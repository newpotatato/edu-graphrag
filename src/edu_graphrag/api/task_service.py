"""Accepting tasks at the system boundary: LOAD SHEDDING, not backpressure.

Backpressure propagates "slow down" upstream to a producer that is part of our control loop
(the batch source waits for confirms and backs off). An external HTTP client is outside that
loop: we cannot make it slow down and must not make it wait. So the API sheds load: if the
queue refuses the task, the request fails *immediately* with 503 + Retry-After, and the client
(or its retry policy) decides what to do. Waiting inside the request for free space would tie
up connections and DB transactions for an unbounded time and push the overload into the API.

Flow of POST /api/v1/tasks, all in ONE database transaction:

    BEGIN
      insert_pending(task)                 row exists, but invisible to others until COMMIT
      publish(task), wait for the confirm  ONE attempt, <= API_CONFIRM_TIMEOUT
    ack           -> COMMIT   -> 202 Accepted + Location
    nack/timeout/
    broker down   -> ROLLBACK -> 503 + Retry-After, no PENDING row is left behind

Why publish inside the transaction: the row is committed only for a task the broker has
taken responsibility for, so a PENDING row always has a message behind it, and a 503 leaves
nothing to clean up. A fast worker cannot "lose" its result either: its upsert of the same
primary key waits on the uncommitted row and then updates it (ON CONFLICT DO UPDATE).

Edge case - COMMIT fails after an ack: the message is already in the queue and will be
processed; the worker's `save_result` is an upsert, so it creates the row itself. We answer
202 (the task *is* accepted; GET may briefly say 404 until the worker stores it) instead of
telling the client to resubmit a task that will run anyway.

Edge case - confirm timeout: the message may still have reached the queue. We answer 503 and
roll back; if it did land, the worker processes it and upserts a row nobody polls. That is the
at-least-once price; an Idempotency-Key header would let the client retry with the same id.

Why no transactional outbox: an outbox commits the task to a table and publishes later, i.e.
it *accepts* the task while the queue is full. That is the opposite of what is required here:
the client must learn about the overload now, with a 503, not after its task sat in an
unbounded outbox table (which just moves the queue into Postgres).
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import asyncpg  # type: ignore[import-untyped]  # no stubs; only exception classes are used
import structlog
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection

from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.config import PipelineSettings
from edu_graphrag.pipeline import store
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import BrokerUnavailableError, PublishError, TaskPublisher
from edu_graphrag.pipeline.retry_after import retry_after_seconds
from edu_graphrag.pipeline.store import TaskRecord

logger = structlog.get_logger(__name__)

# A quorum confirm normally takes milliseconds (a majority fsync). Half a second means the
# broker is struggling; the request must fail fast rather than wait for it.
API_CONFIRM_TIMEOUT = 0.5
# Bound on the extra round trip made on the 503 path to read the consumer count.
CONSUMER_COUNT_TIMEOUT = 0.3
# The consumer count is cached: under overload every request takes the 503 path, and load
# shedding must stay cheap. One second is the granularity of Retry-After anyway.
CONSUMER_COUNT_TTL = 1.0
# Database outage, not overload: the pool reconnects on the next checkout (pre-ping).
STORE_RETRY_AFTER = 5
# SQLAlchemy does not wrap every driver error: connect-time failures (e.g. a rejected
# password) surface as raw asyncpg exceptions.
DB_ERRORS = (SQLAlchemyError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError)


class StoreUnavailableError(Exception):
    pass


class TaskTransaction(Protocol):
    async def insert_pending(self, message: TaskMessage) -> None: ...


class TaskStore(Protocol):
    def transaction(self) -> AbstractAsyncContextManager[TaskTransaction]:
        """Commits on normal exit, rolls back if the block raises."""

    async def get(self, task_id: UUID) -> TaskRecord | None: ...


@dataclass(frozen=True, slots=True)
class _PostgresTransaction:
    conn: AsyncConnection

    async def insert_pending(self, message: TaskMessage) -> None:
        await store.insert_pending(self.conn, message)


class PostgresTaskStore:
    def __init__(self, postgres: PostgresComponent) -> None:
        self._postgres = postgres

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[TaskTransaction]:
        try:
            async with self._postgres.engine.begin() as conn:
                yield _PostgresTransaction(conn)
        except DB_ERRORS as exc:
            raise StoreUnavailableError(f"{type(exc).__name__}: {exc}") from exc

    async def get(self, task_id: UUID) -> TaskRecord | None:
        try:
            async with self._postgres.engine.connect() as conn:
                return await store.get_task(conn, task_id)
        except DB_ERRORS as exc:
            raise StoreUnavailableError(f"{type(exc).__name__}: {exc}") from exc


class TaskRejectedError(Exception):
    """The task was not accepted; nothing was stored. Maps to 503 + Retry-After."""

    def __init__(self, title: str, detail: str, retry_after: int) -> None:
        super().__init__(detail)
        self.title = title
        self.detail = detail
        self.retry_after = retry_after


class TaskService:
    def __init__(
        self,
        task_store: TaskStore,
        publisher: TaskPublisher,
        *,
        pipeline: PipelineSettings,
        broker_retry_after: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = task_store
        self._publisher = publisher
        self._pipeline = pipeline
        self._confirm_timeout = min(API_CONFIRM_TIMEOUT, pipeline.publish_confirm_timeout)
        self._broker_retry_after = broker_retry_after
        self._clock = clock
        self._consumers: tuple[float, int | None] | None = None

    async def submit(self, text: str) -> TaskMessage:
        message = TaskMessage.new(text, source="api")
        published = False
        try:
            async with self._store.transaction() as tx:
                await tx.insert_pending(message)
                await self._publisher.publish(message, confirm_timeout=self._confirm_timeout)
                published = True
        except PublishError as exc:
            retry_after = await self.retry_after(exc)
            logger.warning(
                "task_shed",
                task_id=str(message.task_id),
                reason=type(exc).__name__,
                retry_after=retry_after,
            )
            raise TaskRejectedError(
                "Task queue is full"
                if not isinstance(exc, BrokerUnavailableError)
                else "Task queue is unavailable",
                f"The task was not accepted ({type(exc).__name__}); nothing was stored. "
                f"Retry after {retry_after} s.",
                retry_after,
            ) from exc
        except StoreUnavailableError as exc:
            if not published:
                logger.error("task_store_unavailable", error=str(exc))
                raise TaskRejectedError(
                    "Task store is unavailable",
                    "The task was not accepted: the database is unavailable.",
                    STORE_RETRY_AFTER,
                ) from exc
            # Confirmed by the broker: it will run, the worker's upsert creates the row.
            logger.warning("task_commit_failed_after_publish", task_id=str(message.task_id))
        logger.info("task_accepted", task_id=str(message.task_id))
        return message

    async def get(self, task_id: UUID) -> TaskRecord | None:
        return await self._store.get(task_id)

    async def retry_after(self, exc: PublishError) -> int:
        if isinstance(exc, BrokerUnavailableError):
            # Nothing can be accepted before the API has reconnected.
            return self._broker_retry_after
        consumers = await self._consumer_count()
        return retry_after_seconds(
            queue_max_length=self._pipeline.queue_max_length,
            task_seconds=self._pipeline.inference_delay_ms / 1000,
            # Unknown count, but the broker answered: assume one worker (the longer wait).
            consumers=1 if consumers is None else consumers,
            concurrency=self._pipeline.worker_concurrency,
        )

    async def _consumer_count(self) -> int | None:
        now = self._clock()
        if self._consumers is not None and now - self._consumers[0] < CONSUMER_COUNT_TTL:
            return self._consumers[1]
        try:
            async with asyncio.timeout(CONSUMER_COUNT_TIMEOUT):
                count = await self._publisher.consumer_count()
        except TimeoutError:
            count = None
        self._consumers = (now, count)
        return count
