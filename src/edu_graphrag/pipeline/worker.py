"""Inference worker: consumes `tasks`, classifies the text, commits the result to Postgres.

Run: `uvicorn edu_graphrag.pipeline.worker:create_app --factory --host 0.0.0.0 --port 8001`.

"The worker is an independent process connected to publishers ONLY via RabbitMQ." It shares
nothing with the API but the message contract (`pipeline.messages`) and the topology
(`pipeline.topology`); results go to Postgres, never back to a publisher.

Concurrency and prefetch
------------------------
Measured on FastStream 0.7.7 / RabbitMQ 4.2 (prefetch=4, 8 messages, 1 s each): 2.07 s total,
peak 4 handlers at once; with prefetch=1: 8.14 s. FastStream does NOT serialise a Rabbit
subscriber: aiormq starts one asyncio task per delivery, so the number of handlers running at
once equals the number of unacked deliveries, i.e. `prefetch_count`. Therefore

    prefetch_count = WORKER_CONCURRENCY = size of the inference thread pool.

Throughput by Little's law: X = concurrency / t_inference (4 / 0.2 s = 20 tasks/s per worker).
* prefetch > concurrency: the extra messages sit idle in this worker's buffer. They are
  invisible to a second worker (unfair distribution, the queue looks shorter than the backlog
  really is), all of them are redelivered if this worker crashes, and their latency grows by
  the time spent waiting for a free thread. The broker-side queue, not the worker's memory,
  must be where backlog accumulates, otherwise max-length cannot push back on publishers.
* prefetch < concurrency: threads stay idle, throughput drops to prefetch / t_inference.

Acknowledgement policy: `AckPolicy.NACK_ON_ERROR`
-------------------------------------------------
The message is acked by FastStream only after the handler returned, and the handler returns
only after `save_result` committed its transaction: "never ack before the DB write".
At-least-once delivery + an idempotent store (upsert on `task_id`) = exactly-once effect.
Alternatives:
* ACK_FIRST: acked on receipt, at-most-once; a crash or a DB error loses the task.
* ACK: acks even when the handler raised, the failed task is silently lost.
* REJECT_ON_ERROR: no retries at all, every transient blip (DB restart) goes to the DLQ.
* MANUAL: same semantics as ours, but every code path must ack/nack by hand; the explicit
  exceptions below give the same control with less room for a forgotten ack.

Error flow
----------
* Transient error (model exception, Postgres down): `NackMessage(requeue=True)`. The quorum
  queue counts it in `x-delivery-count`; after `delivery-limit` (policy, DELIVERY_LIMIT)
  failed deliveries the broker dead-letters it to `tasks.dlq` with reason `delivery_limit`.
  Measured: delivery-limit=N gives N+1 deliveries in total (counts None, 1, ..., N).
  No infinite requeue loop.
* Poison message (not a valid TaskMessage): `RejectMessage()` (requeue=False), dead-lettered
  immediately with reason `rejected`: retrying cannot fix a malformed payload.
* Redelivery of a finished task: `save_result` returns False, the message is acked and
  `duplicate_delivery` is logged; nothing is written twice.

Dead letters
------------
On the final allowed delivery the worker records the task as `failed` (best effort: if the
database is the very thing that is down, this write fails too). Messages stay in `tasks.dlq`
so they can be inspected (Management UI) and replayed. The optional DLQ recorder
(DLQ_RECORDER_ENABLED=true, off by default for exactly that reason) consumes `tasks.dlq`,
marks parseable tasks as failed with the `x-first-death-reason`, and acks them.

Graceful shutdown
-----------------
SIGTERM -> uvicorn runs the lifespan shutdown -> the broker stops the subscribers: no new
deliveries are handled, in-flight handlers finish (inference, DB commit, ack) for up to
WORKER_GRACEFUL_TIMEOUT -> the channel closes, so prefetched-but-unstarted deliveries go back
to the queue -> after the broker has stopped, the executor is shut down with wait=True.
Compose's `stop_grace_period` (30 s) is longer than the graceful timeout (20 s).
"""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import structlog
from faststream.asgi import AsgiFastStream, AsgiResponse, get
from faststream.exceptions import NackMessage, RejectMessage
from faststream.middlewares import AckPolicy
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from faststream.rabbit.schemas import Channel
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from edu_graphrag.components.mlflow_model import MlflowModelRegistry, ModelRegistry
from edu_graphrag.config import Settings, get_settings
from edu_graphrag.logging_config import configure_logging
from edu_graphrag.pipeline.inference import InferenceEngine
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.topology import (
    DEAD_LETTER_EXCHANGE,
    DEAD_LETTER_QUEUE,
    TASKS_EXCHANGE,
    TASKS_QUEUE,
)
from edu_graphrag.pipeline.worker_store import PostgresTaskStore, TaskStore

logger = structlog.get_logger(__name__)

DELIVERY_COUNT_HEADER = "x-delivery-count"
_FASTSTREAM_LOGGER = logging.getLogger("edu_graphrag.faststream")


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # How long the broker waits for in-flight handlers on shutdown; < stop_grace_period.
    worker_graceful_timeout: float = Field(default=20.0, gt=0)
    # Drain tasks.dlq into Postgres. Off: dead letters stay in the queue for inspection.
    dlq_recorder_enabled: bool = False


@dataclass(frozen=True, slots=True)
class WorkerDeps:
    """Everything with I/O the worker needs; tests pass in-memory doubles."""

    registry: ModelRegistry
    store: TaskStore


def delivery_attempt(headers: Mapping[str, Any]) -> int:
    """1-based attempt number. Quorum queues put the number of earlier failed deliveries
    into `x-delivery-count`; the header is absent on the first delivery."""
    try:
        return int(headers.get(DELIVERY_COUNT_HEADER) or 0) + 1
    except (TypeError, ValueError):
        return 1


def death_reason(headers: Mapping[str, Any]) -> str:
    """Why the broker dead-lettered a message: `rejected`, `delivery_limit`, `maxlen`..."""
    reason = headers.get("x-first-death-reason")
    if reason:
        return str(reason)
    deaths = headers.get("x-death")
    if isinstance(deaths, list) and deaths and isinstance(deaths[0], Mapping):
        return str(deaths[0].get("reason", "unknown"))
    return "unknown"


class Worker:
    def __init__(
        self,
        settings: Settings,
        deps: WorkerDeps,
        worker_settings: WorkerSettings | None = None,
    ) -> None:
        self._settings = settings
        self._worker_settings = worker_settings or WorkerSettings()
        self._store = deps.store
        pipeline = settings.pipeline
        self.delivery_limit = pipeline.delivery_limit
        # Explicit, see the module docstring: one unacked delivery per inference thread.
        self.prefetch_count = pipeline.worker_concurrency
        self.engine = InferenceEngine(
            deps.registry,
            model_name=settings.mlflow.model_name,
            model_alias=settings.mlflow.model_alias,
            concurrency=pipeline.worker_concurrency,
            delay_ms=pipeline.inference_delay_ms,
        )
        self._in_flight = 0
        self._started_at = time.monotonic()

        self.broker = RabbitBroker(
            settings.rabbit.url,
            graceful_timeout=self._worker_settings.worker_graceful_timeout,
            # A child of the root logger: FastStream records go through the JSON handler.
            logger=_FASTSTREAM_LOGGER,
        )
        self.broker.subscriber(
            TASKS_QUEUE,
            TASKS_EXCHANGE,
            channel=Channel(prefetch_count=self.prefetch_count),
            ack_policy=AckPolicy.NACK_ON_ERROR,
            title="tasks",
        )(self.handle_task)
        if self._worker_settings.dlq_recorder_enabled:
            self.broker.subscriber(
                DEAD_LETTER_QUEUE,
                DEAD_LETTER_EXCHANGE,
                # Bookkeeping only: one message at a time is plenty.
                channel=Channel(prefetch_count=1),
                ack_policy=AckPolicy.NACK_ON_ERROR,
                title="tasks.dlq",
            )(self.record_dead_letter)

        self.app = AsgiFastStream(
            self.broker,
            asgi_routes=[("/health", get(self.health)), ("/ready", get(self.ready))],
            logger=_FASTSTREAM_LOGGER,
            on_startup=[self._on_startup],
            after_shutdown=[self._after_shutdown],
        )

    # ---------- lifecycle ----------

    async def _on_startup(self) -> None:
        # Before the broker starts consuming: no message is handled without the model.
        await self.engine.start()
        await self.declare_dead_lettering()
        logger.info(
            "worker_started",
            model_uri=self.engine.uri,
            model_version=self.engine.ref.version,
            concurrency=self.engine.concurrency,
            prefetch_count=self.prefetch_count,
            inference_delay_ms=self._settings.pipeline.inference_delay_ms,
            delivery_limit=self.delivery_limit,
            dlq_recorder_enabled=self._worker_settings.dlq_recorder_enabled,
        )

    async def declare_dead_lettering(self) -> None:
        """`tasks` and its binding are declared by the subscriber; the dead-letter side is
        declared here so dead letters are never dropped for lack of a target."""
        await self.broker.connect()
        exchange = await self.broker.declare_exchange(DEAD_LETTER_EXCHANGE)
        queue = await self.broker.declare_queue(DEAD_LETTER_QUEUE)
        await queue.bind(exchange, routing_key=DEAD_LETTER_QUEUE.routing())

    async def _after_shutdown(self) -> None:
        # The broker has stopped (in-flight handlers finished or timed out): now wait for
        # any inference thread still running and release resources.
        await self.engine.shutdown()
        await self._store.close()
        logger.info("worker_stopped", **self._stats())

    # ---------- handlers ----------

    async def handle_task(self, message: RabbitMessage) -> None:
        attempt = delivery_attempt(message.headers)
        try:
            task = TaskMessage.model_validate_json(message.body)
        except ValidationError as exc:
            logger.warning(
                "poison_message",
                message_id=message.message_id,
                attempt=attempt,
                error=str(exc).splitlines()[0],
            )
            # requeue=False: dead-lettered at once, retrying cannot fix the payload.
            raise RejectMessage(requeue=False) from exc

        log = logger.bind(task_id=str(task.task_id), attempt=attempt)
        self._in_flight += 1
        started = time.perf_counter()
        try:
            result = await self.engine.predict(task.text)
            ref = self.engine.ref
            saved = await self._store.save_result(
                task,
                result=result,
                model_name=ref.name,
                model_version=ref.version,
                attempts=attempt,
            )
        except Exception as exc:
            await self._on_transient_error(task, attempt, exc, log)
            raise NackMessage(requeue=True) from exc
        finally:
            self._in_flight -= 1

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if saved:
            log.info("task_done", label=result.get("label"), duration_ms=duration_ms)
        else:
            log.info("duplicate_delivery", duration_ms=duration_ms)
        # Returning acks the message: the result is already committed.

    async def _on_transient_error(
        self,
        task: TaskMessage,
        attempt: int,
        exc: Exception,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"
        # delivery-limit=N allows N+1 deliveries; on the last one the broker dead-letters.
        final = attempt > self.delivery_limit
        log.warning("task_failed", error=error, final_attempt=final, requeue=True)
        if not final:
            return
        try:
            await self._store.mark_failed(task, f"delivery limit reached: {error}")
        except Exception as store_exc:  # best effort: the DB may be what is failing
            log.error("mark_failed_error", error=f"{type(store_exc).__name__}: {store_exc}")

    async def record_dead_letter(self, message: RabbitMessage) -> None:
        reason = death_reason(message.headers)
        try:
            task = TaskMessage.model_validate_json(message.body)
        except ValidationError:
            # Nothing to attach the failure to: there is no valid task_id.
            logger.warning("dead_letter_unparseable", reason=reason, body=message.body[:200])
            return
        await self._store.mark_failed(task, f"dead-lettered: {reason}")
        logger.info("dead_letter_recorded", task_id=str(task.task_id), reason=reason)

    # ---------- HTTP ----------

    def _stats(self) -> dict[str, Any]:
        stats = self.engine.stats()
        model: dict[str, Any] = {"uri": self.engine.uri, "loaded": self.engine.is_loaded}
        if self.engine.is_loaded:
            ref = self.engine.ref
            model |= {"name": ref.name, "version": ref.version, "run_id": ref.run_id}
        return {
            "model": model,
            "concurrency": stats.concurrency,
            "prefetch_count": self.prefetch_count,
            # Handlers between receipt and ack; inference_threads_busy is the pool's view.
            "in_flight": self._in_flight,
            "inference_threads_busy": stats.in_flight,
            "completed": stats.completed,
            "failed": stats.failed,
            "uptime_s": round(time.monotonic() - self._started_at, 1),
        }

    async def health(self, scope: Any) -> AsgiResponse:  # noqa: ARG002
        """Liveness. Served by the event loop: answers instantly even when every inference
        thread is busy, which is the proof that inference does not block the loop."""
        return _json({"status": "ok", **self._stats()})

    async def ready(self, scope: Any) -> AsgiResponse:  # noqa: ARG002
        """Readiness: the model is loaded and the broker connection is up."""
        connected = await self.broker.ping(timeout=1.0)
        ok = self.engine.is_loaded and connected
        body = {"status": "ok" if ok else "unavailable", "broker": connected, **self._stats()}
        return _json(body, 200 if ok else 503)


def _json(data: dict[str, Any], status_code: int = 200) -> AsgiResponse:
    return AsgiResponse(
        json.dumps(data).encode(),
        status_code,
        {"Content-Type": "application/json"},
    )


def build_worker(
    settings: Settings | None = None,
    deps: WorkerDeps | None = None,
    worker_settings: WorkerSettings | None = None,
) -> Worker:
    settings = settings or get_settings()
    if deps is None:
        deps = WorkerDeps(
            registry=MlflowModelRegistry(settings.mlflow.mlflow_tracking_uri),
            store=PostgresTaskStore(
                settings.postgres, pool_size=settings.pipeline.worker_concurrency + 1
            ),
        )
    return Worker(settings, deps, worker_settings)


def create_app() -> AsgiFastStream:
    """ASGI factory for uvicorn (`--factory`)."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    return build_worker(settings).app


__all__ = [
    "Worker",
    "WorkerDeps",
    "WorkerSettings",
    "build_worker",
    "create_app",
    "death_reason",
    "delivery_attempt",
]
