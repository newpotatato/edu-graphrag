"""The worker's view of task storage: a small protocol plus its Postgres implementation.

The worker depends on the protocol, so tests run it against an in-memory double; the
Postgres adapter only opens one short transaction per call around `pipeline.store`.
"""

from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from edu_graphrag.components.postgres import build_postgres_url
from edu_graphrag.config import PostgresSettings
from edu_graphrag.pipeline import store
from edu_graphrag.pipeline.messages import TaskMessage


class TaskStore(Protocol):
    async def save_result(
        self,
        message: TaskMessage,
        *,
        result: dict[str, Any],
        model_name: str,
        model_version: str,
        attempts: int,
    ) -> bool:
        """Commit a finished task; False when it was already done (a redelivery)."""

    async def mark_failed(self, message: TaskMessage, error: str) -> None:
        """Record a task that will not be retried any more (unless it is already done)."""

    async def close(self) -> None: ...


class PostgresTaskStore:
    def __init__(self, settings: PostgresSettings, *, pool_size: int) -> None:
        # Every in-flight task may write at the same moment: one connection per thread.
        self._engine: AsyncEngine = create_async_engine(
            build_postgres_url(settings),
            pool_size=pool_size,
            max_overflow=2,
            pool_pre_ping=True,
            connect_args={"timeout": 5},
        )

    async def save_result(
        self,
        message: TaskMessage,
        *,
        result: dict[str, Any],
        model_name: str,
        model_version: str,
        attempts: int,
    ) -> bool:
        # `begin()` commits on exit: when this returns, the result is durable.
        async with self._engine.begin() as conn:
            return await store.save_result(
                conn,
                message,
                result=result,
                model_name=model_name,
                model_version=model_version,
                attempts=attempts,
            )

    async def mark_failed(self, message: TaskMessage, error: str) -> None:
        async with self._engine.begin() as conn:
            await store.mark_failed(conn, message, error)

    async def close(self) -> None:
        await self._engine.dispose()
