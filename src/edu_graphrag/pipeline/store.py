"""Task storage. Idempotency is enforced by the database (primary key on `task_id`), not by
"check, then insert" in the application, which would race between concurrent deliveries."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Uuid,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncConnection

from edu_graphrag.pipeline.messages import TaskMessage

TaskStatus = Literal["pending", "done", "failed"]

metadata = MetaData()

tasks = Table(
    "tasks",
    metadata,
    Column("task_id", Uuid, primary_key=True),
    Column("source", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("input_text", Text, nullable=False),
    Column("result", JSONB, nullable=True),
    Column("error", Text, nullable=True),
    Column("model_name", String(128), nullable=True),
    Column("model_version", String(32), nullable=True),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    CheckConstraint("status in ('pending', 'done', 'failed')", name="tasks_status_check"),
    Index("ix_tasks_status", "status"),
)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: UUID
    source: str
    status: TaskStatus
    result: dict[str, Any] | None
    error: str | None
    model_name: str | None
    model_version: str | None
    attempts: int
    submitted_at: datetime
    finished_at: datetime | None


async def insert_pending(conn: AsyncConnection, message: TaskMessage) -> None:
    """API path: called inside the transaction that also publishes the message; if the publish
    is refused, the caller rolls back and no PENDING row is left behind."""
    await conn.execute(
        insert(tasks)
        .values(
            task_id=message.task_id,
            source=message.source,
            status="pending",
            input_text=message.text,
            submitted_at=message.submitted_at,
        )
        .on_conflict_do_nothing(index_elements=[tasks.c.task_id])
    )


async def save_result(
    conn: AsyncConnection,
    message: TaskMessage,
    *,
    result: dict[str, Any],
    model_name: str,
    model_version: str,
    attempts: int,
) -> bool:
    """Idempotent upsert of a finished task. Returns False when the task was already done
    (a redelivery): the existing result is kept and nothing is duplicated."""
    values = insert(tasks).values(
        task_id=message.task_id,
        source=message.source,
        status="done",
        input_text=message.text,
        result=result,
        model_name=model_name,
        model_version=model_version,
        attempts=attempts,
        submitted_at=message.submitted_at,
        finished_at=datetime.now(UTC),
    )
    stmt = values.on_conflict_do_update(
        index_elements=[tasks.c.task_id],
        set_={
            "status": values.excluded.status,
            "result": values.excluded.result,
            "error": None,
            "model_name": values.excluded.model_name,
            "model_version": values.excluded.model_version,
            "attempts": values.excluded.attempts,
            "finished_at": values.excluded.finished_at,
        },
        where=tasks.c.status != "done",
    ).returning(tasks.c.task_id)
    return (await conn.execute(stmt)).first() is not None


async def mark_failed(conn: AsyncConnection, message: TaskMessage, error: str) -> None:
    """Dead-lettered task: recorded as failed unless it has already succeeded."""
    stmt = insert(tasks).values(
        task_id=message.task_id,
        source=message.source,
        status="failed",
        input_text=message.text,
        error=error,
        submitted_at=message.submitted_at,
        finished_at=datetime.now(UTC),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[tasks.c.task_id],
        set_={"status": "failed", "error": error, "finished_at": stmt.excluded.finished_at},
        where=tasks.c.status != "done",
    )
    await conn.execute(stmt)


async def get_task(conn: AsyncConnection, task_id: UUID) -> TaskRecord | None:
    row = (await conn.execute(select(tasks).where(tasks.c.task_id == task_id))).mappings().first()
    if row is None:
        return None
    return TaskRecord(
        task_id=row["task_id"],
        source=row["source"],
        status=row["status"],
        result=row["result"],
        error=row["error"],
        model_name=row["model_name"],
        model_version=row["model_version"],
        attempts=row["attempts"],
        submitted_at=row["submitted_at"],
        finished_at=row["finished_at"],
    )


async def count_by_status(conn: AsyncConnection) -> dict[str, int]:
    rows = await conn.execute(select(tasks.c.status, func.count()).group_by(tasks.c.status))
    return {status: int(n) for status, n in rows.all()}


__all__ = [
    "TaskRecord",
    "TaskStatus",
    "count_by_status",
    "get_task",
    "insert_pending",
    "mark_failed",
    "metadata",
    "save_result",
    "tasks",
]
