"""Message contract between publishers and workers."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

TaskSource = Literal["api", "batch"]

MAX_TEXT_CHARS = 20_000


class TaskMessage(BaseModel):
    """A command: "classify this text". `task_id` is the idempotency key end to end."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: UUID
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    source: TaskSource
    submitted_at: datetime

    @classmethod
    def new(cls, text: str, source: TaskSource, task_id: UUID | None = None) -> "TaskMessage":
        return cls(
            task_id=task_id or uuid4(),
            text=text,
            source=source,
            submitted_at=datetime.now(UTC),
        )
