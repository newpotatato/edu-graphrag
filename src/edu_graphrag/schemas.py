"""Request and response models."""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from edu_graphrag.pipeline.messages import MAX_TEXT_CHARS
from edu_graphrag.pipeline.store import TaskStatus


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    name: str
    version: str = Field(description="Application version, taken from pyproject.toml")
    commit: str = Field(description="Git commit the image was built from")


class ComponentHealth(BaseModel):
    name: str
    status: Literal["up", "down"]
    version: str | None = None
    latency_ms: float = Field(description="Round trip time of the check, milliseconds")
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


# Same limit as the queue message contract: what the API accepts always fits a TaskMessage.
InputText = Annotated[
    str, Field(min_length=1, max_length=MAX_TEXT_CHARS), AfterValidator(_not_blank)
]


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"text": "Решите квадратное уравнение x^2 - 5x + 6 = 0"}]},
    )

    text: InputText = Field(description=f"Text to classify, 1..{MAX_TEXT_CHARS} characters")


class TaskAccepted(BaseModel):
    task_id: UUID
    status: Literal["pending"] = "pending"
    status_url: str = Field(description="Poll this URL for the result (also in `Location`)")


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: TaskStatus = Field(description="pending -> done | failed")
    source: str = Field(description="Who submitted the task: api or batch")
    result: dict[str, Any] | None = Field(None, description="Model output when done")
    error: str | None = Field(None, description="Why the task failed (dead-lettered)")
    model_name: str | None = None
    model_version: str | None = Field(None, description="Registry version that produced it")
    attempts: int = Field(description="Deliveries it took (redeliveries after worker failures)")
    submitted_at: datetime
    finished_at: datetime | None = None
    processing_ms: float | None = Field(
        None, description="finished_at - submitted_at: queueing + inference, milliseconds"
    )


class Problem(BaseModel):
    """RFC 9457 problem details (`application/problem+json`)."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    retry_after: int | None = Field(None, description="Same value as the Retry-After header, s")
