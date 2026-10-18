"""Request and response models."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


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


# Limits keep one request bounded in memory and inference time.
MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_LENGTH = 20_000


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


InputText = Annotated[
    str, Field(min_length=1, max_length=MAX_TEXT_LENGTH), AfterValidator(_not_blank)
]


class ModelRef(BaseModel):
    name: str
    alias: str
    version: str = Field(description="Concrete registry version the alias resolved to at startup")
    run_id: str | None


class ModelInfo(ModelRef):
    model_uri: str = Field(description="Exact URI that was loaded, e.g. models:/name/3")
    loaded_at: datetime


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[InputText] = Field(min_length=1, max_length=MAX_TEXTS_PER_REQUEST)


class ProcessResponse(BaseModel):
    model: ModelRef
    predictions: list[dict[str, Any]] = Field(
        description="One object per output row: the model's columns, or {'prediction': value}"
    )
