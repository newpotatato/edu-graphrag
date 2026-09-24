"""Response models."""

from typing import Literal

from pydantic import BaseModel, Field


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
