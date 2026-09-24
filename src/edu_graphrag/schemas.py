"""Response models."""

from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class VersionResponse(BaseModel):
    name: str
    version: str = Field(description="Application version, taken from pyproject.toml")
    commit: str = Field(description="Git commit the image was built from")
