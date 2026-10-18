import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edu_graphrag.components.mlflow_model import MlflowModelComponent, ModelVersionRef
from edu_graphrag.config import MlflowSettings


@dataclass
class FakeComponent:
    """In-memory `Component` whose behaviour (version, latency, failure) is set by the test."""

    name: str
    version: str = "1.0.0"
    delay: float = 0.0
    error: Exception | None = None
    startup_error: Exception | None = None
    events: list[str] = field(default_factory=list)
    checks: int = 0

    async def startup(self) -> None:
        if self.startup_error is not None:
            raise self.startup_error
        self.events.append(f"start:{self.name}")

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")

    async def fetch_version(self) -> str:
        self.checks += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.version


@dataclass
class FakePyfuncModel:
    """Pyfunc stand-in: returns `output(frame)` and records the thread each call ran in."""

    output: Callable[[pd.DataFrame], Any] = field(
        default=lambda frame: pd.DataFrame(
            {"label": frame["text"].str.upper(), "confidence": np.float64(0.9)}
        )
    )
    error: Exception | None = None
    inputs: list[pd.DataFrame] = field(default_factory=list)
    thread_ids: list[int] = field(default_factory=list)

    def predict(self, data: pd.DataFrame) -> Any:
        self.thread_ids.append(threading.get_ident())
        self.inputs.append(data)
        if self.error is not None:
            raise self.error
        return self.output(data)


@dataclass
class FakeModelRegistry:
    """In-memory `ModelRegistry`: aliases map to versions, each version to a model."""

    aliases: dict[str, str] = field(default_factory=lambda: {"champion": "3"})
    models: dict[str, FakePyfuncModel] = field(default_factory=dict)
    resolve_error: Exception | None = None
    load_error: Exception | None = None
    resolved: int = 0
    loaded_versions: list[str] = field(default_factory=list)
    closed: int = 0

    def model(self, version: str) -> FakePyfuncModel:
        return self.models.setdefault(version, FakePyfuncModel())

    def resolve(self, name: str, alias: str) -> ModelVersionRef:
        self.resolved += 1
        if self.resolve_error is not None:
            raise self.resolve_error
        if alias not in self.aliases:
            raise LookupError(f"alias {alias!r} not found for {name!r}")
        version = self.aliases[alias]
        return ModelVersionRef(
            name=name,
            alias=alias,
            version=version,
            run_id=f"run-{version}",
            model_uri=f"models:/{name}/{version}",
        )

    def load(self, ref: ModelVersionRef) -> FakePyfuncModel:
        if self.load_error is not None:
            raise self.load_error
        self.loaded_versions.append(ref.version)
        return self.model(ref.version)

    def close(self) -> None:
        self.closed += 1


def make_model_component(
    registry: FakeModelRegistry | None = None, *, name: str = "subject-classifier"
) -> tuple[MlflowModelComponent, FakeModelRegistry]:
    registry = registry or FakeModelRegistry()
    settings = MlflowSettings(mlflow_tracking_uri="http://mlflow.invalid", model_name=name)
    return MlflowModelComponent(settings, registry=registry), registry
