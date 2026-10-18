"""Model from the MLflow Model Registry, loaded once at startup and served from memory.

The component resolves `models:/{MODEL_NAME}@{MODEL_ALIAS}` to a concrete version and loads
exactly that version, so what the app reports is what it actually serves, even if the alias
is moved later. There is deliberately no automatic reload: a restart is the way to pick up a
new alias (immutable deployments: one running process = one model version, easy to roll back
and to reason about). /api/v1/health shows when the alias already points elsewhere.

Degraded start: if MLflow is unreachable or the alias does not exist, the app still starts
(/healthz and /api/v1/version keep working), the failure is logged as `model_load_failed`,
/api/v1/process answers 503 and /api/v1/health reports this component as down.

Only mlflow-skinny modules are used here: the API image does not ship the full mlflow.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import structlog

from edu_graphrag.config import MlflowSettings

logger = structlog.get_logger(__name__)

# Input column every served model receives.
TEXT_COLUMN = "text"
# Column used when a model returns bare values instead of a table.
PREDICTION_COLUMN = "prediction"

# MLflow's own HTTP defaults retry for minutes when the server is down, which would block
# the app startup. These bounded values can still be overridden with the same env variables.
_MLFLOW_HTTP_DEFAULTS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "30",
}


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelVersionRef:
    """A concrete registered model version (what an alias resolved to)."""

    name: str
    alias: str
    version: str
    run_id: str | None
    model_uri: str


class PyfuncModel(Protocol):
    def predict(self, data: pd.DataFrame) -> Any: ...


class ModelRegistry(Protocol):
    """Blocking registry client; the component calls it from a worker thread."""

    def resolve(self, name: str, alias: str) -> ModelVersionRef:
        """Round trip to the registry: which version the alias points to now."""

    def load(self, ref: ModelVersionRef) -> PyfuncModel:
        """Download and load exactly this version."""

    def close(self) -> None:
        """Release local resources (downloaded artifacts)."""


class MlflowModelRegistry:
    """`ModelRegistry` backed by an MLflow tracking server (or any MLflow store URI).

    The tracking URI is passed explicitly instead of relying on mlflow's global state.
    """

    def __init__(self, tracking_uri: str) -> None:
        from mlflow import MlflowClient

        for key, value in _MLFLOW_HTTP_DEFAULTS.items():
            os.environ.setdefault(key, value)
        self._tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self._download_dir: Path | None = None

    def resolve(self, name: str, alias: str) -> ModelVersionRef:
        mv = self._client.get_model_version_by_alias(name, alias)
        return ModelVersionRef(
            name=name,
            alias=alias,
            version=str(mv.version),
            run_id=mv.run_id,
            model_uri=f"models:/{name}/{mv.version}",
        )

    def load(self, ref: ModelVersionRef) -> PyfuncModel:
        import mlflow.artifacts
        import mlflow.pyfunc

        self.close()
        self._download_dir = Path(tempfile.mkdtemp(prefix="edu-graphrag-model-"))
        # Resolve the storage location through our own client and download from it directly:
        # `models:/` URIs would be resolved through mlflow's global tracking URI instead.
        storage_uri = self._client.get_model_version_download_uri(ref.name, ref.version)
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=storage_uri,
            dst_path=str(self._download_dir),
            tracking_uri=self._tracking_uri,
        )
        model: PyfuncModel = mlflow.pyfunc.load_model(local_path)
        return model

    def close(self) -> None:
        if self._download_dir is not None:
            shutil.rmtree(self._download_dir, ignore_errors=True)
            self._download_dir = None


@dataclass(frozen=True, slots=True)
class LoadedModel:
    ref: ModelVersionRef
    model: PyfuncModel
    loaded_at: datetime


def predictions_to_records(raw: Any) -> list[dict[str, Any]]:
    """Turns whatever a pyfunc returned into JSON-able rows.

    A table gives one dict per row; bare values (1-D array, Series, list of scalars) give
    `{"prediction": value}`. Going through `to_json` converts numpy scalars and NaN.
    """
    frame: pd.DataFrame
    if isinstance(raw, pd.DataFrame):
        frame = raw
    elif isinstance(raw, pd.Series):
        frame = pd.DataFrame({PREDICTION_COLUMN: raw.to_numpy()})
    elif isinstance(raw, np.ndarray):
        values = raw.tolist() if raw.ndim > 1 else raw
        frame = pd.DataFrame({PREDICTION_COLUMN: values})
    elif isinstance(raw, list):
        if raw and all(isinstance(item, dict) for item in raw):
            frame = pd.DataFrame(raw)
        else:
            frame = pd.DataFrame({PREDICTION_COLUMN: raw})
    else:
        raise TypeError(f"unsupported prediction type: {type(raw).__name__}")
    records: list[dict[str, Any]] = json.loads(frame.to_json(orient="records"))
    return records


class MlflowModelComponent:
    name = "mlflow_model"

    def __init__(self, settings: MlflowSettings, *, registry: ModelRegistry | None = None) -> None:
        self._model_name = settings.model_name
        self._alias = settings.model_alias
        self._registry = registry or MlflowModelRegistry(settings.mlflow_tracking_uri)
        self._loaded: LoadedModel | None = None
        self._load_error: str | None = "not started"

    @property
    def uri(self) -> str:
        return f"models:/{self._model_name}@{self._alias}"

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise ModelNotLoadedError(f"model {self.uri} is not loaded: {self._load_error}")
        return self._loaded

    async def startup(self) -> None:
        # The registry client is blocking: run it in a thread so the event loop stays free.
        started = time.perf_counter()
        try:
            ref = await asyncio.to_thread(self._registry.resolve, self._model_name, self._alias)
            model = await asyncio.to_thread(self._registry.load, ref)
        except Exception as exc:  # degraded start: the app must come up without a model
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("model_load_failed", model_uri=self.uri, error=self._load_error)
            return
        self._loaded = LoadedModel(ref=ref, model=model, loaded_at=datetime.now(UTC))
        self._load_error = None
        logger.info(
            "model_loaded",
            model_uri=self.uri,
            model_version=ref.version,
            run_id=ref.run_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def shutdown(self) -> None:
        self._loaded = None
        self._load_error = "stopped"
        await asyncio.to_thread(self._registry.close)

    async def fetch_version(self) -> str:
        """Serving version plus a round trip to the registry (is MLflow reachable?)."""
        loaded = self.loaded
        current = await asyncio.to_thread(self._registry.resolve, self._model_name, self._alias)
        served = f"{self._model_name}@{self._alias} v{loaded.ref.version}"
        if current.version != loaded.ref.version:
            return f"{served} (alias now points to v{current.version}, restart to apply)"
        return served

    async def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Inference is CPU-bound and synchronous: it runs in a worker thread."""
        loaded = self.loaded
        frame = pd.DataFrame({TEXT_COLUMN: list(texts)})
        raw = await asyncio.to_thread(loaded.model.predict, frame)
        return predictions_to_records(raw)
