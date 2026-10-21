"""Model inference for the worker: loaded once, executed in a bounded thread pool.

Requirements this module implements:

* "The model is loaded from MLflow by alias (`models:/<name>@<alias>`) once at worker startup,
  never per message, never a hardcoded version or path." `start()` resolves the alias to a
  concrete version through the registry and loads exactly that version; the resolved version
  is logged and stamped on every result.
* "Inference must not block the event loop." `model.predict` and the simulated inference cost
  (`time.sleep(INFERENCE_DELAY_MS)`) are synchronous, so they run in a dedicated
  `ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY)` via `loop.run_in_executor`. The event
  loop only awaits a future: it keeps reading AMQP frames, sending heartbeats, writing to
  Postgres and answering /health while every thread is busy. A dedicated (not the default)
  executor makes the pool size an explicit, measurable number: it is the worker's concurrency
  and therefore its throughput by Little's law, `X = concurrency / t_inference`.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog

from edu_graphrag.components.mlflow_model import (
    TEXT_COLUMN,
    ModelNotLoadedError,
    ModelRegistry,
    ModelVersionRef,
    PyfuncModel,
    predictions_to_records,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceStats:
    concurrency: int
    in_flight: int
    completed: int
    failed: int


class InferenceEngine:
    """Owns the model and the inference thread pool of one worker process."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        model_name: str,
        model_alias: str,
        concurrency: int,
        delay_ms: int,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._registry = registry
        self._model_name = model_name
        self._model_alias = model_alias
        self._concurrency = concurrency
        self._delay_s = delay_ms / 1000
        self._executor: ThreadPoolExecutor | None = None
        self._model: PyfuncModel | None = None
        self._ref: ModelVersionRef | None = None
        self._loaded_at: datetime | None = None
        self._lock = threading.Lock()
        self._in_flight = 0
        self._completed = 0
        self._failed = 0

    @property
    def uri(self) -> str:
        return f"models:/{self._model_name}@{self._model_alias}"

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def ref(self) -> ModelVersionRef:
        if self._ref is None:
            raise ModelNotLoadedError(f"model {self.uri} is not loaded")
        return self._ref

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    def stats(self) -> InferenceStats:
        with self._lock:
            return InferenceStats(self._concurrency, self._in_flight, self._completed, self._failed)

    async def start(self) -> None:
        """Resolve the alias and load that exact version. Called once, before consuming.

        Unlike the API (degraded start), a worker without a model must not start: it would
        nack every message and burn their delivery attempts. A failure here crashes the
        process and the orchestrator restarts it.
        """
        if self._model is not None:
            return
        started = time.perf_counter()
        self._executor = ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="inference"
        )
        # The registry client is blocking: keep it off the event loop too.
        ref = await asyncio.to_thread(self._registry.resolve, self._model_name, self._model_alias)
        self._model = await asyncio.to_thread(self._registry.load, ref)
        self._ref = ref
        self._loaded_at = datetime.now(UTC)
        logger.info(
            "model_loaded",
            model_uri=self.uri,
            model_version=ref.version,
            run_id=ref.run_id,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def predict(self, text: str) -> dict[str, Any]:
        """Classify one text in the inference pool; the coroutine only awaits the future."""
        if self._executor is None or self._model is None:
            raise ModelNotLoadedError(f"model {self.uri} is not loaded")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict_blocking, self._model, text)

    def _predict_blocking(self, model: PyfuncModel, text: str) -> dict[str, Any]:
        """Runs in an inference thread: blocking calls are fine here, never in async code."""
        with self._lock:
            self._in_flight += 1
        try:
            if self._delay_s:
                # Simulated inference cost (INFERENCE_DELAY_MS): reproducible throughput.
                time.sleep(self._delay_s)
            raw = model.predict(pd.DataFrame({TEXT_COLUMN: [text]}))
            records = predictions_to_records(raw)
            if len(records) != 1:
                raise ValueError(f"model returned {len(records)} rows for 1 input")
        except BaseException:
            with self._lock:
                self._in_flight -= 1
                self._failed += 1
            raise
        with self._lock:
            self._in_flight -= 1
            self._completed += 1
        return records[0]

    async def shutdown(self) -> None:
        """Wait for running inferences (wait=True), then release the model.

        Called after the broker has stopped, so no new work can arrive while waiting.
        """
        executor, self._executor = self._executor, None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, wait=True)
        self._model = None
        await asyncio.to_thread(self._registry.close)
