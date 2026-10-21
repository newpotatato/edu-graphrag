"""InferenceEngine: loaded once by alias, runs in its own bounded pool, never on the loop."""

import asyncio
import threading
import time

import pandas as pd
import pytest

from edu_graphrag.components.mlflow_model import ModelNotLoadedError
from edu_graphrag.pipeline.inference import InferenceEngine
from tests.fakes import FakeModelRegistry, FakePyfuncModel


def engine(
    registry: FakeModelRegistry, *, concurrency: int = 2, delay_ms: int = 0
) -> InferenceEngine:
    return InferenceEngine(
        registry,
        model_name="subject-classifier",
        model_alias="champion",
        concurrency=concurrency,
        delay_ms=delay_ms,
    )


async def test_start_resolves_alias_and_loads_once() -> None:
    registry = FakeModelRegistry(aliases={"champion": "7"})
    eng = engine(registry)
    await eng.start()
    await eng.start()  # idempotent: never a second load
    assert registry.resolved == 1
    assert registry.loaded_versions == ["7"]
    assert eng.ref.version == "7"
    assert eng.uri == "models:/subject-classifier@champion"
    assert eng.loaded_at is not None
    await eng.shutdown()
    assert registry.closed == 1
    assert not eng.is_loaded


async def test_start_fails_loudly_without_model() -> None:
    eng = engine(FakeModelRegistry(aliases={}))
    with pytest.raises(LookupError):
        await eng.start()
    with pytest.raises(ModelNotLoadedError):
        await eng.predict("x")
    with pytest.raises(ModelNotLoadedError):
        _ = eng.ref


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        engine(FakeModelRegistry(), concurrency=0)


async def test_predict_runs_in_pool_threads_with_delay() -> None:
    registry = FakeModelRegistry()
    eng = engine(registry, concurrency=3, delay_ms=100)
    await eng.start()
    started = time.perf_counter()
    results = await asyncio.gather(*(eng.predict(f"t{i}") for i in range(3)))
    elapsed = time.perf_counter() - started
    assert [r["label"] for r in results] == ["T0", "T1", "T2"]
    # 3 x 100 ms in parallel on 3 threads.
    assert elapsed < 0.25
    model = registry.models["3"]
    assert threading.get_ident() not in model.thread_ids
    assert eng.stats().completed == 3
    await eng.shutdown()


async def test_pool_is_bounded_by_concurrency() -> None:
    registry = FakeModelRegistry()
    eng = engine(registry, concurrency=2, delay_ms=100)
    await eng.start()
    started = time.perf_counter()
    await asyncio.gather(*(eng.predict("x") for _ in range(4)))
    # 4 tasks on 2 threads: two rounds of 100 ms.
    assert time.perf_counter() - started >= 0.19
    await eng.shutdown()


async def test_failed_inference_is_counted_and_raised() -> None:
    registry = FakeModelRegistry()
    registry.models["3"] = FakePyfuncModel(error=RuntimeError("bad model"))
    eng = engine(registry)
    await eng.start()
    with pytest.raises(RuntimeError, match="bad model"):
        await eng.predict("x")
    stats = eng.stats()
    assert (stats.failed, stats.in_flight) == (1, 0)
    await eng.shutdown()


async def test_wrong_number_of_rows_is_an_error() -> None:
    registry = FakeModelRegistry()
    registry.models["3"] = FakePyfuncModel(output=lambda frame: pd.DataFrame({"label": ["a", "b"]}))
    eng = engine(registry)
    await eng.start()
    with pytest.raises(ValueError, match="2 rows"):
        await eng.predict("x")
    await eng.shutdown()


async def test_shutdown_waits_for_running_inference() -> None:
    registry = FakeModelRegistry()
    eng = engine(registry, concurrency=1, delay_ms=200)
    await eng.start()
    running = asyncio.create_task(eng.predict("x"))
    await asyncio.sleep(0.05)
    await eng.shutdown()
    # wait=True: the thread finished its work before shutdown returned.
    assert eng.stats().completed == 1
    assert (await running)["label"] == "X"
