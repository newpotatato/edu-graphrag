import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edu_graphrag.components import ModelNotLoadedError
from edu_graphrag.components.mlflow_model import MlflowModelComponent, predictions_to_records
from edu_graphrag.config import MlflowSettings
from tests.fakes import FakeModelRegistry, make_model_component


async def test_startup_loads_the_version_the_alias_resolves_to() -> None:
    model, registry = make_model_component(FakeModelRegistry(aliases={"champion": "7"}))

    await model.startup()

    assert registry.loaded_versions == ["7"]
    assert model.loaded.ref.version == "7"
    assert model.loaded.ref.model_uri == "models:/subject-classifier/7"
    assert model.loaded.loaded_at.tzinfo is not None


async def test_load_happens_once_regardless_of_predictions() -> None:
    model, registry = make_model_component()
    await model.startup()

    for _ in range(10):
        await model.predict(["x"])

    assert registry.loaded_versions == ["3"]


async def test_alias_moved_after_startup_does_not_change_served_version() -> None:
    model, registry = make_model_component(FakeModelRegistry(aliases={"champion": "1"}))
    await model.startup()

    registry.aliases["champion"] = "2"

    assert await model.fetch_version() == (
        "subject-classifier@champion v1 (alias now points to v2, restart to apply)"
    )
    assert model.loaded.ref.version == "1"
    assert registry.loaded_versions == ["1"]


async def test_fetch_version_round_trips_to_registry() -> None:
    model, registry = make_model_component()
    await model.startup()
    resolved_at_startup = registry.resolved

    assert await model.fetch_version() == "subject-classifier@champion v3"
    assert registry.resolved == resolved_at_startup + 1


async def test_fetch_version_fails_when_registry_is_unreachable() -> None:
    model, registry = make_model_component()
    await model.startup()
    registry.resolve_error = ConnectionError("mlflow down")

    with pytest.raises(ConnectionError, match="mlflow down"):
        await model.fetch_version()


@pytest.mark.parametrize(
    "registry",
    [
        FakeModelRegistry(resolve_error=ConnectionError("refused")),
        FakeModelRegistry(aliases={}),
        FakeModelRegistry(load_error=OSError("artifact missing")),
    ],
    ids=["unreachable", "no-alias", "load-error"],
)
async def test_failed_load_does_not_crash_startup(
    registry: FakeModelRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    from edu_graphrag.logging_config import configure_logging

    configure_logging("INFO", json_logs=True)
    model, _ = make_model_component(registry)

    await model.startup()  # must not raise

    assert not model.is_loaded
    with pytest.raises(ModelNotLoadedError, match="is not loaded"):
        await model.fetch_version()
    with pytest.raises(ModelNotLoadedError):
        await model.predict(["x"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (failure,) = [r for r in records if r["event"] == "model_load_failed"]
    assert failure["level"] == "error"
    assert failure["model_uri"] == "models:/subject-classifier@champion"


async def test_not_started_component_reports_why() -> None:
    model, _ = make_model_component()

    with pytest.raises(ModelNotLoadedError, match="not started"):
        _ = model.loaded


async def test_shutdown_unloads_model_and_releases_registry() -> None:
    model, registry = make_model_component()
    await model.startup()

    await model.shutdown()

    assert not model.is_loaded
    assert registry.closed == 1


async def test_blocking_calls_run_off_the_event_loop_thread() -> None:
    model, registry = make_model_component()
    await model.startup()

    await model.predict(["x"])

    (predict_thread,) = registry.model("3").thread_ids
    assert predict_thread != threading.get_ident()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            pd.DataFrame({"label": ["a"], "confidence": [np.float32(0.5)]}),
            [{"label": "a", "confidence": 0.5}],
        ),
        (np.array([1, 2]), [{"prediction": 1}, {"prediction": 2}]),
        (np.array([[1, 2]]), [{"prediction": [1, 2]}]),
        (pd.Series(["x"]), [{"prediction": "x"}]),
        (["x", None], [{"prediction": "x"}, {"prediction": None}]),
        ([{"a": 1}], [{"a": 1}]),
        (pd.DataFrame({"v": [np.nan]}), [{"v": None}]),
    ],
    ids=["frame", "1d", "2d", "series", "list", "dicts", "nan"],
)
def test_predictions_are_json_rows(raw: Any, expected: list[dict[str, Any]]) -> None:
    assert predictions_to_records(raw) == expected


def test_unsupported_prediction_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported prediction type: str"):
        predictions_to_records("oops")


@pytest.mark.integration
async def test_real_mlflow_registry_alias_switch_and_restart(tmp_path: Path) -> None:
    """Full path against a throwaway sqlite store (no server): register v1 and v2,
    serve champion=v1, move champion to v2, and a fresh component ("restart") loads v2."""
    mlflow = pytest.importorskip("mlflow")  # the full package lives in the `ml` group

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("test", artifact_location=tmp_path.as_uri())

    class Tagger(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"label": [self.tag] * len(model_input)})

    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        for tag in ("v1", "v2"):
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.pyfunc.log_model(
                    name="model", python_model=Tagger(tag), registered_model_name="clf"
                )
    finally:
        mlflow.set_tracking_uri(previous_uri)
    client.set_registered_model_alias("clf", "champion", "1")

    settings = MlflowSettings(mlflow_tracking_uri=tracking_uri, model_name="clf")
    first = MlflowModelComponent(settings)
    await first.startup()
    assert first.is_loaded
    assert await first.predict(["q"]) == [{"label": "v1"}]

    client.set_registered_model_alias("clf", "champion", "2")
    assert "alias now points to v2" in await first.fetch_version()
    await first.shutdown()

    restarted = MlflowModelComponent(settings)
    await restarted.startup()
    try:
        assert restarted.loaded.ref.version == "2"
        assert await restarted.fetch_version() == "clf@champion v2"
        assert await restarted.predict(["q"]) == [{"label": "v2"}]
    finally:
        await restarted.shutdown()
