import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

from tests.conftest import ClientFactory
from tests.fakes import FakeComponent, FakeModelRegistry, FakePyfuncModel, make_model_component


@pytest.fixture
def model_registry() -> FakeModelRegistry:
    return FakeModelRegistry(aliases={"champion": "3"})


@pytest.fixture
async def model_client(
    make_client: ClientFactory, model_registry: FakeModelRegistry
) -> AsyncIterator[AsyncClient]:
    model, _ = make_model_component(model_registry)
    async with make_client([FakeComponent("postgres"), model]) as client:
        yield client


async def test_process_returns_predictions_and_model_version(model_client: AsyncClient) -> None:
    response = await model_client.post("/api/v1/process", json={"texts": ["algebra", "physics"]})

    assert response.status_code == 200
    assert response.headers["X-Model-Version"] == "3"
    assert response.json() == {
        "model": {
            "name": "subject-classifier",
            "alias": "champion",
            "version": "3",
            "run_id": "run-3",
        },
        "predictions": [
            {"label": "ALGEBRA", "confidence": 0.9},
            {"label": "PHYSICS", "confidence": 0.9},
        ],
    }


async def test_model_receives_text_column(
    model_client: AsyncClient, model_registry: FakeModelRegistry
) -> None:
    await model_client.post("/api/v1/process", json={"texts": ["a", "b"]})

    (frame,) = model_registry.model("3").inputs
    assert list(frame.columns) == ["text"]
    assert frame["text"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    "output",
    [
        lambda frame: np.arange(len(frame)),
        lambda frame: pd.Series(range(len(frame))),
        lambda frame: list(range(len(frame))),
    ],
    ids=["ndarray", "series", "list"],
)
async def test_one_dimensional_output_is_wrapped(
    make_client: ClientFactory, output: Callable[[pd.DataFrame], Any]
) -> None:
    model, _ = make_model_component(FakeModelRegistry(models={"3": FakePyfuncModel(output)}))

    async with make_client([model]) as client:
        response = await client.post("/api/v1/process", json={"texts": ["a", "b"]})

    assert response.status_code == 200
    assert response.json()["predictions"] == [{"prediction": 0}, {"prediction": 1}]


async def test_nested_output_is_returned_as_is(make_client: ClientFactory) -> None:
    def retriever(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [{"results": [{"doc": "d1", "score": 0.5}]} for _ in frame["text"]]

    model, _ = make_model_component(FakeModelRegistry(models={"3": FakePyfuncModel(retriever)}))

    async with make_client([model]) as client:
        response = await client.post("/api/v1/process", json={"texts": ["q"]})

    assert response.json()["predictions"] == [{"results": [{"doc": "d1", "score": 0.5}]}]


async def test_model_is_loaded_once_for_many_requests(
    model_client: AsyncClient, model_registry: FakeModelRegistry
) -> None:
    for _ in range(5):
        response = await model_client.post("/api/v1/process", json={"texts": ["x"]})
        assert response.status_code == 200

    assert model_registry.loaded_versions == ["3"]
    assert len(model_registry.model("3").inputs) == 5


async def test_inference_logs_one_structured_event(
    make_client: ClientFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    model, _ = make_model_component()
    async with make_client([model]) as client:
        await client.post("/api/v1/process", json={"texts": ["a", "b"]})

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (event,) = [r for r in records if r["event"] == "inference_done"]
    assert event["model_version"] == "3"
    assert event["n_inputs"] == 2
    assert event["duration_ms"] >= 0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"texts": []},
        {"texts": [""]},
        {"texts": ["   "]},
        {"texts": ["x" * 20_001]},
        {"texts": ["x"] * 65},
        {"texts": [1]},
        {"texts": ["x"], "unexpected": True},
    ],
    ids=[
        "missing",
        "empty-list",
        "empty-text",
        "blank",
        "too-long",
        "too-many",
        "not-str",
        "extra",
    ],
)
async def test_invalid_request_is_rejected(
    model_client: AsyncClient, model_registry: FakeModelRegistry, body: dict[str, object]
) -> None:
    response = await model_client.post("/api/v1/process", json=body)

    assert response.status_code == 422
    assert model_registry.model("3").inputs == []


async def test_limits_are_inclusive(model_client: AsyncClient) -> None:
    response = await model_client.post("/api/v1/process", json={"texts": ["x" * 20_000] * 64})

    assert response.status_code == 200


async def test_process_is_503_when_model_failed_to_load(make_client: ClientFactory) -> None:
    model, _ = make_model_component(FakeModelRegistry(resolve_error=ConnectionError("refused")))

    async with make_client([model]) as client:
        healthz = await client.get("/healthz")
        version = await client.get("/api/v1/version")
        response = await client.post("/api/v1/process", json={"texts": ["x"]})

    assert healthz.status_code == 200, "the app must start without a model"
    assert version.status_code == 200
    assert response.status_code == 503
    assert "models:/subject-classifier@champion is not loaded" in response.json()["detail"]


async def test_process_is_503_when_serving_is_disabled(client: AsyncClient) -> None:
    # The default `components` fixture has no model component.
    response = await client.post("/api/v1/process", json={"texts": ["x"]})

    assert response.status_code == 503
    assert "MODEL_ENABLED" in response.json()["detail"]


async def test_inference_error_gives_500_without_internals(make_client: ClientFactory) -> None:
    broken = FakePyfuncModel(error=ValueError("secret internals"))
    model, _ = make_model_component(FakeModelRegistry(models={"3": broken}))

    async with make_client([model]) as client:
        response = await client.post("/api/v1/process", json={"texts": ["x"]})

    assert response.status_code == 500
    assert response.json() == {"detail": "Inference failed"}


async def test_model_endpoint_reports_loaded_version(model_client: AsyncClient) -> None:
    response = await model_client.get("/api/v1/model")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "subject-classifier"
    assert body["alias"] == "champion"
    assert body["version"] == "3"
    assert body["run_id"] == "run-3"
    assert body["model_uri"] == "models:/subject-classifier/3"
    assert body["loaded_at"].endswith(("Z", "+00:00"))


async def test_model_endpoint_is_503_without_model(make_client: ClientFactory) -> None:
    model, _ = make_model_component(FakeModelRegistry(aliases={}))

    async with make_client([model]) as client:
        response = await client.get("/api/v1/model")

    assert response.status_code == 503


async def test_health_shows_served_model_version(model_client: AsyncClient) -> None:
    response = await model_client.get("/api/v1/health")

    components = {c["name"]: c for c in response.json()["components"]}
    assert components["mlflow_model"]["status"] == "up"
    assert components["mlflow_model"]["version"] == "subject-classifier@champion v3"


async def test_health_shows_model_down_after_failed_load(make_client: ClientFactory) -> None:
    model, _ = make_model_component(FakeModelRegistry(aliases={}))

    async with make_client([model]) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 503
    (component,) = response.json()["components"]
    assert component["status"] == "down"
    assert component["error"].startswith("ModelNotLoadedError")
    assert "alias 'champion' not found" in component["error"]
