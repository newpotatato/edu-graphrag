import time

from httpx import AsyncClient

from edu_graphrag.services.health import check_component, check_components
from tests.conftest import ClientFactory
from tests.fakes import FakeComponent


async def test_all_components_up(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert [(c["name"], c["status"], c["version"]) for c in body["components"]] == [
        ("postgres", "up", "17.6"),
        ("neo4j", "up", "5.26.0"),
    ]
    assert all(c["latency_ms"] >= 0 and c["error"] is None for c in body["components"])


async def test_failing_component_gives_503_with_details(make_client: ClientFactory) -> None:
    components = [
        FakeComponent("postgres", "17.6"),
        FakeComponent("neo4j", error=ConnectionRefusedError("connection refused")),
    ]

    async with make_client(components) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    postgres, neo4j = body["components"]
    assert postgres["status"] == "up"
    assert neo4j == {
        "name": "neo4j",
        "status": "down",
        "version": None,
        "latency_ms": neo4j["latency_ms"],
        "error": "ConnectionRefusedError: connection refused",
    }


async def test_slow_component_times_out(make_client: ClientFactory) -> None:
    # settings.health_check_timeout is 0.5s in tests
    async with make_client([FakeComponent("neo4j", delay=5)]) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 503
    (neo4j,) = response.json()["components"]
    assert neo4j["status"] == "down"
    assert neo4j["error"] == "timeout after 0.5s"
    assert 450 <= neo4j["latency_ms"] < 1500


async def test_checks_run_concurrently() -> None:
    components = [FakeComponent(f"c{i}", delay=0.2) for i in range(3)]

    started = time.perf_counter()
    report = await check_components(components, per_check_timeout=1)
    elapsed = time.perf_counter() - started

    assert report.status == "ok"
    assert elapsed < 0.5, "checks must not run one after another"


async def test_latency_is_measured() -> None:
    result = await check_component(FakeComponent("pg", delay=0.1), per_check_timeout=1)

    assert result.status == "up"
    assert 90 <= result.latency_ms < 1000


async def test_no_components_is_ok() -> None:
    report = await check_components([], per_check_timeout=1)

    assert report.status == "ok"
    assert report.components == []
