import json
import re

import pytest
from httpx import AsyncClient

from tests.conftest import ClientFactory
from tests.fakes import FakeComponent


async def test_healthz_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_does_not_depend_on_components(make_client: ClientFactory) -> None:
    broken = FakeComponent("postgres", error=ConnectionRefusedError("down"))

    async with make_client([broken]) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert broken.checks == 0


async def test_healthz_is_not_versioned(client: AsyncClient) -> None:
    response = await client.get("/api/v1/healthz")

    assert response.status_code == 404


async def test_request_id_is_generated(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


async def test_incoming_request_id_is_propagated(client: AsyncClient) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "trace-42"})

    assert response.headers["X-Request-ID"] == "trace-42"


@pytest.mark.parametrize("bad_id", ["has spaces", "x" * 65, 'quote"inside'])
async def test_unsafe_request_id_is_replaced(client: AsyncClient, bad_id: str) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": bad_id})

    assert response.headers["X-Request-ID"] != bad_id


async def test_access_log_is_structured(
    make_client: ClientFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    # The app is created inside the test so its log handler writes to the captured stdout.
    async with make_client([]) as client:
        await client.get("/healthz", headers={"X-Request-ID": "trace-7"})

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    access = next(r for r in records if r["event"] == "request_finished")
    assert access["request_id"] == "trace-7"
    assert access["path"] == "/healthz"
    assert access["status_code"] == 200
    assert access["duration_ms"] >= 0
