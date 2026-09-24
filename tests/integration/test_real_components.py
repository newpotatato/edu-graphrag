"""Run against real databases: `just up` (or CI services), then `pytest --integration`."""

import pytest
from httpx import ASGITransport, AsyncClient

from edu_graphrag.components.neo4j import Neo4jComponent
from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.config import Settings
from edu_graphrag.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def real_settings() -> Settings:
    return Settings()  # POSTGRES_* / NEO4J_* from the environment or .env


async def test_postgres_returns_server_version(real_settings: Settings) -> None:
    component = PostgresComponent(real_settings.postgres, connect_timeout=5)
    await component.startup()
    try:
        version = await component.fetch_version()
    finally:
        await component.shutdown()

    assert version[0].isdigit()


async def test_neo4j_returns_server_version(real_settings: Settings) -> None:
    component = Neo4jComponent(real_settings.neo4j, connect_timeout=5)
    await component.startup()
    try:
        version = await component.fetch_version()
    finally:
        await component.shutdown()

    assert version[0].isdigit()


async def test_health_end_to_end(real_settings: Settings) -> None:
    app = create_app(real_settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert {c["name"] for c in body["components"]} == {"postgres", "neo4j"}
    assert all(c["version"] for c in body["components"])
