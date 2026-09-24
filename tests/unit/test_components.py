import pytest
from pydantic import SecretStr

from edu_graphrag.components import build_components, run_components
from edu_graphrag.components.neo4j import Neo4jComponent
from edu_graphrag.components.postgres import PostgresComponent, build_postgres_url
from edu_graphrag.config import Neo4jSettings, PostgresSettings, Settings
from tests.fakes import FakeComponent


async def test_components_start_in_order_and_stop_in_reverse() -> None:
    events: list[str] = []
    components = [FakeComponent("a", events=events), FakeComponent("b", events=events)]

    async with run_components(components):
        assert events == ["start:a", "start:b"]

    assert events == ["start:a", "start:b", "stop:b", "stop:a"]


async def test_started_components_are_stopped_if_later_startup_fails() -> None:
    events: list[str] = []
    components = [
        FakeComponent("a", events=events),
        FakeComponent("b", events=events, startup_error=ValueError("bad config")),
    ]

    with pytest.raises(ValueError, match="bad config"):
        async with run_components(components):
            pytest.fail("must not get here")

    assert events == ["start:a", "stop:a"]


def test_build_components_registers_postgres_and_neo4j(settings: Settings) -> None:
    assert [c.name for c in build_components(settings)] == ["postgres", "neo4j"]


def test_postgres_url_escapes_password_and_uses_async_driver() -> None:
    url = build_postgres_url(
        PostgresSettings(host="db", port=5433, user="u", password=SecretStr("p@ss/word"), db="d")
    )

    assert url.drivername == "postgresql+asyncpg"
    assert url.render_as_string(hide_password=False) == (
        "postgresql+asyncpg://u:p%40ss%2Fword@db:5433/d"
    )
    assert "p@ss" not in str(url)


async def test_postgres_component_lifecycle_without_database() -> None:
    component = PostgresComponent(PostgresSettings(password=SecretStr("x")), connect_timeout=1)
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.engine

    await component.startup()  # lazy: must succeed even though no database is running
    assert component.engine is not None

    await component.shutdown()
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.engine


async def test_neo4j_component_lifecycle_without_database() -> None:
    component = Neo4jComponent(Neo4jSettings(password=SecretStr("x")), connect_timeout=1)
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.driver

    await component.startup()
    assert component.driver is not None

    await component.shutdown()
    with pytest.raises(RuntimeError, match="not started"):
        _ = component.driver
