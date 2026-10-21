from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from edu_graphrag.api.task_service import TaskService
from edu_graphrag.components import Component
from edu_graphrag.config import (
    MlflowSettings,
    Neo4jSettings,
    PostgresSettings,
    RabbitSettings,
    Settings,
)
from edu_graphrag.main import create_app
from tests.fakes import FakeComponent


class ClientFactory(Protocol):
    def __call__(
        self, components: Sequence[Component], task_service: TaskService | None = None
    ) -> AbstractAsyncContextManager[AsyncClient]: ...


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        help="run integration tests against real Postgres and Neo4j (see docker-compose.yml)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="integration test: run with --integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_commit_sha="test-sha",
        health_check_timeout=0.5,
        postgres=PostgresSettings(password=SecretStr("test")),
        neo4j=Neo4jSettings(password=SecretStr("test")),
        mlflow=MlflowSettings(mlflow_tracking_uri="http://mlflow.invalid"),
        rabbit=RabbitSettings(password=SecretStr("test")),
    )


@pytest.fixture
def make_client(settings: Settings) -> ClientFactory:
    """Builds the app with the given components and runs its lifespan."""

    @asynccontextmanager
    async def factory(
        components: Sequence[Component], task_service: TaskService | None = None
    ) -> AsyncIterator[AsyncClient]:
        app = create_app(settings, components, task_service)
        transport = ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=transport, base_url="http://testserver") as client,
        ):
            yield client

    return factory


@pytest.fixture
def components() -> list[FakeComponent]:
    return [FakeComponent("postgres", "17.6"), FakeComponent("neo4j", "5.26.0")]


@pytest.fixture
async def client(
    make_client: ClientFactory, components: list[FakeComponent]
) -> AsyncIterator[AsyncClient]:
    async with make_client(components) as client:
        yield client
