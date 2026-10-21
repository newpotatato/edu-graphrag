"""Application factory. Run with `uvicorn edu_graphrag.main:create_app --factory`."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from edu_graphrag import __version__
from edu_graphrag.api import system, v1
from edu_graphrag.api.task_service import PostgresTaskStore, TaskService
from edu_graphrag.components import Component, build_components, run_components
from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.components.rabbit import RabbitComponent
from edu_graphrag.config import Settings, get_settings
from edu_graphrag.logging_config import configure_logging
from edu_graphrag.middleware import RequestContextMiddleware

logger = structlog.get_logger(__name__)


def build_task_service(settings: Settings, components: Sequence[Component]) -> TaskService | None:
    """Wires POST/GET /api/v1/tasks to the Postgres and RabbitMQ components, if present."""
    postgres = next((c for c in components if isinstance(c, PostgresComponent)), None)
    rabbit = next((c for c in components if isinstance(c, RabbitComponent)), None)
    if postgres is None or rabbit is None:
        return None
    return TaskService(
        PostgresTaskStore(postgres),
        rabbit,
        pipeline=settings.pipeline,
        broker_retry_after=max(1, round(rabbit.reconnect_interval)),
    )


def create_app(
    settings: Settings | None = None,
    components: Sequence[Component] | None = None,
    task_service: TaskService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    if components is None:
        components = build_components(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with run_components(components):
            logger.info("app_started", version=__version__, commit=settings.app_commit_sha)
            yield
        logger.info("app_stopped")

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.components = components
    app.state.task_service = task_service or build_task_service(settings, components)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(system.router)
    app.include_router(v1.router)
    return app
