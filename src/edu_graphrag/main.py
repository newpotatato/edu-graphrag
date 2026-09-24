"""Application factory. Run with `uvicorn edu_graphrag.main:create_app --factory`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from edu_graphrag import __version__
from edu_graphrag.api import system, v1
from edu_graphrag.config import Settings, get_settings
from edu_graphrag.logging_config import configure_logging
from edu_graphrag.middleware import RequestContextMiddleware

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("app_started", version=__version__, commit=settings.app_commit_sha)
        yield
        logger.info("app_stopped")

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.include_router(system.router)
    app.include_router(v1.router)
    return app
