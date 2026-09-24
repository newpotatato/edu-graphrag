from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager

import structlog

from edu_graphrag.components.base import Component
from edu_graphrag.components.neo4j import Neo4jComponent
from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.config import Settings

logger = structlog.get_logger(__name__)


def build_components(settings: Settings) -> list[Component]:
    timeout = settings.health_check_timeout
    return [
        PostgresComponent(settings.postgres, connect_timeout=timeout),
        Neo4jComponent(settings.neo4j, connect_timeout=timeout),
    ]


@asynccontextmanager
async def run_components(components: Sequence[Component]) -> AsyncIterator[None]:
    """Start components in order and shut them down in reverse order,
    including the already started ones if a later startup fails."""
    async with AsyncExitStack() as stack:
        for component in components:
            await component.startup()
            stack.push_async_callback(component.shutdown)
            logger.info("component_started", component=component.name)
        yield
    logger.info("components_stopped")
