"""End-to-end health check of external components."""

import asyncio
import time
from collections.abc import Sequence

import structlog

from edu_graphrag.components import Component
from edu_graphrag.schemas import ComponentHealth, HealthResponse

logger = structlog.get_logger(__name__)


async def check_component(component: Component, per_check_timeout: float) -> ComponentHealth:
    version: str | None = None
    error: str | None = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(per_check_timeout):
            version = await component.fetch_version()
    except TimeoutError:
        error = f"timeout after {per_check_timeout}s"
    except Exception as exc:  # a health check reports any failure instead of crashing
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    result = ComponentHealth(
        name=component.name,
        status="up" if error is None else "down",
        version=version,
        latency_ms=latency_ms,
        error=error,
    )
    log = logger.info if error is None else logger.warning
    log("component_checked", component=result.name, **result.model_dump(exclude={"name"}))
    return result


async def check_components(
    components: Sequence[Component], per_check_timeout: float
) -> HealthResponse:
    """Checks run concurrently, so the response time is bounded by the slowest one."""
    results = await asyncio.gather(*(check_component(c, per_check_timeout) for c in components))
    all_up = all(r.status == "up" for r in results)
    return HealthResponse(status="ok" if all_up else "degraded", components=list(results))
