from fastapi import APIRouter, Response, status

from edu_graphrag.api.deps import ComponentsDep, SettingsDep
from edu_graphrag.schemas import HealthResponse
from edu_graphrag.services.health import check_components

router = APIRouter(tags=["meta"])


@router.get(
    "/health",
    summary="End-to-end health check",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "At least one component is down",
        },
    },
)
async def health(
    response: Response,
    settings: SettingsDep,
    components: ComponentsDep,
) -> HealthResponse:
    """Real round trip to every component: its version and response time.

    Returns 503 if anything is down; the body always contains per-component details.
    """
    report = await check_components(components, per_check_timeout=settings.health_check_timeout)
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
