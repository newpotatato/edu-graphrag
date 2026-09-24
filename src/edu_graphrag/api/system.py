"""Infrastructure endpoints. Not versioned: probes and load balancers rely on a stable path."""

from fastapi import APIRouter

from edu_graphrag.schemas import LivenessResponse

router = APIRouter(tags=["system"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> LivenessResponse:
    """Answers as soon as the process can serve requests.

    Deliberately does not touch external dependencies: a DB outage must not make
    the orchestrator restart a healthy app. Dependencies are checked by /api/v1/health.
    """
    return LivenessResponse()
