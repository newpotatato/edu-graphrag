"""Public API, version 1."""

from fastapi import APIRouter

from edu_graphrag.api.v1 import health, version

router = APIRouter(prefix="/api/v1")
router.include_router(version.router)
router.include_router(health.router)
