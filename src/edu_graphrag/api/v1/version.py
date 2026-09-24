from fastapi import APIRouter

from edu_graphrag import __version__
from edu_graphrag.api.deps import SettingsDep
from edu_graphrag.schemas import VersionResponse

router = APIRouter(tags=["meta"])


@router.get("/version", summary="Application version")
async def get_version(settings: SettingsDep) -> VersionResponse:
    """Version of the application itself (not of the API: `v1` is the API contract version).

    The value is read from package metadata, i.e. from `project.version` in pyproject.toml,
    and the CD pipeline refuses to publish an image whose git tag differs from it.
    """
    return VersionResponse(
        name=settings.app_name,
        version=__version__,
        commit=settings.app_commit_sha,
    )
