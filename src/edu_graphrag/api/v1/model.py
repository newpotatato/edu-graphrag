from fastapi import APIRouter, status

from edu_graphrag.api.deps import ModelDep
from edu_graphrag.schemas import ModelInfo

router = APIRouter(tags=["inference"])


@router.get(
    "/model",
    summary="Model version being served",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No model is loaded"}},
)
async def get_model_info(model: ModelDep) -> ModelInfo:
    """The concrete registry version the alias resolved to when the app started."""
    loaded = model.loaded
    return ModelInfo(
        name=loaded.ref.name,
        alias=loaded.ref.alias,
        version=loaded.ref.version,
        run_id=loaded.ref.run_id,
        model_uri=loaded.ref.model_uri,
        loaded_at=loaded.loaded_at,
    )
