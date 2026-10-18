"""Inference with the model loaded at startup.

The homework's `POST /process` lives under the API version prefix (`/api/v1/process`),
like every other public endpoint of the project; only infrastructure probes are unversioned.
"""

import time

import structlog
from fastapi import APIRouter, HTTPException, Response, status

from edu_graphrag.api.deps import ModelDep
from edu_graphrag.schemas import ModelRef, ProcessRequest, ProcessResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["inference"])

MODEL_VERSION_HEADER = "X-Model-Version"


@router.post(
    "/process",
    summary="Run the served model on a batch of texts",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No model is loaded"}},
)
async def process(payload: ProcessRequest, model: ModelDep, response: Response) -> ProcessResponse:
    """Runs the model loaded from MLflow at startup; nothing is loaded per request.

    The model receives a DataFrame with a `text` column. Its output is returned as one object
    per row, so any registered model with that input works without changing this code.
    The served registry version is in the body and in the `X-Model-Version` header.
    """
    ref = model.loaded.ref
    started = time.perf_counter()
    try:
        predictions = await model.predict(payload.texts)
    except Exception as exc:
        logger.exception("inference_failed", model_version=ref.version, n_inputs=len(payload.texts))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Inference failed") from exc
    logger.info(
        "inference_done",
        model_name=ref.name,
        model_version=ref.version,
        n_inputs=len(payload.texts),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    response.headers[MODEL_VERSION_HEADER] = ref.version
    return ProcessResponse(
        model=ModelRef(name=ref.name, alias=ref.alias, version=ref.version, run_id=ref.run_id),
        predictions=predictions,
    )
