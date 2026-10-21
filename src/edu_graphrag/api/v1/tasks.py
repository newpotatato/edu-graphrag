"""Asynchronous tasks: submit a text for classification, then poll for the result.

Inference runs in the workers behind the `tasks` queue, not in the API. The API only accepts
work (202) or sheds it (503 + Retry-After) - see `api.task_service` for the flow and why this
is load shedding rather than backpressure.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from edu_graphrag.api.deps import TaskServiceDep
from edu_graphrag.api.task_service import (
    STORE_RETRY_AFTER,
    StoreUnavailableError,
    TaskRejectedError,
)
from edu_graphrag.schemas import Problem, TaskAccepted, TaskCreateRequest, TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])

PROBLEM_JSON = "application/problem+json"

_RETRY_AFTER_HEADER = {
    "Retry-After": {
        "description": "Seconds until a retry has a chance to be accepted",
        "schema": {"type": "integer"},
    }
}
_503: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": Problem,
        "description": "Overloaded (queue full) or a dependency is down; nothing was stored",
        "headers": _RETRY_AFTER_HEADER,
        "content": {
            PROBLEM_JSON: {
                "example": {
                    "type": "about:blank",
                    "title": "Task queue is full",
                    "status": 503,
                    "detail": "The task was not accepted (PublishRejectedError); nothing was "
                    "stored. Retry after 2 s.",
                    "retry_after": 2,
                }
            }
        },
    }
}


def problem_response(title: str, detail: str, retry_after: int) -> JSONResponse:
    body = Problem(title=title, status=503, detail=detail, retry_after=retry_after)
    return JSONResponse(
        body.model_dump(),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": str(retry_after)},
        media_type=PROBLEM_JSON,
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a text for classification",
    response_model=TaskAccepted,
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Accepted: the task is in the queue; poll `Location`",
            "headers": {
                "Location": {"description": "URL of the task status", "schema": {"type": "string"}}
            },
        },
        **_503,
    },
)
async def create_task(
    payload: TaskCreateRequest, service: TaskServiceDep, request: Request, response: Response
) -> TaskAccepted | JSONResponse:
    """Returns immediately: 202 once the broker has confirmed the task, otherwise 503.

    The request never waits for space in the queue. On 503 nothing is stored; retry after the
    number of seconds in `Retry-After` (an estimate of when the workers will have freed
    enough of the queue for it to accept again).
    """
    try:
        message = await service.submit(payload.text)
    except TaskRejectedError as exc:
        return problem_response(exc.title, exc.detail, exc.retry_after)
    url = request.app.url_path_for("get_task", task_id=str(message.task_id))
    response.headers["Location"] = url
    return TaskAccepted(task_id=message.task_id, status_url=url)


@router.get(
    "/{task_id}",
    name="get_task",
    summary="Status and result of a task",
    response_model=TaskStatusResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No such task"},
        **_503,
    },
)
async def get_task(task_id: UUID, service: TaskServiceDep) -> TaskStatusResponse | JSONResponse:
    """`pending` while queued or running, then `done` (with the result and the model version
    that produced it) or `failed` (dead-lettered after the delivery limit, with the error)."""
    try:
        record = await service.get(task_id)
    except StoreUnavailableError:
        return problem_response(
            "Task store is unavailable", "The database is unavailable.", STORE_RETRY_AFTER
        )
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Task {task_id} not found")
    processing_ms = None
    if record.finished_at is not None:
        processing_ms = round((record.finished_at - record.submitted_at).total_seconds() * 1000, 1)
    return TaskStatusResponse(
        task_id=record.task_id,
        status=record.status,
        source=record.source,
        result=record.result,
        error=record.error,
        model_name=record.model_name,
        model_version=record.model_version,
        attempts=record.attempts,
        submitted_at=record.submitted_at,
        finished_at=record.finished_at,
        processing_ms=processing_ms,
    )
