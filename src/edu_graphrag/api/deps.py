"""FastAPI dependencies: access to objects created at startup."""

from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from edu_graphrag.api.task_service import TaskService
from edu_graphrag.components import Component
from edu_graphrag.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_components(request: Request) -> Sequence[Component]:
    components: Sequence[Component] = request.app.state.components
    return components


def get_task_service(request: Request) -> TaskService:
    """Built at app creation from the Postgres and RabbitMQ components."""
    service: TaskService | None = request.app.state.task_service
    if service is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Task pipeline is not configured (needs the postgres and rabbitmq components)",
        )
    return service


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ComponentsDep = Annotated[Sequence[Component], Depends(get_components)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
