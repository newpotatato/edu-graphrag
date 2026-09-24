"""FastAPI dependencies: access to objects created at startup."""

from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, Request

from edu_graphrag.components import Component
from edu_graphrag.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_components(request: Request) -> Sequence[Component]:
    components: Sequence[Component] = request.app.state.components
    return components


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ComponentsDep = Annotated[Sequence[Component], Depends(get_components)]
