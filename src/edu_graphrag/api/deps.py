"""FastAPI dependencies: access to objects created at startup."""

from typing import Annotated

from fastapi import Depends, Request

from edu_graphrag.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
