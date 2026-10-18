"""FastAPI dependencies: access to objects created at startup."""

from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from edu_graphrag.components import Component, MlflowModelComponent
from edu_graphrag.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_components(request: Request) -> Sequence[Component]:
    components: Sequence[Component] = request.app.state.components
    return components


ComponentsDep = Annotated[Sequence[Component], Depends(get_components)]


def get_model(components: ComponentsDep) -> MlflowModelComponent:
    """The served model, or 503 if serving is disabled or the model failed to load."""
    model = next((c for c in components if isinstance(c, MlflowModelComponent)), None)
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Model serving is disabled (MODEL_ENABLED=false)"
        )
    if not model.is_loaded:
        # The reason (e.g. MLflow unreachable, alias missing) is in the `model_load_failed` log
        # and in /api/v1/health.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Model {model.uri} is not loaded; fix the registry and restart the app",
        )
    return model


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ModelDep = Annotated[MlflowModelComponent, Depends(get_model)]
