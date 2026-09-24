"""External dependencies of the application behind a common `Component` contract."""

from edu_graphrag.components.base import Component
from edu_graphrag.components.registry import build_components, run_components

__all__ = ["Component", "build_components", "run_components"]
