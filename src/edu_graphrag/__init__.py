"""Multimodal GraphRAG aggregator of study materials."""

from importlib.metadata import version

# Single source of truth for the application version is `project.version`
# in pyproject.toml. It is read from the installed package metadata, so the
# package must be installed (uv does this for local runs and in the image).
__version__ = version("edu-graphrag")
