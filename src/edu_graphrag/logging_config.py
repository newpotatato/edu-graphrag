"""Structured logging: structlog for the app, stdlib loggers (uvicorn, drivers) routed through it."""

import logging
import os
import re
import socket
import sys
import traceback
from pathlib import Path

import structlog
from opentelemetry.trace import get_current_span
from structlog.typing import EventDict, Processor, WrappedLogger

_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s'\"<>]+@")


def _correlate(_logger: WrappedLogger, _method: str, event: EventDict) -> EventDict:
    ctx = get_current_span().get_span_context()
    if ctx.is_valid:
        event["trace_id"] = f"{ctx.trace_id:032x}"
        event["span_id"] = f"{ctx.span_id:016x}"
    return event


def _safe_exception(_logger: WrappedLogger, _method: str, event: EventDict) -> EventDict:
    info = event.pop("exc_info", None)
    if info is True:
        info = sys.exc_info()
    if isinstance(info, BaseException):
        info = (type(info), info, info.__traceback__)
    if isinstance(info, tuple) and info[0] is not None:
        event["exception"] = {
            "type": info[0].__name__,
            "frames": [
                {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(info[2])
            ],
        }
    for key, value in event.items():
        if isinstance(value, str):
            event[key] = _USERINFO.sub(r"\1[redacted]@", value)
    return event


def configure_logging(
    level: str = "INFO", *, json_logs: bool = True, service: str = "edu-api"
) -> None:
    attributes = dict(
        part.split("=", 1)
        for part in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").split(",")
        if "=" in part
    )
    defaults = {
        "service": os.environ.get("OTEL_SERVICE_NAME", service),
        "env": attributes.get("deployment.environment.name", "local"),
        "instance_id": attributes.get("service.instance.id", socket.gethostname()),
    }

    def add_resource(_logger: WrappedLogger, _method: str, event: EventDict) -> EventDict:
        return {**defaults, **event}

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_resource,
        _correlate,
        _safe_exception,
    ]
    renderer: Processor
    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Uvicorn installs its own handlers; route its records through the root handler instead.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # Access logs are written by RequestContextMiddleware with more context.
    logging.getLogger("uvicorn.access").disabled = True
