"""HTTP telemetry and request context for the API."""

import re
import time
import uuid

import structlog
from opentelemetry.trace import SpanKind
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from edu_graphrag import telemetry

logger = structlog.get_logger(__name__)
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_PROBES = frozenset({"/healthz", "/api/v1/health"})
_METHODS = frozenset({"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"})


class RequestContextMiddleware:
    """Pure ASGI middleware so contextvars reach the handler and the access log."""

    def __init__(self, app: ASGIApp, *, api_prefix: str = "") -> None:
        self.app = app
        self.api_prefix = api_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        incoming = headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        method = scope["method"] if scope["method"] in _METHODS else "OTHER"
        probe = scope["path"] in _PROBES
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        status_code = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            if probe:
                await self.app(scope, receive, send_with_request_id)
                return
            parent = telemetry.extract_headers(dict(headers))
            with telemetry.span(method, kind=SpanKind.SERVER, parent=parent) as current:
                try:
                    await self.app(scope, receive, send_with_request_id)
                finally:
                    route = getattr(scope.get("route"), "path", "unmatched")
                    if (
                        route != "unmatched"
                        and self.api_prefix
                        and scope["path"].startswith(self.api_prefix + "/")
                        and not route.startswith(self.api_prefix + "/")
                    ):
                        route = self.api_prefix + route
                    elapsed = time.perf_counter() - started
                    current.update_name(f"{method} {route}")
                    current.set_attributes(
                        {
                            "http.request.method": method,
                            "http.route": route,
                            "http.response.status_code": status_code,
                        }
                    )
                    if status_code >= 500:
                        telemetry.failure(current, reason=f"HTTP {status_code}")
                    labels = {"route": route, "method": method, "status_code": status_code}
                    telemetry.count("edu_api_requests", attrs=labels)
                    telemetry.observe(
                        "edu_api_request_duration", elapsed, telemetry.HTTP_BUCKETS, labels
                    )
                    logger.info(
                        "request_finished",
                        method=method,
                        path=route,
                        status_code=status_code,
                        duration_ms=round(elapsed * 1000, 2),
                    )
        finally:
            structlog.contextvars.clear_contextvars()
