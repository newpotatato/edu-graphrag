"""Process-local OTel SDK wiring and shared message boundaries (no backend dependency)."""

import asyncio
import os
import socket
import time
import traceback
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from opentelemetry import context, metrics, trace
from opentelemetry.metrics import CallbackOptions, Meter, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from opentelemetry.trace import get_current_span as get_current_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from edu_graphrag import __version__

_PROPAGATOR = TraceContextTextMapPropagator()
SENT_AT = "edu-published-at-ns"
HTTP_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0)
TASK_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0)
Attrs = Mapping[str, str | int | float | bool]


@dataclass(frozen=True)
class Runtime:
    tracer: Tracer
    meter: Meter


_NOOP = Runtime(trace.NoOpTracer(), metrics.NoOpMeter("edu_graphrag"))
_runtime = _NOOP


@contextmanager
def use_runtime(runtime: Runtime) -> Iterator[None]:
    """One per process; tests can supply in-memory providers without OTel globals."""
    global _runtime
    previous, _runtime = _runtime, runtime
    try:
        yield
    finally:
        _runtime = previous


def _rss(_options: CallbackOptions) -> Iterable[Observation]:
    # Containers run Linux. VmRSS is current RSS, unlike ru_maxrss (a high-water mark).
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return (Observation(int(line.split()[1]) * 1024),)
    except OSError:
        pass
    return ()


@asynccontextmanager
async def running(service: str) -> AsyncIterator[None]:
    if os.environ.get("OTEL_SDK_DISABLED", "true").lower() == "true" or all(
        os.environ.get(f"OTEL_{signal}_EXPORTER", "none") != "otlp"
        for signal in ("TRACES", "METRICS")
    ):
        yield
        return

    # OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES override these defaults. Docker assigns
    # each worker a different hostname; never configure a fixed hostname on scaled workers.
    resource = (
        Resource.create()
        .merge(
            Resource(
                {
                    "service.name": service,
                    "service.version": __version__,
                    "service.instance.id": socket.gethostname(),
                    "deployment.environment.name": "local",
                }
            )
        )
        .merge(OTELResourceDetector().detect())
    )
    traces: TracerProvider | None = None
    meters: MeterProvider | None = None
    if os.environ.get("OTEL_TRACES_EXPORTER", "none") == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        traces = TracerProvider(resource=resource, shutdown_on_exit=False)
        traces.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if os.environ.get("OTEL_METRICS_EXPORTER", "none") == "otlp":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        meters = MeterProvider(resource=resource, metric_readers=[reader], shutdown_on_exit=False)
    tracer = traces.get_tracer("edu_graphrag") if traces is not None else _NOOP.tracer
    meter = meters.get_meter("edu_graphrag") if meters is not None else _NOOP.meter
    meter.create_observable_gauge("edu_process_rss", callbacks=[_rss], unit="By")
    try:
        with use_runtime(Runtime(tracer, meter)):
            yield
    finally:
        # MetricReader.shutdown exports one last collection even if the batch ran for less
        # than the interval. Trace export uses a bounded queue and an HTTP deadline.
        if meters is not None:
            await asyncio.to_thread(meters.shutdown, timeout_millis=2500)
        if traces is not None:
            await asyncio.to_thread(traces.shutdown)


@contextmanager
def span(
    name: str,
    attrs: Attrs | None = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: context.Context | None = None,
) -> Iterator[Span]:
    with _runtime.tracer.start_as_current_span(
        name,
        context=parent,
        kind=kind,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as exc:
            failure(current, exc.__cause__ or exc)
            raise


def failure(current: Span, exc: BaseException | None = None, *, reason: str = "") -> None:
    current.set_status(Status(StatusCode.ERROR, reason or None))
    if exc is not None and current.is_recording():
        # Do not export str(exc) (DB exceptions can include bind params, URLs or input).
        # traceback.format_exception also includes source lines and chained messages. Record
        # only frame locations and exception type under the standard OTel event keys.
        stack = "\n".join(
            f"{Path(frame.f_code.co_filename).name}:{line} in {frame.f_code.co_name}"
            for frame, line in traceback.walk_tb(exc.__traceback__)
        )
        current.add_event(
            "exception",
            {"exception.type": type(exc).__name__, "exception.stacktrace": stack},
        )


def propagate_headers() -> dict[str, str]:
    headers = {SENT_AT: str(time.time_ns())}
    _PROPAGATOR.inject(headers)
    return headers


def extract_headers(headers: Mapping[str, Any]) -> context.Context:
    # A fresh base context prevents missing or invalid headers from attaching to unrelated work.
    carrier: dict[str, str] = {}
    for key in ("traceparent", "tracestate"):
        value = headers.get(key)
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        if isinstance(value, str):
            carrier[key] = value
    return _PROPAGATOR.extract(carrier, context=context.Context())


def delivery_age(headers: Mapping[str, Any]) -> float | None:
    try:
        age = (time.time_ns() - int(headers.get(SENT_AT, 0))) / 1e9
    except (TypeError, ValueError, OverflowError):
        return None
    return age if 0 <= age <= 86400 else None


def task_uuid(raw: object) -> str | None:
    try:
        return str(UUID(str(raw)))
    except (ValueError, TypeError):
        return None


def bind_task(task_id: str, *, attempt: int | None = None) -> None:
    structlog.contextvars.bind_contextvars(task_id=task_id)
    if attempt is not None:
        structlog.contextvars.bind_contextvars(attempt=attempt)
    trace.get_current_span().set_attribute("task.id", task_id)


def count(
    name: str, amount: int | float = 1, attrs: Attrs | None = None, *, unit: str = "1"
) -> None:
    _runtime.meter.create_counter(name, unit=unit).add(amount, attrs)


def gauge(name: str, value: int | float, attrs: Attrs | None = None) -> None:
    _runtime.meter.create_gauge(name).set(value, attrs)


def observe(
    name: str, seconds: float, buckets: tuple[float, ...], attrs: Attrs | None = None
) -> None:
    _runtime.meter.create_histogram(
        name,
        unit="s",
        explicit_bucket_boundaries_advisory=buckets,
    ).record(seconds, attrs)


def inference_buckets(delay_s: float) -> tuple[float, ...]:
    baseline = delay_s or 0.01
    return tuple(
        sorted(
            {round(baseline * factor, 6) for factor in (0.25, 0.5, 0.75, 1, 1.25, 1.75, 2.5, 5, 10)}
        )
    )
