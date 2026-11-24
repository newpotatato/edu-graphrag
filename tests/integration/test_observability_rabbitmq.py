"""Test the W3C carrier and quorum redeliveries against an actual AMQP broker."""

import asyncio
from collections.abc import Sequence

import pytest
from faststream import TestApp
from opentelemetry.metrics import NoOpMeter
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from edu_graphrag import telemetry
from edu_graphrag.api.task_service import TaskService
from edu_graphrag.components.rabbit import RabbitComponent
from edu_graphrag.config import RabbitSettings, Settings
from edu_graphrag.pipeline.admin import ManagementClient
from tests.conftest import ClientFactory
from tests.fakes import InMemoryTaskStore
from tests.integration.test_backpressure_rabbitmq import rabbit_settings as rabbit_settings
from tests.unit.pipeline.test_worker import make_worker
from tests.unit.test_telemetry import span_context

pytestmark = pytest.mark.integration


async def test_one_trace_survives_amqp_retry_and_dlx(
    rabbit_settings: RabbitSettings,
    settings: Settings,
    make_client: ClientFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    management = ManagementClient(rabbit_settings)
    await asyncio.to_thread(management.declare)
    await asyncio.to_thread(management.apply_policy, max_length=200, delivery_limit=3)

    loop = asyncio.get_running_loop()
    exported = asyncio.Event()

    class NotifyingExporter(InMemorySpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            result = super().export(spans)
            loop.call_soon_threadsafe(exported.set)
            return result

    exporter = NotifyingExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    publisher = RabbitComponent(rabbit_settings, connect_timeout=5)
    worker, registry, _ = make_worker(
        settings.model_copy(update={"rabbit": rabbit_settings}), dlq_recorder=True
    )
    service = TaskService(InMemoryTaskStore(), publisher, pipeline=settings.pipeline)

    async def wait_for_span(name: str) -> None:
        async with asyncio.timeout(20):
            while True:
                exported.clear()
                if any(span.name == name for span in exporter.get_finished_spans()):
                    return
                await exported.wait()

    try:
        with telemetry.use_runtime(
            telemetry.Runtime(provider.get_tracer("test"), NoOpMeter("test"))
        ):
            async with TestApp(worker.app), make_client([publisher], service) as api:
                response = await api.post("/api/v1/tasks", json={"text": "ordinary fixture"})
                assert response.status_code == 202
                await wait_for_span("process")
                good = exporter.get_finished_spans()
                assert {span.name for span in good} >= {
                    "POST /api/v1/tasks",
                    "publish",
                    "process",
                    "inference",
                    "db.save_result",
                }
                assert len({span_context(span).trace_id for span in good}) == 1
                exporter.clear()

                registry.models["3"].error = RuntimeError("poison fixture")
                response = await api.post("/api/v1/tasks", json={"text": "retryable fixture"})
                assert response.status_code == 202
                await wait_for_span("dead_letter")
                spans = exporter.get_finished_spans()
                assert len({span_context(span).trace_id for span in spans}) == 1
                deliveries = [span for span in spans if span.name == "process"]
                attempts = [
                    (span.attributes or {})["messaging.delivery.attempt"] for span in deliveries
                ]
                assert attempts == [1, 2, 3, 4]
                assert all(span.status.status_code == StatusCode.ERROR for span in deliveries)
                (dead_letter,) = [span for span in spans if span.name == "dead_letter"]
                assert (dead_letter.attributes or {})[
                    "messaging.dead_letter.reason"
                ] == "delivery_limit"
    finally:
        provider.shutdown()
