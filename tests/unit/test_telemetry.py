import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pandas as pd
import pytest
import structlog
from faststream import TestApp
from faststream.message import AckStatus
from faststream.rabbit import TestRabbitBroker
from opentelemetry.metrics import NoOpMeter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, StatusCode

from edu_graphrag import telemetry
from edu_graphrag.api.task_service import TaskService
from edu_graphrag.config import Settings
from edu_graphrag.logging_config import configure_logging
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import PublishRejectedError
from edu_graphrag.pipeline.source import Source
from edu_graphrag.pipeline.topology import TASKS_EXCHANGE, TASKS_QUEUE
from edu_graphrag.pipeline.worker_store import PostgresTaskStore
from tests.conftest import ClientFactory
from tests.fakes import FakePyfuncModel, FakeTaskPublisher, InMemoryTaskStore
from tests.unit.pipeline.test_worker import AckSpy, make_worker
from tests.unit.test_components import make_rabbit


def span_context(span: ReadableSpan) -> SpanContext:
    context = span.get_span_context()
    assert context is not None
    return context


@pytest.fixture
def sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[InMemorySpanExporter, InMemoryMetricReader]]:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    tracer = TracerProvider(shutdown_on_exit=False)
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    with telemetry.use_runtime(
        telemetry.Runtime(tracer.get_tracer("test"), meters.get_meter("test"))
    ):
        yield exporter, reader
    tracer.shutdown()
    meters.shutdown()


async def test_publish_carrier_extends_http_trace_after_request_exits(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader],
    make_client: ClientFactory,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exporter, reader = sdk
    rabbit, connect = make_rabbit()
    service = TaskService(InMemoryTaskStore(), rabbit, pipeline=settings.pipeline)
    async with make_client([rabbit], service) as api:
        response = await api.post("/api/v1/tasks", json={"text": "PRIVATE-FIXTURE-CHUNK"})
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert connect.connection.exchange is not None
    ((message, _, _),) = connect.connection.exchange.published
    assert message.correlation_id == task_id
    assert message.headers["traceparent"]
    assert not telemetry.get_current_span().get_span_context().is_valid

    worker, _, store = make_worker(settings)
    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        await broker.publish(
            message.body,
            queue=TASKS_QUEUE,
            exchange=TASKS_EXCHANGE,
            headers=dict(message.headers),
            correlation_id=task_id,
        )
    assert UUID(task_id) in store.done
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans.keys() >= {
        "POST /api/v1/tasks",
        "publish",
        "process",
        "inference",
        "db.save_result",
    }
    contexts = {name: span_context(span) for name, span in spans.items()}
    assert {c.trace_id for c in contexts.values()} == {contexts["publish"].trace_id}
    assert spans["publish"].parent == contexts["POST /api/v1/tasks"]
    parent = spans["process"].parent
    assert parent is not None
    assert parent.is_remote
    assert parent.span_id == contexts["publish"].span_id
    assert spans["inference"].parent == spans["db.save_result"].parent == contexts["process"]
    assert spans["inference"].attributes is not None
    assert spans["inference"].attributes["model.version"] == "3"
    assert spans["inference"].attributes["model.alias"] == "champion"
    expected = f"{contexts['process'].trace_id:032x}"
    output = capsys.readouterr().out
    logs = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert {record["trace_id"] for record in logs if record.get("task_id") == task_id} == {expected}
    assert "PRIVATE-FIXTURE-CHUNK" not in output
    data = reader.get_metrics_data()
    assert data is not None
    text = data.to_json()
    assert "edu_worker_inference_duration" in text
    assert "edu_api_requests" in text
    assert task_id not in text
    assert expected not in text


async def test_overload_is_a_sanitized_error_trace(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader],
    make_client: ClientFactory,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exporter, _ = sdk
    publisher = FakeTaskPublisher(outcomes=[PublishRejectedError("PRIVATE-REJECTED-INPUT")])
    store = InMemoryTaskStore()
    async with make_client([], TaskService(store, publisher, pipeline=settings.pipeline)) as client:
        response = await client.post("/api/v1/tasks", json={"text": "private example"})
    assert response.status_code == 503
    assert store.rows == {}
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]
    assert "PRIVATE-REJECTED-INPUT" not in span.to_json()
    assert "PRIVATE-REJECTED-INPUT" not in capsys.readouterr().out


async def test_each_source_task_owns_a_trace_and_backoff_event(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader],
) -> None:
    exporter, _ = sdk
    source = Source(FakeTaskPublisher(outcomes=[PublishRejectedError("full")]), max_in_flight=1)
    tasks = [TaskMessage.new("synthetic", "batch") for _ in range(2)]
    await source.run(tasks)
    roots = [span for span in exporter.get_finished_spans() if span.name == "source.task"]
    assert len(roots) == 2
    assert all(span.parent is None for span in roots)
    assert len({span_context(span).trace_id for span in roots}) == 2
    assert sum(event.name == "publish.backoff" for span in roots for event in span.events) == 1


def test_stdlib_trace_ids_and_redacted_exception_do_not_leak(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    provider = TracerProvider(shutdown_on_exit=False)
    with telemetry.use_runtime(telemetry.Runtime(provider.get_tracer("test"), NoOpMeter("test"))):
        with (
            telemetry.span("example"),
            structlog.contextvars.bound_contextvars(task_id="test-task"),
        ):
            try:
                raise ValueError("do-not-log-exception-message")
            except ValueError:
                logging.getLogger("dependency").exception(
                    "DB unavailable at postgres://test-user:dsn-password@db:5432/db"
                )
        structlog.get_logger().info("unrelated")
    provider.shutdown()
    output = capsys.readouterr().out
    trace_log, normal = (json.loads(line) for line in output.splitlines())
    assert "dsn-password" not in output
    assert "do-not-log-exception-message" not in output
    assert trace_log["task_id"] == "test-task"
    assert len(trace_log["trace_id"]) == 32
    assert trace_log["exception"]["frames"]
    assert "trace_id" not in normal
    assert "task_id" not in normal


async def test_parallel_worker_threads_keep_separate_trace_contexts(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader], settings: Settings
) -> None:
    exporter, _ = sdk
    barrier = threading.Barrier(2)

    def predict(frame: pd.DataFrame) -> pd.DataFrame:
        barrier.wait(timeout=5)
        return pd.DataFrame({"label": ["fixture"] * len(frame), "confidence": [0.5] * len(frame)})

    worker, _, store = make_worker(settings, model=FakePyfuncModel(output=predict))
    tasks = [TaskMessage.new("parallel fixture", "batch") for _ in range(2)]
    carriers = []
    for task in tasks:
        with telemetry.span("test.publish", {"task.id": str(task.task_id)}) as producer:
            carriers.append((task, telemetry.propagate_headers(), producer.get_span_context()))

    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        await asyncio.gather(
            *(
                broker.publish(
                    task.model_dump_json().encode(),
                    queue=TASKS_QUEUE,
                    exchange=TASKS_EXCHANGE,
                    headers=dict(headers),
                    correlation_id=str(task.task_id),
                )
                for task, headers, _ in carriers
            )
        )

    spans = exporter.get_finished_spans()
    assert store.done.keys() == {task.task_id for task in tasks}
    assert len({span_context(span).trace_id for span in spans}) == 2
    for task, _, parent in carriers:
        children = {
            span.name: span for span in spans if span_context(span).trace_id == parent.trace_id
        }
        assert children.keys() == {"test.publish", "process", "inference", "db.save_result"}
        process = children["process"]
        assert process.parent is not None
        assert process.parent.span_id == parent.span_id
        assert (process.attributes or {})["task.id"] == str(task.task_id)
        assert (
            children["inference"].parent
            == children["db.save_result"].parent
            == span_context(process)
        )


async def test_result_commit_failure_is_traced_and_nacked(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter, reader = sdk
    worker, _, store = make_worker(settings)
    postgres = PostgresTaskStore(settings.postgres, pool_size=1)
    acks = AckSpy()
    worker.broker.add_middleware(acks)
    statements = 0

    class Result:
        def first(self) -> tuple[UUID]:
            return (uuid4(),)

    class Connection:
        async def execute(self, _statement: object) -> Result:
            nonlocal statements
            statements += 1
            return Result()

    @asynccontextmanager
    async def begin(_engine: object) -> AsyncIterator[Connection]:
        yield Connection()
        raise ConnectionError("PRIVATE-COMMIT-FIXTURE")

    monkeypatch.setattr(type(postgres._engine), "begin", begin)
    monkeypatch.setattr(store, "save_result", postgres.save_result)
    task = TaskMessage.new("fixture", "batch")
    try:
        async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
            await broker.publish(
                task.model_dump_json().encode(), queue=TASKS_QUEUE, exchange=TASKS_EXCHANGE
            )
    finally:
        await postgres.close()

    assert statements == 1
    assert acks.statuses == [AckStatus.NACKED]
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["db.save_result"].status.status_code == StatusCode.ERROR
    assert spans["process"].status.status_code == StatusCode.ERROR
    assert spans["db.save_result"].parent == span_context(spans["process"])
    assert "PRIVATE-COMMIT-FIXTURE" not in spans["db.save_result"].to_json()
    data = reader.get_metrics_data()
    assert data is not None
    assert "edu_worker_task_duration" not in data.to_json()


async def test_polling_many_ids_uses_one_route_label(
    sdk: tuple[InMemorySpanExporter, InMemoryMetricReader],
    make_client: ClientFactory,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exporter, reader = sdk
    ids = {str(uuid4()) for _ in range(4)}
    service = TaskService(InMemoryTaskStore(), FakeTaskPublisher(), pipeline=settings.pipeline)
    async with make_client([], service) as api:
        for task_id in ids:
            response = await api.get(f"/api/v1/tasks/{task_id}")
            assert response.status_code == 404
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"GET /api/v1/tasks/{task_id}"}
    assert {(span.attributes or {})["task.id"] for span in spans} == ids
    data = reader.get_metrics_data()
    assert data is not None
    text = data.to_json()
    assert "/api/v1/tasks/{task_id}" in text
    assert not any(task_id in text for task_id in ids)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {line.get("task_id") for line in lines if line["event"] == "request_finished"} == ids
