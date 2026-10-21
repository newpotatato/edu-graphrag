"""Worker message handling with FastStream's in-memory TestRabbitBroker (no real RabbitMQ)."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pandas as pd
import pytest
from faststream import TestApp
from faststream._internal.middlewares import BaseMiddleware
from faststream.message import AckStatus, StreamMessage
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient

from edu_graphrag.config import PipelineSettings, Settings
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.topology import DEAD_LETTER_EXCHANGE, TASKS_EXCHANGE, TASKS_QUEUE
from edu_graphrag.pipeline.worker import (
    Worker,
    WorkerDeps,
    WorkerSettings,
    build_worker,
    death_reason,
    delivery_attempt,
)
from tests.fakes import FakeModelRegistry, FakePyfuncModel


@dataclass
class FakeStore:
    """In-memory TaskStore with the same idempotency rule as the SQL upsert."""

    done: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    failed: dict[UUID, str] = field(default_factory=dict)
    save_error: Exception | None = None
    fail_error: Exception | None = None
    saves: int = 0
    closed: bool = False

    async def save_result(
        self,
        message: TaskMessage,
        *,
        result: dict[str, Any],
        model_name: str,
        model_version: str,
        attempts: int,
    ) -> bool:
        self.saves += 1
        if self.save_error is not None:
            raise self.save_error
        if message.task_id in self.done:
            return False
        self.failed.pop(message.task_id, None)
        self.done[message.task_id] = {
            "result": result,
            "model_name": model_name,
            "model_version": model_version,
            "attempts": attempts,
        }
        return True

    async def mark_failed(self, message: TaskMessage, error: str) -> None:
        if self.fail_error is not None:
            raise self.fail_error
        if message.task_id not in self.done:
            self.failed[message.task_id] = error

    async def close(self) -> None:
        self.closed = True


class AckSpy:
    """Broker middleware that keeps every consumed message to inspect its ack status."""

    def __init__(self) -> None:
        self.messages: list[StreamMessage[Any]] = []

    def __call__(self, msg: Any, context: Any) -> BaseMiddleware:
        spy = self

        class _Spy(BaseMiddleware):
            async def consume_scope(self, call_next: Any, msg: StreamMessage[Any]) -> Any:
                spy.messages.append(msg)
                return await call_next(msg)

        return _Spy(msg, context=context)

    @property
    def statuses(self) -> list[AckStatus | None]:
        return [m.committed for m in self.messages]


@dataclass
class Harness:
    worker: Worker
    registry: FakeModelRegistry
    store: FakeStore
    acks: AckSpy
    broker: Any

    async def send(self, body: Any, headers: dict[str, Any] | None = None) -> None:
        await self.broker.publish(
            body, queue=TASKS_QUEUE, exchange=TASKS_EXCHANGE, headers=headers or {}
        )


def make_settings(settings: Settings, **pipeline: Any) -> Settings:
    defaults: dict[str, Any] = {
        "worker_concurrency": 2,
        "inference_delay_ms": 0,
        "delivery_limit": 3,
    }
    return settings.model_copy(update={"pipeline": PipelineSettings(**(defaults | pipeline))})


def make_worker(
    settings: Settings,
    *,
    dlq_recorder: bool = False,
    model: FakePyfuncModel | None = None,
    **pipeline: Any,
) -> tuple[Worker, FakeModelRegistry, FakeStore]:
    registry = FakeModelRegistry()
    if model is not None:
        registry.models["3"] = model
    store = FakeStore()
    worker = build_worker(
        make_settings(settings, **pipeline),
        WorkerDeps(registry=registry, store=store),
        WorkerSettings(worker_graceful_timeout=5, dlq_recorder_enabled=dlq_recorder),
    )
    return worker, registry, store


@pytest.fixture
async def harness(settings: Settings) -> AsyncIterator[Harness]:
    worker, registry, store = make_worker(settings)
    acks = AckSpy()
    worker.broker.add_middleware(acks)
    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        yield Harness(worker, registry, store, acks, broker)


def task(text: str = "linear algebra") -> TaskMessage:
    return TaskMessage.new(text, "api")


async def test_success_writes_result_then_acks(harness: Harness) -> None:
    msg = task()
    await harness.send(msg.model_dump(mode="json"))

    saved = harness.store.done[msg.task_id]
    assert saved["result"]["label"] == "LINEAR ALGEBRA"
    assert saved["model_name"] == "subject-classifier"
    assert saved["model_version"] == "3"
    assert saved["attempts"] == 1
    assert harness.acks.statuses == [AckStatus.ACKED]


async def test_attempt_number_comes_from_delivery_count(harness: Harness) -> None:
    msg = task()
    await harness.send(msg.model_dump(mode="json"), headers={"x-delivery-count": 2})
    assert harness.store.done[msg.task_id]["attempts"] == 3


async def test_redelivery_of_done_task_is_acked_without_duplicate(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    msg = task()
    await harness.send(msg.model_dump(mode="json"))
    first = dict(harness.store.done[msg.task_id])
    await harness.send(msg.model_dump(mode="json"), headers={"x-delivery-count": 1})

    assert harness.store.done == {msg.task_id: first}
    assert harness.acks.statuses == [AckStatus.ACKED, AckStatus.ACKED]


async def test_transient_store_error_is_nacked_for_requeue(harness: Harness) -> None:
    harness.store.save_error = ConnectionError("postgres is down")
    msg = task()
    await harness.send(msg.model_dump(mode="json"))

    assert harness.acks.statuses == [AckStatus.NACKED]
    assert msg.task_id not in harness.store.done
    # Not the last attempt yet: the task is not marked failed, it will be retried.
    assert harness.store.failed == {}


async def test_model_error_is_nacked(harness: Harness) -> None:
    harness.registry.models["3"].error = RuntimeError("boom")
    await harness.send(task().model_dump(mode="json"))
    assert harness.acks.statuses == [AckStatus.NACKED]


async def test_last_attempt_marks_failed_before_dead_lettering(harness: Harness) -> None:
    harness.registry.models["3"].error = RuntimeError("boom")
    msg = task()
    # delivery_limit=3 -> deliveries with x-delivery-count None, 1, 2, 3; 3 is the last.
    await harness.send(msg.model_dump(mode="json"), headers={"x-delivery-count": 3})

    assert harness.acks.statuses == [AckStatus.NACKED]
    assert "delivery limit reached" in harness.store.failed[msg.task_id]


async def test_mark_failed_error_does_not_break_nack(harness: Harness) -> None:
    harness.store.save_error = ConnectionError("down")
    harness.store.fail_error = ConnectionError("still down")
    await harness.send(task().model_dump(mode="json"), headers={"x-delivery-count": 3})
    assert harness.acks.statuses == [AckStatus.NACKED]


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        {
            "task_id": "not-a-uuid",
            "text": "x",
            "source": "api",
            "submitted_at": "2026-01-01T00:00:00Z",
        },
        {
            "task_id": "7d3f0a3e-0000-4000-8000-000000000000",
            "text": "",
            "source": "api",
            "submitted_at": "2026-01-01T00:00:00Z",
        },
        {"unexpected": "shape"},
    ],
)
async def test_poison_message_is_rejected_without_retry(harness: Harness, body: Any) -> None:
    await harness.send(body)
    assert harness.acks.statuses == [AckStatus.REJECTED]
    assert harness.store.saves == 0
    assert harness.registry.models["3"].inputs == []


async def test_model_is_loaded_once_for_many_messages(harness: Harness) -> None:
    for i in range(5):
        await harness.send(task(f"text {i}").model_dump(mode="json"))
    assert harness.registry.loaded_versions == ["3"]
    assert harness.registry.resolved == 1
    assert len(harness.store.done) == 5


async def test_inference_runs_off_the_event_loop_thread(harness: Harness) -> None:
    await harness.send(task().model_dump(mode="json"))
    (thread_id,) = harness.registry.models["3"].thread_ids
    assert thread_id != threading.get_ident()


def slow_model(seconds: float, started: threading.Event | None = None) -> FakePyfuncModel:
    def output(frame: pd.DataFrame) -> pd.DataFrame:
        if started is not None:
            started.set()
        time.sleep(seconds)  # blocking on purpose: it must not stall the event loop
        return pd.DataFrame({"label": ["x"] * len(frame), "confidence": [0.5] * len(frame)})

    return FakePyfuncModel(output=output)


async def test_event_loop_stays_responsive_during_slow_inference(settings: Settings) -> None:
    worker, _, store = make_worker(settings, model=slow_model(0.5), worker_concurrency=2)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        beat = asyncio.create_task(heartbeat())
        started = time.perf_counter()
        await asyncio.gather(
            *(
                broker.publish(
                    task(str(i)).model_dump(mode="json"), queue=TASKS_QUEUE, exchange=TASKS_EXCHANGE
                )
                for i in range(2)
            )
        )
        elapsed = time.perf_counter() - started
        beat.cancel()

    assert len(store.done) == 2
    # Two 0.5 s inferences ran in parallel on two threads, not one after another.
    assert elapsed < 0.9
    # The heartbeat kept ticking (~25 ticks expected) while both threads were sleeping.
    assert ticks >= 10


async def test_health_answers_while_all_threads_are_busy(settings: Settings) -> None:
    started = threading.Event()
    worker, _, store = make_worker(settings, model=slow_model(0.6, started), worker_concurrency=1)
    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        publish = asyncio.create_task(
            broker.publish(
                task().model_dump(mode="json"), queue=TASKS_QUEUE, exchange=TASKS_EXCHANGE
            )
        )
        await asyncio.to_thread(started.wait, 2)
        transport = ASGITransport(app=worker.app)
        async with AsyncClient(transport=transport, base_url="http://worker") as client:
            t0 = time.perf_counter()
            response = await client.get("/health")
            latency = time.perf_counter() - t0
        await publish

    assert response.status_code == 200
    body = response.json()
    assert body["in_flight"] == 1
    assert body["inference_threads_busy"] == 1
    assert body["concurrency"] == body["prefetch_count"] == 1
    assert body["model"]["version"] == "3"
    assert latency < 0.2
    assert len(store.done) == 1


async def test_ready_reports_model_and_broker(harness: Harness) -> None:
    transport = ASGITransport(app=harness.worker.app)
    async with AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model"]["loaded"] is True


async def test_ready_is_503_before_startup(settings: Settings) -> None:
    worker, _, _ = make_worker(settings)
    transport = ASGITransport(app=worker.app)
    async with AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["model"] == {
        "uri": "models:/subject-classifier@champion",
        "loaded": False,
    }


async def test_graceful_shutdown_finishes_in_flight_tasks(settings: Settings) -> None:
    started = threading.Event()
    worker, registry, store = make_worker(
        settings, model=slow_model(0.4, started), worker_concurrency=2
    )
    acks = AckSpy()
    worker.broker.add_middleware(acks)
    async with TestRabbitBroker(worker.broker) as broker:
        async with TestApp(worker.app):
            publishes = [
                asyncio.create_task(
                    broker.publish(
                        task(str(i)).model_dump(mode="json"),
                        queue=TASKS_QUEUE,
                        exchange=TASKS_EXCHANGE,
                    )
                )
                for i in range(2)
            ]
            await asyncio.to_thread(started.wait, 2)
            # Leaving TestApp runs the shutdown sequence while both tasks are in flight.
        # After shutdown: both finished, were committed and acked; resources released.
        assert all(p.done() for p in publishes)
    assert len(store.done) == 2
    assert acks.statuses == [AckStatus.ACKED, AckStatus.ACKED]
    assert store.closed
    assert registry.closed == 1


async def test_dlq_recorder_marks_failed(settings: Settings) -> None:
    worker, _, store = make_worker(settings, dlq_recorder=True)
    msg = task()
    headers: dict[str, Any] = {"x-first-death-reason": "delivery_limit"}
    async with TestRabbitBroker(worker.broker) as broker, TestApp(worker.app):
        await broker.publish(
            msg.model_dump(mode="json"), exchange=DEAD_LETTER_EXCHANGE, headers=headers
        )
        await broker.publish(
            b"garbage", exchange=DEAD_LETTER_EXCHANGE, headers={"x-first-death-reason": "rejected"}
        )
    assert store.failed == {msg.task_id: "dead-lettered: delivery_limit"}


def test_delivery_attempt() -> None:
    assert delivery_attempt({}) == 1
    assert delivery_attempt({"x-delivery-count": 0}) == 1
    assert delivery_attempt({"x-delivery-count": 2}) == 3
    assert delivery_attempt({"x-delivery-count": "junk"}) == 1


def test_death_reason() -> None:
    assert death_reason({"x-first-death-reason": "rejected"}) == "rejected"
    assert death_reason({"x-death": [{"reason": "delivery_limit", "count": 1}]}) == "delivery_limit"
    assert death_reason({}) == "unknown"


def test_prefetch_equals_concurrency(settings: Settings) -> None:
    worker, _, _ = make_worker(settings, worker_concurrency=7)
    assert worker.prefetch_count == worker.engine.concurrency == 7


async def test_postgres_store_commits_through_pipeline_store(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    from contextlib import asynccontextmanager

    from edu_graphrag.pipeline import store, worker_store

    calls: list[str] = []

    class FakeEngine:
        @asynccontextmanager
        async def begin(self) -> AsyncIterator[str]:
            calls.append("begin")
            yield "conn"
            calls.append("commit")

        async def dispose(self) -> None:
            calls.append("dispose")

    async def fake_save(conn: Any, message: TaskMessage, **kwargs: Any) -> bool:
        calls.append(f"save:{conn}:{kwargs['attempts']}")
        return True

    async def fake_fail(conn: Any, message: TaskMessage, error: str) -> None:
        calls.append(f"fail:{error}")

    monkeypatch.setattr(store, "save_result", fake_save)
    monkeypatch.setattr(store, "mark_failed", fake_fail)
    pg = worker_store.PostgresTaskStore(settings.postgres, pool_size=2)
    pg._engine = FakeEngine()  # type: ignore[assignment]
    msg = task()
    assert await pg.save_result(msg, result={}, model_name="m", model_version="1", attempts=2)
    await pg.mark_failed(msg, "boom")
    await pg.close()
    assert calls == ["begin", "save:conn:2", "commit", "begin", "fail:boom", "commit", "dispose"]


def test_create_app_wires_real_dependencies(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    from edu_graphrag.pipeline import worker as worker_module

    logging_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(worker_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        worker_module,
        "configure_logging",
        lambda level, *, json_logs: logging_calls.append((level, json_logs)),
    )
    app = worker_module.create_app()
    assert app.broker is not None
    assert logging_calls == [(settings.log_level, settings.log_json)]
