import asyncio
import random
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edu_graphrag.components.mlflow_model import MlflowModelComponent, ModelVersionRef
from edu_graphrag.config import MlflowSettings
from edu_graphrag.pipeline.store import TaskRecord


@dataclass
class FakeComponent:
    """In-memory `Component` whose behaviour (version, latency, failure) is set by the test."""

    name: str
    version: str = "1.0.0"
    delay: float = 0.0
    error: Exception | None = None
    startup_error: Exception | None = None
    events: list[str] = field(default_factory=list)
    checks: int = 0

    async def startup(self) -> None:
        if self.startup_error is not None:
            raise self.startup_error
        self.events.append(f"start:{self.name}")

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")

    async def fetch_version(self) -> str:
        self.checks += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.version


@dataclass
class FakePyfuncModel:
    """Pyfunc stand-in: returns `output(frame)` and records the thread each call ran in."""

    output: Callable[[pd.DataFrame], Any] = field(
        default=lambda frame: pd.DataFrame(
            {"label": frame["text"].str.upper(), "confidence": np.float64(0.9)}
        )
    )
    error: Exception | None = None
    inputs: list[pd.DataFrame] = field(default_factory=list)
    thread_ids: list[int] = field(default_factory=list)

    def predict(self, data: pd.DataFrame) -> Any:
        self.thread_ids.append(threading.get_ident())
        self.inputs.append(data)
        if self.error is not None:
            raise self.error
        return self.output(data)


@dataclass
class FakeModelRegistry:
    """In-memory `ModelRegistry`: aliases map to versions, each version to a model."""

    aliases: dict[str, str] = field(default_factory=lambda: {"champion": "3"})
    models: dict[str, FakePyfuncModel] = field(default_factory=dict)
    resolve_error: Exception | None = None
    load_error: Exception | None = None
    resolved: int = 0
    loaded_versions: list[str] = field(default_factory=list)
    closed: int = 0

    def model(self, version: str) -> FakePyfuncModel:
        return self.models.setdefault(version, FakePyfuncModel())

    def resolve(self, name: str, alias: str) -> ModelVersionRef:
        self.resolved += 1
        if self.resolve_error is not None:
            raise self.resolve_error
        if alias not in self.aliases:
            raise LookupError(f"alias {alias!r} not found for {name!r}")
        version = self.aliases[alias]
        return ModelVersionRef(
            name=name,
            alias=alias,
            version=version,
            run_id=f"run-{version}",
            model_uri=f"models:/{name}/{version}",
        )

    def load(self, ref: ModelVersionRef) -> FakePyfuncModel:
        if self.load_error is not None:
            raise self.load_error
        self.loaded_versions.append(ref.version)
        return self.model(ref.version)

    def close(self) -> None:
        self.closed += 1


def make_model_component(
    registry: FakeModelRegistry | None = None, *, name: str = "subject-classifier"
) -> tuple[MlflowModelComponent, FakeModelRegistry]:
    registry = registry or FakeModelRegistry()
    settings = MlflowSettings(mlflow_tracking_uri="http://mlflow.invalid", model_name=name)
    return MlflowModelComponent(settings, registry=registry), registry


# --- task pipeline -----------------------------------------------------------------------


@dataclass
class FakeTaskPublisher:
    """`TaskPublisher` whose answers are scripted: `outcomes` is consumed one per publish
    (None = ack, an exception = raised); when exhausted, every publish is acked."""

    outcomes: list[Exception | None] = field(default_factory=list)
    delay: float = 0.0
    consumers: int | None = 1
    consumer_delay: float = 0.0
    published: list[Any] = field(default_factory=list)
    attempts: list[Any] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)
    consumer_queries: int = 0
    in_flight: int = 0
    max_in_flight: int = 0

    async def publish(self, message: Any, *, confirm_timeout: float) -> None:
        self.attempts.append(message.task_id)
        self.timeouts.append(confirm_timeout)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            outcome = self.outcomes.pop(0) if self.outcomes else None
            if outcome is not None:
                raise outcome
            self.published.append(message)
        finally:
            self.in_flight -= 1

    async def consumer_count(self) -> int | None:
        self.consumer_queries += 1
        if self.consumer_delay:
            await asyncio.sleep(self.consumer_delay)
        return self.consumers


@dataclass
class InMemoryTaskStore:
    """`TaskStore` with real transaction semantics: inserts become visible on commit only."""

    rows: dict[Any, TaskRecord] = field(default_factory=dict)
    commits: int = 0
    rollbacks: int = 0
    insert_error: Exception | None = None
    commit_error: Exception | None = None
    get_error: Exception | None = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["InMemoryTransaction"]:
        tx = InMemoryTransaction(self)
        try:
            yield tx
        except BaseException:
            self.rollbacks += 1
            raise
        if self.commit_error is not None:
            self.rollbacks += 1
            raise self.commit_error
        self.rows.update(tx.staged)
        self.commits += 1

    async def get(self, task_id: Any) -> TaskRecord | None:
        if self.get_error is not None:
            raise self.get_error
        return self.rows.get(task_id)


@dataclass
class InMemoryTransaction:
    store: InMemoryTaskStore
    staged: dict[Any, TaskRecord] = field(default_factory=dict)

    async def insert_pending(self, message: Any) -> None:
        if self.store.insert_error is not None:
            raise self.store.insert_error
        self.staged[message.task_id] = TaskRecord(
            task_id=message.task_id,
            source=message.source,
            status="pending",
            result=None,
            error=None,
            model_name=None,
            model_version=None,
            attempts=0,
            submitted_at=message.submitted_at,
            finished_at=None,
        )


# --- aio-pika stand-ins for RabbitComponent ------------------------------------------------


@dataclass
class FakeDeclareOk:
    message_count: int = 0
    consumer_count: int = 0


@dataclass
class FakeRawChannel:
    declare_ok: FakeDeclareOk = field(default_factory=FakeDeclareOk)
    error: Exception | None = None
    declared: list[tuple[str, bool]] = field(default_factory=list)

    async def queue_declare(self, name: str, *, passive: bool) -> FakeDeclareOk:
        self.declared.append((name, passive))
        if self.error is not None:
            raise self.error
        return self.declare_ok


@dataclass
class FakeExchange:
    name: str
    error: BaseException | None = None
    delay: float = 0.0
    published: list[tuple[Any, str, dict[str, Any]]] = field(default_factory=list)

    async def publish(self, message: Any, routing_key: str, **kwargs: Any) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        self.published.append((message, routing_key, kwargs))


@dataclass
class FakeChannel:
    connection: "FakeConnection"
    kwargs: dict[str, Any]
    closed: bool = False

    async def get_exchange(self, name: str, *, ensure: bool = True) -> FakeExchange:
        assert not ensure, "the component must not declare topology"
        self.connection.exchange = FakeExchange(name)
        return self.connection.exchange

    async def get_underlay_channel(self) -> FakeRawChannel:
        return self.connection.raw

    async def close(self) -> None:
        self.closed = True


@dataclass
class _Underlay:
    server_properties: dict[str, Any]


@dataclass
class _Transport:
    connection: _Underlay


@dataclass
class FakeConnection:
    version: Any = "4.2.9"
    channel_error: Exception | None = None
    raw: FakeRawChannel = field(default_factory=FakeRawChannel)
    exchange: FakeExchange | None = None
    channels: list[FakeChannel] = field(default_factory=list)
    closed: bool = False

    @property
    def transport(self) -> _Transport:
        return _Transport(_Underlay({"version": self.version}))

    async def channel(self, **kwargs: Any) -> FakeChannel:
        if self.channel_error is not None:
            raise self.channel_error
        channel = FakeChannel(self, kwargs)
        self.channels.append(channel)
        return channel

    async def close(self) -> None:
        self.closed = True


@dataclass
class FakeConnect:
    """Replaces `aio_pika.connect_robust`: fails `failures` times, then connects."""

    connection: FakeConnection = field(default_factory=FakeConnection)
    failures: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, url: str, **kwargs: Any) -> FakeConnection:
        self.calls.append({"url": url, **kwargs})
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("connection refused")
        return self.connection


def seeded(seed: int) -> random.Random:
    """Deterministic jitter for tests."""
    return random.Random(seed)  # noqa: S311 - not cryptographic: backoff jitter
