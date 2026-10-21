import asyncio
import csv
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    Backoff,
    PublishCounters,
    PublishRejectedError,
    PublishUnroutableError,
)
from edu_graphrag.pipeline.source import (
    MetricsCsv,
    ReadStats,
    Source,
    amain,
    iter_tasks,
    parse_args,
    rss_mb,
    run_file,
    task_id_for,
)
from tests.fakes import FakeTaskPublisher, seeded


async def no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def write_jsonl(path: Path, lines: list[Any]) -> Path:
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


def messages(n: int) -> list[TaskMessage]:
    return [
        TaskMessage.new(f"text {i}", source="batch", task_id=task_id_for(str(i))) for i in range(n)
    ]


# --- reading ------------------------------------------------------------------------------


def test_task_ids_are_deterministic(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "b.jsonl", [{"id": "a#0", "text": "x"}, {"text": "no id"}])

    first = [m.task_id for m in iter_tasks(path)]
    second = [m.task_id for m in iter_tasks(path)]

    assert first == second, "a restarted source must publish the same ids"
    assert first[0] == task_id_for("a#0")
    assert first[1] == task_id_for("b.jsonl:2")
    assert len(set(first)) == 2


def test_invalid_lines_are_skipped(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "b.jsonl",
        [
            {"id": "1", "text": "ok"},
            "not json",
            {"id": "2"},
            {"id": "3", "text": ""},
            {"id": "4", "text": "x" * 20_001},
            "[1, 2]",
            "",
            {"id": "5", "text": "ok too"},
        ],
    )
    stats = ReadStats()

    got = list(iter_tasks(path, stats=stats))

    assert [m.text for m in got] == ["ok", "ok too"]
    assert all(m.source == "batch" for m in got)
    assert (stats.read, stats.skipped) == (2, 5)


def test_limit(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "b.jsonl", [{"id": str(i), "text": "t"} for i in range(10)])

    assert len(list(iter_tasks(path, limit=3))) == 3


def test_reading_is_lazy(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "b.jsonl", [{"id": str(i), "text": "t"} for i in range(1000)])
    stats = ReadStats()

    tasks = iter_tasks(path, stats=stats)
    next(tasks)

    assert stats.read == 1  # the rest of the file has not been parsed


# --- the window ---------------------------------------------------------------------------


class GatedPublisher(FakeTaskPublisher):
    """Holds every publish until the gate opens: nothing is ever confirmed before that."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def publish(self, message: Any, *, confirm_timeout: float) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await self.gate.wait()
            self.published.append(message)
        finally:
            self.in_flight -= 1


async def test_window_bounds_unconfirmed_publishes_and_reading() -> None:
    pulled = 0

    def lazy() -> Iterator[TaskMessage]:
        nonlocal pulled
        for message in messages(100):
            pulled += 1
            yield message

    publisher = GatedPublisher()
    source = Source(publisher, max_in_flight=4, sleep=no_sleep)
    run = asyncio.create_task(source.run(lazy()))
    for _ in range(20):
        await asyncio.sleep(0)

    assert publisher.in_flight == 4
    assert source.in_flight == 4
    assert pulled == 5, "the source stops reading while the window is full"

    publisher.gate.set()
    counters = await run

    assert counters.accepted == 100
    assert publisher.max_in_flight == 4
    assert source.max_in_flight_seen == 4
    assert source.in_flight == 0


async def test_nacked_messages_keep_their_slot_and_are_all_delivered() -> None:
    outcomes: list[Exception | None] = [PublishRejectedError("nack")] * 30
    publisher = FakeTaskPublisher(outcomes=list(outcomes))
    source = Source(publisher, max_in_flight=3, sleep=no_sleep, backoff=Backoff(rng=seeded(1)))

    counters = await source.run(messages(20))

    assert counters.accepted == 20
    assert counters.rejected == 30
    assert sorted(m.task_id for m in publisher.published) == sorted(
        m.task_id for m in messages(20)
    ), "every task is delivered exactly once from the source's point of view"
    assert publisher.max_in_flight <= 3


async def test_unroutable_aborts_the_run() -> None:
    publisher = FakeTaskPublisher(outcomes=[PublishUnroutableError("no binding")])

    with pytest.raises(ExceptionGroup) as info:
        await Source(publisher, sleep=no_sleep).run(messages(5))

    assert info.group_contains(PublishUnroutableError)


async def test_no_backpressure_mode_ignores_refusals() -> None:
    publisher = FakeTaskPublisher(outcomes=[None, PublishRejectedError("dropped"), None])
    source = Source(publisher, backpressure=False)

    counters = await source.run(messages(3))

    assert len(publisher.attempts) == 3, "no retries"
    assert counters.accepted == 2
    assert source.lost == 1


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        Source(FakeTaskPublisher(), max_in_flight=0)


# --- stats --------------------------------------------------------------------------------


def test_sample_reports_rates_since_previous() -> None:
    source = Source(FakeTaskPublisher())
    source.counters = PublishCounters(accepted=30, rejected=12, timeouts=1)
    source.backoff.last_delay = 0.25
    source.in_flight = 7

    sample = source.sample(
        elapsed=3.0, previous=PublishCounters(accepted=10, rejected=2), interval=2.0
    )

    assert sample["published_total"] == 30
    assert sample["accepted_per_s"] == 10.0
    assert sample["rejected_per_s"] == 5.0
    assert sample["current_backoff_ms"] == 250.0
    assert sample["in_flight"] == 7
    assert sample["elapsed_s"] == 3.0


def test_rss_mb_is_reported() -> None:
    value = rss_mb()  # psutil is installed in the dev environment
    assert value is not None
    assert value > 0


def test_rss_mb_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_psutil(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", no_psutil)
    value = rss_mb()
    if Path("/proc/self/status").exists():  # Linux CI: falls back to /proc
        assert value is not None
        assert value > 0
    else:
        assert value is None


async def test_run_file_logs_stats_and_writes_metrics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_jsonl(tmp_path / "b.jsonl", [{"id": str(i), "text": "t"} for i in range(30)])
    metrics_path = tmp_path / "out" / "metrics.csv"
    metrics = MetricsCsv(metrics_path)
    source = Source(FakeTaskPublisher(delay=0.01), max_in_flight=2)

    summary = await run_file(source, path, limit=20, metrics=metrics, stats_interval=0.02)
    metrics.close()

    assert summary["read"] == 20
    assert summary["published_total"] == 20
    assert summary["max_in_flight_seen"] == 2
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert rows, "at least one sample was written"
    assert set(rows[0]) >= {"published_total", "accepted_per_s", "rejected_per_s", "rss_mb"}
    events = [r.msg["event"] for r in caplog.records if isinstance(r.msg, dict)]
    assert "source_stats" in events
    assert events[-1] == "source_finished"


def test_parse_args() -> None:
    args = parse_args(["--file", "b.jsonl", "--limit", "5", "--no-backpressure"])

    assert args.file == Path("b.jsonl")
    assert args.limit == 5
    assert args.max_in_flight == 16
    assert args.no_backpressure is True
    assert args.metrics_out is None


async def test_amain_wires_the_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    class StubRabbit(FakeTaskPublisher):
        def __init__(self, settings: Any, **kwargs: Any) -> None:
            super().__init__()
            created.append(kwargs)
            self.events: list[str] = []

        async def startup(self) -> None:
            self.events.append("start")

        async def shutdown(self) -> None:
            self.events.append("stop")

    monkeypatch.setenv("RABBITMQ_PASSWORD", "x")
    monkeypatch.setattr("edu_graphrag.components.rabbit.RabbitComponent", StubRabbit)
    path = write_jsonl(tmp_path / "b.jsonl", [{"id": str(i), "text": "t"} for i in range(3)])
    metrics = tmp_path / "m.csv"

    summary = await amain(parse_args(["--file", str(path), "--metrics-out", str(metrics)]))

    assert summary["published_total"] == 3
    assert created == [{"connect_timeout": 5.0, "publisher_confirms": True}]
    assert metrics.exists()
