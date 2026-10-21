"""Batch source: streams tasks from a JSONL file into the `tasks` queue with backpressure.

    python -m edu_graphrag.pipeline.source --file data/tasks/batch.jsonl \
        [--limit N] [--max-in-flight 16] [--no-backpressure] [--metrics-out metrics.csv]

Backpressure loop "worker -> broker -> source"
----------------------------------------------
1. Workers ack a task only after its result is stored; only then the quorum queue frees a slot.
2. The queue is bounded (policy `max-length`, `overflow: reject-publish`): when it is full the
   broker nacks the publish (see `pipeline.publisher` for why this signal and not others).
3. The source keeps at most `max_in_flight` unconfirmed publishes (an `asyncio.Semaphore`).
   A nacked message keeps its slot and is retried after `Backoff` (exponential, full jitter),
   so when the queue is full the window is full of waiting messages and the source *stops
   reading the file*. Nothing is buffered beyond the window: the input is a lazy generator,
   memory is O(max_in_flight), not O(file).

Automatic adaptation: there is no rate parameter. In steady state the queue is full and each
worker ack frees exactly one slot that one retried publish takes, so the accepted publish
rate converges to the aggregate worker throughput. Start a second worker and the queue drains
faster, more publishes are acked, the backoff resets more often: the rate goes up without any
change here. Stop the workers and it drops to zero, with the retries spaced by the cap.

Every second the source logs `source_stats` (and appends it to `--metrics-out` as CSV):
published_total, accepted_per_s, rejected_per_s, current_backoff_ms, in_flight, rss_mb.

Idempotency: task ids are uuid5 of the record id (or file name + line number), so running the
source again over the same file publishes the same ids and the store (primary key on task_id,
"done" is never overwritten) does not create duplicates.

`--no-backpressure` is the ANTI-PATTERN baseline for the experiment: a channel without
publisher confirms, publishes are fire-and-forget, refusals are never seen. With an unbounded
queue (policy max-length 0) the queue, and the broker's memory, grow without limit; with a
bounded one the broker drops the excess silently and tasks are lost.
"""

import argparse
import asyncio
import csv
import importlib
import json
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import structlog
from pydantic import ValidationError

from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.publisher import (
    Backoff,
    PublishCounters,
    PublishError,
    Sleep,
    TaskPublisher,
    publish_with_retry,
)

logger = structlog.get_logger(__name__)

# Fixed namespace: the same record id gives the same task_id on every machine and every run.
TASK_ID_NAMESPACE = uuid.UUID("6f1c3d2e-8a4b-5c7d-9e0f-1a2b3c4d5e6f")

METRIC_FIELDS = (
    "elapsed_s",
    "published_total",
    "accepted_per_s",
    "rejected_total",
    "rejected_per_s",
    "timeouts_total",
    "current_backoff_ms",
    "in_flight",
    "rss_mb",
)


def task_id_for(record_id: str) -> uuid.UUID:
    return uuid.uuid5(TASK_ID_NAMESPACE, record_id)


@dataclass
class ReadStats:
    read: int = 0
    skipped: int = 0


def iter_tasks(
    path: Path, *, limit: int | None = None, stats: ReadStats | None = None
) -> Iterator[TaskMessage]:
    """Lazily yields one `TaskMessage` per valid JSONL line: {"id": "...", "text": "..."}.

    Only the current line is in memory. Invalid lines (bad JSON, no text, too long) are logged
    and skipped, they must not stop a long batch. Without an "id" the record is identified by
    file name + line number (stable as long as the file is not edited).
    """
    stats = stats if stats is not None else ReadStats()
    with path.open(encoding="utf-8") as lines:
        for lineno, line in enumerate(lines, start=1):
            if limit is not None and stats.read >= limit:
                return
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = str(record.get("id") or f"{path.name}:{lineno}")
                message = TaskMessage(
                    task_id=task_id_for(record_id),
                    text=record["text"],
                    source="batch",
                    submitted_at=datetime.now(UTC),
                )
            except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as exc:
                stats.skipped += 1
                logger.warning("task_line_skipped", line=lineno, error=type(exc).__name__)
                continue
            stats.read += 1
            yield message


def rss_mb() -> float | None:
    """Resident memory of this process: psutil if installed (it is a dev dependency only),
    else /proc/self/status on Linux, else None."""
    try:
        psutil: Any = importlib.import_module("psutil")
    except ImportError:
        pass
    else:
        rss: int = psutil.Process().memory_info().rss
        return round(rss / 2**20, 1)
    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None  # pragma: no cover - VmRSS is always present on Linux


class MetricsCsv:
    """Appends one row per stats sample; flushed every row so a killed run keeps its data."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()

    def write(self, sample: dict[str, Any]) -> None:
        self._writer.writerow({key: sample.get(key) for key in METRIC_FIELDS})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class Source:
    publisher: TaskPublisher
    max_in_flight: int = 16
    confirm_timeout: float = 2.0
    backpressure: bool = True
    backoff: Backoff = field(default_factory=Backoff)
    sleep: Sleep = asyncio.sleep
    counters: PublishCounters = field(default_factory=PublishCounters)
    in_flight: int = 0
    max_in_flight_seen: int = 0
    lost: int = 0

    def __post_init__(self) -> None:
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")

    async def run(self, messages: Iterable[TaskMessage]) -> PublishCounters:
        if self.backpressure:
            await self._run_with_window(messages)
        else:
            await self._run_fire_and_forget(messages)
        return self.counters

    async def _run_with_window(self, messages: Iterable[TaskMessage]) -> None:
        window = asyncio.Semaphore(self.max_in_flight)
        async with asyncio.TaskGroup() as tasks:
            for message in messages:
                # The next line is read only once a slot is free: this is where the source
                # slows down when the broker refuses work.
                await window.acquire()
                self.in_flight += 1
                self.max_in_flight_seen = max(self.max_in_flight_seen, self.in_flight)
                tasks.create_task(self._publish_one(message, window))

    async def _publish_one(self, message: TaskMessage, window: asyncio.Semaphore) -> None:
        try:
            await publish_with_retry(
                self.publisher,
                message,
                backoff=self.backoff,
                confirm_timeout=self.confirm_timeout,
                counters=self.counters,
                sleep=self.sleep,
            )
        finally:
            self.in_flight -= 1
            window.release()

    async def _run_fire_and_forget(self, messages: Iterable[TaskMessage]) -> None:
        """ANTI-PATTERN baseline: no window, no retries, refusals counted as lost."""
        for message in messages:
            try:
                await self.publisher.publish(message, confirm_timeout=self.confirm_timeout)
            except PublishError:
                self.lost += 1
                self.counters.rejected += 1
            else:
                self.counters.accepted += 1

    def sample(
        self, *, elapsed: float, previous: PublishCounters, interval: float
    ) -> dict[str, Any]:
        return {
            "elapsed_s": round(elapsed, 2),
            "published_total": self.counters.accepted,
            "accepted_per_s": round((self.counters.accepted - previous.accepted) / interval, 1),
            "rejected_total": self.counters.rejected,
            "rejected_per_s": round((self.counters.rejected - previous.rejected) / interval, 1),
            "timeouts_total": self.counters.timeouts,
            "current_backoff_ms": round(self.backoff.last_delay * 1000, 1),
            "in_flight": self.in_flight,
            "rss_mb": rss_mb(),
        }


async def report_stats(
    source: Source,
    *,
    interval: float = 1.0,
    metrics: MetricsCsv | None = None,
    clock: Any = time.monotonic,
) -> None:
    """Logs `source_stats` every `interval` seconds until cancelled."""
    started = last = clock()
    previous = PublishCounters()
    while True:
        await asyncio.sleep(interval)
        now = clock()
        sample = source.sample(elapsed=now - started, previous=previous, interval=now - last)
        previous, last = PublishCounters(**vars(source.counters)), now
        logger.info("source_stats", **sample)
        if metrics is not None:
            metrics.write(sample)


async def run_file(
    source: Source,
    path: Path,
    *,
    limit: int | None = None,
    metrics: MetricsCsv | None = None,
    stats_interval: float = 1.0,
) -> dict[str, Any]:
    read_stats = ReadStats()
    started = time.monotonic()
    reporter = asyncio.create_task(report_stats(source, interval=stats_interval, metrics=metrics))
    try:
        await source.run(iter_tasks(path, limit=limit, stats=read_stats))
    finally:
        reporter.cancel()
    elapsed = time.monotonic() - started
    summary = {
        "read": read_stats.read,
        "skipped": read_stats.skipped,
        "published_total": source.counters.accepted,
        "rejected_total": source.counters.rejected,
        "timeouts_total": source.counters.timeouts,
        "unavailable_total": source.counters.unavailable,
        "lost": source.lost,
        "max_in_flight_seen": source.max_in_flight_seen,
        "elapsed_s": round(elapsed, 2),
        "avg_accepted_per_s": round(source.counters.accepted / elapsed, 1) if elapsed else None,
    }
    logger.info("source_finished", backpressure=source.backpressure, **summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--file", type=Path, required=True, help="JSONL with {id, text} lines")
    parser.add_argument("--limit", type=int, default=None, help="publish at most N tasks")
    parser.add_argument(
        "--max-in-flight", type=int, default=16, help="unconfirmed publishes at a time"
    )
    parser.add_argument(
        "--no-backpressure",
        action="store_true",
        help="ANTI-PATTERN baseline: fire-and-forget without confirms (for the experiment)",
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="CSV with 1 s samples")
    return parser.parse_args(argv)


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    from edu_graphrag.components.rabbit import RabbitComponent
    from edu_graphrag.config import PipelineSettings, RabbitSettings

    rabbit_settings = RabbitSettings()  # password comes from env / .env
    pipeline = PipelineSettings()
    rabbit = RabbitComponent(
        rabbit_settings, connect_timeout=5.0, publisher_confirms=not args.no_backpressure
    )
    source = Source(
        rabbit,
        max_in_flight=args.max_in_flight,
        confirm_timeout=pipeline.publish_confirm_timeout,
        backpressure=not args.no_backpressure,
    )
    metrics = MetricsCsv(args.metrics_out) if args.metrics_out else None
    # A broker that is down at start is not fatal: publishes fail with BrokerUnavailableError and
    # are retried with backoff while the component reconnects in the background.
    await rabbit.startup()
    try:
        return await run_file(source, args.file, limit=args.limit, metrics=metrics)
    finally:
        await rabbit.shutdown()
        if metrics is not None:
            metrics.close()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs a real broker
    from edu_graphrag.logging_config import configure_logging

    # Not `get_settings()`: the source must not require database passwords it never uses.
    configure_logging("INFO", json_logs=True)
    asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
