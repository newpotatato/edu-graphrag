# ruff: noqa: RUF001, S608
"""Backpressure experiment: the pipeline with and without load limits.

Usage: `uv run python scripts/experiment.py --scenario A|B|C|all` (stack must be up:
`just pipeline-up`), then `uv run python scripts/experiment.py --plots` to draw the figures.

Scenarios (see README, "Backpressure: эксперимент"):
  A  no limits: unbounded queue, source without confirms, 1 worker
  B  backpressure: queue limit 200, publisher confirms, 1 worker
  C  backpressure, 1 worker -> a second worker joins at t = SCALE_AT_S (no code changes)

Each run writes data/experiment/<scenario>/{timeline,latency,source,shed,summary}.* and the
plots go to docs/backpressure/. The source runs as a host process; everything else is observed
through the RabbitMQ management API, `docker stats` and Postgres.
"""

import argparse
import csv
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edu_graphrag.config import RabbitSettings
from edu_graphrag.pipeline.admin import ApiRequest, ManagementClient

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "experiment"
PLOTS = ROOT / "docs" / "backpressure"
BATCH = ROOT / "data" / "tasks" / "batch.jsonl"
API = "http://127.0.0.1:8000"
LIMIT = 200
SCALE_AT_S = 45.0
SHED_AT_S = 25.0
SHED_REQUESTS = 30
BUCKET_S = 5
TIMELINE_FIELDS = (
    "t",
    "messages",
    "ready",
    "unacked",
    "consumers",
    "broker_mb",
    "dlq",
    "done",
    "worker_mb",
    "workers",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    title: str
    max_length: int  # 0 = unbounded
    backpressure: bool
    limit: int  # tasks the source publishes
    observe_s: float  # sampling stops after this long at the latest
    scale_to: int | None = None


SCENARIOS = {
    "A": Scenario("A", "без ограничений, 1 воркер", 0, False, 15000, 45.0),
    "B": Scenario("B", "backpressure, 1 воркер", LIMIT, True, 1300, 90.0),
    "C": Scenario("C", "backpressure, 1 -> 2 воркера", LIMIT, True, 2000, 100.0, scale_to=2),
}


def run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(  # noqa: S603 - fixed argument lists
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=check
    )
    return result.stdout


def psql(sql: str) -> list[list[str]]:
    out = run(
        [
            *("docker", "compose", "exec", "-T", "postgres", "psql", "-U", "edu"),
            *("-d", "edu_graphrag", "-At", "-F", ",", "-c", sql),
        ]
    )
    return [line.split(",") for line in out.splitlines() if line and not line.startswith("DELETE")]


def scalar(sql: str) -> float:
    rows = psql(sql)
    return float(rows[0][0]) if rows and rows[0][0] else 0.0


def scale_workers(n: int) -> None:
    run(["docker", "compose", "up", "-d", "--wait", "--no-recreate", "--scale", f"worker={n}"])


_MEM = re.compile(r"([\d.]+)\s*([KMG]i?B)")
_UNITS = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}


def worker_memory_mb() -> float:
    out = run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}"])
    total = 0.0
    for line in out.splitlines():
        name, _, usage = line.partition("|")
        match = _MEM.match(usage)
        if "worker" in name and match:
            total += float(match.group(1)) * _UNITS[match.group(2)]
    return round(total, 1)


def worker_count() -> int:
    out = run(["docker", "compose", "ps", "--status", "running", "-q", "worker"])
    return len(out.split())


class Broker:
    def __init__(self) -> None:
        self.client = ManagementClient(RabbitSettings())

    def snapshot(self) -> dict[str, Any]:
        stats = self.client.stats()
        tasks = stats["queues"]["tasks"]
        dlq = stats["queues"]["tasks.dlq"]
        node = stats["nodes"][0]
        return {
            "messages": tasks["messages"],
            "ready": tasks["messages_ready"],
            "unacked": tasks["messages_unacknowledged"],
            "consumers": tasks["consumers"],
            "broker_mb": round((node["mem_used"] or 0) / 2**20, 1),
            "dlq": dlq["messages"],
        }

    def purge(self) -> None:
        for queue in ("tasks", "tasks.dlq"):
            self.client.send(ApiRequest("DELETE", f"/api/queues/%2F/{queue}/contents"))

    def apply_policy(self, max_length: int) -> None:
        self.client.apply_policy(max_length=max_length, delivery_limit=5)


@dataclass
class Sampler:
    """Samples broker, database and worker memory into `rows` until stopped."""

    broker: Broker
    t0: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    worker_mb: float = 0.0
    workers: int = 1

    def _slow(self) -> None:
        while not self.stop.is_set():
            self.worker_mb = worker_memory_mb()
            self.workers = worker_count()
            self.stop.wait(1.0)

    def _fast(self) -> None:
        tick = 0
        while not self.stop.is_set():
            tick += 1
            started = time.monotonic()
            row: dict[str, Any] = {"t": round(time.time() - self.t0, 1)}
            row.update(self.broker.snapshot())
            row["done"] = int(
                scalar("select count(*) from tasks where source <> 'api' and status = 'done'")
            )
            row["worker_mb"] = self.worker_mb
            row["workers"] = self.workers
            self.rows.append(row)
            self.stop.wait(max(0.0, 1.0 - (time.monotonic() - started)))

    def start(self) -> list[threading.Thread]:
        threads = [threading.Thread(target=f, daemon=True) for f in (self._slow, self._fast)]
        for thread in threads:
            thread.start()
        return threads


def shed_burst() -> list[dict[str, Any]]:
    """POST /api/v1/tasks while the queue is full; one row per request."""
    results: list[dict[str, Any]] = []
    for i in range(SHED_REQUESTS):
        body = json.dumps({"text": f"Решите уравнение номер {i}: x^2 - 5x + 6 = 0"}).encode()
        request = urllib.request.Request(  # noqa: S310 - local API
            f"{API}/api/v1/tasks", data=body, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        retry_after = ""
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            retry_after = exc.headers.get("Retry-After", "")
        results.append(
            {
                "i": i,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "retry_after": retry_after,
            }
        )
        time.sleep(0.05)
    return results


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reset(broker: Broker, scenario: Scenario) -> None:
    scale_workers(1)
    broker.purge()
    deadline = time.time() + 120
    while broker.snapshot()["unacked"] and time.time() < deadline:
        time.sleep(1)
    broker.purge()
    psql("truncate tasks")
    broker.apply_policy(scenario.max_length)
    time.sleep(3)


def percentile_rows(wall_t0: datetime) -> list[dict[str, Any]]:
    stamp = wall_t0.isoformat()
    epoch = f"extract(epoch from finished_at - timestamptz '{stamp}')"
    lat = "extract(epoch from finished_at - submitted_at)"
    rows = psql(
        f"select floor({epoch} / {BUCKET_S}) * {BUCKET_S}, count(*), "
        f"percentile_cont(0.5) within group (order by {lat}), "
        f"percentile_cont(0.95) within group (order by {lat}) "
        "from tasks where source <> 'api' and status = 'done' group by 1 order by 1"
    )
    return [
        {
            "t": float(r[0]),
            "count": int(r[1]),
            "p50": round(float(r[2]), 2),
            "p95": round(float(r[3]), 2),
        }
        for r in rows
    ]


def source_summary(log: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and "lost" in data:
            summary = data
    return summary


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    out = DATA / scenario.name
    out.mkdir(parents=True, exist_ok=True)
    broker = Broker()
    print(f"[{scenario.name}] reset", flush=True)
    reset(broker, scenario)

    cmd = ["uv", "run", "python", "-m", "edu_graphrag.pipeline.source", "--file", str(BATCH)]
    cmd += ["--limit", str(scenario.limit), "--metrics-out", str(out / "source.csv")]
    # The source CLI flag --no-backpressure is unusable (client rejects it), so scenario A
    # relies on the unbounded queue alone: nothing ever refuses the source.
    log_file = (out / "source.log").open("w", encoding="utf-8")
    t0 = time.time()
    wall_t0 = datetime.now(UTC)
    sampler = Sampler(broker, t0)
    threads = sampler.start()
    source = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)  # noqa: S603
    print(f"[{scenario.name}] source started", flush=True)

    shed: list[dict[str, Any]] = []
    shed_thread = threading.Thread(target=lambda: shed.extend(shed_burst()), daemon=True)
    scale_thread: threading.Thread | None = None
    scale_events: dict[str, float] = {}

    def do_scale() -> None:
        scale_to = scenario.scale_to or 2
        scale_events["requested"] = round(time.time() - t0, 1)
        scale_workers(scale_to)
        scale_events["ready"] = round(time.time() - t0, 1)

    shed_started = scale_started = False
    while time.time() - t0 < scenario.observe_s:
        elapsed = time.time() - t0
        if scenario.backpressure and not shed_started and elapsed >= SHED_AT_S:
            shed_thread.start()
            shed_started = True
        if scenario.scale_to and not scale_started and elapsed >= SCALE_AT_S:
            scale_thread = threading.Thread(target=do_scale, daemon=True)
            scale_thread.start()
            scale_started = True
        if source.poll() is not None and scenario.backpressure:
            break
        time.sleep(0.5)
    if source.poll() is None:
        source.terminate()
    source.wait()
    log_file.close()
    source_end = round(time.time() - t0, 1)
    if shed_started:
        shed_thread.join(timeout=60)
    if scale_thread:
        scale_thread.join(timeout=120)

    drained = False
    if scenario.backpressure:
        deadline = time.time() + 120
        while time.time() < deadline:
            snap = broker.snapshot()
            if snap["messages"] == 0:
                drained = True
                break
            time.sleep(1)
        time.sleep(3)
    sampler.stop.set()
    for thread in threads:
        thread.join(timeout=10)

    summary = source_summary(out / "source.log")
    done = int(scalar("select count(*) from tasks where source <> 'api' and status = 'done'"))
    failed = int(scalar("select count(*) from tasks where status = 'failed'"))
    api_rows = int(scalar("select count(*) from tasks where source = 'api'"))
    api_pending = int(
        scalar("select count(*) from tasks where source = 'api' and status = 'pending'")
    )
    final = broker.snapshot()
    latency = percentile_rows(wall_t0)
    write_csv(out / "timeline.csv", TIMELINE_FIELDS, sampler.rows)
    write_csv(out / "latency.csv", ("t", "count", "p50", "p95"), latency)
    if shed:
        write_csv(out / "shed.csv", ("i", "status", "ms", "retry_after"), shed)
    result: dict[str, Any] = {
        "scenario": scenario.name,
        "published_total": summary.get("published_total"),
        "rejected_total": summary.get("rejected_total"),
        "source_lost": summary.get("lost"),
        "source_end_s": source_end,
        "done": done,
        "failed": failed,
        "dlq": final["dlq"],
        "drained": drained,
        "max_depth": max(r["messages"] for r in sampler.rows),
        "scale": scale_events,
        "shed_202": sum(1 for r in shed if r["status"] == 202),
        "shed_503": sum(1 for r in shed if r["status"] == 503),
        "api_rows": api_rows,
        "api_pending": api_pending,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    broker.purge()
    return result


def restore() -> None:
    broker = Broker()
    broker.apply_policy(LIMIT)
    scale_workers(1)
    time.sleep(3)
    broker.purge()
    psql("truncate tasks")


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        return [
            {k: float(v) if v not in ("", None) else float("nan") for k, v in row.items()}
            for row in csv.DictReader(file)
        ]


def plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    colors = {"A": "#d1495b", "B": "#2e86ab", "C": "#3a9d5d"}
    names = {k: f"{k}: {v.title}" for k, v in SCENARIOS.items()}
    data: dict[str, dict[str, list[dict[str, float]]]] = {}
    for key in SCENARIOS:
        folder = DATA / key
        if (folder / "timeline.csv").exists():
            data[key] = {
                "timeline": read_csv(folder / "timeline.csv"),
                "source": read_csv(folder / "source.csv"),
                "latency": read_csv(folder / "latency.csv"),
            }

    joined = None
    if "C" in data:
        joined = next((r["t"] for r in data["C"]["timeline"] if r["consumers"] >= 2), None)

    def save(fig: Any, name: str) -> None:
        if joined is not None:
            for ax in fig.axes:
                if ax.get_xlabel().startswith("время"):
                    ax.axvline(joined, color=colors["C"], ls=":", lw=1.2)
        fig.tight_layout()
        fig.savefig(PLOTS / name, dpi=110)
        plt.close(fig)

    # 1. queue depth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, zoom in zip(axes, (False, True), strict=True):
        for key, d in data.items():
            ax.plot(
                [r["t"] for r in d["timeline"]],
                [r["messages"] for r in d["timeline"]],
                color=colors[key],
                label=names[key],
            )
        ax.axhline(LIMIT, color="black", ls="--", lw=1, label=f"лимит очереди = {LIMIT}")
        ax.set_xlabel("время, с")
        ax.set_ylabel("сообщений в очереди tasks")
        ax.grid(alpha=0.3)
        if zoom:
            ax.set_ylim(0, 260)
            ax.set_title("то же, масштаб около лимита")
        else:
            ax.set_title("Глубина очереди")
            ax.legend(fontsize=8)
    save(fig, "queue_depth.png")

    # 2. source rate
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, d in data.items():
        if key == "A":
            continue
        ax.plot(
            [r["elapsed_s"] for r in d["source"]],
            [r["accepted_per_s"] for r in d["source"]],
            color=colors[key],
            label=names[key],
        )
    ax.axhline(20, color="gray", ls="--", lw=1, label="1 воркер: 4 потока / 0.2 с = 20 задач/с")
    ax.axhline(40, color="gray", ls=":", lw=1, label="2 воркера: 40 задач/с")
    ax.set_xlabel("время, с")
    ax.set_ylabel("принято брокером, задач/с")
    ax.set_title("Скорость генерации задач источником (backpressure)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "source_rate.png")

    # 3. latency
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, field_name in zip(axes, ("p50", "p95"), strict=True):
        for key, d in data.items():
            ax.plot(
                [r["t"] for r in d["latency"]],
                [r[field_name] for r in d["latency"]],
                color=colors[key],
                marker="o",
                ms=3,
                label=names[key],
            )
        ax.set_xlabel("время завершения задачи, с")
        ax.set_ylabel(f"{field_name} времени в системе, с")
        ax.set_title(f"Время выполнения задачи ({field_name}, окна по {BUCKET_S} с)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    save(fig, "task_latency.png")

    # 4. memory
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for key, d in data.items():
        axes[0].plot(
            [r["t"] for r in d["timeline"]],
            [r["broker_mb"] for r in d["timeline"]],
            color=colors[key],
            label=names[key],
        )
        axes[1].plot(
            [r["elapsed_s"] for r in d["source"]],
            [r["rss_mb"] for r in d["source"]],
            color=colors[key],
            label=names[key],
        )
        axes[2].plot(
            [r["t"] for r in d["timeline"]],
            [r["worker_mb"] for r in d["timeline"]],
            color=colors[key],
            label=names[key],
        )
    for ax, title in zip(
        axes,
        ("RabbitMQ (mem_used)", "источник (RSS)", "воркеры (сумма, docker stats)"),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("время, с")
        ax.set_ylabel("МБ")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    save(fig, "memory.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"])
    parser.add_argument("--plots", action="store_true", help="draw figures from data/experiment")
    parser.add_argument("--restore", action="store_true", help="policy 200, 1 worker, empty queues")
    args = parser.parse_args()
    if args.scenario:
        for key in SCENARIOS if args.scenario == "all" else [args.scenario]:
            run_scenario(SCENARIOS[key])
        restore()
    if args.restore:
        restore()
    if args.plots:
        plots()


if __name__ == "__main__":
    main()
