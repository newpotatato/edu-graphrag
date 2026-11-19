"""Run local Compose modes with matching app and Collector configuration."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

_MODES = {
    "logs": ("alloy", "loki"),
    "traces": ("otel-collector", "tempo"),
    "metrics": ("otel-collector", "prometheus"),
    "observability": ("alloy", "loki", "otel-collector", "tempo", "prometheus", "grafana"),
}
_DEMO = {
    "COMPOSE_PROJECT_NAME": "edu-graphrag-obs-demo",
    "APP_IMAGE": "edu-graphrag:hw5-demo",
    "APP_PORT": "18000",
    "POSTGRES_PORT": "15432",
    "NEO4J_HTTP_PORT": "17474",
    "NEO4J_BOLT_PORT": "17687",
    "MLFLOW_PORT": "15000",
    "SEAWEEDFS_PORT": "18333",
    "RABBITMQ_PORT": "15673",
    "RABBITMQ_MANAGEMENT_PORT": "25672",
    "WORKER_PORT_RANGE": "18101-18104",
    "GRAFANA_PORT": "13000",
    "OBS_MODEL_NAME": "demo-subject-classifier",
    "OBS_MODEL_ALIAS": "champion",
    "DLQ_RECORDER_ENABLED": "false",
    "TELEMETRY_DLQ_TAP": "true",
    "TELEMETRY_DEMO_POISON_ID": "a036b3f2-ec4e-44f0-bfe4-852478b78f3a",
}


class Compose:
    def __init__(self, mode: str, *, demo: bool = False) -> None:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker CLI not found")
        self.command = [
            docker,
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "observability/compose.yaml",
            "--profile",
            mode,
        ]
        self.env = os.environ | {
            "OBS_SDK_DISABLED": "true" if mode == "logs" else "false",
            "OBS_TRACES_EXPORTER": "otlp" if mode in {"traces", "observability"} else "none",
            "OBS_METRICS_EXPORTER": "otlp" if mode in {"metrics", "observability"} else "none",
            "OBS_COLLECTOR_MODE": "full" if mode == "observability" else mode,
            "OBS_LOG_DRAIN_SECONDS": "3" if mode in {"logs", "observability"} else "0",
        }
        if demo:
            self.env.update(_DEMO)
            self.env["OTEL_RESOURCE_ATTRIBUTES"] = "deployment.environment.name=demo"
        self.services = _MODES[mode]

    def run(self, *args: str) -> None:
        # Input is local CLI arguments forwarded verbatim, not shell-evaluated.
        subprocess.run(  # noqa: S603
            [*self.command, *args],
            cwd=Path(__file__).resolve().parents[1],
            env=self.env,
            check=True,
        )

    def up(self, *, demo: bool) -> None:
        # Launch the business services independently of observability readiness.
        self.run("up", "-d", "--wait", "--build", "app")
        if demo:
            self.run("run", "--rm", "--no-deps", "demo-seed")
        self.run("up", "-d", "--wait", "worker")
        self.run("up", "-d", "--wait", "--build", *self.services)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for verb in ("up", "config", "compose"):
        sub = subs.add_parser(verb)
        sub.add_argument("mode", choices=_MODES)
        if verb == "compose":
            sub.add_argument("args", nargs=argparse.REMAINDER)
    subs.add_parser("demo-up")
    demo = subs.add_parser("demo")
    demo.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    is_demo = args.action in {"demo", "demo-up"}
    mode = "observability" if is_demo else args.mode
    project = Compose(mode, demo=is_demo)
    if args.action in {"up", "demo-up"}:
        project.up(demo=is_demo)
    elif args.action == "config":
        project.run("config", "--quiet")
    else:
        project.run(*args.args)


if __name__ == "__main__":
    main()
