"""Small, isolated fixtures for the observability defense; no real study data."""

import argparse
import asyncio
import json
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path
from uuid import UUID, uuid4

import aio_pika
import structlog
from opentelemetry.trace import SpanKind
from sqlalchemy import select

from edu_graphrag import telemetry
from edu_graphrag.components.postgres import PostgresComponent
from edu_graphrag.config import PostgresSettings, RabbitSettings
from edu_graphrag.logging_config import configure_logging
from edu_graphrag.pipeline.messages import TaskMessage
from edu_graphrag.pipeline.source import task_id_for
from edu_graphrag.pipeline.store import tasks
from edu_graphrag.pipeline.topology import TASKS_EXCHANGE, TASKS_ROUTING_KEY

logger = structlog.get_logger(__name__)
MODEL_NAME = "demo-subject-classifier"
ALIAS = "champion"
POISON_ID = UUID("a036b3f2-ec4e-44f0-bfe4-852478b78f3a")
CORPUS = (
    ("matrix determinant eigenvalue vector", "algebra"),
    ("linear combination of basis vectors", "algebra"),
    ("variance sample estimator probability", "statistics"),
    ("confidence interval and standard error", "statistics"),
    ("bounded continuous function derivative", "analysis"),
    ("sequence converges limit calculus", "analysis"),
)


def seed() -> None:
    import joblib
    import mlflow
    import pandas as pd
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from edu_graphrag.components.mlflow_model import MlflowModelRegistry
    from edu_graphrag.ml.classifier.model import PIPELINE_ARTIFACT, SubjectClassifierModel

    registry = MlflowModelRegistry(mlflow.get_tracking_uri())
    try:
        try:
            ref = registry.resolve(MODEL_NAME, ALIAS)
        except MlflowException as exc:
            if exc.error_code not in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                raise
            texts, labels = zip(*CORPUS, strict=True)
            pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=42))
            pipeline.fit(texts, labels)
            example = pd.DataFrame({"text": [texts[0]]})
            output = SubjectClassifierModel(pipeline).predict(None, example)
            mlflow.set_experiment("observability-demo")
            with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name="demo-seed"):
                artifact = Path(tmp) / "pipeline.joblib"
                joblib.dump(pipeline, artifact)
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=SubjectClassifierModel(),
                    artifacts={PIPELINE_ARTIFACT: str(artifact)},
                    input_example=example,
                    signature=infer_signature(example, output),
                    pip_requirements=[
                        f"{p}=={package_version(p)}"
                        for p in ("mlflow-skinny", "scikit-learn", "joblib", "pandas", "numpy")
                    ],
                )
                version = mlflow.register_model(info.model_uri, MODEL_NAME)
                mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
            ref = registry.resolve(MODEL_NAME, ALIAS)
        example = pd.DataFrame({"text": [CORPUS[0][0]]})
        registry.load(ref).predict(example)
        logger.info("demo_model_ready", model=ref.name, alias=ref.alias, version=ref.version)
    finally:
        registry.close()


def fixture(path: Path, *, count: int) -> None:
    run = uuid4().hex
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("x", encoding="utf-8") as file:
        for i in range(count):
            text, _ = CORPUS[i % len(CORPUS)]
            file.write(json.dumps({"id": f"{run}:{i}", "text": text}) + "\n")
    logger.info("demo_fixture_created", file=path.name, batch_id=run, tasks=count)


async def poison(*, malformed: bool) -> None:
    task = TaskMessage.new(
        "synthetic failure fixture", source="batch", task_id=uuid4() if malformed else POISON_ID
    )
    async with telemetry.running("edu-source"):
        with telemetry.span(
            "publish.demo_poison", {"task.id": str(task.task_id)}, kind=SpanKind.PRODUCER
        ):
            connection = await aio_pika.connect(RabbitSettings().url, timeout=5)
            async with connection:
                channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
                exchange = await channel.get_exchange(TASKS_EXCHANGE.name, ensure=False)
                await exchange.publish(
                    aio_pika.Message(
                        b"{}" if malformed else task.model_dump_json().encode(),
                        correlation_id=str(task.task_id),
                        message_id=str(task.task_id),
                        headers=dict(telemetry.propagate_headers()),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    TASKS_ROUTING_KEY,
                    mandatory=True,
                    timeout=3,
                )
            logger.info("demo_poison_sent", task_id=str(task.task_id), malformed=malformed)


async def reconcile(path: Path, threshold: float) -> None:
    """All IDs in a fully published fixture; missing and failed count against the SLO."""
    with path.open(encoding="utf-8") as file:
        ids = {task_id_for(json.loads(line)["id"]) for line in file if line.strip()}
    postgres = PostgresComponent(PostgresSettings(), connect_timeout=5)
    await postgres.startup()
    ok = failed = late = bad = 0
    matched = 0
    try:
        async with postgres.engine.connect() as conn:
            all_ids = list(ids)
            for offset in range(0, len(all_ids), 1000):
                result = await conn.execute(
                    select(tasks).where(tasks.c.task_id.in_(all_ids[offset : offset + 1000]))
                )
                for row in result.mappings():
                    matched += 1
                    if row["status"] == "failed":
                        failed += 1
                    elif row["status"] != "done" or row["finished_at"] is None:
                        bad += 1
                    elif (row["finished_at"] - row["submitted_at"]).total_seconds() <= threshold:
                        ok += 1
                    else:
                        late += 1
    finally:
        await postgres.shutdown()
    logger.info(
        "demo_slo",
        expected=len(ids),
        good=ok,
        late=late,
        failed=failed,
        unfinished=bad,
        missing=len(ids) - matched,
        threshold_s=threshold,
        sli=ok / len(ids) if ids else None,
        meets_99_percent=bool(ids) and ok / len(ids) >= 0.99,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed", help="create and verify the small demo-only MLflow model")
    generate = commands.add_parser("fixture", help="create a new synthetic batch (no overwrite)")
    generate.add_argument("file", type=Path)
    generate.add_argument("--count", type=int, default=6000)
    send = commands.add_parser("poison", help="send one demo message to the configured broker")
    send.add_argument("--malformed", action="store_true")
    verify = commands.add_parser(
        "slo", help="reconcile a fully confirmed cohort after its deadline"
    )
    verify.add_argument("file", type=Path)
    verify.add_argument("--threshold", type=float, default=30)
    args = parser.parse_args()
    configure_logging(service="edu-source" if args.command == "poison" else "edu-demo")
    if args.command == "seed":
        seed()
    elif args.command == "fixture":
        if not 1 <= args.count <= 10_000:
            parser.error("--count must be between 1 and 10000")
        fixture(args.file, count=args.count)
    elif args.command == "poison":
        asyncio.run(poison(malformed=args.malformed))
    else:
        asyncio.run(reconcile(args.file, args.threshold))


if __name__ == "__main__":
    main()
