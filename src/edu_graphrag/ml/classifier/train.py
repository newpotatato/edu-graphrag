"""Train subject-classifier variants and log every run to MLflow.

Every run uses the same prepared data and the same grouped split (fixed seed), so the runs of
the `subject-classifier` experiment are directly comparable. The primary metric is
`test_macro_f1`: classes are heavily imbalanced (see the EDA), and accuracy would be dominated
by the two largest subjects.

Usage:
    uv run python -m edu_graphrag.ml.classifier.train --variant baseline
    uv run python -m edu_graphrag.ml.classifier.train --all

The tracking URI comes from `MLFLOW_TRACKING_URI`.
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import structlog
from sklearn.pipeline import Pipeline

from edu_graphrag.ml.classifier import evaluation
from edu_graphrag.ml.classifier.data import (
    DEFAULT_MIN_DOCS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    GROUP_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    PreparedData,
    Split,
    class_distribution,
    prepare_chunks,
    split_by_document,
    split_stats,
)
from edu_graphrag.ml.classifier.model import (
    INPUT_COLUMN,
    PIPELINE_ARTIFACT,
    SubjectClassifierModel,
)
from edu_graphrag.ml.classifier.variants import (
    VARIANTS,
    Variant,
    build_pipeline,
    linear_coefficients,
)
from edu_graphrag.ml.eda import ProcessedDataset, load_dataset

logger = structlog.get_logger(__name__)

EXPERIMENT_NAME = "subject-classifier"
DATASET_NAME = "study-chunks"
PRIMARY_METRIC = "test_macro_f1"
EXAMPLE_CHARS = 300


@dataclass(frozen=True)
class TrainingData:
    dataset: ProcessedDataset
    prepared: PreparedData
    split: Split
    seed: int
    test_size: float

    @property
    def labels(self) -> list[str]:
        return sorted(self.prepared.chunks[TARGET_COLUMN].unique())


@dataclass
class RunResult:
    pipeline: Pipeline
    params: dict[str, Any]
    metrics: dict[str, float]
    artifacts: list[Path] = field(default_factory=list)


def load_training_data(
    data_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    min_docs: int = DEFAULT_MIN_DOCS,
) -> TrainingData:
    dataset = load_dataset(data_dir)
    prepared = prepare_chunks(dataset.documents, dataset.chunks, min_docs=min_docs)
    split = split_by_document(prepared.chunks, test_size=test_size, seed=seed)
    data = TrainingData(dataset, prepared, split, seed, test_size)
    logger.info(
        "training_data_ready",
        dataset_version=dataset.version,
        dropped_subjects=prepared.dropped_subjects,
        duplicates_removed=prepared.n_duplicate_docs_removed,
        conflicting_hashes=len(prepared.conflicting_hashes),
        **split_stats(split),
    )
    return data


def data_params(data: TrainingData) -> dict[str, Any]:
    """Params describing the data and the split (identical for every run)."""
    prepared = data.prepared
    return {
        "dataset_version": data.dataset.version,
        "seed": data.seed,
        "test_size": data.test_size,
        "n_classes": len(data.labels),
        "n_dropped_subjects": len(prepared.dropped_subjects),
        "dropped_subjects": ",".join(prepared.dropped_subjects) or "none",
        "n_duplicate_docs_removed": prepared.n_duplicate_docs_removed,
        "n_conflicting_duplicate_groups": len(prepared.conflicting_hashes),
        **split_stats(data.split),
    }


def fit_and_evaluate(variant: Variant, data: TrainingData, out_dir: Path) -> RunResult:
    """Fit on train, evaluate on train and test, write report artifacts into `out_dir`."""
    train, test = data.split.train, data.split.test
    pipeline = build_pipeline(variant, data.seed)
    started = time.perf_counter()
    pipeline.fit(train[TEXT_COLUMN].tolist(), train[TARGET_COLUMN].to_numpy())
    fit_seconds = time.perf_counter() - started

    train_pred = pipeline.predict(train[TEXT_COLUMN].tolist())
    started = time.perf_counter()
    test_proba = pipeline.predict_proba(test[TEXT_COLUMN].tolist())
    predict_seconds = time.perf_counter() - started
    classes = [str(c) for c in pipeline.classes_]
    test_pred = [classes[i] for i in test_proba.argmax(axis=1)]
    confidence = test_proba.max(axis=1)
    y_train, y_test = train[TARGET_COLUMN].tolist(), test[TARGET_COLUMN].tolist()

    votes = evaluation.document_votes(test[GROUP_COLUMN].tolist(), y_test, test_pred)
    metrics = {
        **evaluation.classification_metrics(y_test, test_pred, "test"),
        **evaluation.classification_metrics(y_train, list(train_pred), "train"),
        "test_doc_accuracy": evaluation.document_accuracy(votes),
        "fit_seconds": fit_seconds,
        "predict_ms_per_chunk": 1000 * predict_seconds / max(len(test), 1),
        "n_features": float(len(pipeline.named_steps["tfidf"].vocabulary_)),
    }
    metrics["train_test_macro_f1_gap"] = metrics["train_macro_f1"] - metrics["test_macro_f1"]

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = data.labels
    report = evaluation.report_table(y_test, test_pred, labels)
    tables = {
        "classification_report.csv": report,
        "misclassified.csv": evaluation.misclassified(test, test_pred, confidence),
        "document_votes.csv": votes,
        "class_distribution.csv": class_distribution(data.split).reset_index(),
    }
    coef = linear_coefficients(pipeline)
    if coef is not None:
        names = pipeline.named_steps["tfidf"].get_feature_names_out()
        tables["top_features.csv"] = evaluation.top_features(coef, names, classes)
    artifacts = []
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False, encoding="utf-8")
        artifacts.append(out_dir / name)
    report_json = out_dir / "classification_report.json"
    report_json.write_text(
        report.set_index("class").to_json(orient="index", indent=2, force_ascii=False),
        encoding="utf-8",
    )
    prep_json = out_dir / "data_preparation.json"
    prep_json.write_text(
        json.dumps(
            {
                "dropped_subjects": data.prepared.dropped_subjects,
                "conflicting_hashes": data.prepared.conflicting_hashes,
                "n_duplicate_docs_removed": data.prepared.n_duplicate_docs_removed,
                "n_docs_not_ok": data.prepared.n_docs_not_ok,
                "split": split_stats(data.split),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifacts += [report_json, prep_json]
    artifacts.append(
        evaluation.plot_confusion_matrix(
            y_test,
            test_pred,
            labels,
            out_dir / "confusion_matrix.png",
            f"{variant.name}: normalised confusion matrix (test, macro-F1 "
            f"{metrics['test_macro_f1']:.3f})",
        )
    )
    params = {**variant.params(), "variant": variant.name, **data_params(data)}
    logger.info(
        "variant_evaluated",
        variant=variant.name,
        **{k: round(v, 4) for k, v in metrics.items()},
    )
    return RunResult(pipeline=pipeline, params=params, metrics=metrics, artifacts=artifacts)


def git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def model_requirements() -> list[str]:
    """What loading the model needs; all of it is in the API image (web group)."""
    return [
        f"{name}=={package_version(name)}"
        for name in ("mlflow-skinny", "scikit-learn", "pandas", "numpy", "joblib")
    ]


def input_example(test: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    texts = test[TEXT_COLUMN].head(n).str.slice(0, EXAMPLE_CHARS).tolist()
    return pd.DataFrame({INPUT_COLUMN: texts})


def log_run(variant: Variant, data: TrainingData, result: RunResult, work_dir: Path) -> str:
    """One MLflow run: params, tags, datasets (lineage), metrics, reports and the model."""
    import mlflow
    from mlflow.data.pandas_dataset import from_pandas
    from mlflow.models import infer_signature

    source = data.dataset.chunks_path.resolve().as_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{variant.name}-{data.dataset.version}") as run:
        mlflow.set_tags(
            {
                "variant": variant.name,
                "dataset_version": data.dataset.version,
                "git_commit": git_commit(),
                "estimator": variant.estimator,
                "mlflow.note.content": variant.description,
            }
        )
        mlflow.log_params(result.params)
        for frame, context in ((data.split.train, "training"), (data.split.test, "evaluation")):
            ds = from_pandas(frame, source=source, name=DATASET_NAME, targets=TARGET_COLUMN)
            mlflow.log_input(ds, context=context)
        mlflow.log_metrics(result.metrics)
        for path in result.artifacts:
            mlflow.log_artifact(str(path), artifact_path="reports")

        pipeline_path = work_dir / "pipeline.joblib"
        joblib.dump(result.pipeline, pipeline_path)
        example = input_example(data.split.test)
        output = SubjectClassifierModel(result.pipeline).predict(None, example)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=SubjectClassifierModel(),
            artifacts={PIPELINE_ARTIFACT: str(pipeline_path)},
            signature=infer_signature(example, output),
            input_example=example,
            pip_requirements=model_requirements(),
        )
        run_id: str = run.info.run_id
    logger.info("run_logged", variant=variant.name, run_id=run_id)
    return run_id


def train_variant(variant: Variant, data: TrainingData) -> tuple[str, RunResult]:
    with tempfile.TemporaryDirectory(prefix=f"subject-clf-{variant.name}-") as tmp:
        work_dir = Path(tmp)
        result = fit_and_evaluate(variant, data, work_dir / "reports")
        run_id = log_run(variant, data, result, work_dir)
    return run_id, result


def main(argv: Sequence[str] | None = None) -> int:
    from edu_graphrag.logging_config import configure_logging

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("INFO", json_logs=False)
    parser = argparse.ArgumentParser(description="Train subject-classifier variants")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--variant", choices=sorted(VARIANTS))
    target.add_argument("--all", action="store_true", help="train every variant")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--min-docs", type=int, default=DEFAULT_MIN_DOCS)
    args = parser.parse_args(argv)

    data = load_training_data(
        args.data, seed=args.seed, test_size=args.test_size, min_docs=args.min_docs
    )
    names = list(VARIANTS) if args.all else [args.variant]
    print(f"dataset {data.dataset.version}: {split_stats(data.split)}")
    print(f"dropped subjects (< {args.min_docs} docs): {data.prepared.dropped_subjects}")
    for name in names:
        run_id, result = train_variant(VARIANTS[name], data)
        m = result.metrics
        print(
            f"{name:14s} run={run_id} {PRIMARY_METRIC}={m[PRIMARY_METRIC]:.4f} "
            f"acc={m['test_accuracy']:.4f} doc_acc={m['test_doc_accuracy']:.4f} "
            f"gap={m['train_test_macro_f1_gap']:.4f} fit={m['fit_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
