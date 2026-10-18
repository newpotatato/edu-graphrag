"""Register the subject classifier in the MLflow Model Registry and set its aliases.

- the baseline run is registered first (version 1 on a fresh registry);
- the best run by `test_macro_f1` is selected with `mlflow.search_runs`, not hardcoded;
- alias `champion` -> best version (what the API serves), `challenger` -> baseline, or the
  second best run when the baseline itself is the best.

Idempotent: a run that already has a version is not registered again; aliases, tags and
descriptions are simply set again.

Usage: `uv run python -m edu_graphrag.ml.classifier.register [--dataset-version ...]`
The tracking URI comes from `MLFLOW_TRACKING_URI`.
"""

import argparse
import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

from edu_graphrag.ml.classifier.train import EXPERIMENT_NAME, PRIMARY_METRIC

logger = structlog.get_logger(__name__)

MODEL_NAME = "subject-classifier"
CHAMPION = "champion"
CHALLENGER = "challenger"
BASELINE_VARIANT = "baseline"
MODEL_DESCRIPTION = (
    "Subject classifier for study-material fragments: which course a text chunk belongs to. "
    "Input: DataFrame with a `text` column. Output per row: `label`, `confidence` "
    "(max class probability) and `top3` [{label, p}]. TF-IDF + linear model, trained on the "
    "edu-graphrag `study-chunks` dataset with a document-grouped split; primary metric "
    f"`{PRIMARY_METRIC}`. Alias `{CHAMPION}` is served by the API, `{CHALLENGER}` is the "
    "comparison model."
)


@dataclass(frozen=True)
class Selection:
    """Runs chosen for the registry: baseline, best (champion) and challenger."""

    baseline: pd.Series
    best: pd.Series
    challenger: pd.Series


def select_runs(runs: pd.DataFrame) -> Selection:
    """`runs` is `mlflow.search_runs` output ordered by the primary metric, best first.

    The baseline is the latest run of the baseline variant on the best run's dataset version.
    """
    if runs.empty:
        raise ValueError(f"no finished runs with {PRIMARY_METRIC} in {EXPERIMENT_NAME!r}")
    best = runs.iloc[0]
    same_data = runs[runs["tags.dataset_version"] == best["tags.dataset_version"]]
    baselines = same_data[same_data["tags.variant"] == BASELINE_VARIANT]
    if baselines.empty:
        raise ValueError(f"no {BASELINE_VARIANT!r} run for dataset {best['tags.dataset_version']}")
    baseline = baselines.sort_values("start_time", ascending=False).iloc[0]
    challenger = baseline
    if baseline["run_id"] == best["run_id"]:
        others = same_data[same_data["run_id"] != best["run_id"]]
        if not others.empty:
            challenger = others.iloc[0]
    return Selection(baseline=baseline, best=best, challenger=challenger)


def version_tags(run: pd.Series) -> dict[str, str]:
    return {
        "variant": str(run["tags.variant"]),
        PRIMARY_METRIC: f"{float(run[f'metrics.{PRIMARY_METRIC}']):.4f}",
        "test_accuracy": f"{float(run['metrics.test_accuracy']):.4f}",
        "dataset_version": str(run["tags.dataset_version"]),
        "git_commit": str(run.get("tags.git_commit", "unknown")),
    }


def version_description(run: pd.Series) -> str:
    return (
        f"{run['tags.variant']} ({run.get('tags.mlflow.note.content', '')}); "
        f"{PRIMARY_METRIC}={float(run[f'metrics.{PRIMARY_METRIC}']):.4f}, "
        f"test_accuracy={float(run['metrics.test_accuracy']):.4f}, "
        f"test_doc_accuracy={float(run['metrics.test_doc_accuracy']):.4f}; "
        f"dataset {run['tags.dataset_version']}, run {run['run_id']}"
    )


def search_runs(dataset_version: str | None = None) -> pd.DataFrame:
    import mlflow

    filters = ["attributes.status = 'FINISHED'", f"metrics.{PRIMARY_METRIC} >= 0"]
    if dataset_version:
        filters.append(f"tags.dataset_version = '{dataset_version}'")
    runs: pd.DataFrame = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=" and ".join(filters),
        order_by=[f"metrics.{PRIMARY_METRIC} DESC", "attributes.start_time DESC"],
        output_format="pandas",
    )
    return runs


def ensure_version(client: Any, run_id: str) -> str:
    """Version registered from `run_id`; registers it only if it does not exist yet."""
    import mlflow

    existing = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    if existing:
        version = str(min(int(mv.version) for mv in existing))
        logger.info("model_version_exists", run_id=run_id, version=version)
        return version
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    logger.info("model_version_registered", run_id=run_id, version=mv.version)
    return str(mv.version)


def ensure_registered_model(client: Any) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME, description=MODEL_DESCRIPTION)
    client.update_registered_model(MODEL_NAME, description=MODEL_DESCRIPTION)
    client.set_registered_model_tag(MODEL_NAME, "task", "text-classification")
    client.set_registered_model_tag(MODEL_NAME, "primary_metric", PRIMARY_METRIC)


def register(dataset_version: str | None = None) -> dict[str, str]:
    """Registers the selected runs and sets aliases; returns {alias: version}."""
    from mlflow import MlflowClient

    client = MlflowClient()
    selection = select_runs(search_runs(dataset_version))
    ensure_registered_model(client)
    versions: dict[str, str] = {}
    # Baseline first: on a fresh registry it becomes version 1, the best run version 2.
    for run in (selection.baseline, selection.best, selection.challenger):
        run_id = str(run["run_id"])
        if run_id in versions:
            continue
        version = ensure_version(client, run_id)
        versions[run_id] = version
        for key, value in version_tags(run).items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)
        client.update_model_version(MODEL_NAME, version, description=version_description(run))
    aliases = {
        CHAMPION: versions[str(selection.best["run_id"])],
        CHALLENGER: versions[str(selection.challenger["run_id"])],
    }
    for alias, version in aliases.items():
        client.set_registered_model_alias(MODEL_NAME, alias, version)
    logger.info("aliases_set", model=MODEL_NAME, **aliases)
    return aliases


def main(argv: Sequence[str] | None = None) -> int:
    from edu_graphrag.logging_config import configure_logging

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("INFO", json_logs=False)
    parser = argparse.ArgumentParser(description=f"Register {MODEL_NAME} and set its aliases")
    parser.add_argument("--dataset-version", default=None, help="only consider runs on it")
    args = parser.parse_args(argv)
    aliases = register(args.dataset_version)
    for alias, version in aliases.items():
        print(f"models:/{MODEL_NAME}@{alias} -> version {version}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
