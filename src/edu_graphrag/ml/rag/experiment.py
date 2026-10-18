"""Evaluate retriever variants on the QA set, log them to MLflow and register the best.

Each variant is one run in the `study-retriever` experiment: params (variant, tokenizer,
embedding model, chunking, dataset version), both datasets as run inputs, quality and
latency metrics, evaluation artifacts and the self-contained pyfunc model (see `model`).

Idempotent: a run is keyed by (variant config, dataset version, QA digest); an existing
finished run with the same key is reused instead of retrained (`--force` retrains), and a
run that is already registered is not registered again. Registration: the baseline (`bm25`)
becomes the `challenger`, the best run by `recall_at_5` (picked with `search_runs`, ties
broken by MRR) becomes the `champion`.

Usage: `uv run python -m edu_graphrag.ml.rag.experiment --data data/processed --qa data/qa/qa.jsonl`
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess  # nosec B404 - only to read the git commit
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from matplotlib.figure import Figure

from edu_graphrag.ml.rag.index import (
    Embedder,
    FastEmbedder,
    Hit,
    IndexConfig,
    Retriever,
    build_index,
    embed_passages,
    load_index,
)
from edu_graphrag.ml.rag.metrics import (
    first_hit_rank,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from edu_graphrag.ml.rag.model import (
    DEFAULT_K,
    EMBEDDER_ARTIFACT,
    INDEX_ARTIFACT,
    RESULTS_COLUMN,
    TEXT_COLUMN,
    RetrieverModel,
)
from edu_graphrag.ml.rag.qa import QAItem, load_qa, qa_frame, validate_qa

logger = structlog.get_logger(__name__)

EXPERIMENT_NAME = "study-retriever"
MODEL_NAME = "study-retriever"
BASELINE_VARIANT = "bm25"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PRIMARY_METRIC = "recall_at_5"
EVAL_K = 10
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VARIANTS: dict[str, IndexConfig] = {
    "bm25": IndexConfig(method="bm25", tokenizer="word"),
    "bm25_prefix5": IndexConfig(method="bm25", tokenizer="prefix5"),
    "dense": IndexConfig(method="dense", embed_model=EMBED_MODEL),
    "hybrid_rrf": IndexConfig(method="hybrid", tokenizer="word", embed_model=EMBED_MODEL),
    "hybrid_rrf_prefix5": IndexConfig(
        method="hybrid", tokenizer="prefix5", embed_model=EMBED_MODEL
    ),
}

EmbedderLoader = Callable[[str], tuple[Embedder, Path | None]]


# ---------------------------------------------------------------- evaluation (pure)


@dataclass(frozen=True, slots=True)
class QuestionResult:
    item: QAItem
    hits: list[Hit]
    seconds: float

    @property
    def hit_rank(self) -> int | None:
        return first_hit_rank((h.chunk.doc_id for h in self.hits), self.item.gold_docs)

    @property
    def chunk_rank(self) -> int | None:
        return first_hit_rank((h.chunk.chunk_id for h in self.hits), {self.item.chunk_id})


def run_queries(
    retriever: Retriever, items: Sequence[QAItem], k: int = EVAL_K
) -> list[QuestionResult]:
    results = []
    for item in items:
        started = time.perf_counter()
        hits = retriever.search(item.question, k)
        results.append(QuestionResult(item, hits, time.perf_counter() - started))
    return results


def aggregate_metrics(results: Sequence[QuestionResult]) -> dict[str, float]:
    n = len(results) or 1
    ranks = [r.hit_rank for r in results]
    latencies_ms = [r.seconds * 1000 for r in results]
    return {
        "recall_at_1": sum(recall_at_k(rank, 1) for rank in ranks) / n,
        "recall_at_5": sum(recall_at_k(rank, 5) for rank in ranks) / n,
        "recall_at_10": sum(recall_at_k(rank, 10) for rank in ranks) / n,
        "mrr_at_10": sum(mrr_at_k(rank, 10) for rank in ranks) / n,
        "ndcg_at_10": sum(ndcg_at_k(rank, 10) for rank in ranks) / n,
        "chunk_hit_at_5": sum(recall_at_k(r.chunk_rank, 5) for r in results) / n,
        "query_ms_p50": percentile(latencies_ms, 50),
        "query_ms_p95": percentile(latencies_ms, 95),
    }


def per_question_table(results: Sequence[QuestionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "qid": r.item.qid,
                "question": r.item.question,
                "subject": r.item.subject,
                "qtype": r.item.qtype,
                "gold_doc_id": r.item.doc_id,
                "gold_chunk_id": r.item.chunk_id,
                "alt_doc_ids": ";".join(r.item.alt_doc_ids),
                "top5_doc_ids": ";".join(h.chunk.doc_id for h in r.hits[:5]),
                "top1_source_uri": r.hits[0].chunk.source_uri if r.hits else "",
                "hit_rank": r.hit_rank,
                "chunk_hit_rank": r.chunk_rank,
                "query_ms": round(r.seconds * 1000, 3),
            }
        )
    return pd.DataFrame(rows)


def per_subject_table(per_question: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    frame = per_question.assign(
        subject=per_question["subject"].fillna("(none)"),
        hit=per_question["hit_rank"].map(
            lambda rank: recall_at_k(None if pd.isna(rank) else int(rank), k)
        ),
    )
    table = frame.groupby("subject").agg(n_questions=("qid", "size"), recall=("hit", "mean"))
    table = table.rename(columns={"recall": f"recall_at_{k}"}).reset_index()
    return table.sort_values([f"recall_at_{k}", "subject"], ascending=[False, True]).reset_index(
        drop=True
    )


def failure_table(
    results: Sequence[QuestionResult], gold_uri: dict[str, str], k: int = 5
) -> pd.DataFrame:
    """Questions whose gold document is not in the top-k, with what came back instead."""
    rows = [
        {
            "qid": r.item.qid,
            "question": r.item.question,
            "subject": r.item.subject,
            "gold_source_uri": gold_uri.get(r.item.doc_id, r.item.doc_id),
            "hit_rank": r.hit_rank,
            "top3_source_uris": " | ".join(h.chunk.source_uri for h in r.hits[:3]),
            "top1_snippet": r.hits[0].chunk.snippet if r.hits else "",
        }
        for r in results
        if recall_at_k(r.hit_rank, k) == 0.0
    ]
    columns = [
        "qid",
        "question",
        "subject",
        "gold_source_uri",
        "hit_rank",
        "top3_source_uris",
        "top1_snippet",
    ]
    return pd.DataFrame(rows, columns=columns)


def plot_per_subject(table: pd.DataFrame, path: Path, title: str) -> Path:
    fig = Figure(figsize=(9, max(4, 0.35 * len(table))))
    ax = fig.subplots()
    ordered = table.iloc[::-1]
    labels = [
        f"{s} (n={n})" for s, n in zip(ordered["subject"], ordered["n_questions"], strict=True)
    ]
    ax.barh(labels, ordered["recall_at_5"], color="#3b75af")
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall@5 (doc level)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


# ---------------------------------------------------------------- run bookkeeping


@dataclass(frozen=True, slots=True)
class ExperimentInputs:
    chunks: pd.DataFrame
    manifest: dict[str, Any]
    qa: list[QAItem]
    chunks_path: Path
    qa_path: Path

    @property
    def dataset_version(self) -> str:
        return str(self.manifest.get("dataset_version", "unknown"))

    @property
    def qa_digest(self) -> str:
        return hashlib.sha256(self.qa_path.read_bytes()).hexdigest()[:12]


def load_inputs(data_dir: Path, qa_path: Path) -> ExperimentInputs:
    chunks_path = data_dir / "chunks.parquet"
    chunks = pd.read_parquet(chunks_path)
    qa = load_qa(qa_path)
    validate_qa(qa, chunks)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return ExperimentInputs(chunks, manifest, qa, chunks_path, qa_path)


def run_key(variant: str, config: IndexConfig, inputs: ExperimentInputs) -> str:
    payload = json.dumps(
        {"variant": variant, "config": asdict(config), "qa": inputs.qa_digest}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def run_params(variant: str, config: IndexConfig, inputs: ExperimentInputs) -> dict[str, Any]:
    chunking = inputs.manifest.get("chunking", {})
    return {
        "variant": variant,
        "method": config.method,
        "k": DEFAULT_K,
        "eval_k": EVAL_K,
        "tokenizer": config.tokenizer if config.uses_bm25 else "-",
        "embed_model": config.embed_model or "-",
        "query_prefix": config.query_prefix or "-",
        "passage_prefix": config.passage_prefix or "-",
        "rrf_k": config.rrf_k if config.method == "hybrid" else "-",
        "fusion_depth": config.fusion_depth if config.method == "hybrid" else "-",
        "bm25_k1": config.bm25_k1 if config.uses_bm25 else "-",
        "bm25_b": config.bm25_b if config.uses_bm25 else "-",
        "chunk_target_chars": chunking.get("target_chars", "-"),
        "chunk_overlap_chars": chunking.get("overlap_chars", "-"),
        "chunk_min_chars": chunking.get("min_chars", "-"),
        "dataset_version": inputs.dataset_version,
        "n_chunks": len(inputs.chunks),
        "n_questions": len(inputs.qa),
        "qa_digest": inputs.qa_digest,
    }


def _requirements(dense: bool) -> list[str]:
    packages = ["mlflow-skinny", "pandas", "numpy", "rank-bm25", "scikit-learn"]
    if dense:
        packages.append("fastembed")
    pinned = []
    for name in packages:
        try:
            pinned.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:  # pragma: no cover - e.g. full mlflow only
            pinned.append(name)
    return pinned


def _signature(example: pd.DataFrame, output: pd.DataFrame) -> Any:
    from mlflow.models import infer_signature

    return infer_signature(example, output, params={"k": DEFAULT_K})


@dataclass(slots=True)
class VariantOutcome:
    variant: str
    run_id: str
    metrics: dict[str, float]
    reused: bool
    extra_artifacts: list[Path] = field(default_factory=list)


def find_finished_run(client: Any, experiment_id: str, key: str) -> Any | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_key = '{key}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def evaluate_variant(
    variant: str,
    config: IndexConfig,
    inputs: ExperimentInputs,
    work_dir: Path,
    *,
    embedder: Embedder | None,
    embeddings: np.ndarray | None,
    embed_seconds: float,
) -> tuple[Path, dict[str, float], list[QuestionResult]]:
    """Build the index into `work_dir/index`, query it like the served model and score it."""
    report = build_index(inputs.chunks, config, work_dir / "index", embeddings=embeddings)
    retriever = Retriever(load_index(report.out_dir), embedder if config.uses_dense else None)
    results = run_queries(retriever, inputs.qa)
    metrics = aggregate_metrics(results)
    metrics["index_build_seconds"] = report.seconds + (embed_seconds if config.uses_dense else 0.0)
    logger.info(
        "variant_evaluated", variant=variant, **{k: round(v, 4) for k, v in metrics.items()}
    )
    return report.out_dir, metrics, results


def write_eval_artifacts(
    variant: str,
    results: Sequence[QuestionResult],
    metrics: dict[str, float],
    inputs: ExperimentInputs,
    out: Path,
) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    per_question = per_question_table(results)
    per_question.to_csv(out / "per_question.csv", index=False, encoding="utf-8")
    gold_uri = dict(zip(inputs.chunks["doc_id"], inputs.chunks["source_uri"], strict=True))
    failure_table(results, gold_uri).to_csv(out / "failures.csv", index=False, encoding="utf-8")
    subjects = per_subject_table(per_question)
    subjects.to_csv(out / "per_subject_recall_at_5.csv", index=False, encoding="utf-8")
    plot_per_subject(
        subjects, out / "per_subject_recall_at_5.png", f"{variant}: recall@5 per subject"
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return subjects


def log_variant(
    variant: str,
    config: IndexConfig,
    inputs: ExperimentInputs,
    work_dir: Path,
    *,
    embedder: Embedder | None,
    embeddings: np.ndarray | None,
    embed_seconds: float,
    embedder_dir: Path | None,
    key: str,
) -> VariantOutcome:
    import mlflow
    from mlflow.data.pandas_dataset import from_pandas

    index_dir, metrics, results = evaluate_variant(
        variant,
        config,
        inputs,
        work_dir,
        embedder=embedder,
        embeddings=embeddings,
        embed_seconds=embed_seconds,
    )
    eval_dir = work_dir / "eval"
    subjects = write_eval_artifacts(variant, results, metrics, inputs, eval_dir)
    example = pd.DataFrame({TEXT_COLUMN: [r.item.question for r in results[:2]]})
    output = pd.DataFrame(
        {RESULTS_COLUMN: [[h.to_dict() for h in r.hits[:DEFAULT_K]] for r in results[:2]]}
    )
    del results  # release before logging: the model is loaded again for input validation

    artifacts = {INDEX_ARTIFACT: str(index_dir)}
    if config.uses_dense and embedder_dir is not None:
        artifacts[EMBEDDER_ARTIFACT] = str(embedder_dir)
    with mlflow.start_run(run_name=f"{variant}-{inputs.dataset_version}") as run:
        mlflow.set_tags(
            {
                "dataset_version": inputs.dataset_version,
                "git_commit": git_commit(),
                "variant": variant,
                "run_key": key,
                "qa_digest": inputs.qa_digest,
            }
        )
        mlflow.log_params(run_params(variant, config, inputs))
        mlflow.log_metrics(metrics)
        mlflow.log_input(
            from_pandas(
                inputs.chunks, source=inputs.chunks_path.resolve().as_uri(), name="study-chunks"
            ),
            context="index",
        )
        mlflow.log_input(
            from_pandas(
                qa_frame(inputs.qa), source=inputs.qa_path.resolve().as_uri(), name="study-qa"
            ),
            context="evaluation",
        )
        mlflow.log_artifacts(str(eval_dir), artifact_path="eval")
        mlflow.log_table(subjects, artifact_file="eval/per_subject_recall_at_5.json")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RetrieverModel(),
            artifacts=artifacts,
            signature=_signature(example, output),
            input_example=example,
            pip_requirements=_requirements(config.uses_dense),
        )
        mlflow.set_tag("model_uri", info.model_uri)
        run_id: str = run.info.run_id
    logger.info("variant_logged", variant=variant, run_id=run_id)
    return VariantOutcome(variant, run_id, metrics, reused=False)


# ---------------------------------------------------------------- dense embedder


def locate_model_dir(cache_dir: Path, model_name: str) -> Path:
    """Directory with the ONNX file of a fastembed model inside its download cache."""
    from fastembed import TextEmbedding

    description = next(
        m for m in TextEmbedding.list_supported_models() if m["model"].lower() == model_name.lower()
    )
    candidates = sorted(cache_dir.rglob(Path(description["model_file"]).name))
    if not candidates:
        raise FileNotFoundError(f"{description['model_file']} for {model_name} not in {cache_dir}")
    model_file = candidates[0]
    # `model_file` may be nested (e.g. `onnx/model.onnx`): bundle from the snapshot root.
    depth = len(Path(description["model_file"]).parts)
    return model_file.parents[depth - 1]


def fastembed_loader(cache_dir: Path) -> EmbedderLoader:
    """Download (once, at build time) and load a fastembed model; returns it with its files."""

    def load(model_name: str) -> tuple[Embedder, Path | None]:
        embedder = FastEmbedder(model_name, cache_dir=cache_dir)
        return embedder, locate_model_dir(cache_dir, model_name)

    return load


# ---------------------------------------------------------------- orchestration


def run_experiment(
    inputs: ExperimentInputs,
    variants: Sequence[str],
    work_root: Path,
    *,
    embedder_loader: EmbedderLoader | None = None,
    force: bool = False,
) -> list[VariantOutcome]:
    import mlflow

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    outcomes: list[VariantOutcome] = []
    dense_state: dict[str, tuple[Embedder, Path | None, np.ndarray, float]] = {}
    for variant in variants:
        config = replace(VARIANTS[variant], dataset_version=inputs.dataset_version)
        key = run_key(variant, config, inputs)
        existing = None if force else find_finished_run(client, experiment.experiment_id, key)
        if existing is not None:
            logger.info("variant_reused", variant=variant, run_id=existing.info.run_id)
            outcomes.append(
                VariantOutcome(
                    variant, existing.info.run_id, dict(existing.data.metrics), reused=True
                )
            )
            continue
        embedder = embedder_dir = embeddings = None
        embed_seconds = 0.0
        if config.uses_dense:
            model_name = str(config.embed_model)
            if model_name not in dense_state:
                if embedder_loader is None:
                    raise ValueError(f"variant {variant!r} needs an embedder loader")
                loaded, files = embedder_loader(model_name)
                started = time.perf_counter()
                matrix = embed_passages(
                    loaded, inputs.chunks["text"].astype(str).tolist(), config.passage_prefix
                )
                dense_state[model_name] = (loaded, files, matrix, time.perf_counter() - started)
            embedder, embedder_dir, embeddings, embed_seconds = dense_state[model_name]
        outcomes.append(
            log_variant(
                variant,
                config,
                inputs,
                work_root / variant,
                embedder=embedder,
                embeddings=embeddings,
                embed_seconds=embed_seconds,
                embedder_dir=embedder_dir,
                key=key,
            )
        )
    log_comparison(client, outcomes, work_root)
    return outcomes


def log_comparison(client: Any, outcomes: Sequence[VariantOutcome], work_root: Path) -> Path:
    """`metrics_comparison.json` (all variants side by side) attached to every new run."""
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / "metrics_comparison.json"
    payload = {o.variant: {"run_id": o.run_id, **o.metrics} for o in outcomes}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for outcome in outcomes:
        if not outcome.reused:
            client.log_artifact(outcome.run_id, str(path))
    return path


# ---------------------------------------------------------------- registry


@dataclass(frozen=True, slots=True)
class RegistryState:
    champion: str
    challenger: str
    champion_run_id: str
    challenger_run_id: str
    created_versions: list[str]


def _ensure_version(client: Any, run: Any, created: list[str]) -> str:
    from mlflow.exceptions import MlflowException

    versions = client.search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run.info.run_id}'"
    )
    if versions:
        return str(min(int(v.version) for v in versions))
    try:
        client.create_registered_model(
            MODEL_NAME,
            description="Retriever over study-material chunks; version == index version.",
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    model_uri = run.data.tags.get("model_uri") or f"runs:/{run.info.run_id}/model"
    tags = {
        "variant": run.data.tags.get("variant", ""),
        "dataset_version": run.data.tags.get("dataset_version", ""),
        PRIMARY_METRIC: str(run.data.metrics.get(PRIMARY_METRIC, "")),
    }
    # MLflow 3 logged models live at `models:/m-<id>`; such a source must come with its id.
    model_id = model_uri.removeprefix("models:/") if model_uri.startswith("models:/m-") else None
    mv = client.create_model_version(
        MODEL_NAME, source=model_uri, run_id=run.info.run_id, tags=tags, model_id=model_id
    )
    created.append(str(mv.version))
    logger.info("model_version_registered", version=mv.version, variant=tags["variant"])
    return str(mv.version)


def register_best(dataset_version: str, qa_digest: str) -> RegistryState:
    """Register baseline and best runs (by `search_runs` ordering) and point the aliases."""
    import mlflow

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"experiment {EXPERIMENT_NAME!r} does not exist")
    found = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.dataset_version = '{dataset_version}' and tags.qa_digest = '{qa_digest}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=[
            f"metrics.{PRIMARY_METRIC} DESC",
            "metrics.mrr_at_10 DESC",
            "attributes.start_time DESC",
        ],
    )
    runs = [r for r in found if "model_uri" in r.data.tags]
    baseline = next((r for r in runs if r.data.tags.get("variant") == BASELINE_VARIANT), None)
    if not runs or baseline is None:
        raise RuntimeError("no finished runs (with a baseline) to register")
    best = runs[0]
    created: list[str] = []
    challenger = _ensure_version(client, baseline, created)  # first: v1 on a fresh registry
    champion = _ensure_version(client, best, created)
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, challenger)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, champion)
    logger.info(
        "aliases_set", champion=champion, challenger=challenger, best=best.data.tags.get("variant")
    )
    return RegistryState(champion, challenger, best.info.run_id, baseline.info.run_id, created)


def main(argv: Sequence[str] | None = None) -> int:
    from edu_graphrag.logging_config import configure_logging

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("INFO", json_logs=False)
    parser = argparse.ArgumentParser(description="Evaluate and register study retrievers")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--qa", type=Path, default=Path("data/qa/qa.jsonl"))
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument(
        "--embed-cache",
        type=Path,
        default=Path(tempfile.gettempdir()) / "edu-graphrag-fastembed",
        help="download cache for the fastembed model (only used at build time)",
    )
    parser.add_argument("--work-dir", type=Path, default=None, help="keep built indexes here")
    parser.add_argument("--force", action="store_true", help="retrain even if a run exists")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args(argv)

    inputs = load_inputs(args.data, args.qa)
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="edu-graphrag-rag-"))
    try:
        outcomes = run_experiment(
            inputs,
            args.variants,
            work_root,
            embedder_loader=fastembed_loader(args.embed_cache),
            force=args.force,
        )
        for o in outcomes:
            shown = {k: round(v, 3) for k, v in sorted(o.metrics.items())}
            print(f"{o.variant:<20} run {o.run_id} {'(reused) ' if o.reused else ''}{shown}")
        if not args.no_register:
            state = register_best(inputs.dataset_version, inputs.qa_digest)
            print(
                f"{MODEL_NAME}: champion=v{state.champion} challenger=v{state.challenger} "
                f"(new versions: {state.created_versions or 'none'})"
            )
    finally:
        if args.work_dir is None:
            shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
