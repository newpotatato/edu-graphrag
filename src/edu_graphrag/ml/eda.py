"""Exploratory analysis of the processed dataset (`documents.parquet` + `chunks.parquet`).

Pure stat functions take DataFrames and return DataFrames; plotting functions write PNGs;
`run_eda` ties them together and `log_to_mlflow` records the run.

Usage: `uv run python -m edu_graphrag.ml.eda --data data/processed [--log-to-mlflow]`
"""

import argparse
import io
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import structlog
from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

EXPERIMENT_NAME = "edu-graphrag-eda"
NO_SUBJECT = "(none)"
_COPY_SUFFIX_RE = re.compile(r"\s*\((\d+)\)$")
# Words of 4+ letters in any script (Latin, Cyrillic, ...).
_TOKEN_RE = re.compile(r"[^\W\d_]{4,}")


@dataclass
class ProcessedDataset:
    documents: pd.DataFrame
    chunks: pd.DataFrame
    manifest: dict[str, Any]
    chunks_path: Path

    @property
    def version(self) -> str:
        return str(self.manifest.get("dataset_version", "unknown"))


@dataclass
class EdaReport:
    dataset_version: str
    out_dir: Path
    params: dict[str, Any]
    metrics: dict[str, float]
    artifacts: list[Path] = field(default_factory=list)


def load_dataset(data_dir: Path) -> ProcessedDataset:
    chunks_path = data_dir / "chunks.parquet"
    return ProcessedDataset(
        documents=pd.read_parquet(data_dir / "documents.parquet"),
        chunks=pd.read_parquet(chunks_path),
        manifest=json.loads((data_dir / "manifest.json").read_text(encoding="utf-8")),
        chunks_path=chunks_path,
    )


# ---------------------------------------------------------------- pure statistics


def _subject(df: pd.DataFrame) -> pd.Series:
    return df["subject"].fillna(NO_SUBJECT).astype(str)


def docs_per_subject(docs: pd.DataFrame) -> pd.DataFrame:
    """Documents per subject split into sorted/unsorted, largest first."""
    table = pd.crosstab(_subject(docs), docs["is_sorted"].map({True: "sorted", False: "unsorted"}))
    table = table.reindex(columns=["sorted", "unsorted"], fill_value=0)
    table["total"] = table["sorted"] + table["unsorted"]
    table.index.name = "subject"
    return table.sort_values(["total", "sorted"], ascending=False)


def class_stats(docs: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """Per-subject class statistics for a chunk-level subject classifier."""
    doc_groups = docs.assign(subject=_subject(docs)).groupby("subject")
    stats = pd.DataFrame(
        {
            "n_docs": doc_groups.size(),
            "n_docs_ok": doc_groups["status"].apply(lambda s: int((s == "ok").sum())),
            "n_docs_sorted": doc_groups["is_sorted"].sum().astype(int),
            "median_doc_chars": doc_groups["n_chars"].median(),
        }
    )
    chunk_groups = chunks.assign(subject=_subject(chunks)).groupby("subject")
    stats["n_chunks"] = chunk_groups.size().reindex(stats.index, fill_value=0)
    stats["mean_chunk_chars"] = chunk_groups["n_chars"].mean().reindex(stats.index)
    total = stats["n_chunks"].sum()
    stats["chunk_share"] = stats["n_chunks"] / total if total else 0.0
    stats.index.name = "subject"
    return stats.sort_values("n_chunks", ascending=False)


def imbalance_ratio(stats: pd.DataFrame) -> float:
    """Largest / smallest chunk count among subjects that have chunks."""
    counts = stats.loc[stats["n_chunks"] > 0, "n_chunks"]
    return float(counts.max() / counts.min()) if len(counts) else 0.0


def content_duplicates(docs: pd.DataFrame) -> pd.DataFrame:
    """Groups of documents with identical bytes stored under different paths."""
    groups = docs.groupby("content_hash")["source_uri"].agg(list)
    groups = groups[groups.map(len) > 1]
    return pd.DataFrame(
        {
            "content_hash": groups.index,
            "n_copies": groups.map(len).to_numpy(),
            "source_uris": groups.map(lambda uris: "; ".join(sorted(uris))).to_numpy(),
        }
    ).reset_index(drop=True)


def strip_copy_suffix(source_uri: str) -> str:
    """`dir/lecture (2).pdf` -> `dir/lecture.pdf` (browser/OS copy suffixes)."""
    path = PurePosixPath(source_uri)
    return str(path.with_name(_COPY_SUFFIX_RE.sub("", path.stem) + path.suffix))


def filename_duplicates(docs: pd.DataFrame) -> pd.DataFrame:
    """Files whose name carries a copy suffix like `(2)`; `has_original` if the base exists."""
    uris = set(docs["source_uri"])
    rows = [
        {"source_uri": uri, "base_uri": base, "has_original": base in uris}
        for uri in sorted(uris)
        if (base := strip_copy_suffix(uri)) != uri
    ]
    return pd.DataFrame(rows, columns=["source_uri", "base_uri", "has_original"])


def problem_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """Failed and empty documents (e.g. scanned PDFs without a text layer)."""
    problems = docs.loc[docs["status"] != "ok", ["source_uri", "source_type", "status", "error"]]
    return problems.sort_values(["status", "source_uri"]).reset_index(drop=True)


def grade_semester_counts(docs: pd.DataFrame) -> pd.DataFrame:
    """Documents per (grade, semester); semester is `-` for unsorted data."""
    grade = docs["grade"].map(lambda g: "-" if pd.isna(g) else f"grade {int(g)}")
    sem = docs["semester"].map(lambda s: "-" if pd.isna(s) else f"sem {int(s)}")
    counts = pd.DataFrame({"grade": grade, "semester": sem}).value_counts().rename("n_docs")
    return counts.reset_index().sort_values(["grade", "semester"]).reset_index(drop=True)


def length_by_type(docs: pd.DataFrame) -> pd.DataFrame:
    """Distribution of converted document length (chars) per source type."""
    described = docs.groupby("source_type")["n_chars"].describe()
    return described[["count", "mean", "50%", "max"]].rename(columns={"50%": "median"})


def top_terms(chunks: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Most distinctive tokens per subject: frequency weighted by lift over the corpus."""
    per_subject: dict[str, Counter[str]] = {}
    for subject, text in zip(_subject(chunks), chunks["text"], strict=True):
        per_subject.setdefault(subject, Counter()).update(
            t.lower() for t in _TOKEN_RE.findall(text)
        )
    overall: Counter[str] = Counter()
    for counter in per_subject.values():
        overall.update(counter)
    total = sum(overall.values()) or 1
    rows = []
    for subject, counter in sorted(per_subject.items()):
        size = sum(counter.values()) or 1
        # count * log(lift): frequent in the subject and rare elsewhere.
        scored = sorted(
            (-count * math.log((count / size) / (overall[term] / total)), term)
            for term, count in counter.items()
            if count >= 3
        )
        terms = [term for _, term in scored[:k]]
        rows.append({"subject": subject, "top_terms": ", ".join(terms)})
    return pd.DataFrame(rows, columns=["subject", "top_terms"])


def build_insights(
    dataset: ProcessedDataset,
    stats: pd.DataFrame,
    content_dups: pd.DataFrame,
    name_dups: pd.DataFrame,
    problems: pd.DataFrame,
) -> list[str]:
    """Human-readable findings computed from the data."""
    docs, chunks = dataset.documents, dataset.chunks
    n_docs, n_chunks = len(docs), len(chunks)
    insights = [f"Dataset `{dataset.version}`: {n_docs} documents, {n_chunks} chunks."]
    status = docs["status"].value_counts()
    insights.append(
        "Status: " + ", ".join(f"{name} = {int(count)}" for name, count in status.items()) + "."
    )
    types = docs["source_type"].value_counts()
    insights.append(
        "Source types: "
        + ", ".join(f"{name} {count / n_docs:.1%}" for name, count in types.items())
        + "."
    )
    n_sorted = int(docs["is_sorted"].sum())
    insights.append(
        f"{n_sorted} documents ({n_sorted / n_docs:.0%}) are sorted into grade/semester/subject; "
        f"{n_docs - n_sorted} are unsorted (new data without a semester level)."
    )
    if len(stats) and stats["n_chunks"].sum():
        top = stats.index[0]
        with_chunks = stats[stats["n_chunks"] > 0]
        smallest = with_chunks.index[-1]
        insights.append(
            f"Class imbalance: `{top}` holds {stats.loc[top, 'chunk_share']:.0%} of chunks, "
            f"`{smallest}` only {int(with_chunks.loc[smallest, 'n_chunks'])}; "
            f"max/min ratio {imbalance_ratio(stats):.1f}. "
            "Use macro-F1 and class weights (or stratified sampling) for the subject classifier."
        )
        rare = with_chunks[with_chunks["n_chunks"] < 20]
        if len(rare):
            insights.append(
                f"{len(rare)} subjects have fewer than 20 chunks ("
                + ", ".join(f"`{s}`" for s in rare.index)
                + "): merge, drop, or evaluate them separately."
            )
    if n_chunks:
        lengths = chunks["n_chars"]
        insights.append(
            f"Chunk length: median {lengths.median():.0f}, p5 {lengths.quantile(0.05):.0f}, "
            f"p95 {lengths.quantile(0.95):.0f} chars."
        )
    if len(content_dups):
        extra = int(content_dups["n_copies"].sum() - len(content_dups))
        insights.append(
            f"{len(content_dups)} groups of byte-identical files ({extra} redundant copies): "
            "deduplicate by content_hash before splitting train/test to avoid leakage."
        )
    if len(name_dups):
        insights.append(
            f"{len(name_dups)} files carry a copy suffix like `(2)`; "
            f"{int(name_dups['has_original'].sum())} of them sit next to the original."
        )
    if len(problems):
        by_status = problems["status"].value_counts()
        insights.append(
            "Problem documents: "
            + ", ".join(f"{name} = {int(count)}" for name, count in by_status.items())
            + " (empty PDFs are most likely scans without a text layer: OCR candidates)."
        )
    return insights


# ---------------------------------------------------------------- plots


def _save(fig: Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def plot_docs_per_subject(table: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(10, max(4, 0.35 * len(table))))
    ax = fig.subplots()
    ordered = table.iloc[::-1]
    ax.barh(ordered.index, ordered["sorted"], color="#3b75af", label="sorted")
    ax.barh(
        ordered.index,
        ordered["unsorted"],
        left=ordered["sorted"],
        color="#ef8636",
        label="unsorted",
    )
    ax.set_title("Documents per subject")
    ax.set_xlabel("documents")
    ax.legend()
    return _save(fig, path)


def plot_chunks_per_subject(stats: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(10, max(4, 0.35 * len(stats))))
    ax = fig.subplots()
    ordered = stats.iloc[::-1]
    ax.barh(ordered.index, ordered["n_chunks"], color="#3b75af")
    ax.set_title(f"Chunks per subject (class balance, max/min = {imbalance_ratio(stats):.1f})")
    ax.set_xlabel("chunks")
    return _save(fig, path)


def plot_doc_length_by_type(docs: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    docs = docs[docs["n_chars"] > 0]  # failed/empty documents have no length to show
    types = sorted(docs["source_type"].unique())
    data = [docs.loc[docs["source_type"] == t, "n_chars"].to_numpy() for t in types]
    ax.boxplot(data, tick_labels=types)
    ax.set_yscale("log")
    ax.set_title("Converted document length by source type")
    ax.set_xlabel("source type")
    ax.set_ylabel("characters (log scale)")
    return _save(fig, path)


def plot_chunk_length_hist(chunks: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.hist(chunks["n_chars"], bins=40, color="#3b75af")
    ax.set_title("Chunk length distribution")
    ax.set_xlabel("characters per chunk")
    ax.set_ylabel("chunks")
    return _save(fig, path)


def plot_source_type_mix(docs: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    counts = docs["source_type"].value_counts()
    ax.bar(counts.index, counts.to_numpy(), color="#3b75af")
    ax.set_title("Documents per source type")
    ax.set_xlabel("source type")
    ax.set_ylabel("documents")
    return _save(fig, path)


def plot_grade_semester(counts: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    labels = [f"{g} / {s}" for g, s in zip(counts["grade"], counts["semester"], strict=True)]
    ax.bar(labels, counts["n_docs"], color="#3b75af")
    ax.set_title("Documents per grade / semester ('-' = unsorted)")
    ax.set_ylabel("documents")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


def plot_duplicates(content_dups: pd.DataFrame, name_dups: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    labels = [
        "identical content\n(redundant copies)",
        "copy suffix\n(with original)",
        "copy suffix\n(no original)",
    ]
    redundant = int(content_dups["n_copies"].sum() - len(content_dups)) if len(content_dups) else 0
    with_original = int(name_dups["has_original"].sum()) if len(name_dups) else 0
    values = [redundant, with_original, len(name_dups) - with_original]
    ax.bar(labels, values, color=["#c44e52", "#ef8636", "#8c8c8c"])
    ax.set_title("Duplicate documents")
    ax.set_ylabel("documents")
    return _save(fig, path)


# ---------------------------------------------------------------- orchestration


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body])


def run_eda(data_dir: Path, out_dir: Path) -> tuple[EdaReport, ProcessedDataset]:
    """Compute statistics, write plots/tables/summary into `out_dir`."""
    dataset = load_dataset(data_dir)
    docs, chunks = dataset.documents, dataset.chunks
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = docs_per_subject(docs)
    stats = class_stats(docs, chunks)
    content_dups = content_duplicates(docs)
    name_dups = filename_duplicates(docs)
    problems = problem_documents(docs)
    grade_sem = grade_semester_counts(docs)
    lengths = length_by_type(docs)
    terms = top_terms(chunks)

    artifacts = [
        plot_docs_per_subject(subjects, out_dir / "docs_per_subject.png"),
        plot_chunks_per_subject(stats, out_dir / "chunks_per_subject.png"),
        plot_doc_length_by_type(docs, out_dir / "doc_length_by_type.png"),
        plot_chunk_length_hist(chunks, out_dir / "chunk_length_hist.png"),
        plot_source_type_mix(docs, out_dir / "source_type_mix.png"),
        plot_grade_semester(grade_sem, out_dir / "docs_per_grade_semester.png"),
        plot_duplicates(content_dups, name_dups, out_dir / "duplicates.png"),
    ]
    tables = {
        "class_stats.csv": stats.reset_index(),
        "docs_per_subject.csv": subjects.reset_index(),
        "content_duplicates.csv": content_dups,
        "filename_duplicates.csv": name_dups,
        "problem_documents.csv": problems,
        "length_by_type.csv": lengths.reset_index(),
        "top_terms.csv": terms,
    }
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False, encoding="utf-8")
        artifacts.append(out_dir / name)

    insights = build_insights(dataset, stats, content_dups, name_dups, problems)
    summary = "\n\n".join(
        [
            f"# EDA summary: dataset {dataset.version}",
            "## Insights\n\n" + "\n".join(f"- {line}" for line in insights),
            "## Class statistics\n\n" + _markdown_table(stats.round(3).reset_index()),
            "## Document length by source type\n\n"
            + _markdown_table(lengths.round(0).reset_index()),
            "## Failed / empty documents\n\n" + _markdown_table(problems),
            "## Top terms per subject\n\n" + _markdown_table(terms),
        ]
    )
    summary_path = out_dir / "eda_summary.md"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    artifacts.append(summary_path)

    n_ok = int((docs["status"] == "ok").sum())
    report = EdaReport(
        dataset_version=dataset.version,
        out_dir=out_dir,
        params={
            "dataset_version": dataset.version,
            "n_docs": len(docs),
            "n_chunks": len(chunks),
            "n_subjects": int((stats["n_chunks"] > 0).sum()),
        },
        metrics={
            "imbalance_ratio": imbalance_ratio(stats),
            "share_docs_ok": n_ok / len(docs) if len(docs) else 0.0,
            "n_failed_docs": float((docs["status"] == "failed").sum()),
            "n_empty_docs": float((docs["status"] == "empty").sum()),
            "n_content_duplicate_groups": float(len(content_dups)),
            "n_copy_suffix_files": float(len(name_dups)),
            "median_chunk_chars": float(chunks["n_chars"].median()) if len(chunks) else 0.0,
        },
        artifacts=artifacts,
    )
    logger.info("eda_done", out_dir=str(out_dir), artifacts=len(artifacts), **report.params)
    return report, dataset


def log_to_mlflow(report: EdaReport, dataset: ProcessedDataset) -> str:
    """Log params, metrics, the chunks dataset and all artifacts; returns the run id.

    The tracking URI comes from `MLFLOW_TRACKING_URI`.
    """
    import mlflow
    from mlflow.data.pandas_dataset import from_pandas

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"eda-{report.dataset_version}") as run:
        mlflow.log_params(report.params)
        mlflow.log_metrics(report.metrics)
        chunks_dataset = from_pandas(
            dataset.chunks,
            source=dataset.chunks_path.resolve().as_uri(),
            name="study-chunks",
            targets="subject",
        )
        mlflow.log_input(chunks_dataset, context="eda")
        mlflow.log_artifacts(str(report.out_dir), artifact_path="eda")
        run_id: str = run.info.run_id
    logger.info("eda_logged_to_mlflow", run_id=run_id)
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    from edu_graphrag.logging_config import configure_logging

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("INFO", json_logs=False)
    parser = argparse.ArgumentParser(description="EDA of the processed study-materials dataset")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/eda"))
    parser.add_argument("--log-to-mlflow", action="store_true")
    args = parser.parse_args(argv)

    report, dataset = run_eda(args.data, args.out)
    print(
        f"EDA for dataset {report.dataset_version}: {len(report.artifacts)} artifacts in {args.out}"
    )
    if args.log_to_mlflow:
        run_id = log_to_mlflow(report, dataset)
        print(f"MLflow run id: {run_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
