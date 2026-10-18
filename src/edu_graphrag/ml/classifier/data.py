"""Dataset preparation for the subject classifier: filtering, deduplication, grouped split.

All functions are pure: they take DataFrames and return new ones.

Why the split is grouped by document (leakage): chunks of one document share its vocabulary,
notation, author style and even literal text (chunks overlap by `overlap_chars`). If chunks of
the same document landed both in train and in test, the model could recognise *the document*
instead of *the subject*, and test metrics would be optimistic. So every document goes either
to train or to test as a whole. Byte-identical files stored under several paths are the same
document too, therefore they are deduplicated by `content_hash` *before* the split.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TEXT_COLUMN = "text"
TARGET_COLUMN = "subject"
GROUP_COLUMN = "doc_id"
# Columns of chunks.parquet the classifier keeps (and logs as the MLflow dataset).
CHUNK_COLUMNS = ["chunk_id", GROUP_COLUMN, "chunk_index", TEXT_COLUMN, TARGET_COLUMN]

DEFAULT_MIN_DOCS = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PreparedData:
    """Chunks ready for splitting plus a record of what was removed and why."""

    chunks: pd.DataFrame
    n_docs_not_ok: int
    n_duplicate_docs_removed: int
    conflicting_hashes: list[str] = field(default_factory=list)
    dropped_subjects: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def keep_ok_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """Documents that converted successfully and have a subject label."""
    return docs[(docs["status"] == "ok") & docs[TARGET_COLUMN].notna()].copy()


def deduplicate_documents(docs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One document per `content_hash`; returns the kept documents and conflicting hashes.

    A sorted copy is preferred over an unsorted one, then the shortest path wins, so the
    choice is deterministic. Groups whose copies carry *different* subjects are ambiguous
    labels (the same file filed under two courses) and are dropped entirely.
    """
    subjects_per_hash = docs.groupby("content_hash")[TARGET_COLUMN].nunique()
    conflicting = sorted(subjects_per_hash[subjects_per_hash > 1].index)
    clean = docs[~docs["content_hash"].isin(conflicting)]
    ordered = clean.assign(_uri_len=clean["source_uri"].str.len()).sort_values(
        ["content_hash", "is_sorted", "_uri_len", "source_uri"],
        ascending=[True, False, True, True],
    )
    kept = ordered.drop_duplicates("content_hash", keep="first").drop(columns="_uri_len")
    return kept.sort_values("doc_id").reset_index(drop=True), conflicting


def rare_subjects(docs: pd.DataFrame, min_docs: int) -> dict[str, int]:
    """Subjects with fewer than `min_docs` distinct documents -> their document count.

    A grouped split needs at least one document in test and a couple in train per class.
    """
    counts = docs.groupby(TARGET_COLUMN)[GROUP_COLUMN].nunique()
    return {str(s): int(n) for s, n in counts[counts < min_docs].sort_index().items()}


def prepare_chunks(
    docs: pd.DataFrame, chunks: pd.DataFrame, *, min_docs: int = DEFAULT_MIN_DOCS
) -> PreparedData:
    """ok documents -> deduplicated -> rare subjects dropped -> their chunks."""
    ok = keep_ok_documents(docs)
    unique, conflicting = deduplicate_documents(ok)
    n_conflicting_docs = int(ok["content_hash"].isin(conflicting).sum())
    dropped = rare_subjects(unique, min_docs)
    kept_docs = unique[~unique[TARGET_COLUMN].isin(dropped)]
    selected = chunks[chunks[GROUP_COLUMN].isin(kept_docs[GROUP_COLUMN])]
    selected = selected[selected[TEXT_COLUMN].fillna("").str.strip() != ""]
    return PreparedData(
        chunks=selected[CHUNK_COLUMNS].sort_values("chunk_id").reset_index(drop=True),
        n_docs_not_ok=len(docs) - len(ok),
        n_duplicate_docs_removed=len(ok) - len(unique) - n_conflicting_docs,
        conflicting_hashes=conflicting,
        dropped_subjects=dropped,
    )


def split_by_document(
    chunks: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Grouped and stratified split: ~`test_size` of each subject's *documents* go to test.

    This is what `StratifiedGroupKFold` approximates, done per subject so that every class
    is guaranteed at least one test and one train document (with 3-5 documents per small
    class a fold-based splitter may leave a class out of test). The fixed seed makes the
    split identical for every run, so runs are comparable.
    """
    rng = np.random.default_rng(seed)
    docs = chunks[[GROUP_COLUMN, TARGET_COLUMN]].drop_duplicates(GROUP_COLUMN)
    test_docs: list[str] = []
    for _, group in docs.sort_values(GROUP_COLUMN).groupby(TARGET_COLUMN, sort=True):
        doc_ids = group[GROUP_COLUMN].to_numpy()
        if len(doc_ids) < 2:
            continue  # a single document cannot be split; it stays in train
        n_test = min(len(doc_ids) - 1, max(1, round(test_size * len(doc_ids))))
        test_docs.extend(str(d) for d in rng.permutation(doc_ids)[:n_test])
    is_test = chunks[GROUP_COLUMN].isin(test_docs)
    return Split(
        train=chunks[~is_test].reset_index(drop=True),
        test=chunks[is_test].reset_index(drop=True),
    )


def split_stats(split: Split) -> dict[str, int]:
    """Sizes of both parts in documents, chunks and classes."""
    stats: dict[str, int] = {}
    for part, frame in (("train", split.train), ("test", split.test)):
        stats[f"n_{part}_docs"] = int(frame[GROUP_COLUMN].nunique())
        stats[f"n_{part}_chunks"] = len(frame)
        stats[f"n_{part}_classes"] = int(frame[TARGET_COLUMN].nunique())
    return stats


def class_distribution(split: Split) -> pd.DataFrame:
    """Per-subject documents and chunks in train and test."""
    parts = {}
    for part, frame in (("train", split.train), ("test", split.test)):
        grouped = frame.groupby(TARGET_COLUMN)
        parts[f"{part}_docs"] = grouped[GROUP_COLUMN].nunique()
        parts[f"{part}_chunks"] = grouped.size()
    table = pd.DataFrame(parts).fillna(0).astype(int)
    table.index.name = TARGET_COLUMN
    return table.sort_values("train_chunks", ascending=False)
