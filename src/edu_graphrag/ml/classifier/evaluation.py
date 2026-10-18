"""Metrics and report tables for the subject classifier (pure functions + one plot)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from edu_graphrag.ml.classifier.data import GROUP_COLUMN, TARGET_COLUMN, TEXT_COLUMN

SNIPPET_CHARS = 200


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], prefix: str
) -> dict[str, float]:
    """Accuracy, macro/weighted F1 and balanced accuracy with a `train_`/`test_` prefix."""
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def document_votes(
    doc_ids: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str]
) -> pd.DataFrame:
    """Majority vote of chunk predictions per document (ties -> alphabetically first label)."""
    frame = pd.DataFrame({"doc_id": doc_ids, "true": y_true, "pred": y_pred})
    rows = []
    for doc_id, group in frame.groupby("doc_id", sort=True):
        counts = group["pred"].value_counts()
        top = sorted(counts[counts == counts.max()].index)[0]
        rows.append(
            {"doc_id": doc_id, "true": group["true"].iloc[0], "pred": top, "n_chunks": len(group)}
        )
    return pd.DataFrame(rows, columns=["doc_id", "true", "pred", "n_chunks"])


def document_accuracy(votes: pd.DataFrame) -> float:
    return float((votes["true"] == votes["pred"]).mean()) if len(votes) else 0.0


def report_table(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support plus the macro and weighted averages."""
    report: dict[str, Any] = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    report.pop("accuracy", None)
    table = pd.DataFrame(report).T
    table.index.name = "class"
    table["support"] = table["support"].astype(int)
    return table.reset_index()


def misclassified(
    test: pd.DataFrame, y_pred: Sequence[str], confidence: Sequence[float]
) -> pd.DataFrame:
    """Wrong test predictions, most confident mistakes first."""
    frame = pd.DataFrame(
        {
            "chunk_id": test["chunk_id"].to_numpy(),
            GROUP_COLUMN: test[GROUP_COLUMN].to_numpy(),
            "true": test[TARGET_COLUMN].to_numpy(),
            "pred": np.asarray(y_pred),
            "confidence": np.round(np.asarray(confidence, dtype=float), 4),
            "snippet": test[TEXT_COLUMN]
            .str.slice(0, SNIPPET_CHARS)
            .str.replace(r"\s+", " ", regex=True)
            .to_numpy(),
        }
    )
    wrong = frame[frame["true"] != frame["pred"]]
    return wrong.sort_values("confidence", ascending=False).reset_index(drop=True)


def top_features(
    coef: Any, feature_names: Sequence[str], classes: Sequence[str], k: int = 15
) -> pd.DataFrame:
    """Highest-weight features per class of a linear model (one row per class)."""
    weights = np.asarray(coef.toarray() if hasattr(coef, "toarray") else coef)
    names = np.asarray(feature_names)
    if weights.shape[0] == 1 and len(classes) == 2:
        # Binary models keep one weight row (for the positive class).
        weights = np.vstack([-weights[0], weights[0]])
    rows = []
    for label, row in zip(classes, weights, strict=True):
        best = np.argsort(-row, kind="stable")[:k]
        rows.append({"class": str(label), "top_features": " | ".join(str(names[i]) for i in best)})
    return pd.DataFrame(rows, columns=["class", "top_features"])


def plot_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], path: Path, title: str
) -> Path:
    """Row-normalised confusion matrix: each row shows where a true class goes (recall)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels), normalize="true")
    size = max(6.0, 0.55 * len(labels) + 2)
    fig = Figure(figsize=(size + 1, size))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > 0.5 else "black",
                )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path
