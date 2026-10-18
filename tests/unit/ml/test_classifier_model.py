import json
import pickle
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
import pytest

from edu_graphrag.ml.classifier import evaluation, variants
from edu_graphrag.ml.classifier.model import (
    OUTPUT_COLUMNS,
    PIPELINE_ARTIFACT,
    SubjectClassifierModel,
    format_predictions,
    input_texts,
)

TEXTS = {
    "algebra": "matrix vector eigenvalue determinant linear space basis",
    "topology": "open set neighbourhood continuous map compact hausdorff",
    "stats": "probability distribution variance estimator sample hypothesis",
}


def _fitted(variant: variants.Variant) -> Any:
    texts, labels = [], []
    for label, text in TEXTS.items():
        words = text.split()
        for i in range(6):
            texts.append(" ".join(words[i:] + words[:i]))
            labels.append(label)
    pipeline = variants.build_pipeline(variant, seed=0)
    return pipeline.fit(texts, labels)


def test_format_predictions() -> None:
    proba = np.array([[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]])
    out = format_predictions(["a", "b", "c"], proba)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["label"].tolist() == ["b", "a"]
    assert out["confidence"].tolist() == [0.7, 0.5]
    assert out["top3"][0] == [
        {"label": "b", "p": 0.7},
        {"label": "c", "p": 0.2},
        {"label": "a", "p": 0.1},
    ]
    assert out["top3"][1][1] == {"label": "b", "p": 0.25}  # stable order on ties
    json.loads(out.to_json(orient="records"))


def test_input_texts() -> None:
    assert input_texts(pd.DataFrame({"text": ["a", None]})) == ["a", ""]
    assert input_texts(pd.DataFrame({"other": ["x"]})) == ["x"]
    assert input_texts(pd.Series(["s"])) == ["s"]
    assert input_texts(["l", 1]) == ["l", "1"]


def test_model_predict_and_pickle_without_pipeline(tmp_path: Path) -> None:
    pipeline = _fitted(variants.BASELINE)
    model = SubjectClassifierModel(pipeline)
    out = model.predict(
        None, pd.DataFrame({"text": ["eigenvalue of a matrix", "compact open set"]})
    )
    assert out["label"].tolist() == ["algebra", "topology"]
    assert all(0 < c <= 1 for c in out["confidence"])
    assert len(out["top3"][0]) == 3
    assert model.predict(None, pd.DataFrame({"text": []})).empty

    restored = pickle.loads(pickle.dumps(model))  # noqa: S301 - our own object
    with pytest.raises(RuntimeError, match="not loaded"):
        restored.predict(None, pd.DataFrame({"text": ["x"]}))

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)

    class Context:
        artifacts: ClassVar[dict[str, str]] = {PIPELINE_ARTIFACT: str(path)}

    restored.load_context(Context())
    assert restored.predict(None, pd.DataFrame({"text": ["matrix basis"]}))["label"][0] == "algebra"


def test_variants_params_and_coefficients() -> None:
    assert set(variants.VARIANTS) >= {"baseline", "char_ngrams", "balanced", "svm"}
    assert variants.BASELINE.params()["class_weight"] == "none"
    assert variants.BALANCED.params()["analyzer"] == "char_wb"

    svm = _fitted(variants.SVM)
    coef = variants.linear_coefficients(svm)
    assert coef.shape == (3, len(svm.named_steps["tfidf"].vocabulary_))
    assert svm.predict_proba(["variance of estimator"]).shape == (1, 3)
    logreg = _fitted(variants.CHAR_NGRAMS)
    assert variants.linear_coefficients(logreg).shape[0] == 3

    class NoCoef:
        named_steps: dict[str, object] = {"clf": object()}  # noqa: RUF012

    assert variants.linear_coefficients(NoCoef()) is None


def test_metrics_and_votes() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    metrics = evaluation.classification_metrics(y_true, y_pred, "test")
    assert metrics["test_accuracy"] == 0.75
    assert metrics["test_balanced_accuracy"] == 0.75
    assert set(metrics) == {
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_balanced_accuracy",
    }

    votes = evaluation.document_votes(["d1", "d1", "d2", "d2"], y_true, ["b", "a", "b", "b"])
    assert votes["pred"].tolist() == ["a", "b"]  # tie in d1 -> alphabetically first
    assert evaluation.document_accuracy(votes) == 1.0
    assert evaluation.document_accuracy(votes.iloc[:0]) == 0.0


class _Sparse:
    """Stands in for a scipy sparse matrix (only `toarray` is used)."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def toarray(self) -> np.ndarray:
        return np.array(self._rows)


def test_report_misclassified_top_features_and_plot(tmp_path: Path) -> None:
    report = evaluation.report_table(["a", "b"], ["a", "a"], ["a", "b"])
    assert report["class"].tolist() == ["a", "b", "macro avg", "weighted avg"]
    assert report["support"].tolist() == [1, 1, 2, 2]

    test = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2", "c3"],
            "doc_id": ["d1", "d1", "d2"],
            "text": ["x  y\nz", "ok", "w" * 500],
            "subject": ["a", "a", "b"],
        }
    )
    wrong = evaluation.misclassified(test, ["b", "a", "a"], [0.4, 0.9, 0.8])
    assert wrong["chunk_id"].tolist() == ["c3", "c1"]
    assert wrong["snippet"].tolist() == ["w" * evaluation.SNIPPET_CHARS, "x y z"]

    features = evaluation.top_features(
        _Sparse([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]]), ["f0", "f1", "f2"], ["a", "b"], k=2
    )
    assert features["top_features"].tolist() == ["f1 | f0", "f0 | f2"]
    binary = evaluation.top_features(np.array([[1.0, -2.0]]), ["p", "n"], ["neg", "pos"], k=1)
    assert binary["top_features"].tolist() == ["n", "p"]

    path = evaluation.plot_confusion_matrix(
        ["a", "a", "b"], ["a", "b", "b"], ["a", "b"], tmp_path / "cm.png", "title"
    )
    assert path.stat().st_size > 0
