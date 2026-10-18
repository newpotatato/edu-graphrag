"""MLflow pyfunc wrapper around the fitted sklearn pipeline, as served by the API.

The fitted pipeline is stored as a separate joblib artifact and loaded in `load_context`, so
the pickled wrapper itself holds no sklearn state. Loading needs only what the API image has
(mlflow-skinny, scikit-learn with its joblib, pandas, numpy and this package): this module must
not import anything from the `ml` dependency group.

Input: a DataFrame with a `text` column. Output: one row per text with `label` (str),
`confidence` (max class probability, float) and `top3` (list of `{"label", "p"}`), all
JSON-serialisable for `/api/v1/process`.
"""

from typing import Any

import joblib
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModel, PythonModelContext

PIPELINE_ARTIFACT = "pipeline"
INPUT_COLUMN = "text"
OUTPUT_COLUMNS = ["label", "confidence", "top3"]
TOP_K = 3


def format_predictions(classes: Any, proba: np.ndarray, top_k: int = TOP_K) -> pd.DataFrame:
    """Class probabilities (n_samples x n_classes) -> `label`, `confidence`, `top3` rows."""
    labels = np.asarray(classes).astype(str)
    order = np.argsort(-proba, axis=1, kind="stable")[:, :top_k]
    rows = []
    for row_proba, row_order in zip(proba, order, strict=True):
        top = [{"label": str(labels[i]), "p": round(float(row_proba[i]), 6)} for i in row_order]
        rows.append(
            {"label": top[0]["label"], "confidence": float(row_proba[row_order[0]]), "top3": top}
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def input_texts(model_input: Any) -> list[str]:
    """Texts from a DataFrame `text` column (or its first column), a Series or a list."""
    if isinstance(model_input, pd.DataFrame):
        column = INPUT_COLUMN if INPUT_COLUMN in model_input.columns else model_input.columns[0]
        values = model_input[column]
    elif isinstance(model_input, pd.Series):
        values = model_input
    else:
        values = pd.Series(list(model_input))
    return ["" if pd.isna(v) else str(v) for v in values.tolist()]


class SubjectClassifierModel(PythonModel):
    """Pyfunc model: `predict(DataFrame[text]) -> DataFrame[label, confidence, top3]`."""

    def __init__(self, pipeline: Any = None) -> None:
        # Set directly in tests and right after training; restored by `load_context` otherwise.
        self._pipeline = pipeline

    def __getstate__(self) -> dict[str, Any]:
        # The pipeline travels as its own artifact, not inside the pickled wrapper.
        return {"_pipeline": None}

    def load_context(self, context: PythonModelContext) -> None:
        self._pipeline = joblib.load(context.artifacts[PIPELINE_ARTIFACT])

    # `pd.DataFrame` is a hint mlflow 3 accepts without its own type-hint validation (`Any`
    # or no hint make it warn on every load); the logged signature validates `text` instead.
    def predict(
        self,
        context: PythonModelContext | None,  # noqa: ARG002 - pyfunc API
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,  # noqa: ARG002 - pyfunc API
    ) -> pd.DataFrame:
        if self._pipeline is None:
            raise RuntimeError("pipeline is not loaded")
        texts = input_texts(model_input)
        if not texts:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        proba = self._pipeline.predict_proba(texts)
        return format_predictions(self._pipeline.classes_, proba)
