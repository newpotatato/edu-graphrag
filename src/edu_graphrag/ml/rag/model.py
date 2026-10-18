"""The retriever as an MLflow pyfunc model.

Model version == index version. The whole index (chunk metadata, frozen BM25 tokens, the
passage embedding matrix) and, for dense methods, the ONNX embedding model files are stored
as artifacts of the logged model. A registered version is therefore self-contained and
immutable: `study-retriever` vN always searches exactly the corpus snapshot it was built and
evaluated on (its `dataset_version` tag/param says which), and serving it needs no access to
`data/processed`, no index service and no network (the embedder is loaded from the bundled
files). Re-indexing a new dataset means logging and registering a new version; rolling back
the index is moving an alias.

Contract (same as every model served by `/api/v1/process`):
input  - DataFrame with a `text` column (one query per row);
output - DataFrame with a `results` column: per query, a list of top-k dicts
         {rank, doc_id, chunk_id, source_uri, subject, score, snippet}. Optional param `k`.

Loading needs only mlflow-skinny, pandas, numpy, rank-bm25 and (dense) fastembed.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.pyfunc.model import PythonModel

from edu_graphrag.ml.rag.index import (
    EmbedderFactory,
    Retriever,
    fastembed_factory,
    load_index,
)

TEXT_COLUMN = "text"
RESULTS_COLUMN = "results"
DEFAULT_K = 5
MAX_K = 50
INDEX_ARTIFACT = "index"
EMBEDDER_ARTIFACT = "embedder"


class RetrieverModel(PythonModel):
    def __init__(self, embedder_factory: EmbedderFactory | None = None) -> None:
        # The factory is a test seam; the logged model uses fastembed.
        self._embedder_factory = embedder_factory
        self._retriever: Retriever | None = None

    def load_context(self, context: Any) -> None:
        index = load_index(Path(context.artifacts[INDEX_ARTIFACT]))
        embedder = None
        if index.config.uses_dense:
            model_dir = context.artifacts.get(EMBEDDER_ARTIFACT)
            factory = self._embedder_factory or fastembed_factory
            embedder = factory(index.config, Path(model_dir) if model_dir else None)
        self._retriever = Retriever(index, embedder)

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("RetrieverModel.load_context() was not called")
        return self._retriever

    def predict(
        self,
        context: Any,  # noqa: ARG002 - part of the pyfunc interface
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        k = min(max(int((params or {}).get("k", DEFAULT_K)), 1), MAX_K)
        queries = _queries(model_input)
        results = [[hit.to_dict() for hit in self.retriever.search(q, k)] for q in queries]
        return pd.DataFrame({RESULTS_COLUMN: results})


def _queries(model_input: Any) -> list[str]:
    """Accept a DataFrame with `text`, a dict of lists, or a plain list of strings."""
    if isinstance(model_input, pd.DataFrame):
        column = TEXT_COLUMN if TEXT_COLUMN in model_input.columns else model_input.columns[0]
        values = model_input[column].tolist()
    elif isinstance(model_input, dict):
        values = list(model_input[TEXT_COLUMN])
    else:
        values = list(model_input)
    return ["" if v is None else str(v) for v in values]
