"""Index build/load and the retriever itself (no MLflow here).

An index is a directory of plain files, so it can be shipped as MLflow model artifacts:

- `config.json`        - `IndexConfig` (method, tokenizer, embedding model, fusion params, ...)
- `chunks.jsonl`       - per-chunk metadata returned with every hit (ids, uri, subject, snippet)
- `bm25_tokens.json`   - tokenized corpus for BM25 (bm25 / hybrid), frozen at build time
- `embeddings.npy`     - L2-normalised float32 passage embeddings (dense / hybrid)

Only numpy and rank-bm25 are needed to load it; fastembed is imported lazily, and only for
methods that embed queries.
"""

import json
import re
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import structlog
from rank_bm25 import BM25Okapi

from edu_graphrag.ml.rag.fusion import RRF_K, reciprocal_rank_fusion
from edu_graphrag.ml.rag.tokenizer import get_tokenizer

logger = structlog.get_logger(__name__)

CONFIG_FILE = "config.json"
CHUNKS_FILE = "chunks.jsonl"
TOKENS_FILE = "bm25_tokens.json"
EMBEDDINGS_FILE = "embeddings.npy"

METHODS = ("bm25", "dense", "hybrid")
SNIPPET_CHARS = 300
_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class IndexConfig:
    method: str  # bm25 | dense | hybrid
    tokenizer: str = "word"
    embed_model: str | None = None
    query_prefix: str = ""  # e.g. "query: " for e5 models
    passage_prefix: str = ""  # e.g. "passage: " for e5 models
    rrf_k: int = RRF_K
    fusion_depth: int = 100  # candidates taken from each retriever before fusion
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dataset_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}, expected one of {METHODS}")
        if self.uses_dense and not self.embed_model:
            raise ValueError(f"method {self.method!r} needs embed_model")
        get_tokenizer(self.tokenizer)  # fail fast on a bad tokenizer name

    @property
    def uses_bm25(self) -> bool:
        return self.method in ("bm25", "hybrid")

    @property
    def uses_dense(self) -> bool:
        return self.method in ("dense", "hybrid")


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    chunk_id: str
    doc_id: str
    source_uri: str
    subject: str | None
    snippet: str


@dataclass(frozen=True, slots=True)
class Hit:
    rank: int
    score: float
    chunk: ChunkMeta

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable row of the model output."""
        return {
            "rank": self.rank,
            "doc_id": self.chunk.doc_id,
            "chunk_id": self.chunk.chunk_id,
            "source_uri": self.chunk.source_uri,
            "subject": self.chunk.subject,
            "score": round(float(self.score), 6),
            "snippet": self.chunk.snippet,
        }


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(len(texts), dim) float matrix."""


EmbedderFactory = Callable[[IndexConfig, Path | None], Embedder]


class FastEmbedder:
    """fastembed ONNX embedder; with `model_dir` it loads only those local files (offline).

    One thread and small batches keep the memory footprint low.
    """

    def __init__(
        self,
        model_name: str,
        model_dir: Path | None = None,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 16,
        threads: int = 1,
    ) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"threads": threads}
        if model_dir is not None:
            kwargs.update(specific_model_path=str(model_dir), local_files_only=True)
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self._model = TextEmbedding(model_name, **kwargs)
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = list(self._model.embed(list(texts), batch_size=self._batch_size))
        return np.asarray(vectors, dtype=np.float32)


def fastembed_factory(config: IndexConfig, model_dir: Path | None) -> Embedder:
    if config.embed_model is None:  # pragma: no cover - IndexConfig validates this
        raise ValueError("embed_model is not set")
    return FastEmbedder(config.embed_model, model_dir)


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = _SPACES_RE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized: np.ndarray = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalized


def _batches(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_passages(
    embedder: Embedder, texts: Sequence[str], prefix: str = "", batch_size: int = 64
) -> np.ndarray:
    """Embed in slices so a progress line is logged and peak memory stays bounded."""
    parts = []
    for done, batch in enumerate(_batches([prefix + t for t in texts], batch_size), 1):
        parts.append(embedder.embed(batch))
        if done % 10 == 0:
            logger.info("passages_embedded", n=min(done * batch_size, len(texts)))
    return normalize_rows(np.vstack(parts))


@dataclass(frozen=True, slots=True)
class BuildReport:
    out_dir: Path
    n_chunks: int
    seconds: float


def build_index(
    chunks: pd.DataFrame,
    config: IndexConfig,
    out_dir: Path,
    *,
    embedder: Embedder | None = None,
    embeddings: np.ndarray | None = None,
) -> BuildReport:
    """Write the index for `chunks` into `out_dir` (recreated).

    Dense methods need either precomputed `embeddings` (rows aligned with `chunks`, reused
    across variants) or an `embedder` to compute them.
    """
    started = time.perf_counter()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    texts = chunks["text"].astype(str).tolist()
    (out_dir / CONFIG_FILE).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    with (out_dir / CHUNKS_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        for row, text in zip(chunks.itertuples(index=False), texts, strict=True):
            subject = None if pd.isna(row.subject) else str(row.subject)
            meta = ChunkMeta(
                str(row.chunk_id), str(row.doc_id), str(row.source_uri), subject, make_snippet(text)
            )
            fh.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
    if config.uses_bm25:
        tokenizer = get_tokenizer(config.tokenizer)
        tokens = [tokenizer(t) for t in texts]
        (out_dir / TOKENS_FILE).write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    if config.uses_dense:
        if embeddings is None:
            if embedder is None:
                raise ValueError("dense index needs an embedder or precomputed embeddings")
            embeddings = embed_passages(embedder, texts, config.passage_prefix)
        if embeddings.shape[0] != len(texts):
            raise ValueError(f"{embeddings.shape[0]} embeddings for {len(texts)} chunks")
        np.save(out_dir / EMBEDDINGS_FILE, normalize_rows(embeddings))
    seconds = time.perf_counter() - started
    logger.info("index_built", method=config.method, n_chunks=len(texts), seconds=round(seconds, 2))
    return BuildReport(out_dir=out_dir, n_chunks=len(texts), seconds=seconds)


@dataclass(slots=True)
class LoadedIndex:
    config: IndexConfig
    chunks: list[ChunkMeta]
    bm25: BM25Okapi | None
    embeddings: np.ndarray | None


def load_index(index_dir: Path) -> LoadedIndex:
    config = IndexConfig(**json.loads((index_dir / CONFIG_FILE).read_text(encoding="utf-8")))
    with (index_dir / CHUNKS_FILE).open(encoding="utf-8") as fh:
        chunks = [ChunkMeta(**json.loads(line)) for line in fh if line.strip()]
    bm25 = None
    if config.uses_bm25:
        tokens = json.loads((index_dir / TOKENS_FILE).read_text(encoding="utf-8"))
        bm25 = BM25Okapi(tokens, k1=config.bm25_k1, b=config.bm25_b)
    embeddings = np.load(index_dir / EMBEDDINGS_FILE) if config.uses_dense else None
    return LoadedIndex(config=config, chunks=chunks, bm25=bm25, embeddings=embeddings)


class Retriever:
    """Top-k chunk search over a loaded index."""

    def __init__(self, index: LoadedIndex, embedder: Embedder | None = None) -> None:
        if index.config.uses_dense and embedder is None:
            raise ValueError(f"method {index.config.method!r} needs a query embedder")
        self._index = index
        self._embedder = embedder
        self._tokenizer = get_tokenizer(index.config.tokenizer)

    @property
    def config(self) -> IndexConfig:
        return self._index.config

    def _bm25_ranking(self, query: str, depth: int) -> list[tuple[int, float]]:
        bm25 = self._index.bm25
        if bm25 is None:  # pragma: no cover - guarded by the method check in search()
            raise RuntimeError("index has no BM25 part")
        tokens = self._tokenizer(query)
        if not tokens:
            return []
        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:depth]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _dense_ranking(self, query: str, depth: int) -> list[tuple[int, float]]:
        matrix, embedder = self._index.embeddings, self._embedder
        if matrix is None or embedder is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError("index has no dense part")
        vector = normalize_rows(embedder.embed([self.config.query_prefix + query]))[0]
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[:depth]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        cfg = self.config
        ranked: list[tuple[int, float]]
        if cfg.method == "bm25":
            ranked = self._bm25_ranking(query, k)
        elif cfg.method == "dense":
            ranked = self._dense_ranking(query, k)
        else:
            depth = max(k, cfg.fusion_depth)
            lexical = [i for i, _ in self._bm25_ranking(query, depth)]
            semantic = [i for i, _ in self._dense_ranking(query, depth)]
            ranked = reciprocal_rank_fusion([lexical, semantic], k=cfg.rrf_k)[:k]
        return [
            Hit(rank=rank, score=score, chunk=self._index.chunks[i])
            for rank, (i, score) in enumerate(ranked, 1)
        ]
