"""QA loading/validation and the index + retriever (BM25, dense with a fake embedder, hybrid)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edu_graphrag.ml.rag import index as index_mod
from edu_graphrag.ml.rag.index import (
    EMBEDDINGS_FILE,
    TOKENS_FILE,
    IndexConfig,
    Retriever,
    build_index,
    embed_passages,
    load_index,
    make_snippet,
    normalize_rows,
)
from edu_graphrag.ml.rag.qa import QAValidationError, load_qa, parse_item, qa_frame, validate_qa
from tests.unit.ml.test_rag_support import FakeEmbedder, make_chunks, write_dataset

# ---------------------------------------------------------------- QA set


def test_load_and_validate_qa(tmp_path: Path) -> None:
    _, qa_path = write_dataset(tmp_path)
    items = load_qa(qa_path)
    assert [i.qid for i in items] == ["q001", "q002", "q003", "q004"]
    assert items[2].gold_docs == {"doc-web", "doc-web-copy"}
    assert items[0].alt_doc_ids == ()
    validate_qa(items, make_chunks())
    frame = qa_frame(items)
    assert list(frame.columns) == [
        "qid",
        "question",
        "doc_id",
        "chunk_id",
        "subject",
        "qtype",
        "alt_doc_ids",
    ]
    assert frame.loc[2, "alt_doc_ids"] == "doc-web-copy"


def test_validate_qa_reports_every_problem() -> None:
    base = {"question": "q?", "doc_id": "doc-lin", "chunk_id": "doc-lin:0000", "qtype": "fact"}
    items = [
        parse_item({"qid": "a", **base}),
        parse_item({"qid": "a", **base}),  # duplicate
        parse_item({**base, "qid": "b", "chunk_id": "nope:0000"}),
        parse_item({**base, "qid": "c", "doc_id": "doc-top"}),  # chunk not in that doc
        parse_item({**base, "qid": "d", "alt_doc_ids": ["ghost"]}),
        parse_item({**base, "qid": "e", "question": "  ", "qtype": "poem"}),
    ]
    with pytest.raises(QAValidationError) as err:
        validate_qa(items, make_chunks())
    message = str(err.value)
    for expected in (
        "a: duplicate qid",
        "b: unknown chunk_id nope:0000",
        "c: chunk doc-lin:0000 is not in doc doc-top",
        "d: unknown doc_id ghost",
        "e: empty question",
        "e: unknown qtype 'poem'",
    ):
        assert expected in message


def test_parse_item_rejects_bad_alt_ids() -> None:
    with pytest.raises(QAValidationError, match="alt_doc_ids must be a list"):
        parse_item({"qid": "x", "alt_doc_ids": "doc", "question": "", "doc_id": "", "chunk_id": ""})
    assert (
        parse_item(
            {
                "qid": "x",
                "question": "q",
                "doc_id": "d",
                "chunk_id": "c",
                "qtype": "fact",
                "subject": None,
            }
        ).subject
        is None
    )


# ---------------------------------------------------------------- config / helpers


def test_index_config_validation() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        IndexConfig(method="tfidf")
    with pytest.raises(ValueError, match="needs embed_model"):
        IndexConfig(method="dense")
    with pytest.raises(ValueError, match="unknown tokenizer"):
        IndexConfig(method="bm25", tokenizer="porter")
    hybrid = IndexConfig(method="hybrid", embed_model="m")
    assert hybrid.uses_bm25
    assert hybrid.uses_dense


def test_snippet_and_normalisation() -> None:
    assert make_snippet("  a \n\n b  ") == "a b"
    long = make_snippet("слово " * 100)
    assert len(long) <= 300
    assert long.endswith("…")
    rows = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert rows.dtype == np.float32
    assert np.allclose(rows[0], [0.6, 0.8])
    assert np.allclose(rows[1], [0.0, 0.0])


def test_embed_passages_batches_and_prefixes() -> None:
    seen: list[list[str]] = []

    class Recorder(FakeEmbedder):
        def embed(self, texts: Any) -> np.ndarray:
            seen.append(list(texts))
            return super().embed(texts)

    matrix = embed_passages(Recorder(), [f"text {i}" for i in range(25)], "passage: ", batch_size=2)
    assert matrix.shape == (25, 32)
    assert len(seen) == 13
    assert seen[0] == ["passage: text 0", "passage: text 1"]
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


# ---------------------------------------------------------------- build / load / search


def test_bm25_index_round_trip_and_search(tmp_path: Path) -> None:
    chunks = make_chunks()
    config = IndexConfig(method="bm25", tokenizer="prefix5", dataset_version="v1")
    report = build_index(chunks, config, tmp_path / "idx")
    assert report.n_chunks == len(chunks)
    assert not (tmp_path / "idx" / EMBEDDINGS_FILE).exists()
    tokens = json.loads((tmp_path / "idx" / TOKENS_FILE).read_text(encoding="utf-8"))
    assert tokens[0][:2] == ["собст", "значе"]

    loaded = load_index(tmp_path / "idx")
    assert loaded.config == config
    assert loaded.embeddings is None
    retriever = Retriever(loaded)
    hits = retriever.search("собственное значение матриц", k=3)
    assert hits[0].chunk.chunk_id == "doc-lin:0000"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)
    row = hits[0].to_dict()
    assert set(row) == {"rank", "doc_id", "chunk_id", "source_uri", "subject", "score", "snippet"}
    json.dumps(row)  # JSON-serialisable
    assert retriever.search("", k=3) == []
    assert retriever.search("!!! ?", k=3) == []  # no tokens
    assert retriever.search("зебра", k=3) == []  # no matching term
    assert retriever.search("матрица", k=0) == []


def test_build_index_overwrites_and_keeps_missing_subject(tmp_path: Path) -> None:
    out = tmp_path / "idx"
    (out / "stale").mkdir(parents=True)
    build_index(make_chunks(), IndexConfig(method="bm25"), out)
    assert not (out / "stale").exists()
    loaded = load_index(out)
    assert {c.subject for c in loaded.chunks} >= {None, "difgem"}


def test_dense_and_hybrid_search_with_fake_embedder(tmp_path: Path) -> None:
    chunks = make_chunks()
    embedder = FakeEmbedder()
    dense_cfg = IndexConfig(method="dense", embed_model="fake", query_prefix="query: ")
    build_index(chunks, dense_cfg, tmp_path / "dense", embedder=embedder)
    dense = Retriever(load_index(tmp_path / "dense"), embedder)
    assert dense.search("связное топологическое пространство", k=2)[0].chunk.doc_id == "doc-top"
    assert len(dense.search("что угодно", k=4)) == 4  # dense always returns k

    precomputed = embed_passages(embedder, chunks["text"].tolist())
    calls = embedder.calls
    hybrid_cfg = IndexConfig(method="hybrid", embed_model="fake", fusion_depth=3)
    build_index(chunks, hybrid_cfg, tmp_path / "hybrid", embeddings=precomputed)
    assert embedder.calls == calls  # precomputed embeddings are reused
    hybrid = Retriever(load_index(tmp_path / "hybrid"), embedder)
    hits = hybrid.search("компонент Vue пропсы", k=2)
    assert hits[0].chunk.doc_id in {"doc-web", "doc-web-copy"}
    assert hits[0].score == pytest.approx(2 / 61)  # first in both lists
    # Lexical miss: fusion still returns the dense candidates.
    assert len(hybrid.search("зебра", k=3)) == 3


def test_dense_index_errors(tmp_path: Path) -> None:
    chunks = make_chunks()
    cfg = IndexConfig(method="dense", embed_model="fake")
    with pytest.raises(ValueError, match="needs an embedder or precomputed"):
        build_index(chunks, cfg, tmp_path / "a")
    with pytest.raises(ValueError, match="embeddings for"):
        build_index(chunks, cfg, tmp_path / "b", embeddings=np.ones((2, 4), dtype=np.float32))
    build_index(chunks, cfg, tmp_path / "c", embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="needs a query embedder"):
        Retriever(load_index(tmp_path / "c"))


def test_fastembed_wrapper_offline_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FastEmbedder passes local files / cache / threads to fastembed (fake class, no download)."""
    created: dict[str, Any] = {}

    class FakeTextEmbedding:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            created.update(model_name=model_name, **kwargs)

        def embed(self, texts: list[str], batch_size: int) -> Any:
            created["batch_size"] = batch_size
            return iter([np.ones(3) for _ in texts])

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    embedder = index_mod.fastembed_factory(IndexConfig(method="dense", embed_model="m"), tmp_path)
    assert embedder.embed(["a", "b"]).shape == (2, 3)
    assert created == {
        "model_name": "m",
        "threads": 1,
        "specific_model_path": str(tmp_path),
        "local_files_only": True,
        "batch_size": 16,
    }
    created.clear()
    index_mod.FastEmbedder("m", cache_dir=tmp_path)
    assert created == {"model_name": "m", "threads": 1, "cache_dir": str(tmp_path)}


def test_chunks_without_subject_column_values(tmp_path: Path) -> None:
    chunks = make_chunks().assign(subject=pd.NA)
    build_index(chunks, IndexConfig(method="bm25"), tmp_path / "idx")
    assert all(c.subject is None for c in load_index(tmp_path / "idx").chunks)
