import pandas as pd
import pytest

from edu_graphrag.ml.classifier import data


def _docs() -> pd.DataFrame:
    rows = [
        # doc_id, source_uri, hash, subject, is_sorted, status
        ("d1", "a/matan/x.pdf", "h1", "matan", True, "ok"),
        ("d2", "a/matan/x (2).pdf", "h1", "matan", True, "ok"),  # duplicate of d1
        ("d3", "unsorted/x.pdf", "h1", "matan", False, "ok"),  # unsorted duplicate of d1
        ("d4", "a/matan/y.pdf", "h2", "matan", True, "ok"),
        ("d5", "a/matan/z.pdf", "h3", "matan", True, "ok"),
        ("d6", "a/ml/same.pdf", "h4", "ml", True, "ok"),  # same bytes, other subject
        ("d7", "a/opt/same.pdf", "h4", "opt", True, "ok"),
        ("d8", "a/opt/scan.pdf", "h5", "opt", True, "empty"),
        ("d9", "a/none.pdf", "h6", None, False, "ok"),
        ("d10", "a/ml/a.md", "h7", "ml", True, "ok"),
    ]
    return pd.DataFrame(
        rows, columns=["doc_id", "source_uri", "content_hash", "subject", "is_sorted", "status"]
    )


def _chunks(doc_subjects: dict[str, str], per_doc: int = 2) -> pd.DataFrame:
    rows = [
        {
            "chunk_id": f"{doc_id}-{i}",
            "doc_id": doc_id,
            "chunk_index": i,
            "text": f"{subject} text {i}",
            "subject": subject,
        }
        for doc_id, subject in doc_subjects.items()
        for i in range(per_doc)
    ]
    return pd.DataFrame(rows)


def test_keep_ok_documents() -> None:
    kept = data.keep_ok_documents(_docs())
    assert set(kept["doc_id"]) == {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d10"}


def test_deduplicate_prefers_sorted_shortest_path_and_drops_conflicts() -> None:
    kept, conflicting = data.deduplicate_documents(data.keep_ok_documents(_docs()))
    assert conflicting == ["h4"]
    assert list(kept["doc_id"]) == ["d1", "d10", "d4", "d5"]


def test_rare_subjects() -> None:
    docs = pd.DataFrame({"doc_id": ["a", "b", "c", "d"], "subject": ["x", "x", "x", "y"]})
    assert data.rare_subjects(docs, 3) == {"y": 1}
    assert data.rare_subjects(docs, 1) == {}


def test_prepare_chunks() -> None:
    docs = _docs()
    chunks = _chunks(dict(zip(docs["doc_id"], docs["subject"].fillna("?"), strict=True)))
    chunks.loc[chunks["chunk_id"] == "d4-1", "text"] = "   "
    prepared = data.prepare_chunks(docs, chunks, min_docs=3)
    assert prepared.n_docs_not_ok == 2
    assert prepared.n_duplicate_docs_removed == 2
    assert prepared.conflicting_hashes == ["h4"]
    assert prepared.dropped_subjects == {"ml": 1}
    assert list(prepared.chunks.columns) == data.CHUNK_COLUMNS
    assert sorted(prepared.chunks["chunk_id"]) == ["d1-0", "d1-1", "d4-0", "d5-0", "d5-1"]


def _many_docs() -> pd.DataFrame:
    subjects = {f"a{i}": "alpha" for i in range(10)}
    subjects |= {f"b{i}": "beta" for i in range(3)}
    subjects |= {"c0": "gamma"}
    return _chunks(subjects, per_doc=3)


def test_split_by_document_is_grouped_stratified_and_deterministic() -> None:
    chunks = _many_docs()
    split = data.split_by_document(chunks, test_size=0.2, seed=7)
    train_docs, test_docs = set(split.train["doc_id"]), set(split.test["doc_id"])
    assert not train_docs & test_docs
    assert len(split.train) + len(split.test) == len(chunks)
    test_subjects = split.test.drop_duplicates("doc_id")["subject"].value_counts()
    assert test_subjects.to_dict() == {"alpha": 2, "beta": 1}  # gamma has one doc: train only
    assert "c0" in train_docs

    again = data.split_by_document(chunks, test_size=0.2, seed=7)
    pd.testing.assert_frame_equal(split.test, again.test)


@pytest.mark.parametrize("test_size", [0.01, 0.99])
def test_split_keeps_one_doc_on_each_side(test_size: float) -> None:
    split = data.split_by_document(_many_docs(), test_size=test_size, seed=1)
    for subject in ("alpha", "beta"):
        assert (split.test["subject"] == subject).any()
        assert (split.train["subject"] == subject).any()


def test_split_stats_and_class_distribution() -> None:
    split = data.split_by_document(_many_docs(), test_size=0.2, seed=7)
    stats = data.split_stats(split)
    assert stats == {
        "n_train_docs": 11,
        "n_train_chunks": 33,
        "n_train_classes": 3,
        "n_test_docs": 3,
        "n_test_chunks": 9,
        "n_test_classes": 2,
    }
    table = data.class_distribution(split)
    assert table.loc["gamma"].to_dict() == {
        "train_docs": 1,
        "train_chunks": 3,
        "test_docs": 0,
        "test_chunks": 0,
    }
    assert table.index[0] == "alpha"
