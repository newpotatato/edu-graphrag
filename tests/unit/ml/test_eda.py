from pathlib import Path

import pandas as pd
import pytest

from edu_graphrag.ml import eda
from edu_graphrag.ml.ingest.build_dataset import build_dataset
from edu_graphrag.ml.ingest.connectors import FilesystemConnector


def _docs() -> pd.DataFrame:
    rows = [
        # source_uri, type, hash, grade, sem, subject, sorted, n_chars, n_chunks, status, error
        ("1_grade/1_sem/matan/a.pdf", "pdf", "h1", 1, 1, "matan", True, 5000, 4, "ok", None),
        ("2_grade/3_sem/matan/b.md", "md", "h2", 2, 3, "matan", True, 3000, 3, "ok", None),
        ("3_grade/difgem/c.pdf", "pdf", "h3", 3, None, "difgem", False, 1500, 1, "ok", None),
        ("3_grade/difgem/c (2).pdf", "pdf", "h3", 3, None, "difgem", False, 1500, 1, "ok", None),
        ("3_grade/difgem/d (3).pdf", "pdf", "h4", 3, None, "difgem", False, 0, 0, "empty", None),
        ("1_grade/x.docx", "docx", "h5", 1, None, None, False, 0, 0, "failed", "BadZipFile"),
    ]
    columns = [
        "source_uri",
        "source_type",
        "content_hash",
        "grade",
        "semester",
        "subject",
        "is_sorted",
        "n_chars",
        "n_chunks",
        "status",
        "error",
    ]
    return pd.DataFrame(rows, columns=columns)


def _chunks(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc in docs.itertuples():
        for index in range(doc.n_chunks):
            text = f"{doc.subject} lecture topology manifold {index} " * 5
            rows.append(
                {"subject": doc.subject, "text": text, "n_chars": len(text), "chunk_index": index}
            )
    return pd.DataFrame(rows)


def test_docs_per_subject() -> None:
    table = eda.docs_per_subject(_docs())
    assert table.loc["matan"].to_dict() == {"sorted": 2, "unsorted": 0, "total": 2}
    assert table.loc["difgem"].to_dict() == {"sorted": 0, "unsorted": 3, "total": 3}
    assert table.loc[eda.NO_SUBJECT, "total"] == 1
    assert table.index[0] == "difgem"


def test_class_stats_and_imbalance() -> None:
    docs = _docs()
    stats = eda.class_stats(docs, _chunks(docs))
    assert stats.loc["matan", "n_chunks"] == 7
    assert stats.loc["difgem", "n_chunks"] == 2
    assert stats.loc["difgem", "n_docs_ok"] == 2
    assert stats.loc[eda.NO_SUBJECT, "n_chunks"] == 0
    assert stats["chunk_share"].sum() == pytest.approx(1.0)
    assert eda.imbalance_ratio(stats) == pytest.approx(3.5)
    assert eda.imbalance_ratio(stats.iloc[0:0]) == 0.0


def test_duplicates() -> None:
    docs = _docs()
    content = eda.content_duplicates(docs)
    assert content.to_dict("records") == [
        {
            "content_hash": "h3",
            "n_copies": 2,
            "source_uris": "3_grade/difgem/c (2).pdf; 3_grade/difgem/c.pdf",
        }
    ]
    names = eda.filename_duplicates(docs)
    assert names.set_index("source_uri")["has_original"].to_dict() == {
        "3_grade/difgem/c (2).pdf": True,
        "3_grade/difgem/d (3).pdf": False,
    }
    assert eda.strip_copy_suffix("a/b/lecture_1 (2).pdf") == "a/b/lecture_1.pdf"
    assert eda.strip_copy_suffix("a/b.pdf") == "a/b.pdf"


def test_problem_documents_and_grade_semester() -> None:
    docs = _docs()
    problems = eda.problem_documents(docs)
    assert problems["status"].tolist() == ["empty", "failed"]
    counts = eda.grade_semester_counts(docs)
    as_dict = {(r.grade, r.semester): r.n_docs for r in counts.itertuples()}
    assert as_dict == {
        ("grade 1", "sem 1"): 1,
        ("grade 1", "-"): 1,
        ("grade 2", "sem 3"): 1,
        ("grade 3", "-"): 3,
    }
    lengths = eda.length_by_type(docs)
    assert lengths.loc["pdf", "count"] == 4


def test_top_terms() -> None:
    docs = _docs()
    terms = eda.top_terms(_chunks(docs), k=3).set_index("subject")["top_terms"]
    assert terms["matan"].split(", ")[0] == "matan"
    assert terms["difgem"].split(", ")[0] == "difgem"


@pytest.fixture
def processed(archive: Path, tmp_path: Path) -> Path:
    out = tmp_path / "processed"
    build_dataset(FilesystemConnector(archive), out)
    return out


def test_run_eda_and_log_to_mlflow(
    processed: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mlflow

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    out = tmp_path / "eda"
    assert eda.main(["--data", str(processed), "--out", str(out), "--log-to-mlflow"]) == 0
    assert "MLflow run id" in capsys.readouterr().out

    for name in ("eda_summary.md", "class_stats.csv", "docs_per_subject.png", "duplicates.png"):
        assert (out / name).stat().st_size > 0
    summary = (out / "eda_summary.md").read_text(encoding="utf-8")
    assert "Class imbalance" in summary
    assert "byte-identical" in summary
    assert "empty = 1" in summary

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.search_runs(experiment_names=[eda.EXPERIMENT_NAME], output_format="list")[0]
    version = eda.load_dataset(processed).version
    assert run.info.run_name == f"eda-{version}"
    assert run.data.params["dataset_version"] == version
    assert int(run.data.params["n_docs"]) == 9
    assert run.inputs is not None
    dataset_input = run.inputs.dataset_inputs[0]
    assert dataset_input.dataset.name == "study-chunks"
    assert dataset_input.tags[0].value == "eda"
