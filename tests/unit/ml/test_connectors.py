from pathlib import Path

import pytest

from edu_graphrag.ml.ingest.connectors import (
    FilesystemConnector,
    PathInfo,
    is_excluded,
    make_doc_id,
    parse_path,
    source_type_for,
)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("2_grade/4_sem/func_analyse/x.pdf", PathInfo(2, 4, "func_analyse", True)),
        ("1_grade/1_sem/matan/deep/dir/a.md", PathInfo(1, 1, "matan", True)),
        ("2_grade/3_sem/matan/b.md", PathInfo(2, 3, "matan", True)),
        ("3_grade/difgem/lecture_1 (2).pdf", PathInfo(3, None, "difgem", False)),
        ("3_grade/bd/hw_hdfs/report.md", PathInfo(3, None, "bd", False)),
        ("1_grade/1_sem/loose.pdf", PathInfo(1, 1, None, False)),
        ("1_grade/loose.pdf", PathInfo(1, None, None, False)),
        ("misc/file.pdf", PathInfo(None, None, None, False)),
        ("file.pdf", PathInfo(None, None, None, False)),
    ],
)
def test_parse_path(uri: str, expected: PathInfo) -> None:
    assert parse_path(uri) == expected


@pytest.mark.parametrize(
    ("uri", "excluded"),
    [
        ("3_grade/MLops/edu-graphrag/README.md", True),
        ("3_grade/MLops", True),
        ("3_grade/MLopsX/a.md", False),
        ("1_grade/1_sem/python/node_modules/pkg/a.md", True),
        ("1_grade/1_sem/python/.venv/a.md", True),
        ("1_grade/1_sem/python/.ipynb_checkpoints/a.ipynb", True),
        ("1_grade/1_sem/python/.obsidian/a.md", True),
        ("2_grade/4_sem/func_analyse/x.pdf", False),
    ],
)
def test_is_excluded(uri: str, excluded: bool) -> None:
    assert is_excluded(uri) is excluded


def test_doc_id_and_source_type() -> None:
    doc_id = make_doc_id("3_grade/difgem/a.pdf")
    assert len(doc_id) == 16
    assert doc_id == make_doc_id("3_grade/difgem/a.pdf") != make_doc_id("3_grade/difgem/b.pdf")
    assert source_type_for("A.PDF") == "pdf"
    assert source_type_for("a.png") is None


def test_filesystem_connector(archive: Path) -> None:
    docs = list(FilesystemConnector(archive).iter_documents())
    uris = [d.source_uri for d in docs]
    assert uris == sorted(uris)
    assert not any("MLops" in u or "node_modules" in u or "hidden" in u for u in uris)
    assert not any(u.endswith(".png") or "lock" in u for u in uris)
    assert {d.source_type for d in docs} == {"pdf", "md", "ipynb", "tex", "docx", "pptx"}

    by_uri = {d.source_uri: d for d in docs}
    notes = by_uri["2_grade/4_sem/func_analyse/notes.md"]
    assert (notes.grade, notes.semester, notes.subject, notes.is_sorted) == (
        2,
        4,
        "func_analyse",
        True,
    )
    assert notes.read_bytes().startswith(b"# Banach")
    assert notes.doc_id == make_doc_id(notes.source_uri)
    lecture = by_uri["3_grade/difgem/lecture.pdf"]
    assert lecture.subject == "difgem"
    assert not lecture.is_sorted
    assert lecture.content_hash == by_uri["3_grade/difgem/lecture (2).pdf"].content_hash
