import hashlib
import importlib
from pathlib import Path

import pytest

from edu_graphrag.ml.ingest import converters
from edu_graphrag.ml.ingest.connectors import RawDocument, SourceType
from edu_graphrag.ml.ingest.converters import CONVERTERS, convert, convert_pdf


def _doc(path: Path, source_type: SourceType) -> RawDocument:
    return RawDocument(
        source_uri=path.name,
        source_type=source_type,
        content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        grade=None,
        semester=None,
        subject=None,
        is_sorted=False,
        path=path,
    )


def test_registry_covers_all_source_types() -> None:
    assert set(CONVERTERS) == {"pdf", "md", "ipynb", "tex", "docx", "pptx"}


@pytest.mark.parametrize(
    ("relative", "source_type", "expected"),
    [
        ("2_grade/4_sem/func_analyse/notes.md", "md", ["# Banach", "Topological space"]),
        ("2_grade/4_sem/func_analyse/hw.tex", "tex", ["\\section{Hilbert}"]),
        (
            "2_grade/4_sem/func_analyse/lab.ipynb",
            "ipynb",
            ["# Lab 1", "```python\n# not a heading\nx = 1\n```"],
        ),
        (
            "2_grade/4_sem/func_analyse/report.docx",
            "docx",
            ["# Report", "# Section", "Topological space", "metric | value"],
        ),
        ("3_grade/difgem/slides.pptx", "pptx", ["## Slide 1", "Manifolds", "## Slide 2", "a | "]),
        ("3_grade/difgem/lecture.pdf", "pdf", ["Topological space"]),
    ],
)
def test_converters(
    archive: Path, relative: str, source_type: SourceType, expected: list[str]
) -> None:
    result = convert(_doc(archive / relative, source_type))
    assert result.status == "ok"
    assert result.error is None
    for fragment in expected:
        assert fragment in result.text


def test_broken_file_is_marked_failed(archive: Path) -> None:
    result = convert(_doc(archive / "3_grade/difgem/broken.docx", "docx"))
    assert result.status == "failed"
    assert result.error
    assert result.text == ""


def test_missing_converter_is_marked_failed(archive: Path) -> None:
    result = convert(_doc(archive / "3_grade/difgem/lecture.pdf", "pdf"), converters={})
    assert result.status == "failed"
    assert "no converter" in (result.error or "")


def test_nul_bytes_are_removed(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_bytes(b"abc\x00def")
    assert convert(_doc(path, "md")).text == "abcdef"


def test_pdf_falls_back_to_plain_text(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pymupdf4llm = importlib.import_module("pymupdf4llm")

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("layout failed")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", boom)
    text = convert_pdf(archive / "3_grade/difgem/lecture.pdf")
    assert "Topological space" in text
    assert converters.CONVERTERS["pdf"] is convert_pdf
