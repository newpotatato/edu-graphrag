import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from edu_graphrag.ml.ingest import build_dataset as bd
from edu_graphrag.ml.ingest.chunking import ChunkingConfig
from edu_graphrag.ml.ingest.connectors import FilesystemConnector

N_DOCS = 9  # supported, non-excluded files in the `archive` fixture


def _build(root: Path, out: Path, chunking: ChunkingConfig | None = None) -> bd.BuildStats:
    return bd.build_dataset(FilesystemConnector(root), out, chunking)


def test_build_writes_documents_chunks_and_manifest(archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    stats = _build(archive, out)
    assert (stats.n_documents, stats.converted, stats.reused, stats.removed) == (
        N_DOCS,
        N_DOCS,
        0,
        0,
    )
    assert stats.by_status == {"empty": 1, "failed": 1, "ok": N_DOCS - 2}

    docs = pq.read_table(out / bd.DOCUMENTS_FILE)
    chunks = pq.read_table(out / bd.CHUNKS_FILE)
    assert docs.schema.equals(bd.DOCUMENTS_SCHEMA)
    assert chunks.schema.equals(bd.CHUNKS_SCHEMA)
    rows = {r["source_uri"]: r for r in docs.to_pylist()}
    assert rows["3_grade/difgem/scan.pdf"]["status"] == "empty"
    assert rows["3_grade/difgem/broken.docx"]["status"] == "failed"
    assert rows["3_grade/difgem/broken.docx"]["error"]
    notes = rows["2_grade/4_sem/func_analyse/notes.md"]
    assert (notes["grade"], notes["semester"], notes["subject"], notes["is_sorted"]) == (
        2,
        4,
        "func_analyse",
        True,
    )

    chunk_rows = chunks.to_pylist()
    assert len(chunk_rows) == stats.n_chunks == sum(r["n_chunks"] for r in rows.values())
    keys = [(c["source_uri"], c["chunk_index"]) for c in chunk_rows]
    assert keys == sorted(keys)
    first = chunk_rows[0]
    assert first["chunk_id"] == f"{first['doc_id']}:{first['chunk_index']:04d}"
    assert first["n_chars"] == len(first["text"])

    manifest = json.loads((out / bd.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == stats.dataset_version
    assert len(stats.dataset_version) == 12
    assert manifest["chunking"] == {"target_chars": 1200, "overlap_chars": 150, "min_chars": 80}


def test_incremental_rebuild(archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    first = _build(archive, out)
    first_chunks = pq.read_table(out / bd.CHUNKS_FILE).to_pylist()

    second = _build(archive, out)
    # Everything but the failed document is reused; the dataset is byte-for-byte the same.
    assert (second.converted, second.reused, second.removed) == (1, N_DOCS - 1, 0)
    assert second.dataset_version == first.dataset_version
    assert pq.read_table(out / bd.CHUNKS_FILE).to_pylist() == first_chunks

    notes = archive / "2_grade/4_sem/func_analyse/notes.md"
    notes.write_text(
        notes.read_text(encoding="utf-8") + "\n\n# New\n\n" + "y " * 60, encoding="utf-8"
    )
    (archive / "3_grade/difgem/lecture (2).pdf").unlink()
    third = _build(archive, out)
    assert (third.converted, third.reused, third.removed) == (2, N_DOCS - 3, 1)
    assert third.dataset_version != first.dataset_version

    # A different chunking config invalidates the cache.
    fourth = _build(archive, out, ChunkingConfig(target_chars=500, overlap_chars=50))
    assert (fourth.converted, fourth.reused) == (N_DOCS - 1, 0)


def test_dataset_version_is_order_independent() -> None:
    chunks = [{"chunk_id": "a:0000", "text": "x"}, {"chunk_id": "b:0000", "text": "y"}]
    assert bd.dataset_version(chunks) == bd.dataset_version(list(reversed(chunks)))
    assert bd.dataset_version(chunks) != bd.dataset_version([{**chunks[0], "text": "z"}, chunks[1]])


def test_cli(
    archive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(archive))
    layout_calls: list[bool] = []

    def fake_pdf(path: Path, *, use_layout: bool = False) -> str:
        layout_calls.append(use_layout)
        return "# PDF " + "text " * 40

    # The real layout parser is heavy; only check that the flag is wired through.
    monkeypatch.setattr(bd, "convert_pdf", fake_pdf)
    out = tmp_path / "cli"
    assert bd.main(["--out", str(out), "--pdf-layout", "--target-chars", "800"]) == 0
    assert "documents" in capsys.readouterr().out
    assert (out / bd.MANIFEST_FILE).exists()
    assert layout_calls
    assert all(layout_calls)


def test_cli_requires_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(SystemExit):
        bd.main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        bd.main(["--data-root", str(tmp_path / "missing"), "--out", str(tmp_path)])
