"""Build the processed dataset: `documents.parquet`, `chunks.parquet` and `manifest.json`.

Incremental: documents whose content hash did not change since the previous build (and that
did not fail) are not converted again; their rows and chunks are reused.

Usage: `uv run python -m edu_graphrag.ml.ingest.build_dataset --data-root ... --out data/processed`
"""

import argparse
import functools
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from edu_graphrag.ml.ingest.chunking import ChunkingConfig, chunk_text
from edu_graphrag.ml.ingest.connectors import Connector, FilesystemConnector, SourceType
from edu_graphrag.ml.ingest.converters import CONVERTERS, Converter, convert, convert_pdf

logger = structlog.get_logger(__name__)

DOCUMENTS_FILE = "documents.parquet"
CHUNKS_FILE = "chunks.parquet"
MANIFEST_FILE = "manifest.json"

DOCUMENTS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_uri", pa.string()),
        ("source_type", pa.string()),
        ("content_hash", pa.string()),
        ("grade", pa.int32()),
        ("semester", pa.int32()),
        ("subject", pa.string()),
        ("is_sorted", pa.bool_()),
        ("n_chars", pa.int64()),
        ("n_chunks", pa.int64()),
        ("status", pa.string()),  # ok | failed | empty
        ("error", pa.string()),
    ]
)

CHUNKS_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("text", pa.string()),
        ("n_chars", pa.int64()),
        ("source_uri", pa.string()),
        ("source_type", pa.string()),
        ("grade", pa.int32()),
        ("semester", pa.int32()),
        ("subject", pa.string()),
        ("is_sorted", pa.bool_()),
    ]
)

Row = dict[str, Any]


@dataclass
class BuildStats:
    dataset_version: str
    n_documents: int
    n_chunks: int
    converted: int
    reused: int
    removed: int
    by_status: dict[str, int] = field(default_factory=dict)
    by_source_type: dict[str, int] = field(default_factory=dict)
    chunking: dict[str, int] = field(default_factory=dict)


def dataset_version(chunks: Sequence[Row]) -> str:
    """Short digest over sorted (chunk_id, sha256(text)) pairs; stable across rebuilds."""
    digest = hashlib.sha256()
    for chunk_id, text in sorted((c["chunk_id"], c["text"]) for c in chunks):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        digest.update(f"{chunk_id}\t{text_hash}\n".encode())
    return digest.hexdigest()[:12]


def _load_previous(out_dir: Path, chunking: ChunkingConfig) -> tuple[dict[str, Row], list[Row]]:
    """Previous documents (by doc_id) and chunks, or nothing if absent or built differently."""
    docs_path, chunks_path = out_dir / DOCUMENTS_FILE, out_dir / CHUNKS_FILE
    manifest_path = out_dir / MANIFEST_FILE
    if not (docs_path.exists() and chunks_path.exists() and manifest_path.exists()):
        return {}, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chunking") != asdict(chunking):
        logger.info("chunking_config_changed_full_rebuild")
        return {}, []
    docs = pq.read_table(docs_path).to_pylist()
    return {d["doc_id"]: d for d in docs}, pq.read_table(chunks_path).to_pylist()


def build_dataset(
    connector: Connector,
    out_dir: Path,
    chunking: ChunkingConfig | None = None,
    converters: dict[SourceType, Converter] = CONVERTERS,
) -> BuildStats:
    """Convert, chunk and write the dataset into `out_dir`."""
    cfg = chunking or ChunkingConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_docs, prev_chunks = _load_previous(out_dir, cfg)
    prev_chunks_by_doc: dict[str, list[Row]] = {}
    for chunk in prev_chunks:
        prev_chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    doc_rows: list[Row] = []
    chunk_rows: list[Row] = []
    converted = reused = 0
    seen: set[str] = set()
    for doc in connector.iter_documents():
        doc_id = doc.doc_id
        seen.add(doc_id)
        meta: Row = {
            "source_uri": doc.source_uri,
            "source_type": doc.source_type,
            "grade": doc.grade,
            "semester": doc.semester,
            "subject": doc.subject,
            "is_sorted": doc.is_sorted,
        }
        prev = prev_docs.get(doc_id)
        if (
            prev is not None
            and prev["content_hash"] == doc.content_hash
            and prev["status"] != "failed"
        ):
            reused += 1
            doc_rows.append({**prev, **meta})
            chunk_rows.extend({**c, **meta} for c in prev_chunks_by_doc.get(doc_id, []))
            continue

        converted += 1
        started = time.perf_counter()
        result = convert(doc, converters)
        chunks = chunk_text(result.text, cfg) if result.status == "ok" else []
        status = result.status if result.status == "failed" or chunks else "empty"
        logger.info(
            "document_converted",
            source_uri=doc.source_uri,
            status=status,
            n_chunks=len(chunks),
            seconds=round(time.perf_counter() - started, 2),
        )
        doc_rows.append(
            {
                "doc_id": doc_id,
                "content_hash": doc.content_hash,
                **meta,
                "n_chars": len(result.text),
                "n_chunks": len(chunks),
                "status": status,
                "error": result.error,
            }
        )
        chunk_rows.extend(
            {
                "chunk_id": f"{doc_id}:{index:04d}",
                "doc_id": doc_id,
                "chunk_index": index,
                "text": text,
                "n_chars": len(text),
                **meta,
            }
            for index, text in enumerate(chunks)
        )

    removed = len(set(prev_docs) - seen)
    doc_rows.sort(key=lambda r: r["source_uri"])
    chunk_rows.sort(key=lambda r: (r["source_uri"], r["chunk_index"]))
    pq.write_table(
        pa.Table.from_pylist(doc_rows, schema=DOCUMENTS_SCHEMA), out_dir / DOCUMENTS_FILE
    )
    pq.write_table(pa.Table.from_pylist(chunk_rows, schema=CHUNKS_SCHEMA), out_dir / CHUNKS_FILE)

    stats = BuildStats(
        dataset_version=dataset_version(chunk_rows),
        n_documents=len(doc_rows),
        n_chunks=len(chunk_rows),
        converted=converted,
        reused=reused,
        removed=removed,
        by_status=dict(sorted(Counter(r["status"] for r in doc_rows).items())),
        by_source_type=dict(sorted(Counter(r["source_type"] for r in doc_rows).items())),
        chunking=asdict(cfg),
    )
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(asdict(stats), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "dataset_built",
        dataset_version=stats.dataset_version,
        documents=stats.n_documents,
        chunks=stats.n_chunks,
        converted=converted,
        reused=reused,
        removed=removed,
    )
    return stats


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--data-root", type=Path, default=None, help="archive root (default: $DATA_ROOT)"
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--target-chars", type=int, default=ChunkingConfig.target_chars)
    parser.add_argument("--overlap-chars", type=int, default=ChunkingConfig.overlap_chars)
    parser.add_argument("--min-chars", type=int, default=ChunkingConfig.min_chars)
    parser.add_argument(
        "--pdf-layout", action="store_true", help="use the slower pymupdf-layout PDF parser"
    )
    args = parser.parse_args(argv)
    if args.data_root is None:
        env_root = os.environ.get("DATA_ROOT")
        if not env_root:
            parser.error("--data-root or the DATA_ROOT environment variable is required")
        args.data_root = Path(env_root)
    if not args.data_root.is_dir():
        parser.error(f"data root {args.data_root} is not a directory")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from edu_graphrag.logging_config import configure_logging

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic paths on Windows consoles
    configure_logging("INFO", json_logs=False)
    args = _parse_args(argv)
    chunking = ChunkingConfig(args.target_chars, args.overlap_chars, args.min_chars)
    converters = dict(CONVERTERS)
    if args.pdf_layout:
        converters["pdf"] = functools.partial(convert_pdf, use_layout=True)
    connector = FilesystemConnector(args.data_root.resolve())
    stats = build_dataset(connector, args.out, chunking, converters)
    print(
        f"dataset {stats.dataset_version}: {stats.n_documents} documents, {stats.n_chunks} chunks "
        f"(converted {stats.converted}, reused {stats.reused}, removed {stats.removed}); "
        f"status {stats.by_status}; types {stats.by_source_type}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
