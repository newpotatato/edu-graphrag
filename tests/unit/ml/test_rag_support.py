"""Shared helpers for the retriever tests: a tiny synthetic corpus and a fake embedder."""

import json
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from edu_graphrag.ml.rag.tokenizer import tokenize

DIM = 32

DOCS = {
    "doc-lin": (
        "linear_algebra",
        [
            "Собственные значения матрицы находят из характеристического уравнения.",
            "Ортогональный проектор переводит вектор в ближайшую точку подпространства.",
        ],
    ),
    "doc-top": (
        "difgem",
        [
            "Топологическое пространство называется связным, если оно не разбивается на два открытых множества.",
            "База топологии порождает все открытые множества объединениями.",
        ],
    ),
    "doc-web": (
        "Frontend",
        ["Компонент Vue получает данные через пропсы и обновляет родителя через v-model."],
    ),
    "doc-web-copy": (
        "Frontend",
        ["Компонент Vue получает данные через пропсы и обновляет родителя через v-model."],
    ),
    "doc-misc": (None, ["Random words without any subject " * 30]),
}


def make_chunks() -> pd.DataFrame:
    rows = []
    for doc_id, (subject, texts) in DOCS.items():
        for index, text in enumerate(texts):
            rows.append(
                {
                    "chunk_id": f"{doc_id}:{index:04d}",
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": text,
                    "n_chars": len(text),
                    "source_uri": f"notes/{doc_id}.md",
                    "source_type": "md",
                    "subject": subject,
                }
            )
    return pd.DataFrame(rows)


QA_ROWS = [
    {
        "qid": "q001",
        "question": "Как найти собственные значения матрицы?",
        "doc_id": "doc-lin",
        "chunk_id": "doc-lin:0000",
        "subject": "linear_algebra",
        "qtype": "howto",
    },
    {
        "qid": "q002",
        "question": "Какое пространство называется связным?",
        "doc_id": "doc-top",
        "chunk_id": "doc-top:0000",
        "subject": "difgem",
        "qtype": "definition",
    },
    {
        "qid": "q003",
        "question": "Как компонент Vue передаёт данные через пропсы?",
        "doc_id": "doc-web",
        "chunk_id": "doc-web:0000",
        "subject": "Frontend",
        "qtype": "code",
        "alt_doc_ids": ["doc-web-copy"],
    },
    {
        "qid": "q004",
        "question": "Что такое гомеоморфизм многообразий?",
        "doc_id": "doc-top",
        "chunk_id": "doc-top:0001",
        "subject": "difgem",
        "qtype": "definition",
    },
]


def write_dataset(root: Path) -> tuple[Path, Path]:
    """`root/processed/{chunks.parquet,manifest.json}` and `root/qa.jsonl`."""
    data_dir = root / "processed"
    data_dir.mkdir(parents=True)
    make_chunks().to_parquet(data_dir / "chunks.parquet")
    manifest = {
        "dataset_version": "testver00001",
        "chunking": {"target_chars": 1200, "overlap_chars": 150, "min_chars": 80},
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    qa_path = root / "qa.jsonl"
    qa_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in QA_ROWS), encoding="utf-8"
    )
    return data_dir, qa_path


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors. No model download."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token[:5].encode()) % DIM] += 1.0
        out[:, 0] += 0.01  # no all-zero rows
        return out
