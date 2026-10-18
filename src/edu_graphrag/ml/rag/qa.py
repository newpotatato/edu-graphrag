"""Evaluation QA set: `data/qa/qa.jsonl`, one hand-written question per sampled chunk.

Relevance is judged at document level (`doc_id` plus `alt_doc_ids`: byte-identical or
near-duplicate copies of the same material), so runs with different chunking stay
comparable; `chunk_id` is kept for the stricter exact-chunk metric.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

QTYPES = frozenset({"definition", "howto", "why", "theorem", "code", "fact"})


class QAValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QAItem:
    qid: str
    question: str
    doc_id: str
    chunk_id: str
    subject: str | None
    qtype: str
    alt_doc_ids: tuple[str, ...] = field(default=())

    @property
    def gold_docs(self) -> frozenset[str]:
        return frozenset((self.doc_id, *self.alt_doc_ids))


def parse_item(raw: dict[str, object]) -> QAItem:
    alt = raw.get("alt_doc_ids") or []
    if not isinstance(alt, list):
        raise QAValidationError(f"{raw.get('qid')}: alt_doc_ids must be a list")
    subject = raw.get("subject")
    return QAItem(
        qid=str(raw["qid"]),
        question=str(raw["question"]),
        doc_id=str(raw["doc_id"]),
        chunk_id=str(raw["chunk_id"]),
        subject=None if subject is None else str(subject),
        qtype=str(raw["qtype"]),
        alt_doc_ids=tuple(str(a) for a in alt),
    )


def load_qa(path: Path) -> list[QAItem]:
    with path.open(encoding="utf-8") as fh:
        return [parse_item(json.loads(line)) for line in fh if line.strip()]


def validate_qa(items: Sequence[QAItem], chunks: pd.DataFrame) -> None:
    """Raise `QAValidationError` listing every problem: unknown ids, chunk/doc mismatch,
    duplicate qids, empty questions or unknown question types."""
    chunk_doc = dict(zip(chunks["chunk_id"], chunks["doc_id"], strict=True))
    doc_ids = set(chunk_doc.values())
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.qid in seen:
            problems.append(f"{item.qid}: duplicate qid")
        seen.add(item.qid)
        if not item.question.strip():
            problems.append(f"{item.qid}: empty question")
        if item.qtype not in QTYPES:
            problems.append(f"{item.qid}: unknown qtype {item.qtype!r}")
        if item.chunk_id not in chunk_doc:
            problems.append(f"{item.qid}: unknown chunk_id {item.chunk_id}")
        elif chunk_doc[item.chunk_id] != item.doc_id:
            problems.append(f"{item.qid}: chunk {item.chunk_id} is not in doc {item.doc_id}")
        problems.extend(
            f"{item.qid}: unknown doc_id {doc}" for doc in item.gold_docs if doc not in doc_ids
        )
    if problems:
        raise QAValidationError("; ".join(problems))


def qa_frame(items: Sequence[QAItem]) -> pd.DataFrame:
    """Flat table (alt ids joined by `;`) for logging as an MLflow dataset."""
    return pd.DataFrame(
        {
            "qid": [i.qid for i in items],
            "question": [i.question for i in items],
            "doc_id": [i.doc_id for i in items],
            "chunk_id": [i.chunk_id for i in items],
            "subject": [i.subject for i in items],
            "qtype": [i.qtype for i in items],
            "alt_doc_ids": [";".join(i.alt_doc_ids) for i in items],
        }
    )
