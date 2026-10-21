"""Builds the batch-source input: one JSON line {"id": ..., "text": ...} per task.

    python -m edu_graphrag.pipeline.make_batch --chunks data/processed/chunks.parquet \
        --out data/tasks/batch.jsonl --repeat 5

`--repeat` multiplies the corpus to get a long enough run for the experiment; each copy gets
its own id (`<chunk_id>#<copy>`), so they are distinct tasks with stable ids. Texts longer
than the message limit (MAX_TEXT_CHARS) or blank are skipped: the source would reject them.
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from edu_graphrag.pipeline.messages import MAX_TEXT_CHARS


def batch_records(
    chunks: Iterable[tuple[str, str]], *, repeat: int = 1
) -> Iterator[dict[str, str]]:
    """Yields task records; `chunks` must be re-iterable when repeat > 1 (a list or a frame)."""
    for copy in range(repeat):
        for chunk_id, text in chunks:
            if not text or not text.strip() or len(text) > MAX_TEXT_CHARS:
                continue
            yield {"id": f"{chunk_id}#{copy}", "text": text}


def read_chunks(path: Path) -> list[tuple[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["chunk_id", "text"])
    return [(str(chunk_id), str(text)) for chunk_id, text in frame.itertuples(index=False)]


def write_jsonl(records: Iterable[dict[str, str]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a JSONL batch of tasks from chunks")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/tasks/batch.jsonl"))
    parser.add_argument("--repeat", type=int, default=1, help="copies of the corpus (>= 1)")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    chunks = read_chunks(args.chunks)
    n = write_jsonl(batch_records(chunks, repeat=args.repeat), args.out)
    print(f"wrote {n} tasks ({len(chunks)} chunks x {args.repeat}) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
