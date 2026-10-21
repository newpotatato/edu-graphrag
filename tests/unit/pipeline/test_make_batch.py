import json
from pathlib import Path

import pandas as pd
import pytest

from edu_graphrag.pipeline.make_batch import batch_records, main
from edu_graphrag.pipeline.source import iter_tasks


def test_batch_records_repeat_and_filter() -> None:
    chunks = [("c1", "alpha"), ("c2", "x" * 20_001), ("c3", "   "), ("c4", "beta")]

    records = list(batch_records(chunks, repeat=2))

    assert records == [
        {"id": "c1#0", "text": "alpha"},
        {"id": "c4#0", "text": "beta"},
        {"id": "c1#1", "text": "alpha"},
        {"id": "c4#1", "text": "beta"},
    ]


def test_cli_writes_jsonl_the_source_can_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks = tmp_path / "chunks.parquet"
    pd.DataFrame(
        {"chunk_id": ["a", "b"], "text": ["first", "second"], "subject": ["m", "p"]}
    ).to_parquet(chunks)
    out = tmp_path / "tasks" / "batch.jsonl"

    assert main(["--chunks", str(chunks), "--out", str(out), "--repeat", "3"]) == 0

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 6
    assert lines[0] == {"id": "a#0", "text": "first"}
    assert "wrote 6 tasks" in capsys.readouterr().out
    ids = [m.task_id for m in iter_tasks(out)]
    assert len(set(ids)) == 6, "repeated copies are distinct tasks"


def test_cli_rejects_bad_repeat(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--chunks", str(tmp_path / "x.parquet"), "--repeat", "0"])
