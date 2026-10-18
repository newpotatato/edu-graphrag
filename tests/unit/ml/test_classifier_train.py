"""End-to-end: synthetic processed dataset -> train -> register -> load like the API."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edu_graphrag.components.mlflow_model import MlflowModelRegistry
from edu_graphrag.ml.classifier import register, train

VOCAB = {
    "algebra": "matrix vector eigenvalue determinant basis kernel rank linear",
    "topology": "open closed neighbourhood compact hausdorff continuous homeomorphism",
    "stats": "probability variance estimator sample hypothesis likelihood bayes",
}
COMMON = ["the", "of", "and", "is", "we", "let", "then", "proof"]


def _write_processed(root: Path) -> Path:
    rng = np.random.default_rng(0)
    docs, chunks = [], []

    def add_doc(doc_id: str, subject: str, content_hash: str, n_chunks: int, status: str) -> None:
        docs.append(
            {
                "doc_id": doc_id,
                "source_uri": f"1_grade/1_sem/{subject}/{doc_id}.md",
                "source_type": "md",
                "content_hash": content_hash,
                "grade": 1,
                "semester": 1.0,
                "subject": subject,
                "is_sorted": True,
                "n_chars": 100 * n_chunks,
                "n_chunks": n_chunks,
                "status": status,
                "error": None,
            }
        )
        words = VOCAB.get(subject, "rare words only here").split() + COMMON
        for i in range(n_chunks):
            text = " ".join(rng.choice(words, size=30))
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-{i}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": text,
                    "n_chars": len(text),
                    "source_uri": f"{doc_id}.md",
                    "source_type": "md",
                    "grade": 1,
                    "semester": 1.0,
                    "subject": subject,
                    "is_sorted": True,
                }
            )

    for subject in VOCAB:
        for d in range(5):
            add_doc(f"{subject}{d}", subject, f"h-{subject}{d}", 6, "ok")
    add_doc("copy", "algebra", "h-algebra0", 6, "ok")  # byte-identical duplicate
    add_doc("rare0", "rare", "h-rare0", 4, "ok")  # dropped: one document
    add_doc("broken", "stats", "h-broken", 0, "failed")

    processed = root / "processed"
    processed.mkdir()
    pd.DataFrame(docs).to_parquet(processed / "documents.parquet")
    pd.DataFrame(chunks).to_parquet(processed / "chunks.parquet")
    manifest = {"dataset_version": "synthetic01", "n_documents": len(docs)}
    (processed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return processed


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)  # the sqlite store keeps artifacts in ./mlruns
    # Only the env variable: `mlflow.set_tracking_uri` would leak into later tests.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def test_train_register_and_serve(
    tmp_path: Path, tracking_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlflow
    from mlflow import MlflowClient

    processed = _write_processed(tmp_path)
    for variant in ("baseline", "word_balanced"):
        assert train.main(["--variant", variant, "--data", str(processed)]) == 0
    out = capsys.readouterr().out
    assert "dropped subjects (< 3 docs): {'rare': 1}" in out
    assert "test_macro_f1=" in out

    runs = register.search_runs("synthetic01")
    assert len(runs) == 2
    run = mlflow.get_run(runs.iloc[0]["run_id"])
    params, metrics = run.data.params, run.data.metrics
    assert params["dataset_version"] == "synthetic01"
    assert params["n_duplicate_docs_removed"] == "1"
    assert params["dropped_subjects"] == "rare"
    assert int(params["n_test_docs"]) == 3
    assert run.data.tags["dataset_version"] == "synthetic01"
    for name in ("test_macro_f1", "test_doc_accuracy", "train_macro_f1", "predict_ms_per_chunk"):
        assert name in metrics
    assert run.inputs is not None
    contexts = sorted(d.tags[0].value for d in run.inputs.dataset_inputs)
    assert contexts == ["evaluation", "training"]
    artifacts = {a.path for a in MlflowClient().list_artifacts(run.info.run_id, "reports")}
    for name in ("confusion_matrix.png", "classification_report.csv", "misclassified.csv"):
        assert f"reports/{name}" in artifacts

    assert register.main([]) == 0
    aliases = register.register()  # second call must not create new versions
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{register.MODEL_NAME}'")
    assert len(versions) == 2
    assert set(aliases.values()) <= {"1", "2"}
    baseline_version = client.get_model_version(register.MODEL_NAME, "1")
    assert baseline_version.tags["variant"] == "baseline"
    assert "test_macro_f1" in baseline_version.tags

    # Load exactly like the API does.
    registry = MlflowModelRegistry(tracking_uri)
    try:
        ref = registry.resolve(register.MODEL_NAME, register.CHAMPION)
        assert ref.version == aliases[register.CHAMPION]
        model = registry.load(ref)
        frame = model.predict(pd.DataFrame({"text": ["eigenvalue of the matrix kernel"]}))
    finally:
        registry.close()
    assert frame["label"].tolist() == ["algebra"]
    assert frame["top3"][0][0]["label"] == "algebra"


def _runs(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["run_id", "tags.variant", "metrics.test_macro_f1", "start_time"]
    )
    frame["tags.dataset_version"] = "v1"
    return frame


def test_select_runs() -> None:
    selection = register.select_runs(
        _runs([("r2", "balanced", 0.8, 2), ("r1", "baseline", 0.6, 1), ("r0", "baseline", 0.5, 3)])
    )
    assert selection.best["run_id"] == "r2"
    assert selection.baseline["run_id"] == "r0"  # latest baseline run
    assert selection.challenger["run_id"] == "r0"

    selection = register.select_runs(
        _runs([("r1", "baseline", 0.9, 1), ("r2", "svm", 0.7, 2), ("r3", "svm", 0.6, 3)])
    )
    assert selection.best["run_id"] == "r1"
    assert selection.challenger["run_id"] == "r2"  # second best when baseline is best

    alone = register.select_runs(_runs([("r1", "baseline", 0.9, 1)]))
    assert alone.challenger["run_id"] == "r1"


@pytest.mark.parametrize(
    ("rows", "message"),
    [([], "no finished runs"), ([("r1", "svm", 0.9, 1)], "no 'baseline' run")],
)
def test_select_runs_errors(rows: list[tuple[str, str, float, int]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        register.select_runs(_runs(rows))


def test_git_commit_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    assert train.git_commit()  # real repo or "unknown"
    monkeypatch.setattr("shutil.which", lambda _: "git")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("subprocess.run", fail)
    assert train.git_commit() == "unknown"
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert train.git_commit() == "unknown"
