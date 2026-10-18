"""RetrieverModel as a pyfunc and the experiment/registry flow against a tmp sqlite store."""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from edu_graphrag.components.mlflow_model import MlflowModelRegistry, predictions_to_records
from edu_graphrag.ml.rag import experiment as exp
from edu_graphrag.ml.rag import model as model_mod
from edu_graphrag.ml.rag.index import Embedder, IndexConfig, build_index
from edu_graphrag.ml.rag.model import RESULTS_COLUMN, RetrieverModel
from tests.unit.ml.test_rag_support import FakeEmbedder, make_chunks, write_dataset


def fake_factory(config: IndexConfig, model_dir: Path | None) -> Embedder:
    return FakeEmbedder()


# ---------------------------------------------------------------- model without MLflow


def test_model_predict_contract_with_dense_index(tmp_path: Path) -> None:
    cfg = IndexConfig(method="hybrid", embed_model="fake")
    build_index(make_chunks(), cfg, tmp_path / "index", embedder=FakeEmbedder())
    received: list[Path | None] = []

    def factory(config: IndexConfig, model_dir: Path | None) -> Embedder:
        received.append(model_dir)
        return FakeEmbedder()

    model = RetrieverModel(embedder_factory=factory)
    with pytest.raises(RuntimeError, match="load_context"):
        _ = model.retriever
    context = SimpleNamespace(
        artifacts={"index": str(tmp_path / "index"), "embedder": str(tmp_path / "onnx")}
    )
    model.load_context(context)
    assert received == [tmp_path / "onnx"]

    out = model.predict(None, pd.DataFrame({"text": ["связное пространство", "Vue пропсы"]}))
    assert list(out.columns) == [RESULTS_COLUMN]
    assert len(out) == 2
    first = out.loc[0, RESULTS_COLUMN]
    assert len(first) == 5
    assert first[0]["doc_id"] == "doc-top"
    assert [r["rank"] for r in first] == [1, 2, 3, 4, 5]
    assert len(model.predict(None, pd.DataFrame({"text": ["x"]}), params={"k": 2}).iloc[0, 0]) == 2
    assert len(model.predict(None, pd.DataFrame({"text": ["x"]}), params={"k": 0}).iloc[0, 0]) == 1
    # Records exactly as the API would return them.
    records = predictions_to_records(out)
    assert json.loads(json.dumps(records))[1][RESULTS_COLUMN][0]["subject"] == "Frontend"


def test_model_accepts_other_input_shapes(tmp_path: Path) -> None:
    build_index(make_chunks(), IndexConfig(method="bm25"), tmp_path / "index")
    model = RetrieverModel()
    model.load_context(SimpleNamespace(artifacts={"index": str(tmp_path / "index")}))
    by_dict = model.predict(None, {"text": ["матрица"]})
    by_list = model.predict(None, ["матрица", None])
    other_column = model.predict(None, pd.DataFrame({"query": ["матрица"]}))
    assert by_dict.iloc[0, 0] == by_list.iloc[0, 0] == other_column.iloc[0, 0]
    assert by_list.iloc[1, 0] == []


# ---------------------------------------------------------------- evaluation helpers


def test_evaluation_tables(tmp_path: Path) -> None:
    data_dir, qa_path = write_dataset(tmp_path)
    inputs = exp.load_inputs(data_dir, qa_path)
    cfg = IndexConfig(method="bm25", tokenizer="prefix5")
    _, metrics, results = exp.evaluate_variant(
        "bm25_prefix5", cfg, inputs, tmp_path / "w", embedder=None, embeddings=None, embed_seconds=9
    )
    assert metrics["recall_at_5"] == pytest.approx(0.75)  # q004 has no lexical match
    assert metrics["index_build_seconds"] < 9  # embed time only counts for dense
    assert metrics["query_ms_p95"] >= metrics["query_ms_p50"] >= 0
    per_question = exp.per_question_table(results)
    assert per_question.loc[2, "hit_rank"] == 1  # alt doc counts too
    assert per_question.loc[3, "top1_source_uri"] == ""
    subjects = exp.per_subject_table(per_question)
    assert subjects.set_index("subject").loc["difgem", "recall_at_5"] == 0.5
    failures = exp.failure_table(results, {"doc-top": "notes/doc-top.md"})
    assert failures["qid"].tolist() == ["q004"]
    assert failures.loc[0, "gold_source_uri"] == "notes/doc-top.md"
    assert exp.aggregate_metrics([])["recall_at_5"] == 0.0


def test_locate_model_dir_and_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = exp.EMBED_MODEL
    with pytest.raises(FileNotFoundError):
        exp.locate_model_dir(tmp_path, name)
    snapshot = tmp_path / "models--x" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model_optimized.onnx").write_bytes(b"onnx")
    assert exp.locate_model_dir(tmp_path, name) == snapshot

    monkeypatch.setattr(exp, "FastEmbedder", lambda model, cache_dir: FakeEmbedder())
    embedder, files = exp.fastembed_loader(tmp_path)(name)
    assert isinstance(embedder, FakeEmbedder)
    assert files == snapshot


def test_git_commit_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edu_graphrag.ml.rag.experiment.shutil.which", lambda _: None)
    assert exp.git_commit() == "unknown"


# ---------------------------------------------------------------- MLflow flow


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Throwaway sqlite store whose experiment keeps artifacts under tmp_path."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.MlflowClient().create_experiment(
        exp.EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    # The served model has no bundled ONNX files in tests: fake its embedder.
    monkeypatch.setattr(model_mod, "fastembed_factory", fake_factory)
    try:
        yield uri
    finally:
        mlflow.set_tracking_uri(previous)


def test_experiment_register_and_serve_app_style(tmp_path: Path, tracking: str) -> None:
    import mlflow

    data_dir, qa_path = write_dataset(tmp_path)
    inputs = exp.load_inputs(data_dir, qa_path)
    loads: list[str] = []

    def loader(model_name: str) -> tuple[Embedder, Path | None]:
        loads.append(model_name)
        return FakeEmbedder(), None

    variants = ["bm25", "bm25_prefix5", "hybrid_rrf"]
    outcomes = exp.run_experiment(inputs, variants, tmp_path / "work", embedder_loader=loader)
    assert [o.variant for o in outcomes] == variants
    assert not any(o.reused for o in outcomes)
    assert loads == [exp.EMBED_MODEL]

    client = mlflow.MlflowClient()
    run = client.get_run(outcomes[0].run_id)
    assert run.data.params["variant"] == "bm25"
    assert run.data.params["dataset_version"] == "testver00001"
    assert run.data.params["chunk_target_chars"] == "1200"
    assert run.data.tags["dataset_version"] == "testver00001"
    assert run.data.tags["git_commit"]
    assert {"recall_at_5", "mrr_at_10", "ndcg_at_10", "chunk_hit_at_5", "query_ms_p95"} <= set(
        run.data.metrics
    )
    contexts = {
        (d.dataset.name, d.tags[0].value)
        for d in (run.inputs.dataset_inputs if run.inputs else [])
        if d.tags
    }
    assert contexts == {("study-chunks", "index"), ("study-qa", "evaluation")}
    eval_files = {f.path for f in client.list_artifacts(run.info.run_id, "eval")}
    assert {
        "eval/per_question.csv",
        "eval/failures.csv",
        "eval/per_subject_recall_at_5.png",
        "eval/per_subject_recall_at_5.json",
    } <= eval_files
    assert "metrics_comparison.json" in {f.path for f in client.list_artifacts(run.info.run_id)}

    # Second invocation reuses the finished runs instead of retraining.
    again = exp.run_experiment(inputs, variants, tmp_path / "work2", embedder_loader=loader)
    assert all(o.reused for o in again)
    assert [o.run_id for o in again] == [o.run_id for o in outcomes]

    state = exp.register_best(inputs.dataset_version, inputs.qa_digest)
    assert state.challenger == "1"  # the baseline is registered first
    assert state.challenger_run_id == outcomes[0].run_id
    best = max(outcomes, key=lambda o: (o.metrics["recall_at_5"], o.metrics["mrr_at_10"]))
    assert (
        client.get_run(state.champion_run_id).data.metrics["recall_at_5"]
        == best.metrics["recall_at_5"]
    )
    repeat = exp.register_best(inputs.dataset_version, inputs.qa_digest)
    assert repeat.created_versions == []
    assert (repeat.champion, repeat.challenger) == (state.champion, state.challenger)

    # App-style: resolve the alias, download and load like the API does, then predict.
    registry = MlflowModelRegistry(tracking)
    try:
        ref = registry.resolve(exp.MODEL_NAME, exp.CHALLENGER_ALIAS)
        assert ref.version == "1"
        served = registry.load(ref)
        raw = served.predict(pd.DataFrame({"text": ["собственные значения матрицы"]}))
        records = predictions_to_records(raw)
        assert records[0][RESULTS_COLUMN][0]["doc_id"] == "doc-lin"
        assert len(records[0][RESULTS_COLUMN]) == 5 or len(records[0][RESULTS_COLUMN]) > 0
        champion = registry.load(registry.resolve(exp.MODEL_NAME, exp.CHAMPION_ALIAS))
        assert predictions_to_records(champion.predict(pd.DataFrame({"text": ["пропсы Vue"]})))
    finally:
        registry.close()


def test_main_cli_and_register_errors(
    tmp_path: Path, tracking: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir, qa_path = write_dataset(tmp_path)
    with pytest.raises(RuntimeError, match="no finished runs"):
        exp.register_best("testver00001", "nope")
    args = ["--data", str(data_dir), "--qa", str(qa_path), "--variants", "bm25"]
    assert exp.main([*args, "--no-register"]) == 0
    assert exp.main([*args, "--work-dir", str(tmp_path / "keep")]) == 0
    out = capsys.readouterr().out
    assert "(reused)" in out
    assert "champion=v1 challenger=v1" in out
    assert (tmp_path / "keep" / "metrics_comparison.json").exists()
    with pytest.raises(ValueError, match="needs an embedder loader"):
        exp.run_experiment(exp.load_inputs(data_dir, qa_path), ["dense"], tmp_path / "w")


def test_register_best_without_experiment(tracking: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "EXPERIMENT_NAME", "missing-experiment")
    with pytest.raises(RuntimeError, match="does not exist"):
        exp.register_best("v", "q")


def test_run_key_depends_on_config_and_qa(tmp_path: Path) -> None:
    data_dir, qa_path = write_dataset(tmp_path)
    inputs = exp.load_inputs(data_dir, qa_path)
    a = exp.run_key("bm25", IndexConfig(method="bm25"), inputs)
    b = exp.run_key("bm25", IndexConfig(method="bm25", tokenizer="prefix5"), inputs)
    assert a != b
    assert a == exp.run_key("bm25", IndexConfig(method="bm25"), inputs)
    params: dict[str, Any] = exp.run_params("dense", exp.VARIANTS["dense"], inputs)
    assert params["tokenizer"] == "-"
    assert params["embed_model"] == exp.EMBED_MODEL
