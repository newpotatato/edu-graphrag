"""Tokenizers, ranking metrics and reciprocal rank fusion."""

import math

import pytest

from edu_graphrag.ml.rag.fusion import reciprocal_rank_fusion
from edu_graphrag.ml.rag.metrics import (
    first_hit_rank,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from edu_graphrag.ml.rag.tokenizer import get_tokenizer, prefix_tokenize, tokenize


def test_tokenize_handles_cyrillic_latin_and_noise() -> None:
    text = "Матрица ГРАМА: x_1 + l2-норма, ёмкость (2026)!"
    assert tokenize(text) == ["матрица", "грама", "l2", "норма", "емкость", "2026"]


def test_tokenize_drops_single_chars_and_underscores() -> None:
    assert tokenize("a b _ __init__ я") == ["init"]
    assert tokenize("") == []


def test_prefix_tokenizer_merges_inflections() -> None:
    assert prefix_tokenize("матрица матрицы матрицей", 5) == ["матри"] * 3
    assert get_tokenizer("prefix5")("Матрицами") == ["матри"]
    assert get_tokenizer("word")("Матрицами") == ["матрицами"]


@pytest.mark.parametrize("name", ["stem", "prefix0", "prefix", "prefixX"])
def test_unknown_tokenizer_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer(name)


def test_first_hit_rank_on_hand_made_ranking() -> None:
    ranking = ["d3", "d1", "d1", "d2"]
    assert first_hit_rank(ranking, {"d1"}) == 2
    assert first_hit_rank(ranking, {"d2", "d9"}) == 4
    assert first_hit_rank(ranking, {"d9"}) is None
    assert first_hit_rank([], {"d1"}) is None


@pytest.mark.parametrize(
    ("rank", "recall1", "recall5", "mrr", "ndcg"),
    [
        (1, 1.0, 1.0, 1.0, 1.0),
        (3, 0.0, 1.0, 1 / 3, 1 / math.log2(4)),
        (7, 0.0, 0.0, 1 / 7, 1 / math.log2(8)),
        (11, 0.0, 0.0, 0.0, 0.0),
        (None, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_rank_metrics(
    rank: int | None, recall1: float, recall5: float, mrr: float, ndcg: float
) -> None:
    assert recall_at_k(rank, 1) == recall1
    assert recall_at_k(rank, 5) == recall5
    assert mrr_at_k(rank, 10) == pytest.approx(mrr)
    assert ndcg_at_k(rank, 10) == pytest.approx(ndcg)


def test_mean_metrics_over_rankings() -> None:
    gold = [{"a"}, {"b"}, {"c"}]
    rankings = [["a", "x"], ["x", "y", "b"], ["x", "y"]]
    ranks = [first_hit_rank(r, g) for r, g in zip(rankings, gold, strict=True)]
    assert sum(recall_at_k(r, 5) for r in ranks) / 3 == pytest.approx(2 / 3)
    assert sum(mrr_at_k(r, 10) for r in ranks) / 3 == pytest.approx((1 + 1 / 3) / 3)


def test_percentile_interpolates() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([4.0, 1.0, 3.0, 2.0], 100) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0


def test_rrf_fuses_and_orders() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    scores = dict(fused)
    assert [item for item, _ in fused][:2] == ["b", "a"]
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert set(scores) == {"a", "b", "c", "d"}


def test_rrf_edge_cases() -> None:
    assert reciprocal_rank_fusion([]) == []
    [(item, score)] = reciprocal_rank_fusion([[], ["x"]])
    assert item == "x"
    assert score == pytest.approx(1 / 61)
    # Equal scores keep first-seen order.
    assert [i for i, _ in reciprocal_rank_fusion([["p"], ["q"]])] == ["p", "q"]
