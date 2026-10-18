"""Reciprocal rank fusion (Cormack et al., 2009)."""

from collections.abc import Hashable, Sequence

RRF_K = 60


def reciprocal_rank_fusion[T: Hashable](
    rankings: Sequence[Sequence[T]], k: int = RRF_K
) -> list[tuple[T, float]]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank(d)), rank is 1-based.

    Returns (item, score) sorted by score descending; ties keep first-seen order.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
