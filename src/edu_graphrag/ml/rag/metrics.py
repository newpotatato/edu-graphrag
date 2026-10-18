"""Ranking metrics with binary relevance.

A ranking is a list of retrieved chunks; the gold for a question is a *set* of doc ids (the
source document plus its byte-identical / near-duplicate copies). Every metric depends only
on the 1-based position of the first chunk whose document is in the gold set:

- recall@k: 1 if that position <= k (the "any of the top-k chunks comes from the gold doc").
- MRR@k:    1 / position, 0 beyond k.
- nDCG@k:   1 / log2(position + 1), 0 beyond k (one relevant target, so the ideal DCG is 1).
"""

import math
from collections.abc import Collection, Iterable, Sequence


def first_hit_rank(ranked: Iterable[str], gold: Collection[str]) -> int | None:
    """1-based position of the first item that is in `gold`, or None."""
    for position, item in enumerate(ranked, 1):
        if item in gold:
            return position
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def mrr_at_k(rank: int | None, k: int) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def ndcg_at_k(rank: int | None, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, `q` in [0, 100]; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
