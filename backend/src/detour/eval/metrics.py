"""Offline evaluation metrics. Pure functions over ids, vectors and metadata.

Accuracy (against held-out relevant items H)
  recall@k      |top_k ∩ H| / |H|
  precision@k   |top_k ∩ H| / k
  ndcg@k        binary-relevance NDCG
Beyond accuracy (properties of the list itself; NOT evidence of quality on their own,
because MMR/novelty optimize them directly; see docs/evaluation.md §1)
  ild           mean pairwise (1 - cosine) within the list
  coverage      fraction of the user's interests matched by >= 1 listed item
  owner_div     unique owners / list length
  topic_entropy Shannon entropy (bits) of meaningful topics in the list
  mean_log_stars popularity of the list
  unfamiliarity mean (1 - max cosine to the user's known repos)
Serendipity proxy
  serendipity@k |top_k ∩ H_ser| / k, where H_ser are held-out items that are NOT
                near-duplicates of anything the user already knew (max cos to known < 0.6)
"""

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from detour.representation.embedder import Vectors


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal


def intra_list_diversity(vectors: Vectors) -> float:
    n = vectors.shape[0]
    if n < 2:
        return float("nan")
    sims = vectors @ vectors.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))


def interest_coverage(matched: Sequence[str | None], interests: Sequence[str]) -> float:
    if not interests:
        return float("nan")
    return len({m for m in matched if m} & set(interests)) / len(interests)


def owner_diversity(owners: Sequence[str]) -> float:
    if not owners:
        return float("nan")
    return len({o.lower() for o in owners}) / len(owners)


def topic_entropy(topic_lists: Sequence[Sequence[str]]) -> float:
    counts = Counter(t for topics in topic_lists for t in topics)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def mean_log_stars(stars: Sequence[int]) -> float:
    if not stars:
        return float("nan")
    return float(np.mean(np.log10(np.asarray(stars, dtype=float) + 1)))


def mean_unfamiliarity(vectors: Vectors, known: Vectors) -> float:
    if vectors.shape[0] == 0 or known.shape[0] == 0:
        return float("nan")
    return float(np.mean(1.0 - (vectors @ known.T).max(axis=1)))


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over users (NaNs dropped)."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
