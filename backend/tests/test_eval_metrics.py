import math

import numpy as np
import pytest

from detour.eval.metrics import (
    bootstrap_ci,
    interest_coverage,
    intra_list_diversity,
    mean_log_stars,
    mean_unfamiliarity,
    ndcg_at_k,
    owner_diversity,
    precision_at_k,
    recall_at_k,
    topic_entropy,
)


def test_accuracy_metrics_by_hand() -> None:
    ranked = [5, 1, 9, 2]
    rel = {1, 2, 7}
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert precision_at_k(ranked, rel, 4) == pytest.approx(2 / 4)
    # DCG = 1/log2(3) + 1/log2(5); IDCG (3 relevant, k=4) = 1 + 1/log2(3) + 1/log2(4)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(ranked, rel, 4) == pytest.approx(expected)
    assert ndcg_at_k([1, 2, 7], rel, 3) == pytest.approx(1.0)
    assert math.isnan(recall_at_k(ranked, set(), 3))


def test_ild_identical_vs_orthogonal() -> None:
    same = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    ortho = np.eye(3, dtype=np.float32)
    assert intra_list_diversity(same) == pytest.approx(0.0)
    assert intra_list_diversity(ortho) == pytest.approx(1.0)
    assert math.isnan(intra_list_diversity(ortho[:1]))


def test_list_property_metrics() -> None:
    assert interest_coverage(["ML", "ML", None], ["ML", "DS"]) == 0.5
    assert owner_diversity(["a", "A", "b", "c"]) == 0.75
    assert topic_entropy([["x"], ["x"]]) == 0.0
    assert topic_entropy([["x"], ["y"]]) == pytest.approx(1.0)
    assert mean_log_stars([9, 99]) == pytest.approx(1.5)
    known = np.array([[1.0, 0.0]], dtype=np.float32)
    lst = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert mean_unfamiliarity(lst, known) == pytest.approx(0.5)


def test_bootstrap_ci_contains_mean_and_drops_nan() -> None:
    mean, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, float("nan")])
    assert mean == pytest.approx(2.0)
    assert lo <= mean <= hi
    assert all(math.isnan(x) for x in bootstrap_ci([]))
