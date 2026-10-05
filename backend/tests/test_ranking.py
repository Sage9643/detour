"""Deterministic ranking tests on hand-built vectors.

Each test encodes one claim from docs/recommendation-engine.md with numbers small enough
to verify by hand.
"""

import numpy as np
import pytest

from detour.enums import RankerVariant
from detour.ranking.config import RankerConfig
from detour.ranking.engine import rank, rank_normalize

from .ranking_helpers import item, profile, repo, vec

V = RankerVariant
U = vec(1, 0, 0, 0)  # the single interest direction in most tests


def fillers(start_id: int, n: int = 7) -> list:
    """Moderately relevant items (cos 0.30..0.42 with U), all pointing the same way."""
    out = []
    for i in range(n):
        x = 0.30 + 0.02 * i
        out.append(item(repo(start_id + i), vec(x, 0, 0, float(np.sqrt(1 - x * x)))))
    return out


# --- relevance-only vs diversity ------------------------------------------------------


def mmr_pool() -> list:
    a = item(repo(1), vec(0.9, 0.436, 0, 0))  # rel 0.90
    b = item(repo(2), vec(0.88, 0.475, 0, 0))  # rel 0.88, cos(a,b) = 0.999 (near-duplicate)
    c = item(repo(3), vec(0.85, -0.527, 0, 0))  # rel 0.85, cos(a,c) = 0.535
    return [a, b, c, *fillers(10)]


def test_relevance_only_baseline_keeps_the_near_duplicate() -> None:
    res = rank(mmr_pool(), profile([("ML", U)]), V.B1_RELEVANCE, RankerConfig(k=2))
    assert [s.repo_id for s in res.shown] == [1, 2]


def test_mmr_picks_dissimilar_item_over_near_duplicate() -> None:
    # Second pick: B = 0.7*0.889 - 0.3*0.999 = 0.323 ; C = 0.7*0.778 - 0.3*0.535 = 0.384
    res = rank(mmr_pool(), profile([("ML", U)]), V.D2_RELEVANCE_MMR, RankerConfig(k=2))
    assert [s.repo_id for s in res.shown] == [1, 3]
    c = res.shown[1]
    assert c.diversity_promoted is True
    assert c.max_sim_to_selected == pytest.approx(0.535, abs=0.01)
    assert res.shown[0].diversity_promoted is False


def test_mmr_lambda_one_reduces_to_utility_order() -> None:
    res = rank(
        mmr_pool(), profile([("ML", U)]), V.D2_RELEVANCE_MMR, RankerConfig(k=2, mmr_lambda=1.0)
    )
    assert [s.repo_id for s in res.shown] == [1, 2]


def test_owner_cap_limits_repos_per_owner() -> None:
    same = [item(repo(i, owner="bigco"), vec(1.0 - 0.01 * i, 0.1 * i, 0, 0)) for i in range(1, 6)]
    other = [item(repo(20 + i), vec(0.5, 0, 0.866 - 0.01 * i, 0)) for i in range(3)]
    res = rank(same + other, profile([("ML", U)]), V.D3_FULL, RankerConfig(k=5, owner_cap=2))
    owners = [s.item.candidate.repo.owner_login for s in res.shown]
    assert owners.count("bigco") == 2
    # Relevance-only has no cap.
    b1 = rank(same + other, profile([("ML", U)]), V.B1_RELEVANCE, RankerConfig(k=5))
    assert [s.item.candidate.repo.owner_login for s in b1.shown].count("bigco") == 5


# --- relevance gate: novelty cannot rescue irrelevance --------------------------------------


def test_gate_excludes_novel_but_irrelevant_item() -> None:
    relevant = item(repo(1, stars=50_000, topics=["machine-learning"]), vec(0.8, 0.6, 0, 0))
    # Maximally novel: tiny, unknown topic, but cos(U) = 0.1 < tau = 0.18
    irrelevant = item(repo(2, stars=25, topics=["knitting"]), vec(0.1, 0, 0.995, 0))
    p = profile([("ML", U)], known_topics=["machine-learning"])
    cfg = RankerConfig(k=10, beta_novelty=100.0)  # absurd novelty weight on purpose
    for variant in (V.D1_RELEVANCE_NOVELTY, V.D3_FULL):
        res = rank([relevant, irrelevant], p, variant, cfg)
        assert [s.repo_id for s in res.shown] == [1]
        gated = next(s for s in res.scored if s.repo_id == 2)
        assert gated.filter_reason == "below_relevance_gate"
        assert gated.novelty is not None and gated.novelty > 0.9  # novel, still excluded


def test_novelty_reorders_among_relevant_items() -> None:
    famous = item(repo(1, stars=80_000, topics=["machine-learning"]), vec(0.9, 0.436, 0, 0))
    niche = item(repo(2, stars=150, topics=["federated-learning"]), vec(0.88, 0, 0.475, 0))
    p = profile([("ML", U)], known_topics=["machine-learning"])
    b1 = rank([famous, niche], p, V.B1_RELEVANCE, RankerConfig(k=1))
    d1 = rank([famous, niche], p, V.D1_RELEVANCE_NOVELTY, RankerConfig(k=1, beta_novelty=1.5))
    assert b1.shown[0].repo_id == 1
    # utility famous = 1 + 1.5*mean(0 topical, 0 pop) = 1.0 ; niche = 0 + 1.5*mean(1,1) = 1.5
    assert d1.shown[0].repo_id == 2


# --- familiarity vs preference -------------------------------------------------------------


def test_already_known_lowers_unfamiliarity_not_relevance() -> None:
    target = item(repo(1), vec(0.9, 0.436, 0, 0))
    lookalike_known = (99, vec(0.88, 0.475, 0, 0))  # near-duplicate of target
    cold = rank([target], profile([("ML", U)]), V.D3_FULL, RankerConfig())
    warm = rank([target], profile([("ML", U)], known=[lookalike_known]), V.D3_FULL, RankerConfig())
    c, w = cold.scored[0], warm.scored[0]
    assert c.relevance_raw == pytest.approx(w.relevance_raw)  # preference untouched
    assert c.negative_penalty == w.negative_penalty == 0.0
    assert c.unfamiliarity is None  # undefined at cold start: no familiarity data
    assert w.unfamiliarity is not None and w.unfamiliarity < 0.01  # familiar now


def test_known_repo_itself_is_filtered_not_penalized() -> None:
    target = item(repo(1), vec(0.9, 0.436, 0, 0))
    res = rank(
        [target],
        profile([("ML", U)], known=[(1, vec(0.9, 0.436, 0, 0))]),
        V.D3_FULL,
        RankerConfig(),
    )
    assert res.scored[0].filter_reason == "already_known"
    assert res.shown == []


def test_negative_feedback_penalizes_near_duplicates_only() -> None:
    near_rejected = item(repo(1), vec(0.9, 0.436, 0, 0))
    same_interest_different = item(repo(2), vec(0.85, -0.527, 0, 0))
    rejected = (77, vec(0.88, 0.475, 0, 0))  # cos with repo 1 = 0.999, with repo 2 = 0.50
    pool = [near_rejected, same_interest_different, *fillers(10)]
    p = profile([("ML", U)], negatives=[rejected])
    res = rank(pool, p, V.D3_FULL, RankerConfig(k=3))
    by_id = {s.repo_id: s for s in res.scored}
    assert by_id[1].negative_penalty == pytest.approx(0.999, abs=0.01)
    assert by_id[1].nearest_negative == "disliked/r77"
    assert by_id[2].negative_penalty == 0.0  # below theta=0.6: the interest is not punished
    # utility(1) = 1.0 + 0.3*0.5 - 0.999 = 0.151 ; utility(2) = 0.889 + 0.15 = 1.039
    assert by_id[1].utility == pytest.approx(0.151, abs=0.01)
    assert res.shown[0].repo_id == 2
    assert by_id[1].pre_rerank_rank is not None and by_id[1].pre_rerank_rank > 5
    # Variants without personalization ignore negative feedback.
    d2 = rank(pool, p, V.D2_RELEVANCE_MMR, RankerConfig(k=3))
    assert d2.shown[0].repo_id == 1


def test_positive_exemplar_adds_relevance_in_full_variant_only() -> None:
    # Weak interest match (0.15 < tau) but very similar to a repo the user liked.
    candidate = item(repo(1), vec(0.15, 0, 0.989, 0))
    liked = (55, vec(0.1, 0, 0.995, 0))
    p = profile([("ML", U)], positives=[liked])
    d3 = rank([candidate], p, V.D3_FULL, RankerConfig())
    s = d3.scored[0]
    assert s.matched_via_exemplar == "liked/r55"
    assert s.relevance_raw == pytest.approx(0.5 * 0.999, abs=0.01)  # alpha * cos
    assert d3.shown and d3.shown[0].repo_id == 1
    d1 = rank([candidate], p, V.D1_RELEVANCE_NOVELTY, RankerConfig())
    assert d1.scored[0].filter_reason == "below_relevance_gate"


def test_interest_weights_apply_only_to_full_variant() -> None:
    ml, ds = vec(1, 0, 0, 0), vec(0, 1, 0, 0)
    a = item(repo(1), vec(0.9, 0.3, 0, 0.316))  # mostly ML
    b = item(repo(2), vec(0.3, 0.85, 0, 0.433))  # mostly DS
    p = profile([("ML", ml, 0.5), ("DS", ds, 1.0)])
    assert rank([a, b], p, V.B1_RELEVANCE, RankerConfig(k=1)).shown[0].repo_id == 1
    d3 = rank([a, b], p, V.D3_FULL, RankerConfig(k=1))
    assert d3.shown[0].repo_id == 2
    assert d3.scored[0].matched_interest == "ML"
    assert d3.scored[0].relevance_raw == pytest.approx(0.45, abs=0.01)


# --- filters, baselines, determinism -----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"archived": True}, "archived"),
        ({"fork": True}, "fork"),
        ({"description": None}, "no_text"),
        ({"stars": 3}, "below_star_floor"),
    ],
)
def test_filters_apply_to_every_variant(kwargs: dict, reason: str) -> None:
    bad = item(repo(1, **kwargs), vec(1, 0, 0, 0))
    for variant in V:
        res = rank([bad], profile([("ML", U)]), variant, RankerConfig())
        assert res.scored[0].filter_reason == reason
        assert res.shown == []


def test_recently_shown_items_are_excluded() -> None:
    res = rank(
        [item(repo(1), vec(1, 0, 0, 0)), item(repo(2), vec(0.9, 0.436, 0, 0))],
        profile([("ML", U)], recently_shown=[1]),
        V.D3_FULL,
        RankerConfig(),
    )
    assert [s.repo_id for s in res.shown] == [2]


def test_popularity_baseline_sorts_by_stars_without_gate() -> None:
    pool = [
        item(repo(1, stars=10_000), vec(0.05, 0.998, 0, 0)),  # irrelevant but popular
        item(repo(2, stars=500), vec(1, 0, 0, 0)),
    ]
    res = rank(pool, profile([("ML", U)]), V.B0_POPULARITY, RankerConfig(k=2))
    assert [s.repo_id for s in res.shown] == [1, 2]


def test_ranking_is_deterministic_with_ties_broken_by_id() -> None:
    pool = [item(repo(i), vec(1, 0, 0, 0)) for i in (5, 3, 9, 1)]
    for variant in V:
        first = [s.repo_id for s in rank(pool, profile([("ML", U)]), variant, RankerConfig()).shown]
        again = [s.repo_id for s in rank(pool, profile([("ML", U)]), variant, RankerConfig()).shown]
        assert first == again
    b1 = rank(pool, profile([("ML", U)]), V.B1_RELEVANCE, RankerConfig())
    assert [s.repo_id for s in b1.shown] == [1, 3, 5, 9]


def test_rank_normalize_handles_ties_and_edge_cases() -> None:
    assert rank_normalize(np.array([])).shape == (0,)
    assert rank_normalize(np.array([7.0])).tolist() == [1.0]
    assert rank_normalize(np.array([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]
    assert rank_normalize(np.array([1.0, 1.0, 5.0])).tolist() == [0.25, 0.25, 1.0]


def test_fewer_eligible_than_k_returns_fewer() -> None:
    res = rank(
        [item(repo(1), vec(1, 0, 0, 0))], profile([("ML", U)]), V.D3_FULL, RankerConfig(k=10)
    )
    assert len(res.shown) == 1
    assert res.eligible_count == 1


def test_empty_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="no interests"):
        rank([], profile([]), V.D3_FULL, RankerConfig())


def test_shown_positions_are_contiguous_and_logged() -> None:
    res = rank(mmr_pool(), profile([("ML", U)]), V.D3_FULL, RankerConfig(k=5))
    assert [s.position for s in res.shown] == list(range(5))
    assert all(s.shown for s in res.shown)
    unshown = [s for s in res.scored if not s.shown]
    assert all(s.position is None for s in unshown)
