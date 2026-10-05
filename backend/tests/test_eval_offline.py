"""Offline evaluator plumbing on a tiny synthetic snapshot (tests the code path, not results)."""

from typing import Any

from detour.eval.collect import build_example, derive_interests
from detour.eval.offline import aggregate, evaluate_user
from detour.ranking.config import RankerConfig
from detour.representation.embedder import default_embedder
from detour.retrieval.github import StarredRepo

from .fake_github import CATALOG
from .ranking_helpers import repo


def _rec(gid: int) -> dict[str, Any]:
    from detour.retrieval.models import RepoRecord

    item = next(i for i in CATALOG if i["id"] == gid)
    return RepoRecord.from_github(item).to_json()


def _example() -> dict[str, Any]:
    known = [
        _rec(g) | {"starred_at": "2026-01-01T00:00:00+00:00"} for g in (101, 102, 201, 202, 301)
    ]
    heldout = [_rec(g) | {"starred_at": "2026-08-01T00:00:00+00:00"} for g in (103, 204)]
    return {
        "user": "u1",
        "interests": ["Distributed Systems", "Machine Learning"],
        "known": known,
        "heldout": heldout,
    }


def _pool(ids: list[tuple[int, list[str]]]) -> dict[str, Any]:
    return {
        "user": "u1",
        "queries": [],
        "candidates": [
            {
                "repo": _rec(g),
                "families": fams,
                "query_labels": [],
                "adjacent_topics": [],
                "bridge_pairs": [],
            }
            for g, fams in ids
        ],
    }


def test_evaluate_user_protocols_and_ablation() -> None:
    pool = _pool(
        [
            (103, ["core"]),
            (104, ["core"]),
            (105, ["core"]),
            (205, ["core", "bridge"]),
            (206, ["core"]),
            (207, ["adjacent"]),
            (106, ["adjacent"]),
        ]
    )
    res = evaluate_user(_example(), pool, default_embedder(), RankerConfig(k=3))
    assert res["n_heldout"] == 2
    assert res["pool_recall"] == {"core": 0.5, "core+bridge": 0.5, "all": 0.5}  # 103 in, 204 not
    for protocol in ("end_to_end", "ranking_only"):
        assert set(res[protocol]) == {"B0", "B1", "D1", "D2", "D3"}
        for m in res[protocol].values():
            assert 0.0 <= m["recall"] <= 1.0
            assert m["list_len"] <= 3
    # known repos are never recommended, in any variant
    agg = aggregate([res, res])
    assert agg["end_to_end"]["paired_vs_B1"]["D3"]["recall"]["n"] == 2
    assert set(agg["ablation_end_to_end_D3"]) == {"core", "core+bridge", "all"}


def test_derive_interests_and_example_split() -> None:
    from datetime import UTC, datetime

    from detour.retrieval.models import RepoRecord

    recs = [RepoRecord.from_github(i) for i in CATALOG]
    assert derive_interests([r for r in recs if r.github_id in (101, 102, 103, 104)]) == [
        "Distributed Systems"
    ]
    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    early = [StarredRepo(datetime(2026, 1, 1, tzinfo=UTC), r) for r in recs[:20]]
    late = [StarredRepo(datetime(2026, 7, 1, tzinfo=UTC), r) for r in recs[:2]]
    # re-starring repos already known before the cutoff yields no held-out items -> skipped
    assert build_example("someone", late + early, cutoff) is None
    ex = build_example(
        "someone",
        [StarredRepo(datetime(2026, 7, 1, tzinfo=UTC), repo(900, stars=50, topics=["x"]))] * 3
        + early,
        cutoff,
    )
    assert ex is not None
    assert ex["user"] != "someone"  # anonymized
    assert len(ex["known"]) == 20
    assert {r["github_id"] for r in ex["heldout"]} == {900}
