"""Representation (text, embedder) and explanation tests."""

import numpy as np
import pytest

from detour.enums import QueryFamily, RankerVariant
from detour.ranking.config import RankerConfig
from detour.ranking.engine import rank
from detour.ranking.explain import build_reasons, popularity_band, render
from detour.representation.embedder import default_embedder, l2_normalize
from detour.representation.text import repo_text, split_name, text_hash

from .ranking_helpers import item, profile, repo, vec


def test_split_name() -> None:
    assert split_name("raft-rs") == "raft rs"
    assert split_name("FastAPI") == "fast api"
    assert split_name("etcd_io.v2") == "etcd io v2"


def test_repo_text_excludes_language_and_metadata() -> None:
    r = repo(1, topics=["key-value-store"], description="Distributed KV store", stars=999)
    t = repo_text(r)
    assert t == "r1. Distributed KV store. topics: key value store"
    assert "999" not in t
    assert text_hash(t) == text_hash(t) and len(text_hash(t)) == 64


def test_l2_normalize_handles_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert out[0].tolist() == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == [0.0, 0.0]


def test_pretrained_embedder_captures_topical_similarity() -> None:
    """Sanity check of the REAL model (not a calibration): related > unrelated."""
    emb = default_embedder()
    v = emb.embed(
        [
            "distributed systems: consensus, replication, fault tolerance",
            "etcd: distributed reliable key-value store using raft consensus",
            "react component library for building user interfaces",
        ]
    )
    assert v.shape == (3, emb.dim)
    assert np.linalg.norm(v, axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) + 0.2
    # deterministic
    assert np.array_equal(emb.embed(["same text"]), emb.embed(["same text"]))
    assert emb.embed([]).shape == (0, emb.dim)


# --- explanations: every claim must be backed by a computed signal ---------------------------


def _ranked(pool, prof, variant=RankerVariant.D3_FULL, **cfg):  # type: ignore[no-untyped-def]
    return rank(pool, prof, variant, RankerConfig(**cfg))


def test_explanation_claims_match_scores() -> None:
    u_ml, u_ds = vec(1, 0, 0, 0), vec(0, 1, 0, 0)
    a = item(
        repo(1, stars=150, topics=["federated-learning", "python"]),
        vec(0.7, 0.6, 0, 0.387),
        families=[QueryFamily.CORE, QueryFamily.BRIDGE],
    )
    a.candidate.bridge_pairs.append(("ML", "DS"))
    b = item(repo(2, stars=50_000, topics=["machine-learning"]), vec(0.5, 0, 0, 0.866))
    p = profile([("ML", u_ml), ("DS", u_ds)], known_topics=["machine-learning"])
    res = _ranked([a, b], p, k=2)
    s = next(x for x in res.shown if x.repo_id == 1)
    reasons = build_reasons(s, p, tau=0.18)

    assert reasons["matched_interest"] == "ML"
    assert reasons["secondary_interest"] == "DS"  # cos 0.6 >= tau
    assert reasons["strong_match"] == (s.relevance_norm is not None and s.relevance_norm >= 0.8)
    assert reasons["new_topics"] == ["federated-learning"]  # generic 'python' never claimed
    assert reasons["bridge"] == ["ML", "DS"]
    assert reasons["popularity_band"] == "niche"
    assert "unfamiliar" not in reasons  # no familiarity data -> no familiarity claim
    assert reasons["cold_start"] is True

    text = render(reasons)
    assert text[0] == "Strong match with your ML interest."
    assert "Also relevant to DS." in text
    assert "Found by a query bridging ML and DS." in text
    assert "Introduces topics you haven't interacted with: federated-learning." in text
    assert "Less-known project (150 stars)." in text

    other = next(x for x in res.scored if x.repo_id == 2)
    r2 = build_reasons(other, p, tau=0.18)
    assert "secondary_interest" not in r2  # cos(DS) = 0 < tau
    assert "new_topics" not in r2  # only a known topic
    assert not any("Less-known" in line for line in render(r2))


def test_exemplar_and_unfamiliarity_explanations() -> None:
    cand = item(repo(1), vec(0.15, 0, 0.989, 0))
    p = profile(
        [("ML", vec(1, 0, 0, 0))],
        positives=[(55, vec(0.1, 0, 0.995, 0))],
        known=[(55, vec(0.1, 0, 0.995, 0)), (56, vec(0, 1, 0, 0))],
    )
    res = _ranked([cand], p)
    s = res.shown[0]
    reasons = build_reasons(s, p, tau=0.18)
    assert reasons["similar_to_liked"] == "liked/r55"
    assert render(reasons)[0] == "Similar to liked/r55, which you marked as interesting."
    # it is near a known repo, so it must NOT be called unfamiliar
    assert s.unfamiliarity is not None and s.unfamiliarity < 0.6
    assert "unfamiliar" not in reasons


@pytest.mark.parametrize(
    ("stars", "band"),
    [
        (10, "niche"),
        (199, "niche"),
        (200, "emerging"),
        (5_000, "established"),
        (90_000, "widely used"),
    ],
)
def test_popularity_bands(stars: int, band: str) -> None:
    assert popularity_band(stars) == band
