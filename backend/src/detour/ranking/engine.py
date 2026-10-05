"""Ranking engine: pure functions over a candidate pool. No I/O.

Stages (docs/recommendation-engine.md §6):
  filters -> relevance -> relevance gate -> novelty -> negative penalty -> utility
  -> selection (variant-specific; MMR + owner cap for diversified variants)

Relevance (§6.1) is calibrated per interest, because raw cosine scales differ a lot
between interests with static embeddings (measured on a live pool: mean on-target cosine
0.39 for "Distributed Systems" vs 0.58 for "Competitive Programming"):
  gate      margin = cosine - background(interest) >= relevance_margin
  ranking   rel = max over interests of  weight * percentile of cosine within the pool

Variants share the SAME filtered pool so that comparisons isolate the ranking logic:
  B0  popularity: stars desc                                   (no gate)
  B1  relevance only: calibrated relevance desc                (no gate)
  D1  gate -> utility = rel_norm + beta*novelty                (no MMR)
  D2  gate -> utility = rel_norm -> MMR + owner cap
  D3  gate -> utility = rel_norm + beta*novelty - gamma*neg -> MMR + owner cap
      with personalization: decayed interest weights + INTERESTED exemplars
Personalization signals are used only by D3 so its contribution is attributable; filters
(already known, recently shown) apply to every variant.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from detour.enums import QueryFamily, RankerVariant
from detour.ranking.config import RankerConfig
from detour.ranking.profile import UserProfile
from detour.representation.embedder import Vectors
from detour.representation.interests import GENERIC_TOPICS
from detour.retrieval.models import Candidate

Floats = npt.NDArray[np.float64]

MAX_DESCRIPTION_CHARS = 1000

GATED = {RankerVariant.D1_RELEVANCE_NOVELTY, RankerVariant.D2_RELEVANCE_MMR, RankerVariant.D3_FULL}
DIVERSIFIED = {RankerVariant.D2_RELEVANCE_MMR, RankerVariant.D3_FULL}
NOVELTY_USED = {RankerVariant.D1_RELEVANCE_NOVELTY, RankerVariant.D3_FULL}


@dataclass(slots=True)
class PoolItem:
    candidate: Candidate
    vector: Vectors  # (dim,), L2-normalized


@dataclass(slots=True)
class ScoredCandidate:
    """Everything the ranker computed for one candidate (logged for every candidate)."""

    item: PoolItem
    filter_reason: str | None = None
    # relevance
    interest_sims: dict[str, float] = field(default_factory=dict)  # raw cosine per interest
    matched_interest: str | None = None
    matched_via_exemplar: str | None = None  # full_name of the INTERESTED repo, if any
    relevance_raw: float | None = None  # raw cosine to the matched anchor
    relevance_margin: float | None = None  # raw cosine minus that interest's background
    relevance_norm: float | None = None  # calibrated relevance in [0, 1] used for ranking
    # novelty
    unfamiliarity: float | None = None
    topical_novelty: float | None = None
    popularity_novelty: float | None = None
    novelty: float | None = None
    nearest_known: str | None = None
    # utility / selection
    negative_penalty: float | None = None
    nearest_negative: str | None = None
    utility: float | None = None
    pre_rerank_rank: int | None = None
    mmr_score: float | None = None
    max_sim_to_selected: float | None = None
    diversity_promoted: bool = False  # MMR picked this over a higher-utility candidate
    shown: bool = False
    position: int | None = None

    @property
    def repo_id(self) -> int:
        return self.item.candidate.repo.github_id


KeyFn = Callable[[ScoredCandidate], float]


@dataclass(slots=True)
class RankResult:
    variant: RankerVariant
    scored: list[ScoredCandidate]  # every candidate, in pool order
    shown: list[ScoredCandidate]  # selected list, in display order
    eligible_count: int  # passed filters (+ gate for gated variants)
    filtered_counts: dict[str, int]


# ----------------------------------------------------------------------------------------
# helpers


def rank_normalize(values: Floats) -> Floats:
    """Map values to [0, 1] by rank (ties share their average rank). n<=1 -> 1.0."""
    n = values.shape[0]
    if n == 0:
        return values.astype(np.float64)
    if n == 1:
        return np.ones(1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ranks for ties
    uniq, inverse = np.unique(values, return_inverse=True)
    if uniq.shape[0] != n:
        sums = np.bincount(inverse, weights=ranks)
        cnts = np.bincount(inverse)
        ranks = (sums / cnts)[inverse]
    out: Floats = ranks / (n - 1)
    return out


def _filter_reason(c: Candidate, profile: UserProfile, cfg: RankerConfig) -> str | None:
    r = c.repo
    if r.archived:
        return "archived"
    if r.is_fork:
        return "fork"
    if not (r.description and r.description.strip()) and not r.topics:
        return "no_text"
    if r.description and len(r.description) > MAX_DESCRIPTION_CHARS:
        # Observed in live results: keyword-stuffed spam repos with 260 KB descriptions
        # matching bridge queries. Legitimate descriptions are a sentence or two.
        return "suspicious_description"
    if r.stars < cfg.star_floor:
        return "below_star_floor"
    if r.github_id in profile.known_ids:
        return "already_known"
    if r.github_id in profile.recently_shown:
        return "recently_shown"
    return None


def meaningful_topics(topics: Sequence[str]) -> list[str]:
    return [t for t in topics if t not in GENERIC_TOPICS]


# ----------------------------------------------------------------------------------------
# main entry point


def rank(
    pool: Sequence[PoolItem],
    profile: UserProfile,
    variant: RankerVariant,
    cfg: RankerConfig,
) -> RankResult:
    if not profile.interests:
        raise ValueError("profile has no interests")
    personalize = variant is RankerVariant.D3_FULL
    scored = [ScoredCandidate(item=p) for p in pool]

    filtered_counts: dict[str, int] = {}
    for s in scored:
        s.filter_reason = _filter_reason(s.item.candidate, profile, cfg)
        if s.filter_reason:
            filtered_counts[s.filter_reason] = filtered_counts.get(s.filter_reason, 0) + 1
    live = [s for s in scored if s.filter_reason is None]
    if not live:
        return RankResult(variant, scored, [], 0, filtered_counts)

    V = np.stack([s.item.vector for s in live]).astype(np.float64)  # (n, d)

    # --- relevance -------------------------------------------------------------------
    n = len(live)
    rows = np.arange(n)
    U = np.stack([iv.vector for iv in profile.interests]).astype(np.float64)  # (I, d)
    sims = V @ U.T  # (n, I) raw cosine
    background = np.array([iv.background for iv in profile.interests], dtype=np.float64)
    margins = sims - background
    weights = np.array(
        [iv.weight if personalize else 1.0 for iv in profile.interests], dtype=np.float64
    )
    pct = np.stack([rank_normalize(sims[:, j]) for j in range(sims.shape[1])], axis=1)
    weighted = pct * weights
    best = weighted.argmax(axis=1)
    rel = weighted[rows, best]
    passes_gate = margins[rows, best] >= cfg.relevance_margin

    ex_rel = np.full(n, -np.inf)
    ex_idx = np.zeros(n, dtype=int)
    ex_sim = np.zeros(n)
    if personalize and profile.positives:
        P = np.stack([e.vector for e in profile.positives]).astype(np.float64)
        psims = V @ P.T
        ex_idx = psims.argmax(axis=1)
        ex_sim = psims[rows, ex_idx]
        ex_rel = cfg.alpha_exemplar * rank_normalize(ex_sim)
        passes_gate = passes_gate | (ex_sim >= cfg.exemplar_gate)

    via_exemplar = ex_rel > rel
    rel_score = np.maximum(rel, ex_rel)
    labels = [iv.label for iv in profile.interests]
    gate_ok: list[bool] = []
    for i, s in enumerate(live):
        s.interest_sims = {lab: round(float(sims[i, j]), 4) for j, lab in enumerate(labels)}
        s.matched_interest = labels[int(best[i])]
        s.relevance_margin = float(margins[i, best[i]])
        if via_exemplar[i]:
            s.matched_via_exemplar = profile.positives[int(ex_idx[i])].full_name
            s.relevance_raw = float(ex_sim[i])
        else:
            s.relevance_raw = float(sims[i, best[i]])
        s.relevance_norm = float(rel_score[i])
        gate_ok.append(bool(passes_gate[i]))

    # --- novelty (computed for every variant so it can be analysed; used by D1/D3) -----
    stars = np.array([s.item.candidate.repo.stars for s in live], dtype=np.float64)
    pop_nov = 1.0 - rank_normalize(np.log1p(stars))
    known = profile.known_vectors
    for i, s in enumerate(live):
        comps: list[float] = []
        if known.shape[0] > 0:
            ksims = known.astype(np.float64) @ V[i]
            j = int(ksims.argmax())
            s.unfamiliarity = float(np.clip(1.0 - ksims[j], 0.0, 1.0))
            comps.append(s.unfamiliarity)
        topics = meaningful_topics(s.item.candidate.repo.topics)
        if topics:
            new = [t for t in topics if t not in profile.known_topics]
            s.topical_novelty = len(new) / len(topics)
            comps.append(s.topical_novelty)
        s.popularity_novelty = float(pop_nov[i])
        comps.append(s.popularity_novelty)
        s.novelty = float(np.mean(comps))

    # --- negative penalty (D3 only) ----------------------------------------------------
    if personalize and profile.negatives:
        N = np.stack([e.vector for e in profile.negatives]).astype(np.float64)
        nsims = V @ N.T
        nidx = nsims.argmax(axis=1)
        for i, s in enumerate(live):
            top = float(nsims[i, nidx[i]])
            if top >= cfg.theta_negative:
                s.negative_penalty = top
                s.nearest_negative = profile.negatives[int(nidx[i])].full_name
            else:
                s.negative_penalty = 0.0
    elif personalize:
        for s in live:
            s.negative_penalty = 0.0

    # --- gate ---------------------------------------------------------------------------
    if variant in GATED:
        eligible_idx = []
        for i, s in enumerate(live):
            if gate_ok[i]:
                eligible_idx.append(i)
            else:
                s.filter_reason = "below_relevance_gate"
                filtered_counts["below_relevance_gate"] = (
                    filtered_counts.get("below_relevance_gate", 0) + 1
                )
    else:
        eligible_idx = list(range(len(live)))

    # --- utility ------------------------------------------------------------------------
    for i in eligible_idx:
        s = live[i]
        assert s.relevance_norm is not None and s.novelty is not None
        u = s.relevance_norm
        if variant in NOVELTY_USED:
            u += cfg.beta_novelty * s.novelty
        if personalize and s.negative_penalty:
            u -= cfg.gamma_negative * s.negative_penalty
        s.utility = u

    # --- selection ---------------------------------------------------------------------
    elig = [live[i] for i in eligible_idx]
    vecs = V[eligible_idx] if eligible_idx else np.zeros((0, V.shape[1]))

    def key_for(s: ScoredCandidate) -> float:
        if variant is RankerVariant.B0_POPULARITY:
            return float(s.item.candidate.repo.stars)
        if variant is RankerVariant.B1_RELEVANCE:
            return float(s.relevance_norm or 0.0)
        return float(s.utility or 0.0)

    # deterministic: score desc, then github_id asc
    pre_order = sorted(range(len(elig)), key=lambda j: (-key_for(elig[j]), elig[j].repo_id))
    for r, j in enumerate(pre_order):
        elig[j].pre_rerank_rank = r

    chosen = _mmr(elig, vecs, cfg, key_for) if variant in DIVERSIFIED else pre_order[: cfg.k]

    shown: list[ScoredCandidate] = []
    for pos, j in enumerate(chosen):
        s = elig[j]
        s.shown = True
        s.position = pos
        shown.append(s)

    # similarity of every eligible candidate to the final list (diversity context feature)
    if shown and len(elig):
        sel = vecs[chosen]
        sims_to_sel = vecs @ sel.T
        for j, s in enumerate(elig):
            if not s.shown:
                s.max_sim_to_selected = float(sims_to_sel[j].max())

    return RankResult(variant, scored, shown, len(elig), filtered_counts)


def _mmr(
    elig: list[ScoredCandidate],
    vecs: Floats,
    cfg: RankerConfig,
    key_for: KeyFn,
) -> list[int]:
    """Greedy Maximal Marginal Relevance with a per-owner cap and an interest-coverage floor.

    next = argmax  lambda*utility(r) - (1-lambda)*max_{s in S} cos(r, s)

    Coverage floor: MMR removes redundancy but does not guarantee that every interest is
    represented. When the remaining slots equal the number of still-uncovered interests
    (that have eligible candidates), selection is restricted to those interests. This only
    changes the tail of the list, and only when coverage would otherwise be lost.
    """
    n = len(elig)
    if n == 0:
        return []
    util = np.array([key_for(s) for s in elig], dtype=np.float64)
    ids = np.array([s.repo_id for s in elig])
    owners = [s.item.candidate.repo.owner_login.lower() for s in elig]
    interest_of = [s.matched_interest for s in elig]
    max_sim = np.zeros(n)  # max similarity to anything selected so far
    available = np.ones(n, dtype=bool)
    owner_count: dict[str, int] = {}
    chosen: list[int] = []
    lam = cfg.mmr_lambda

    while len(chosen) < cfg.k:
        allowed = available & np.array([owner_count.get(o, 0) < cfg.owner_cap for o in owners])
        if not allowed.any():
            break
        if cfg.coverage_floor:
            covered = {interest_of[j] for j in chosen}
            uncovered = {interest_of[j] for j in np.flatnonzero(allowed)} - covered
            if uncovered and cfg.k - len(chosen) <= len(uncovered):
                restricted = allowed & np.array([i in uncovered for i in interest_of])
                if restricted.any():
                    allowed = restricted
        score = lam * util - (1.0 - lam) * max_sim
        cand = np.flatnonzero(allowed)
        # deterministic tie-break: highest score, then smallest github_id
        best = min(cand, key=lambda j: (-score[j], ids[j]))
        top_util = min(cand, key=lambda j: (-util[j], ids[j]))
        s = elig[best]
        s.mmr_score = float(score[best])
        s.max_sim_to_selected = float(max_sim[best]) if chosen else 0.0
        s.diversity_promoted = bool(best != top_util)
        chosen.append(int(best))
        available[best] = False
        owner_count[owners[best]] = owner_count.get(owners[best], 0) + 1
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    return chosen


def families_of(s: ScoredCandidate) -> list[str]:
    order = [QueryFamily.CORE, QueryFamily.BRIDGE, QueryFamily.ADJACENT]
    return [f.value for f in order if f in s.item.candidate.families]
