"""Recommendation orchestration: profile -> retrieval -> embeddings -> ranking -> log.

This is the only module that wires I/O (database, GitHub, embedder) to the pure ranking
engine. Every run is persisted with its configuration, profile snapshot, per-stage
timings, and EVERY scored candidate (shown or not) with point-in-time features.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from detour.db.models import Interest, RecommendationRun, RunCandidate, User
from detour.enums import RankerVariant, RunStatus
from detour.profile.builder import build_profile
from detour.ranking.config import ENGINE_VERSION, RankerConfig, config_hash
from detour.ranking.engine import PoolItem, ScoredCandidate, families_of, rank
from detour.ranking.explain import build_reasons, render
from detour.ranking.profile import UserProfile
from detour.representation.embedder import Embedder
from detour.representation.interests import resolve_interest
from detour.representation.store import embed_repos_cached
from detour.retrieval.github import GitHubClient
from detour.retrieval.planner import RetrievalConfig
from detour.retrieval.service import (
    CandidateRetriever,
    DbSearchCache,
    RetrievalFailed,
    RetrievalOutcome,
)

logger = logging.getLogger(__name__)


class UserNotFound(Exception):
    pass


class NoInterests(Exception):
    pass


@dataclass(slots=True)
class RunView:
    run_id: uuid.UUID
    variant: RankerVariant
    status: RunStatus
    created_at: datetime
    items: list[dict[str, Any]]
    stats: dict[str, Any]
    profile: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecommendResult:
    primary: RunView
    comparison: RunView | None = None


@dataclass(slots=True)
class Deps:
    github: GitHubClient
    embedder: Embedder
    cache_ttl_s: int
    retrieval_cfg: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker_cfg: RankerConfig = field(default_factory=RankerConfig)


def recommend(
    session: Session,
    user_id: uuid.UUID,
    deps: Deps,
    variant: RankerVariant = RankerVariant.D3_FULL,
    k: int | None = None,
    compare_variant: RankerVariant | None = None,
) -> RecommendResult:
    if session.get(User, user_id) is None:
        raise UserNotFound(str(user_id))
    cfg = deps.ranker_cfg if k is None else deps.ranker_cfg.with_overrides(k=k)
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    t = time.perf_counter()
    profile, snapshot = build_profile(session, user_id, deps.embedder, cfg)
    timings["profile_ms"] = _ms(t)
    if not profile.interests:
        raise NoInterests(str(user_id))

    interest_rows = session.scalars(
        select(Interest).where(Interest.user_id == user_id, Interest.active.is_(True))
    ).all()
    interest_ids = {row.label: row.id for row in interest_rows}
    specs = [resolve_interest(iv.label) for iv in profile.interests]

    t = time.perf_counter()
    retriever = CandidateRetriever(deps.github, DbSearchCache(session), deps.cache_ttl_s)
    try:
        retrieval = retriever.retrieve(specs, set(profile.known_topics), deps.retrieval_cfg)
    except RetrievalFailed as exc:
        timings["retrieval_ms"] = _ms(t)
        _log_failed_run(session, user_id, variant, cfg, deps, snapshot, timings, str(exc))
        raise
    timings["retrieval_ms"] = _ms(t)

    candidates = list(retrieval.candidates.values())
    t = time.perf_counter()
    vectors, embed_stats = embed_repos_cached(session, deps.embedder, [c.repo for c in candidates])
    timings["embedding_ms"] = _ms(t)
    pool = [PoolItem(candidate=c, vector=vectors[c.repo.github_id]) for c in candidates]

    common_stats = {
        "candidates": len(pool),
        "by_family": _family_counts(retrieval),
        "queries": [q.to_dict() for q in retrieval.queries],
        "github_calls": retrieval.github_calls,
        "github_authenticated": deps.github.authenticated,
        "search_rate_remaining": deps.github.search_rate.remaining,
        "embedding_cache": {"hits": embed_stats.hits, "misses": embed_stats.misses},
        "embedding_model": deps.embedder.model_name,
        "data_fetched_at": (
            retrieval.oldest_data_at.isoformat() if retrieval.oldest_data_at else None
        ),
    }

    primary = _rank_and_log(
        session,
        user_id,
        variant,
        pool,
        profile,
        snapshot,
        cfg,
        deps,
        retrieval,
        dict(timings),
        t_total,
        common_stats,
        interest_ids,
    )
    comparison = None
    if compare_variant is not None and compare_variant is not variant:
        comparison = _rank_and_log(
            session,
            user_id,
            compare_variant,
            pool,
            profile,
            snapshot,
            cfg,
            deps,
            retrieval,
            dict(timings),
            t_total,
            common_stats,
            interest_ids,
            paired_with=primary.run_id,
        )
    return RecommendResult(primary=primary, comparison=comparison)


def _rank_and_log(
    session: Session,
    user_id: uuid.UUID,
    variant: RankerVariant,
    pool: list[PoolItem],
    profile: UserProfile,
    snapshot: dict[str, Any],
    cfg: RankerConfig,
    deps: Deps,
    retrieval: RetrievalOutcome,
    timings: dict[str, float],
    t_total: float,
    common_stats: dict[str, Any],
    interest_ids: dict[str, uuid.UUID],
    paired_with: uuid.UUID | None = None,
) -> RunView:
    t = time.perf_counter()
    result = rank(pool, profile, variant, cfg)
    reasons = {s.repo_id: build_reasons(s, profile, cfg.tau) for s in result.shown}
    timings["ranking_ms"] = _ms(t)

    status = RunStatus.DEGRADED if retrieval.degraded else RunStatus.OK
    run_config = {
        "variant": variant.value,
        "ranker": cfg.to_dict(),
        "retrieval": deps.retrieval_cfg.to_dict(),
        "embedding_model": deps.embedder.model_name,
    }
    if paired_with:
        run_config["paired_with_run"] = str(paired_with)

    t = time.perf_counter()
    run = RecommendationRun(
        user_id=user_id,
        variant=variant.value,
        engine_version=ENGINE_VERSION,
        config=run_config,
        config_hash=config_hash(run_config | {"paired_with_run": None}),
        profile_snapshot=snapshot,
        status=status.value,
        candidate_count=len(pool),
        shown_count=len(result.shown),
        timings_ms={},
    )
    session.add(run)
    session.flush()
    rows = [_candidate_row(run.id, s, reasons.get(s.repo_id), interest_ids) for s in result.scored]
    if rows:
        session.execute(insert(RunCandidate), rows)
    timings["logging_ms"] = _ms(t)
    timings["total_ms"] = _ms(t_total)
    run.timings_ms = {k: round(v, 1) for k, v in timings.items()}
    session.flush()

    stats = dict(common_stats)
    stats.update(
        {
            "eligible": result.eligible_count,
            "filtered": result.filtered_counts,
            "shown": len(result.shown),
            "timings_ms": run.timings_ms,
        }
    )
    items = [_item_view(s, reasons[s.repo_id]) for s in result.shown]
    return RunView(
        run_id=run.id,
        variant=variant,
        status=status,
        created_at=datetime.now(UTC),
        items=items,
        stats=stats,
        profile=snapshot,
        warnings=list(retrieval.warnings),
    )


def _candidate_row(
    run_id: uuid.UUID,
    s: ScoredCandidate,
    reasons: dict[str, Any] | None,
    interest_ids: dict[str, uuid.UUID],
) -> dict[str, Any]:
    repo = s.item.candidate.repo
    now = datetime.now(UTC)
    features: dict[str, Any] = {
        # point-in-time metadata (as of this request)
        "stars": repo.stars,
        "forks": repo.forks,
        "language": repo.language,
        "topic_count": len(repo.topics),
        "repo_age_days": (now - repo.gh_created_at).days if repo.gh_created_at else None,
        "days_since_push": (now - repo.pushed_at).days if repo.pushed_at else None,
        "interest_sims": s.interest_sims,
        "matched_via_exemplar": s.matched_via_exemplar,
        "nearest_negative": s.nearest_negative,
        "diversity_promoted": s.diversity_promoted,
        "query_labels": s.item.candidate.query_labels,
    }
    return {
        "run_id": run_id,
        "repo_id": repo.github_id,
        "query_families": families_of(s),
        "filter_reason": s.filter_reason,
        "shown": s.shown,
        "position": s.position,
        "pre_rerank_rank": s.pre_rerank_rank,
        "matched_interest_id": interest_ids.get(s.matched_interest or ""),
        "matched_interest_label": s.matched_interest,
        "relevance_raw": s.relevance_raw,
        "relevance_norm": s.relevance_norm,
        "unfamiliarity": s.unfamiliarity,
        "topical_novelty": s.topical_novelty,
        "popularity_novelty": s.popularity_novelty,
        "novelty": s.novelty,
        "negative_penalty": s.negative_penalty,
        "utility": s.utility,
        "mmr_score": s.mmr_score,
        "max_sim_to_selected": s.max_sim_to_selected,
        "features": features,
        "reasons": reasons,
    }


def _item_view(s: ScoredCandidate, reasons: dict[str, Any]) -> dict[str, Any]:
    repo = s.item.candidate.repo
    return {
        "position": s.position,
        "repo": {
            "github_id": repo.github_id,
            "full_name": repo.full_name,
            "owner": repo.owner_login,
            "description": repo.description,
            "html_url": repo.html_url,
            "language": repo.language,
            "stars": repo.stars,
            "forks": repo.forks,
            "topics": list(repo.topics),
            "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
        },
        "scores": {
            "relevance_raw": _r(s.relevance_raw),
            "relevance_norm": _r(s.relevance_norm),
            "novelty": _r(s.novelty),
            "unfamiliarity": _r(s.unfamiliarity),
            "topical_novelty": _r(s.topical_novelty),
            "popularity_novelty": _r(s.popularity_novelty),
            "negative_penalty": _r(s.negative_penalty),
            "utility": _r(s.utility),
            "mmr_score": _r(s.mmr_score),
            "max_sim_to_selected": _r(s.max_sim_to_selected),
            "pre_rerank_rank": s.pre_rerank_rank,
            "interest_sims": s.interest_sims,
        },
        "matched_interest": s.matched_interest,
        "query_families": families_of(s),
        "reasons": reasons,
        "explanation": render(reasons),
    }


def _log_failed_run(
    session: Session,
    user_id: uuid.UUID,
    variant: RankerVariant,
    cfg: RankerConfig,
    deps: Deps,
    snapshot: dict[str, Any],
    timings: dict[str, float],
    error: str,
) -> None:
    run_config = {
        "variant": variant.value,
        "ranker": cfg.to_dict(),
        "retrieval": deps.retrieval_cfg.to_dict(),
        "embedding_model": deps.embedder.model_name,
    }
    session.add(
        RecommendationRun(
            user_id=user_id,
            variant=variant.value,
            engine_version=ENGINE_VERSION,
            config=run_config,
            config_hash=config_hash(run_config),
            profile_snapshot=snapshot,
            status=RunStatus.FAILED.value,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            error=error[:1000],
        )
    )
    session.flush()


def _family_counts(retrieval: RetrievalOutcome) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in retrieval.candidates.values():
        for f in c.families:
            counts[f.value] = counts.get(f.value, 0) + 1
    return counts


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


__all__ = [
    "Deps",
    "NoInterests",
    "RecommendResult",
    "RetrievalFailed",
    "RunView",
    "UserNotFound",
    "recommend",
]
