"""Build the ranker's UserProfile from persisted interests and the feedback event log.

Profile state is DERIVED from the append-only feedback log on every request rather than
mutated in place. Consequences:
- a user changing their mind (NOT_INTERESTED -> INTERESTED) is handled by "latest event
  per repository wins";
- interest weight decay is reproducible and reversible;
- the exact profile used by a run is stored in recommendation_runs.profile_snapshot.

Feedback semantics:
  INTERESTED      positive exemplar + known (familiar) + its topics become known topics
  NOT_INTERESTED  negative exemplar (local penalty) + counts toward its interest's decay
  ALREADY_KNOW    known (familiar) + known topics. NOT a negative signal.
"""

import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from detour.db.models import Feedback, Interest, RecommendationRun, RunCandidate
from detour.db.repo_store import load_records
from detour.enums import FeedbackType
from detour.ranking.config import RankerConfig
from detour.ranking.profile import Exemplar, InterestVector, UserProfile
from detour.representation.embedder import Embedder
from detour.representation.interests import GENERIC_TOPICS, resolve_interest
from detour.representation.store import embed_repos_cached


@dataclass(frozen=True, slots=True)
class FeedbackState:
    latest: dict[int, FeedbackType]  # repo_id -> latest feedback type
    rejections_by_interest: Counter[uuid.UUID]


def interest_multiplier(rejections: int, cfg: RankerConfig) -> float:
    """1.0 until `decay_after` rejections, then decay_factor per extra rejection (floored)."""
    extra = max(0, rejections - cfg.decay_after)
    return max(cfg.min_weight, cfg.decay_factor**extra)


def load_feedback_state(session: Session, user_id: uuid.UUID) -> FeedbackState:
    rows = session.execute(
        select(Feedback.repo_id, Feedback.type, Feedback.run_id)
        .where(Feedback.user_id == user_id)
        .order_by(Feedback.created_at, Feedback.id)
    ).all()
    latest: dict[int, FeedbackType] = {}
    latest_run: dict[int, uuid.UUID | None] = {}
    for repo_id, ftype, run_id in rows:
        latest[repo_id] = FeedbackType(ftype)
        latest_run[repo_id] = run_id

    # Attribute each (current) rejection to the interest the repo was matched to when shown.
    rejected = [
        (rid, latest_run[rid]) for rid, t in latest.items() if t is FeedbackType.NOT_INTERESTED
    ]
    counts: Counter[uuid.UUID] = Counter()
    with_runs = [(rid, run) for rid, run in rejected if run is not None]
    if with_runs:
        matched = session.execute(
            select(
                RunCandidate.repo_id, RunCandidate.run_id, RunCandidate.matched_interest_id
            ).where(
                RunCandidate.run_id.in_({run for _, run in with_runs}),
                RunCandidate.repo_id.in_({rid for rid, _ in with_runs}),
            )
        ).all()
        lookup = {(rid, run): iid for rid, run, iid in matched}
        for rid, run in with_runs:
            iid = lookup.get((rid, run))
            if iid is not None:
                counts[iid] += 1
    return FeedbackState(latest=latest, rejections_by_interest=counts)


def build_profile(
    session: Session,
    user_id: uuid.UUID,
    embedder: Embedder,
    cfg: RankerConfig,
    now: datetime | None = None,
) -> tuple[UserProfile, dict[str, Any]]:
    now = now or datetime.now(UTC)
    interests = session.scalars(
        select(Interest)
        .where(Interest.user_id == user_id, Interest.active.is_(True))
        .order_by(Interest.created_at, Interest.id)
    ).all()
    state = load_feedback_state(session, user_id)

    # expanded_text holds an optional user override of the curated expansion.
    specs = [resolve_interest(i.label, i.expanded_text) for i in interests]
    ivecs = embedder.embed([spec.expansion for spec in specs])
    interest_vectors: list[InterestVector] = []
    interest_snapshot: list[dict[str, Any]] = []
    for row, spec, vec in zip(interests, specs, ivecs, strict=True):
        rejections = state.rejections_by_interest.get(row.id, 0)
        eff = row.weight * interest_multiplier(rejections, cfg)
        interest_vectors.append(
            InterestVector(label=row.label, vector=vec, weight=eff, topics=spec.topics, id=row.id)
        )
        interest_snapshot.append(
            {
                "id": str(row.id),
                "label": row.label,
                "curated": spec.curated,
                "base_weight": row.weight,
                "effective_weight": round(eff, 4),
                "rejections": rejections,
                "topics": list(spec.topics),
            }
        )

    positive_ids = [r for r, t in state.latest.items() if t is FeedbackType.INTERESTED]
    negative_ids = [r for r, t in state.latest.items() if t is FeedbackType.NOT_INTERESTED]
    known_ids = [
        r
        for r, t in state.latest.items()
        if t in (FeedbackType.INTERESTED, FeedbackType.ALREADY_KNOW)
    ]
    needed = sorted(set(positive_ids) | set(negative_ids) | set(known_ids))
    records = load_records(session, needed)
    vectors, _ = embed_repos_cached(session, embedder, [records[i] for i in needed if i in records])

    def exemplars(ids: list[int]) -> list[Exemplar]:
        return [
            Exemplar(repo_id=i, full_name=records[i].full_name, vector=vectors[i])
            for i in sorted(ids)
            if i in vectors
        ]

    known_present = [i for i in sorted(known_ids) if i in vectors]
    known_vectors = (
        np.stack([vectors[i] for i in known_present])
        if known_present
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    known_topics = {t for spec in specs for t in spec.topics}
    for i in known_present:
        known_topics.update(t for t in records[i].topics if t not in GENERIC_TOPICS)

    since = now - timedelta(days=cfg.recently_shown_days)
    recently_shown = set(
        session.scalars(
            select(RunCandidate.repo_id)
            .join(RecommendationRun, RecommendationRun.id == RunCandidate.run_id)
            .where(
                RecommendationRun.user_id == user_id,
                RecommendationRun.created_at >= since,
                RunCandidate.shown.is_(True),
            )
        ).all()
    )

    profile = UserProfile(
        interests=interest_vectors,
        positives=exemplars(positive_ids),
        negatives=exemplars(negative_ids),
        known_ids=set(known_ids),
        known_vectors=known_vectors,
        known_topics=known_topics,
        recently_shown=recently_shown,
    )
    snapshot = {
        "interests": interest_snapshot,
        "positives": [e.full_name for e in profile.positives],
        "negatives": [e.full_name for e in profile.negatives],
        "known_count": len(profile.known_ids),
        "known_topics_count": len(known_topics),
        "recently_shown_count": len(recently_shown),
        "cold_start": profile.cold_start,
    }
    return profile, snapshot
