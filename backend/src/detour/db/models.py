"""SQLAlchemy models.

Schema overview (see docs/data-model.md for rationale):

    users ─< interests
      │
      ├─< recommendation_runs ─< run_candidates >─ repositories
      │                                   │
      └─< feedback >──────────────────────┘ (repo), optionally linked to a run

Design rules:
- run_candidates logs EVERY scored candidate in a run, not only the shown Top-K, so that
  alternative rankers can be replayed on identical pools and future learned models are not
  trained only on items the current ranker already selected (selection bias).
- Feature values are written at request time (point-in-time). Never recompute them later
  from current data: that would leak future information into training/evaluation.
- feedback is an append-only event log; current state = latest event per (user, repo).
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from detour.enums import FeedbackType, QueryFamily, RankerVariant, RunStatus

# Deterministic constraint names: required for reliable Alembic migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _in_enum(column: str, enum: type[StrEnum]) -> str:
    values = ", ".join(f"'{member.value}'" for member in enum)
    return f"{column} IN ({values})"


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


class User(Base):
    """Anonymous in v1 (no auth). github_username optionally seeds familiarity from stars."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    github_username: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class Interest(Base):
    """One explicit interest. Kept separate (never averaged into one vector) so relevance
    can be attributed to a specific interest and explained."""

    __tablename__ = "interests"
    __table_args__ = (
        CheckConstraint("char_length(label) BETWEEN 1 AND 100", name="label_length"),
        CheckConstraint("weight > 0", name="weight_positive"),
        Index("uq_interests_user_label_ci", "user_id", func.lower(text("label")), unique=True),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    # No separate index on user_id: uq_interests_user_label_ci already leads with it.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    # Expanded description that gets embedded (e.g. "distributed systems: consensus, ...").
    expanded_text: Mapped[str | None] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("1.0"))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Repository(Base):
    """Latest known metadata for a GitHub repository. Keyed by GitHub's numeric id, which
    survives renames (full_name does not)."""

    __tablename__ = "repositories"
    __table_args__ = (CheckConstraint("stars >= 0 AND forks >= 0", name="counts_non_negative"),)

    github_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    owner_login: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    language: Mapped[str | None] = mapped_column(Text)
    stars: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    forks: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_fork: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    license_spdx: Mapped[str | None] = mapped_column(Text)
    html_url: Mapped[str] = mapped_column(Text, nullable=False)
    gh_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pushed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = _created_at()
    # Metadata freshness is tracked separately from embedding freshness (Phase 3).
    metadata_fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RecommendationRun(Base):
    """One recommendation request: which ranker, with which config, on which profile."""

    __tablename__ = "recommendation_runs"
    __table_args__ = (
        CheckConstraint(_in_enum("variant", RankerVariant), name="variant_valid"),
        CheckConstraint(_in_enum("status", RunStatus), name="status_valid"),
        Index("ix_recommendation_runs_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    variant: Mapped[str] = mapped_column(Text, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    # Full ranker configuration (weights, thresholds, lambda, model name) + its hash, so
    # results can be grouped by exact configuration.
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # Profile as it was at request time (interests, weights, known-set sizes).
    profile_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_count: Mapped[int | None] = mapped_column(Integer)
    shown_count: Mapped[int | None] = mapped_column(Integer)
    timings_ms: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class RunCandidate(Base):
    """Every candidate scored in a run, shown or not, with point-in-time features.

    Score columns are nullable because simpler variants (B0, B1) do not compute them all.
    """

    __tablename__ = "run_candidates"
    __table_args__ = (
        UniqueConstraint("run_id", "repo_id", name="uq_run_candidates_run_repo"),
        CheckConstraint(
            "(shown AND position IS NOT NULL) OR (NOT shown AND position IS NULL)",
            name="shown_iff_position",
        ),
        CheckConstraint("position IS NULL OR position >= 0", name="position_non_negative"),
        CheckConstraint("NOT (shown AND filter_reason IS NOT NULL)", name="shown_not_filtered"),
        CheckConstraint(
            f"cardinality(query_families) >= 1 AND query_families <@ ARRAY["
            f"{', '.join(repr(f.value) for f in QueryFamily)}]::text[]",
            name="query_families_valid",
        ),
        Index(
            "uq_run_candidates_run_position",
            "run_id",
            "position",
            unique=True,
            postgresql_where=text("position IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"), nullable=False
    )
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.github_id", ondelete="RESTRICT"), nullable=False, index=True
    )

    # Retrieval provenance
    query_families: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    # Why the candidate was dropped before ranking (e.g. "below_relevance_gate",
    # "already_known"); NULL means it was eligible.
    filter_reason: Mapped[str | None] = mapped_column(Text)

    # Outcome
    shown: Mapped[bool] = mapped_column(Boolean, nullable=False)
    position: Mapped[int | None] = mapped_column(Integer)  # 0-based rank in the shown list
    pre_rerank_rank: Mapped[int | None] = mapped_column(Integer)  # rank by utility before MMR

    # Relevance
    matched_interest_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interests.id", ondelete="SET NULL")
    )
    matched_interest_label: Mapped[str | None] = mapped_column(Text)  # snapshot
    relevance_raw: Mapped[float | None] = mapped_column(Float)  # cosine, pre-normalization
    relevance_norm: Mapped[float | None] = mapped_column(Float)  # within-pool normalized

    # Novelty components (each in [0, 1])
    unfamiliarity: Mapped[float | None] = mapped_column(Float)
    topical_novelty: Mapped[float | None] = mapped_column(Float)
    popularity_novelty: Mapped[float | None] = mapped_column(Float)
    novelty: Mapped[float | None] = mapped_column(Float)

    # Utility and diversification
    negative_penalty: Mapped[float | None] = mapped_column(Float)
    utility: Mapped[float | None] = mapped_column(Float)
    mmr_score: Mapped[float | None] = mapped_column(Float)
    max_sim_to_selected: Mapped[float | None] = mapped_column(Float)

    # Full point-in-time feature vector (stars at request time, recency, language match,
    # topic overlap, ...). Extensible without migrations; typed columns above are the
    # stable core used in analysis queries.
    features: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Structured explanation reasons (Phase 8). Rendered text is derived, not stored.
    reasons: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Feedback(Base):
    """Append-only feedback events."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(_in_enum("type", FeedbackType), name="type_valid"),
        Index("ix_feedback_user_repo_created", "user_id", "repo_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.github_id", ondelete="RESTRICT"), nullable=False
    )
    # Which run the item was shown in; NULL for feedback outside a run (e.g. onboarding
    # "tick repos you already know"). Position is recoverable via run_candidates.
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("recommendation_runs.id", ondelete="SET NULL")
    )
    type: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()
