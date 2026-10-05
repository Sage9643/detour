"""Initial schema: users, interests, repositories, and the interaction log
(recommendation_runs, run_candidates, feedback).

Generated with autogenerate, then reviewed by hand. search_cache (Phase 2) and
repo_embeddings (Phase 3) are intentionally NOT here; they arrive as later migrations once
their shape is decided by measurement.

Revision ID: 0001
Revises:
Create Date: 2026-10-05 11:14:26.283055
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "repositories",
        sa.Column("github_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("owner_login", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "topics", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("stars", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("forks", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_fork", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("license_spdx", sa.Text(), nullable=True),
        sa.Column("html_url", sa.Text(), nullable=False),
        sa.Column("gh_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pushed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("metadata_fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "stars >= 0 AND forks >= 0", name=op.f("ck_repositories_counts_non_negative")
        ),
        sa.PrimaryKeyConstraint("github_id", name=op.f("pk_repositories")),
    )
    op.create_index(op.f("ix_repositories_full_name"), "repositories", ["full_name"], unique=False)
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("github_username", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_table(
        "interests",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("expanded_text", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), server_default=sa.text("1.0"), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(label) BETWEEN 1 AND 100", name=op.f("ck_interests_label_length")
        ),
        sa.CheckConstraint("weight > 0", name=op.f("ck_interests_weight_positive")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_interests_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interests")),
    )
    op.create_index(
        "uq_interests_user_label_ci",
        "interests",
        ["user_id", sa.literal_column("lower(label)")],
        unique=True,
    )
    op.create_table(
        "recommendation_runs",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("variant", sa.Text(), nullable=False),
        sa.Column("engine_version", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("profile_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=True),
        sa.Column("shown_count", sa.Integer(), nullable=True),
        sa.Column(
            "timings_ms",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'degraded', 'failed')",
            name=op.f("ck_recommendation_runs_status_valid"),
        ),
        sa.CheckConstraint(
            "variant IN ('B0', 'B1', 'D1', 'D2', 'D3')",
            name=op.f("ck_recommendation_runs_variant_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_recommendation_runs_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_recommendation_runs")),
    )
    op.create_index(
        "ix_recommendation_runs_user_created",
        "recommendation_runs",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "type IN ('INTERESTED', 'NOT_INTERESTED', 'ALREADY_KNOW')",
            name=op.f("ck_feedback_type_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.github_id"],
            name=op.f("fk_feedback_repo_id_repositories"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["recommendation_runs.id"],
            name=op.f("fk_feedback_run_id_recommendation_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_feedback_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    op.create_index(
        "ix_feedback_user_repo_created",
        "feedback",
        ["user_id", "repo_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "run_candidates",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("query_families", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("filter_reason", sa.Text(), nullable=True),
        sa.Column("shown", sa.Boolean(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("pre_rerank_rank", sa.Integer(), nullable=True),
        sa.Column("matched_interest_id", sa.UUID(), nullable=True),
        sa.Column("matched_interest_label", sa.Text(), nullable=True),
        sa.Column("relevance_raw", sa.Float(), nullable=True),
        sa.Column("relevance_norm", sa.Float(), nullable=True),
        sa.Column("unfamiliarity", sa.Float(), nullable=True),
        sa.Column("topical_novelty", sa.Float(), nullable=True),
        sa.Column("popularity_novelty", sa.Float(), nullable=True),
        sa.Column("novelty", sa.Float(), nullable=True),
        sa.Column("negative_penalty", sa.Float(), nullable=True),
        sa.Column("utility", sa.Float(), nullable=True),
        sa.Column("mmr_score", sa.Float(), nullable=True),
        sa.Column("max_sim_to_selected", sa.Float(), nullable=True),
        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "cardinality(query_families) >= 1 AND query_families <@ ARRAY['core', 'bridge', 'adjacent']::text[]",
            name=op.f("ck_run_candidates_query_families_valid"),
        ),
        sa.CheckConstraint(
            "(shown AND position IS NOT NULL) OR (NOT shown AND position IS NULL)",
            name=op.f("ck_run_candidates_shown_iff_position"),
        ),
        sa.CheckConstraint(
            "NOT (shown AND filter_reason IS NOT NULL)",
            name=op.f("ck_run_candidates_shown_not_filtered"),
        ),
        sa.CheckConstraint(
            "position IS NULL OR position >= 0",
            name=op.f("ck_run_candidates_position_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["matched_interest_id"],
            ["interests.id"],
            name=op.f("fk_run_candidates_matched_interest_id_interests"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.github_id"],
            name=op.f("fk_run_candidates_repo_id_repositories"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["recommendation_runs.id"],
            name=op.f("fk_run_candidates_run_id_recommendation_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_candidates")),
        sa.UniqueConstraint("run_id", "repo_id", name="uq_run_candidates_run_repo"),
    )
    op.create_index(op.f("ix_run_candidates_repo_id"), "run_candidates", ["repo_id"], unique=False)
    op.create_index(
        "uq_run_candidates_run_position",
        "run_candidates",
        ["run_id", "position"],
        unique=True,
        postgresql_where=sa.text("position IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_run_candidates_run_position",
        table_name="run_candidates",
        postgresql_where=sa.text("position IS NOT NULL"),
    )
    op.drop_index(op.f("ix_run_candidates_repo_id"), table_name="run_candidates")
    op.drop_table("run_candidates")
    op.drop_index("ix_feedback_user_repo_created", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_recommendation_runs_user_created", table_name="recommendation_runs")
    op.drop_table("recommendation_runs")
    op.drop_index("uq_interests_user_label_ci", table_name="interests")
    op.drop_table("interests")
    op.drop_table("users")
    op.drop_index(op.f("ix_repositories_full_name"), table_name="repositories")
    op.drop_table("repositories")
