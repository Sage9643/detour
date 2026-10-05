"""Search cache and repository embeddings (cached external data).

search_cache: normalized GitHub query -> ordered repo ids (metadata lives in repositories).
repo_embeddings: (repo, model) -> vector, keyed by a hash of the embedded text.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05 12:09:10.791576
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_cache",
        sa.Column("query_key", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("sort", sa.Text(), nullable=True),
        sa.Column("per_page", sa.Integer(), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("repo_ids", postgresql.ARRAY(sa.BigInteger()), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "per_page BETWEEN 1 AND 100", name=op.f("ck_search_cache_per_page_range")
        ),
        sa.PrimaryKeyConstraint("query_key", name=op.f("pk_search_cache")),
    )
    op.create_table(
        "repo_embeddings",
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.Text(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vector", postgresql.ARRAY(sa.REAL()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "dim > 0 AND cardinality(vector) = dim",
            name=op.f("ck_repo_embeddings_dim_matches_vector"),
        ),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.github_id"],
            name=op.f("fk_repo_embeddings_repo_id_repositories"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("repo_id", "model_name", name=op.f("pk_repo_embeddings")),
    )


def downgrade() -> None:
    op.drop_table("repo_embeddings")
    op.drop_table("search_cache")
