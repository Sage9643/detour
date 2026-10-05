"""Migration correctness: round-trip and model/migration drift."""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect

from detour.db.migrations_info import alembic_config, current_revision, expected_head
from detour.db.models import Base

pytestmark = pytest.mark.db

EXPECTED_TABLES = {
    "users",
    "interests",
    "repositories",
    "recommendation_runs",
    "run_candidates",
    "feedback",
}


def test_database_is_at_head(engine: Engine) -> None:
    with engine.connect() as conn:
        assert current_revision(conn) == expected_head()
    assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES


def test_models_match_migrations(engine: Engine) -> None:
    """If this fails, a model changed without a migration (or vice versa)."""
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_downgrade_to_base_and_upgrade_again(engine: Engine, test_db_url: str) -> None:
    cfg = alembic_config(test_db_url)
    try:
        command.downgrade(cfg, "base")
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert remaining == set()
    finally:
        # Always restore head so other tests see the full schema.
        command.upgrade(cfg, "head")
    assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
