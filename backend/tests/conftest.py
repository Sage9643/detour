"""Shared fixtures.

DB-backed tests need TEST_DATABASE_URL pointing at a DISPOSABLE PostgreSQL database whose
name ends in `_test` (the suite drops and recreates its schema). Without it, DB tests are
skipped locally; in CI, DETOUR_REQUIRE_DB_TESTS=1 turns that skip into a failure so the
suite can never pass green while silently skipping the database layer.
"""

import os
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from detour.db.migrations_info import alembic_config


def _require_db_tests() -> bool:
    return os.environ.get("DETOUR_REQUIRE_DB_TESTS") == "1"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if _require_db_tests():
            pytest.fail("TEST_DATABASE_URL is not set but DETOUR_REQUIRE_DB_TESTS=1")
        pytest.skip("TEST_DATABASE_URL not set; skipping database tests")
    db_name = make_url(url).database or ""
    if not db_name.endswith("_test"):
        pytest.fail(f"Refusing to run destructive tests against database {db_name!r}")
    return url


def reset_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))


@pytest.fixture(scope="session")
def engine(test_db_url: str) -> Iterator[Engine]:
    """Engine for a database freshly migrated to head."""
    eng = create_engine(test_db_url)
    reset_schema(eng)
    command.upgrade(alembic_config(test_db_url), "head")
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Session inside an outer transaction that is always rolled back."""
    conn = engine.connect()
    outer = conn.begin()
    sess = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield sess
    finally:
        sess.close()
        outer.rollback()
        conn.close()


ALL_DATA_TABLES = (
    "feedback, run_candidates, recommendation_runs, interests, users, "
    "repo_embeddings, search_cache, repositories"
)


@pytest.fixture
def clean_db(engine: Engine) -> Iterator[Engine]:
    """For tests whose app commits real transactions: empty tables before AND after, so
    committed rows never leak into other tests."""

    def truncate() -> None:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {ALL_DATA_TABLES} CASCADE"))

    truncate()
    yield engine
    truncate()
