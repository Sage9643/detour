"""Readiness against a real database."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from detour.config import Settings
from detour.db.migrations_info import expected_head
from detour.main import create_app

pytestmark = pytest.mark.db


def _client(url: str) -> TestClient:
    settings = Settings(database_url=url, _env_file=None)  # type: ignore[call-arg]
    return TestClient(create_app(settings))


def test_ready_when_schema_at_head(engine: Engine, test_db_url: str) -> None:
    with _client(test_db_url) as client:
        resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready",
        "database": "ok",
        "schema_revision": expected_head(),
        "expected_revision": expected_head(),
        "detail": None,
    }


def test_not_ready_when_schema_revision_mismatches(engine: Engine, test_db_url: str) -> None:
    """Simulates deploying new code without running migrations."""
    head = expected_head()
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = 'stale'"))
    try:
        with _client(test_db_url) as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["database"] == "ok"
        assert body["schema_revision"] == "stale"
        assert body["expected_revision"] == head
    finally:
        with engine.begin() as conn:
            conn.execute(text("UPDATE alembic_version SET version_num = :h"), {"h": head})
