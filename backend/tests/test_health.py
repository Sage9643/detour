"""Health endpoint tests that do NOT need a database.

DB-backed readiness tests live in test_health_db.py.
"""

from fastapi.testclient import TestClient

from detour import __version__
from detour.config import Settings
from detour.main import create_app

# Port 1 on localhost: nothing listens there, so connections fail fast.
UNREACHABLE_DB = "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"


def _settings(url: str) -> Settings:
    return Settings(database_url=url, DETOUR_DB_CONNECT_TIMEOUT_S=1, _env_file=None)  # type: ignore[call-arg]


def test_liveness_does_not_touch_database() -> None:
    with TestClient(create_app(_settings(UNREACHABLE_DB))) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_readiness_returns_503_when_database_unreachable() -> None:
    with TestClient(create_app(_settings(UNREACHABLE_DB))) as client:
        resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["database"] == "unreachable"
    # Credentials from the DSN must never leak into responses.
    assert "nothing" not in resp.text
