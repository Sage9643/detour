import pytest
from pydantic import ValidationError

from detour.config import Environment, Settings


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_reads_values_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("DETOUR_ENV", "production")
    monkeypatch.setenv("DETOUR_DB_POOL_SIZE", "3")

    s = Settings(_env_file=None)  # type: ignore[call-arg]

    assert s.database_url == "postgresql+psycopg://u:p@db:5432/x"
    assert s.env is Environment.PRODUCTION
    assert s.db_pool_size == 3


def test_rejects_invalid_environment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("DETOUR_ENV", "staging-ish")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_rejects_non_positive_pool_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("DETOUR_DB_POOL_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
