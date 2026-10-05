"""Application configuration.

All configuration comes from environment variables (optionally a local `.env` file for
development). Nothing secret has a default: a missing DATABASE_URL fails at startup
instead of silently connecting to an unintended database.

Ranking parameters live in detour.ranking.config.RankerConfig, not here: they are part of
the experiment definition and are logged with every run.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    database_url: str = Field(
        description="SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db",
    )
    env: Environment = Field(default=Environment.DEVELOPMENT, alias="DETOUR_ENV")
    log_level: str = Field(default="INFO", alias="DETOUR_LOG_LEVEL")

    # Connection pool sizing. Small defaults: Neon's free tier limits connections and a
    # single Render instance does not need many.
    db_pool_size: int = Field(default=5, ge=1, alias="DETOUR_DB_POOL_SIZE")
    db_connect_timeout_s: int = Field(default=5, ge=1, alias="DETOUR_DB_CONNECT_TIMEOUT_S")

    # --- GitHub ---------------------------------------------------------------------
    # Optional. Without a token, GitHub allows 10 search requests/minute; with one, 30.
    github_token: SecretStr | None = Field(default=None, alias="GITHUB_TOKEN")
    github_api_url: str = Field(default="https://api.github.com", alias="DETOUR_GITHUB_API_URL")
    github_timeout_s: float = Field(default=10.0, gt=0, alias="DETOUR_GITHUB_TIMEOUT_S")
    # Search results are cached per normalized query. The driver is GitHub's search rate
    # limit, not latency (see docs/decisions.md D-011).
    search_cache_ttl_s: int = Field(default=12 * 3600, ge=0, alias="DETOUR_SEARCH_CACHE_TTL_S")
    # Record/replay of GitHub HTTP traffic (see detour.retrieval.cassette).
    #   off     normal live operation
    #   record  live, and write every response to github_cassette_dir
    #   replay  never touch the network; serve only recorded responses
    github_cassette_mode: str = Field(default="off", alias="DETOUR_GITHUB_CASSETTE_MODE")
    github_cassette_dir: str | None = Field(default=None, alias="DETOUR_GITHUB_CASSETTE_DIR")

    # --- HTTP -------------------------------------------------------------------------
    # Comma-separated list of allowed browser origins (the Vercel frontend URL in prod).
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", alias="DETOUR_CORS_ORIGINS"
    )

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, v: str) -> str:
        """Accept plain postgres URLs as given by Neon/Render; select the psycopg 3 driver."""
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()  # values come from the environment
