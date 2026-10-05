"""Application configuration.

All configuration comes from environment variables (optionally a local `.env` file for
development). Nothing secret has a default: a missing DATABASE_URL fails at startup
instead of silently connecting to an unintended database.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        description="SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db",
    )
    env: Environment = Field(default=Environment.DEVELOPMENT, alias="DETOUR_ENV")
    log_level: str = Field(default="INFO", alias="DETOUR_LOG_LEVEL")

    # Connection pool sizing. Small defaults: Neon's free tier limits connections and a
    # single Render instance does not need many.
    db_pool_size: int = Field(default=5, ge=1, alias="DETOUR_DB_POOL_SIZE")
    db_connect_timeout_s: int = Field(default=5, ge=1, alias="DETOUR_DB_CONNECT_TIMEOUT_S")


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()  # values come from the environment
