"""Engine and session construction.

The engine is created from Settings rather than at import time so tests and scripts can
build engines against different databases.
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from detour.config import Settings


def build_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,  # survive Neon/serverless idle disconnects
        connect_args={"connect_timeout": settings.db_connect_timeout_s},
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
