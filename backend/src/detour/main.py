"""FastAPI application factory.

Run locally:  uvicorn detour.main:create_default_app --factory --reload
Tests build apps via create_app(settings) with their own settings.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from detour import __version__
from detour.api import health
from detour.config import Settings, get_settings
from detour.db.session import build_engine, build_session_factory
from detour.logging_setup import configure_logging


def create_app(settings: Settings) -> FastAPI:
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Engine creation is lazy: no connection is opened until first use, so the app
        # starts (and /health answers) even if the database is down.
        engine = build_engine(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Detour API", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    return app


def create_default_app() -> FastAPI:
    """ASGI factory for uvicorn: `uvicorn detour.main:create_default_app --factory`.

    Settings are read here, not at import time, so importing this module never depends
    on the environment. A missing DATABASE_URL fails at startup with a pydantic error.
    """
    return create_app(get_settings())
