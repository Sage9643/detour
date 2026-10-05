"""FastAPI application factory.

Run locally:  uvicorn detour.main:create_default_app --factory --reload
Tests build apps via create_app(settings, ...) with their own settings and dependencies.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from detour import __version__
from detour.api import health, users
from detour.config import Settings, get_settings
from detour.db.session import build_engine, build_session_factory
from detour.logging_setup import configure_logging
from detour.recommend.service import Deps
from detour.representation.embedder import Embedder, default_embedder
from detour.retrieval.cassette import build_transport
from detour.retrieval.github import GitHubClient

logger = logging.getLogger(__name__)


def build_github_client(settings: Settings) -> GitHubClient:
    token = settings.github_token.get_secret_value() if settings.github_token else None
    return GitHubClient(
        base_url=settings.github_api_url,
        token=token,
        timeout_s=settings.github_timeout_s,
        transport=build_transport(settings.github_cassette_mode, settings.github_cassette_dir),
    )


def create_app(
    settings: Settings,
    github: GitHubClient | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Engine creation is lazy: no connection is opened until first use, so the app
        # starts (and /health answers) even if the database is down.
        engine = build_engine(settings)
        client = github or build_github_client(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        app.state.deps = Deps(
            github=client,
            embedder=embedder or default_embedder(),
            cache_ttl_s=settings.search_cache_ttl_s,
        )
        logger.info(
            "startup github_authenticated=%s cassette=%s embedder=%s",
            client.authenticated,
            settings.github_cassette_mode,
            app.state.deps.embedder.model_name,
        )
        yield
        if github is None:
            client.close()
        engine.dispose()

    app = FastAPI(title="Detour API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        # Never leak SQL or DSNs; log the type only.
        logger.error("database error: %s", type(exc).__name__)
        unavailable = isinstance(exc, OperationalError)
        return JSONResponse(
            status_code=503 if unavailable else 500,
            content={
                "detail": {
                    "code": "database_unavailable" if unavailable else "database_error",
                    "message": "database unavailable" if unavailable else "database error",
                }
            },
        )

    app.include_router(health.router)
    app.include_router(users.router)
    return app


def create_default_app() -> FastAPI:
    """ASGI factory for uvicorn: `uvicorn detour.main:create_default_app --factory`.

    Settings are read here, not at import time, so importing this module never depends
    on the environment. A missing DATABASE_URL fails at startup with a pydantic error.
    """
    return create_app(get_settings())
