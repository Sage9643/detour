"""Liveness and readiness endpoints.

/health        liveness: the process is serving requests. No dependencies are touched, so
               a database outage does not cause the host to restart a healthy process.
/health/ready  readiness: the database is reachable AND the schema is at the revision this
               code expects. Returns 503 otherwise.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from detour import __version__
from detour.db.migrations_info import current_revision, expected_head

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class LivenessResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: Literal["ok", "unreachable"]
    schema_revision: str | None
    expected_revision: str | None
    detail: str | None = None


@router.get("/health", response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    return LivenessResponse(status="ok", version=__version__)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    engine: Engine = request.app.state.engine
    expected = expected_head()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            revision = current_revision(conn)
    except SQLAlchemyError as exc:
        # Log the exception type only; connection errors can contain the DSN.
        logger.warning("readiness check failed: database unreachable (%s)", type(exc).__name__)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            database="unreachable",
            schema_revision=None,
            expected_revision=expected,
            detail="database unreachable",
        )

    if revision != expected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            database="ok",
            schema_revision=revision,
            expected_revision=expected,
            detail="schema revision does not match code; run `alembic upgrade head`",
        )

    return ReadinessResponse(
        status="ready", database="ok", schema_revision=revision, expected_revision=expected
    )
