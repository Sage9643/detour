"""FastAPI dependencies and error mapping."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from detour.recommend.service import Deps


def get_session(request: Request) -> Iterator[Session]:
    """One transaction per request: commit on success, roll back on any error."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_deps(request: Request) -> Deps:
    deps: Deps = request.app.state.deps
    return deps


SessionDep = Annotated[Session, Depends(get_session)]
EngineDeps = Annotated[Deps, Depends(get_deps)]


def api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})
