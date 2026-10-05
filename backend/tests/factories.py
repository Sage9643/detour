"""Small explicit test factories (deliberately not a factory library)."""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from detour.db.models import RecommendationRun, Repository, User


def make_user(session: Session) -> User:
    user = User()
    session.add(user)
    session.flush()
    return user


def make_repo(session: Session, github_id: int = 1, **overrides: object) -> Repository:
    fields: dict[str, object] = {
        "github_id": github_id,
        "full_name": f"owner/repo-{github_id}",
        "owner_login": "owner",
        "name": f"repo-{github_id}",
        "html_url": f"https://github.com/owner/repo-{github_id}",
        "metadata_fetched_at": datetime.now(UTC),
    }
    fields.update(overrides)
    repo = Repository(**fields)
    session.add(repo)
    session.flush()
    return repo


def make_run(session: Session, user: User, **overrides: object) -> RecommendationRun:
    fields: dict[str, object] = {
        "user_id": user.id,
        "variant": "B1",
        "engine_version": "test",
        "config": {"k": 10},
        "config_hash": "deadbeef",
        "profile_snapshot": {"interests": []},
        "status": "ok",
    }
    fields.update(overrides)
    run = RecommendationRun(**fields)
    session.add(run)
    session.flush()
    return run


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()
