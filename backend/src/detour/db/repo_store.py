"""Persistence helpers for repository metadata (cached external data)."""

from collections.abc import Iterable, Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from detour.db.models import Repository
from detour.retrieval.models import RepoRecord

_UPDATABLE = (
    "full_name",
    "owner_login",
    "name",
    "description",
    "topics",
    "language",
    "stars",
    "forks",
    "archived",
    "is_fork",
    "license_spdx",
    "html_url",
    "gh_created_at",
    "pushed_at",
    "metadata_fetched_at",
)


def upsert_repositories(
    session: Session, records: Iterable[RepoRecord], fetched_at: datetime
) -> None:
    rows = {}
    for r in records:  # dedupe: ON CONFLICT cannot touch the same row twice in one statement
        rows[r.github_id] = {
            "github_id": r.github_id,
            "full_name": r.full_name,
            "owner_login": r.owner_login,
            "name": r.name,
            "description": r.description,
            "topics": list(r.topics),
            "language": r.language,
            "stars": r.stars,
            "forks": r.forks,
            "archived": r.archived,
            "is_fork": r.is_fork,
            "license_spdx": r.license_spdx,
            "html_url": r.html_url,
            "gh_created_at": r.gh_created_at,
            "pushed_at": r.pushed_at,
            "metadata_fetched_at": fetched_at,
        }
    if not rows:
        return
    stmt = insert(Repository).values(list(rows.values()))
    stmt = stmt.on_conflict_do_update(
        index_elements=[Repository.github_id],
        set_={col: stmt.excluded[col] for col in _UPDATABLE},
    )
    session.execute(stmt)


def to_record(row: Repository) -> RepoRecord:
    return RepoRecord(
        github_id=row.github_id,
        full_name=row.full_name,
        owner_login=row.owner_login,
        name=row.name,
        description=row.description,
        topics=tuple(row.topics or ()),
        language=row.language,
        stars=row.stars,
        forks=row.forks,
        archived=row.archived,
        is_fork=row.is_fork,
        license_spdx=row.license_spdx,
        html_url=row.html_url,
        gh_created_at=row.gh_created_at,
        pushed_at=row.pushed_at,
    )


def load_records(session: Session, ids: Sequence[int]) -> dict[int, RepoRecord]:
    if not ids:
        return {}
    rows = session.scalars(select(Repository).where(Repository.github_id.in_(list(ids))))
    return {row.github_id: to_record(row) for row in rows}
