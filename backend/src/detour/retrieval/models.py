"""Plain data types shared by retrieval, representation and ranking.

These are deliberately framework-free (no SQLAlchemy, no FastAPI) so ranking can run on
frozen offline snapshots exactly as it runs online.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from detour.enums import QueryFamily


@dataclass(frozen=True, slots=True)
class RepoRecord:
    """Point-in-time repository metadata as returned by GitHub."""

    github_id: int
    full_name: str
    owner_login: str
    name: str
    description: str | None
    topics: tuple[str, ...]
    language: str | None
    stars: int
    forks: int
    archived: bool
    is_fork: bool
    license_spdx: str | None
    html_url: str
    gh_created_at: datetime | None
    pushed_at: datetime | None

    @staticmethod
    def from_github(item: dict[str, Any]) -> "RepoRecord":
        lic = item.get("license") or {}
        return RepoRecord(
            github_id=int(item["id"]),
            full_name=item["full_name"],
            owner_login=(item.get("owner") or {}).get("login") or item["full_name"].split("/")[0],
            name=item["name"],
            description=item.get("description"),
            topics=tuple(item.get("topics") or ()),
            language=item.get("language"),
            stars=int(item.get("stargazers_count") or 0),
            forks=int(item.get("forks_count") or 0),
            archived=bool(item.get("archived")),
            is_fork=bool(item.get("fork")),
            license_spdx=lic.get("spdx_id") if isinstance(lic, dict) else None,
            html_url=item["html_url"],
            gh_created_at=_parse_ts(item.get("created_at")),
            pushed_at=_parse_ts(item.get("pushed_at")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "github_id": self.github_id,
            "full_name": self.full_name,
            "owner_login": self.owner_login,
            "name": self.name,
            "description": self.description,
            "topics": list(self.topics),
            "language": self.language,
            "stars": self.stars,
            "forks": self.forks,
            "archived": self.archived,
            "is_fork": self.is_fork,
            "license_spdx": self.license_spdx,
            "html_url": self.html_url,
            "gh_created_at": self.gh_created_at.isoformat() if self.gh_created_at else None,
            "pushed_at": self.pushed_at.isoformat() if self.pushed_at else None,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "RepoRecord":
        return RepoRecord(
            github_id=int(d["github_id"]),
            full_name=d["full_name"],
            owner_login=d["owner_login"],
            name=d["name"],
            description=d.get("description"),
            topics=tuple(d.get("topics") or ()),
            language=d.get("language"),
            stars=int(d["stars"]),
            forks=int(d["forks"]),
            archived=bool(d["archived"]),
            is_fork=bool(d["is_fork"]),
            license_spdx=d.get("license_spdx"),
            html_url=d["html_url"],
            gh_created_at=_parse_ts(d.get("gh_created_at")),
            pushed_at=_parse_ts(d.get("pushed_at")),
        )


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    """One GitHub search the planner decided to run, and why."""

    family: QueryFamily
    q: str
    per_page: int
    label: str  # human-readable, e.g. "Machine Learning + Distributed Systems"
    interest_labels: tuple[str, ...] = ()
    topic: str | None = None  # for adjacent queries: the exploratory topic
    sort: str | None = None  # None = GitHub best-match


@dataclass(slots=True)
class Candidate:
    repo: RepoRecord
    families: set[QueryFamily] = field(default_factory=set)
    query_labels: list[str] = field(default_factory=list)
    adjacent_topics: set[str] = field(default_factory=set)
    bridge_pairs: list[tuple[str, ...]] = field(default_factory=list)
