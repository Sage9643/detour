"""Thin GitHub REST client.

Responsibilities: timeouts, one retry on transient failures, rate-limit detection, and
mapping HTTP failures to typed errors the service layer can degrade on. It does NOT cache;
caching is the retrieval service's job (detour.retrieval.service).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from detour.retrieval.cassette import CassetteMiss
from detour.retrieval.models import RepoRecord

logger = logging.getLogger(__name__)

_API_VERSION = "2022-11-28"
_RETRY_DELAY_S = 0.5


class GitHubError(Exception):
    """Base class. Messages never include tokens."""


class GitHubUnavailable(GitHubError):
    """Network failure, timeout, or 5xx after retry."""


class GitHubRateLimited(GitHubError):
    def __init__(self, reset_at: datetime | None) -> None:
        self.reset_at = reset_at
        when = reset_at.isoformat() if reset_at else "unknown"
        super().__init__(f"GitHub rate limit exhausted (resets at {when})")


class GitHubBadRequest(GitHubError):
    """422 and similar: the query itself is invalid. Not retried."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    total_count: int
    items: list[RepoRecord]


@dataclass(frozen=True, slots=True)
class StarredRepo:
    starred_at: datetime
    repo: RepoRecord


@dataclass(slots=True)
class RateLimitState:
    remaining: int | None = None
    limit: int | None = None
    reset_at: datetime | None = None


class GitHubClient:
    def __init__(
        self,
        base_url: str = "https://api.github.com",
        token: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _API_VERSION,
            "User-Agent": "detour-recommender",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.authenticated = bool(token)
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )
        self._lock = threading.Lock()
        self.search_rate = RateLimitState()
        self.core_rate = RateLimitState()
        self.calls = 0  # actual HTTP requests issued (for per-run accounting)

    def close(self) -> None:
        self._http.close()

    # -- public API --------------------------------------------------------------------

    def search_repositories(
        self, q: str, per_page: int = 50, sort: str | None = None
    ) -> SearchResult:
        rate = self.search_rate
        if rate.remaining == 0 and rate.reset_at and rate.reset_at > datetime.now(UTC):
            # Known to be exhausted: fail fast instead of spending a request on a 403.
            raise GitHubRateLimited(rate.reset_at)
        params: dict[str, str | int] = {"q": q, "per_page": per_page}
        if sort:
            params["sort"] = sort
            params["order"] = "desc"
        data = self._get_json("/search/repositories", params, self.search_rate)
        items = [RepoRecord.from_github(it) for it in data.get("items", [])]
        return SearchResult(total_count=int(data.get("total_count", 0)), items=items)

    def get(self, path: str, params: dict[str, str | int], accept: str | None = None) -> Any:
        """Generic GET on the core API (used by offline dataset collection)."""
        return self._get_json(path, params, self.core_rate, accept=accept)

    def starred(self, username: str, max_pages: int = 3) -> list[StarredRepo]:
        """A user's starred repositories with timestamps (newest first)."""
        out: list[StarredRepo] = []
        for page in range(1, max_pages + 1):
            data = self._get_json(
                f"/users/{username}/starred",
                {"per_page": 100, "page": page, "sort": "created", "direction": "desc"},
                self.core_rate,
                accept="application/vnd.github.star+json",
            )
            if not isinstance(data, list) or not data:
                break
            for row in data:
                out.append(
                    StarredRepo(
                        starred_at=datetime.fromisoformat(row["starred_at"].replace("Z", "+00:00")),
                        repo=RepoRecord.from_github(row["repo"]),
                    )
                )
            if len(data) < 100:
                break
        return out

    # -- internals -------------------------------------------------------------------------

    def _get_json(
        self,
        path: str,
        params: dict[str, str | int],
        rate: RateLimitState,
        accept: str | None = None,
    ) -> Any:
        headers = {"Accept": accept} if accept else None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with self._lock:
                    self.calls += 1
                response = self._http.get(path, params=params, headers=headers)
            except httpx.HTTPError as exc:  # timeouts, DNS, connection, decoding, cassette miss
                last_error = exc
                if isinstance(exc, CassetteMiss):  # deterministic: retrying cannot help
                    raise GitHubUnavailable(f"not in recording: {path}") from exc
                logger.warning("github request failed path=%s error=%s", path, type(exc).__name__)
                if attempt == 0:
                    time.sleep(_RETRY_DELAY_S)
                    continue
                raise GitHubUnavailable(f"GitHub request failed: {type(exc).__name__}") from exc

            self._update_rate(rate, response)
            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise GitHubUnavailable(f"GitHub returned invalid JSON for {path}") from exc
            if status in (403, 429) and self._is_rate_limited(response):
                raise GitHubRateLimited(rate.reset_at)
            if status == 422:
                raise GitHubBadRequest(f"GitHub rejected query (422) for {path}")
            if status == 404:
                raise GitHubBadRequest(f"Not found: {path}")
            if status >= 500 and attempt == 0:
                logger.warning("github %s on %s; retrying once", status, path)
                time.sleep(_RETRY_DELAY_S)
                continue
            raise GitHubUnavailable(f"GitHub returned HTTP {status} for {path}")
        raise GitHubUnavailable("GitHub request failed") from last_error

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        if response.headers.get("retry-after"):
            return True
        return "rate limit" in response.text.lower()

    @staticmethod
    def _update_rate(rate: RateLimitState, response: httpx.Response) -> None:
        h = response.headers
        if "x-ratelimit-remaining" in h:
            rate.remaining = int(h["x-ratelimit-remaining"])
        if "x-ratelimit-limit" in h:
            rate.limit = int(h["x-ratelimit-limit"])
        if "x-ratelimit-reset" in h:
            rate.reset_at = datetime.fromtimestamp(int(h["x-ratelimit-reset"]), tz=UTC)
