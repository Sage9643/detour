"""Candidate retrieval: execute the planned queries with caching and graceful degradation.

Flow:
  plan_primary -> (cache | GitHub) -> core results -> plan_adjacent -> (cache | GitHub)
  -> merge + dedupe by github_id, remembering every family that retrieved each repo.

Degradation rules (never fabricate results):
  fresh cache hit           -> served, source="cache"
  miss/stale + GitHub ok    -> fetched, source="live", cache updated
  stale + GitHub failing    -> stale entry served, source="stale", warning recorded
  miss + GitHub failing     -> query contributes nothing, source="failed", warning recorded
  every query failed        -> RetrievalFailed (the API turns this into a 503)
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from detour.db.models import SearchCacheEntry
from detour.db.repo_store import load_records, upsert_repositories
from detour.enums import QueryFamily
from detour.representation.interests import InterestSpec
from detour.retrieval.github import (
    GitHubBadRequest,
    GitHubClient,
    GitHubError,
    GitHubRateLimited,
    SearchResult,
)
from detour.retrieval.models import Candidate, PlannedQuery, RepoRecord
from detour.retrieval.planner import RetrievalConfig, plan_adjacent, plan_primary

logger = logging.getLogger(__name__)

Source = Literal["live", "cache", "stale", "failed"]


@dataclass(frozen=True, slots=True)
class CachedSearch:
    total_count: int
    items: list[RepoRecord]
    fetched_at: datetime


class SearchCache(Protocol):
    def get(self, key: str) -> CachedSearch | None: ...

    def put(
        self, key: str, pq: PlannedQuery, result: SearchResult, fetched_at: datetime
    ) -> None: ...


class MemorySearchCache:
    """In-process cache for offline tooling (evaluation collection) and tests."""

    def __init__(self) -> None:
        self._data: dict[str, CachedSearch] = {}

    def get(self, key: str) -> CachedSearch | None:
        return self._data.get(key)

    def put(self, key: str, pq: PlannedQuery, result: SearchResult, fetched_at: datetime) -> None:
        self._data[key] = CachedSearch(result.total_count, list(result.items), fetched_at)


class DbSearchCache:
    """search_cache rows hold ordered repo ids; metadata comes from `repositories`."""

    def __init__(self, session: Session) -> None:
        self._s = session

    def get(self, key: str) -> CachedSearch | None:
        row = self._s.scalar(select(SearchCacheEntry).where(SearchCacheEntry.query_key == key))
        if row is None:
            return None
        records = load_records(self._s, row.repo_ids)
        items = [records[i] for i in row.repo_ids if i in records]
        return CachedSearch(row.total_count, items, row.fetched_at)

    def put(self, key: str, pq: PlannedQuery, result: SearchResult, fetched_at: datetime) -> None:
        upsert_repositories(self._s, result.items, fetched_at)
        values = {
            "query_key": key,
            "query": pq.q,
            "sort": pq.sort,
            "per_page": pq.per_page,
            "total_count": result.total_count,
            "repo_ids": [r.github_id for r in result.items],
            "fetched_at": fetched_at,
        }
        stmt = insert(SearchCacheEntry).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[SearchCacheEntry.query_key],
            set_={k: stmt.excluded[k] for k in values if k != "query_key"},
        )
        self._s.execute(stmt)


def cache_key(pq: PlannedQuery) -> str:
    raw = f"{' '.join(pq.q.split())}|{pq.sort or ''}|{pq.per_page}"
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(slots=True)
class QueryOutcome:
    query: PlannedQuery
    source: Source
    result_count: int
    total_count: int | None
    latency_ms: float
    fetched_at: datetime | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.query.family.value,
            "label": self.query.label,
            "q": self.query.q,
            "source": self.source,
            "result_count": self.result_count,
            "total_count": self.total_count,
            "latency_ms": round(self.latency_ms, 1),
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "error": self.error,
        }


@dataclass(slots=True)
class RetrievalOutcome:
    candidates: dict[int, Candidate]
    queries: list[QueryOutcome]
    warnings: list[str] = field(default_factory=list)
    github_calls: int = 0

    @property
    def degraded(self) -> bool:
        return any(q.source in ("stale", "failed") for q in self.queries)

    @property
    def oldest_data_at(self) -> datetime | None:
        stamps = [q.fetched_at for q in self.queries if q.fetched_at]
        return min(stamps) if stamps else None


class RetrievalFailed(Exception):
    """No query produced any candidates (GitHub down/rate-limited and nothing cached)."""


class CandidateRetriever:
    def __init__(
        self,
        client: GitHubClient,
        cache: SearchCache,
        cache_ttl_s: int,
        max_workers: int = 4,
    ) -> None:
        self._client = client
        self._cache = cache
        self._ttl_s = cache_ttl_s
        self._workers = max_workers

    def retrieve(
        self,
        interests: list[InterestSpec],
        known_topics: set[str],
        cfg: RetrievalConfig,
        now: datetime | None = None,
    ) -> RetrievalOutcome:
        now = now or datetime.now(UTC)
        today: date = now.date()
        calls_before = self._client.calls

        primary = plan_primary(interests, cfg, today)
        outcomes, results = self._run(primary, now)

        core_items = [
            r
            for o, items in zip(outcomes, results, strict=True)
            if o.query.family is QueryFamily.CORE
            for r in items
        ]
        adjacent = plan_adjacent(core_items, known_topics, cfg, today)
        if adjacent:
            adj_outcomes, adj_results = self._run(adjacent, now)
            outcomes += adj_outcomes
            results += adj_results

        candidates: dict[int, Candidate] = {}
        for outcome, items in zip(outcomes, results, strict=True):
            pq = outcome.query
            for repo in items:
                cand = candidates.get(repo.github_id)
                if cand is None:
                    cand = candidates[repo.github_id] = Candidate(repo=repo)
                cand.families.add(pq.family)
                cand.query_labels.append(pq.label)
                if pq.family is QueryFamily.ADJACENT and pq.topic:
                    cand.adjacent_topics.add(pq.topic)
                if pq.family is QueryFamily.BRIDGE:
                    cand.bridge_pairs.append(pq.interest_labels)

        warnings = [
            f"{o.query.label}: {o.error}" for o in outcomes if o.source in ("stale", "failed")
        ]
        if outcomes and all(o.source == "failed" for o in outcomes):
            raise RetrievalFailed("; ".join(warnings) or "all GitHub queries failed")
        return RetrievalOutcome(
            candidates=candidates,
            queries=outcomes,
            warnings=warnings,
            github_calls=self._client.calls - calls_before,
        )

    # ------------------------------------------------------------------------------------

    def _run(
        self, plans: list[PlannedQuery], now: datetime
    ) -> tuple[list[QueryOutcome], list[list[RepoRecord]]]:
        # 1. Cache lookups on the calling thread (DB sessions are not thread-safe).
        cached: list[CachedSearch | None] = []
        to_fetch: list[int] = []
        for i, pq in enumerate(plans):
            entry = self._cache.get(cache_key(pq))
            cached.append(entry)
            fresh = entry is not None and (now - entry.fetched_at).total_seconds() < self._ttl_s
            if not fresh:
                to_fetch.append(i)

        # 2. Network calls in parallel.
        fetched: dict[int, tuple[SearchResult | None, str | None, float]] = {}
        if to_fetch:
            with ThreadPoolExecutor(max_workers=min(self._workers, len(to_fetch))) as pool:
                futures = {i: pool.submit(self._fetch, plans[i]) for i in to_fetch}
                fetched = {i: f.result() for i, f in futures.items()}

        # 3. Cache writes + outcome bookkeeping, back on the calling thread.
        outcomes: list[QueryOutcome] = []
        results: list[list[RepoRecord]] = []
        for i, pq in enumerate(plans):
            entry = cached[i]
            if i not in fetched:
                assert entry is not None
                outcomes.append(
                    QueryOutcome(
                        pq, "cache", len(entry.items), entry.total_count, 0.0, entry.fetched_at
                    )
                )
                results.append(entry.items)
                continue
            result, error, latency = fetched[i]
            if result is not None:
                self._cache.put(cache_key(pq), pq, result, now)
                outcomes.append(
                    QueryOutcome(pq, "live", len(result.items), result.total_count, latency, now)
                )
                results.append(result.items)
            elif entry is not None:
                outcomes.append(
                    QueryOutcome(
                        pq,
                        "stale",
                        len(entry.items),
                        entry.total_count,
                        latency,
                        entry.fetched_at,
                        f"served stale cache ({error})",
                    )
                )
                results.append(entry.items)
            else:
                outcomes.append(QueryOutcome(pq, "failed", 0, None, latency, None, error))
                results.append([])
        return outcomes, results

    def _fetch(self, pq: PlannedQuery) -> tuple[SearchResult | None, str | None, float]:
        t0 = time.perf_counter()
        try:
            res = self._client.search_repositories(pq.q, per_page=pq.per_page, sort=pq.sort)
            return res, None, (time.perf_counter() - t0) * 1000
        except GitHubRateLimited as exc:
            return None, str(exc), (time.perf_counter() - t0) * 1000
        except GitHubBadRequest as exc:
            return None, str(exc), (time.perf_counter() - t0) * 1000
        except GitHubError as exc:
            logger.warning("github search failed label=%s error=%s", pq.label, exc)
            return None, str(exc), (time.perf_counter() - t0) * 1000
