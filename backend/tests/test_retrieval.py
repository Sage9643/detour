"""Planner, GitHub client and cassette tests (no database, no network)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from detour.enums import QueryFamily
from detour.representation.interests import resolve_interest
from detour.retrieval.cassette import (
    CassetteMiss,
    RecordingTransport,
    ReplayTransport,
    build_transport,
    request_key,
)
from detour.retrieval.github import (
    GitHubBadRequest,
    GitHubClient,
    GitHubRateLimited,
    GitHubUnavailable,
)
from detour.retrieval.planner import RetrievalConfig, adjacent_topics, plan_adjacent, plan_primary
from detour.retrieval.service import CandidateRetriever, MemorySearchCache, RetrievalFailed

from .fake_github import FakeGitHub
from .ranking_helpers import repo

TODAY = date(2026, 10, 5)


# --- planner ----------------------------------------------------------------------------------


def test_primary_plan_has_one_core_query_per_interest_and_bridges() -> None:
    specs = [resolve_interest(x) for x in ("ML", "Distributed Systems", "CP")]
    plans = plan_primary(specs, RetrievalConfig(), TODAY)
    core = [p for p in plans if p.family is QueryFamily.CORE]
    bridge = [p for p in plans if p.family is QueryFamily.BRIDGE]
    assert [p.label for p in core] == [
        "Machine Learning",
        "Distributed Systems",
        "Competitive Programming",
    ]
    assert len(bridge) == 3
    assert bridge[0].interest_labels == ("Machine Learning", "Distributed Systems")
    for p in plans:
        assert "pushed:>2025-10-05" in p.q
        assert "stars:>=20" in p.q
        assert "archived:false" in p.q and "fork:false" in p.q
        assert "sort" not in p.q  # best-match, not stars-sorted


def test_bridges_skip_pairs_with_the_same_term_and_respect_flags() -> None:
    specs = [resolve_interest("Competitive Programming"), resolve_interest("Algorithms")]
    assert [
        p for p in plan_primary(specs, RetrievalConfig(), TODAY) if p.family is QueryFamily.BRIDGE
    ] == []
    specs = [resolve_interest("ML"), resolve_interest("Backend")]
    off = plan_primary(specs, RetrievalConfig(enable_bridge=False), TODAY)
    assert {p.family for p in off} == {QueryFamily.CORE}


def test_interest_count_is_capped() -> None:
    specs = [resolve_interest(f"interest {i}") for i in range(10)]
    plans = plan_primary(specs, RetrievalConfig(max_interests=6), TODAY)
    assert len([p for p in plans if p.family is QueryFamily.CORE]) == 6


def test_adjacent_topics_exclude_known_and_generic_and_need_support() -> None:
    results = [
        repo(1, topics=["raft", "observability", "python"]),
        repo(2, topics=["raft", "observability", "hacktoberfest"]),
        repo(3, topics=["observability", "tracing"]),
        repo(4, topics=["tracing"]),
        repo(1, topics=["raft", "observability"]),  # duplicate id counted once
    ]
    cfg = RetrievalConfig(adjacent_min_support=2, max_adjacent=3)
    ranked = adjacent_topics(results, known_topics={"raft"}, cfg=cfg)
    assert ranked == [("observability", 3), ("tracing", 2)]
    plans = plan_adjacent(results, {"raft"}, cfg, TODAY)
    assert [p.topic for p in plans] == ["observability", "tracing"]
    assert plans[0].q.startswith("topic:observability ")
    assert plan_adjacent(results, {"raft"}, RetrievalConfig(enable_adjacent=False), TODAY) == []


def test_unknown_interest_falls_back_to_raw_text() -> None:
    spec = resolve_interest("  Bioinformatics   pipelines ")
    assert spec.curated is False
    assert spec.label == "Bioinformatics pipelines"
    assert spec.core_terms == "Bioinformatics pipelines"
    assert spec.topics == ("bioinformatics-pipelines",)
    assert resolve_interest("ml").label == "Machine Learning"
    assert "consensus" in resolve_interest("distributed systems").expansion
    override = resolve_interest("ML", "only reinforcement learning please")
    assert override.expansion.endswith("only reinforcement learning please")


# --- GitHub client ----------------------------------------------------------------------------


def client_for(handler: object) -> GitHubClient:
    return GitHubClient(
        base_url="https://api.github.test",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def test_search_parses_items_and_rate_headers() -> None:
    gh = GitHubClient(base_url="https://api.github.test", transport=FakeGitHub().transport())
    res = gh.search_repositories("raft in:name,description,topics", per_page=5)
    assert res.total_count >= 1
    first = res.items[0]
    assert first.full_name == "hashicorp/raft"
    assert first.topics == ("raft", "consensus", "distributed-systems")
    assert gh.search_rate.remaining == 29
    assert gh.calls == 1


def test_transient_failure_retried_once_then_succeeds() -> None:
    fake = FakeGitHub(mode="flaky")
    gh = GitHubClient(base_url="https://api.github.test", transport=fake.transport())
    gh.search_repositories("raft")
    assert gh.calls == 2


def test_persistent_5xx_raises_unavailable() -> None:
    gh = GitHubClient(
        base_url="https://api.github.test", transport=FakeGitHub(mode="down").transport()
    )
    with pytest.raises(GitHubUnavailable):
        gh.search_repositories("raft")
    assert gh.calls == 2


def test_rate_limit_raises_and_later_calls_fail_fast() -> None:
    fake = FakeGitHub(mode="rate_limited")
    gh = GitHubClient(base_url="https://api.github.test", transport=fake.transport())
    with pytest.raises(GitHubRateLimited) as exc:
        gh.search_repositories("raft")
    assert exc.value.reset_at is not None
    with pytest.raises(GitHubRateLimited):
        gh.search_repositories("paxos")
    assert len(fake.calls) == 1, "known-exhausted budget must not spend another request"


def test_timeouts_and_network_errors_become_unavailable() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    gh = client_for(boom)
    with pytest.raises(GitHubUnavailable):
        gh.search_repositories("raft")


def test_invalid_query_is_not_retried() -> None:
    gh = client_for(lambda r: httpx.Response(422, json={"message": "Validation Failed"}))
    with pytest.raises(GitHubBadRequest):
        gh.search_repositories("bad:::")
    assert gh.calls == 1


def test_token_sent_as_bearer_header_only_when_configured() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"total_count": 0, "items": []})

    GitHubClient(
        base_url="https://x.test", transport=httpx.MockTransport(handler)
    ).search_repositories("a")
    GitHubClient(
        base_url="https://x.test", token="t0ken", transport=httpx.MockTransport(handler)
    ).search_repositories("a")
    assert seen == [None, "Bearer t0ken"]


# --- retriever with memory cache ------------------------------------------------------------------


def test_retriever_merges_families_and_dedupes() -> None:
    gh = GitHubClient(base_url="https://api.github.test", transport=FakeGitHub().transport())
    retriever = CandidateRetriever(gh, MemorySearchCache(), cache_ttl_s=3600)
    specs = [resolve_interest("Distributed Systems"), resolve_interest("Machine Learning")]
    out = retriever.retrieve(specs, known_topics=set(), cfg=RetrievalConfig(adjacent_min_support=2))
    horovod = out.candidates[204]
    assert {QueryFamily.CORE, QueryFamily.BRIDGE} <= horovod.families
    assert horovod.bridge_pairs == [("Distributed Systems", "Machine Learning")]
    assert len(out.candidates) == len({c.repo.github_id for c in out.candidates.values()})
    assert out.github_calls == len(out.queries)
    assert not out.degraded

    again = retriever.retrieve(
        specs, known_topics=set(), cfg=RetrievalConfig(adjacent_min_support=2)
    )
    assert again.github_calls == 0
    assert {q.source for q in again.queries} == {"cache"}


def test_retriever_raises_when_every_query_fails() -> None:
    gh = GitHubClient(
        base_url="https://api.github.test", transport=FakeGitHub(mode="down").transport()
    )
    retriever = CandidateRetriever(gh, MemorySearchCache(), cache_ttl_s=3600)
    with pytest.raises(RetrievalFailed):
        retriever.retrieve([resolve_interest("ML")], set(), RetrievalConfig())


# --- cassette record / replay ---------------------------------------------------------


def test_cassette_records_then_replays_without_network(tmp_path: Path) -> None:
    fake = FakeGitHub()
    rec = GitHubClient(
        base_url="https://api.github.test",
        token="secret-token",
        transport=RecordingTransport(tmp_path, inner=fake.transport()),
    )
    live = rec.search_repositories("raft", per_page=3)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "secret-token" not in files[0].read_text()
    assert json.loads(files[0].read_text())["status"] == 200

    replay = GitHubClient(base_url="https://api.github.test", transport=ReplayTransport(tmp_path))
    again = replay.search_repositories("raft", per_page=3)
    assert [r.github_id for r in again.items] == [r.github_id for r in live.items]
    with pytest.raises(GitHubUnavailable):
        replay.search_repositories("something never recorded")


def test_request_key_ignores_param_order_and_headers() -> None:
    a = httpx.Request("GET", "https://x/search?q=a&per_page=5", headers={"Authorization": "x"})
    b = httpx.Request("GET", "https://x/search?per_page=5&q=a")
    assert request_key(a) == request_key(b)


def test_build_transport_modes(tmp_path: Path) -> None:
    assert build_transport("off", None) is None
    assert isinstance(build_transport("replay", str(tmp_path)), ReplayTransport)
    with pytest.raises(ValueError):
        build_transport("replay", None)
    with pytest.raises(ValueError):
        build_transport("bogus", str(tmp_path))
    assert issubclass(CassetteMiss, httpx.TransportError)
