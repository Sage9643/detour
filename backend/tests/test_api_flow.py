"""End-to-end API tests: real PostgreSQL, real embedding model, fake GitHub transport.

These exercise the full product loop:
  create user -> interests -> recommendations -> feedback -> different recommendations
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from detour.config import Settings
from detour.db.models import Feedback, RecommendationRun, RunCandidate
from detour.main import create_app
from detour.retrieval.github import GitHubClient

from .fake_github import FakeGitHub

pytestmark = pytest.mark.db

INTERESTS = [{"label": "Distributed Systems"}, {"label": "Machine Learning"}]


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub()


def make_client(test_db_url: str, fake: FakeGitHub, **settings: Any) -> TestClient:
    s = Settings(database_url=test_db_url, _env_file=None, **settings)  # type: ignore[call-arg]
    gh = GitHubClient(base_url="https://api.github.test", transport=fake.transport())
    return TestClient(create_app(s, github=gh))


@pytest.fixture
def client(clean_db: Engine, test_db_url: str, fake: FakeGitHub) -> Iterator[TestClient]:
    with make_client(test_db_url, fake) as c:
        yield c


def new_user(client: TestClient, interests: list[dict[str, str]] = INTERESTS) -> str:
    uid = client.post("/users", json={}).json()["id"]
    r = client.put(f"/users/{uid}/interests", json={"interests": interests})
    assert r.status_code == 200, r.text
    return str(uid)


def recommend(client: TestClient, uid: str, **body: Any) -> dict[str, Any]:
    r = client.post(f"/users/{uid}/recommendations", json=body)
    assert r.status_code == 200, r.text
    return dict(r.json())


# --- happy path -----------------------------------------------------------------------------


def test_full_recommendation_run_is_served_and_logged(client: TestClient, clean_db: Engine) -> None:
    uid = new_user(client)
    body = recommend(client, uid, k=5)
    run = body["run"]
    assert run["status"] == "ok"
    assert run["variant"] == "D3"
    assert 1 <= len(run["items"]) <= 5
    stats = run["stats"]
    families = {q["family"] for q in stats["queries"]}
    assert {"core", "bridge"} <= families
    assert stats["candidates"] > len(run["items"])
    assert stats["embedding_model"] == "wordllama-l2_supercat-256"

    for it in run["items"]:
        assert it["explanation"], "every recommendation carries an explanation"
        assert it["matched_interest"] in {"Distributed Systems", "Machine Learning"}
        assert it["scores"]["relevance_raw"] >= 0.18  # passed the gate
        # The archived repo and the fork must never be shown.
        assert it["repo"]["github_id"] not in (501, 502)

    with Session(clean_db) as s:
        db_run = s.get(RecommendationRun, uuid.UUID(run["run_id"]))
        assert db_run is not None
        assert db_run.candidate_count == stats["candidates"]
        rows = s.scalars(select(RunCandidate).where(RunCandidate.run_id == db_run.id)).all()
        assert len(rows) == stats["candidates"]  # EVERY candidate logged, not only shown
        shown = sorted((r for r in rows if r.shown), key=lambda r: r.position or 0)
        assert [r.repo_id for r in shown] == [it["repo"]["github_id"] for it in run["items"]]
        reasons = {r.repo_id: r.filter_reason for r in rows}
        assert reasons[501] == "fork"
        assert reasons[502] == "archived"
        assert all(r.features.get("stars") is not None for r in rows)
        assert set(db_run.timings_ms) >= {"retrieval_ms", "embedding_ms", "ranking_ms", "total_ms"}


def test_second_request_uses_caches_and_shows_new_items(
    client: TestClient, fake: FakeGitHub
) -> None:
    uid = new_user(client)
    first = recommend(client, uid, k=4)["run"]
    calls_after_first = len(fake.calls)
    second = recommend(client, uid, k=4)["run"]

    assert len(fake.calls) == calls_after_first, "second run must not call GitHub"
    assert {q["source"] for q in second["stats"]["queries"]} == {"cache"}
    assert second["stats"]["github_calls"] == 0
    emb = second["stats"]["embedding_cache"]
    assert emb["misses"] == 0 and emb["hits"] == second["stats"]["candidates"]
    first_ids = {it["repo"]["github_id"] for it in first["items"]}
    second_ids = {it["repo"]["github_id"] for it in second["items"]}
    assert first_ids.isdisjoint(second_ids), "recently shown repos are excluded"


def test_comparison_variant_ranks_the_same_pool(client: TestClient, clean_db: Engine) -> None:
    uid = new_user(client)
    body = recommend(client, uid, k=5, compare_variant="B1")
    run, cmp = body["run"], body["comparison"]
    assert cmp is not None and cmp["variant"] == "B1"
    assert cmp["stats"]["candidates"] == run["stats"]["candidates"]
    with Session(clean_db) as s:
        b1 = s.get(RecommendationRun, uuid.UUID(cmp["run_id"]))
        assert b1 is not None and b1.config["paired_with_run"] == run["run_id"]


# --- feedback loop ------------------------------------------------------------------------------


def feedback(client: TestClient, uid: str, repo_id: int, ftype: str, run_id: str | None) -> Any:
    return client.post(
        f"/users/{uid}/feedback", json={"repo_id": repo_id, "type": ftype, "run_id": run_id}
    )


def test_already_know_is_familiarity_not_dislike(client: TestClient, clean_db: Engine) -> None:
    uid = new_user(client)
    run = recommend(client, uid, k=5)["run"]
    target = run["items"][0]["repo"]["github_id"]

    r = feedback(client, uid, target, "ALREADY_KNOW", run["run_id"])
    assert r.status_code == 201, r.text
    prof = r.json()["profile"]
    assert prof["known_count"] == 1
    assert prof["cold_start"] is False
    assert prof["negatives"] == []  # not treated as a negative preference
    assert all(i["effective_weight"] == 1.0 for i in prof["interests"])

    nxt = recommend(client, uid, k=5)["run"]
    with Session(clean_db) as s:
        row = s.scalar(
            select(RunCandidate).where(
                RunCandidate.run_id == uuid.UUID(nxt["run_id"]), RunCandidate.repo_id == target
            )
        )
        assert row is not None and row.filter_reason == "already_known"
        # novelty now has a user-relative familiarity component
        eligible = s.scalars(
            select(RunCandidate).where(
                RunCandidate.run_id == uuid.UUID(nxt["run_id"]),
                RunCandidate.filter_reason.is_(None),
            )
        ).all()
        assert eligible and all(c.unfamiliarity is not None for c in eligible)


def test_interested_feedback_becomes_a_relevance_anchor(client: TestClient) -> None:
    uid = new_user(client)
    run = recommend(client, uid, k=5)["run"]
    liked = run["items"][0]["repo"]
    r = feedback(client, uid, liked["github_id"], "INTERESTED", run["run_id"])
    assert r.json()["profile"]["positives"] == [liked["full_name"]]
    nxt = recommend(client, uid, k=10)["run"]
    assert liked["github_id"] not in {it["repo"]["github_id"] for it in nxt["items"]}


def test_repeated_rejections_decay_the_matched_interest(client: TestClient) -> None:
    uid = new_user(client)
    rejected: list[str] = []
    for _ in range(3):
        run = recommend(client, uid, k=10)["run"]
        for it in run["items"]:
            if it["matched_interest"] == "Machine Learning" and len(rejected) < 3:
                r = feedback(client, uid, it["repo"]["github_id"], "NOT_INTERESTED", run["run_id"])
                assert r.status_code == 201
                rejected.append(it["repo"]["full_name"])
        if len(rejected) >= 3:
            break
    assert len(rejected) == 3, "fixture should surface >= 3 ML repos"
    prof = client.get(f"/users/{uid}").json()["profile"]
    ml = next(i for i in prof["interests"] if i["label"] == "Machine Learning")
    ds = next(i for i in prof["interests"] if i["label"] == "Distributed Systems")
    assert ml["rejections"] == 3
    assert ml["effective_weight"] == pytest.approx(0.85)  # one rejection past decay_after=2
    assert ds["effective_weight"] == 1.0
    assert sorted(prof["negatives"]) == sorted(rejected)


def test_changing_mind_latest_feedback_wins(client: TestClient, clean_db: Engine) -> None:
    uid = new_user(client)
    run = recommend(client, uid, k=3)["run"]
    rid = run["items"][0]["repo"]["github_id"]
    feedback(client, uid, rid, "NOT_INTERESTED", run["run_id"])
    prof = feedback(client, uid, rid, "INTERESTED", run["run_id"]).json()["profile"]
    assert prof["negatives"] == [] and len(prof["positives"]) == 1
    with Session(clean_db) as s:
        assert s.scalar(select(func.count()).select_from(Feedback)) == 2  # history kept


def test_feedback_validation(client: TestClient) -> None:
    uid = new_user(client)
    other = new_user(client)
    run = recommend(client, uid, k=3)["run"]
    shown = run["items"][0]["repo"]["github_id"]
    assert feedback(client, uid, 999_999, "INTERESTED", None).status_code == 404
    assert feedback(client, other, shown, "INTERESTED", run["run_id"]).status_code == 404
    assert feedback(client, uid, 501, "INTERESTED", run["run_id"]).status_code == 422  # not shown
    bad = client.post(f"/users/{uid}/feedback", json={"repo_id": shown, "type": "MEH"})
    assert bad.status_code == 422
    # Feedback outside a run (e.g. onboarding) is allowed for known repositories.
    assert feedback(client, uid, shown, "ALREADY_KNOW", None).status_code == 201


# --- input validation --------------------------------------------------------------------


def test_interest_validation(client: TestClient) -> None:
    uid = client.post("/users", json={}).json()["id"]
    too_many = [{"label": f"topic {i}"} for i in range(7)]
    assert client.put(f"/users/{uid}/interests", json={"interests": too_many}).status_code == 422
    dupes = [{"label": "ML"}, {"label": "ml"}]
    assert client.put(f"/users/{uid}/interests", json={"interests": dupes}).status_code == 422
    assert client.put(f"/users/{uid}/interests", json={"interests": []}).status_code == 422
    blank = [{"label": "   "}]
    assert client.put(f"/users/{uid}/interests", json={"interests": blank}).status_code == 422


def test_interest_replacement_preserves_rows(client: TestClient) -> None:
    uid = new_user(client)
    out = client.put(f"/users/{uid}/interests", json={"interests": [{"label": "ml"}]}).json()
    assert [i["label"] for i in out] == ["ml"]
    assert out[0]["curated"] is True and "neural networks" in out[0]["expansion"]
    back = client.put(f"/users/{uid}/interests", json={"interests": INTERESTS}).json()
    assert {i["label"] for i in back} == {"Distributed Systems", "Machine Learning"}


def test_recommend_requires_user_and_interests(client: TestClient) -> None:
    missing = client.post(f"/users/{uuid.uuid4()}/recommendations", json={})
    assert missing.status_code == 404
    uid = client.post("/users", json={}).json()["id"]
    r = client.post(f"/users/{uid}/recommendations", json={})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "no_interests"


def test_catalog_lists_curated_interests(client: TestClient) -> None:
    labels = {e["label"] for e in client.get("/interests/catalog").json()}
    assert {"Machine Learning", "Distributed Systems", "Competitive Programming"} <= labels


# --- failure handling ------------------------------------------------------------------------


def test_github_down_without_cache_returns_503_and_logs_failed_run(
    clean_db: Engine, test_db_url: str
) -> None:
    fake = FakeGitHub(mode="down")
    with make_client(test_db_url, fake) as c:
        uid = new_user(c)
        r = c.post(f"/users/{uid}/recommendations", json={})
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "github_unavailable"
    with Session(clean_db) as s:
        statuses = s.scalars(select(RecommendationRun.status)).all()
        assert statuses == ["failed"]


def test_github_down_with_stale_cache_serves_degraded_results(
    clean_db: Engine, test_db_url: str
) -> None:
    fake = FakeGitHub()
    # TTL 0: every cache entry is immediately stale, so a refresh is attempted each time.
    with make_client(test_db_url, fake, DETOUR_SEARCH_CACHE_TTL_S=0) as c:
        uid = new_user(c)
        recommend(c, uid, k=3)
        fake.mode = "down"
        run = recommend(c, uid, k=3)["run"]
    assert run["status"] == "degraded"
    assert run["warnings"]
    assert {q["source"] for q in run["stats"]["queries"]} == {"stale"}
    assert run["items"], "stale-but-real candidates are still ranked"


def test_rate_limit_is_reported_not_retried(clean_db: Engine, test_db_url: str) -> None:
    fake = FakeGitHub(mode="rate_limited")
    with make_client(test_db_url, fake) as c:
        uid = new_user(c)
        r = c.post(f"/users/{uid}/recommendations", json={})
    assert r.status_code == 503
    assert "rate limit" in r.json()["detail"]["message"].lower()
    # first 403 marks the budget exhausted; later queries fail fast without HTTP calls
    assert len(fake.calls) < 4


def test_transient_5xx_is_retried_once(clean_db: Engine, test_db_url: str) -> None:
    fake = FakeGitHub(mode="flaky")
    with make_client(test_db_url, fake) as c:
        uid = new_user(c)
        run = recommend(c, uid, k=3)["run"]
    assert run["status"] == "ok"
    assert {q["source"] for q in run["stats"]["queries"]} == {"live"}


def test_database_unavailable_returns_503_without_leaking_details(fake: FakeGitHub) -> None:
    url = "postgresql+psycopg://nobody:secretpw@127.0.0.1:1/none"
    with make_client(url, fake, DETOUR_DB_CONNECT_TIMEOUT_S=1) as c:
        r = c.post("/users", json={})
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "database_unavailable"
    assert "secretpw" not in r.text


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    r = client.options(
        "/users",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_blind_study_feedback_is_attributed_to_each_system(
    client: TestClient, clean_db: Engine
) -> None:
    from detour.eval.study import analyse, wilson

    uid = new_user(client)
    body = recommend(client, uid, k=4, compare_variant="B1")
    d3, b1 = body["run"], body["comparison"]
    # rate one item unique to each list, from within that list (as the study UI does)
    d3_set = {x["repo"]["github_id"] for x in d3["items"]}
    b1_set = {x["repo"]["github_id"] for x in b1["items"]}
    d3_only, b1_only = sorted(d3_set - b1_set), sorted(b1_set - d3_set)
    assert d3_only and b1_only, "fixture should produce lists that differ"
    assert feedback(client, uid, d3_only[0], "INTERESTED", d3["run_id"]).status_code == 201
    assert feedback(client, uid, b1_only[0], "NOT_INTERESTED", b1["run_id"]).status_code == 201

    out = analyse(clean_db)
    assert out["pairs"] == 1
    assert out["per_variant"]["D3"]["INTERESTED"] == 1
    assert out["per_variant"]["B1"]["NOT_INTERESTED"] == 1
    assert out["d3_wins"] == 1 and out["d3_losses"] == 0
    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.98


def test_study_rating_of_a_repo_shown_in_both_lists_counts_for_both(
    client: TestClient, clean_db: Engine
) -> None:
    from detour.eval.study import analyse

    uid = new_user(client)
    body = recommend(client, uid, k=6, compare_variant="B1")
    d3, b1 = body["run"], body["comparison"]
    shared = {i["repo"]["github_id"] for i in d3["items"]} & {
        i["repo"]["github_id"] for i in b1["items"]
    }
    assert shared, "fixture should produce overlapping lists"
    feedback(client, uid, next(iter(shared)), "ALREADY_KNOW", b1["run_id"])
    out = analyse(clean_db)
    assert out["per_variant"]["D3"]["ALREADY_KNOW"] == 1
    assert out["per_variant"]["B1"]["ALREADY_KNOW"] == 1
