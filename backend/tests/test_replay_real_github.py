"""The full API on REAL GitHub data: responses recorded live on 2026-10-05 and replayed.

Assertions are about invariants (gate, families, explanations, logging, filters), not
about specific repositories, so they hold for any genuine recording.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from detour.config import Settings
from detour.db.models import RunCandidate
from detour.main import create_app

pytestmark = pytest.mark.db

RECORDING = Path(__file__).parent / "data" / "github_2026-10-05"


@pytest.fixture
def client(clean_db: Engine, test_db_url: str) -> Iterator[TestClient]:
    settings = Settings(  # type: ignore[call-arg]
        database_url=test_db_url,
        _env_file=None,
        GITHUB_TOKEN=None,
        DETOUR_GITHUB_CASSETTE_MODE="replay",
        DETOUR_GITHUB_CASSETTE_DIR=str(RECORDING),
        DETOUR_AS_OF_DATE="2026-10-05",
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_recorded_live_data_flows_through_the_whole_pipeline(
    client: TestClient, clean_db: Engine
) -> None:
    uid = client.post("/users", json={}).json()["id"]
    client.put(
        f"/users/{uid}/interests",
        json={"interests": [{"label": "Machine Learning"}, {"label": "Distributed Systems"}]},
    )
    body = client.post(
        f"/users/{uid}/recommendations", json={"k": 10, "compare_variant": "B1"}
    ).json()
    run, base = body["run"], body["comparison"]

    assert run["status"] == "ok", run["warnings"]  # every query was served from the recording
    stats = run["stats"]
    assert {q["family"] for q in stats["queries"]} == {"core", "bridge", "adjacent"}
    assert stats["candidates"] >= 150
    assert stats["filtered"].get("below_relevance_gate", 0) > 0  # the gate does real work
    assert stats["filtered"].get("suspicious_description", 0) > 0  # live spam repos caught
    assert len(run["items"]) == 10

    matched = {it["matched_interest"] for it in run["items"]}
    assert matched == {"Machine Learning", "Distributed Systems"}  # coverage
    owners = [it["repo"]["owner"].lower() for it in run["items"]]
    assert max(owners.count(o) for o in owners) <= 2  # owner cap
    for it in run["items"]:
        assert it["scores"]["relevance_margin"] >= 0.10
        assert it["explanation"]
        assert len(it["repo"]["description"] or "") <= 1000

    # D3 differs from the relevance-only list on the same pool, and is more novel.
    d3_ids = {it["repo"]["github_id"] for it in run["items"]}
    b1_ids = {it["repo"]["github_id"] for it in base["items"]}
    assert d3_ids != b1_ids
    mean = lambda items, key: sum(i["scores"][key] for i in items) / len(items)  # noqa: E731
    assert mean(run["items"], "novelty") > mean(base["items"], "novelty")

    with Session(clean_db) as s:
        rows = s.scalars(
            select(RunCandidate).where(RunCandidate.run_id == uuid.UUID(run["run_id"]))
        ).all()
        assert len(rows) == stats["candidates"]
        assert {r.filter_reason for r in rows if not r.shown} >= {"below_relevance_gate"}
