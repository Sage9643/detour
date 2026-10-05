"""Collect the offline evaluation dataset from public GitHub data (resumable).

Dataset: "GitHub stars, temporal split" (docs/evaluation.md §3)
  1. users    sample owners of the newest forks of a few seed repositories spanning several
              areas (recently active, engaged developers; the stargazers endpoint requires
              authentication), then fetch each user's starred repositories WITH timestamps
  2. split    cutoff t = collection date - window. Stars before t = what the user already
              knew (familiarity + derived interests); stars after t = held-out H
  3. pools    run Detour's real retrieval for each user's derived interests and record
              every GitHub response (cassettes) so evaluation is replayable offline

Privacy: only public data is used; logins are replaced by salted hashes in every file
except users_private.json, which never leaves the data directory (gitignored). Only
aggregate metrics are reported.

GitHub's unauthenticated budget (60 core calls/hour, 10 searches/minute) is small, so
every step persists progress and stops cleanly when the budget or a time limit is hit;
re-run the same command to continue.

    python -m detour.eval.collect users --data DIR [--max-users 40] [--budget-s 150]
    python -m detour.eval.collect pools --data DIR [--budget-s 150]
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from detour.representation.interests import CURATED_LABELS, GENERIC_TOPICS, resolve_interest
from detour.retrieval.cassette import ReplayOrRecordTransport
from detour.retrieval.github import GitHubBadRequest, GitHubClient, GitHubRateLimited
from detour.retrieval.models import RepoRecord
from detour.retrieval.planner import RetrievalConfig
from detour.retrieval.service import CandidateRetriever, MemorySearchCache, RetrievalFailed

SEEDS = [
    "jepsen-io/maelstrom",  # distributed systems
    "pingcap/talent-plan",  # distributed systems / databases
    "cmu-db/bustub",  # databases
    "kth-competitive-programming/kactl",  # competitive programming
    "karpathy/micrograd",  # machine learning
    "tokio-rs/mini-redis",  # backend / systems
    "open-telemetry/opentelemetry-collector",  # observability / cloud native
    "huggingface/smolagents",  # LLMs
]
CUTOFF_DAYS = 120
MIN_KNOWN = 20
MIN_HELDOUT = 3
MAX_HELDOUT = 100
INTEREST_MIN_SUPPORT = 3
MAX_INTERESTS = 3
STAR_FLOOR = 20


def _salt() -> str:
    return os.environ.get("DETOUR_EVAL_SALT", "detour-eval-v1")


def anon(login: str) -> str:
    return hashlib.sha256(f"{_salt()}:{login.lower()}".encode()).hexdigest()[:12]


def _load(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def _client(data: Path) -> GitHubClient:
    token = os.environ.get("GITHUB_TOKEN") or None
    return GitHubClient(
        token=token, transport=ReplayOrRecordTransport(data / "cassettes"), timeout_s=20.0
    )


# --- interest derivation ---------------------------------------------------------------------

_TOPIC_TO_LABELS: dict[str, set[str]] = {}
for _label in CURATED_LABELS:
    for _t in resolve_interest(_label).topics:
        if _t not in GENERIC_TOPICS:
            _TOPIC_TO_LABELS.setdefault(_t, set()).add(_label)


def derive_interests(known: list[RepoRecord]) -> list[str]:
    """Curated interests supported by >= INTEREST_MIN_SUPPORT known repos' topics."""
    support: Counter[str] = Counter()
    for repo in known:
        labels: set[str] = set()
        for t in repo.topics:
            labels |= _TOPIC_TO_LABELS.get(t, set())
        support.update(labels)
    ranked = sorted(
        ((lab, c) for lab, c in support.items() if c >= INTEREST_MIN_SUPPORT),
        key=lambda x: (-x[1], x[0]),
    )
    return [lab for lab, _ in ranked[:MAX_INTERESTS]]


def eligible_heldout(repo: RepoRecord) -> bool:
    has_text = bool(repo.description and repo.description.strip()) or bool(repo.topics)
    return not repo.archived and not repo.is_fork and has_text and repo.stars >= STAR_FLOOR


# --- step 1: users -----------------------------------------------------------------------------


def collect_users(data: Path, max_users: int, budget_s: float) -> bool:
    t0 = time.monotonic()
    data.mkdir(parents=True, exist_ok=True)
    state = _load(data / "state.json", {"collected_at": datetime.now(UTC).isoformat()})
    collected_at = datetime.fromisoformat(state["collected_at"])
    cutoff = collected_at - timedelta(days=CUTOFF_DAYS)
    state["cutoff"] = cutoff.isoformat()
    gh = _client(data)
    private = _load(data / "users_private.json", {"stargazers": {}, "examined": []})

    try:
        # 1a. owners of the newest forks of each seed repository
        for seed in SEEDS:
            if seed in private["stargazers"]:
                continue
            rows = gh.get(f"/repos/{seed}/forks", {"sort": "newest", "per_page": 100})
            private["stargazers"][seed] = [
                r["owner"]["login"] for r in rows if (r.get("owner") or {}).get("type") == "User"
            ]
            _save(data / "users_private.json", private)

        logins = sorted({lg for v in private["stargazers"].values() for lg in v})
        random.Random(42).shuffle(logins)

        # 1b. stars of sampled users
        examples = _load(data / "examples.json", [])
        for login in logins:
            if len(private["examined"]) >= max_users:
                break
            if login in private["examined"]:
                continue
            if time.monotonic() - t0 > budget_s:
                print("time budget reached; re-run to continue")
                return False
            try:
                starred = gh.starred(login, max_pages=3, stop=enough_pages(cutoff))
            except GitHubBadRequest:
                private["examined"].append(login)
                continue
            private["examined"].append(login)
            ex = build_example(login, starred, cutoff)
            if ex is not None:
                examples.append(ex)
                _save(data / "examples.json", examples)
            _save(data / "users_private.json", private)
    except GitHubRateLimited as exc:
        print(f"rate limited ({exc}); re-run after the reset")
        _save(data / "state.json", state)
        return False
    _save(data / "state.json", state)
    print(f"examined={len(private['examined'])} eligible={len(_load(data / 'examples.json', []))}")
    return True


KNOWN_TARGET = 50  # enough pre-cutoff stars to derive interests reliably


def enough_pages(cutoff: datetime) -> Callable[[list[Any]], bool]:
    """Stop paging once we have enough pre-cutoff stars, or once it is clear the user is a
    bulk starrer (more post-cutoff stars than the held-out cap): either way, further pages
    cannot change the outcome."""

    def stop(stars: list[Any]) -> bool:
        known = sum(1 for s in stars if s.starred_at <= cutoff)
        after = len(stars) - known
        return known >= KNOWN_TARGET or after > MAX_HELDOUT

    return stop


def build_example(login: str, starred: list[Any], cutoff: datetime) -> dict[str, Any] | None:
    known = [s for s in starred if s.starred_at <= cutoff]
    after = [s for s in starred if s.starred_at > cutoff]
    known_ids = {s.repo.github_id for s in known}
    heldout = [s for s in after if eligible_heldout(s.repo) and s.repo.github_id not in known_ids]
    if len(known) < MIN_KNOWN or not (MIN_HELDOUT <= len(heldout) <= MAX_HELDOUT):
        return None
    interests = derive_interests([s.repo for s in known[:100]])
    if not interests:
        return None
    return {
        "user": anon(login),
        "interests": interests,
        "known": [s.repo.to_json() | {"starred_at": s.starred_at.isoformat()} for s in known],
        "heldout": [s.repo.to_json() | {"starred_at": s.starred_at.isoformat()} for s in heldout],
    }


# --- step 2: pools ------------------------------------------------------------------------------


def collect_pools(data: Path, budget_s: float) -> bool:
    t0 = time.monotonic()
    examples = _load(data / "examples.json", [])
    pools_dir = data / "pools"
    pools_dir.mkdir(exist_ok=True)
    gh = _client(data)
    cfg = RetrievalConfig()
    for ex in examples:
        out = pools_dir / f"{ex['user']}.json"
        if out.exists():
            continue
        if time.monotonic() - t0 > budget_s:
            print("time budget reached; re-run to continue")
            return False
        specs = [resolve_interest(lab) for lab in ex["interests"]]
        known_topics = {t for s in specs for t in s.topics}
        for r in ex["known"]:
            known_topics.update(t for t in r["topics"] if t not in GENERIC_TOPICS)
        while True:
            # A fresh in-memory cache each attempt: already-successful requests are served
            # from the cassette directory without spending API budget.
            retriever = CandidateRetriever(
                gh, MemorySearchCache(), cache_ttl_s=10**9, max_workers=1
            )
            try:
                outcome = retriever.retrieve(specs, known_topics, cfg)
                if not any(q.source == "failed" for q in outcome.queries):
                    break
            except RetrievalFailed:
                pass
            reset = gh.search_rate.reset_at
            wait = (reset - datetime.now(UTC)).total_seconds() + 2 if reset else None
            remaining = budget_s - (time.monotonic() - t0)
            if wait is None or wait > remaining:
                print(f"user {ex['user']}: search budget exhausted; re-run later")
                return False
            print(f"user {ex['user']}: waiting {wait:.0f}s for the search rate limit")
            time.sleep(max(0.0, wait))
        pool = {
            "user": ex["user"],
            "queries": [q.to_dict() for q in outcome.queries],
            "candidates": [
                {
                    "repo": c.repo.to_json(),
                    "families": sorted(f.value for f in c.families),
                    "query_labels": c.query_labels,
                    "adjacent_topics": sorted(c.adjacent_topics),
                    "bridge_pairs": [list(p) for p in c.bridge_pairs],
                }
                for c in outcome.candidates.values()
            ],
        }
        _save(out, pool)
        print(f"user {ex['user']}: pool={len(pool['candidates'])} queries={len(pool['queries'])}")
    print(f"pools complete: {len(list(pools_dir.glob('*.json')))}/{len(examples)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["users", "pools"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-users", type=int, default=40)
    ap.add_argument("--budget-s", type=float, default=150)
    args = ap.parse_args()
    data = Path(args.data)
    done = (
        collect_users(data, args.max_users, args.budget_s)
        if args.step == "users"
        else collect_pools(data, args.budget_s)
    )
    sys.exit(0 if done else 3)


if __name__ == "__main__":
    main()
