"""Query planner: interests -> GitHub searches. Pure functions, no I/O.

Three query families (see docs/recommendation-engine.md §5):
- core      one keyword query per interest: baseline coverage
- bridge    two interests combined: candidates relevant to both ("relevant surprise")
- adjacent  topics that co-occur with the user's interests in core results but that the
            user's profile does not cover: exploratory, topically novel candidates

Retrieval only decides what CAN be recommended. Whether something is relevant is decided
later by the ranker; adjacent results in particular still have to pass the relevance gate.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import combinations
from typing import Any

from detour.enums import QueryFamily
from detour.representation.interests import GENERIC_TOPICS, InterestSpec
from detour.retrieval.models import PlannedQuery, RepoRecord


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    star_floor: int = 20  # quality floor; also keeps abandoned toy repos out
    recency_days: int = 365  # "currently relevant": pushed within this window
    core_per_page: int = 50
    bridge_per_page: int = 30
    adjacent_per_page: int = 30
    max_interests: int = 6
    max_bridges: int = 3
    max_adjacent: int = 2
    adjacent_min_support: int = 3  # topic must appear on >= N distinct core results
    enable_bridge: bool = True
    enable_adjacent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qualifiers(cfg: RetrievalConfig, today: date) -> str:
    since = (today - timedelta(days=cfg.recency_days)).isoformat()
    return f"pushed:>{since} stars:>={cfg.star_floor} archived:false fork:false"


def plan_primary(
    interests: Sequence[InterestSpec], cfg: RetrievalConfig, today: date
) -> list[PlannedQuery]:
    """Core queries for every interest, then bridge queries for interest pairs."""
    quals = qualifiers(cfg, today)
    chosen = list(interests)[: cfg.max_interests]
    plans = [
        PlannedQuery(
            family=QueryFamily.CORE,
            q=f"{spec.core_terms} in:name,description,topics {quals}",
            per_page=cfg.core_per_page,
            label=spec.label,
            interest_labels=(spec.label,),
        )
        for spec in chosen
    ]
    if cfg.enable_bridge:
        bridges = 0
        for a, b in combinations(chosen[:4], 2):
            if bridges >= cfg.max_bridges:
                break
            if a.bridge_term.lower() == b.bridge_term.lower():
                continue  # e.g. two interests sharing the term "algorithms": no bridge
            plans.append(
                PlannedQuery(
                    family=QueryFamily.BRIDGE,
                    q=f"{a.bridge_term} {b.bridge_term} in:name,description,topics {quals}",
                    per_page=cfg.bridge_per_page,
                    label=f"{a.label} + {b.label}",
                    interest_labels=(a.label, b.label),
                )
            )
            bridges += 1
    return plans


def adjacent_topics(
    core_results: Iterable[RepoRecord],
    known_topics: set[str],
    cfg: RetrievalConfig,
) -> list[tuple[str, int]]:
    """Topics frequent among core results but outside the user's known topics.

    Returns (topic, support) pairs, most supported first; ties broken alphabetically so
    the plan is deterministic.
    """
    seen: set[int] = set()
    counts: Counter[str] = Counter()
    for repo in core_results:
        if repo.github_id in seen:
            continue
        seen.add(repo.github_id)
        for topic in set(repo.topics):
            if topic in known_topics or topic in GENERIC_TOPICS:
                continue
            counts[topic] += 1
    ranked = sorted(
        ((t, c) for t, c in counts.items() if c >= cfg.adjacent_min_support),
        key=lambda tc: (-tc[1], tc[0]),
    )
    return ranked[: cfg.max_adjacent]


def plan_adjacent(
    core_results: Iterable[RepoRecord],
    known_topics: set[str],
    cfg: RetrievalConfig,
    today: date,
) -> list[PlannedQuery]:
    if not cfg.enable_adjacent:
        return []
    quals = qualifiers(cfg, today)
    return [
        PlannedQuery(
            family=QueryFamily.ADJACENT,
            q=f"topic:{topic} {quals}",
            per_page=cfg.adjacent_per_page,
            label=f"adjacent topic: {topic}",
            topic=topic,
        )
        for topic, _support in adjacent_topics(core_results, known_topics, cfg)
    ]
