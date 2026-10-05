"""Explanations derived only from what the ranker actually computed.

`build_reasons` turns a ScoredCandidate into a structured dict (logged as
run_candidates.reasons). `render` turns that dict into sentences with fixed templates.
Every sentence corresponds to one reason key, and tests assert each claim against the
underlying scores. No LLM is involved.
"""

from typing import Any

from detour.enums import QueryFamily
from detour.ranking.engine import ScoredCandidate, meaningful_topics
from detour.ranking.profile import UserProfile

STRONG_REL_NORM = 0.8  # top 20% of the candidate pool by relevance
UNFAMILIAR_MIN = 0.6  # unfamiliarity at/above this is worth mentioning


def popularity_band(stars: int) -> str:
    if stars < 200:
        return "niche"
    if stars < 2_000:
        return "emerging"
    if stars < 20_000:
        return "established"
    return "widely used"


def build_reasons(s: ScoredCandidate, profile: UserProfile, tau: float) -> dict[str, Any]:
    repo = s.item.candidate.repo
    cand = s.item.candidate
    reasons: dict[str, Any] = {
        "matched_interest": s.matched_interest,
        "relevance_raw": _r(s.relevance_raw),
        "relevance_pct": _r(s.relevance_norm),
        "strong_match": bool(s.relevance_norm is not None and s.relevance_norm >= STRONG_REL_NORM),
        "popularity_band": popularity_band(repo.stars),
        "stars": repo.stars,
        "language": repo.language,
    }
    if s.matched_via_exemplar:
        reasons["similar_to_liked"] = s.matched_via_exemplar

    # A second interest the repo also clears the relevance gate for.
    others = sorted(
        ((lab, sim) for lab, sim in s.interest_sims.items() if lab != s.matched_interest),
        key=lambda x: -x[1],
    )
    if others and others[0][1] >= tau:
        reasons["secondary_interest"] = others[0][0]

    new_topics = [t for t in meaningful_topics(repo.topics) if t not in profile.known_topics]
    if new_topics:
        reasons["new_topics"] = new_topics[:3]

    if s.unfamiliarity is not None and s.unfamiliarity >= UNFAMILIAR_MIN:
        reasons["unfamiliar"] = _r(s.unfamiliarity)
    if profile.cold_start:
        reasons["cold_start"] = True

    if QueryFamily.BRIDGE in cand.families and cand.bridge_pairs:
        reasons["bridge"] = list(cand.bridge_pairs[0])
    if QueryFamily.ADJACENT in cand.families and cand.adjacent_topics:
        reasons["adjacent_topic"] = sorted(cand.adjacent_topics)[0]
    if s.diversity_promoted:
        reasons["diversity_promoted"] = True
    if s.negative_penalty:
        reasons["near_rejected"] = s.nearest_negative
    return reasons


def render(reasons: dict[str, Any]) -> list[str]:
    out: list[str] = []
    interest = reasons.get("matched_interest")
    if reasons.get("similar_to_liked"):
        out.append(f"Similar to {reasons['similar_to_liked']}, which you marked as interesting.")
    elif interest:
        if reasons.get("strong_match"):
            out.append(f"Strong match with your {interest} interest.")
        else:
            out.append(f"Relevant to your {interest} interest.")
    if reasons.get("secondary_interest"):
        out.append(f"Also relevant to {reasons['secondary_interest']}.")
    if reasons.get("bridge"):
        a, b = reasons["bridge"][:2]
        out.append(f"Found by a query bridging {a} and {b}.")
    if reasons.get("adjacent_topic"):
        out.append(
            f"Exploration pick: found via '{reasons['adjacent_topic']}', a topic that often "
            "appears next to your interests."
        )
    if reasons.get("new_topics"):
        topics = ", ".join(reasons["new_topics"])
        out.append(f"Introduces topics you haven't interacted with: {topics}.")
    if reasons.get("unfamiliar") is not None:
        out.append("Unlike the repositories you've marked as known.")
    if reasons.get("popularity_band") in ("niche", "emerging"):
        out.append(f"Less-known project ({reasons['stars']:,} stars).")
    if reasons.get("diversity_promoted"):
        out.append(
            "Picked over a higher-scoring but more similar repository to keep the list varied."
        )
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)
