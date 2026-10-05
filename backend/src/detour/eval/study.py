"""Analyse the blind A/B study (frontend `?study` mode) from the interaction log.

    DATABASE_URL=... python -m detour.eval.study

A study session ranks one candidate pool twice: a D3 run, plus a B1 run linked through
`config.paired_with_run`. Participants rate items in both lists. Each feedback event
references the run whose list it was given in, so it can be attributed to a system.

Per variant: share of shown items marked INTERESTED (would explore), ALREADY_KNOW, and
NOT_INTERESTED. Per pair: which list received more INTERESTED marks, with a Wilson 95%
interval on D3's win share (ties excluded).
"""

import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from detour.config import Settings
from detour.db.models import Feedback, RecommendationRun, RunCandidate
from detour.enums import FeedbackType


@dataclass
class Pair:
    d3: uuid.UUID
    other: uuid.UUID
    other_variant: str


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def analyse(engine: Engine) -> dict[str, Any]:
    with Session(engine) as s:
        runs = s.scalars(
            select(RecommendationRun).where(RecommendationRun.status != "failed")
        ).all()
        pairs = [
            Pair(uuid.UUID(r.config["paired_with_run"]), r.id, r.variant)
            for r in runs
            if r.config.get("paired_with_run")
        ]
        run_ids = {p.d3 for p in pairs} | {p.other for p in pairs}
        if not run_ids:
            return {"pairs": 0}
        shown: dict[uuid.UUID, set[int]] = {}
        for run_id, repo_id in s.execute(
            select(RunCandidate.run_id, RunCandidate.repo_id).where(
                RunCandidate.run_id.in_(run_ids), RunCandidate.shown.is_(True)
            )
        ):
            shown.setdefault(run_id, set()).add(repo_id)
        # latest feedback per (run, repo)
        latest: dict[tuple[uuid.UUID, int], str] = {}
        for fb_run, fb_repo, ftype in s.execute(
            select(Feedback.run_id, Feedback.repo_id, Feedback.type)
            .where(Feedback.run_id.in_(run_ids))
            .order_by(Feedback.created_at, Feedback.id)
        ):
            if fb_run is not None:
                latest[(fb_run, fb_repo)] = ftype

    def counts(run_id: uuid.UUID) -> dict[str, int]:
        c = {t.value: 0 for t in FeedbackType} | {"shown": len(shown.get(run_id, ()))}
        for repo_id in shown.get(run_id, ()):
            t = latest.get((run_id, repo_id))
            if t:
                c[t] += 1
        return c

    per_variant: dict[str, dict[str, int]] = {}
    wins = losses = ties = 0
    for p in pairs:
        a, b = counts(p.d3), counts(p.other)
        for variant, c in (("D3", a), (p.other_variant, b)):
            agg = per_variant.setdefault(variant, {k: 0 for k in c})
            for k, v in c.items():
                agg[k] += v
        if a["INTERESTED"] > b["INTERESTED"]:
            wins += 1
        elif a["INTERESTED"] < b["INTERESTED"]:
            losses += 1
        else:
            ties += 1
    rates = {
        v: {
            **c,
            "interested_rate": c["INTERESTED"] / c["shown"] if c["shown"] else float("nan"),
            "already_know_rate": c["ALREADY_KNOW"] / c["shown"] if c["shown"] else float("nan"),
            "not_interested_rate": c["NOT_INTERESTED"] / c["shown"] if c["shown"] else float("nan"),
        }
        for v, c in per_variant.items()
    }
    lo, hi = wilson(wins, wins + losses)
    return {
        "pairs": len(pairs),
        "per_variant": rates,
        "d3_wins": wins,
        "d3_losses": losses,
        "ties": ties,
        "d3_win_share_ci95": [lo, hi],
    }


def main() -> None:
    settings = Settings() if os.environ.get("DATABASE_URL") else None
    if settings is None:
        sys.exit("set DATABASE_URL")
    print(json.dumps(analyse(create_engine(settings.database_url)), indent=1))


if __name__ == "__main__":
    main()
