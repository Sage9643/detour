"""Ranking parameters.

Every value here is a STARTING HYPOTHESIS, not a calibrated or optimized number (see the
parameter register in docs/recommendation-engine.md §11). The full config is stored with
every recommendation run (recommendation_runs.config + config_hash) so results can always
be traced back to the exact parameters that produced them.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RankerConfig:
    k: int = 10
    # Relevance gate on raw cosine similarity. For the WordLlama model, on real candidate
    # pools, related text typically scores 0.2-0.6 and unrelated text 0.0-0.15; 0.18 is a
    # hypothesis from inspecting score distributions, NOT a calibrated threshold (no
    # labeled relevance data exists yet).
    tau: float = 0.18
    alpha_exemplar: float = 0.5  # weight of similarity to INTERESTED repos in relevance
    beta_novelty: float = 0.3  # novelty weight in item utility
    gamma_negative: float = 1.0  # penalty weight for near-duplicates of rejected repos
    theta_negative: float = 0.6  # similarity above which a rejected repo penalizes
    mmr_lambda: float = 0.7  # 1.0 = pure utility, 0.0 = pure dissimilarity
    owner_cap: int = 2  # max repos per owner in one list (D2/D3)
    star_floor: int = 20
    # Interest weight decay from NOT_INTERESTED: after `decay_after` rejections attributed
    # to an interest, each further rejection multiplies its weight by `decay_factor`.
    decay_after: int = 2
    decay_factor: float = 0.85
    min_weight: float = 0.5
    # Repos shown to the user within this window are excluded, so "more" shows new items.
    recently_shown_days: int = 14

    def with_overrides(self, **kw: Any) -> "RankerConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
