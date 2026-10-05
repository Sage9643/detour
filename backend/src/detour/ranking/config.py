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
    # Relevance gate: a candidate must beat the interest's BACKGROUND similarity (its mean
    # cosine with unrelated topic descriptions) by this margin. Weakly calibrated on one live
    # pool using retrieval provenance as labels (see docs/evaluation.md §7.1): at 0.10,
    # 94% of on-target and 6% of off-target candidates pass. Not a human-labeled calibration.
    relevance_margin: float = 0.10
    # A candidate this similar (raw cosine) to a repo the user marked INTERESTED also
    # passes the gate (D3 only).
    exemplar_gate: float = 0.5
    alpha_exemplar: float = 0.5  # weight of similarity to INTERESTED repos in relevance
    beta_novelty: float = 0.3  # novelty weight in item utility
    gamma_negative: float = 1.0  # penalty weight for near-duplicates of rejected repos
    theta_negative: float = 0.6  # similarity above which a rejected repo penalizes
    mmr_lambda: float = 0.7  # 1.0 = pure utility, 0.0 = pure dissimilarity
    owner_cap: int = 2  # max repos per owner in one list (D2/D3)
    coverage_floor: bool = True  # D2/D3: reserve tail slots so every interest appears once
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
