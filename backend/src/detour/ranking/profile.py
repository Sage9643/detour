"""The user representation the ranker consumes (pure data; built by detour.profile)."""

import uuid
from dataclasses import dataclass, field

import numpy as np

from detour.representation.embedder import Vectors


@dataclass(frozen=True, slots=True)
class InterestVector:
    label: str
    vector: Vectors  # (dim,)
    weight: float  # effective weight after feedback-derived decay (1.0 = neutral)
    topics: tuple[str, ...] = ()
    # Mean cosine between this interest and unrelated topic descriptions: the similarity an
    # off-topic repository typically gets. Differs a lot per interest with static embeddings.
    background: float = 0.0
    id: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class Exemplar:
    repo_id: int
    full_name: str
    vector: Vectors


@dataclass(slots=True)
class UserProfile:
    """A SET of interest vectors (never averaged) plus feedback-derived signals.

    positives      INTERESTED repos: extra relevance anchors (personalization)
    negatives      NOT_INTERESTED repos: local penalty for near-duplicates
    known_*        familiarity: ALREADY_KNOW + INTERESTED repos. Drives novelty only;
                   knowing a repo is NOT evidence of disliking similar repos.
    recently_shown repos shown in recent runs (excluded so each request shows new items)
    """

    interests: list[InterestVector]
    positives: list[Exemplar] = field(default_factory=list)
    negatives: list[Exemplar] = field(default_factory=list)
    known_ids: set[int] = field(default_factory=set)
    known_vectors: Vectors = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    known_topics: set[str] = field(default_factory=set)
    recently_shown: set[int] = field(default_factory=set)

    @property
    def cold_start(self) -> bool:
        return bool(self.known_vectors.shape[0] == 0)
