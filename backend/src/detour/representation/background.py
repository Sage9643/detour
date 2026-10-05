"""Per-interest background similarity: how similar an UNRELATED text typically is.

Static embeddings are anisotropic: every pair of texts has a positive baseline cosine, and
that baseline differs per interest (measured on a live pool: off-target candidates scored
0.13 against "Machine Learning" but 0.21 against "Competitive Programming"). A fixed raw
cosine threshold therefore lets off-topic repositories through for some interests and
rejects on-topic ones for others.

Background(interest) = mean cosine between the interest's expansion and the expansions of
all OTHER curated interests (a fixed, pool-independent reference set spanning computing
topics). On the measured live pool this closely matched the observed off-target mean
(0.142 vs 0.134, 0.161 vs 0.120, 0.212 vs 0.205).
"""

import numpy as np

from detour.representation.embedder import Embedder, Vectors
from detour.representation.interests import CURATED_LABELS, InterestSpec, resolve_interest

_REFERENCE: dict[str, tuple[list[str], Vectors]] = {}


def _reference(embedder: Embedder) -> tuple[list[str], Vectors]:
    ref = _REFERENCE.get(embedder.model_name)
    if ref is None:
        labels = list(CURATED_LABELS)
        vecs = embedder.embed([resolve_interest(lab).expansion for lab in labels])
        ref = _REFERENCE[embedder.model_name] = (labels, vecs)
    return ref


def background_similarity(embedder: Embedder, spec: InterestSpec, vector: Vectors) -> float:
    labels, vecs = _reference(embedder)
    mask = np.array([lab != spec.label for lab in labels])
    return float((vecs[mask] @ vector).mean())
