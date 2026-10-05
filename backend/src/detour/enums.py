"""Domain enumerations shared by the engine, API, and persistence layers.

Stored in PostgreSQL as TEXT with CHECK constraints (see db/models.py) rather than native
ENUM types: adding a value is a one-line constraint change instead of an ALTER TYPE.
"""

from enum import StrEnum


class FeedbackType(StrEnum):
    INTERESTED = "INTERESTED"  # preference evidence (+)
    NOT_INTERESTED = "NOT_INTERESTED"  # preference evidence (-)
    ALREADY_KNOW = "ALREADY_KNOW"  # familiarity evidence; NOT a negative preference


class RankerVariant(StrEnum):
    """Systems compared in evaluation. Every logged run records which one produced it."""

    B0_POPULARITY = "B0"
    B1_RELEVANCE = "B1"
    D1_RELEVANCE_NOVELTY = "D1"
    D2_RELEVANCE_MMR = "D2"
    D3_FULL = "D3"


class RunStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"  # served, but a dependency fell back (e.g. stale cache)
    FAILED = "failed"


class QueryFamily(StrEnum):
    """Which retrieval strategy produced a candidate (a candidate may have several)."""

    CORE = "core"  # one query per explicit interest
    BRIDGE = "bridge"  # combination of two interests
    ADJACENT = "adjacent"  # topic co-occurring with interests but not yet known
