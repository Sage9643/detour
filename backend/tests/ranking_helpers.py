"""Hand-built pools for deterministic ranking tests (no embedding model involved)."""

from collections.abc import Sequence

import numpy as np

from detour.enums import QueryFamily
from detour.ranking.engine import PoolItem
from detour.ranking.profile import Exemplar, InterestVector, UserProfile
from detour.representation.embedder import Vectors, l2_normalize
from detour.retrieval.models import Candidate, RepoRecord


def vec(*xs: float) -> Vectors:
    return l2_normalize(np.array([xs], dtype=np.float32))[0]


def repo(
    gid: int,
    *,
    owner: str | None = None,
    stars: int = 100,
    topics: Sequence[str] = (),
    description: str | None = "a repo",
    archived: bool = False,
    fork: bool = False,
) -> RepoRecord:
    owner = owner or f"owner{gid}"
    return RepoRecord(
        github_id=gid,
        full_name=f"{owner}/r{gid}",
        owner_login=owner,
        name=f"r{gid}",
        description=description,
        topics=tuple(topics),
        language=None,
        stars=stars,
        forks=0,
        archived=archived,
        is_fork=fork,
        license_spdx=None,
        html_url=f"https://github.com/{owner}/r{gid}",
        gh_created_at=None,
        pushed_at=None,
    )


def item(
    r: RepoRecord, v: Vectors, families: Sequence[QueryFamily] = (QueryFamily.CORE,)
) -> PoolItem:
    return PoolItem(candidate=Candidate(repo=r, families=set(families)), vector=v)


def profile(
    interests: Sequence[tuple[str, Vectors]] | Sequence[tuple[str, Vectors, float]],
    *,
    known: Sequence[tuple[int, Vectors]] = (),
    known_topics: Sequence[str] = (),
    positives: Sequence[tuple[int, Vectors]] = (),
    negatives: Sequence[tuple[int, Vectors]] = (),
    recently_shown: Sequence[int] = (),
) -> UserProfile:
    ivs = []
    for spec in interests:
        label, v = spec[0], spec[1]
        w = spec[2] if len(spec) == 3 else 1.0  # type: ignore[misc]
        ivs.append(InterestVector(label=label, vector=v, weight=float(w)))
    kv = np.stack([v for _, v in known]) if known else np.zeros((0, 0), dtype=np.float32)
    return UserProfile(
        interests=ivs,
        positives=[Exemplar(i, f"liked/r{i}", v) for i, v in positives],
        negatives=[Exemplar(i, f"disliked/r{i}", v) for i, v in negatives],
        known_ids={i for i, _ in known},
        known_vectors=kv,
        known_topics=set(known_topics),
        recently_shown=set(recently_shown),
    )
