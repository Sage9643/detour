"""Embedding cache: reuse a repository vector when (model, embedded text) is unchanged."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from detour.db.models import RepoEmbedding
from detour.representation.embedder import Embedder, Vectors
from detour.representation.text import repo_text, text_hash
from detour.retrieval.models import RepoRecord


@dataclass(frozen=True, slots=True)
class EmbedStats:
    hits: int
    misses: int
    embed_ms: float


def embed_repos_cached(
    session: Session, embedder: Embedder, repos: Sequence[RepoRecord]
) -> tuple[dict[int, Vectors], EmbedStats]:
    """Vectors for `repos` (keyed by github_id), computing only cache misses.

    Repositories must already exist in `repositories` (FK).
    """
    texts = {r.github_id: repo_text(r) for r in repos}
    hashes = {rid: text_hash(t) for rid, t in texts.items()}
    out: dict[int, Vectors] = {}

    if texts:
        rows = session.execute(
            select(RepoEmbedding.repo_id, RepoEmbedding.text_hash, RepoEmbedding.vector).where(
                tuple_(RepoEmbedding.repo_id, RepoEmbedding.model_name).in_(
                    [(rid, embedder.model_name) for rid in texts]
                )
            )
        )
        for rid, h, vec in rows:
            if hashes.get(rid) == h:
                out[rid] = np.asarray(vec, dtype=np.float32)

    missing = [rid for rid in texts if rid not in out]
    t0 = time.perf_counter()
    if missing:
        vectors = embedder.embed([texts[rid] for rid in missing])
        values = []
        for rid, vec in zip(missing, vectors, strict=True):
            out[rid] = vec
            values.append(
                {
                    "repo_id": rid,
                    "model_name": embedder.model_name,
                    "text_hash": hashes[rid],
                    "dim": embedder.dim,
                    "vector": vec.tolist(),
                }
            )
        stmt = insert(RepoEmbedding).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[RepoEmbedding.repo_id, RepoEmbedding.model_name],
            set_={
                "text_hash": stmt.excluded.text_hash,
                "dim": stmt.excluded.dim,
                "vector": stmt.excluded.vector,
            },
        )
        session.execute(stmt)
    embed_ms = (time.perf_counter() - t0) * 1000
    return out, EmbedStats(hits=len(texts) - len(missing), misses=len(missing), embed_ms=embed_ms)


def embed_repos_uncached(embedder: Embedder, repos: Sequence[RepoRecord]) -> dict[int, Vectors]:
    """Offline tooling path (no database)."""
    if not repos:
        return {}
    vectors = embedder.embed([repo_text(r) for r in repos])
    return {r.github_id: v for r, v in zip(repos, vectors, strict=True)}
