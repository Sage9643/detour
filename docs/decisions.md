# Engineering Decisions

Each entry: decision · why · alternatives · cost · status.

## D-001 Modular monolith, single FastAPI service
**Why:** One team, low traffic, and one request path whose stages are synchronous and
cheap except for external calls.
**Alternatives:** microservices or a queue-based pipeline. These add network hops,
deployment units, and failure modes with no measured benefit.
**Status:** adopted (Phase 1).

## D-002 PostgreSQL for everything stateful, no vector DB
**Why:** We need relational user, feedback and log data, and each request scores about
300 vectors, which is a NumPy dot product rather than an ANN search problem.
**Revisit if:** a precomputed global index over many repos becomes necessary
(pgvector on Neon would be the first step).
**Status:** adopted.

## D-003 Real PostgreSQL in tests (no SQLite)
**Why:** The schema relies on partial unique indexes, ARRAY containment checks, JSONB and
`gen_random_uuid()`. A SQLite test would validate a different database.
**Cost:** tests need a running Postgres; CI uses a service container.
**Status:** adopted.

## D-004 CI fails rather than skips when the DB is missing
**Why:** A green build that silently skipped 45 DB tests is worse than a red build.
`DETOUR_REQUIRE_DB_TESTS=1` is set in CI. Local runs without a DB skip, for convenience.
**Status:** adopted; verified by a mutation check (see development-log).

## D-005 Initial migration contains only tables whose shape is settled
**Why:** `search_cache` (Phase 2) and `repo_embeddings` (Phase 3) depend on decisions
those phases will measure (TTL keying, embedding dimension and storage).
**Status:** adopted.

## D-006 Log every scored candidate, with point-in-time features
See `data-model.md`. Avoids selection bias in future learned ranking, and enables
offline replay of variants on real pools.
**Status:** adopted.

## D-007 API not containerized in Phase 1
**Why:** Locally, `uvicorn --reload` iterates faster; compose runs only Postgres. A
Dockerfile is added when deployment needs one (Phase 11).
**Status:** adopted. *Changed from the Phase 0 plan, which listed Docker generally.*

## D-008 Dev test client uses `httpx2`
**Why:** The installed Starlette version deprecates plain `httpx` for `TestClient`;
tests run with `-W error::DeprecationWarning`.
**Status:** adopted.

## D-009 Embedder default (superseded by D-010)
The Phase 0 lean is a local ONNX model, for reproducibility and zero cost. The
alternative is an external API, which is lighter to deploy. The decision will be made by
the Phase 3 benchmark (latency, memory on Render, quality on the labeled calibration
set).
**Status:** superseded. Hugging Face (the source for both candidate models) was
unreachable, so D-010 chose a bundled pretrained model instead.

## D-010 Embedding model: WordLlama l2_supercat (static, 256-d), bundled weights
**Context:** Hugging Face was unreachable from both development sandboxes, so
transformer encoders (bge-small, MiniLM) could not be downloaded or tested. Render's free
tier has 512 MB of RAM.
**Decision:** WordLlama, a pretrained static embedding model whose weights (16 MB) and
tokenizer ship inside its PyPI wheel. The files are loaded directly, bypassing the
library's own downloader. It needs no PyTorch or ONNX runtime, the measured API RSS
after startup is ~170 MB, and embedding 176 repo texts took 174 ms on the sandbox.
**Evidence it works:** on a live pool, cosine to the interest separated a query's own
results from other interests' results with AUC 0.96–0.997 (evaluation §7.1).
**Trade-off:** static embeddings ignore word order, are weaker than transformer encoders,
and are anisotropic. That anisotropy is the reason for D-012.
**Reversible:** everything depends on the `Embedder` protocol, and cached vectors are
keyed by `model_name`.
**Status:** adopted. Benchmarking a transformer encoder where Hugging Face is reachable
is future work.

## D-011 Search cache keyed by normalized query; stale-on-failure
**Why:** GitHub's search budget (10/min unauthenticated, 30/min with a token) is the
binding constraint, not latency: one cold request spends 6–10 searches. Caching per
query, with metadata upserted into `repositories`, makes repeat requests cost zero API
calls. The live run measured retrieval at 1706 ms cold and 29 ms warm.
**Stale-on-failure:** expired entries are kept and served when GitHub fails; the run is
marked `degraded` and the UI says so. Results are never invented.
**Status:** adopted. Redis is not needed: the cache is one indexed Postgres lookup.

## D-012 Per-interest calibrated relevance (background-margin gate + percentile ranking)
**Context (measured):** with raw cosine, on-target scales differed by interest (0.39 vs
0.58), and the raw-max ranking showed zero repos for one interest. A raw threshold of
0.18 let 38% of off-target candidates through.
**Decision:** gate on cosine minus the interest's background similarity (≥ 0.10). Rank by
per-interest percentile within the pool.
**Measured effect (same pool):** off-target pass rate 38% → 6% with on-target 94%; top-24
interest mix 14/0/10 → 8/8/8.
**Caveat:** calibrated on one pool with weak labels (retrieval provenance), not human
judgements.
**Status:** adopted.

## D-013 Interest-coverage floor in MMR (instead of xQuAD)
**Why:** MMR removes redundancy but does not guarantee every interest appears. The
floor reserves tail slots only when coverage would otherwise be lost. It is one boolean,
deterministic, and unit-tested.
**Status:** adopted.

## D-014 Log every candidate, including filtered ones
This extends D-006: candidates removed by filters or the gate are logged too, with
`filter_reason`. Pool-level analyses (what the gate removed, what a different variant
would have shown) need the complete pool.

## D-015 Record/replay transport for GitHub
**Why:** reproducible evaluation (frozen responses), and exercising the full pipeline on
*real* data where GitHub is unreachable. In development, the cloud sandbox could not call
GitHub search; the linked machine could, so it recorded and the cloud replayed.
Authorization headers are never recorded. Replay-first recording makes the offline
collector resumable without re-spending API budget.
**Status:** adopted.

## D-016 Deploy the API with Render's native Python runtime, not a Dockerfile
**Why:** a Docker image could not be built or tested in the development environment
(Docker Hub blocked). Render's native runtime uses the same commands that were verified
locally in a clean virtualenv: `pip install .`, then
`alembic upgrade head && uvicorn …` against a fresh database with a plain
`postgresql://` URL.
**Changed from Phase 0,** which planned a Dockerfile at deploy time. docker-compose is
still used for local Postgres.
**Status:** adopted.

## D-017 No learned ranker; no LLM
There is no real interaction data yet (0 real-user feedback events). Training on
synthetic or self-generated labels would be meaningless. Explanations are templates over
logged signals, so an LLM would add cost and risk, not information.
**Status:** adopted. See recommendation-engine §10 for the conditions to revisit.
