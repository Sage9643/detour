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

## D-009 Embedder default (open)
The Phase 0 lean is a local ONNX model, for reproducibility and zero cost. The
alternative is an external API, which is lighter to deploy. The decision will be made by
the Phase 3 benchmark (latency, memory on Render, quality on the labeled calibration
set).
**Status:** open.
