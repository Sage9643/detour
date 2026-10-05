# Architecture

A modular monolith: one FastAPI process, one Postgres database, two external
dependencies (GitHub's REST API and an in-process embedding model).

```
React + Vite + TS (Vercel)
        │  HTTPS / JSON
FastAPI app (Render) ─────────────────────────────────────────────────────────────
  api/            routes + schemas; error mapping (no stack traces, no DSNs)
  recommend/      orchestration: profile → retrieval → embeddings → rank → log
  profile/        feedback log → UserProfile (derived each request); feedback validation
  retrieval/      planner (pure) · GitHub client · search cache · record/replay transport
  representation/ interest dictionary · repo text · Embedder (WordLlama) · embedding cache
                  · background similarity
  ranking/        engine (pure: filters, relevance, gate, novelty, utility, MMR) · explain
  eval/           metrics · dataset collector · offline runner  (CLI only, never served)
  tools/          demo_session (end-to-end driver / smoke test)
  db/             SQLAlchemy models · Alembic migrations 0001, 0002 · repo store
──────────────────────────────────────────────────────────────────────────────────
        │                                  │
PostgreSQL (Neon)                    GitHub REST API (search, starred, forks)
users, interests, feedback,
recommendation_runs, run_candidates,
repositories, search_cache, repo_embeddings
```

**Deliberately absent:** Redis, queues, Celery, Kubernetes, microservices, a vector
database, LLM calls. Each absence is justified by a measurement or a decision in
`decisions.md`, e.g. the cache read is 17 ms p50 from Postgres, and scoring ~300 vectors
in NumPy takes about 3 ms.

## Module boundary rules (enforced by structure and tests)
- `ranking/` is pure: no database, network, or FastAPI. The same `rank()` runs online
  and in the offline evaluator, on identical inputs.
- `retrieval/planner.py` is pure. I/O lives in `retrieval/service.py` and
  `retrieval/github.py`.
- `api/` holds no business logic.
- Only `db/`, `profile/`, `recommend/`, `retrieval/service.py` and
  `representation/store.py` touch the database.

## Request: `POST /users/{id}/recommendations`
1. **profile** – load active interests and the feedback log; derive weights, exemplars,
   known set and recently shown; embed the interest expansions; compute each interest's
   background similarity.
2. **retrieval** – plan core and bridge queries; serve each from cache, or fetch it live
   (4 threads), or fall back to stale; then plan adjacent-topic queries from the core
   results and run them; dedupe.
3. **embedding** – reuse cached vectors keyed by text hash; embed misses; upsert.
4. **ranking** – `rank(pool, profile, variant, cfg)`; optionally a second variant on the
   same pool.
5. **explain** – structured reasons, then template sentences.
6. **log** – one `recommendation_runs` row (config, hash, profile snapshot, timings,
   status) plus one `run_candidates` row per candidate.
7. Respond with the items, stats (every number is real), profile and warnings.

A request is one database transaction. If retrieval fails completely, a `failed` run is
committed and 503 is returned.

## Dependency failure matrix

| Failure | Behaviour | Tested by |
|---|---|---|
| GitHub 5xx / timeout | one retry; then stale cache, else the query is skipped | `test_transient_5xx_is_retried_once`, `test_github_down_with_stale_cache_serves_degraded_results` |
| GitHub rate limit | no retry; fail fast for later queries; stale cache, else skip | `test_rate_limit_is_reported_not_retried` |
| All queries fail | 503 `github_unavailable`, `failed` run logged | `test_github_down_without_cache_returns_503_and_logs_failed_run` |
| Malformed response / unexpected error in one query | that query fails, the run degrades | `test_unexpected_error_in_one_query_degrades_instead_of_crashing` |
| Database down | 503 `database_unavailable`; `/health` still 200; `/health/ready` 503 | `test_database_unavailable_returns_503_without_leaking_details` |
| Schema behind code | `/health/ready` 503 with both revisions | `test_not_ready_when_schema_revision_mismatches` |
| Embedding model | in-process and bundled, so there is no network dependency to fail. A load failure fails startup loudly. | – |

## Frontend
A single page (`frontend/src`):
- **Setup:** interest picker (curated and free text, max 6), with an optional
  "compare with relevance-only" toggle.
- **Results:**
  - Cards show the explanation sentences, three signal bars (relevance percentile,
    novelty, distinctness = 1 − similarity to the items above), topics, and feedback
    buttons.
  - A left border marks how each repo was found: solid for an interest search, double for
    a bridge, dashed for exploration.
  - Side panels show the derived profile (effective interest weights, known/liked/rejected
    counts, cold-start note) and the run funnel (searches, candidates per family,
    filtered-out reasons, shown, data age, embedding cache hits, latency).
  - A compare mode puts D3 and B1 side by side and outlines picks that relevance alone
    would not have made.
- The user id is kept in `localStorage`; the app still works if storage is blocked.
