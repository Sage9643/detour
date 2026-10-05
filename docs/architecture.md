# Architecture

## Target (end of project)

```
React + Vite + TS (Vercel)
        │ HTTPS / JSON
FastAPI modular monolith (Render)
  ├─ api/             routes, request/response schemas, error mapping
  ├─ retrieval/       query planner, GitHub client, search cache         (Phase 2)
  ├─ representation/  text builder, interest expansion, Embedder, cache  (Phase 3)
  ├─ ranking/         relevance, novelty, utility, MMR, explanations     (Phases 4–8)
  ├─ profile/         feedback → profile state                           (Phase 7)
  ├─ eval/            snapshots, offline runner, metrics (CLI, not served) (Phase 4+)
  └─ db/              SQLAlchemy models, Alembic migrations
        │
PostgreSQL (Neon)        GitHub REST API        Embedding model (local ONNX or API)
```

**Deliberately absent:** Redis, message queues, Celery, Kubernetes, microservices, and a
vector database. Each is reconsidered only if a measurement shows a need (see
`decisions.md`).

## Implemented now (Phase 1)

```
backend/
  src/detour/
    main.py             create_app(settings) factory; uvicorn entry: create_default_app
    config.py           Settings from env vars (DATABASE_URL required)
    logging_setup.py    stdlib logging, key=value format
    enums.py            FeedbackType, RankerVariant, RunStatus, QueryFamily
    api/health.py       GET /health (liveness), GET /health/ready (DB + schema revision)
    db/
      models.py         6 tables (see data-model.md)
      session.py        engine/session construction
      migrations_info.py  expected vs current Alembic revision
      migrations/       Alembic env + versions/0001_initial_schema.py
  tests/                51 tests (config, health, migrations, schema constraints)
docker-compose.yml      local PostgreSQL 16 (+ detour_test database)
.github/workflows/ci.yml  lint → mypy → migrate/check/round-trip → pytest
```

### Request flow currently available
```
GET /health        → 200 {"status":"ok","version":...}        (no dependencies)
GET /health/ready  → 200 when DB is reachable and schema == code's Alembic head
                   → 503 otherwise, with the reason
```

### Module boundary rules
- `api/` contains no business logic; it calls domain modules.
- Domain modules (`retrieval`, `representation`, `ranking`, `profile`) do not import
  FastAPI.
- Ranking functions are pure functions over arrays and dataclasses, so they can be tested
  without a database or network and replayed offline on frozen snapshots.
- Only `db/` imports SQLAlchemy models.
