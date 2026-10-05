# Detour: a serendipity-aware recommendation engine

Detour recommends GitHub repositories that are **relevant** to your interests, **new to
you**, and **different from each other**. It deliberately avoids the "more of exactly
what you already like" failure mode.

> **Status: Phase 1 of 12 (foundation).** The schema, configuration, migrations, CI and
> specification exist. No retrieval, embeddings or ranking are implemented yet, and no
> evaluation results exist yet.

## How it will work

```
interests ─► GitHub search (core / bridge / adjacent queries) ─► ~150–300 candidates
          ─► embeddings ─► relevance gate ─► novelty ─► MMR diversity rerank ─► top-K
          ─► explanation from the actual scoring features ─► feedback ─► updated profile
```

Every run logs **all** scored candidates and their features, so ranking variants
(popularity B0, relevance-only B1, … full Detour D3) can be compared on identical pools,
and a learned ranker can be trained later *if* the data justifies one.

## Docs
- [Recommendation engine specification](docs/recommendation-engine.md)
- [Evaluation protocol](docs/evaluation.md)
- [Data model](docs/data-model.md)
- [Architecture](docs/architecture.md)
- [Engineering decisions](docs/decisions.md)
- [Development log](docs/development-log.md)

## Local development

Requirements: Python ≥ 3.12, Docker (for Postgres), or any local PostgreSQL 16.

```bash
docker compose up -d                       # Postgres 16 with detour + detour_test DBs
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp ../.env.example .env                    # local defaults; never commit .env

alembic upgrade head                       # apply migrations
uvicorn detour.main:create_default_app --factory --reload
curl localhost:8000/health/ready
```

### Checks (identical to CI)
```bash
cd backend
ruff check . && ruff format --check .
mypy
alembic upgrade head && alembic check
TEST_DATABASE_URL=postgresql+psycopg://detour:detour@localhost:5432/detour_test \
  pytest -W error::DeprecationWarning
```

The tests drop and recreate the schema of `TEST_DATABASE_URL` and refuse to run unless
the database name ends in `_test`.

## Stack
FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL 16 (Neon in production) · GitHub REST API ·
React + Vite + TypeScript (later) · GitHub Actions. No Redis, queue or vector DB unless a
measurement demands one; see [decisions](docs/decisions.md).
