# Detour: a serendipity-aware recommendation engine

Detour recommends GitHub repositories that are **relevant** to your interests, **new to
you**, and **different from each other**. It deliberately avoids the "more of exactly
what you already like" failure mode of relevance-only ranking, and every pick explains
itself using the signals that actually produced it.

```
interests ─► GitHub search: core / bridge / adjacent-topic queries ─► ~150–300 candidates
          ─► pretrained embeddings ─► per-interest calibrated relevance + gate
          ─► user-relative novelty ─► MMR diversification (+ owner cap, interest coverage)
          ─► top-K with explanations ─► feedback ─► updated profile ─► different next list
```

## What works today
- **End to end on live GitHub data:** React UI, then the FastAPI API, then the
  recommendation engine, with PostgreSQL behind it. Verified with real GitHub responses.
- **Three feedback signals with distinct meanings:**
  - *Interested* → a relevance anchor.
  - *Not for me* → a local penalty, and interest decay after repeated rejections.
  - *Already know it* → familiarity only (lowers novelty, never relevance).
- **Every run is logged** with its config, profile snapshot and timings, together with
  **every** scored candidate (shown or not) and point-in-time features. This is the
  dataset a future learned ranker would need.
- **Five comparable rankers** on identical candidate pools: B0 popularity, B1 relevance
  only, D1 + novelty, D2 + MMR, D3 full. The UI can show D3 next to B1 for the same
  request.
- **Offline evaluation** on a GitHub-stars temporal split (anonymized, frozen,
  replayable), with bootstrap confidence intervals and a retrieval ablation.
- **Graceful degradation:** cache, then stale cache, then 503. Results are never
  fabricated.

## Results so far (measured; see [evaluation](docs/evaluation.md))
- **Relevance calibration (live pool, 176 candidates, 3 interests):** raw cosine let
  38% of off-topic candidates through the gate and ranked zero repos for one interest.
  The per-interest background-margin gate lets 6% through, keeps 94% of on-topic
  candidates, and balances the interests (8/8/8 in the top 24).
- **Diversity and popularity:** on the offline benchmark, D3 raises intra-list diversity
  over B1 (paired bootstrap CI above zero) and recommends far less popular repositories,
  with interest coverage kept at 100%.
- **Accuracy on the GitHub-stars benchmark is poor for every semantic variant,** and the
  popularity baseline wins. The cause is diagnosed, not hidden: retrieval finds under 1%
  of later-starred repos, and stars skew popular and off-profile. Small sample; see the
  evaluation doc for the exact numbers and limitations.

## Docs
| | |
|---|---|
| [Recommendation engine](docs/recommendation-engine.md) | every stage, formula, parameter, and why |
| [Evaluation](docs/evaluation.md) | protocol, measured results, limitations |
| [Architecture](docs/architecture.md) | modules, request flow, failure matrix |
| [Data model](docs/data-model.md) | schema and logging design |
| [API](docs/api.md) | endpoints, errors, smoke test |
| [Performance](docs/performance.md) | measured latency and memory |
| [Deployment](docs/deployment.md) | Neon + Render + Vercel |
| [Decisions](docs/decisions.md) | D-001 … D-017 |
| [Development log](docs/development-log.md) | what was done and verified, in order |

## Run it locally

Requires Python ≥ 3.12, Node ≥ 20, and Docker (or any PostgreSQL 16).

```bash
# 1. database
docker compose up -d                 # Postgres 16 with the detour + detour_test databases

# 2. API
cd backend
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp ../.env.example .env              # optionally add GITHUB_TOKEN=... (no scopes needed)
alembic upgrade head
uvicorn detour.main:create_default_app --factory --reload     # http://localhost:8000/docs

# 3. frontend (new terminal)
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

Without `GITHUB_TOKEN`, GitHub allows 10 searches a minute and a first request uses up to
10, so expect occasional "some searches didn't complete" notices. A token without scopes
raises the limit to 30 a minute.

### Checks (identical to CI)
```bash
cd backend
ruff check . && ruff format --check . && mypy
alembic upgrade head && alembic check
TEST_DATABASE_URL=postgresql+psycopg://detour:detour@localhost:5432/detour_test \
  DETOUR_REQUIRE_DB_TESTS=1 pytest -W error::DeprecationWarning
cd ../frontend && npm run typecheck && npm test && npm run build
```

### Demo / smoke test against any running API
```bash
python -m detour.tools.demo_session --api http://localhost:8000 \
  --interest "Distributed Systems" --interest "Machine Learning" --interest "Competitive Programming"
```

### Offline evaluation
```bash
python -m detour.eval.collect users --data ../data/eval --max-users 40   # resumable; respects rate limits
python -m detour.eval.collect pools --data ../data/eval
python -m detour.eval.offline --data ../data/eval --out ../data/eval/results.json --sweep
```

## Stack and deliberate omissions
FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL 16 · NumPy · WordLlama (pretrained static
embeddings, bundled weights) · React + Vite + TypeScript · GitHub Actions.

There is no Redis, queue, vector DB, microservice, LLM call or trained model. Each was
considered, and none is justified by a measurement yet ([decisions](docs/decisions.md)).

## Known limitations
- **Static embeddings** (WordLlama) are weaker than transformer encoders and needed
  per-interest calibration. A transformer model could not be downloaded in the build
  environment.
- **The relevance gate** is weakly calibrated on one live pool using retrieval provenance
  as labels. It has not been checked against human judgements.
- **Novelty at cold start** is topic- and popularity-based. It becomes personal only
  after the user marks repos as known or liked.
- **The offline benchmark is small** and biased toward popular repositories. No human
  evaluation has been run yet; the protocol is defined in the evaluation doc.
- **There is no learned ranker,** because there is no real interaction data yet. The
  logging needed to train one is in place.
- **Anonymous users, no authentication** (v1).
