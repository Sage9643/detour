# Detour

## GitHub Repository Discovery & Recommendation Engine

> Detour finds GitHub repositories relevant to your interests while deliberately
> introducing projects you are less likely to discover yourself. It balances relevance,
> novelty, and diversity instead of simply recommending popular repositories.

## Live Demo

🚀 Try Detour — https://detour-pink.vercel.app/

📚 API Documentation — https://detour-api-sgia.onrender.com/docs

❤️ API Health — https://detour-api-sgia.onrender.com/health/ready

The backend runs on Render's free tier, so the first request after a period of inactivity
can take noticeably longer while the service wakes up.

---

## The problem

A standard recommender ranks repositories by how similar they are to what you already
like. That tends to return the same famous projects, and lists of near-duplicates.

Detour looks for repositories that are:

- **relevant** enough to your interests;
- **unfamiliar** enough to be worth discovering;
- **different** enough from the other recommendations in the same list.

Detour treats these as three separate properties and handles each at a different stage:

| Property | What it describes | How Detour handles it |
|---|---|---|
| **Relevance** | the repo matches your interests | semantic similarity, plus a relevance gate |
| **Novelty** | the repo is new *to you* | unfamiliarity, new topics, lower popularity |
| **Diversity** | the *list* isn't repetitive | MMR reranking, an owner cap, interest coverage |

## Features

- **Live GitHub retrieval.** Three search strategies:
  - **core:** one search per interest;
  - **bridge:** searches that combine two interests;
  - **adjacent:** topics that frequently appear next to your interests.
- **Candidate pool.** Results are deduplicated, and every strategy that found a repo is
  remembered.
- **GitHub search caching** in PostgreSQL. When GitHub fails or rate-limits, a stale
  cached result is served instead (the run is marked *degraded*).
- **Relevance.** Pretrained embeddings give a semantic similarity score, calibrated per
  interest, and a **relevance gate** removes off-topic candidates.
- **Novelty scoring:** unfamiliarity relative to repos you know, topics you haven't seen,
  and lower popularity.
- **Diversity:** MMR reranking, an **owner cap** (at most 2 repos per owner), and an
  **interest-coverage** guarantee.
- **Feedback:** *Interested*, *Not for me*, *Already know it*. It changes your next
  recommendations.
- **Persistent profile:** anonymous user, interests, and feedback history are stored in
  PostgreSQL.
- **Structured explanations.** Every recommendation says why it was picked, using only
  signals the ranker actually computed.
- **Relevance-only comparison.** The UI can show Detour's list next to a relevance-only
  ranking of the same candidates.
- **Logging.** Every recommendation run is stored, together with **every** scored
  candidate and its point-in-time features.
- **Evaluation tooling:** an offline GitHub-stars benchmark, baselines, a parameter
  sweep, and a blind A/B study mode (`?study`).

## How it works

```
User interests
      ↓
Profile construction        (interests, liked / rejected / known repos)
      ↓
Query planning
      ↓
GitHub retrieval            (core + bridge + adjacent, cached)
      ↓
Candidate pool              (~150–300 deduplicated repositories)
      ↓
Repository embeddings       (cached by content hash)
      ↓
Relevance gate
      ↓
Novelty + feedback signals
      ↓
Utility scoring
      ↓
MMR diversification  + owner cap  + interest coverage
      ↓
Top-K recommendations
      ↓
Structured explanations
      ↓
Feedback updates the profile
```

**Candidate generation and ranking are separate stages.** Retrieval decides what *can*
be recommended. Ranking decides the order and the final list. A ranker cannot surface a
repository that retrieval never fetched, so the two are evaluated separately.

## Recommendation engine

Full details, formulas and every parameter are in
[docs/recommendation-engine.md](docs/recommendation-engine.md). In summary:

**User representation**
- One embedding **per interest**, never a single averaged vector. A repo is matched to
  the specific interest it fits, and the explanation names it.
- Repos you marked *Interested* act as extra relevance anchors.
- Repos you marked *Not for me* penalize near-duplicates.
- Repos you know, and their topics, feed the novelty score.

**Repository representation**
- Embedded text: repository name, description, and topics.
- Embedding model: **WordLlama `l2_supercat`, 256 dimensions**. It is a pretrained static
  embedding model whose weights ship inside its Python package.
- Metadata such as stars, language, and dates is kept separate from the embedding and
  used as its own signal.

**Candidate generation**
- **Searches:** core searches per interest, bridge searches between pairs of interests,
  and adjacent-topic searches for exploration.
- **Deduplication** by repository ID.
- **Caching:** each search is cached for 12 hours, with a stale fallback when GitHub is
  unavailable or rate-limited.

**Relevance**
- Semantic similarity between a repo and each interest.
- Raw similarity scales turned out to differ a lot between interests. Relevance is
  therefore calibrated per interest: a margin over that interest's "background"
  similarity, plus a percentile within the candidate pool.
- **Relevance gate:** a candidate must clear that margin. This stops a highly novel but
  irrelevant repository from being "rescued" by its novelty.

**Novelty**
- **Unfamiliarity:** distance from the repos you've marked as known or liked.
- **Topic novelty:** the share of the repo's topics you haven't interacted with.
- **Popularity novelty:** less-starred repos within the candidate pool score higher.

**Diversity**
- **MMR (Maximal Marginal Relevance):** each pick trades off its own score against its
  similarity to repos already in the list.
- **Owner cap:** at most 2 repositories per owner.
- **Interest coverage:** every interest appears in the list at least once, when it has
  eligible candidates.

**Feedback**

| Action | Meaning | Effect |
|---|---|---|
| **Interested** | positive preference | similar repos become more relevant |
| **Not for me** | negative preference | near-duplicates are penalized; repeated rejections lower that interest's weight |
| **Already know it** | **familiarity, not dislike** | similar repos become less *novel*, but **not** less relevant |

Detour does **not** use a learned ranking model. It combines pretrained embeddings with
transparent, deterministic scoring rules. See [Limitations](#limitations).

## Architecture

```
React + TypeScript + Vite   (Vercel)
            ↓
FastAPI backend             (Render)
 ├── GitHub REST API        (repository search)
 └── WordLlama embeddings   (in-process)
            ↓
PostgreSQL                  (Neon)
```

| Component | Hosting |
|---|---|
| Frontend | Vercel |
| Backend | Render |
| Database | Neon (PostgreSQL) |
| CI | GitHub Actions |

The backend is intentionally a **modular monolith**: one FastAPI service with separate
modules for retrieval, representation, ranking, profiles and evaluation. The current
workload doesn't need Kafka, RabbitMQ, Redis, or Kubernetes. GitHub's rate limit is the
real bottleneck, and caching search results in PostgreSQL already handles it. See
[docs/architecture.md](docs/architecture.md) and [docs/decisions.md](docs/decisions.md).

## Evaluation

Full method, tables and confidence intervals are in
[docs/evaluation.md](docs/evaluation.md).

**Offline benchmark.** It uses a temporal split of GitHub stars for **24 users**. Repos a
user starred before a cutoff date count as "known"; repos they starred afterwards are
the held-out targets. Five rankers are compared on identical candidate pools:

| ID | Ranker |
|---|---|
| B0 | popularity |
| B1 | relevance only |
| D1 | relevance + novelty |
| D2 | relevance + diversity |
| D3 | full Detour |

Values are means over users, with 95% bootstrap confidence intervals.

| Finding | Result |
|---|---|
| **Retrieval is the main bottleneck** | Only **0.5%** [0.0, 1.4] of later-starred repos reached the candidate pool with interest-only searches; **1.9%** [0.4, 3.8] with core + bridge + adjacent. The extra search strategies add +1.4 points [+0.1, +2.9]. |
| **Accuracy vs. relevance-only** | No measurable difference: D3 − B1 recall = **+0.002** [−0.008, +0.014] (held-out repos injected into the pool) |
| **Diversity** | D3 raised intra-list diversity by **+0.097** [+0.075, +0.119] over B1, higher for 23 of 24 users |
| **Popularity** | D3's picks: mean log10 stars **2.53**, vs. 3.61 for relevance-only and 4.96 for the popularity baseline (B0), i.e. far less popular repositories |
| **Interest coverage** | **100%** for D3; 87.5% for novelty without the diversity stage (D1) |
| **Benchmark bias** | The popularity baseline scored best (recall 0.155 when held-out repos were injected). Stars skew toward popular repositories, so this benchmark structurally favours popularity. |

**What this means.** Detour reliably produces more diverse, less popular,
interest-balanced lists, without a measurable accuracy cost on this benchmark. It
has **not** been shown to improve user satisfaction:

- **Human evaluation was not completed.** Blind A/B tooling exists, but no study was run.
- **There is not enough real-user feedback to justify a learned ranker.**

The relevance gate was calibrated on a live candidate pool (176 repositories). It cut the
share of off-topic candidates passing the gate from 38% to 6%, while keeping 94% of
on-topic ones.

## Engineering highlights

- **Separate stages.** Candidate generation is separated from ranking and measured
  independently, which is how retrieval was identified as the bottleneck.
- **A relevance gate** prevents novelty from rescuing irrelevant repositories. A unit test
  proves it, even with an extreme novelty weight.
- **Separate objectives.** Novelty and diversity are distinct: novelty is an item-level
  score relative to the user; diversity is a list-level reranking (MMR).
- **Point-in-time candidate logging.** Every scored candidate is stored, shown or not,
  with the features it had at request time. This supports leakage-safe offline analysis
  and any future learned ranker.
- **Append-only feedback events.** The profile is derived from the log, so changing your
  mind works and history is kept.
- **Caching and stale fallback** cope with GitHub API limits. Results are never
  fabricated: if nothing is available, the API returns a clear 503.
- **Deterministic explanations** are generated from the actual ranking signals, and tests
  check each claim against the scores behind it.
- **Multiple baselines** (popularity, relevance-only, and three Detour variants) are
  evaluated on identical candidate pools.

## Tech stack

| Area | Tools |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI, SQLAlchemy, Alembic, Pydantic, httpx |
| ML / recommendation | WordLlama (pretrained embeddings), NumPy; semantic similarity, novelty scoring, MMR |
| Database | PostgreSQL (Neon in production, Docker locally) |
| External API | GitHub REST API |
| Testing / quality | pytest, Vitest, Ruff, mypy (strict) |
| Deployment / CI | Vercel, Render, Neon, GitHub Actions |

## Project structure

```
detour/
├── backend/            FastAPI app, recommendation engine, migrations, tests
│   ├── src/detour/
│   ├── tests/
│   └── pyproject.toml
├── frontend/           React + TypeScript + Vite app (vercel.json)
├── docs/               architecture, engine, evaluation, API, deployment, ...
├── scripts/            local database init
├── docker-compose.yml  local PostgreSQL
├── render.yaml         Render deployment blueprint
└── README.md
```

## Local development

Requirements: Python ≥ 3.12, Node ≥ 20, and Docker (or any PostgreSQL 16). Locally,
PostgreSQL runs in Docker; production uses Neon.

```bash
git clone https://github.com/Sage9643/detour.git
cd detour

# 1. PostgreSQL (creates the detour and detour_test databases)
docker compose up -d

# 2. Backend
cd backend
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp ../.env.example .env                # Windows: copy ..\.env.example .env
alembic upgrade head
uvicorn detour.main:create_default_app --factory --reload    # http://localhost:8000/docs

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

`GITHUB_TOKEN` is optional. GitHub allows 10 searches a minute without it and 30 with
one. A token with no scopes is enough.

**Checks (same as CI):**

```bash
cd backend
ruff check . && ruff format --check . && mypy
alembic upgrade head && alembic check
TEST_DATABASE_URL=postgresql+psycopg://detour:detour@localhost:5432/detour_test \
  DETOUR_REQUIRE_DB_TESTS=1 pytest
cd ../frontend && npm run typecheck && npm test && npm run build
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/health/ready` | readiness: database reachable and schema up to date |
| GET | `/interests/catalog` | suggested interests |
| POST | `/users` | create an anonymous user |
| GET | `/users/{id}` | user, interests, and derived profile |
| PUT | `/users/{id}/interests` | set interests (1–6) |
| POST | `/users/{id}/recommendations` | run the pipeline (optionally with a relevance-only comparison) |
| POST | `/users/{id}/feedback` | record *Interested* / *Not for me* / *Already know it* |

Request/response formats and error codes are in [docs/api.md](docs/api.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Recommendation Engine](docs/recommendation-engine.md)
- [Evaluation](docs/evaluation.md)
- [Data Model](docs/data-model.md)
- [API](docs/api.md)
- [Deployment](docs/deployment.md)
- [Performance](docs/performance.md)
- [Engineering Decisions](docs/decisions.md)
- [Development Log](docs/development-log.md)

## Limitations

- **Retrieval is the primary bottleneck.** Most repositories a user later stars never
  enter the candidate pool.
- **The offline stars benchmark is popularity-biased,** and its sample is small (24
  users).
- **Human evaluation was not completed,** so user-perceived quality is unmeasured.
- **There is not enough real-user feedback for a learned ranker.** Ranking uses
  transparent rules over pretrained embeddings.
- **Cold-start novelty is limited.** For a new user it is based mainly on topics and
  popularity, and becomes personal only after feedback.
- **The embedding model** is a lightweight static one, chosen to fit free-tier hosting.
  It is weaker than transformer encoders.
- **Render free-tier cold starts** can add noticeable latency to the first request after
  inactivity.

## Security

- Secrets (`DATABASE_URL`, `GITHUB_TOKEN`) are set as deployment environment variables on
  Render. They are never committed.
- `.env` files are gitignored. Only `.env.example`, with local-development defaults, is
  in the repository.
- The Neon connection string and the GitHub token do not appear in the code, the docs,
  or this README.
- The GitHub client never logs or records its authorization header.
