# Deployment

Target: **Neon** (Postgres), **Render** (API, native Python runtime), **Vercel**
(frontend). Nothing has been deployed yet: these steps need your accounts. The
build and start commands below were verified locally in a clean virtualenv against a
fresh database.

## 1. Database: Neon
1. Create a project and copy its connection string
   (`postgresql://…neon.tech/…?sslmode=require`).
2. No manual schema work is needed: the API runs `alembic upgrade head` on start.
   Plain `postgresql://` / `postgres://` URLs are mapped to the psycopg driver
   automatically.

## 2. API: Render
1. New, Blueprint, select this repository. `render.yaml` defines the `detour-api` service:
   - rootDir `backend`
   - build: `pip install .`
   - start: `alembic upgrade head && uvicorn detour.main:create_default_app --factory --host 0.0.0.0 --port $PORT`
   - health check: `/health`
2. Set the environment variables when prompted:

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | the Neon connection string |
   | `GITHUB_TOKEN` | a GitHub token with **no scopes** (public read). Optional, but raises search from 10 to 30 requests/min |
   | `DETOUR_CORS_ORIGINS` | your Vercel URL, e.g. `https://detour.vercel.app` |
3. Verify: `curl https://<service>.onrender.com/health/ready` should return `"status":"ready"`
   with `schema_revision` equal to `expected_revision`.

## 3. Frontend: Vercel
1. New Project, select this repository, root directory `frontend` (framework: Vite;
   `vercel.json` is included).
2. Environment variable `VITE_API_URL=https://<service>.onrender.com`.
3. Deploy, then add the resulting URL to `DETOUR_CORS_ORIGINS` on Render.

## 4. Smoke test the deployment
```bash
cd backend && pip install -e . && \
python -m detour.tools.demo_session --api https://<service>.onrender.com \
  --interest "Distributed Systems" --interest "Machine Learning"
```

## Secrets
No secret is committed. `.env` files are gitignored, and `GITHUB_TOKEN` and
`DATABASE_URL` live only in the Render dashboard (or your local `.env`). The GitHub client
never logs or records the Authorization header.

## CI
`.github/workflows/ci.yml` runs on push and PR:
- **backend:** ruff, mypy `--strict`, then migrations apply, `alembic check` (no drift)
  and a down/up round-trip, then pytest against a `postgres:16` service with database
  tests *required*.
- **frontend:** `npm ci`, typecheck, vitest, then a production build.

Deploys are triggered by Render/Vercel's GitHub integrations on `main`, not by CI.
