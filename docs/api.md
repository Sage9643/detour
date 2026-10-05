# API

FastAPI serves interactive docs at `/docs` and the schema at `/openapi.json`. Errors use
the shape `{"detail": {"code": "...", "message": "..."}}`. Validation errors are FastAPI's
standard 422.

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/health` | liveness | no dependencies touched |
| GET | `/health/ready` | readiness | 200 only if the DB is reachable **and** the schema is at the code's Alembic head |
| GET | `/interests/catalog` | curated interests | any other label is also accepted |
| POST | `/users` | create an anonymous user | 201 |
| GET | `/users/{id}` | user, active interests, **derived profile** | effective weights, liked, rejected, known counts |
| PUT | `/users/{id}/interests` | replace the active interest set | 1–6 labels, unique case-insensitively; rows are deactivated, not deleted |
| POST | `/users/{id}/recommendations` | run the pipeline once and log it | body `{k: 1–30, variant: "D3", compare_variant?: "B1"}` |
| POST | `/users/{id}/feedback` | append a feedback event | body `{repo_id, type, run_id?}`; returns the updated profile |

## Recommendation response (abridged)

```json
{
  "run": {
    "run_id": "…", "variant": "D3", "status": "ok | degraded",
    "items": [{
      "position": 0,
      "repo": {"github_id": 1, "full_name": "…", "stars": 812, "topics": ["…"], "…": "…"},
      "scores": {"relevance_raw": 0.48, "relevance_margin": 0.32, "relevance_norm": 0.97,
                 "novelty": 0.86, "unfamiliarity": null, "topical_novelty": 1.0,
                 "popularity_novelty": 0.73, "utility": 1.23, "mmr_score": 0.71,
                 "max_sim_to_selected": 0.41, "pre_rerank_rank": 4,
                 "interest_sims": {"Distributed Systems": 0.48, "…": 0.1}},
      "matched_interest": "Distributed Systems",
      "query_families": ["core", "bridge"],
      "reasons": {"…": "structured, logged"},
      "explanation": ["Strong match with your Distributed Systems interest.", "…"]
    }],
    "stats": {"candidates": 176, "eligible": 163, "shown": 8,
              "by_family": {"core": 149, "bridge": 30},
              "filtered": {"below_relevance_gate": 12, "no_text": 1},
              "queries": [{"family": "core", "label": "…", "source": "live|cache|stale|failed", "…": "…"}],
              "github_calls": 5, "embedding_cache": {"hits": 0, "misses": 176},
              "timings_ms": {"profile_ms": 9, "retrieval_ms": 1706, "embedding_ms": 174,
                             "ranking_ms": 5, "logging_ms": 86, "total_ms": 1982}},
    "profile": {"interests": ["…"], "positives": [], "negatives": [], "known_count": 0, "cold_start": true},
    "warnings": []
  },
  "comparison": null
}
```

Every number in `stats` comes from the actual run; the UI shows them as they are.

## Status codes

| Code | When |
|---|---|
| 404 `user_not_found`, `repo_not_found`, `run_not_found` | unknown or foreign ids |
| 409 `no_interests` | recommendations requested before any interest was set |
| 422 `not_shown` | feedback that references a run in which the repo was not shown |
| 503 `github_unavailable` | every GitHub query failed and nothing was cached (a `failed` run is still logged) |
| 503 `database_unavailable` | Postgres unreachable; no SQL or connection details are leaked |

## Smoke test / demo
```bash
python -m detour.tools.demo_session --api http://localhost:8000 \
  --interest "Distributed Systems" --interest "Machine Learning" --out report.json
```
This runs the full loop: user, interests, D3 next to B1 on the same pool, three kinds of
feedback, then a second run. It exits non-zero on any failure.
