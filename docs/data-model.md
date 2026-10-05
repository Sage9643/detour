# Data Model

> Implemented in migration `0001` (Phase 1). Verified by `tests/test_migrations.py` and
> `tests/test_schema_constraints.py`.

## Three kinds of data

| Kind | Examples | Freshness | Where |
|---|---|---|---|
| **Persistent user data** | users, interests, feedback | permanent | `users`, `interests`, `feedback` |
| **Interaction log** (for evaluation and future training) | runs, every scored candidate with features | permanent, append-only | `recommendation_runs`, `run_candidates` |
| **Cached external data** | GitHub metadata, search results, embeddings | TTL / content-hash | `repositories` (now); `search_cache` (Phase 2), `repo_embeddings` (Phase 3) |

## Entity diagram

```
users ─┬─< interests
       ├─< recommendation_runs ─< run_candidates >─ repositories
       └─< feedback >────────────────────────────── repositories
                 └── run_id (nullable) ─> recommendation_runs
run_candidates.matched_interest_id ─> interests (SET NULL; label snapshot kept)
```

## Tables

### `users`
`id uuid PK` · `github_username text?` · `created_at`. Users are anonymous in v1.

### `interests`
`id uuid PK` · `user_id → users (CASCADE)` · `label` (1–100 chars) · `expanded_text?` ·
`weight > 0 (default 1)` · `active` · timestamps.
Unique on `(user_id, lower(label))`, so "ML" and "ml" are the same interest.

### `repositories`
Keyed by **GitHub numeric id**, which survives renames (`full_name` does not).
Metadata only: `full_name, owner_login, name, description, topics[], language, stars,
forks, archived, is_fork, license_spdx, html_url, gh_created_at, pushed_at,
first_seen_at, metadata_fetched_at`. The schema rejects negative stars or forks.

### `recommendation_runs`
One row per request: `variant ∈ {B0,B1,D1,D2,D3}`, `engine_version`, `config` (JSONB),
`config_hash`, `profile_snapshot` (the profile at request time), `status ∈
{ok,degraded,failed}`, `candidate_count`, `shown_count`, `timings_ms` (JSONB per
stage), `error`.

### `run_candidates`: the core of the interaction dataset
**Every** candidate scored in a run, whether shown or not.

| Group | Columns |
|---|---|
| identity | `run_id → runs (CASCADE)`, `repo_id → repositories (RESTRICT)`; unique `(run_id, repo_id)` |
| provenance | `query_families text[]` (non-empty, values in `{core,bridge,adjacent}`), `filter_reason?` |
| outcome | `shown`, `position?` (0-based), `pre_rerank_rank?` |
| relevance | `matched_interest_id?`, `matched_interest_label?` (snapshot), `relevance_raw?`, `relevance_norm?` |
| novelty | `unfamiliarity?`, `topical_novelty?`, `popularity_novelty?`, `novelty?` |
| utility / diversity | `negative_penalty?`, `utility?`, `mmr_score?`, `max_sim_to_selected?` |
| extensible | `features jsonb` (point-in-time feature vector), `reasons jsonb?` (structured explanation) |

Invariants enforced by the database:
- `shown ⇔ position IS NOT NULL`; `position ≥ 0`
- positions are unique within a run (partial unique index), while any number of
  unshown rows is allowed
- a filtered candidate cannot be shown

### `feedback`
Append-only events: `user_id → users (CASCADE)`, `repo_id → repositories (RESTRICT)`,
`run_id → runs (SET NULL)`, `type ∈ {INTERESTED, NOT_INTERESTED, ALREADY_KNOW}`,
`created_at`. Current state is the latest event per `(user, repo)`. `run_id` is NULL for
feedback given outside a run (e.g. onboarding "repos I already know").

## Design decisions

- **Log unshown candidates.** A learned ranker trained only on shown items learns from
  data the current ranker already filtered (selection bias). Keeping full pools also
  lets new rankers be replayed offline on real request pools.
  *Cost:* about 150–300 rows per run instead of about 10. Negligible at this scale;
  sampling would be needed at real scale.
- **Point-in-time features.** Values are written at request time and never recomputed,
  because recomputing with today's stars or profile would leak future information into
  evaluation and training.
- **Typed columns plus JSONB.** The features analysis depends on are typed columns
  (type-checked and easy to query). `features` holds the full vector, so a new feature
  does not require a migration.
- **TEXT + CHECK instead of native ENUM types.** Changing an allowed value means
  replacing a constraint; native enums need `ALTER TYPE`, which is harder to downgrade.
  The Python `StrEnum`s generate the CHECK expressions, and tests insert every enum
  value, so the two cannot silently drift apart.
- **Deletion semantics.** Deleting a user removes all of their data (privacy). Deleting
  an interest keeps the log row and its label snapshot. A repository referenced by the
  log cannot be deleted.
- **Embeddings are not stored yet.** Their storage format (`real[]` vs `bytea`) and
  dimension depend on the embedder chosen in Phase 3; they will arrive as a new
  migration. pgvector is not needed at about 300 vectors per request.
