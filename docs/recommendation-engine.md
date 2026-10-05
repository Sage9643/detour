# Recommendation Engine Specification

> Status: **specification (v0.1, Phase 1)**. Nothing in this document has been
> implemented or measured yet. All numeric parameters below are **starting hypotheses**,
> marked ⚙️, to be calibrated or swept in the phase indicated. No result in this file is
> an experimental finding; findings go in `evaluation.md` once measured.

## 1. Goal

Recommend GitHub repositories that are:

1. **relevant** to the user's interests,
2. **unfamiliar** to the user (novelty), and
3. **different from each other** within a list (diversity),

and adapt to explicit feedback (personalization). Every recommendation carries an
explanation derived from the logged scoring features.

**Novelty and diversity are different properties.**
- **Novelty** is a property of an *item relative to a user*: is it new to them?
- **Diversity** is a property of a *list*: are its items different from each other?

They are therefore computed at different stages.

## 2. Pipeline

```
Profile ─► Query planner ─► GitHub search (cached) ─► Candidate pool (~150–300)
        ─► Filters ─► Representation/embeddings ─► Relevance ─► Relevance gate
        ─► Novelty components ─► Negative penalty ─► Utility
        ─► Diversity rerank (MMR + owner cap) ─► Top-K ─► Structured reasons
        ─► Log run + ALL candidates ─► Feedback ─► Profile update
```

**Candidate generation and ranking are separate problems.** Ranking can only reorder what
retrieval returned. Retrieval quality is therefore measured on its own (pool recall,
§9), separately from ranking quality.

## 3. User representation

A **set** of interest vectors, never a single averaged vector.

| Component | Contents | Source |
|---|---|---|
| Interests | label, expanded text, embedding, weight `w_i` (default 1), active flag | user input |
| Positive exemplars | embeddings of repos marked INTERESTED | feedback |
| Negative exemplars | embeddings of repos marked NOT_INTERESTED | feedback |
| Known set `K` | repos marked ALREADY_KNOW or INTERESTED; optionally the user's GitHub stars | feedback / optional username |
| Known topics `T_K` | topics mapped from interests, plus topics of repos in `K` | derived |

**Why a set:** the centroid of unrelated interests (e.g. ML and competitive programming)
is a point that matches neither. Keeping vectors separate also preserves which interest
matched, which the explanations need.

**Interest expansion:** short labels ("ML", "CP") embed poorly. Each label is expanded
to a short description before embedding, using a curated dictionary with raw-text
fallback. The user can edit the expansion. No LLM is involved.

## 4. Repository representation

| Role | Fields |
|---|---|
| Embedded text | name split into tokens, description, topics |
| Metadata features (not embedded) | stars, forks, `pushed_at`, language, archived, is_fork, license |
| Display only | URL, owner |

- **Language is excluded from the embedding.** Otherwise all repos in one language become
  similar to each other.
- **README is excluded in v1.** Fetching it costs one API call per repo and the text is
  noisy. It is a candidate experiment for repos with missing descriptions (Phase 3).
- **Embedding cache key:** `(repo_id, model_name, sha256(embedded_text))`. A metadata
  change (e.g. stars) does not trigger re-embedding.

## 5. Candidate generation (Phase 2; bridge and adjacent queries in Phase 5)

| Family | Query | Purpose |
|---|---|---|
| `core` | key terms of one interest + `pushed:>now-12mo` ⚙️ + `stars:>=20` ⚙️ + `archived:false` | baseline coverage |
| `bridge` | combination of two interests | relevant surprise (relevant to two interests at once) |
| `adjacent` | topics that co-occur with interest topics in core results but are not in `T_K` | topical novelty |

- Best-match sort plus a recently-pushed sort. Stars sort is avoided because it always
  returns the same famous repos, which works against novelty.
- One page of up to 100 results per query; no pagination.
- Results are deduplicated by `github_id`. A candidate records *all* families that
  retrieved it (`query_families`).
- Search results are cached per normalized query with a TTL ⚙️ of about 6–12h, set in
  Phase 2. The main driver is GitHub's search rate limit, not latency.
- **Filters:** archived, forks, no description *and* no topics, below the star floor,
  repos already in `K`, repos shown to this user in the last N days ⚙️.
  Every filtered candidate is still **logged** with a `filter_reason`.

## 6. Scoring

Let `e_r` be the repository embedding and `u_i` the interest embeddings. All embeddings
are L2-normalized, so cosine similarity is a dot product.

### 6.1 Relevance
```
sim_i(r)      = cos(e_r, u_i)
rel_raw(r)    = max( max_i w_i·sim_i(r),  α·max_p cos(e_r, e_p) )    p ∈ positive exemplars, α=0.5 ⚙️
matched(r)    = argmax interest (or the exemplar's source interest)
rel_norm(r)   = within-pool rank normalization of rel_raw to [0,1]
```

- **Max, not mean:** a repo that is deeply relevant to one interest is not penalized for
  ignoring the others. Covering all interests is the diversity stage's job.
- **Raw score for gating, normalized score for combining.** Raw cosine ranges depend on
  the embedding model. Rank normalization is model-agnostic, but it hides the case where
  every candidate in the pool is weak. That is why the gate uses the raw score.

### 6.2 Relevance gate
```
eligible(r) ⇔ rel_raw(r) ≥ τ   and r ∉ K   and r not recently shown
```
τ ⚙️ is **calibrated** on a hand-labeled set of about 100 (interest, repo) pairs in
Phase 4. It is not guessed.

**The gate is what separates novelty from randomness:** novelty only reorders
candidates that are already relevant. An unrelated item cannot be rescued by being
unfamiliar.

### 6.3 Novelty components (each in [0, 1], all logged separately)

| Component | Definition | Works at cold start? |
|---|---|---|
| `unfamiliarity` | `1 − max_{k∈K} cos(e_r, e_k)`; 1 if `K` is empty | No (uninformative until feedback or stars exist) |
| `topical_novelty` | fraction of the repo's topics not in `T_K`; undefined if the repo has no topics | Partially |
| `popularity_novelty` | `1 − rank_norm(log(1+stars))` within the pool | Yes |

```
novelty(r) = mean of the defined components     (equal weights ⚙️; ablated in Phase 5)
```

**Limitations, stated explicitly:**
- At cold start, novelty is essentially popularity- and topic-based. It becomes
  personal only once familiarity data exists (feedback, onboarding ticks, or GitHub
  stars).
- Stars measure popularity *and* quality, so inverse popularity can promote obscure,
  low-quality repos. The retrieval star floor partly mitigates this.

### 6.4 Negative-feedback penalty
```
neg(r) = max_n cos(e_r, e_n)   if ≥ θ_neg ⚙️ else 0          n ∈ negative exemplars
```
The penalty is **local**: it suppresses near-duplicates of a rejected repo, not the whole
interest the repo belonged to.

### 6.5 Utility (item-level)
```
utility(r) = rel_norm(r) + β·novelty(r) − γ·neg(r)          β≈0.3 ⚙️, γ≈1.0 ⚙️ (swept)
```

### 6.6 Diversity rerank (list-level), Phase 6
Greedy **MMR** (Maximal Marginal Relevance) over embedding similarity:
```
S = ∅
repeat K times:
    r* = argmax_{r ∉ S, eligible, owner cap ok}  λ·utility(r) − (1−λ)·max_{s∈S} cos(e_r, e_s)
    S  = S ∪ {r*}
```
- λ ⚙️ is swept to produce a relevance–diversity tradeoff curve; the operating point is
  chosen using human judgments.
- **Owner cap:** at most 2 ⚙️ repos per owner. This is a hard constraint that embedding
  similarity does not reliably enforce.
- **Known weakness:** MMR reduces redundancy but does not guarantee coverage of every
  interest. Interest coverage is tracked; if it is poor, xQuAD-style intent-aware
  diversification is the next experiment.

Alternatives considered: xQuAD (more parameters), DPP (harder to explain and tune,
overkill at K≈10–20), fixed per-interest quotas (brittle, ignores relevance strength).

## 7. Ranker variants

All variants run on the **same candidate pool** (frozen snapshots offline, the same live
pool online).

| ID | Ranking | Uses |
|---|---|---|
| **B0** | stars, descending | popularity sanity floor |
| **B1** | `rel_raw`, descending, top-K | **primary baseline** |
| **D1** | gate → utility with novelty (no MMR) | novelty's contribution |
| **D2** | gate → `rel_norm` → MMR | diversity's contribution |
| **D3** | gate → full utility (novelty + negative penalty + feedback-derived profile) → MMR + owner cap | full Detour |
| *(future)* LTR | learned scorer over logged features, then MMR | only if §10's conditions hold |

The retrieval contribution (bridge and adjacent query families) is ablated separately by
running each variant on pools with and without those families.

## 8. Feedback → profile updates (Phase 7)

| Feedback | Evidence type | Update |
|---|---|---|
| INTERESTED | preference (+) | add positive exemplar; add to `K`; add topics to `T_K` |
| NOT_INTERESTED | preference (−) | add negative exemplar; after ≥3 ⚙️ rejections attributed to one interest, `w_i ← max(w_min, 0.85·w_i)` ⚙️ |
| ALREADY_KNOW | **familiarity**, and weak (+) preference | add to `K` and its topics to `T_K`; relevance is **not** penalized |

- Feedback is an append-only log. Profile state is derived from the latest event per
  (user, repo).
- **No feedback is not treated as negative feedback**: whether an item was clicked is
  confounded by its position and by user attention.

## 9. Explanations (Phase 8)

The ranker emits structured reasons, e.g.
`{matched_interest, rel_band, secondary_interest, new_topics, popularity_band, query_family}`.
A template renders them as text. Every phrase must map to a logged field; tests assert
each rendered claim against the candidate's scores. An LLM is used, at most, to rephrase
these reasons, never to decide them.

## 10. Interaction log and learned-model readiness

Every run logs **all** scored candidates (`run_candidates`), shown or not, with
point-in-time features. See `data-model.md` for the schema.

| Training feature (planned) | Where it lives |
|---|---|
| semantic relevance (raw, normalized) | `relevance_raw`, `relevance_norm` |
| novelty components | `unfamiliarity`, `topical_novelty`, `popularity_novelty`, `novelty` |
| negative penalty, utility | `negative_penalty`, `utility` |
| diversity context | `mmr_score`, `max_sim_to_selected`, `pre_rerank_rank` |
| matched interest | `matched_interest_id`, `matched_interest_label` |
| query family | `query_families` |
| position | `position` |
| popularity, recency, language match, topic overlap, prior feedback counts | `features` JSONB (point-in-time) |
| label | `feedback` joined on (user, repo, run) |

**Conditions for introducing a learned ranker. All must hold; none are assumed:**
1. There are enough labeled impressions that a per-user-grouped cross-validated model is
   stable. The threshold will be decided from learning curves on the actual data, not
   picked in advance.
2. **Position bias is handled.** Items shown higher get more feedback regardless of
   quality. Options: include position as a training feature and fix it to a constant at
   inference, or use inverse-propensity weighting.
3. On held-out users or sessions, the model beats D3 on the pre-registered metrics in
   `evaluation.md`.

If these conditions are never met, the project reports that, with the actual number of
interactions collected. That is an acceptable outcome.

## 11. Parameter register

| Parameter | Start | Decided in | Method |
|---|---|---|---|
| τ (relevance gate) | — | Phase 4 | calibration on labeled pairs |
| β (novelty weight) | 0.3 | Phase 5 | sweep + offline/human eval |
| γ, θ_neg | 1.0, — | Phase 7 | unit-level behaviour + eval |
| λ (MMR) | 0.7 | Phase 6 | tradeoff curve + human eval |
| owner cap | 2 | Phase 6 | inspection |
| star floor, recency window | 20, 12 months | Phase 2 | pool-quality inspection, pool recall |
| search cache TTL | 6–12 h | Phase 2 | rate-limit budget |
| K (list length) | 10 | Phase 9 | product choice |
