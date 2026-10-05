# Recommendation Engine

> Status: **implemented** (engine version 1.0.0). This document describes what the code
> does. Parameters marked ⚙️ are starting hypotheses, not tuned values; the parameter
> register (§11) says how each one was chosen. Measured results live in `evaluation.md`.

Code map: `retrieval/` (planner, GitHub client, cache) → `representation/` (interest
expansion, repo text, embeddings) → `ranking/engine.py` (pure ranking) →
`ranking/explain.py` → `recommend/service.py` (orchestration + logging) →
`profile/` (feedback → profile).

## 1. Goal

Recommend GitHub repositories that are

1. **relevant** to the user's interests,
2. **unfamiliar** to the user (novelty), and
3. **different from each other** within a list (diversity),

adapt to explicit feedback, and explain every pick from the signals that produced it.

Novelty and diversity are different properties and live in different stages:

- **Novelty** belongs to an item relative to a user ("is this new to them?"). It is an
  item-level score.
- **Diversity** belongs to a list ("are these different from each other?"). It is handled
  by list-level reranking.

## 2. Pipeline

```
profile (interests, weights, exemplars, known set)          profile/builder.py
  └─► query planner: core / bridge / adjacent               retrieval/planner.py
        └─► GitHub search (cache → live → stale)            retrieval/service.py
              └─► candidate pool, deduped, ~150–300         (families remembered)
                    └─► embeddings (content-hash cache)     representation/store.py
                          └─► filters → relevance → gate → novelty → negative penalty
                                → utility → MMR + owner cap + coverage floor     ranking/engine.py
                                └─► structured reasons → text                     ranking/explain.py
                                      └─► log run + EVERY candidate               recommend/service.py
feedback (append-only) ─► next profile
```

**Candidate generation and ranking are separate problems.** Ranking can only reorder what
retrieval returned. The offline evaluation measures retrieval recall separately, and that
measurement shows retrieval is currently the bottleneck (`evaluation.md`).

## 3. User representation (`ranking/profile.py`, `profile/builder.py`)

| Component | Contents | Source |
|---|---|---|
| Interests | label, embedded expansion, **background similarity**, effective weight | user input + curated dictionary |
| Positive exemplars | embeddings of repos whose latest feedback is INTERESTED | feedback |
| Negative exemplars | embeddings of repos whose latest feedback is NOT_INTERESTED | feedback |
| Known set `K` | INTERESTED and ALREADY_KNOW repos | feedback |
| Known topics `T_K` | curated topics of the interests, plus topics of repos in `K` | derived |
| Recently shown | repos shown to the user in the last 14 days ⚙️ | run log |

- **A set of interest vectors, never one average.** The average of unrelated interests
  matches none of them, and keeping them separate lets every pick name the interest it
  matched.
- **Interest expansion.** 34 curated interests map aliases (e.g. "ml", "cp", "k8s") to an
  embedded description, a keyword query, a bridge term and GitHub topic slugs. Unknown
  labels fall back to the raw text. Users can override the expansion. No LLM is involved.
- **The profile is derived, not mutated.** It is rebuilt from the append-only feedback log
  on every request, and the latest event per repository wins. Changing your mind therefore
  works, and the exact profile each run used is stored in `profile_snapshot`.

## 4. Repository representation (`representation/text.py`)

- **Embedded text:** the name split into words (`raft-rs` → "raft rs"), the description
  (first 400 characters), and topics.
- **Not embedded:** language (it would make every repo in a language similar), stars,
  forks and dates (used as separate metadata features), and the README (costs one API call
  per repo and is noisy; not implemented).
- **Embedding cache:** `repo_embeddings(repo_id, model_name)`, keyed by
  `sha256(embedded_text)`. A change to the description or topics triggers re-embedding; a
  change in star count does not.

**Embedding model:** WordLlama `l2_supercat` (256-d), a pretrained *static* sentence
embedding model whose weights ship inside its PyPI wheel. Static embeddings ignore word
order and are weaker than transformer encoders. Decision D-010 explains the trade-off.
Everything depends only on the `Embedder` protocol, and cached vectors are keyed by model
name.

## 5. Candidate generation (`retrieval/planner.py`, `retrieval/service.py`)

| Family | Query | Purpose |
|---|---|---|
| `core` | interest keywords `in:name,description,topics` | baseline coverage, one per interest (max 6) |
| `bridge` | bridge terms of two interests | repos relevant to two interests at once, up to 3 pairs |
| `adjacent` | `topic:X`, where X is frequent in core results (≥3 repos) but not in `T_K` and not generic | exploration, up to 2 topics |

- **Qualifiers on every query:** `pushed:>today-365d stars:>=20 archived:false fork:false` ⚙️.
- **Sort:** GitHub best-match. Sorting by stars is never used: it returns the same famous
  repos every time, which works against novelty.
- **Page size:** one page per query (50 core, 30 bridge/adjacent), no pagination.
  Candidates are deduped by `github_id`, and every family that retrieved a candidate is
  remembered.
- **Canonical planning:** interests are sorted before planning, so the same interest set
  always produces the same queries and therefore the same cache keys.
- **Search cache** (`search_cache`): stores an ordered list of repo ids per normalized
  query, with a 12 h TTL ⚙️. Metadata is upserted into `repositories` from the same
  response, so a cache hit costs zero API calls.
- **Degradation, never fabrication:**

  | Situation | What happens |
  |---|---|
  | Fresh cache entry | served from cache |
  | GitHub available | fetched live and cached |
  | GitHub failing, stale entry exists | stale entry served; run marked `degraded` with warnings |
  | GitHub failing, no entry | that query is skipped |
  | Every query fails | HTTP 503, and a `failed` run is logged |

- **GitHub client:** 10 s timeout, one retry on 5xx or transport errors, and typed errors.
  It fails fast without spending a request once the rate limit is known to be exhausted.

## 6. Scoring (`ranking/engine.py`)

All vectors are L2-normalized, so cosine similarity is a dot product.

### 6.0 Filters (every variant)
A candidate is removed if it is archived, a fork, has no description and no topics, has a
description longer than 1,000 characters (keyword-stuffed spam; live bridge results
contained 260 KB descriptions), has fewer than 20 stars, is already in `K`, or was shown
recently. Every filtered candidate
is still logged, with its `filter_reason`.

### 6.1 Relevance: calibrated per interest

Raw cosine scales differ a lot per interest with static embeddings. On a live pool, the
mean on-target cosine was 0.39 for "Distributed Systems" and 0.58 for "Competitive
Programming", so ranking by raw max-cosine showed **zero** distributed-systems repos.
Relevance is therefore calibrated per interest:

```
s_ij        = cos(repo_i, interest_j)
background_j = mean cos(interest_j, other curated interest descriptions)   (representation/background.py)
margin_ij   = s_ij − background_j                    ← used by the gate
pct_ij      = percentile of s_ij among this pool     ← used for ranking
rel_i       = max_j  w_j · pct_ij                    (w_j = 1 except D3, §8)
D3 only:    rel_i = max(rel_i, α · percentile(max_p cos(repo_i, exemplar_p)))     α = 0.5 ⚙️
matched     = argmax (the interest, or the liked repo, that produced rel_i)
```

- **Max, not mean:** a repo deeply relevant to one interest is not penalized for ignoring
  the others. Representing every interest is the reranker's job (§6.6).
- **Background similarity** is the cosine an off-topic text typically gets. On the live
  pool it closely matched the observed off-target mean (0.142 vs 0.134, 0.161 vs 0.120,
  0.212 vs 0.205).

### 6.2 Relevance gate (D1, D2, D3)
```
eligible ⇔ margin(matched interest) ≥ 0.10 ⚙️   or (D3) cos to a liked repo ≥ 0.5 ⚙️
```

**The gate is what separates novelty from randomness:** an unrelated item cannot be
rescued by being unfamiliar. The test `test_gate_excludes_novel_but_irrelevant_item` sets
β = 100 to prove this.

### 6.3 Novelty components (each in [0, 1], each logged)

| Component | Definition | Available at cold start? |
|---|---|---|
| `unfamiliarity` | 1 − max cos to repos in `K` | **No:** undefined (not 1.0) until familiarity data exists |
| `topical_novelty` | fraction of the repo's non-generic topics not in `T_K` | yes, if the repo has topics |
| `popularity_novelty` | 1 − percentile of log(1 + stars) in the pool | yes |

`novelty` is the mean of the defined components (equal weights ⚙️).

**Limitations:**
- At cold start, novelty is topic- and popularity-based only. It becomes personal once
  the user marks repos as known or liked; the UI says so.
- Inverse popularity can promote obscure, low-quality repos. The 20-star floor only
  partly mitigates this.

### 6.4 Negative penalty (D3)
```
neg = max cos to a rejected repo,  if that cosine ≥ 0.6 ⚙️, else 0
```
The penalty is **local**: it hits near-duplicates of a rejected repo, not the whole
interest the repo belonged to. The interest itself only decays after repeated
rejections (§8).

### 6.5 Utility (item level)
```
D1:  u = rel + β·novelty                    β = 0.3 ⚙️
D2:  u = rel
D3:  u = rel + β·novelty − γ·neg            γ = 1.0 ⚙️
```

### 6.6 Diversity: list-level reranking (D2, D3)
Greedy **MMR**:
```
next = argmax  λ·u(r) − (1−λ)·max_{s∈S} cos(r, s)        λ = 0.7 ⚙️
```
It runs subject to:
- **Owner cap:** at most 2 repos per owner ⚙️.
- **Interest-coverage floor:** when the number of remaining slots equals the number of
  interests not yet represented (and those interests have eligible candidates), the
  choice is restricted to them. This only ever changes the tail of the list.

Ties are broken by `github_id`, so the whole ranking is deterministic.

Alternatives considered:
- **xQuAD:** explicit intent coverage, but more parameters. The coverage floor captures
  the part that mattered.
- **DPP:** harder to explain and tune, and overkill at K ≈ 10.
- **Fixed quotas per interest:** ignore relevance strength.

## 7. Variants (`enums.RankerVariant`)

| ID | Ranking |
|---|---|
| **B0** | stars, descending (no gate) |
| **B1** | calibrated relevance `rel`, descending (no gate). **Primary baseline.** |
| **D1** | gate, then `rel + β·novelty`; no MMR |
| **D2** | gate, then `rel`, then MMR + owner cap + coverage floor |
| **D3** | gate (+ exemplar gate), then `rel + β·novelty − γ·neg` with decayed interest weights and liked-repo exemplars, then MMR + owner cap + coverage floor |

- Filters apply to every variant, so all variants rank the same pool.
- Personalization signals are used by D3 only, so their effect can be attributed to it.
- The API can rank a second variant on the **same** candidate pool (`compare_variant`).
  The UI uses this to show Detour next to the relevance-only list.

## 8. Feedback → profile

| Feedback | Evidence | Effect on the next run |
|---|---|---|
| INTERESTED | preference (+) | becomes a relevance anchor (exemplar) and passes the gate; added to `K` (not re-recommended); topics join `T_K` |
| NOT_INTERESTED | preference (−) | local penalty for near-duplicates; counts toward its matched interest's decay |
| ALREADY_KNOW | **familiarity** | added to `K` and `T_K`. This makes similar repos less *novel* but **not less relevant**; no penalty, no decay |

Interest decay is computed from the log:
`w = max(0.5, 0.85^(rejections − 2))` once an interest has more than 2 rejections ⚙️.

The end-to-end tests check each of these behaviours through the API.

Not clicking is **not** treated as negative feedback, because clicks are confounded by
position and attention.

## 9. Explanations (`ranking/explain.py`)

Structured reasons are logged in `run_candidates.reasons` and rendered with fixed
templates. Each claim maps to a computed signal:

| Claim | Signal |
|---|---|
| "Strong match with your X interest" | matched-interest percentile ≥ 0.8 |
| "Also relevant to Y" | second interest's margin ≥ 0.10 (same rule as the gate) |
| "Similar to R, which you marked as interesting" | exemplar produced the relevance |
| "Found by a query bridging A and B" | retrieved by a bridge query |
| "Exploration pick: found via 'T'" | retrieved by an adjacent-topic query |
| "Introduces topics you haven't interacted with" | non-generic topics not in `T_K` |
| "Unlike the repositories you've marked as known" | unfamiliarity ≥ 0.6, only when `K` is non-empty |
| "Less-known project (N stars)" | stars < 2,000 |
| "Moved up from #N … to keep the list varied" | MMR promoted it ≥ 3 places over pure-utility order |

Tests assert both directions: claims appear when their signal holds, and do not appear
when it doesn't.

## 10. Interaction log and learned-model readiness

Every run logs **all** scored candidates (shown, unshown and filtered) with point-in-time
features: typed score columns, plus `features` JSONB holding stars, forks, language,
repo age, days since push, per-interest similarities, relevance margin, query labels,
nearest rejected repo, and whether MMR promoted the item. Labels come from `feedback`
joined on (user, repo, run).

**No learned ranker exists, deliberately.** It should be introduced only when all three
of these hold:
1. there are enough labeled impressions for a stable per-user cross-validated model,
   judged from learning curves on real data;
2. position bias is handled (position as a feature fixed at inference time, or
   inverse-propensity weighting);
3. it beats D3 on held-out users with the pre-registered metrics.

At the time of writing there are **zero real-user feedback events**, so none of these
conditions can be met.

## 11. Parameter register

| Parameter | Value | How it was chosen |
|---|---|---|
| relevance margin | 0.10 | weakly calibrated on one live pool using retrieval provenance as labels: 94% on-target vs 6% off-target pass (`evaluation.md` §7.1) |
| ranking relevance | per-interest percentile | live pool: top-24 interest mix 8/8/8 vs 14/0/10 with raw cosine |
| exemplar gate / α | 0.5 / 0.5 | hypothesis |
| β novelty | 0.3 | hypothesis; not tuned |
| γ, θ_neg | 1.0, 0.6 | hypothesis; 0.6 ≈ near-duplicate cosine for this model |
| λ MMR | 0.7 | hypothesis; not tuned |
| owner cap | 2 | judgement |
| star floor, recency | 20, 365 days | judgement |
| cache TTL | 12 h | GitHub search budget |
| decay | after 2 rejections, ×0.85, floor 0.5 | judgement |
| K | 10 (API), 8 (UI) | product choice |

None of the hypothesis parameters were tuned on evaluation data. Sweeping them
(especially λ and β) against the offline benchmark is listed as future work.
