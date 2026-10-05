# Evaluation

> Every number in this document was measured, and each one names the data it came from.
> Sample sizes are small. Read §9 (limitations) before quoting any result.

## 1. Principles

1. **Separate retrieval from ranking.** A ranker cannot surface what retrieval never
   fetched, so pool recall is measured separately from ranking metrics.
2. **Frozen, replayable data.** Live GitHub results change daily. Every evaluation input
   is recorded, including the raw GitHub responses, and can be replayed.
3. **Same pool for all variants.** B0, B1, D1, D2 and D3 always rank the identical
   candidate pool for a given user.
4. **Beyond-accuracy metrics are partly circular.** MMR optimizes intra-list diversity
   directly, so "diversity went up" is true by construction. The non-trivial claims
   concern relevance and human preference.
5. **Pre-registered criteria** (§6). **Negative results are reported, not hidden.**

## 2. Evaluation layers and their status

| Layer | Status | Where |
|---|---|---|
| 1. Deterministic tests (hand-computed vectors) | **done**: MMR choice, gate vs. novelty, familiarity ≠ preference, local negative penalty, coverage floor, owner cap, calibration | `tests/test_ranking.py` |
| 2a. Relevance calibration on a live pool | **done** (§7.1) | live run, 2026-10-05 |
| 2b. Real-data replay of the full API | **done**: invariants asserted on recorded GitHub responses | `tests/test_replay_real_github.py` |
| 2c. Offline GitHub-stars temporal split | **done, small n** (§7.3–7.5) | `detour.eval` |
| 3. Human evaluation | **not run**: protocol and tooling ready (§8) | |
| 4. Live interaction log | **logging in place; no real users yet** (0 feedback events from real users) | `run_candidates`, `feedback` |

## 3. Offline dataset: GitHub stars, temporal split (`detour.eval.collect`)

**Construction, as implemented:**
1. **Users.** Owners of the 100 newest forks of 8 seed repositories spanning
   distributed systems, databases, competitive programming, ML, backend, observability
   and LLMs. Forks were used because GitHub's stargazers endpoint now requires
   authentication; fork owners are recently active developers. Logins are shuffled with
   a fixed seed.
2. **Stars.** Each user's starred repositories, with timestamps, newest first. Paging
   stops early once the outcome is decided, to save API budget.
3. **Split.** Cutoff `t` = collection date − 120 days.
   - `K` = stars before `t` (what the user already knew).
   - `H` = stars after `t`, excluding repos already in `K` and anything that would fail
     the candidate filters (archived, fork, no text, < 20 stars).
4. **Eligibility:** |K| ≥ 20, 3 ≤ |H| ≤ 100, and at least one derived interest.
5. **Derived interests.** Count the curated topics of the user's 100 most recent
   pre-cutoff stars, and keep the top 3 curated interests with ≥ 3 supporting repos.
6. **Pools.** Detour's real retrieval runs for those interests, and every GitHub response
   is recorded.

**Privacy.** Only public data is used. Logins are replaced by salted hashes in every file
except `users_private.json`, which stays in the gitignored data directory. Only
aggregate results are reported here.

## 4. Protocols (`detour.eval.offline`)

For each user, the profile is: derived interests; `K` as the known set (filtered and used
for unfamiliarity); and, for D3 only, the 30 most recent pre-cutoff stars as INTERESTED
exemplars, since a star is a weak positive signal.

- **end_to_end:** rank the retrieved pool as it is. Retrieval misses count against
  every variant. This is what a user would actually experience.
- **ranking_only:** inject `H` into the pool. This isolates ranking from retrieval.
- **retrieval ablation:** restrict the pool to `core`, `core+bridge`, or all families.

Aggregation: the mean over users with a 95% percentile bootstrap CI (2000 resamples,
seed 7), plus paired per-user differences against B1.

## 5. Metrics (`detour.eval.metrics`)

| Metric | Definition |
|---|---|
| recall@10, NDCG@10 | against `H` |
| recall_on_profile | against `H_profile` = items of `H` that pass the relevance-gate rule for the derived interests |
| recall_long_tail | against items of `H` below the pool's median star count |
| serendipity@10 | share of the list in `H_ser` = held-out items whose max cosine to `K` is < 0.6 (relevant, and not a near-duplicate of something known) |
| ILD | mean pairwise (1 − cos) within the list |
| interest coverage | fraction of the derived interests matched by ≥ 1 listed item |
| mean log10 stars | popularity of the list |
| unfamiliarity | mean (1 − max cos to `K`) |

## 6. Pre-registered success criteria

D3 is better than B1 if **all** of these hold:
1. Serendipity@K (human) is higher, and a blind preference test favours D3.
2. ILD and interest coverage are higher.
3. Precision@K (human) drops by no more than **10%** relative to B1.

Criteria 1 and 3 need human judgements, which have not been collected (§8). **The
pre-registered comparison therefore cannot be concluded yet.** The offline benchmark below
is supporting evidence only.

## 7. Results

### 7.1 Relevance calibration on a live pool (2026-10-05)
Data: one live retrieval on the linked machine for {Machine Learning, Distributed
Systems, Competitive Programming}, giving 176 candidates.

Weak labels: a candidate returned by interest *j*'s core query counts as on-target for
*j*; a candidate returned only by other interests' core queries counts as off-target for
*j*.

| | ML | DS | CP |
|---|---|---|---|
| mean cosine, on-target | 0.562 | 0.392 | 0.584 |
| mean cosine, off-target | 0.134 | 0.120 | 0.205 |
| AUC (on vs. off) | 0.997 | 0.957 | 0.996 |
| background similarity (pool-independent) | 0.142 | 0.161 | 0.212 |

- **The embeddings work:** AUC is 0.96–0.997.
- **Raw scales differ by interest.** This is the motivation for D-012.

Gate rule comparison:

| Gate | on-target pass | off-target pass |
|---|---|---|
| raw cosine ≥ 0.18 (old) | 0.97 | **0.38** |
| raw cosine ≥ 0.30 | 0.91 | 0.04 |
| margin over background ≥ 0.10 (**adopted**) | 0.94 | **0.06** |

Interest balance of the top 24 by relevance score:

| Score | ML / DS / CP |
|---|---|
| raw max cosine | 14 / 0 / 10 |
| margin | 17 / 1 / 6 |
| background z-score | 3 / 1 / 20 |
| per-interest percentile (**adopted**) | 8 / 8 / 8 |

The matched interest agreed with provenance for 98.6% of single-provenance candidates
under every option.

*Caveat:* one pool and weak labels, with no human judgements.

### 7.2 Live end-to-end runs (linked machine, real GitHub, unauthenticated)
| Interests | Candidates | Gate removed | Shown | Status | Cold total | Next request |
|---|---|---|---|---|---|---|
| DS, ML, CP (before calibration) | 176 | 6 | 8 | degraded (4 of 9 searches rate-limited) | 1982 ms | 178 ms |
| DS, ML | 182 | 40 | 8 | ok | 3571 ms | 270 ms |
| Backend, Databases, Observability | 284 | 39 | 8 | ok | 4710 ms | 361 ms |

**Overlap between the D3 and B1 lists on the same pool:**
- DS/ML/CP: 4/8 (engine before calibration); 2/8 when the same recording was replayed after calibration
- DS/ML: 2/8 (median stars D3 645 vs. B1 7,516)
- Backend/DB/Observability: 0/8 (median stars D3 85 vs. B1 2,315)

**After feedback,** in the second request for the DS/ML user:
- the two known repos were filtered (`already_known`), as were the 10 shown before;
- unfamiliarity became defined, and "Unlike the repositories you've marked as known"
  explanations appeared.

**Observation, not yet a judgement:** D3's lists are much less popular than B1's. In
the DS/ML run, most D3 picks had under 2,000 stars, while B1 showed etcd, keras and onnx.
Whether users see this as discovery or as obscurity is exactly what the human evaluation
must decide.

### 7.3 Offline benchmark
Generated by `python -m detour.eval.offline --sweep --markdown`. The machine-readable
aggregate is in `docs/eval/offline-results-2026-10-05.json`. Data: 93 users examined, 24
eligible (fork owners of 8 seed repos), 24 recorded pools.

Snapshot: 24 users (of 24 eligible), cutoff 2026-06-07, mean |H| = 15.7, mean pool = 243 candidates, k = 10, engine 1.0.0, embeddings wordllama-l2_supercat-256. Cells: mean [95% bootstrap CI] over users.

**Retrieval: share of held-out stars present in the candidate pool**

| families | pool recall |
|---|---|
| core | 0.005 [0.000, 0.014] |
| core+bridge | 0.009 [0.001, 0.019] |
| all | 0.019 [0.004, 0.038] |

**End-to-end (rank the retrieved pool as-is)**

| variant | recall@k | recall on-profile | recall long-tail | serendipity@k | ILD | coverage | log10 stars | unfamiliarity |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.004 | 0.018 | 0.000 | 0.008 | 0.662 | 0.944 | 4.959 | 0.468 |
| B1 | 0.003 | 0.004 | 0.000 | 0.000 | 0.596 | 1.000 | 3.608 | 0.414 |
| D1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.594 | 0.875 | 2.382 | 0.497 |
| D2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.665 | 1.000 | 3.410 | 0.473 |
| D3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.692 | 1.000 | 2.525 | 0.516 |

**Ranking-only (held-out stars injected into the pool)**

| variant | recall@k | recall on-profile | recall long-tail | serendipity@k | ILD | coverage | log10 stars | unfamiliarity |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.155 | 0.199 | 0.000 | 0.183 | 0.696 | 0.931 | 5.022 | 0.470 |
| B1 | 0.005 | 0.009 | 0.000 | 0.008 | 0.596 | 1.000 | 3.618 | 0.414 |
| D1 | 0.004 | 0.016 | 0.005 | 0.004 | 0.595 | 0.875 | 2.359 | 0.498 |
| D2 | 0.008 | 0.022 | 0.003 | 0.017 | 0.668 | 1.000 | 3.442 | 0.471 |
| D3 | 0.007 | 0.023 | 0.008 | 0.017 | 0.694 | 1.000 | 2.526 | 0.519 |

**Paired difference D3 minus B1 (per user)**

| metric | end-to-end | ranking-only |
|---|---|---|
| recall@k | -0.003 [-0.010, 0.000] | 0.002 [-0.008, 0.014] |
| recall on-profile | -0.004 [-0.012, 0.000] | 0.014 [-0.008, 0.050] |
| recall long-tail | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.020] |
| serendipity@k | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.021] |
| ILD | 0.097 [0.075, 0.119] | 0.098 [0.075, 0.122] |
| coverage | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| log10 stars | -1.084 [-1.285, -0.885] | -1.092 [-1.285, -0.898] |
| unfamiliarity | 0.102 [0.085, 0.120] | 0.105 [0.088, 0.122] |

**D3 sweep (means over users)**

| lambda | beta | ILD | coverage | log10 stars | unfamiliarity | recall (ranking-only) | recall on-profile |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.0 | 0.797 | 1.000 | 3.270 | 0.539 | 0.062 | 0.112 |
| 0.30 | 0.3 | 0.795 | 1.000 | 2.812 | 0.561 | 0.071 | 0.112 |
| 0.30 | 0.6 | 0.792 | 1.000 | 2.539 | 0.579 | 0.079 | 0.126 |
| 0.50 | 0.0 | 0.725 | 1.000 | 3.320 | 0.506 | 0.020 | 0.033 |
| 0.50 | 0.3 | 0.745 | 1.000 | 2.592 | 0.542 | 0.002 | 0.006 |
| 0.50 | 0.6 | 0.739 | 1.000 | 2.343 | 0.558 | 0.015 | 0.026 |
| 0.70 | 0.0 | 0.665 | 1.000 | 3.410 | 0.473 | 0.008 | 0.022 |
| 0.70 | 0.3 | 0.692 | 1.000 | 2.525 | 0.516 | 0.007 | 0.023 |
| 0.70 | 0.6 | 0.699 | 1.000 | 2.181 | 0.544 | 0.004 | 0.016 |
| 0.85 | 0.0 | 0.629 | 1.000 | 3.498 | 0.447 | 0.003 | 0.007 |
| 0.85 | 0.3 | 0.653 | 1.000 | 2.369 | 0.501 | 0.005 | 0.018 |
| 0.85 | 0.6 | 0.665 | 1.000 | 2.155 | 0.532 | 0.004 | 0.016 |
| 1.00 | 0.0 | 0.596 | 1.000 | 3.608 | 0.414 | 0.005 | 0.009 |
| 1.00 | 0.3 | 0.608 | 1.000 | 2.390 | 0.490 | 0.004 | 0.016 |
| 1.00 | 0.6 | 0.630 | 1.000 | 2.153 | 0.523 | 0.004 | 0.016 |


### 7.4 What the benchmark says (and does not say)
1. **Retrieval is the bottleneck.**
   - Only **0.5%** of later-starred repos appear in the pool with core queries alone,
     **0.9%** with bridges, and **1.9%** with bridges plus adjacent-topic exploration.
   - The exploratory families add +1.4 points (paired 95% CI [+0.1, +2.9]), with gains
     for 5 of 24 users. That is a real but small effect.
   - With ~250 candidates drawn from millions of repositories, end-to-end recall is
     essentially zero for **every** variant, including popularity. No ranking change can
     fix this; better candidate generation could. Personalized retrieval from the user's
     known repos is the obvious next step.
2. **The stars benchmark favours popularity.**
   - Held-out stars are more popular than the pools: mean log10 stars 3.77 vs. 3.33.
   - Only 42% of them are on-profile for the derived interests.
   - So when H is injected (ranking-only), **B0 wins clearly**: recall 0.155 vs. ≤ 0.008
     for every semantic variant. This was the anticipated bias (§9), and it is reported
     rather than tuned away.
3. **Relevance-only (B1) vs. Detour (D2/D3) on accuracy:** no measurable difference.
   Ranking-only recall, D3 − B1 = +0.002 [−0.008, +0.014]. Novelty and diversity did
   **not** cost measurable accuracy here, but the benchmark has little power to detect
   either a gain or a loss.
4. **Beyond-accuracy effects are large and consistent.** These are partly by
   construction (§1.4).
   - **Diversity:** D3 raises ILD by +0.097 [+0.075, +0.119], higher for 23 of 24 users.
   - **Popularity:** it recommends far less popular repositories (mean log10 stars 2.53
     vs. 3.61, about 12× fewer stars).
   - **Unfamiliarity:** higher, 0.516 vs. 0.414.
   - **Coverage:** interest coverage stays at 100%.
5. **The coverage floor matters.** D1 (novelty without MMR or the floor) drops interest
   coverage to 0.875 (−0.125 [−0.208, −0.056] vs. B1). D3, which adds the floor, keeps
   1.000.
6. **Sweep (hypothesis-generating only).**
   - Strong diversification (λ = 0.3) raises ranking-only recall to 0.06–0.08 vs. ≤ 0.02
     at λ ≥ 0.5.
   - Likely mechanism: held-out stars are often off-profile, and aggressive MMR reaches
     outside the dense on-topic cluster.
   - The defaults were **not** changed: that would tune on the test set, toward a
     benchmark known to reward off-profile picks. Whether users prefer λ = 0.3 lists is a
     question for the human study.


## 8. Human evaluation (tooling built, not run)
**Study mode:** open the frontend with `?study`, e.g. `https://<frontend>/?study`.
- Each request ranks one candidate pool with D3 and with B1 (the paired run is logged as
  `config.paired_with_run`).
- The two lists are shown as **List A / List B** in random order, without explanations,
  scores or panels.
- Participants rate every item:
  - "Interested" means they would explore it;
  - "Already know it" means it was familiar;
  - "Not for me" means it isn't relevant.
- Each rating is logged against the run whose list it was given in, so it can be
  attributed to a system.

**Analysis:** `DATABASE_URL=… python -m detour.eval.study` reports, per system, the rates
of each rating among shown items, and per session which list got more "Interested" marks,
with a Wilson 95% CI on D3's win share. The end-to-end test
`test_blind_study_feedback_is_attributed_to_each_system` checks this path.

**Protocol:** 10–20 participants × 2–3 sessions each, using their own interests. Apply
the criteria in §6. "Interested" without "Already know it" approximates the
relevant ∧ unknown ∧ worth-exploring definition of serendipity.

**Not run:** there were no participants during development. No human-preference
result exists.

## 9. Limitations (apply to every number above)
- **Small samples.** The offline benchmark covers few users; CIs are wide. Treat
  directions as hypotheses.
- **Stars are a weak, popularity-biased signal.** People star bookmarks, friends' repos
  and trending projects. *Not* starring is not dislike. The measured held-out stars were
  more popular than the candidate pools (§7.4), which structurally favours B0.
- **Derived interests are coarse:** 1–3 curated labels from topics. They are not what
  the user would type. Many later stars are off-profile for them.
- **Metadata is as of collection time, not as of the cutoff:** stars and topics may have
  changed since `t`. Retrieval also ran at collection time and can return repos created
  after `t`.
- **User sampling** via forks of 8 seed repos is not representative of GitHub users.
- **The relevance gate was calibrated on one live pool** (§7.1) and was not re-tuned on
  the benchmark, which avoids tuning on the test set.
- **Bridge query word order differs in 8 pools.** Those 8 pools were recorded before
  query planning became order-canonical, so their bridge queries list the two terms in the
  derived-interest order. GitHub's keyword search is AND-based, so results should be
  essentially the same.
- **No human labels exist.** Relevance, serendipity and preference as people experience
  them are unmeasured.
