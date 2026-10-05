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
| Backend, Databases, Observability | 284 | 39 | 8 | ok | 4710 ms | – |

**Overlap between the D3 and B1 lists on the same pool:** 2/8 in both runs where it was
measured.

**After feedback,** in the second request for the DS/ML user:
- the two known repos were filtered (`already_known`), as were the 10 shown before;
- unfamiliarity became defined, and "Unlike the repositories you've marked as known"
  explanations appeared.

**Observation, not yet a judgement:** D3's lists are much less popular than B1's. In
the DS/ML run, most D3 picks had under 2,000 stars, while B1 showed etcd, keras and onnx.
Whether users see this as discovery or as obscurity is exactly what the human evaluation
must decide.

### 7.3 Offline benchmark
RESULTS_TABLE_PLACEHOLDER

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
- **No human labels exist.** Relevance, serendipity and preference as people experience
  them are unmeasured.
