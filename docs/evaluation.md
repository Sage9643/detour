# Evaluation

> Status: **protocol only (Phase 1)**. No experiments have been run. The *Results*
> section is empty on purpose. Numbers appear here only after they have been measured,
> together with the snapshot ID and configuration hash that produced them.

## 1. Principles

1. **Separate retrieval from ranking.** A ranker cannot surface what retrieval never
   fetched. Pool recall is measured separately from ranking metrics.
2. **Frozen snapshots.** Live GitHub data changes daily. Offline comparisons run on saved
   candidate pools: metadata, embeddings, and the query that produced each item.
3. **Same pool for all variants.** B0, B1, D1, D2 and D3 always rank the same snapshot.
4. **Beyond-accuracy metrics are partly circular.** MMR directly optimizes intra-list
   diversity, so "diversity went up" is true by construction. The non-trivial claim is
   that diversity and novelty rose *while relevance stayed within a pre-set margin*,
   **and** that humans preferred the result.
5. **Pre-register success criteria** (§6) before looking at results.
6. **Negative results are recorded**, not hidden.

## 2. Layers

| Layer | What it shows | Data |
|---|---|---|
| 1. Deterministic tests | algorithms behave as specified (MMR picks the orthogonal item, the gate excludes novel but irrelevant items, ALREADY_KNOW lowers unfamiliarity but not relevance) | hand-built vectors |
| 2. Offline: GitHub stars, temporal split | retrieval recall; ranking of later-starred repos | public starred lists |
| 3. Human evaluation | relevance, novelty and serendipity as people perceive them; preference between systems | blind study |
| 4. Live interaction log | feedback rates per variant (and interleaving, if traffic allows) | `run_candidates` + `feedback` |

## 3. Offline dataset: GitHub stars with a temporal split

**Construction**
1. Select 30–50 public GitHub users with sufficient starring activity and recognizable
   technical interests. Only their public starred lists are used.
2. For each user, choose a cutoff `t` (e.g. 6 months before the snapshot date):
   - stars **before** `t` → known set `K` (familiarity), plus derived interests
     (dominant topics, mapped to interest labels);
   - stars **after** `t` that pass the candidate filters → **held-out set** `H`
     (relevant *and* new to the user at time `t`).
3. Freeze: store the user's derived profile, `K`, `H`, the retrieved pool, and every
   embedding, under a snapshot ID.

**Retrieval evaluation:** pool recall = |H ∩ pool| / |H|, reported per query family.
This identifies whether candidate generation is the bottleneck.

**Ranking evaluation:** inject `H` into the pool (so retrieval failures don't contaminate
ranking metrics), then compute Recall@K and NDCG@K per variant. A "long-tail" variant
of the metric counts only items of `H` below the pool's median popularity.

**Limitations (stated in every report that uses this data):**
- A star is a weak, noisy preference signal: people star as a bookmark, to show support,
  or because something is trending. Not starring is **not** evidence of dislike.
- Starring is biased toward popular repositories, which favours B0 and penalizes
  novelty-seeking variants.
- Interests derived from earlier stars approximate, but are not the same as, what a user
  would type in.
- Users with public, active star lists are not representative of all users.
- The results are **directional evidence**, not ground truth.

## 4. Human evaluation

- **Blind side-by-side:** each participant gives their interests and sees two lists (B1
  and D3, in random order, with labels hidden), then picks the better one or "no
  difference".
- **Per-item labels:** *Relevant to me?* · *Already knew it?* · *Would explore it?*
- **Serendipity@K** = fraction of items that are relevant ∧ previously unknown ∧ worth
  exploring. This is the headline metric.
- Target: 10–20 participants × 2–3 sessions. Report the raw counts, the preference split
  with a binomial confidence interval, and an explicit small-sample caveat.
- Optional live alternative: **team-draft interleaving** of B1 and D3 in the real feed,
  crediting feedback to the system that contributed each item.

## 5. Metrics

| Category | Metric | Definition |
|---|---|---|
| Accuracy | Precision@K | fraction of the top-K labeled relevant (human) |
| | Recall@K, NDCG@K | against held-out stars `H` |
| Novelty | mean unfamiliarity | from logged components |
| | mean self-information | mean −log₂(popularity share within the pool) |
| | unknown rate | fraction labeled "didn't know it" (human) |
| Diversity | ILD | mean pairwise (1 − cos) within the list |
| | interest coverage | fraction of the user's interests matched by ≥1 item |
| | unique owners, topic entropy | per list |
| Serendipity | Serendipity@K | human; relevant ∧ unknown ∧ worth exploring |
| | long-tail recall | Recall@K restricted to below-median-popularity items of `H` |
| Coverage | catalog coverage | distinct repos recommended / distinct repos in pools |
| | repeat rate | fraction of items repeated across a user's consecutive sessions |
| Retrieval | pool recall, share of pool per family | §3 |
| System | p50/p95 per stage, GitHub calls per request, cache hit rate, error rate per dependency | p99 only from a load test, never from low-traffic production |

## 6. Pre-registered success criteria (to be confirmed before the first D3 run)

D3 is considered better than B1 if **all** of the following hold:
1. Serendipity@K (human) is higher, and the preference test favours D3.
2. ILD and interest coverage are higher.
3. Precision@K (human) drops by no more than **10%** relative to B1 ⚙️ (margin to be
   confirmed before running).

If any criterion fails, the outcome is recorded here as it is, followed by an
investigation.

## 7. Results

_No experiments run yet._
