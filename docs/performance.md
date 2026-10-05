# Performance

Everything here was measured; the environment is stated for each number. Nothing was
tuned for speed yet: no measured bottleneck justifies it (see "What would change this").

## Cold request (live GitHub, nothing cached)
One observation, recorded on the development sandbox of the linked machine (2 vCPU,
local Postgres, **unauthenticated** GitHub). Interests: Distributed Systems, Machine
Learning, Competitive Programming.

| Stage | ms |
|---|---|
| profile | 8.8 |
| retrieval (5 live searches; 4 more were rate-limited) | 1705.6 |
| embedding (176 misses) | 174.0 |
| ranking | 5.3 |
| logging (run + 176 candidate rows) | 86.1 |
| **total** | **1982.3** |

The same user's next request, with all searches cached, took **178 ms**: retrieval
29 ms, embedding 23 ms (176 cache hits).

**Bottleneck: GitHub.** In latency terms it dominates the cold path. In budget terms,
unauthenticated search allows 10 calls/min per IP, and a single cold request with 3
interests plans 8 searches, so 4 were rate-limited in that run. A `GITHUB_TOKEN` raises
the limit to 30/min.

## Warm requests (all searches cached)
Cloud development container, local Postgres 16. 29 requests (6 users × 5, excluding the
very first one), recorded GitHub responses (replay mode, so failed queries return
immediately), K = 10.

| | p50 | p95 |
|---|---|---|
| client end-to-end | 113 ms | 141 ms |
| server total | 107 ms | 130 ms |
| profile | 4.7 | 9.0 |
| retrieval (cache lookups) | 16.8 | 25.4 |
| embedding (cache hits) | 14.0 | 15.8 |
| ranking (filters → MMR, ~170 candidates) | 3.4 | 5.0 |
| **logging (~170 run_candidates rows)** | **64.3** | **87.8** |

p99 is not reported: 29 samples cannot support it. A load test with a few hundred
requests is the way to get it.

## Memory
API process RSS after startup, with the embedding model loaded: **~170 MB**. That fits
Render's 512 MB free tier.

## What would change this
- **Logging is the largest warm-path cost.** Every scored candidate is written on purpose
  (D-006/D-014). If it ever matters: use COPY instead of a multi-row INSERT, or write
  candidates after the response. Measure first.
- **Cold-path retrieval** would benefit from a token (more budget) and from pre-warming
  the cache for popular curated interests. Redis would not help: GitHub is the bottleneck,
  not cache-read latency (17 ms p50 from Postgres).
- **Render free tier** sleeps idle services. The first request after idling includes a
  cold start, which is not measured here. Measure it after deploying.
