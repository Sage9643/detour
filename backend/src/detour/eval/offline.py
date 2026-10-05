"""Offline evaluation on the GitHub-stars temporal-split dataset (frozen snapshot).

    python -m detour.eval.offline --data DIR --out results.json [--k 10]

For every user in the snapshot:
  profile   derived interests (from pre-cutoff stars' topics), known set K = pre-cutoff
            stars (familiarity, filtered from recommendations); for D3, the 30 most recent
            pre-cutoff stars also act as INTERESTED exemplars (a star is a weak positive)
  truth     H = repos starred after the cutoff (relevant AND new to the user at cutoff)
  pool      the candidates Detour's real retrieval returned for those interests (recorded)

Two protocols, reported separately:
  end_to_end    rank the retrieved pool as-is; misses caused by retrieval count against
                every variant (what a user would actually experience)
  ranking_only  H is injected into the pool, isolating ranking quality from retrieval
And a retrieval ablation: the pool restricted to core / core+bridge / all query families.

Every variant ranks exactly the same pool for a given user. Results are aggregated per
user (mean with 95% bootstrap CI) and as paired differences against B1.
"""

import argparse
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from detour.enums import QueryFamily, RankerVariant
from detour.eval.metrics import (
    bootstrap_ci,
    interest_coverage,
    intra_list_diversity,
    mean_log_stars,
    mean_unfamiliarity,
    ndcg_at_k,
    owner_diversity,
    precision_at_k,
    recall_at_k,
    topic_entropy,
)
from detour.ranking.config import ENGINE_VERSION, RankerConfig
from detour.ranking.engine import PoolItem, meaningful_topics, rank
from detour.ranking.profile import Exemplar, InterestVector, UserProfile
from detour.representation.background import background_similarity
from detour.representation.embedder import Embedder, Vectors, default_embedder
from detour.representation.interests import GENERIC_TOPICS, resolve_interest
from detour.representation.store import embed_repos_uncached
from detour.retrieval.models import Candidate, RepoRecord

VARIANTS = list(RankerVariant)
N_POSITIVE_EXEMPLARS = 30
SERENDIPITY_MAX_KNOWN_SIM = 0.6
FAMILY_SETS = {
    "core": {QueryFamily.CORE},
    "core+bridge": {QueryFamily.CORE, QueryFamily.BRIDGE},
    "all": {QueryFamily.CORE, QueryFamily.BRIDGE, QueryFamily.ADJACENT},
}


def _records(rows: Sequence[dict[str, Any]]) -> list[RepoRecord]:
    return [RepoRecord.from_json(r) for r in rows]


def build_profile(ex: dict[str, Any], embedder: Embedder, vecs: dict[int, Vectors]) -> UserProfile:
    specs = [resolve_interest(lab) for lab in ex["interests"]]
    ivecs = embedder.embed([s.expansion for s in specs])
    interests = [
        InterestVector(
            label=s.label,
            vector=v,
            weight=1.0,
            topics=s.topics,
            background=background_similarity(embedder, s, v),
        )
        for s, v in zip(specs, ivecs, strict=True)
    ]
    known = _records(ex["known"])
    known_ids = {r.github_id for r in known}
    known_topics = {t for s in specs for t in s.topics}
    for r in known:
        known_topics.update(t for t in r.topics if t not in GENERIC_TOPICS)
    recent = sorted(ex["known"], key=lambda r: r["starred_at"], reverse=True)
    positives = [
        Exemplar(repo_id=r["github_id"], full_name=r["full_name"], vector=vecs[r["github_id"]])
        for r in recent[:N_POSITIVE_EXEMPLARS]
    ]
    return UserProfile(
        interests=interests,
        positives=positives,
        known_ids=known_ids,
        known_vectors=np.stack([vecs[i] for i in sorted(known_ids)]),
        known_topics=known_topics,
    )


def list_metrics(
    shown: Sequence[Any],
    heldout: set[int],
    serendipitous: set[int],
    interests: Sequence[str],
    known_vectors: Vectors,
    k: int,
) -> dict[str, float]:
    ids = [s.repo_id for s in shown]
    repos = [s.item.candidate.repo for s in shown]
    vecs = (
        np.stack([s.item.vector for s in shown]).astype(np.float32)
        if shown
        else np.zeros((0, 1), dtype=np.float32)
    )
    return {
        "recall": recall_at_k(ids, heldout, k),
        "precision": precision_at_k(ids, heldout, k),
        "ndcg": ndcg_at_k(ids, heldout, k),
        "serendipity": precision_at_k(ids, serendipitous, k),
        "ild": intra_list_diversity(vecs) if shown else float("nan"),
        "interest_coverage": interest_coverage([s.matched_interest for s in shown], interests),
        "owner_diversity": owner_diversity([r.owner_login for r in repos]),
        "topic_entropy": topic_entropy([meaningful_topics(r.topics) for r in repos]),
        "mean_log10_stars": mean_log_stars([r.stars for r in repos]),
        "unfamiliarity": mean_unfamiliarity(vecs, known_vectors) if shown else float("nan"),
        "list_len": float(len(shown)),
    }


def evaluate_user(
    ex: dict[str, Any],
    pool_doc: dict[str, Any],
    embedder: Embedder,
    cfg: RankerConfig,
    variants: Sequence[RankerVariant] = VARIANTS,
) -> dict[str, Any]:
    known = _records(ex["known"])
    heldout_recs = _records(ex["heldout"])
    heldout = {r.github_id for r in heldout_recs}
    candidates: dict[int, Candidate] = {}
    for c in pool_doc["candidates"]:
        rec = RepoRecord.from_json(c["repo"])
        cand = Candidate(repo=rec, families={QueryFamily(f) for f in c["families"]})
        cand.query_labels = list(c["query_labels"])
        cand.adjacent_topics = set(c["adjacent_topics"])
        cand.bridge_pairs = [tuple(p) for p in c["bridge_pairs"]]
        candidates[rec.github_id] = cand

    every = {r.github_id: r for r in [*known, *heldout_recs]}
    every.update({i: c.repo for i, c in candidates.items()})
    vecs = embed_repos_uncached(embedder, list(every.values()))
    profile = build_profile(ex, embedder, vecs)

    # Held-out items that are not near-duplicates of anything already known.
    serendipitous = {
        h
        for h in heldout
        if float((profile.known_vectors @ vecs[h]).max()) < SERENDIPITY_MAX_KNOWN_SIM
    }
    # Held-out items consistent with the derived interests (pass the relevance-gate rule).
    # Many stars are off-topic for a 1-3 interest profile; this subset asks the narrower
    # question "among on-profile items, does the ranker surface what was starred later?"
    U = np.stack([iv.vector for iv in profile.interests])
    bg = np.array([iv.background for iv in profile.interests])
    on_profile = {h for h in heldout if float(((U @ vecs[h]) - bg).max()) >= cfg.relevance_margin}
    # Long tail: held-out items below the pool's median popularity.
    pool_median_stars = float(np.median([c.repo.stars for c in candidates.values()]))
    long_tail = {h for h in heldout if every[h].stars < pool_median_stars}

    def items(ids: Sequence[int]) -> list[PoolItem]:
        out = []
        for i in ids:
            cand = candidates.get(i) or Candidate(repo=every[i])
            out.append(PoolItem(candidate=cand, vector=vecs[i]))
        return out

    result: dict[str, Any] = {
        "user": ex["user"],
        "interests": ex["interests"],
        "n_known": len(known),
        "n_heldout": len(heldout),
        "n_serendipitous": len(serendipitous),
        "n_on_profile": len(on_profile),
        "n_long_tail": len(long_tail),
        "heldout_mean_log10_stars": mean_log_stars([every[h].stars for h in heldout]),
        "pool_mean_log10_stars": mean_log_stars([c.repo.stars for c in candidates.values()]),
        "pool_size": len(candidates),
        "pool_recall": {},
        "end_to_end": {},
        "ranking_only": {},
        "ablation_end_to_end_D3": {},
    }
    for name, fams in FAMILY_SETS.items():
        ids = [i for i, c in candidates.items() if c.families & fams]
        result["pool_recall"][name] = len(set(ids) & heldout) / len(heldout)
        if name != "all" and RankerVariant.D3_FULL in variants:
            res = rank(items(ids), profile, RankerVariant.D3_FULL, cfg)
            result["ablation_end_to_end_D3"][name] = list_metrics(
                res.shown, heldout, serendipitous, ex["interests"], profile.known_vectors, cfg.k
            )

    pool_ids = sorted(candidates)
    injected_ids = sorted(set(pool_ids) | heldout)
    for variant in variants:
        for protocol, ids in (("end_to_end", pool_ids), ("ranking_only", injected_ids)):
            res = rank(items(ids), profile, variant, cfg)
            m = list_metrics(
                res.shown, heldout, serendipitous, ex["interests"], profile.known_vectors, cfg.k
            )
            top = [s.repo_id for s in res.shown]
            m["recall_on_profile"] = recall_at_k(top, on_profile, cfg.k)
            m["recall_long_tail"] = recall_at_k(top, long_tail, cfg.k)
            result[protocol][variant.value] = m
    if "D3" in result["end_to_end"]:
        result["ablation_end_to_end_D3"]["all"] = result["end_to_end"]["D3"]
    return result


def sweep(
    examples: list[dict[str, Any]],
    pools: dict[str, dict[str, Any]],
    embedder: Embedder,
    base: RankerConfig,
) -> list[dict[str, Any]]:
    """D3 relevance-diversity-novelty tradeoff over (lambda, beta), end-to-end protocol for
    list properties and ranking-only protocol for held-out recall."""
    grid = [(lam, beta) for lam in (0.3, 0.5, 0.7, 0.85, 1.0) for beta in (0.0, 0.3, 0.6)]
    rows = []
    for lam, beta in grid:
        cfg = base.with_overrides(mmr_lambda=lam, beta_novelty=beta)
        per_user = [
            evaluate_user(ex, pools[ex["user"]], embedder, cfg, variants=[RankerVariant.D3_FULL])
            for ex in examples
            if ex["user"] in pools
        ]
        row: dict[str, Any] = {"mmr_lambda": lam, "beta_novelty": beta}
        for protocol, metric in (
            ("end_to_end", "ild"),
            ("end_to_end", "interest_coverage"),
            ("end_to_end", "mean_log10_stars"),
            ("end_to_end", "unfamiliarity"),
            ("ranking_only", "recall"),
            ("ranking_only", "recall_on_profile"),
        ):
            row[f"{protocol}.{metric}"] = bootstrap_ci(
                [u[protocol]["D3"][metric] for u in per_user]
            )[0]
        rows.append(row)
    return rows


def aggregate(users: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(values: list[float]) -> dict[str, float]:
        mean, lo, hi = bootstrap_ci(values)
        n = sum(1 for v in values if not math.isnan(v))
        return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n}

    out: dict[str, Any] = {"pool_recall": {}, "ablation_end_to_end_D3": {}}
    for name in FAMILY_SETS:
        out["pool_recall"][name] = stat([u["pool_recall"][name] for u in users])
        out["ablation_end_to_end_D3"][name] = {
            m: stat([u["ablation_end_to_end_D3"][name][m] for u in users])
            for m in users[0]["ablation_end_to_end_D3"][name]
        }
    for protocol in ("end_to_end", "ranking_only"):
        out[protocol] = {}
        metrics = list(users[0][protocol]["B1"])
        for v in VARIANTS:
            out[protocol][v.value] = {
                m: stat([u[protocol][v.value][m] for u in users]) for m in metrics
            }
        out[protocol]["paired_vs_B1"] = {
            v.value: {
                m: stat(
                    [
                        u[protocol][v.value][m] - u[protocol]["B1"][m]
                        for u in users
                        if not (
                            math.isnan(u[protocol][v.value][m]) or math.isnan(u[protocol]["B1"][m])
                        )
                    ]
                )
                for m in metrics
            }
            for v in VARIANTS
            if v is not RankerVariant.B1_RELEVANCE
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sweep", action="store_true", help="also sweep D3's lambda and beta")
    args = ap.parse_args()
    data = Path(args.data)
    examples = json.loads((data / "examples.json").read_text(encoding="utf-8"))
    state = json.loads((data / "state.json").read_text(encoding="utf-8"))
    embedder = default_embedder()
    cfg = RankerConfig(k=args.k)
    pools = {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "pools").glob("*.json")
    }
    users = [
        evaluate_user(ex, pools[ex["user"]], embedder, cfg)
        for ex in examples
        if ex["user"] in pools
    ]
    if not users:
        raise SystemExit("no users with pools")
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": ENGINE_VERSION,
        "embedding_model": embedder.model_name,
        "ranker_config": cfg.to_dict(),
        "dataset": {
            "collected_at": state.get("collected_at"),
            "cutoff": state.get("cutoff"),
            "users": len(users),
            "examples_total": len(examples),
            "mean_heldout": float(np.mean([u["n_heldout"] for u in users])),
            "mean_pool": float(np.mean([u["pool_size"] for u in users])),
        },
        "aggregate": aggregate(users),
        "per_user": users,
    }
    if args.sweep:
        report["sweep_D3"] = sweep(examples, pools, embedder, cfg)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(render_summary(report))
    if args.sweep:
        print("\nD3 sweep: lambda beta | ILD coverage log10stars unfamiliarity | recall on-profile")
        sweep_rows: list[dict[str, float]] = report["sweep_D3"]
        for r in sweep_rows:
            vals = "  ".join(f"{v:.3f}" for k, v in r.items() if "." in k)
            print(f"  {r['mmr_lambda']:.2f} {r['beta_novelty']:.1f}  {vals}")


def render_summary(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    d = report["dataset"]
    lines = [
        f"users={d['users']} cutoff={d['cutoff']} mean |H|={d['mean_heldout']:.1f} "
        f"mean pool={d['mean_pool']:.0f}",
        "",
        "Pool recall of held-out stars by retrieval families:",
    ]
    for name, s in agg["pool_recall"].items():
        lines.append(f"  {name:12} {s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
    cols = [
        "recall",
        "ndcg",
        "recall_on_profile",
        "recall_long_tail",
        "serendipity",
        "ild",
        "interest_coverage",
        "mean_log10_stars",
        "unfamiliarity",
    ]
    for protocol in ("end_to_end", "ranking_only"):
        lines += ["", f"{protocol} (mean over users, @k={report['ranker_config']['k']}):"]
        lines.append("  variant " + " ".join(f"{c[:12]:>13}" for c in cols))
        for v, ms in agg[protocol].items():
            if v == "paired_vs_B1":
                continue
            lines.append(f"  {v:7} " + " ".join(f"{ms[c]['mean']:13.3f}" for c in cols))
    lines += ["", "paired difference vs B1, end_to_end (mean [95% CI]):"]
    for v, ms in agg["end_to_end"]["paired_vs_B1"].items():
        parts = [
            f"{c}={ms[c]['mean']:+.3f} [{ms[c]['ci_low']:+.3f},{ms[c]['ci_high']:+.3f}]"
            for c in ("recall", "ndcg", "ild", "interest_coverage")
        ]
        lines.append(f"  {v}: " + "  ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
