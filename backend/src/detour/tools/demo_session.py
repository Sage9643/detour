"""Drive the full product loop against a running Detour API (local or deployed).

    python -m detour.tools.demo_session --api http://localhost:8000 \
        --interest "Distributed Systems" --interest "Machine Learning" --out report.json

Steps: create user -> set interests -> recommend (D3, compared with B1 on the same pool)
-> feedback on the top items (ALREADY_KNOW, INTERESTED, NOT_INTERESTED) -> recommend again.
Prints a readable summary and optionally writes the raw responses to a JSON report.
Doubles as a post-deployment smoke test: exits non-zero if any step fails.
"""

import argparse
import json
import sys
from typing import Any

import httpx


def _check(resp: httpx.Response, expected: int) -> Any:
    if resp.status_code != expected:
        print(f"FAILED {resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()


def _print_run(title: str, run: dict[str, Any]) -> None:
    st = run["stats"]
    print(f"\n=== {title}: variant={run['variant']} status={run['status']} ===")
    print(
        f"candidates={st['candidates']} eligible={st['eligible']} shown={st['shown']} "
        f"by_family={st['by_family']} github_calls={st['github_calls']} "
        f"embedding_cache={st['embedding_cache']}"
    )
    print(f"filtered={st['filtered']}")
    print(f"timings_ms={st['timings_ms']}")
    for q in st["queries"]:
        print(f"  [{q['source']:6}] {q['family']:8} {q['result_count']:3} results  {q['label']}")
    for w in run["warnings"]:
        print(f"  WARNING: {w}")
    for it in run["items"]:
        r, s = it["repo"], it["scores"]
        print(
            f"{it['position'] + 1:2}. {r['full_name']}  ({r['stars']}★, {r['language']})  "
            f"rel={s['relevance_raw']} nov={s['novelty']} fam={','.join(it['query_families'])}"
        )
        for line in it["explanation"]:
            print(f"      - {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--interest", action="append", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    report: dict[str, Any] = {}
    with httpx.Client(base_url=args.api, timeout=120) as c:
        _check(c.get("/health/ready"), 200)
        user = _check(c.post("/users", json={}), 201)
        uid = user["id"]
        report["interests"] = _check(
            c.put(
                f"/users/{uid}/interests", json={"interests": [{"label": i} for i in args.interest]}
            ),
            200,
        )
        first = _check(
            c.post(f"/users/{uid}/recommendations", json={"k": args.k, "compare_variant": "B1"}),
            200,
        )
        report["first"] = first
        _print_run("Run 1 (Detour D3)", first["run"])
        _print_run("Run 1 (baseline B1, same pool)", first["comparison"])
        d3 = {it["repo"]["github_id"] for it in first["run"]["items"]}
        b1 = {it["repo"]["github_id"] for it in first["comparison"]["items"]}
        print(f"\nOverlap D3 vs B1: {len(d3 & b1)}/{len(d3)}")

        items = first["run"]["items"]
        actions = list(
            zip(items[:3], ["ALREADY_KNOW", "INTERESTED", "NOT_INTERESTED"], strict=False)
        )
        report["feedback"] = []
        for it, ftype in actions:
            fb = _check(
                c.post(
                    f"/users/{uid}/feedback",
                    json={
                        "repo_id": it["repo"]["github_id"],
                        "type": ftype,
                        "run_id": first["run"]["run_id"],
                    },
                ),
                201,
            )
            report["feedback"].append({"repo": it["repo"]["full_name"], "type": ftype})
            print(f"feedback {ftype:15} {it['repo']['full_name']}")
        print(f"profile after feedback: {json.dumps(fb['profile'], indent=None)[:600]}")

        second = _check(c.post(f"/users/{uid}/recommendations", json={"k": args.k}), 200)
        report["second"] = second
        _print_run("Run 2 (after feedback)", second["run"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print("\nOK")


if __name__ == "__main__":
    main()
