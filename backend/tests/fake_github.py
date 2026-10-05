"""A small in-process imitation of GitHub's search API for tests.

It serves a fixed catalog of synthetic repositories, matching query keywords against
name/description/topics roughly the way GitHub does. Test-only: real runs use the live
API or recordings of it (see detour.retrieval.cassette).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

_CATALOG_SPEC = [
    # id, owner, name, description, topics, stars
    (
        101,
        "hashicorp",
        "raft",
        "Golang implementation of the Raft consensus protocol",
        ["raft", "consensus", "distributed-systems"],
        8000,
    ),
    (
        102,
        "etcd-io",
        "etcd",
        "Distributed reliable key-value store for the most critical data of a distributed system",
        ["distributed-systems", "raft", "key-value-store", "consensus"],
        47000,
    ),
    (
        103,
        "tikv",
        "tikv",
        "Distributed transactional key-value database",
        ["distributed-systems", "database", "raft", "key-value-store"],
        15000,
    ),
    (
        104,
        "smallco",
        "minipaxos",
        "Minimal Paxos consensus implementation for learning distributed systems",
        ["paxos", "consensus", "distributed-systems"],
        120,
    ),
    (
        105,
        "jepsen-io",
        "maelstrom",
        "A workbench for writing toy implementations of distributed systems",
        ["distributed-systems", "testing", "fault-injection"],
        3000,
    ),
    (
        106,
        "open-telemetry",
        "opentelemetry-go",
        "OpenTelemetry Go API and SDK for distributed tracing and metrics",
        ["opentelemetry", "tracing", "observability", "distributed-systems"],
        5000,
    ),
    (
        107,
        "jaegertracing",
        "jaeger",
        "Distributed tracing platform for monitoring microservices",
        ["tracing", "observability", "distributed-systems", "opentelemetry"],
        20000,
    ),
    (
        201,
        "pytorch",
        "pytorch",
        "Tensors and dynamic neural networks in Python with strong GPU acceleration",
        ["machine-learning", "deep-learning", "neural-network", "pytorch"],
        80000,
    ),
    (
        202,
        "scikit-learn",
        "scikit-learn",
        "Machine learning in Python",
        ["machine-learning", "scikit-learn", "data-science"],
        60000,
    ),
    (
        203,
        "tinyml",
        "micrograd-lite",
        "A tiny autograd engine for learning how neural networks train",
        ["machine-learning", "neural-network", "autograd"],
        90,
    ),
    (
        204,
        "horovod",
        "horovod",
        "Distributed training framework for machine learning with TensorFlow and PyTorch",
        ["machine-learning", "distributed-training", "deep-learning", "distributed-systems"],
        14000,
    ),
    (
        205,
        "ray-project",
        "ray",
        "Distributed computing framework for scaling machine learning workloads",
        ["machine-learning", "distributed-systems", "distributed-training", "python"],
        33000,
    ),
    (
        206,
        "mlflow",
        "mlflow",
        "Open source platform for the machine learning lifecycle and experiment tracking",
        ["machine-learning", "mlops", "experiment-tracking"],
        18000,
    ),
    (
        207,
        "federated",
        "flower",
        "A friendly federated learning framework for machine learning research",
        ["federated-learning", "machine-learning", "distributed-training"],
        4500,
    ),
    (
        301,
        "kth",
        "kactl",
        "KTH algorithm competition template library for competitive programming",
        ["competitive-programming", "algorithms", "icpc"],
        2500,
    ),
    (
        302,
        "cpdude",
        "cp-snippets",
        "Competitive programming snippets: segment trees, graphs and dynamic programming",
        ["competitive-programming", "algorithms", "data-structures"],
        300,
    ),
    (
        303,
        "atcoder",
        "ac-library",
        "AtCoder library of algorithms and data structures for competitive programming",
        ["competitive-programming", "algorithms"],
        3500,
    ),
    (
        401,
        "facebook",
        "react",
        "The library for web and native user interfaces",
        ["react", "frontend", "javascript", "ui"],
        220000,
    ),
    (
        402,
        "uiorg",
        "fancy-buttons",
        "Accessible react component library for buttons",
        ["react", "ui-components", "frontend"],
        500,
    ),
    (
        501,
        "forkedco",
        "raft-fork",
        "A fork of a Raft consensus implementation",
        ["raft", "consensus", "distributed-systems"],
        50,
    ),
    (
        502,
        "oldco",
        "archived-db",
        "Archived distributed database experiment",
        ["database", "distributed-systems"],
        900,
    ),
]


def _item(spec: tuple[Any, ...]) -> dict[str, Any]:
    gid, owner, name, desc, topics, stars = spec
    return {
        "id": gid,
        "full_name": f"{owner}/{name}",
        "name": name,
        "owner": {"login": owner},
        "description": desc,
        "topics": topics,
        "language": "Go" if gid < 200 else "Python",
        "stargazers_count": stars,
        "forks_count": stars // 10,
        "archived": gid == 502,
        "fork": gid == 501,
        "license": {"spdx_id": "MIT"},
        "html_url": f"https://github.com/{owner}/{name}",
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2026-09-01T00:00:00Z",
    }


CATALOG: list[dict[str, Any]] = [_item(s) for s in _CATALOG_SPEC]


@dataclass
class FakeGitHub:
    """Callable handler for httpx.MockTransport with switchable failure modes."""

    mode: str = "ok"  # ok | down | rate_limited | flaky (first call per path 500s)
    calls: list[str] = field(default_factory=list)
    search_remaining: int = 30
    _flaky_seen: set[str] = field(default_factory=set)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = urlsplit(str(request.url))
        self.calls.append(f"{request.method} {url.path}?{url.query}")
        if self.mode == "down":
            return httpx.Response(502, text="bad gateway")
        if self.mode == "rate_limited":
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "4102444800"},
                json={"message": "API rate limit exceeded"},
            )
        if self.mode == "flaky" and url.query not in self._flaky_seen:
            self._flaky_seen.add(url.query)
            return httpx.Response(500, text="oops")
        if url.path == "/search/repositories":
            q = parse_qs(url.query)["q"][0]
            per_page = int(parse_qs(url.query).get("per_page", ["30"])[0])
            items = [it for it in CATALOG if _matches(q, it)][:per_page]
            self.search_remaining -= 1
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-remaining": str(self.search_remaining),
                    "x-ratelimit-limit": "30",
                    "x-ratelimit-reset": "4102444800",
                },
                content=json.dumps({"total_count": len(items), "items": items}),
            )
        return httpx.Response(404, json={"message": "Not Found"})


def _matches(q: str, item: dict[str, Any]) -> bool:
    words, topic = [], None
    for tok in q.split():
        if tok.startswith("topic:"):
            topic = tok.split(":", 1)[1]
        elif ":" not in tok:
            words.append(tok.lower())
    hay = (
        " ".join([item["name"], item["description"] or "", " ".join(item["topics"])])
        .lower()
        .replace("-", " ")
    )
    if topic and topic not in item["topics"]:
        return False
    # GitHub-like: every keyword must appear (prefix match is close enough here).
    return all(w.rstrip("s") in hay for w in words)


Handler = Callable[[httpx.Request], httpx.Response]
