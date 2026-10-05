"""Interest expansion: short labels -> text worth embedding + GitHub search hints.

Why: labels like "ML" or "CP" embed poorly and search poorly. A small curated dictionary
expands common interests into a descriptive sentence (embedded), a keyword query
(searched), a short bridge term (combined with other interests), and GitHub topic slugs
(used for topical novelty and adjacent-topic exploration).

Deterministic and inspectable: no LLM involved. Unknown labels fall back to the raw text.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterestSpec:
    label: str
    expansion: str  # text that gets embedded
    core_terms: str  # GitHub keyword query (all words must match)
    bridge_term: str  # short term combined with another interest's bridge term
    topics: tuple[str, ...]  # GitHub topic slugs this interest "covers"
    curated: bool  # False when produced by the raw-text fallback


def _spec(
    label: str,
    aliases: list[str],
    expansion: str,
    core_terms: str,
    bridge_term: str,
    topics: list[str],
) -> tuple[list[str], InterestSpec]:
    return (
        [label.lower(), *aliases],
        InterestSpec(
            label=label,
            expansion=f"{label.lower()}: {expansion}",
            core_terms=core_terms,
            bridge_term=bridge_term,
            topics=tuple(topics),
            curated=True,
        ),
    )


_ENTRIES = [
    _spec(
        "Machine Learning",
        ["ml", "machine-learning"],
        "neural networks, model training, deep learning, classification, prediction, scikit-learn, pytorch",
        "machine learning",
        "machine learning",
        ["machine-learning", "deep-learning", "ml", "neural-network", "scikit-learn", "pytorch"],
    ),
    _spec(
        "Deep Learning",
        ["dl", "deep-learning", "neural networks"],
        "neural networks, transformers, pytorch, tensorflow, gpu training, model architectures",
        "deep learning",
        "deep learning",
        ["deep-learning", "neural-network", "pytorch", "tensorflow", "transformers"],
    ),
    _spec(
        "Natural Language Processing",
        ["nlp", "natural-language-processing"],
        "text processing, language models, tokenization, embeddings, transformers, text classification",
        "natural language processing",
        "nlp",
        ["nlp", "natural-language-processing", "transformers", "text-classification"],
    ),
    _spec(
        "Computer Vision",
        ["cv", "computer-vision", "image processing"],
        "image recognition, object detection, segmentation, opencv, convolutional neural networks",
        "computer vision",
        "vision",
        ["computer-vision", "object-detection", "opencv", "image-processing", "segmentation"],
    ),
    _spec(
        "Large Language Models",
        ["llm", "llms", "generative ai", "genai"],
        "LLM inference, fine-tuning, retrieval augmented generation, prompt engineering, agents",
        "llm",
        "llm",
        ["llm", "large-language-models", "rag", "generative-ai", "langchain", "openai"],
    ),
    _spec(
        "Reinforcement Learning",
        ["rl", "reinforcement-learning"],
        "agents, policy optimization, reward, environments, gym, simulation",
        "reinforcement learning",
        "reinforcement learning",
        ["reinforcement-learning", "deep-reinforcement-learning", "gym", "rl"],
    ),
    _spec(
        "Recommender Systems",
        ["recsys", "recommendation systems", "recommender"],
        "recommendation algorithms, collaborative filtering, ranking, personalization, retrieval",
        "recommender system",
        "recommendation",
        ["recommender-system", "recommendation-system", "collaborative-filtering", "recsys"],
    ),
    _spec(
        "MLOps",
        ["ml ops", "machine learning operations"],
        "model deployment, model serving, experiment tracking, feature stores, ml pipelines",
        "mlops",
        "mlops",
        ["mlops", "model-serving", "ml-pipelines", "experiment-tracking"],
    ),
    _spec(
        "Data Engineering",
        ["etl", "data pipelines", "data-engineering"],
        "ETL pipelines, data warehouses, stream processing, orchestration, spark, airflow",
        "data engineering",
        "data pipeline",
        ["data-engineering", "etl", "apache-spark", "airflow", "data-pipeline", "streaming"],
    ),
    _spec(
        "Data Science",
        ["data analysis", "data-science", "analytics"],
        "data analysis, statistics, visualization, pandas, notebooks, exploratory analysis",
        "data science",
        "data science",
        ["data-science", "data-analysis", "pandas", "data-visualization", "jupyter-notebook"],
    ),
    _spec(
        "Backend",
        ["backend development", "web backend", "backend engineering", "server-side"],
        "web servers, REST APIs, microservices, authentication, web frameworks, server-side development",
        "backend framework",
        "backend",
        ["backend", "rest-api", "api", "web-framework", "microservices", "http-server"],
    ),
    _spec(
        "Distributed Systems",
        ["distributed computing", "distributed-systems"],
        "consensus, replication, fault tolerance, distributed databases, raft, paxos, RPC",
        "distributed systems",
        "distributed",
        ["distributed-systems", "consensus", "raft", "distributed-computing", "replication"],
    ),
    _spec(
        "Databases",
        ["database", "dbms", "database systems", "databases internals"],
        "database engines, storage engines, query processing, SQL, indexing, transactions",
        "database engine",
        "database",
        ["database", "sql", "storage-engine", "key-value-store", "dbms", "postgresql"],
    ),
    _spec(
        "Competitive Programming",
        ["cp", "competitive-programming", "competitive coding"],
        "algorithms, data structures, contest problems, codeforces, leetcode, icpc",
        "competitive programming",
        "algorithms",
        ["competitive-programming", "codeforces", "leetcode", "algorithms", "icpc"],
    ),
    _spec(
        "Algorithms",
        ["algorithms and data structures", "dsa", "data structures"],
        "algorithms, data structures, graph algorithms, dynamic programming, complexity",
        "algorithms data structures",
        "algorithms",
        ["algorithms", "data-structures", "graph-algorithms", "dynamic-programming"],
    ),
    _spec(
        "DevOps",
        ["devops", "ci/cd", "infrastructure"],
        "continuous integration, deployment automation, infrastructure as code, containers",
        "devops",
        "devops",
        ["devops", "ci-cd", "infrastructure-as-code", "terraform", "ansible", "docker"],
    ),
    _spec(
        "Cloud Native",
        ["kubernetes", "k8s", "cloud-native"],
        "kubernetes, containers, service mesh, operators, cloud infrastructure",
        "kubernetes",
        "kubernetes",
        ["kubernetes", "cloud-native", "k8s", "containers", "service-mesh", "helm"],
    ),
    _spec(
        "Observability",
        ["monitoring", "tracing", "logging"],
        "metrics, distributed tracing, logging, monitoring, telemetry, alerting",
        "observability",
        "observability",
        ["observability", "monitoring", "opentelemetry", "tracing", "metrics", "prometheus"],
    ),
    _spec(
        "Security",
        ["cybersecurity", "infosec", "application security"],
        "vulnerability scanning, cryptography, penetration testing, authentication, threat detection",
        "security",
        "security",
        ["security", "cybersecurity", "cryptography", "pentesting", "vulnerability"],
    ),
    _spec(
        "Systems Programming",
        ["systems", "low-level programming"],
        "memory management, concurrency, performance, operating system interfaces, rust, c",
        "systems programming",
        "systems",
        ["systems-programming", "low-level", "concurrency", "performance"],
    ),
    _spec(
        "Compilers",
        ["compiler", "programming languages", "pl"],
        "parsers, interpreters, code generation, type systems, llvm, language design",
        "compiler",
        "compiler",
        ["compiler", "interpreter", "programming-language", "llvm", "parser"],
    ),
    _spec(
        "Operating Systems",
        ["os", "kernel", "operating-systems"],
        "kernels, schedulers, file systems, virtual memory, device drivers",
        "operating system kernel",
        "kernel",
        ["operating-system", "kernel", "os", "osdev", "filesystem"],
    ),
    _spec(
        "Networking",
        ["computer networks", "network programming"],
        "network protocols, TCP/IP, HTTP, proxies, load balancing, packet processing",
        "networking",
        "network",
        ["networking", "network", "proxy", "tcp", "load-balancer", "protocol"],
    ),
    _spec(
        "Frontend",
        ["frontend development", "web frontend", "ui development"],
        "user interfaces, react, components, css, browser applications, single page apps",
        "frontend",
        "frontend",
        ["frontend", "react", "vue", "css", "ui-components", "javascript"],
    ),
    _spec(
        "Mobile Development",
        ["mobile", "android", "ios"],
        "android apps, ios apps, cross-platform mobile, flutter, react native",
        "mobile app",
        "mobile",
        ["android", "ios", "flutter", "react-native", "mobile"],
    ),
    _spec(
        "Game Development",
        ["gamedev", "game dev", "games"],
        "game engines, rendering, physics, game design, unity, godot",
        "game engine",
        "game",
        ["game-engine", "gamedev", "game-development", "godot", "unity"],
    ),
    _spec(
        "Computer Graphics",
        ["graphics", "rendering"],
        "rendering, ray tracing, shaders, gpu programming, opengl, vulkan",
        "rendering graphics",
        "rendering",
        ["graphics", "rendering", "ray-tracing", "opengl", "vulkan", "shaders"],
    ),
    _spec(
        "Embedded Systems",
        ["embedded", "iot", "microcontrollers", "firmware"],
        "microcontrollers, firmware, real-time systems, arduino, esp32, iot devices",
        "embedded firmware",
        "embedded",
        ["embedded", "iot", "firmware", "arduino", "esp32", "rtos"],
    ),
    _spec(
        "Blockchain",
        ["web3", "crypto", "smart contracts"],
        "smart contracts, ethereum, decentralized applications, consensus, cryptography",
        "blockchain",
        "blockchain",
        ["blockchain", "ethereum", "web3", "smart-contracts", "solidity"],
    ),
    _spec(
        "Functional Programming",
        ["fp", "functional-programming", "haskell"],
        "immutability, type systems, haskell, ocaml, lambda calculus, category theory",
        "functional programming",
        "functional",
        ["functional-programming", "haskell", "ocaml", "elixir", "lambda-calculus"],
    ),
    _spec(
        "Developer Tools",
        ["devtools", "cli tools", "developer productivity", "tooling"],
        "command line tools, editors, build systems, linters, developer productivity",
        "developer tools",
        "cli",
        ["developer-tools", "cli", "devtools", "terminal", "build-tool"],
    ),
    _spec(
        "Robotics",
        ["robots", "ros"],
        "robot control, motion planning, ROS, perception, SLAM, simulation",
        "robotics",
        "robotics",
        ["robotics", "ros", "slam", "motion-planning"],
    ),
    _spec(
        "Quantum Computing",
        ["quantum"],
        "quantum circuits, qubits, quantum algorithms, qiskit, quantum simulation",
        "quantum computing",
        "quantum",
        ["quantum-computing", "quantum", "qiskit", "quantum-algorithms"],
    ),
    _spec(
        "Testing",
        ["software testing", "test automation", "qa"],
        "unit testing, integration testing, property-based testing, fuzzing, test frameworks",
        "testing framework",
        "testing",
        ["testing", "test-automation", "fuzzing", "unit-testing"],
    ),
]

_BY_ALIAS: dict[str, InterestSpec] = {}
for _aliases, _entry in _ENTRIES:
    for _alias in _aliases:
        _BY_ALIAS[_alias.lower()] = _entry

CURATED_LABELS: tuple[str, ...] = tuple(entry.label for _, entry in _ENTRIES)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def resolve_interest(label: str, expansion_override: str | None = None) -> InterestSpec:
    """Resolve a user-entered label to its spec (curated, or raw-text fallback)."""
    clean = normalize_label(label)
    entry = _BY_ALIAS.get(clean.lower())
    if entry is None:
        entry = InterestSpec(
            label=clean,
            expansion=clean,
            core_terms=clean,
            bridge_term=clean,
            topics=(_slug(clean),),
            curated=False,
        )
    if expansion_override:
        entry = InterestSpec(
            label=entry.label,
            expansion=f"{entry.label.lower()}: {expansion_override.strip()}",
            core_terms=entry.core_terms,
            bridge_term=entry.bridge_term,
            topics=entry.topics,
            curated=entry.curated,
        )
    return entry


# Topics too generic to count as an "exploratory direction" for adjacent-topic retrieval
# or as evidence of topical novelty (languages, list/meta tags, platforms).
GENERIC_TOPICS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "go",
        "golang",
        "rust",
        "java",
        "javascript",
        "typescript",
        "cpp",
        "c-plus-plus",
        "c",
        "csharp",
        "ruby",
        "php",
        "kotlin",
        "swift",
        "scala",
        "haskell",
        "elixir",
        "lua",
        "r",
        "julia",
        "zig",
        "nodejs",
        "node",
        "deno",
        "hacktoberfest",
        "hacktoberfest2020",
        "hacktoberfest2021",
        "hacktoberfest2022",
        "hacktoberfest2023",
        "awesome",
        "awesome-list",
        "list",
        "open-source",
        "opensource",
        "library",
        "framework",
        "tutorial",
        "tutorials",
        "learning",
        "education",
        "examples",
        "linux",
        "macos",
        "windows",
        "cross-platform",
        "github",
        "api",
        "sdk",
        "cli",
        "hacktoberfest-accepted",
        "beginner-friendly",
        "good-first-issue",
        "self-hosted",
        "free",
        "tool",
        "tools",
        "utility",
        "utilities",
        "app",
        "application",
    }
)
