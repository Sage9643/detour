"""What text represents a repository semantically.

Embedded: name (split into words), description, topics.
NOT embedded: language (it would make every repo in a language similar to every other),
stars/forks/dates (metadata features, used separately), README (cost + noise; deferred).
"""

import hashlib
import re

from detour.retrieval.models import RepoRecord

_MAX_DESCRIPTION_CHARS = 400


def split_name(name: str) -> str:
    """`raft-rs` -> "raft rs"; `FastAPI` -> "fast api"; `etcd_io` -> "etcd io"."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[-_./]+", " ", spaced).lower().strip()


def repo_text(repo: RepoRecord) -> str:
    parts = [split_name(repo.name)]
    if repo.description:
        parts.append(repo.description.strip()[:_MAX_DESCRIPTION_CHARS])
    if repo.topics:
        parts.append("topics: " + ", ".join(t.replace("-", " ") for t in repo.topics))
    return ". ".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
