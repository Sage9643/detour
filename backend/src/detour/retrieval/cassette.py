"""Record/replay of GitHub HTTP traffic.

Why this exists:
- Reproducibility: an evaluation or demo run can be replayed byte-for-byte later, even
  though live GitHub results change daily.
- Environments without GitHub access (e.g. a sandboxed CI or dev box) can still exercise
  the full pipeline on *real* recorded responses rather than invented fixtures.

Requests are keyed by method + URL path + sorted query parameters. Authorization headers
are never part of the key and never written to disk.
"""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx

# Response headers worth keeping (rate-limit bookkeeping); everything else is dropped.
_KEPT_HEADERS = ("content-type", "x-ratelimit-remaining", "x-ratelimit-reset", "x-ratelimit-limit")


def request_key(request: httpx.Request) -> str:
    url = urlsplit(str(request.url))
    params = sorted(parse_qsl(url.query, keep_blank_values=True))
    canonical = json.dumps([request.method, url.path, params])
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def env_http_transport() -> httpx.HTTPTransport:
    """A real transport that honours HTTPS_PROXY.

    httpx only applies environment proxies when it builds the transport itself; a custom
    transport (like the recorder below) must be given the proxy explicitly.
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return httpx.HTTPTransport(proxy=proxy) if proxy else httpx.HTTPTransport()


class RecordingTransport(httpx.BaseTransport):
    """Pass requests to a real transport and save every response."""

    def __init__(self, directory: Path, inner: httpx.BaseTransport | None = None) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._inner = inner or env_http_transport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        body = response.read()
        record = {
            "method": request.method,
            "url": str(request.url),
            "status": response.status_code,
            "headers": {k: v for k, v in response.headers.items() if k.lower() in _KEPT_HEADERS},
            "body": body.decode("utf-8", errors="replace"),
        }
        path = self._dir / f"{request_key(request)}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        # `body` is already decoded, so encoding/length headers must not be passed on.
        headers = [
            (k, v)
            for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(
            status_code=response.status_code,
            headers=headers,
            content=body,
            request=request,
        )

    def close(self) -> None:
        self._inner.close()


class ReplayOrRecordTransport(httpx.BaseTransport):
    """Serve successful recorded responses; fetch and record anything else.

    Used by resumable offline collection: re-running a step never spends API budget on a
    request that already succeeded. Error responses (e.g. 403 rate limit) are re-fetched.
    """

    def __init__(self, directory: Path, inner: httpx.BaseTransport | None = None) -> None:
        self._replay = ReplayTransport(directory)
        self._record = RecordingTransport(directory, inner)
        self._dir = directory

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = self._dir / f"{request_key(request)}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["status"] == 200:
            return self._replay.handle_request(request)
        return self._record.handle_request(request)

    def close(self) -> None:
        self._record.close()


class CassetteMiss(httpx.TransportError):
    """Replay mode: no recording exists for this request."""


class ReplayTransport(httpx.BaseTransport):
    """Serve only recorded responses; never touch the network."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = self._dir / f"{request_key(request)}.json"
        if not path.exists():
            raise CassetteMiss(f"no recording for {request.method} {request.url}", request=request)
        record = json.loads(path.read_text(encoding="utf-8"))
        return httpx.Response(
            status_code=record["status"],
            headers=record["headers"],
            content=record["body"].encode("utf-8"),
            request=request,
        )


def build_transport(mode: str, directory: str | None) -> httpx.BaseTransport | None:
    if mode == "off":
        return None
    if not directory:
        raise ValueError("DETOUR_GITHUB_CASSETTE_DIR is required when cassette mode is on")
    if mode == "record":
        return RecordingTransport(Path(directory))
    if mode == "replay":
        return ReplayTransport(Path(directory))
    raise ValueError(f"unknown cassette mode {mode!r} (expected off|record|replay)")
