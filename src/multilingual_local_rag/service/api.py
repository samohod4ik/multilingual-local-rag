"""Versioned loopback API. It refuses any non-loopback bind address."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from multilingual_local_rag.generation.answer import AnswerResult, answer_question
from multilingual_local_rag.index.builder import load_chunks, read_current
from multilingual_local_rag.index.vector_cache import VectorCache
from multilingual_local_rag.providers.deterministic import HashEmbedder, OverlapReranker
from multilingual_local_rag.providers.pinned import PinnedEmbedder, PinnedReranker
from multilingual_local_rag.retrieval.hybrid import search_snapshot
from multilingual_local_rag.service.state import ServiceState

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
_MAX_BODY = 16_384


class LoopbackServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, state: ServiceState) -> None:
        if host not in _LOOPBACK:
            raise ValueError("service host must be loopback")
        self.state = state
        super().__init__((host, port), Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        name = host.split(":", 1)[0]
        return name in _HOSTS

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            self._send({"error": "bad_host"}, status=421)
            return
        path = urlparse(self.path).path
        if path == "/v1/capabilities":
            self._send(
                {"version": "1", "endpoints": ["capabilities", "health", "search", "answer"]}
            )
            return
        if path == "/v1/health":
            current = read_current(cast(LoopbackServer, self.server).state.data_root)
            self._send(
                {
                    "ok": current is not None,
                    "snapshot": None if current is None else current["snapshot_id"],
                }
            )
            return
        self._send({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            self._send({"error": "bad_host"}, status=421)
            return
        path = urlparse(self.path).path
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self.close_connection = True
            self._send({"error": "bad_length"}, status=400)
            return
        if length < 0 or length > _MAX_BODY:
            self.close_connection = True
            self._send({"error": "body_too_large"}, status=413)
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send({"error": "bad_json"}, status=400)
            return
        if (
            not isinstance(payload, dict)
            or "query" not in payload
            or not isinstance(payload["query"], str)
        ):
            self._send({"error": "query_required"}, status=400)
            return
        state = cast(LoopbackServer, self.server).state
        if path == "/v1/search":
            self._send(asdict(_search(state, payload["query"])))
            return
        if path == "/v1/answer":
            generate = payload.get("generate", True)
            self._send(asdict(_answer(state, payload["query"], bool(generate))))
            return
        self._send({"error": "not_found"}, status=404)

    def _send(self, body: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _search(state: ServiceState, query: str) -> Any:
    embedder, reranker, model_id = _models(state)
    cache = (
        VectorCache(state.data_root / "caches" / "vectors.sqlite") if embedder is not None else None
    )
    try:
        return search_snapshot(
            state.chunks,
            query,
            profile=state.profile,
            embedder=embedder,  # type: ignore[arg-type]
            reranker=reranker,  # type: ignore[arg-type]
            cache=cache,
            model_id=model_id,
        )
    finally:
        if cache is not None:
            cache.close()


def _answer(state: ServiceState, query: str, generate: bool) -> AnswerResult:
    found = _search(state, query)
    generator: Callable[[str], str] | None = _scripted if generate else None
    return answer_question(found, query, generator)


def _scripted(prompt: str) -> str:
    if "[S1]" not in prompt:
        return "I abstain."
    return "Supported by the first source [S1]."


def _models(state: ServiceState) -> tuple[object | None, object | None, str]:
    if state.profile == "lexical" or state.embedder_name == "pinned":
        if state.profile == "lexical":
            return None, None, "lexical"
        return PinnedEmbedder(), PinnedReranker(), PinnedEmbedder.model_id
    return HashEmbedder(), OverlapReranker(), HashEmbedder.model_id


def load_state(data_root: str | Path, *, profile: str, embedder_name: str) -> ServiceState:
    root = Path(data_root)
    current = read_current(root)
    if current is None:
        raise ValueError("no snapshot")
    return ServiceState(
        chunks=load_chunks(root, current["snapshot_id"]),
        profile=profile,
        model_id=current["model_id"],
        data_root=root,
        embedder_name=embedder_name,
    )
