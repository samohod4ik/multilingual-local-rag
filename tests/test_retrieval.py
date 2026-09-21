import json
from pathlib import Path

import pytest

from multilingual_local_rag.adapters.filesystem import FilesystemAdapter, is_reparse
from multilingual_local_rag.contracts import sha256_text
from multilingual_local_rag.index.builder import build_snapshot, chunk_document, read_current
from multilingual_local_rag.index.vector_cache import VectorCache, validate_vector
from multilingual_local_rag.providers.deterministic import HashEmbedder, OverlapReranker
from multilingual_local_rag.providers.pinned import (
    PinnedEmbedder,
    PinnedReranker,
    ProviderUnavailable,
)
from multilingual_local_rag.retrieval.hybrid import search_snapshot


def test_filesystem_skips_secrets_and_keeps_relative_uris(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Pump\n\nThe limit is 42.\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_bytes(b"\xef\xbb\xbfPlain body.\n")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")
    docs = list(FilesystemAdapter(tmp_path).iter_documents())
    uris = {doc.source_uri for doc in docs}
    assert uris == {"note.md", "plain.txt"}
    plain = next(doc for doc in docs if doc.source_uri == "plain.txt")
    assert plain.text == "Plain body.\n"
    assert ":" not in plain.source_uri
    assert not hasattr(plain, "group_id")


def test_filesystem_rejects_non_utf8(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="utf-8"):
        list(FilesystemAdapter(tmp_path).iter_documents())


def test_reparse_helper_is_false_for_a_normal_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("ok", encoding="utf-8")
    assert is_reparse(path) is False


def test_failed_embed_does_not_publish(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha\n", encoding="utf-8")
    data = tmp_path / "data"

    class Boom:
        model_id = "boom"

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ProviderUnavailable("missing")

    with pytest.raises(ProviderUnavailable):
        build_snapshot(
            FilesystemAdapter(tmp_path), data, embedder=Boom(), profile="quality", model_id="boom"
        )
    assert read_current(data) is None


def test_vector_cache_namespaces_and_rejects_bad_vectors(tmp_path: Path) -> None:
    cache = VectorCache(tmp_path / "vectors.sqlite")
    good = [1.0] + [0.0] * 3
    cache.put("quality", "model-a", "abc", good)
    cache.put("quality", "model-b", "abc", [0.0, 1.0, 0.0, 0.0])
    assert cache.get("quality", "model-a", "abc") == good
    assert cache.get("quality", "model-b", "abc") != good
    with pytest.raises(ValueError):
        validate_vector([float("nan")])
    cache.close()


def test_union_rerank_and_lexical_fallback() -> None:
    chunks = chunk_document("s-a", "a.txt", "alpha beta", "text/plain") + chunk_document(
        "s-b", "b.txt", "unrelated gamma", "text/plain"
    )
    embedder = HashEmbedder()
    query_vector = [float(item) for item in embedder.embed(["alpha"])[0]]

    class Cache:
        def get(self, profile: str, model_id: str, content_hash: str) -> list[float] | None:
            if content_hash == chunks[1].content_hash:
                return query_vector
            return [float(item) for item in embedder.embed([chunks[0].text])[0]]

    result = search_snapshot(
        chunks,
        "alpha",
        profile="quality",
        embedder=embedder,
        reranker=OverlapReranker(),
        cache=Cache(),  # type: ignore[arg-type]
        model_id=embedder.model_id,
    )
    assert {item.source_id for item in result.evidence} == {"s-a", "s-b"}
    assert result.evidence[0].source_id == "s-a"
    assert result.degraded is False
    assert result.omitted_before_rerank == 0
    fallback = search_snapshot(
        chunks,
        "alpha",
        profile="quality",
        embedder=PinnedEmbedder(),
        reranker=PinnedReranker(),
        cache=Cache(),  # type: ignore[arg-type]
        model_id="pinned-embedder",
    )
    assert fallback.degraded is True
    assert fallback.profile == "lexical"
    assert all(item.source_uri.count(":") == 0 for item in fallback.evidence)


def test_markdown_chunks_follow_headings() -> None:
    chunks = chunk_document("s", "n.md", "# One\n\nAlpha.\n\n# Two\n\nBeta.\n", "text/markdown")
    assert len(chunks) == 2
    assert chunks[0].text.startswith("# One")
    assert chunks[0].chunk_id.startswith("s:0:")
    assert chunks[0].content_hash == sha256_text(chunks[0].text)


def test_index_roundtrip_writes_relative_evidence(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("alpha limit\n", encoding="utf-8")
    data = tmp_path / "idx"
    embedder = HashEmbedder()
    snapshot = build_snapshot(
        FilesystemAdapter(source),
        data,
        embedder=embedder,
        profile="quality",
        model_id=embedder.model_id,
    )
    current = read_current(data)
    assert current is not None
    assert current["snapshot_id"] == snapshot
    encoded = json.dumps(current)
    assert "Users" not in encoded
    assert snapshot
