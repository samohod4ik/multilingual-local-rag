"""BM25 plus dense union, then rerank. Failures stay explicit."""

from __future__ import annotations

import math
from dataclasses import dataclass

from multilingual_local_rag.index.builder import ChunkRecord
from multilingual_local_rag.index.vector_cache import VectorCache
from multilingual_local_rag.providers.base import Embedder, Reranker
from multilingual_local_rag.providers.pinned import ProviderUnavailable
from multilingual_local_rag.retrieval.bm25 import BM25Index

_CANDIDATES = 20
_EVIDENCE = 8


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    source_id: str
    source_uri: str
    locator: str
    excerpt: str
    content_hash: str
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    evidence: tuple[EvidenceItem, ...]
    degraded: bool
    degradation: str | None
    omitted_before_rerank: int
    profile: str


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _diverse(items: list[EvidenceItem], limit: int) -> tuple[EvidenceItem, ...]:
    chosen: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        if item.source_id in seen:
            continue
        chosen.append(item)
        seen.add(item.source_id)
        if len(chosen) == limit:
            return tuple(chosen)
    for item in items:
        if item in chosen:
            continue
        chosen.append(item)
        if len(chosen) == limit:
            break
    return tuple(chosen)


def search_snapshot(
    chunks: tuple[ChunkRecord, ...],
    query: str,
    *,
    profile: str,
    embedder: Embedder | None,
    reranker: Reranker | None,
    cache: VectorCache | None,
    model_id: str,
) -> SearchResult:
    lexical = BM25Index([(chunk.chunk_id, chunk.text) for chunk in chunks])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lexical_ids = [chunk_id for chunk_id, _score in lexical.search(query, _CANDIDATES)]
    if profile == "lexical" or embedder is None:
        ordered = lexical_ids
        return _pack(
            ordered,
            by_id,
            len(chunks) - len(ordered),
            degraded=False,
            reason=None,
            profile="lexical",
        )
    try:
        query_vector = [float(item) for item in embedder.embed([query])[0]]
        dense_scores: list[tuple[str, float]] = []
        for chunk in chunks:
            if cache is None:
                raise ProviderUnavailable("vector cache is missing")
            vector = cache.get(profile, model_id, chunk.content_hash)
            if vector is None:
                continue
            dense_scores.append((chunk.chunk_id, _cosine(query_vector, vector)))
        dense_scores.sort(key=lambda item: (-item[1], item[0]))
        dense_ids = [chunk_id for chunk_id, score in dense_scores if score > 0][:_CANDIDATES]
        union: list[str] = []
        for chunk_id in lexical_ids + dense_ids:
            if chunk_id not in union:
                union.append(chunk_id)
        omitted = len(chunks) - len(union)
        if reranker is None:
            raise ProviderUnavailable("reranker is missing")
        scores = reranker.rerank(query, [by_id[chunk_id].text for chunk_id in union])
        if len(scores) != len(union) or any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("reranker returned invalid scores")
        order = sorted(range(len(union)), key=lambda index: (-float(scores[index]), union[index]))
        ranked = [union[index] for index in order]
        items = _items(ranked, by_id, [float(scores[union.index(chunk_id)]) for chunk_id in ranked])
        return SearchResult(
            evidence=_diverse(items, _EVIDENCE),
            degraded=False,
            degradation=None,
            omitted_before_rerank=omitted,
            profile=profile,
        )
    except (ProviderUnavailable, ValueError, OSError) as exc:
        fallback = _pack(
            lexical_ids,
            by_id,
            len(chunks) - len(lexical_ids),
            degraded=True,
            reason=type(exc).__name__ + ": " + str(exc),
            profile="lexical",
        )
        return fallback


def _items(
    ranked: list[str], by_id: dict[str, ChunkRecord], scores: list[float]
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for chunk_id, score in zip(ranked, scores, strict=True):
        chunk = by_id[chunk_id]
        items.append(
            EvidenceItem(
                source_id=chunk.source_id,
                source_uri=chunk.source_uri,
                locator=str(chunk.index),
                excerpt=chunk.text,
                content_hash=chunk.content_hash,
                score=score,
            )
        )
    return items


def _pack(
    ranked: list[str],
    by_id: dict[str, ChunkRecord],
    omitted: int,
    *,
    degraded: bool,
    reason: str | None,
    profile: str,
) -> SearchResult:
    scores = [float(len(ranked) - index) for index in range(len(ranked))]
    return SearchResult(
        evidence=_diverse(_items(ranked, by_id, scores), _EVIDENCE),
        degraded=degraded,
        degradation=reason,
        omitted_before_rerank=omitted,
        profile=profile,
    )
