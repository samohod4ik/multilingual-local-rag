"""Deterministic stand-ins. They are not the pinned GPU models."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

_DIM = 16


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        raise ValueError("zero vector")
    return [item / norm for item in values]


class HashEmbedder:
    """Bag-of-tokens hashing. Tests use this. It is not Nemotron."""

    model_id = "hash-embed-v1"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            buckets = [0.0] * _DIM
            for token in text.casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                buckets[digest[0] % _DIM] += 1.0
            if not any(buckets):
                buckets[0] = 1.0
            vectors.append(_normalize(buckets))
        return vectors


class OverlapReranker:
    """Scores query-token overlap. Tests use this. It is not the NVIDIA reranker."""

    model_id = "overlap-rerank-v1"

    def rerank(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        wanted = set(query.casefold().split())
        scores: list[float] = []
        for text in texts:
            seen = set(text.casefold().split())
            scores.append(float(len(wanted & seen)))
        return scores
