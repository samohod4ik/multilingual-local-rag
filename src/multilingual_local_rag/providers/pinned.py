"""Pinned model adapters. Importing this module does not import a model runtime."""

from __future__ import annotations

from collections.abc import Sequence


class ProviderUnavailable(RuntimeError):
    """The pinned local model runtime is not installed."""


class PinnedEmbedder:
    """Placeholder for the pinned local embedding model. Weights stay outside git."""

    model_id = "pinned-embedder"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        del texts
        raise ProviderUnavailable("pinned embedder runtime is not installed")


class PinnedReranker:
    """Placeholder for the pinned local cross-encoder. Weights stay outside git."""

    model_id = "pinned-reranker"

    def rerank(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        del query, texts
        raise ProviderUnavailable("pinned reranker runtime is not installed")
