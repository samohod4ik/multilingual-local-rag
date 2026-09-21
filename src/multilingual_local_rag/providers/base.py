"""Provider protocols. This module must not import a model runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one finite vector per text."""
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """Return one score per text, higher is better."""
        ...
