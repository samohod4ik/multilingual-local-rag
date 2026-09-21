"""Adapter protocol. Implementations yield product records only."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from multilingual_local_rag.contracts import SourceDocument


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    adapter_id: str
    version: str
    source_kind: str

    def __post_init__(self) -> None:
        for name in ("adapter_id", "version", "source_kind"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


@runtime_checkable
class SourceAdapter(Protocol):
    @property
    def info(self) -> AdapterInfo:
        """Static adapter metadata."""
        ...

    def iter_documents(self) -> Iterator[SourceDocument]:
        """Yield product records in a deterministic order."""
        ...

    def content_hash(self) -> str:
        """Stable hash of the underlying source bytes this adapter reads."""
        ...
