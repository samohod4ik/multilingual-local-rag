"""In-memory service state. Queries are not written to a log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multilingual_local_rag.index.builder import ChunkRecord


@dataclass
class ServiceState:
    chunks: tuple[ChunkRecord, ...]
    profile: str
    model_id: str
    data_root: Path
    embedder_name: str

    def search_kwargs(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "model_id": self.model_id,
            "embedder_name": self.embedder_name,
        }
