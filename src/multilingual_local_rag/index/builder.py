"""Build a candidate snapshot, then publish current.json only if it validates."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from multilingual_local_rag.adapters.base import SourceAdapter
from multilingual_local_rag.chunking.markdown import chunk_markdown
from multilingual_local_rag.chunking.text import chunk_plain
from multilingual_local_rag.contracts import sha256_text
from multilingual_local_rag.index.vector_cache import VectorCache
from multilingual_local_rag.providers.base import Embedder

_KEEP = 2


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_uri: str
    index: int
    text: str
    content_hash: str


def chunk_document(
    source_id: str, source_uri: str, text: str, media_type: str
) -> tuple[ChunkRecord, ...]:
    pieces = chunk_markdown(text) if media_type == "text/markdown" else chunk_plain(text)
    records: list[ChunkRecord] = []
    for index, piece in enumerate(pieces):
        content_hash = sha256_text(piece)
        records.append(
            ChunkRecord(
                chunk_id=f"{source_id}:{index}:{content_hash[:12]}",
                source_id=source_id,
                source_uri=source_uri,
                index=index,
                text=piece,
                content_hash=content_hash,
            )
        )
    return tuple(records)


def build_snapshot(
    adapter: SourceAdapter,
    data_root: str | Path,
    *,
    embedder: Embedder | None,
    profile: str,
    model_id: str,
) -> str:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkRecord] = []
    for document in adapter.iter_documents():
        chunks.extend(
            chunk_document(
                document.source_id, document.source_uri, document.text, document.media_type
            )
        )
    snapshot_id = sha256_text("\n".join(chunk.chunk_id for chunk in chunks) + profile + model_id)[
        :16
    ]
    final = root / "snapshots" / snapshot_id
    if final.is_dir() and (final / "chunks.jsonl").is_file():
        _publish(root, snapshot_id, profile, model_id, adapter.content_hash())
        return snapshot_id
    folder = root / "snapshots" / (snapshot_id + ".partial")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    payload = [asdict(chunk) for chunk in chunks]
    (folder / "chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        encoding="utf-8",
        newline="\n",
    )
    if embedder is not None and chunks:
        cache = VectorCache(root / "caches" / "vectors.sqlite")
        try:
            missing = [
                chunk
                for chunk in chunks
                if cache.get(profile, model_id, chunk.content_hash) is None
            ]
            if missing:
                vectors = embedder.embed([chunk.text for chunk in missing])
                if len(vectors) != len(missing):
                    raise ValueError("embedder returned the wrong number of vectors")
                for chunk, vector in zip(missing, vectors, strict=True):
                    cache.put(
                        profile, model_id, chunk.content_hash, [float(item) for item in vector]
                    )
        finally:
            cache.close()
    os.replace(folder, final)
    _publish(root, snapshot_id, profile, model_id, adapter.content_hash())
    _retain(root, snapshot_id)
    return snapshot_id


def read_current(data_root: str | Path) -> dict[str, str] | None:
    path = Path(data_root) / "current.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("current.json must be an object")
    return {str(key): str(value) for key, value in data.items()}


def load_chunks(data_root: str | Path, snapshot_id: str) -> tuple[ChunkRecord, ...]:
    path = Path(data_root) / "snapshots" / snapshot_id / "chunks.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(ChunkRecord(**json.loads(line)))
    return tuple(rows)


def _publish(root: Path, snapshot_id: str, profile: str, model_id: str, source_hash: str) -> None:
    body = json.dumps(
        {
            "snapshot_id": snapshot_id,
            "profile": profile,
            "model_id": model_id,
            "source_hash": source_hash,
        },
        sort_keys=True,
    )
    temporary = root / "current.json.tmp"
    temporary.write_text(body, encoding="utf-8", newline="\n")
    os.replace(temporary, root / "current.json")


def _retain(root: Path, current_id: str) -> None:
    folder = root / "snapshots"
    names = sorted(path.name for path in folder.iterdir() if path.is_dir())
    stale = [name for name in names if name != current_id]
    for name in stale[:-_KEEP] if len(stale) > _KEEP else []:
        target = folder / name
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
