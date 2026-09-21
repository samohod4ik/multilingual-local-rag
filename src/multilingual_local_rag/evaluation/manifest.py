"""Manifest paths must stay inside the manifest directory after resolution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BENCHMARK_FILES = ("groups.jsonl", "documents.jsonl", "queries.jsonl", "SCHEMA.md")


def _contained(root: Path, candidate: Path) -> bool:
    root_text = str(root)
    cand_text = str(candidate)
    if os.name == "nt":
        root_text = root_text.casefold()
        cand_text = cand_text.casefold()
        separator = "\\"
    else:
        separator = "/"
    return cand_text == root_text or cand_text.startswith(root_text + separator)


def resolve_contained(root: Path, relative: str) -> Path:
    """Reject absolute paths and any `..` segment, then require containment."""
    normalized = relative.replace("\\", "/")
    raw = Path(normalized)
    if raw.is_absolute() or ".." in raw.parts or ":" in normalized:
        raise ValueError(f"path escape: {relative!r}")
    root_resolved = root.resolve()
    candidate = (root_resolved / raw).resolve()
    if not _contained(root_resolved, candidate):
        raise ValueError(f"path escapes manifest root: {relative!r}")
    return candidate


def dataset_hash(benchmark_dir: str | Path) -> str:
    root = Path(benchmark_dir)
    digest = hashlib.sha256()
    for name in BENCHMARK_FILES:
        file_path = resolve_contained(root, name)
        if not file_path.is_file():
            raise FileNotFoundError(name)
        digest.update(hashlib.sha256(name.encode("utf-8")).digest())
        digest.update(hashlib.sha256(file_path.read_bytes()).digest())
    return digest.hexdigest()


def build_manifest(benchmark_dir: str | Path, *, run_id: str, config_hash: str) -> dict[str, str]:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "dataset_hash": dataset_hash(benchmark_dir),
        "config_hash": config_hash,
    }


def verify_manifest(benchmark_dir: str | Path, manifest: dict[str, str]) -> None:
    allowed = {"schema_version", "run_id", "dataset_hash", "config_hash"}
    extra = set(manifest) - allowed
    if extra:
        raise ValueError(f"unknown manifest fields {sorted(extra)}")
    for key in allowed:
        if key not in manifest or not str(manifest[key]).strip():
            raise ValueError(f"missing manifest field {key}")
    if manifest["schema_version"] != "1":
        raise ValueError("unsupported schema_version")
    actual = dataset_hash(benchmark_dir)
    if manifest["dataset_hash"] != actual:
        raise ValueError("dataset_hash does not match benchmark files")


def read_manifest(path: str | Path, benchmark_dir: str | Path) -> dict[str, str]:
    root = Path(benchmark_dir)
    file_path = resolve_contained(root, Path(path).as_posix())
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    normalized = {str(key): str(value) for key, value in data.items()}
    verify_manifest(root, normalized)
    return normalized
