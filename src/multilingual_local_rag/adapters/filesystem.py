"""Recursive UTF-8 text adapter. It does not follow links or emit host paths."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from pathlib import Path

from multilingual_local_rag.adapters.base import AdapterInfo
from multilingual_local_rag.contracts import SourceDocument, sha256_text, validate_source_uri

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_SKIP_NAMES = {".env", "credentials.json", "data.json", "id_rsa"}
_SKIP_SUFFIXES = {".pem", ".key", ".p12", ".sqlite"}
_REPARSE = 0x400


def is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    kernel = getattr(ctypes, "windll").kernel32
    attrs = kernel.GetFileAttributesW(str(path))
    return attrs != -1 and bool(attrs & _REPARSE)


def paths_contained(root_text: str, cand_text: str, *, windows: bool) -> bool:
    if windows:
        root_text = root_text.casefold()
        cand_text = cand_text.casefold()
        separator = "\\"
    else:
        separator = "/"
    return cand_text == root_text or cand_text.startswith(root_text + separator)


def _contained(root: Path, candidate: Path) -> bool:
    return paths_contained(str(root.resolve()), str(candidate.resolve()), windows=os.name == "nt")


class FilesystemAdapter:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError("filesystem root must be a directory")
        if is_reparse(self._root):
            raise ValueError("filesystem root must not be a link")

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo("filesystem", "1", "filesystem")

    def content_hash(self) -> str:
        digest = sha256_text("\n".join(doc.content_hash for doc in self.iter_documents()))
        return digest

    def iter_documents(self) -> Iterator[SourceDocument]:
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            current = Path(dirpath)
            kept: list[str] = []
            for name in sorted(dirnames):
                child = current / name
                if is_reparse(child) or not _contained(self._root, child):
                    continue
                kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                path = current / name
                if not self._allowed(path):
                    continue
                if is_reparse(path) or not _contained(self._root, path):
                    continue
                relative = path.relative_to(self._root).as_posix()
                text = self._read(path, relative)
                if not text.strip():
                    continue
                validate_source_uri(relative)
                content_hash = sha256_text(text)
                suffix = path.suffix.casefold()
                media = "text/markdown" if suffix in {".md", ".markdown"} else "text/plain"
                yield SourceDocument(
                    source_id="src-" + sha256_text(relative)[:16],
                    source_uri=relative,
                    text=text,
                    content_hash=content_hash,
                    revision=content_hash[:12],
                    media_type=media,
                    metadata={"title": path.stem},
                    language=None,
                )

    def _allowed(self, path: Path) -> bool:
        name = path.name.casefold()
        stem = path.stem.casefold()
        if name in _SKIP_NAMES or stem in {".env", "credentials", "id_rsa", "data"}:
            return False
        if path.suffix.casefold() in _SKIP_SUFFIXES:
            return False
        return path.suffix.casefold() in _TEXT_SUFFIXES

    def _read(self, path: Path, relative: str) -> str:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"not utf-8: {relative}") from exc
