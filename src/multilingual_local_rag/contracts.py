"""Product records emitted by a text-source adapter.

Benchmark annotations are not part of this module. A store adapter must be
able to represent a document without groups, translation pairs, or grades.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LANG_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source_uri(source_uri: str) -> None:
    """Accept only a relative URI that cannot escape its root."""
    _require(bool(source_uri), "source_uri must be non-empty")
    _require("\\" not in source_uri, f"source_uri must be posix-relative: {source_uri!r}")
    _require(":" not in source_uri, f"source_uri must not contain a drive: {source_uri!r}")
    _require(not source_uri.startswith("/"), f"source_uri must be relative: {source_uri!r}")
    parts = source_uri.split("/")
    _require(
        all(part not in ("", ".", "..") for part in parts), f"source_uri escapes: {source_uri!r}"
    )


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One textual source. Language, when present, is an open BCP 47 tag."""

    source_id: str
    source_uri: str
    text: str
    content_hash: str
    revision: str
    media_type: str
    metadata: Mapping[str, str]
    language: str | None = None

    def __post_init__(self) -> None:
        _require(bool(_SOURCE_ID_RE.match(self.source_id)), f"bad source_id {self.source_id!r}")
        validate_source_uri(self.source_uri)
        _require(bool(self.text), "text must be non-empty")
        _require(bool(_HASH_RE.match(self.content_hash)), "content_hash must be sha256 hex")
        _require(self.content_hash == sha256_text(self.text), "content_hash does not match text")
        _require(bool(self.revision.strip()), "revision must be non-empty")
        _require(bool(self.media_type) and " " not in self.media_type, "bad media_type")
        if self.language is not None:
            _require(bool(_LANG_RE.match(self.language)), f"bad language tag {self.language!r}")
        frozen: dict[str, str] = {}
        for key, value in self.metadata.items():
            _require(isinstance(key, str) and bool(key), "metadata keys must be non-empty strings")
            _require(isinstance(value, str), f"metadata[{key!r}] must be a string")
            frozen[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(frozen))
