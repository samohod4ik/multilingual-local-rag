"""Paragraph windows for plain text."""

from __future__ import annotations

import re


def chunk_plain(text: str) -> tuple[str, ...]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts and text.strip():
        return (text.strip(),)
    return tuple(parts)
