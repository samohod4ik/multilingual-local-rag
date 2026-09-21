"""Heading and paragraph chunks for Markdown."""

from __future__ import annotations

import re

_HEADING = re.compile(r"(?m)^#{1,6} .+$")


def chunk_markdown(text: str) -> tuple[str, ...]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        from multilingual_local_rag.chunking.text import chunk_plain

        return chunk_plain(text)
    chunks: list[str] = []
    if matches[0].start() > 0:
        preface = text[: matches[0].start()].strip()
        if preface:
            chunks.append(preface)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start() : end].strip()
        if section:
            chunks.append(section)
    return tuple(chunks)
