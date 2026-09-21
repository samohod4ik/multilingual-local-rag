"""Keep only citation ids that exist in the evidence packet."""

from __future__ import annotations

import re

_CITE = re.compile(r"\[S(\d+)\]")


def valid_citations(answer: str, evidence_count: int) -> tuple[str, ...] | None:
    """Return citation labels, or None if any label is outside the evidence packet."""
    found: list[str] = []
    for match in _CITE.finditer(answer):
        number = int(match.group(1))
        if number < 1 or number > evidence_count:
            return None
        label = f"S{number}"
        if label not in found:
            found.append(label)
    return tuple(found)
