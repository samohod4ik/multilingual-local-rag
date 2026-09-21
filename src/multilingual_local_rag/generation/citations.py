"""Keep only citation ids that exist in the evidence packet."""

from __future__ import annotations

import re

_CITE = re.compile(r"\[S(\d+)\]")


def valid_citations(answer: str, evidence_count: int) -> tuple[str, ...]:
    found: list[str] = []
    for match in _CITE.finditer(answer):
        number = int(match.group(1))
        label = f"S{number}"
        if 1 <= number <= evidence_count and label not in found:
            found.append(label)
    return tuple(found)
