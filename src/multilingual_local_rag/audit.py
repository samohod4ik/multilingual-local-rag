"""Repository text audit for host residue and release claims."""

from __future__ import annotations

import os
from pathlib import Path

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".egg-info",
        "dist",
        "build",
    }
)
_MAX_BYTES = 1_000_000


def _forbidden_tokens() -> tuple[tuple[str, str], ...]:
    personal = "Aleksandr " + "Muraviov"
    other_owner = "aleksandr-" + "muraviov"
    host_token = "Her" + "mes"
    user_prefix = "C:\\" + "Users\\"
    host_drive = "C:\\" + host_token
    vault = "G:\\" + "Agents" + "KnowledgeBase"
    fake_date = "2026-01-" + "01"
    return (
        ("personal-name", personal),
        ("other-owner", other_owner),
        ("her" + "mes", host_token),
        ("user-prefix", user_prefix),
        ("her" + "mes-drive", host_drive),
        ("vault", vault),
        ("fake-release-date", fake_date),
    )


def audit_tree(root: str | Path) -> list[str]:
    """Return problem strings. LICENSE may keep the bootstrap copyright name."""
    base = Path(root)
    problems: list[str] = []
    tokens = _forbidden_tokens()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [
            name for name in dirnames if name not in _SKIP_DIRS and not name.endswith(".egg-info")
        ]
        for filename in filenames:
            if filename in _SKIP_DIRS:
                continue
            path = Path(dirpath) / filename
            if path.stat().st_size > _MAX_BYTES:
                problems.append(f"oversized {path.relative_to(base).as_posix()}")
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                problems.append(f"binary {path.relative_to(base).as_posix()}")
                continue
            text = raw.decode("utf-8").casefold()
            relative = path.relative_to(base).as_posix()
            for label, token in tokens:
                needle = token.casefold()
                haystack = text
                if label == "user-prefix":
                    haystack = haystack.replace("\\", "/")
                    needle = needle.replace("\\", "/")
                if needle not in haystack:
                    continue
                if label == "personal-name" and relative == "LICENSE":
                    continue
                problems.append(f"{relative}: {label}")
    return problems
