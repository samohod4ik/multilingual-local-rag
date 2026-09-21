"""Runtime configuration that cannot point at an arbitrary host path or URL."""

from __future__ import annotations

from dataclasses import dataclass

_PROFILES = frozenset({"quality", "lexical"})
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    profile: str
    data_root: str
    loopback_host: str

    def __post_init__(self) -> None:
        if self.profile not in _PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}")
        if not self.data_root or self.data_root.startswith(("/", "\\")) or ":" in self.data_root:
            raise ValueError("data_root must be a relative path")
        if ".." in self.data_root.replace("\\", "/").split("/"):
            raise ValueError("data_root must not contain ..")
        if self.loopback_host not in _LOOPBACK:
            raise ValueError("loopback_host must be a loopback name")
