"""SQLite cache of normalized vectors, namespaced by profile and model."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


def validate_vector(values: list[float]) -> None:
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("vector must be finite")
    norm = math.sqrt(sum(item * item for item in values))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("vector must be normalized")


class VectorCache:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                profile TEXT NOT NULL,
                model_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY (profile, model_id, content_hash)
            )
            """
        )

    def close(self) -> None:
        self._db.close()

    def get(self, profile: str, model_id: str, content_hash: str) -> list[float] | None:
        row = self._db.execute(
            "SELECT vector FROM vectors WHERE profile=? AND model_id=? AND content_hash=?",
            (profile, model_id, content_hash),
        ).fetchone()
        if row is None:
            return None
        values = [float(item) for item in json.loads(row[0])]
        validate_vector(values)
        return values

    def put(self, profile: str, model_id: str, content_hash: str, values: list[float]) -> None:
        validate_vector(values)
        self._db.execute(
            "INSERT OR REPLACE INTO vectors (profile, model_id, content_hash, vector) VALUES (?, ?, ?, ?)",
            (profile, model_id, content_hash, json.dumps(values)),
        )
        self._db.commit()
