"""Strict prediction envelope. Unknown top-level fields are rejected."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multilingual_local_rag.evaluation.metrics import Prediction

_ENVELOPE_KEYS = frozenset(
    {"schema_version", "run_id", "dataset_hash", "config_hash", "predictions"}
)
_ROW_KEYS = frozenset(
    {"query_id", "ranked_ids", "candidate_ids", "evidence", "latency_ms", "degraded"}
)
_EVIDENCE_KEYS = frozenset({"source_id", "snippet"})


def _reject_unknown(row: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    extra = set(row) - allowed
    if extra:
        raise ValueError(f"{where} has unknown fields {sorted(extra)}")


def _require_str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{where} must be a list of strings")
    return value


def parse_predictions(
    payload: Mapping[str, Any], *, dataset_hash: str, config_hash: str
) -> tuple[str, tuple[Prediction, ...]]:
    _reject_unknown(payload, _ENVELOPE_KEYS, "envelope")
    for key in _ENVELOPE_KEYS:
        if key not in payload:
            raise ValueError(f"envelope missing {key}")
    if payload["schema_version"] != "1":
        raise ValueError("unsupported schema_version")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if payload["dataset_hash"] != dataset_hash:
        raise ValueError("dataset_hash does not match the benchmark")
    if payload["config_hash"] != config_hash:
        raise ValueError("config_hash does not match the run config")
    rows = payload["predictions"]
    if not isinstance(rows, list):
        raise ValueError("predictions must be a list")
    parsed: list[Prediction] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"predictions[{index}] must be an object")
        _reject_unknown(row, _ROW_KEYS, f"predictions[{index}]")
        ranked_ids = _require_str_list(row["ranked_ids"], f"predictions[{index}] ranked_ids")
        candidate_ids = _require_str_list(
            row["candidate_ids"], f"predictions[{index}] candidate_ids"
        )
        raw_evidence = row["evidence"]
        if not isinstance(raw_evidence, list):
            raise ValueError(f"predictions[{index}] evidence must be a list")
        evidence: list[tuple[str, str]] = []
        for span in raw_evidence:
            if not isinstance(span, dict):
                raise ValueError(f"predictions[{index}] evidence items must be objects")
            _reject_unknown(span, _EVIDENCE_KEYS, f"predictions[{index}] evidence")
            source_id = span["source_id"]
            snippet = span["snippet"]
            if not isinstance(source_id, str) or not isinstance(snippet, str):
                raise ValueError(
                    f"predictions[{index}] evidence source_id and snippet must be strings"
                )
            evidence.append((source_id, snippet))
        degraded = row["degraded"]
        if not isinstance(degraded, bool):
            raise ValueError(f"predictions[{index}] degraded must be a bool")
        parsed.append(
            Prediction(
                query_id=row["query_id"],
                ranked_ids=tuple(ranked_ids),
                candidate_ids=tuple(candidate_ids),
                evidence=tuple(evidence),
                latency_ms=float(row["latency_ms"]),
                degraded=degraded,
            )
        )
    return run_id, tuple(parsed)


def load_predictions(
    path: str | Path, *, dataset_hash: str, config_hash: str
) -> tuple[str, tuple[Prediction, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prediction file must be an object")
    return parse_predictions(payload, dataset_hash=dataset_hash, config_hash=config_hash)
