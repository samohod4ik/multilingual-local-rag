"""Strict JSONL loader. Unknown keys are errors."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from multilingual_local_rag.benchmark import (
    BenchmarkDataset,
    BenchmarkDocument,
    EvidenceSpan,
    Qrel,
    QueryCase,
    SourceGroup,
    validate_dataset,
)

_GROUP_KEYS = frozenset({"group_id", "fold", "topic", "category"})
_DOC_KEYS = frozenset(
    {
        "source_id",
        "group_id",
        "language",
        "title",
        "text",
        "source_uri",
        "content_hash",
        "revision",
        "media_type",
        "pair_id",
        "role",
        "supersedes",
    }
)
_QUERY_KEYS = frozenset(
    {"query_id", "group_id", "language", "category", "text", "qrels", "evidence"}
)
_QREL_KEYS = frozenset({"source_id", "grade"})
_EVIDENCE_KEYS = frozenset({"source_id", "snippet"})


def load_dataset(directory: str | Path) -> BenchmarkDataset:
    root = Path(directory)
    groups = tuple(_load_groups(root / "groups.jsonl"))
    documents = tuple(_load_documents(root / "documents.jsonl"))
    queries = tuple(_load_queries(root / "queries.jsonl"))
    dataset = BenchmarkDataset(groups=groups, documents=documents, queries=queries)
    validate_dataset(dataset)
    return dataset


def _load_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_no} must be an object")
        yield row


def _reject_unknown(row: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    extra = set(row) - allowed
    if extra:
        raise ValueError(f"{where} has unknown fields {sorted(extra)}")


def _load_groups(path: Path) -> Iterable[SourceGroup]:
    for index, row in enumerate(_load_jsonl(path), start=1):
        _reject_unknown(row, _GROUP_KEYS, f"groups.jsonl:{index}")
        yield SourceGroup(
            group_id=row["group_id"],
            fold=row["fold"],
            topic=row["topic"],
            category=row["category"],
        )


def _load_documents(path: Path) -> Iterable[BenchmarkDocument]:
    for index, row in enumerate(_load_jsonl(path), start=1):
        _reject_unknown(row, _DOC_KEYS, f"documents.jsonl:{index}")
        yield BenchmarkDocument(
            source_id=row["source_id"],
            group_id=row["group_id"],
            language=row["language"],
            title=row["title"],
            text=row["text"],
            source_uri=row["source_uri"],
            content_hash=row["content_hash"],
            revision=row["revision"],
            media_type=row["media_type"],
            pair_id=row.get("pair_id"),
            role=row.get("role", "primary"),
            supersedes=row.get("supersedes"),
        )


def _load_queries(path: Path) -> Iterable[QueryCase]:
    for index, row in enumerate(_load_jsonl(path), start=1):
        _reject_unknown(row, _QUERY_KEYS, f"queries.jsonl:{index}")
        qrels = []
        for qrel in row["qrels"]:
            _reject_unknown(qrel, _QREL_KEYS, f"queries.jsonl:{index} qrel")
            qrels.append(Qrel(source_id=qrel["source_id"], grade=qrel["grade"]))
        evidence = []
        for span in row["evidence"]:
            _reject_unknown(span, _EVIDENCE_KEYS, f"queries.jsonl:{index} evidence")
            evidence.append(EvidenceSpan(source_id=span["source_id"], snippet=span["snippet"]))
        yield QueryCase(
            query_id=row["query_id"],
            group_id=row["group_id"],
            language=row["language"],
            category=row["category"],
            text=row["text"],
            qrels=tuple(qrels),
            evidence=tuple(evidence),
        )
