import json
from pathlib import Path

import pytest

from multilingual_local_rag.benchmark import BenchmarkDataset, Qrel, QueryCase, SourceGroup
from multilingual_local_rag.config import RuntimeConfig
from multilingual_local_rag.evaluation.compare import compare_runs
from multilingual_local_rag.evaluation.envelope import parse_predictions
from multilingual_local_rag.evaluation.manifest import (
    build_manifest,
    read_manifest,
    resolve_contained,
)
from multilingual_local_rag.evaluation.metrics import Prediction, evaluate_run

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "public-v1"


def _tiny() -> BenchmarkDataset:
    query = QueryCase(
        query_id="QRY-001",
        group_id="GRP-001",
        language="en",
        category="same_language_lexical",
        text="limit?",
        qrels=(Qrel("DOC-A", 3),),
        evidence=(),
    )
    group = SourceGroup("GRP-001", "fold-1", "Amberlathe", "same_language_lexical")
    return BenchmarkDataset(groups=(group,), documents=(), queries=(query,))


def _pred(query_id: str, first: str) -> Prediction:
    return Prediction(query_id, (first,), (first,), (), 1.0, False)


def test_compare_rejects_partial_overlap() -> None:
    dataset = _tiny()
    other = Prediction("QRY-002", ("DOC-A",), ("DOC-A",), (), 1.0, False)
    with pytest.raises(ValueError, match="full query-id set"):
        compare_runs(dataset, (_pred("QRY-001", "DOC-A"),), (other,))


def test_evaluate_rejects_missing_query() -> None:
    dataset = _tiny()
    with pytest.raises(ValueError, match="exactly"):
        evaluate_run(dataset, ())


def test_evaluate_rejects_duplicate_query_id() -> None:
    dataset = _tiny()
    pred = _pred("QRY-001", "DOC-A")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_run(dataset, (pred, pred))


def test_manifest_rejects_parent_segments(tmp_path: Path) -> None:
    (tmp_path / "groups.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="path escape"):
        resolve_contained(tmp_path, "..\\outside")
    with pytest.raises(ValueError, match="path escape"):
        resolve_contained(tmp_path, str(tmp_path / "groups.jsonl"))


def test_manifest_accepts_child(tmp_path: Path) -> None:
    child = tmp_path / "groups.jsonl"
    child.write_text("{}\n", encoding="utf-8")
    assert resolve_contained(tmp_path, "groups.jsonl") == child.resolve()


def _envelope_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": "QRY-001",
        "ranked_ids": ["DOC-A"],
        "candidate_ids": ["DOC-A"],
        "evidence": [{"source_id": "DOC-A", "snippet": "span"}],
        "latency_ms": 1.0,
        "degraded": False,
    }
    row.update(overrides)
    return {
        "schema_version": "1",
        "run_id": "run-1",
        "dataset_hash": "abc",
        "config_hash": "def",
        "predictions": [row],
    }


def test_envelope_rejects_string_ranked_ids() -> None:
    with pytest.raises(ValueError, match="list of strings"):
        parse_predictions(_envelope_row(ranked_ids="DOC-A"), dataset_hash="abc", config_hash="def")


def test_envelope_rejects_string_degraded() -> None:
    with pytest.raises(ValueError, match="degraded must be a bool"):
        parse_predictions(
            _envelope_row(degraded="false"), dataset_hash="abc", config_hash="def"
        )


def test_data_root_rejects_windows_parent_segments() -> None:
    for root in (r"foo\..\bar", r"..\data"):
        with pytest.raises(ValueError, match=r"\.\."):
            RuntimeConfig(profile="lexical", data_root=root, loopback_host="127.0.0.1")


def test_envelope_rejects_unknown_and_hash_mismatch() -> None:
    payload = {
        "schema_version": "1",
        "run_id": "run-1",
        "dataset_hash": "abc",
        "config_hash": "def",
        "predictions": [],
        "extra": True,
    }
    with pytest.raises(ValueError, match="unknown fields"):
        parse_predictions(payload, dataset_hash="abc", config_hash="def")
    payload.pop("extra")
    with pytest.raises(ValueError, match="dataset_hash"):
        parse_predictions(payload, dataset_hash="nope", config_hash="def")


def test_read_manifest_roundtrip(tmp_path: Path) -> None:
    for name in ("groups.jsonl", "documents.jsonl", "queries.jsonl", "SCHEMA.md"):
        (tmp_path / name).write_text(name + "\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, run_id="run-1", config_hash="cfg")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = read_manifest("manifest.json", tmp_path)
    assert loaded["run_id"] == "run-1"
    with pytest.raises(ValueError, match="path escape"):
        read_manifest("..\\manifest.json", tmp_path)
