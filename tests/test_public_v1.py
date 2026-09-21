import importlib.util
from pathlib import Path
from typing import Any

from multilingual_local_rag.evaluation.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "public-v1"


def _load_generate() -> Any:
    spec = importlib.util.spec_from_file_location("public_v1_generate", ROOT / "generate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_v1_regenerated_bytes_match_committed(tmp_path: Path) -> None:
    _load_generate().write_fixture(tmp_path)
    for name in ("groups.jsonl", "documents.jsonl", "queries.jsonl"):
        generated = (tmp_path / name).read_bytes().replace(b"\r\n", b"\n")
        committed = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        assert generated == committed


def test_public_v1_gold_grades_and_pairs() -> None:
    dataset = load_dataset(ROOT)
    assert len(dataset.groups) == 40
    assert len(dataset.queries) == 80
    folds: dict[str, str] = {}
    for group in dataset.groups:
        folds[group.group_id] = group.fold
    pair_groups: dict[str, set[str]] = {}
    for doc in dataset.documents:
        if doc.role == "primary" and doc.pair_id is not None:
            pair_groups.setdefault(doc.pair_id, set()).add(doc.group_id)
    for pair_id, group_ids in pair_groups.items():
        assert len(group_ids) == 1, pair_id
        assert len({folds[gid] for gid in group_ids}) == 1
    for query in dataset.queries:
        grades = {qrel.grade for qrel in query.qrels}
        assert 2 not in grades
        assert 3 in grades
