from multilingual_local_rag.benchmark import EvidenceSpan, Qrel, QueryCase
from multilingual_local_rag.evaluation.metrics import (
    Prediction,
    average_precision,
    evidence_recall,
    hit_at,
    ndcg_at,
)


def _query() -> QueryCase:
    return QueryCase(
        query_id="QRY-001",
        group_id="GRP-001",
        language="en",
        category="same_language_lexical",
        text="What is the limit?",
        qrels=(Qrel("DOC-A", 3), Qrel("DOC-B", 1)),
        evidence=(EvidenceSpan("DOC-A", "exact span"),),
    )


def test_map_uses_the_full_ranked_list() -> None:
    ranked = tuple(f"X{i}" for i in range(9)) + ("DOC-A",)
    score = average_precision(ranked, frozenset({"DOC-A"}))
    assert score > 0
    assert "map_at_8" not in average_precision.__name__


def test_grade_1_is_not_a_hit() -> None:
    assert hit_at(("DOC-B",), frozenset({"DOC-A"}), 1) == 0.0
    assert hit_at(("DOC-A",), frozenset({"DOC-A"}), 1) == 1.0


def test_ndcg_cutoff_ignores_rank_nine() -> None:
    grades = {"DOC-A": 3}
    early = ndcg_at(("DOC-A",), grades, 8)
    late = ndcg_at(tuple(f"X{i}" for i in range(8)) + ("DOC-A",), grades, 8)
    assert early == 1.0
    assert late == 0.0


def test_evidence_recall_requires_exact_span() -> None:
    query = _query()
    assert evidence_recall(query, (("DOC-A", "prefix exact span suffix"),)) == 1.0
    assert evidence_recall(query, (("DOC-A", "exact"),)) == 0.0
    assert evidence_recall(query, (("DOC-B", "exact span"),)) == 0.0


def test_prediction_rejects_duplicate_ids() -> None:
    try:
        Prediction("QRY-001", ("DOC-A", "DOC-A"), (), (), 1.0, False)
    except ValueError:
        return
    raise AssertionError("duplicate ranked ids")
