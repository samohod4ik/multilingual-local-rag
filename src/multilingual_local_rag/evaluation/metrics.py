"""Rank metrics. MAP is mean average precision over the full ranked list."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from multilingual_local_rag.benchmark import BenchmarkDataset, QueryCase


@dataclass(frozen=True, slots=True)
class Prediction:
    query_id: str
    ranked_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]
    latency_ms: float
    degraded: bool

    def __post_init__(self) -> None:
        if len(self.ranked_ids) != len(set(self.ranked_ids)):
            raise ValueError(f"{self.query_id} has duplicate ranked ids")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")


def relevant_ids(query: QueryCase) -> frozenset[str]:
    """Binary relevance is grade 3 only."""
    return frozenset(qrel.source_id for qrel in query.qrels if qrel.grade == 3)


def grade_map(query: QueryCase) -> dict[str, int]:
    return {qrel.source_id: qrel.grade for qrel in query.qrels}


def hit_at(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def recall_at(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """Average precision over the entire ranked list. This is not AP@k."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / index
    return total / len(relevant)


def ndcg_at(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    def dcg(values: Sequence[int]) -> float:
        score = 0.0
        for index, grade in enumerate(values[:k], start=1):
            score += (2**grade - 1) / math.log2(index + 1)
        return score

    ideal = sorted(grades.values(), reverse=True)
    ideal_score = dcg(ideal)
    if ideal_score == 0.0:
        return 0.0
    gains = [grades.get(doc_id, 0) for doc_id in ranked]
    return dcg(gains) / ideal_score


def evidence_recall(query: QueryCase, evidence: Sequence[tuple[str, str]]) -> float:
    """Fraction of gold snippets returned as exact substrings of retrieved evidence."""
    if not query.evidence:
        return 0.0
    hits = 0
    for span in query.evidence:
        for source_id, snippet in evidence:
            if source_id == span.source_id and span.snippet in snippet:
                hits += 1
                break
    return hits / len(query.evidence)


def candidate_recall(candidate_ids: Sequence[str], relevant: frozenset[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc_id in relevant if doc_id in set(candidate_ids)) / len(relevant)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass(frozen=True, slots=True)
class _ScoreRow:
    hit_at_1: float
    hit_at_3: float
    recall_at_8: float
    mrr: float
    ap: float
    ndcg_at_8: float
    evidence_recall: float
    candidate_recall: float
    latency_ms: float
    degraded: float
    language: str
    category: str


def evaluate_run(dataset: BenchmarkDataset, predictions: Sequence[Prediction]) -> dict[str, float]:
    expected = {query.query_id for query in dataset.queries}
    query_ids = [prediction.query_id for prediction in predictions]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate prediction query_id")
    got = set(query_ids)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"prediction query ids must match the dataset exactly; missing={missing} extra={extra}"
        )
    by_id = {query.query_id: query for query in dataset.queries}
    rows: list[_ScoreRow] = []
    for prediction in predictions:
        query = by_id[prediction.query_id]
        relevant = relevant_ids(query)
        grades = grade_map(query)
        rows.append(
            _ScoreRow(
                hit_at_1=hit_at(prediction.ranked_ids, relevant, 1),
                hit_at_3=hit_at(prediction.ranked_ids, relevant, 3),
                recall_at_8=recall_at(prediction.ranked_ids, relevant, 8),
                mrr=reciprocal_rank(prediction.ranked_ids, relevant),
                ap=average_precision(prediction.ranked_ids, relevant),
                ndcg_at_8=ndcg_at(prediction.ranked_ids, grades, 8),
                evidence_recall=evidence_recall(query, prediction.evidence),
                candidate_recall=candidate_recall(prediction.candidate_ids, relevant),
                latency_ms=prediction.latency_ms,
                degraded=1.0 if prediction.degraded else 0.0,
                language=query.language,
                category=query.category,
            )
        )
    summary = _mean_metrics(rows)
    summary["latency_p50_ms"] = _percentile([row.latency_ms for row in rows], 0.50)
    summary["latency_p95_ms"] = _percentile([row.latency_ms for row in rows], 0.95)
    summary["degraded_rate"] = summary.pop("degraded")
    for language in sorted({row.language for row in rows}):
        sliced = _mean_metrics([row for row in rows if row.language == language])
        for key, value in sliced.items():
            if key != "degraded":
                summary[f"{language}_{key}"] = value
    for category in sorted({row.category for row in rows}):
        sliced = _mean_metrics([row for row in rows if row.category == category])
        for key, value in sliced.items():
            if key != "degraded":
                summary[f"{category}_{key}"] = value
    if "map_at_8" in summary:
        raise ValueError("map_at_8 is not a metric")
    return summary


def _mean_metrics(rows: Sequence[_ScoreRow]) -> dict[str, float]:
    count = len(rows)
    totals = {
        "hit_at_1": sum(row.hit_at_1 for row in rows) / count,
        "hit_at_3": sum(row.hit_at_3 for row in rows) / count,
        "recall_at_8": sum(row.recall_at_8 for row in rows) / count,
        "mrr": sum(row.mrr for row in rows) / count,
        "map": sum(row.ap for row in rows) / count,
        "ndcg_at_8": sum(row.ndcg_at_8 for row in rows) / count,
        "evidence_recall": sum(row.evidence_recall for row in rows) / count,
        "candidate_recall": sum(row.candidate_recall for row in rows) / count,
        "degraded": sum(row.degraded for row in rows) / count,
    }
    return totals
