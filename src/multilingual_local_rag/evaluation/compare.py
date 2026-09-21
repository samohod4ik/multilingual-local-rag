"""Paired comparison. Both runs must cover the same complete query set."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from multilingual_local_rag.benchmark import BenchmarkDataset
from multilingual_local_rag.evaluation.metrics import (
    Prediction,
    average_precision,
    grade_map,
    ndcg_at,
    relevant_ids,
)


def _index(predictions: Sequence[Prediction]) -> dict[str, Prediction]:
    indexed = {item.query_id: item for item in predictions}
    if len(indexed) != len(predictions):
        raise ValueError("duplicate query_id in predictions")
    return indexed


def _query_map(prediction: Prediction, dataset_query_relevant: frozenset[str]) -> float:
    return average_precision(prediction.ranked_ids, dataset_query_relevant)


@dataclass(frozen=True, slots=True)
class _QueryDelta:
    query_id: str
    group_id: str
    baseline_map: float
    candidate_map: float
    delta_map: float
    baseline_ndcg_at_8: float
    candidate_ndcg_at_8: float


def compare_runs(
    dataset: BenchmarkDataset,
    baseline: Sequence[Prediction],
    candidate: Sequence[Prediction],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    expected = {query.query_id for query in dataset.queries}
    base = _index(baseline)
    cand = _index(candidate)
    if set(base) != expected or set(cand) != expected:
        raise ValueError("baseline and candidate must each contain the full query-id set")
    by_query = {query.query_id: query for query in dataset.queries}
    per_query: list[_QueryDelta] = []
    group_deltas: dict[str, list[float]] = {}
    for query_id in sorted(expected):
        query = by_query[query_id]
        relevant = relevant_ids(query)
        grades = grade_map(query)
        base_map = _query_map(base[query_id], relevant)
        cand_map = _query_map(cand[query_id], relevant)
        base_ndcg = ndcg_at(base[query_id].ranked_ids, grades, 8)
        cand_ndcg = ndcg_at(cand[query_id].ranked_ids, grades, 8)
        delta = cand_map - base_map
        per_query.append(
            _QueryDelta(
                query_id=query_id,
                group_id=query.group_id,
                baseline_map=base_map,
                candidate_map=cand_map,
                delta_map=delta,
                baseline_ndcg_at_8=base_ndcg,
                candidate_ndcg_at_8=cand_ndcg,
            )
        )
        group_deltas.setdefault(query.group_id, []).append(delta)
    group_means = [sum(values) / len(values) for values in group_deltas.values()]
    low, high = _bootstrap_interval(group_means, samples=samples, seed=seed)
    gains = sum(1 for row in per_query if row.delta_map > 0)
    losses = sum(1 for row in per_query if row.delta_map < 0)
    return {
        "per_query": [asdict(row) for row in per_query],
        "gains": gains,
        "losses": losses,
        "ties": len(per_query) - gains - losses,
        "map_delta_mean": sum(group_means) / len(group_means),
        "map_delta_low": low,
        "map_delta_high": high,
    }


def _bootstrap_interval(
    group_means: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    stats: list[float] = []
    size = len(group_means)
    for _ in range(samples):
        draw = [group_means[rng.randrange(size)] for _ in range(size)]
        stats.append(sum(draw) / size)
    stats.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return stats[low_index], stats[high_index]
