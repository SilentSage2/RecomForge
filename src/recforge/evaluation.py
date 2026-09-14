"""Exact temporal-corpus evaluation for retrieval models."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from recforge.data.features import ItemFeatureTable, aggregate_history_features
from recforge.data.mind import iter_mind_behaviors
from recforge.data.protocol import (
    TemporalCatalogIndex,
    iter_positive_retrieval_examples,
)
from recforge.metrics import recall_at_k, reciprocal_rank_at_k
from recforge.models.two_tower import TwoTowerRetriever


def positive_target_popularity(behavior_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for behavior in iter_mind_behaviors(behavior_path):
        counts.update(
            item_id
            for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        )
    return counts


def _popularity_thresholds(popularity: Counter[str]) -> tuple[int, int]:
    nonzero = sorted(popularity.values())
    if not nonzero:
        return 0, 0
    return nonzero[len(nonzero) // 3], nonzero[(2 * len(nonzero)) // 3]


def _bucket(count: int, thresholds: tuple[int, int]) -> str:
    low, high = thresholds
    if count <= low:
        return "tail"
    if count <= high:
        return "mid"
    return "head"


def _encode_items(
    model: TwoTowerRetriever,
    table: ItemFeatureTable,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(table.item_ids), 4096):
            features = np.array(table.features[start : start + 4096], copy=True)
            chunks.append(model.encode_items(torch.from_numpy(features).to(device)))
    return torch.cat(chunks)


def evaluate_temporal_corpus(
    model: TwoTowerRetriever,
    table: ItemFeatureTable,
    *,
    catalog: TemporalCatalogIndex,
    behavior_path: Path,
    training_popularity: Counter[str],
    max_history_items: int,
    device: torch.device,
    max_queries: int | None = None,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Run exact top-100 retrieval against each query's time-eligible catalog."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    examples = tuple(iter_positive_retrieval_examples(behavior_path, namespace="eval"))
    if max_queries is not None:
        examples = examples[:max_queries]
    if not examples:
        raise ValueError("evaluation contains no positive queries")

    row_by_item = table.row_by_item_id()
    try:
        temporal_rows = [row_by_item[item_id] for item_id in catalog.item_ids]
    except KeyError as error:
        raise ValueError(f"catalog item {error.args[0]!r} has no features") from error
    position_by_item = {item_id: position for position, item_id in enumerate(catalog.item_ids)}
    thresholds = _popularity_thresholds(training_popularity)
    slice_hits = Counter[str]()
    slice_targets = Counter[str]()
    recommended: set[str] = set()
    recall_20 = 0.0
    recall_100 = 0.0
    mrr_20 = 0.0
    search_seconds = 0.0

    model.eval()
    item_embeddings = _encode_items(model, table, device)[temporal_rows]
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        user_array = np.stack(
            [
                aggregate_history_features(
                    example.history_item_ids,
                    table,
                    max_history_items=max_history_items,
                    row_by_item_id=row_by_item,
                )
                for example in batch
            ]
        ).astype(np.float32, copy=False)
        with torch.no_grad():
            user_embeddings = model.encode_users(torch.from_numpy(user_array).to(device))
            eligible_sizes = [
                bisect_right(catalog.observed_at, example.local_timestamp) for example in batch
            ]
            max_eligible = max(eligible_sizes)
            started = time.perf_counter()
            scores = user_embeddings @ item_embeddings[:max_eligible].transpose(0, 1)
            for row, (example, eligible_size) in enumerate(zip(batch, eligible_sizes, strict=True)):
                scores[row, eligible_size:] = float("-inf")
                for item_id in set(example.history_item_ids).difference(example.positive_item_ids):
                    position = position_by_item.get(item_id)
                    if position is not None and position < eligible_size:
                        scores[row, position] = float("-inf")
            top_count = min(100, max_eligible)
            top_scores, top_positions = torch.topk(scores, top_count, dim=1)
            if device.type == "mps":
                torch.mps.synchronize()
            search_seconds += time.perf_counter() - started

        for row, example in enumerate(batch):
            ranked = [
                catalog.item_ids[int(position)]
                for score, position in zip(
                    top_scores[row].cpu().tolist(),
                    top_positions[row].cpu().tolist(),
                    strict=True,
                )
                if score != float("-inf")
            ]
            recall_20 += recall_at_k(ranked, example.positive_item_ids, 20)
            recall_100 += recall_at_k(ranked, example.positive_item_ids, 100)
            mrr_20 += reciprocal_rank_at_k(ranked, example.positive_item_ids, 20)
            recommended.update(ranked[:100])
            retrieved = set(ranked[:100])
            for target in example.positive_item_ids:
                bucket = _bucket(training_popularity[target], thresholds)
                slice_targets[bucket] += 1
                slice_hits[bucket] += target in retrieved

    query_count = len(examples)
    metrics: dict[str, float | int] = {
        "query_count": query_count,
        "recall@20": recall_20 / query_count,
        "recall@100": recall_100 / query_count,
        "mrr@20": mrr_20 / query_count,
        "coverage@100": len(recommended) / len(catalog.item_ids),
        "exact_search_ms_per_query": 1000 * search_seconds / query_count,
    }
    for bucket in ("head", "mid", "tail"):
        denominator = slice_targets[bucket]
        metrics[f"{bucket}_target_recall@100"] = (
            slice_hits[bucket] / denominator if denominator else 0.0
        )
        metrics[f"{bucket}_target_count"] = denominator
    return metrics
