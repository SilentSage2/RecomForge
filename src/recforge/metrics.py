"""Dependency-free retrieval and ranking metrics with explicit edge-case behavior."""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _unique_prefix(ranked_item_ids: Sequence[str], k: int) -> tuple[str, ...]:
    _validate_k(k)
    seen: set[str] = set()
    result: list[str] = []
    for item_id in ranked_item_ids:
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
        if len(result) == k:
            break
    return tuple(result)


def recall_at_k(
    ranked_item_ids: Sequence[str], relevant_item_ids: Collection[str], k: int
) -> float:
    """Fraction of unique relevant items retrieved in the first K unique results."""
    relevant = set(relevant_item_ids)
    if not relevant:
        raise ValueError("relevant_item_ids must not be empty")
    hits = relevant.intersection(_unique_prefix(ranked_item_ids, k))
    return len(hits) / len(relevant)


def reciprocal_rank_at_k(
    ranked_item_ids: Sequence[str], relevant_item_ids: Collection[str], k: int
) -> float:
    """Reciprocal rank of the first relevant item, or zero when no item is found."""
    relevant = set(relevant_item_ids)
    if not relevant:
        raise ValueError("relevant_item_ids must not be empty")
    for rank, item_id in enumerate(_unique_prefix(ranked_item_ids, k), start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(
    ranked_item_ids: Sequence[str], relevant_item_ids: Collection[str], k: int
) -> float:
    """MIND-style mean reciprocal rank over every relevant item in the impression.

    Unlike first-relevant reciprocal rank, the numerator sums reciprocal ranks for
    every relevant item found in the top K. The denominator remains the total
    number of relevant items, including relevant items outside K.
    """
    relevant = set(relevant_item_ids)
    if not relevant:
        raise ValueError("relevant_item_ids must not be empty")
    reciprocal_sum = sum(
        1.0 / rank
        for rank, item_id in enumerate(_unique_prefix(ranked_item_ids, k), start=1)
        if item_id in relevant
    )
    return reciprocal_sum / len(relevant)


def ndcg_at_k(ranked_item_ids: Sequence[str], relevant_item_ids: Collection[str], k: int) -> float:
    """Binary normalized discounted cumulative gain at K."""
    relevant = set(relevant_item_ids)
    if not relevant:
        raise ValueError("relevant_item_ids must not be empty")
    ranked = _unique_prefix(ranked_item_ids, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked, start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


def catalog_coverage_at_k(
    ranked_lists: Iterable[Sequence[str]], catalog_item_ids: Collection[str], k: int
) -> float:
    """Fraction of the declared catalog appearing in top-K recommendations."""
    _validate_k(k)
    catalog = set(catalog_item_ids)
    if not catalog:
        raise ValueError("catalog_item_ids must not be empty")
    recommended = {item_id for ranked in ranked_lists for item_id in _unique_prefix(ranked, k)}
    unknown = recommended.difference(catalog)
    if unknown:
        raise ValueError(f"recommendations contain items outside the catalog: {sorted(unknown)}")
    return len(recommended) / len(catalog)


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Pairwise binary AUC; tied positive/negative scores receive half credit."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one positive and one negative")
    credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return credit / (len(positives) * len(negatives))
