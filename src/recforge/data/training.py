"""Deterministic training batches for MIND two-tower retrieval."""

from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from recforge.data.features import ItemFeatureTable, aggregate_history_features
from recforge.data.protocol import (
    MindRetrievalExample,
    TemporalCatalogIndex,
    iter_positive_retrieval_examples,
)


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    user_features: NDArray[np.float32]
    positive_item_features: NDArray[np.float32]
    positive_item_rows: NDArray[np.int64]
    query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = self.user_features.shape[0]
        if self.user_features.ndim != 2 or self.positive_item_features.ndim != 2:
            raise ValueError("batch features must be rank-two")
        if self.positive_item_features.shape != self.user_features.shape:
            raise ValueError("user and item feature batches must have equal shapes")
        if self.positive_item_rows.shape != (batch_size,):
            raise ValueError("positive_item_rows must have shape [batch]")
        if len(self.query_ids) != batch_size:
            raise ValueError("query IDs and feature rows must have equal length")


@dataclass(frozen=True, slots=True)
class UniformNegativePool:
    features: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]


def load_training_examples(
    behavior_path: Path, *, namespace: str = "train"
) -> tuple[MindRetrievalExample, ...]:
    return tuple(iter_positive_retrieval_examples(behavior_path, namespace=namespace))


def _choose_positive(example: MindRetrievalExample, *, seed: int, epoch: int) -> str:
    positives = sorted(example.positive_item_ids)
    digest = hashlib.blake2b(
        f"{seed}:{epoch}:{example.query_id}".encode(),
        digest_size=8,
        person=b"recforge",
    ).digest()
    return positives[int.from_bytes(digest, "little") % len(positives)]


def iter_feature_batches(
    examples: tuple[MindRetrievalExample, ...],
    table: ItemFeatureTable,
    *,
    batch_size: int,
    max_history_items: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> Iterator[FeatureBatch]:
    """Yield all examples once, with deterministic order and one positive per query."""
    if batch_size <= 1:
        raise ValueError("batch_size must exceed one for in-batch negatives")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    row_by_item_id = table.row_by_item_id()

    for start in range(0, len(order), batch_size):
        batch_examples = [examples[index] for index in order[start : start + batch_size]]
        if len(batch_examples) < 2:
            continue
        user_features: list[NDArray[np.float32]] = []
        item_rows: list[int] = []
        query_ids: list[str] = []
        for example in batch_examples:
            positive_item_id = _choose_positive(example, seed=seed, epoch=epoch)
            try:
                item_row = row_by_item_id[positive_item_id]
            except KeyError as error:
                raise ValueError(
                    f"query {example.query_id} target {positive_item_id!r} has no features"
                ) from error
            user_features.append(
                aggregate_history_features(
                    example.history_item_ids,
                    table,
                    max_history_items=max_history_items,
                    row_by_item_id=row_by_item_id,
                )
            )
            item_rows.append(item_row)
            query_ids.append(example.query_id)
        rows = np.asarray(item_rows, dtype=np.int64)
        yield FeatureBatch(
            user_features=np.stack(user_features).astype(np.float32, copy=False),
            positive_item_features=np.asarray(table.features[rows], dtype=np.float32),
            positive_item_rows=rows,
            query_ids=tuple(query_ids),
        )


def sample_uniform_negative_pool(
    examples: tuple[MindRetrievalExample, ...],
    table: ItemFeatureTable,
    catalog: TemporalCatalogIndex,
    *,
    seed: int,
    epoch: int,
) -> UniformNegativePool:
    """Sample one legal negative per query, then share and mask the resulting pool."""
    if not examples:
        raise ValueError("examples must not be empty")
    row_by_item = table.row_by_item_id()
    position_by_item = {item_id: position for position, item_id in enumerate(catalog.item_ids)}
    sampled_ids: list[str] = []
    for example in examples:
        eligible_size = bisect_right(catalog.observed_at, example.local_timestamp)
        excluded = set(example.history_item_ids).union(example.positive_item_ids)
        rng = random.Random(f"{seed}:{epoch}:{example.query_id}")
        sampled: str | None = None
        for _ in range(100):
            candidate = catalog.item_ids[rng.randrange(eligible_size)]
            if candidate not in excluded and candidate in row_by_item:
                sampled = candidate
                break
        if sampled is None:
            sampled = next(
                (
                    item_id
                    for item_id in catalog.item_ids[:eligible_size]
                    if item_id not in excluded and item_id in row_by_item
                ),
                None,
            )
        if sampled is None:
            raise ValueError(f"query {example.query_id} has no legal uniform negative")
        sampled_ids.append(sampled)

    valid_mask = np.ones((len(examples), len(sampled_ids)), dtype=np.bool_)
    for row, example in enumerate(examples):
        eligible_size = bisect_right(catalog.observed_at, example.local_timestamp)
        excluded = set(example.history_item_ids).union(example.positive_item_ids)
        for column, item_id in enumerate(sampled_ids):
            valid_mask[row, column] = (
                position_by_item[item_id] < eligible_size and item_id not in excluded
            )
    rows = np.asarray([row_by_item[item_id] for item_id in sampled_ids], dtype=np.int64)
    return UniformNegativePool(
        features=np.asarray(table.features[rows], dtype=np.float32),
        valid_mask=valid_mask,
    )
