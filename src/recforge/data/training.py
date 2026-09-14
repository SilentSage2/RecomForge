"""Deterministic training batches for MIND two-tower retrieval."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from recforge.data.features import ItemFeatureTable, aggregate_history_features
from recforge.data.protocol import MindRetrievalExample, iter_positive_retrieval_examples


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
