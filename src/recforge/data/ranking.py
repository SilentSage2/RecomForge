"""Leakage-safe impression-local examples and padded batches for L1 ranking."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from recforge.data.features import ItemFeatureTable
from recforge.data.mind import MindNews, iter_mind_behaviors, iter_mind_news
from recforge.data.text import Vocabulary, encode_title


@dataclass(frozen=True, slots=True)
class RankingExample:
    query_id: str
    history_item_ids: tuple[str, ...]
    candidate_item_ids: tuple[str, ...]
    target_index: int = 0

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        if len(self.candidate_item_ids) < 2:
            raise ValueError("ranking examples require a positive and at least one negative")
        if not 0 <= self.target_index < len(self.candidate_item_ids):
            raise ValueError("target_index is outside the candidate list")


@dataclass(frozen=True, slots=True)
class TitleTable:
    item_ids: tuple[str, ...]
    token_ids: NDArray[np.int64]
    attention_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.attention_mask.shape != self.token_ids.shape:
            raise ValueError("title tokens and masks must have equal [items, tokens] shapes")
        if self.token_ids.shape[0] != len(self.item_ids):
            raise ValueError("item IDs and title rows must have equal length")
        if self.token_ids.dtype != np.int64 or self.attention_mask.dtype != np.bool_:
            raise ValueError("title token IDs must be int64 and masks must be bool")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("title table item IDs must be unique")

    @property
    def max_title_tokens(self) -> int:
        return int(self.token_ids.shape[1])

    def row_by_item_id(self) -> dict[str, int]:
        return {item_id: row for row, item_id in enumerate(self.item_ids)}


@dataclass(frozen=True, slots=True)
class RankingBatch:
    history_token_ids: NDArray[np.int64]
    history_token_mask: NDArray[np.bool_]
    history_item_mask: NDArray[np.bool_]
    candidate_token_ids: NDArray[np.int64]
    candidate_token_mask: NDArray[np.bool_]
    target_indices: NDArray[np.int64]
    query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.query_ids)
        if self.history_token_ids.ndim != 3 or self.candidate_token_ids.ndim != 3:
            raise ValueError("history and candidate title IDs must be rank-three")
        if self.history_token_mask.shape != self.history_token_ids.shape:
            raise ValueError("history token mask shape mismatch")
        if self.candidate_token_mask.shape != self.candidate_token_ids.shape:
            raise ValueError("candidate token mask shape mismatch")
        if self.history_item_mask.shape != self.history_token_ids.shape[:2]:
            raise ValueError("history item mask shape mismatch")
        if self.history_token_ids.shape[0] != batch_size:
            raise ValueError("history batch size mismatch")
        if self.candidate_token_ids.shape[0] != batch_size:
            raise ValueError("candidate batch size mismatch")
        if self.target_indices.shape != (batch_size,):
            raise ValueError("target indices must have shape [batch]")


@dataclass(frozen=True, slots=True)
class FeatureRankingBatch:
    history_features: NDArray[np.float32]
    history_item_mask: NDArray[np.bool_]
    candidate_features: NDArray[np.float32]
    target_indices: NDArray[np.int64]
    query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.query_ids)
        if self.history_features.ndim != 3 or self.candidate_features.ndim != 3:
            raise ValueError("history and candidate features must be rank-three")
        if self.history_features.shape[0] != batch_size:
            raise ValueError("history batch size mismatch")
        if self.candidate_features.shape[0] != batch_size:
            raise ValueError("candidate batch size mismatch")
        if self.history_features.shape[2] != self.candidate_features.shape[2]:
            raise ValueError("history and candidate feature dimensions must match")
        if self.history_item_mask.shape != self.history_features.shape[:2]:
            raise ValueError("history item mask shape mismatch")
        if self.target_indices.shape != (batch_size,):
            raise ValueError("target indices must have shape [batch]")


def build_title_table(
    news_paths: list[Path], vocabulary: Vocabulary, *, max_title_tokens: int
) -> TitleTable:
    if not news_paths:
        raise ValueError("at least one news file is required")
    news_by_id: dict[str, MindNews] = {}
    for path in news_paths:
        for news in iter_mind_news(path):
            existing = news_by_id.get(news.news_id)
            if existing is not None and existing != news:
                raise ValueError(f"conflicting metadata for item {news.news_id!r}")
            news_by_id[news.news_id] = news
    if not news_by_id:
        raise ValueError("news inputs contain no items")

    item_ids = tuple(sorted(news_by_id))
    encoded = [
        encode_title(news_by_id[item_id].title, vocabulary, max_title_tokens=max_title_tokens)
        for item_id in item_ids
    ]
    return TitleTable(
        item_ids=item_ids,
        token_ids=np.asarray([title.token_ids for title in encoded], dtype=np.int64),
        attention_mask=np.asarray([title.attention_mask for title in encoded], dtype=np.bool_),
    )


def load_ranking_examples(
    behavior_path: Path,
    *,
    negative_count: int,
    seed: int,
    epoch: int,
    namespace: str = "train",
    max_examples: int | None = None,
) -> tuple[RankingExample, ...]:
    """Create one example per click using negatives from that same impression only."""
    if negative_count <= 0:
        raise ValueError("negative_count must be positive")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    examples: list[RankingExample] = []
    for behavior in iter_mind_behaviors(behavior_path):
        positives = [
            item_id
            for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        ]
        negatives = [
            item_id
            for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 0
        ]
        if not positives or not negatives:
            continue
        for positive_index, positive_item_id in enumerate(positives):
            rng = random.Random(f"{seed}:{epoch}:{behavior.impression_id}:{positive_index}")
            if len(negatives) >= negative_count:
                sampled_negatives = rng.sample(negatives, negative_count)
            else:
                sampled_negatives = [rng.choice(negatives) for _ in range(negative_count)]
            examples.append(
                RankingExample(
                    query_id=f"{namespace}:{behavior.impression_id}:{positive_index}",
                    history_item_ids=behavior.history_item_ids,
                    candidate_item_ids=(positive_item_id, *sampled_negatives),
                )
            )
            if max_examples is not None and len(examples) == max_examples:
                return tuple(examples)
    return tuple(examples)


def iter_ranking_batches(
    examples: tuple[RankingExample, ...],
    table: TitleTable,
    *,
    batch_size: int,
    max_history_items: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> Iterator[RankingBatch]:
    if batch_size <= 0 or max_history_items <= 0:
        raise ValueError("batch_size and max_history_items must be positive")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if not examples:
        return
    candidate_count = len(examples[0].candidate_item_ids)
    if any(len(example.candidate_item_ids) != candidate_count for example in examples):
        raise ValueError("all ranking examples must have equal candidate counts")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    row_by_item = table.row_by_item_id()

    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start : start + batch_size]]
        current_size = len(selected)
        title_width = table.max_title_tokens
        history_ids = np.zeros((current_size, max_history_items, title_width), dtype=np.int64)
        history_token_mask = np.zeros_like(history_ids, dtype=np.bool_)
        history_item_mask = np.zeros((current_size, max_history_items), dtype=np.bool_)
        candidate_ids = np.empty((current_size, candidate_count, title_width), dtype=np.int64)
        candidate_token_mask = np.empty_like(candidate_ids, dtype=np.bool_)

        for batch_row, example in enumerate(selected):
            known_history = [
                row_by_item[item_id]
                for item_id in example.history_item_ids
                if item_id in row_by_item
            ][-max_history_items:]
            if known_history:
                history_count = len(known_history)
                history_ids[batch_row, :history_count] = table.token_ids[known_history]
                history_token_mask[batch_row, :history_count] = table.attention_mask[known_history]
                history_item_mask[batch_row, :history_count] = True
            try:
                candidate_rows = [row_by_item[item] for item in example.candidate_item_ids]
            except KeyError as error:
                raise ValueError(
                    f"query {example.query_id} candidate {error.args[0]!r} has no title"
                ) from error
            candidate_ids[batch_row] = table.token_ids[candidate_rows]
            candidate_token_mask[batch_row] = table.attention_mask[candidate_rows]

        yield RankingBatch(
            history_token_ids=history_ids,
            history_token_mask=history_token_mask,
            history_item_mask=history_item_mask,
            candidate_token_ids=candidate_ids,
            candidate_token_mask=candidate_token_mask,
            target_indices=np.asarray(
                [example.target_index for example in selected], dtype=np.int64
            ),
            query_ids=tuple(example.query_id for example in selected),
        )


def iter_feature_ranking_batches(
    examples: tuple[RankingExample, ...],
    table: ItemFeatureTable,
    *,
    batch_size: int,
    max_history_items: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> Iterator[FeatureRankingBatch]:
    """Materialize ranking batches from an immutable item-feature artifact."""
    if batch_size <= 0 or max_history_items <= 0:
        raise ValueError("batch_size and max_history_items must be positive")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if not examples:
        return
    candidate_count = len(examples[0].candidate_item_ids)
    if any(len(example.candidate_item_ids) != candidate_count for example in examples):
        raise ValueError("all ranking examples must have equal candidate counts")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    row_by_item = table.row_by_item_id()

    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start : start + batch_size]]
        current_size = len(selected)
        history = np.zeros((current_size, max_history_items, table.dimension), dtype=np.float32)
        history_mask = np.zeros((current_size, max_history_items), dtype=np.bool_)
        candidates = np.empty((current_size, candidate_count, table.dimension), dtype=np.float32)

        for batch_row, example in enumerate(selected):
            history_rows = [
                row_by_item[item_id]
                for item_id in example.history_item_ids
                if item_id in row_by_item
            ][-max_history_items:]
            if history_rows:
                history_count = len(history_rows)
                history[batch_row, :history_count] = table.features[history_rows]
                history_mask[batch_row, :history_count] = True
            try:
                candidate_rows = [row_by_item[item] for item in example.candidate_item_ids]
            except KeyError as error:
                raise ValueError(
                    f"query {example.query_id} candidate {error.args[0]!r} has no features"
                ) from error
            candidates[batch_row] = table.features[candidate_rows]

        yield FeatureRankingBatch(
            history_features=history,
            history_item_mask=history_mask,
            candidate_features=candidates,
            target_indices=np.asarray(
                [example.target_index for example in selected], dtype=np.int64
            ),
            query_ids=tuple(example.query_id for example in selected),
        )
