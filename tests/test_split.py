from dataclasses import replace
from datetime import UTC, datetime

import pytest

from recforge.schemas import Interaction
from recforge.split import temporal_split, validate_retrieval_query
from recforge.synthetic import build_synthetic_dataset


def test_temporal_split_uses_half_open_boundaries() -> None:
    dataset = build_synthetic_dataset()
    train_end = datetime(2026, 1, 4, tzinfo=UTC)
    validation_end = datetime(2026, 1, 5, tzinfo=UTC)
    split = temporal_split(dataset.training_interactions, train_end, validation_end)
    assert len(split.train) == 3
    assert len(split.validation) == 2
    assert not split.test


def test_query_rejects_future_history() -> None:
    dataset = build_synthetic_dataset()
    query = dataset.queries[0]
    future = Interaction(query.user_id, "article-a", query.timestamp)
    invalid = replace(query, history=query.history + (future,))
    with pytest.raises(ValueError, match="future or simultaneous"):
        validate_retrieval_query(invalid, {item.item_id: item for item in dataset.items})


def test_query_rejects_future_candidate() -> None:
    dataset = build_synthetic_dataset()
    query = dataset.queries[0]
    invalid = replace(
        query,
        candidate_item_ids=query.candidate_item_ids + ("article-future",),
    )
    with pytest.raises(ValueError, match="unavailable"):
        validate_retrieval_query(invalid, {item.item_id: item for item in dataset.items})
