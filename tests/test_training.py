from datetime import datetime

import numpy as np
import pytest

from recforge.data.features import ItemFeatureTable
from recforge.data.protocol import MindRetrievalExample, TemporalCatalogIndex
from recforge.data.training import iter_feature_batches, sample_uniform_negative_pool


def _example(query_id: str, positives: set[str]) -> MindRetrievalExample:
    return MindRetrievalExample(
        query_id=query_id,
        user_id="user",
        local_timestamp=datetime(2019, 11, 11),
        history_item_ids=("i0",),
        positive_item_ids=frozenset(positives),
    )


def _table() -> ItemFeatureTable:
    return ItemFeatureTable(
        item_ids=("i0", "i1", "i2", "i3"),
        features=np.eye(4, dtype=np.float32),
    )


def test_batches_are_deterministic_and_cover_examples() -> None:
    examples = tuple(_example(f"q{i}", {"i1", "i2"}) for i in range(5))
    first = list(
        iter_feature_batches(
            examples, _table(), batch_size=3, max_history_items=5, seed=17, epoch=2
        )
    )
    second = list(
        iter_feature_batches(
            examples, _table(), batch_size=3, max_history_items=5, seed=17, epoch=2
        )
    )

    assert [batch.query_ids for batch in first] == [batch.query_ids for batch in second]
    assert [batch.positive_item_rows.tolist() for batch in first] == [
        batch.positive_item_rows.tolist() for batch in second
    ]
    assert sum(len(batch.query_ids) for batch in first) == len(examples)


def test_singleton_tail_is_skipped_for_in_batch_training() -> None:
    examples = tuple(_example(f"q{i}", {"i1"}) for i in range(5))
    batches = list(
        iter_feature_batches(
            examples,
            _table(),
            batch_size=2,
            max_history_items=5,
            seed=0,
            epoch=0,
            shuffle=False,
        )
    )
    assert [len(batch.query_ids) for batch in batches] == [2, 2]


def test_batch_requires_target_features() -> None:
    examples = (_example("q", {"missing"}), _example("q2", {"i1"}))
    with pytest.raises(ValueError, match="has no features"):
        list(
            iter_feature_batches(
                examples,
                _table(),
                batch_size=2,
                max_history_items=5,
                seed=0,
                epoch=0,
            )
        )


def test_uniform_pool_is_deterministic_and_has_a_valid_negative_per_user() -> None:
    examples = (_example("q1", {"i1"}), _example("q2", {"i2"}))
    timestamp = examples[0].local_timestamp
    catalog = TemporalCatalogIndex(
        observed_at=(timestamp, timestamp, timestamp, timestamp),
        item_ids=("i0", "i1", "i2", "i3"),
    )
    first = sample_uniform_negative_pool(examples, _table(), catalog, seed=4, epoch=0)
    second = sample_uniform_negative_pool(examples, _table(), catalog, seed=4, epoch=0)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.valid_mask, second.valid_mask)
    assert first.valid_mask.any(axis=1).all()
