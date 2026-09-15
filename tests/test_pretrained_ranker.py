import numpy as np
import torch

from recforge.data.features import ItemFeatureTable
from recforge.data.ranking import RankingExample, iter_feature_ranking_batches
from recforge.models.nrms import sampled_softmax_loss
from recforge.models.pretrained_ranker import FrozenFeatureRanker


def _table() -> ItemFeatureTable:
    return ItemFeatureTable(
        ("old", "recent", "positive", "negative"),
        np.eye(4, dtype=np.float32),
    )


def test_feature_batches_use_recent_known_history_and_are_deterministic() -> None:
    examples = (
        RankingExample(
            query_id="q1",
            history_item_ids=("old", "missing", "recent"),
            candidate_item_ids=("positive", "negative"),
        ),
    )
    first = next(
        iter_feature_ranking_batches(
            examples,
            _table(),
            batch_size=1,
            max_history_items=2,
            seed=7,
            epoch=0,
            shuffle=False,
        )
    )
    second = next(
        iter_feature_ranking_batches(
            examples,
            _table(),
            batch_size=1,
            max_history_items=2,
            seed=7,
            epoch=0,
            shuffle=False,
        )
    )
    assert first.query_ids == ("q1",)
    assert first.history_item_mask.tolist() == [[True, True]]
    assert np.array_equal(first.history_features, second.history_features)
    assert np.array_equal(first.candidate_features, second.candidate_features)


def test_frozen_feature_ranker_handles_empty_history_and_backpropagates() -> None:
    model = FrozenFeatureRanker(input_dim=4, embedding_dim=2)
    history = torch.zeros((2, 2, 4))
    history[0, 0, 0] = 1
    mask = torch.tensor([[True, False], [False, False]])
    candidates = torch.eye(4)[:2].reshape(2, 1, 4).repeat(1, 2, 1)
    candidates[:, 1] = torch.roll(candidates[:, 1], shifts=1, dims=1)
    scores = model(history, mask, candidates)
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()
    loss = sampled_softmax_loss(scores, torch.zeros(2, dtype=torch.long))
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 18
