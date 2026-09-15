from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from recforge.data.features import ItemFeatureTable
from recforge.l2_experiment import (
    L2ExperimentConfig,
    evaluate_frozen_ranker,
    train_frozen_ranker,
)


def _write_behaviors(path: Path) -> Path:
    path.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN0\tN0-1 N1-0 N2-0\n"
        "2\tU2\t11/11/2019 9:01:00 AM\tN1\tN1-1 N0-0 N2-0\n"
        "3\tU3\t11/11/2019 9:02:00 AM\tN2\tN2-1 N0-0 N1-0\n",
        encoding="utf-8",
    )
    return path


def _table() -> ItemFeatureTable:
    return ItemFeatureTable(
        ("N0", "N1", "N2"),
        np.asarray(
            [[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.1, 0.0], [0.1, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
    )


def _config(epochs: int = 2) -> L2ExperimentConfig:
    return L2ExperimentConfig(
        feature_artifact="features",
        train_behaviors="train",
        eval_behaviors="eval",
        seed=17,
        epochs=epochs,
        batch_size=2,
        embedding_dim=2,
        negative_count=2,
        max_history_items=2,
        device="cpu",
    )


def test_tiny_l2_train_and_official_evaluation(tmp_path: Path) -> None:
    behaviors = _write_behaviors(tmp_path / "behaviors.tsv")
    model, losses, trained, epoch_seconds, resumed_from = train_frozen_ranker(
        _config(), _table(), behavior_path=behaviors, device=torch.device("cpu")
    )
    metrics, predictions = evaluate_frozen_ranker(
        model,
        _table(),
        behavior_path=behaviors,
        max_history_items=2,
        device=torch.device("cpu"),
        max_impressions=None,
        batch_size=2,
    )
    assert trained == 6
    assert len(losses) == len(epoch_seconds) == 2
    assert all(value > 0 for value in epoch_seconds)
    assert resumed_from == 0
    assert metrics["query_count"] == 3
    assert 0.0 <= metrics["auc"] <= 1.0
    assert len(predictions) == 3


def test_l2_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    behaviors = _write_behaviors(tmp_path / "behaviors.tsv")
    config = _config()
    uninterrupted, full_losses, full_count, _, _ = train_frozen_ranker(
        config, _table(), behavior_path=behaviors, device=torch.device("cpu")
    )
    checkpoint = tmp_path / "training-state.pt"
    train_frozen_ranker(
        replace(config, epochs=1),
        _table(),
        behavior_path=behaviors,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        resume_metadata={"split": "fixture-v1"},
    )
    resumed, resumed_losses, resumed_count, _, start_epoch = train_frozen_ranker(
        config,
        _table(),
        behavior_path=behaviors,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        resume_metadata={"split": "fixture-v1"},
    )
    assert start_epoch == 1
    assert resumed_losses == full_losses
    assert resumed_count == full_count
    for expected, actual in zip(
        uninterrupted.state_dict().values(), resumed.state_dict().values(), strict=True
    ):
        assert torch.equal(expected, actual)
