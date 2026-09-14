from pathlib import Path

import numpy as np
import pytest
import torch

from recforge.data.features import ItemFeatureTable
from recforge.experiment import R1ExperimentConfig, evaluate_logged_impressions, train_model


def _config() -> R1ExperimentConfig:
    return R1ExperimentConfig(
        feature_artifact="features",
        train_behaviors="train.tsv",
        eval_behaviors="eval.tsv",
        seed=3,
        epochs=8,
        batch_size=2,
        learning_rate=0.02,
        embedding_dim=4,
        hidden_dim=8,
        device="cpu",
    )


def _write_behaviors(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "1\tU1\t11/11/2019 9:00:00 AM\ti0\ti0-1 i1-0 i2-0",
                "2\tU2\t11/11/2019 9:01:00 AM\ti1\ti1-1 i2-0 i3-0",
                "3\tU3\t11/11/2019 9:02:00 AM\ti2\ti2-1 i3-0 i0-0",
                "4\tU4\t11/11/2019 9:03:00 AM\ti3\ti3-1 i0-0 i1-0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_tiny_train_and_logged_evaluation(tmp_path: Path) -> None:
    behaviors = _write_behaviors(tmp_path / "behaviors.tsv")
    table = ItemFeatureTable(
        item_ids=("i0", "i1", "i2", "i3"),
        features=np.eye(4, dtype=np.float32),
    )
    model, losses, trained_pairs = train_model(
        _config(),
        table,
        behavior_path=behaviors,
        max_history_items=5,
        device=torch.device("cpu"),
    )
    metrics = evaluate_logged_impressions(
        model,
        table,
        behavior_path=behaviors,
        max_history_items=5,
        device=torch.device("cpu"),
        max_queries=None,
    )

    assert trained_pairs == 32
    assert losses[-1] < losses[0]
    assert metrics["query_count"] == 4
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ndcg@5"] <= 1.0


def test_config_rejects_singleton_batches() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        R1ExperimentConfig("features", "train", "eval", batch_size=1)
