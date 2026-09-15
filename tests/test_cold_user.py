from pathlib import Path

import numpy as np

from recforge.cold_user import (
    ColdUserConfig,
    _coverage,
    build_propensity_dataset,
    score_all_items,
    train_propensity_head,
)
from recforge.data.features import ItemFeatureTable
from recforge.data.mind import iter_mind_behaviors


def _behaviors(path: Path) -> Path:
    path.write_text(
        "1\tU1\t11/13/2019 1:00:00 AM\t\tN0-1 N1-0 N2-0\n"
        "2\tU2\t11/13/2019 2:00:00 AM\tN0\tN1-1 N0-0 N2-0\n",
        encoding="utf-8",
    )
    return path


def _table() -> ItemFeatureTable:
    return ItemFeatureTable(("N0", "N1", "N2"), np.eye(3, dtype=np.float32))


def _config() -> ColdUserConfig:
    return ColdUserConfig(
        feature_artifact="features",
        fit_behaviors="fit",
        calibration_behaviors="calibration",
        dev_behaviors="dev",
        train_news="train-news",
        dev_news="dev-news",
        base_prediction="base",
        epochs=2,
        batch_size=3,
        bootstrap_resamples=10,
    )


def test_propensity_training_and_item_scoring_are_deterministic(tmp_path: Path) -> None:
    dataset = build_propensity_dataset(_behaviors(tmp_path / "behaviors.tsv"), _table())
    first, losses, durations, weight = train_propensity_head(_config(), dataset, _table())
    second, _, _, _ = train_propensity_head(_config(), dataset, _table())
    assert len(losses) == len(durations) == 2
    assert weight == 2.0
    assert np.array_equal(score_all_items(first, _table()), score_all_items(second, _table()))


def test_coverage_uses_displayed_slice_catalog(tmp_path: Path) -> None:
    behaviors = list(iter_mind_behaviors(_behaviors(tmp_path / "behaviors.tsv")))
    predictions = [[3.0, 2.0, 1.0], [3.0, 1.0, 2.0]]
    assert _coverage(behaviors, predictions, k=1) == 2 / 3
    assert _coverage(behaviors, predictions, k=5) == 1.0
