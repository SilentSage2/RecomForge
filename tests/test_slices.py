from pathlib import Path

import numpy as np

from recforge.data.mind import iter_mind_behaviors
from recforge.slices import _evaluate_family, build_slice_labels


def _write_news(path: Path, suffix: str = "") -> Path:
    path.write_text(
        f"N0\tcat\tsub\tcommon title\tabs\thttps://a.test/0{suffix}\t[]\t[]\n"
        f"N1\tcat\tsub\trare token\tabs\thttps://b.test/1{suffix}\t[]\t[]\n"
        f"N2\tcat\tsub\tnew title\tabs\thttps://a.test/2{suffix}\t[]\t[]\n",
        encoding="utf-8",
    )
    return path


def test_build_slice_labels_uses_training_counts_and_prior_exposure(tmp_path: Path) -> None:
    train_news = _write_news(tmp_path / "train-news.tsv")
    dev_news = _write_news(tmp_path / "dev-news.tsv")
    train_behaviors = tmp_path / "train.tsv"
    train_behaviors.write_text(
        "1\tU1\t11/14/2019 9:00:00 AM\tN0\tN0-1 N1-0\n",
        encoding="utf-8",
    )
    dev_path = tmp_path / "dev.tsv"
    dev_path.write_text(
        "2\tU2\t11/15/2019 9:00:00 AM\t\tN0-1 N2-0\n"
        "3\tU3\t11/15/2019 10:00:00 AM\tN0 N1\tN2-1 N1-0\n",
        encoding="utf-8",
    )
    behaviors = tuple(iter_mind_behaviors(dev_path))
    labels, metadata = build_slice_labels(train_behaviors, train_news, behaviors, dev_news)
    assert labels["history_length"] == ["0", "1-9"]
    assert labels["positive_exposure_age"] == ["6-24h", "0-6h"]
    assert labels["positive_item_popularity"] == ["tail", "cold"]
    assert metadata["recency_interpretation"] == "logged_candidate_exposure_age_proxy"


def test_family_report_contains_multiplicity_adjusted_intervals() -> None:
    baseline = np.zeros((4, 4), dtype=np.float64)
    candidate = np.ones((4, 4), dtype=np.float64)
    report = _evaluate_family(
        "fixture",
        ["a", "a", "b", "b"],
        baseline,
        candidate,
        resamples=20,
        seed=7,
    )
    assert report["slice_count"] == 2
    for item in report["slices"].values():
        metric = item["metrics"]["auc"]
        assert metric["mean_difference"] == 1.0
        assert metric["confidence_familywise_bonferroni"] == [1.0, 1.0]
