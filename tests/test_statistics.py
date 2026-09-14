from pathlib import Path

import numpy as np
import pytest

from recforge.statistics import paired_bootstrap, per_impression_metrics


def test_paired_bootstrap_is_deterministic_and_detects_uniform_improvement() -> None:
    baseline = np.zeros((20, 4), dtype=np.float64)
    candidate = np.full((20, 4), 0.1, dtype=np.float64)
    first = paired_bootstrap(baseline, candidate, resamples=100, seed=7)
    second = paired_bootstrap(baseline, candidate, resamples=100, seed=7)

    assert first == second
    assert first["auc"]["mean_difference"] == pytest.approx(0.1)
    assert first["auc"]["confidence_lower"] > 0


def test_per_impression_metrics_reads_candidate_aligned_ranks(tmp_path: Path) -> None:
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\t\tN1-1 N2-0 N3-0\n"
        "2\tU2\t11/11/2019 9:01:00 AM\tN1\tN2-0 N3-1\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.txt"
    prediction.write_text("1 [1,2,3]\n2 [2,1]\n", encoding="utf-8")

    metrics = per_impression_metrics(behaviors, prediction)
    assert metrics.shape == (2, 4)
    assert np.array_equal(metrics, np.ones((2, 4)))
