from datetime import datetime
from pathlib import Path

import numpy as np

from recforge.calibration import (
    fit_monotonic_platt,
    probability_metrics,
    split_calibration_behaviors,
)


def test_temporal_calibration_split_preserves_relative_source_order(tmp_path: Path) -> None:
    source = tmp_path / "behaviors.tsv"
    source.write_text(
        "1\tU1\t11/14/2019 1:00:00 AM\tN0\tN0-1 N1-0\n"
        "2\tU2\t11/13/2019 1:00:00 AM\tN0\tN0-1 N1-0\n"
        "3\tU3\t11/14/2019 2:00:00 AM\tN0\tN0-1 N1-0\n",
        encoding="utf-8",
    )
    manifest = split_calibration_behaviors(
        source,
        tmp_path / "split",
        cutoff=datetime.fromisoformat("2019-11-14T00:00:00"),
        repository_root=Path.cwd(),
    )
    assert (tmp_path / "split/fit.tsv").read_text().startswith("2\tU2")
    calibration_lines = (tmp_path / "split/calibration.tsv").read_text().splitlines()
    assert [line.split("\t")[0] for line in calibration_lines] == ["1", "3"]
    assert manifest["source_timestamp_inversion_count"] == 1


def test_platt_scaling_is_positive_and_improves_simple_probabilities() -> None:
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    scale, bias = fit_monotonic_platt(scores, labels)
    assert scale > 0
    raw = 1 / (1 + np.exp(-scores))
    calibrated = 1 / (1 + np.exp(-(scale * scores + bias)))
    assert probability_metrics(labels, calibrated)["nll"] < probability_metrics(labels, raw)["nll"]
