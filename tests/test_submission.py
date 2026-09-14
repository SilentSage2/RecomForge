from pathlib import Path

import pytest

from recforge.data.submission import (
    format_prediction_line,
    ranks_from_scores,
    validate_prediction_file,
    write_prediction_file,
)


def test_prediction_line_rejects_empty_ranks() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        format_prediction_line("1", [])


def _behaviors(path: Path) -> Path:
    path.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\t\tN1 N2 N3\n2\tU2\t11/11/2019 9:01:00 AM\tN1\tN2 N4\n",
        encoding="utf-8",
    )
    return path


def test_scores_become_candidate_aligned_stable_ranks() -> None:
    assert ranks_from_scores([0.2, 0.9, 0.2]) == (2, 1, 3)
    assert format_prediction_line("7", (2, 1, 3)) == "7 [2,1,3]\n"


def test_writer_and_validator_accept_exact_submission(tmp_path: Path) -> None:
    behaviors = _behaviors(tmp_path / "behaviors.tsv")
    prediction = tmp_path / "prediction.txt"
    write_prediction_file(prediction, [("1", [0.2, 0.9, 0.1]), ("2", [1.0, 0.0])])
    assert validate_prediction_file(behaviors, prediction) == 2


def test_validator_rejects_incomplete_rank_permutation(tmp_path: Path) -> None:
    behaviors = _behaviors(tmp_path / "behaviors.tsv")
    prediction = tmp_path / "prediction.txt"
    prediction.write_text("1 [1,1,3]\n2 [1,2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="permutation"):
        validate_prediction_file(behaviors, prediction)


def test_validator_rejects_wrong_impression_order(tmp_path: Path) -> None:
    behaviors = _behaviors(tmp_path / "behaviors.tsv")
    prediction = tmp_path / "prediction.txt"
    prediction.write_text("2 [1,2,3]\n1 [1,2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="impression"):
        validate_prediction_file(behaviors, prediction)
