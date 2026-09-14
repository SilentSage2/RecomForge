from pathlib import Path

from recforge.logged_baselines import popularity_predictions, rank_scores


def test_rank_scores_use_item_id_instead_of_logged_order_for_ties() -> None:
    assert rank_scores(("B", "A", "C"), {}) == [-2.0, -1.0, -3.0]


def test_popularity_predictions_do_not_consult_evaluation_labels(tmp_path: Path) -> None:
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN0\tN2-1 N1-0\n",
        encoding="utf-8",
    )
    second.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN0\tN2-0 N1-1\n",
        encoding="utf-8",
    )
    scores = {"N1": 5.0, "N2": 2.0}

    assert popularity_predictions(first, scores) == popularity_predictions(second, scores)
