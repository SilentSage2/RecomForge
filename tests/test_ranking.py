from pathlib import Path

import numpy as np
import pytest

from recforge.data.ranking import (
    RankingExample,
    build_title_table,
    iter_ranking_batches,
    load_ranking_examples,
)
from recforge.data.text import Vocabulary


def _write_news(path: Path) -> Path:
    path.write_text(
        "N0\tcat\tsub\tzero title\tabs\turl\t[]\t[]\n"
        "N1\tcat\tsub\tone title\tabs\turl\t[]\t[]\n"
        "N2\tcat\tsub\ttwo title\tabs\turl\t[]\t[]\n"
        "N3\tcat\tsub\tthree secret\tabs\turl\t[]\t[]\n",
        encoding="utf-8",
    )
    return path


def _write_behaviors(path: Path) -> Path:
    path.write_text(
        "7\tU1\t11/11/2019 9:00:00 AM\tmissing N0 N1\tN1-1 N2-0 N3-0\n"
        "8\tU2\t11/11/2019 9:01:00 AM\tN2\tN0-1 N1-1 N3-0\n",
        encoding="utf-8",
    )
    return path


def test_examples_use_only_same_impression_negatives_and_are_deterministic(tmp_path: Path) -> None:
    behaviors = _write_behaviors(tmp_path / "behaviors.tsv")
    first = load_ranking_examples(behaviors, negative_count=2, seed=3, epoch=0)
    second = load_ranking_examples(behaviors, negative_count=2, seed=3, epoch=0)

    assert first == second
    assert len(first) == 3
    assert first[0].candidate_item_ids[0] == "N1"
    assert set(first[0].candidate_item_ids[1:]).issubset({"N2", "N3"})
    assert set(first[1].candidate_item_ids[1:]) == {"N3"}


def test_title_table_and_batches_mask_padding_and_unknown_tokens(tmp_path: Path) -> None:
    news = _write_news(tmp_path / "news.tsv")
    vocabulary = Vocabulary(("<pad>", "<unk>", "zero", "one", "two", "title"))
    table = build_title_table([news], vocabulary, max_title_tokens=3)
    examples = (RankingExample("q", ("missing", "N0", "N1"), ("N2", "N3")),)
    batch = next(
        iter_ranking_batches(
            examples,
            table,
            batch_size=1,
            max_history_items=1,
            seed=0,
            epoch=0,
            shuffle=False,
        )
    )

    assert batch.history_token_ids.shape == (1, 1, 3)
    assert batch.candidate_token_ids.shape == (1, 2, 3)
    assert batch.history_item_mask.tolist() == [[True]]
    assert batch.history_token_mask.tolist() == [[[True, True, False]]]
    assert batch.candidate_token_ids[0, 1, 1] == 1
    assert batch.candidate_token_mask[0, 1].tolist() == [True, True, False]
    assert batch.target_indices.tolist() == [0]


def test_batches_are_deterministic_and_reject_missing_candidate_titles(tmp_path: Path) -> None:
    table = build_title_table(
        [_write_news(tmp_path / "news.tsv")],
        Vocabulary(("<pad>", "<unk>", "title")),
        max_title_tokens=2,
    )
    examples = tuple(RankingExample(f"q{i}", (), ("N1", "N2")) for i in range(3))
    first = list(
        iter_ranking_batches(examples, table, batch_size=2, max_history_items=2, seed=4, epoch=1)
    )
    second = list(
        iter_ranking_batches(examples, table, batch_size=2, max_history_items=2, seed=4, epoch=1)
    )
    assert [batch.query_ids for batch in first] == [batch.query_ids for batch in second]
    assert np.array_equal(first[0].candidate_token_ids, second[0].candidate_token_ids)

    missing = (RankingExample("bad", (), ("N1", "unknown")),)
    with pytest.raises(ValueError, match="has no title"):
        list(
            iter_ranking_batches(
                missing,
                table,
                batch_size=1,
                max_history_items=2,
                seed=0,
                epoch=0,
            )
        )
