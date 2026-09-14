from datetime import UTC, datetime
from pathlib import Path

import pytest

from recforge.data.protocol import TemporalCatalogIndex, iter_positive_retrieval_examples


def _write_behaviors(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "1\tU1\t11/11/2019 9:00:00 AM\told-item\tcurrent-1 future-0",
                "2\tU1\t11/12/2019 9:00:00 AM\told-item current\tcurrent-1 future-0",
                "3\tU2\t11/13/2019 9:00:00 AM\t\tother-0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_catalog_excludes_items_observed_only_in_the_future(tmp_path: Path) -> None:
    path = _write_behaviors(tmp_path / "behaviors.tsv")
    index = TemporalCatalogIndex.from_behavior_files([path])
    examples = list(iter_positive_retrieval_examples(path, namespace="train"))
    first_candidates = index.candidates_for(examples[0], exclude_history=False)
    assert set(first_candidates) == {"old-item", "current", "future"}
    assert "other" not in first_candidates


def test_seen_filter_retains_repeated_positive(tmp_path: Path) -> None:
    path = _write_behaviors(tmp_path / "behaviors.tsv")
    index = TemporalCatalogIndex.from_behavior_files([path])
    second = list(iter_positive_retrieval_examples(path, namespace="train"))[1]
    candidates = index.candidates_for(second)
    assert "current" in candidates
    assert "old" not in candidates


def test_examples_skip_impressions_without_clicks(tmp_path: Path) -> None:
    path = _write_behaviors(tmp_path / "behaviors.tsv")
    examples = list(iter_positive_retrieval_examples(path, namespace="train"))
    assert [example.query_id for example in examples] == ["train:1", "train:2"]


def test_catalog_rejects_timezone_aware_query(tmp_path: Path) -> None:
    path = _write_behaviors(tmp_path / "behaviors.tsv")
    index = TemporalCatalogIndex.from_behavior_files([path])
    with pytest.raises(ValueError, match="timezone-naive"):
        index.eligible_at(datetime.now(UTC))
