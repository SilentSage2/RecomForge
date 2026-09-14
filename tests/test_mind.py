from pathlib import Path

import pytest

from recforge.data.mind import audit_mind_split, iter_mind_behaviors, iter_mind_news

NEWS_ROW = "\t".join(
    ["N1", "science", "space", "A title", "An abstract", "https://example.test", "[]", "[]"]
)
BEHAVIOR_ROW = "1\tU1\t11/11/2019 9:05:58 AM\tN0 N2\tN1-1 N3-0\n"


def _write_split(root: Path) -> Path:
    root.mkdir()
    (root / "news.tsv").write_text(NEWS_ROW + "\n", encoding="utf-8")
    (root / "behaviors.tsv").write_text(BEHAVIOR_ROW, encoding="utf-8")
    return root


def test_parses_news_and_behavior_without_inventing_timezone(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "MINDsmall_train")
    news = list(iter_mind_news(split / "news.tsv"))
    behaviors = list(iter_mind_behaviors(split / "behaviors.tsv"))
    assert news[0].news_id == "N1"
    assert behaviors[0].history_item_ids == ("N0", "N2")
    assert behaviors[0].candidate_item_ids == ("N1", "N3")
    assert behaviors[0].labels == (1, 0)
    assert behaviors[0].local_timestamp.tzinfo is None


def test_audit_reports_counts_and_hashes(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "MINDsmall_dev")
    audit = audit_mind_split(split)
    assert audit.behavior_count == 1
    assert audit.news_count == 1
    assert audit.unique_user_count == 1
    assert audit.duplicate_news_id_count == 0
    assert audit.missing_candidate_news_count == 1
    assert audit.missing_history_news_count == 2
    assert audit.positive_count == 1
    assert audit.negative_count == 1
    assert len(audit.behaviors_sha256) == 64
    assert len(audit.news_sha256) == 64


def test_rejects_malformed_candidate(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "MINDsmall_train")
    (split / "behaviors.tsv").write_text(
        "1\tU1\t11/11/2019 9:05:58 AM\t\tN1-maybe\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid labeled candidate"):
        list(iter_mind_behaviors(split / "behaviors.tsv"))


def test_rejects_wrong_column_count(tmp_path: Path) -> None:
    split = _write_split(tmp_path / "MINDsmall_train")
    (split / "news.tsv").write_text("too\tfew\tcolumns\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 8"):
        list(iter_mind_news(split / "news.tsv"))
