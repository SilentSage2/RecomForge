import json
from pathlib import Path

import numpy as np
import pytest

from recforge.data.features import (
    HashingFeatureConfig,
    ItemFeatureTable,
    aggregate_history_features,
    build_item_feature_table,
    hash_item_features,
    load_feature_artifact,
    write_feature_artifact,
)
from recforge.data.mind import MindNews


def _news(news_id: str, title: str = "Alpha beta") -> MindNews:
    return MindNews(news_id, "Science", "Space", title, "Abstract", "url", "[]", "[]")


def _write_news(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_hash_features_are_deterministic_normalized_and_namespaced() -> None:
    config = HashingFeatureConfig(dimension=128)
    first = hash_item_features(_news("N1"), config)
    second = hash_item_features(_news("N2"), config)

    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.count_nonzero(first) <= 4


def test_table_merges_identical_items_and_sorts_ids(tmp_path: Path) -> None:
    row_1 = "N2\tcat\tsub\tSecond title\tabs\turl\t[]\t[]"
    row_2 = "N1\tcat\tsub\tFirst title\tabs\turl\t[]\t[]"
    first = _write_news(tmp_path / "first.tsv", [row_1, row_2])
    second = _write_news(tmp_path / "second.tsv", [row_2])

    table = build_item_feature_table([first, second], HashingFeatureConfig(dimension=32))

    assert table.item_ids == ("N1", "N2")
    assert table.features.shape == (2, 32)


def test_table_rejects_conflicting_duplicate_metadata(tmp_path: Path) -> None:
    first = _write_news(tmp_path / "first.tsv", ["N1\tcat\tsub\tFirst\tabs\turl\t[]\t[]"])
    second = _write_news(tmp_path / "second.tsv", ["N1\tcat\tsub\tChanged\tabs\turl\t[]\t[]"])
    with pytest.raises(ValueError, match="conflicting"):
        build_item_feature_table([first, second], HashingFeatureConfig())


def test_history_uses_recent_known_items_and_handles_empty_history() -> None:
    config = HashingFeatureConfig(dimension=64)
    item_ids = ("old", "recent")
    features = np.stack(
        [
            hash_item_features(_news("old", "old"), config),
            hash_item_features(_news("recent", "new"), config),
        ]
    )
    table = ItemFeatureTable(item_ids, features)

    actual = aggregate_history_features(("old", "unknown", "recent"), table, max_history_items=2)
    assert np.allclose(actual, features[1], atol=1e-7)
    assert np.array_equal(
        aggregate_history_features((), table, max_history_items=2),
        np.zeros(64, dtype=np.float32),
    )


def test_feature_artifact_is_immutable_and_fingerprinted(tmp_path: Path) -> None:
    news_path = _write_news(tmp_path / "news.tsv", ["N1\tcat\tsub\tFirst\tabs\turl\t[]\t[]"])
    config = HashingFeatureConfig(dimension=16)
    table = build_item_feature_table([news_path], config)
    output = tmp_path / "artifact"

    write_feature_artifact(output, table, config, [news_path])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["item_count"] == 1
    assert len(manifest["feature_sha256"]) == 64
    restored, restored_config = load_feature_artifact(output)
    assert restored.item_ids == table.item_ids
    assert np.array_equal(restored.features, table.features)
    assert restored_config == config
    with pytest.raises(FileExistsError):
        write_feature_artifact(output, table, config, [news_path])


def test_feature_artifact_detects_tampering(tmp_path: Path) -> None:
    news_path = _write_news(tmp_path / "news.tsv", ["N1\tcat\tsub\tFirst\tabs\turl\t[]\t[]"])
    config = HashingFeatureConfig(dimension=16)
    table = build_item_feature_table([news_path], config)
    output = tmp_path / "artifact"
    write_feature_artifact(output, table, config, [news_path])
    (output / "item_ids.json").write_text('["changed"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_feature_artifact(output)
