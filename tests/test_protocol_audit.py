import json
from dataclasses import asdict
from pathlib import Path

import pytest

from recforge.data.protocol_audit import audit_temporal_protocol


def _write_split(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_audit_summarizes_temporal_catalog_and_seen_filter(tmp_path: Path) -> None:
    train = _write_split(
        tmp_path / "train.tsv",
        [
            "1\tU1\t11/11/2019 9:00:00 AM\told\tnew-1 distractor-0",
            "2\tU1\t11/12/2019 9:00:00 AM\told new\tnew-1 later-0",
        ],
    )
    dev = _write_split(
        tmp_path / "dev.tsv",
        ["1\tU2\t11/13/2019 9:00:00 AM\told\tlater-1 another-1"],
    )

    payload = asdict(audit_temporal_protocol([("train", train), ("dev", dev)]))

    assert payload["protocol"] == "temporal_corpus_v1"
    assert payload["catalog_item_count"] == 5
    train_audit = payload["splits"][0]
    assert train_audit["positive_query_count"] == 2
    assert train_audit["repeated_positive_query_count"] == 1
    assert train_audit["min_eligible_catalog_size"] == 3
    assert train_audit["max_eligible_catalog_size"] == 4
    assert train_audit["mean_history_items_removed"] == 1.0
    dev_audit = payload["splits"][1]
    assert dev_audit["multi_positive_query_count"] == 1
    assert dev_audit["positive_target_count"] == 2


def test_audit_payload_is_json_serializable(tmp_path: Path) -> None:
    path = _write_split(
        tmp_path / "behaviors.tsv",
        ["1\tU1\t11/11/2019 9:00:00 AM\t\titem-1"],
    )
    payload = asdict(audit_temporal_protocol([("train", path)]))
    assert json.loads(json.dumps(payload))["catalog_item_count"] == 1


def test_audit_rejects_duplicate_split_names(tmp_path: Path) -> None:
    path = _write_split(
        tmp_path / "behaviors.tsv",
        ["1\tU1\t11/11/2019 9:00:00 AM\t\titem-1"],
    )
    with pytest.raises(ValueError, match="unique"):
        audit_temporal_protocol([("train", path), ("train", path)])
