"""Audit statistics for the versioned MIND temporal retrieval protocol."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from recforge.data.mind import sha256_file
from recforge.data.protocol import TemporalCatalogIndex, iter_positive_retrieval_examples


@dataclass(frozen=True, slots=True)
class ProtocolSplitAudit:
    split: str
    behaviors_sha256: str
    positive_query_count: int
    positive_target_count: int
    multi_positive_query_count: int
    repeated_positive_query_count: int
    min_eligible_catalog_size: int
    median_eligible_catalog_size: int
    p95_eligible_catalog_size: int
    max_eligible_catalog_size: int
    mean_eligible_catalog_size: float
    mean_candidates_after_history_filter: float
    mean_history_items_removed: float


@dataclass(frozen=True, slots=True)
class TemporalProtocolAudit:
    audit_schema_version: int
    protocol: str
    catalog_item_count: int
    first_observed_at: str
    last_observed_at: str
    splits: tuple[ProtocolSplitAudit, ...]


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot compute a percentile of an empty collection")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def audit_temporal_protocol(
    split_paths: Iterable[tuple[str, Path]],
) -> TemporalProtocolAudit:
    """Audit temporal-corpus query sizes without materializing every candidate tuple."""
    splits = tuple(split_paths)
    if not splits:
        raise ValueError("at least one split is required")
    names = [name for name, _ in splits]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("split names must be non-empty and unique")

    index = TemporalCatalogIndex.from_behavior_files(path for _, path in splits)
    if not index.item_ids:
        raise ValueError("temporal catalog is empty")
    first_observed_by_item = dict(zip(index.item_ids, index.observed_at, strict=True))
    split_audits: list[ProtocolSplitAudit] = []

    for split, path in splits:
        eligible_sizes: list[int] = []
        filtered_sizes: list[int] = []
        positive_target_count = 0
        multi_positive_query_count = 0
        repeated_positive_query_count = 0

        for example in iter_positive_retrieval_examples(path, namespace=split):
            eligible_size = bisect_right(index.observed_at, example.local_timestamp)
            missing = [
                item_id
                for item_id in example.positive_item_ids
                if first_observed_by_item.get(item_id, example.local_timestamp)
                > example.local_timestamp
                or item_id not in first_observed_by_item
            ]
            if missing:
                raise ValueError(
                    f"query {example.query_id} has positives outside the temporal catalog: "
                    f"{sorted(missing)}"
                )
            excluded_history = {
                item_id
                for item_id in example.history_item_ids
                if item_id not in example.positive_item_ids
                and first_observed_by_item.get(item_id, example.local_timestamp)
                <= example.local_timestamp
            }
            eligible_sizes.append(eligible_size)
            filtered_sizes.append(eligible_size - len(excluded_history))
            positive_target_count += len(example.positive_item_ids)
            multi_positive_query_count += len(example.positive_item_ids) > 1
            repeated_positive_query_count += bool(
                example.positive_item_ids.intersection(example.history_item_ids)
            )

        if not eligible_sizes:
            raise ValueError(f"split {split!r} contains no positive queries")
        removed_total = sum(eligible_sizes) - sum(filtered_sizes)
        query_count = len(eligible_sizes)
        split_audits.append(
            ProtocolSplitAudit(
                split=split,
                behaviors_sha256=sha256_file(path),
                positive_query_count=query_count,
                positive_target_count=positive_target_count,
                multi_positive_query_count=multi_positive_query_count,
                repeated_positive_query_count=repeated_positive_query_count,
                min_eligible_catalog_size=min(eligible_sizes),
                median_eligible_catalog_size=_nearest_rank(eligible_sizes, 0.5),
                p95_eligible_catalog_size=_nearest_rank(eligible_sizes, 0.95),
                max_eligible_catalog_size=max(eligible_sizes),
                mean_eligible_catalog_size=round(sum(eligible_sizes) / query_count, 6),
                mean_candidates_after_history_filter=round(sum(filtered_sizes) / query_count, 6),
                mean_history_items_removed=round(removed_total / query_count, 6),
            )
        )

    return TemporalProtocolAudit(
        audit_schema_version=1,
        protocol="temporal_corpus_v1",
        catalog_item_count=len(index.item_ids),
        first_observed_at=index.observed_at[0].isoformat(),
        last_observed_at=index.observed_at[-1].isoformat(),
        splits=tuple(split_audits),
    )


def _write_json(audit: TemporalProtocolAudit, handle: TextIO) -> None:
    json.dump(asdict(audit), handle, indent=2, sort_keys=True)
    handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit temporal MIND retrieval corpora.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=2,
        metavar=("NAME", "BEHAVIORS_TSV"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split_paths = [(name, Path(path)) for name, path in args.split]
    audit = audit_temporal_protocol(split_paths)
    if args.output is None:
        import sys

        _write_json(audit, sys.stdout)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        _write_json(audit, handle)


if __name__ == "__main__":
    main()
