"""Strict, streaming adapter for the MIND news recommendation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True, slots=True)
class MindNews:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities_json: str
    abstract_entities_json: str


@dataclass(frozen=True, slots=True)
class MindBehavior:
    impression_id: str
    user_id: str
    local_timestamp: datetime
    history_item_ids: tuple[str, ...]
    candidate_item_ids: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.local_timestamp.tzinfo is not None:
            raise ValueError("MIND local_timestamp must remain timezone-naive")
        if len(self.candidate_item_ids) != len(self.labels):
            raise ValueError("candidate_item_ids and labels must have equal length")
        if not self.candidate_item_ids:
            raise ValueError("a MIND behavior must contain candidates")
        if len(set(self.candidate_item_ids)) != len(self.candidate_item_ids):
            raise ValueError("MIND candidate_item_ids must be unique within an impression")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("MIND labels must be binary")


@dataclass(frozen=True, slots=True)
class MindAudit:
    split: str
    behaviors_sha256: str
    news_sha256: str
    behavior_count: int
    news_count: int
    duplicate_news_id_count: int
    unique_user_count: int
    unique_candidate_count: int
    unique_history_item_count: int
    missing_candidate_news_count: int
    missing_history_news_count: int
    positive_count: int
    negative_count: int
    empty_history_count: int
    min_local_timestamp: str
    max_local_timestamp: str
    mean_history_length: float
    mean_candidate_count: float


def _rows(path: Path, expected_columns: int) -> Iterator[tuple[int, list[str]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != expected_columns:
                raise ValueError(
                    f"{path}:{line_number} expected {expected_columns} tab-separated "
                    f"columns, found {len(fields)}"
                )
            yield line_number, fields


def iter_mind_news(path: Path) -> Iterator[MindNews]:
    """Yield MIND news rows without materializing the corpus."""
    for _, fields in _rows(path, expected_columns=8):
        yield MindNews(*fields)


def _parse_candidate(raw: str, *, path: Path, line_number: int) -> tuple[str, int]:
    try:
        item_id, raw_label = raw.rsplit("-", maxsplit=1)
        label = int(raw_label)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}:{line_number} invalid labeled candidate {raw!r}") from error
    if not item_id or label not in (0, 1):
        raise ValueError(f"{path}:{line_number} invalid labeled candidate {raw!r}")
    return item_id, label


def iter_mind_behaviors(path: Path) -> Iterator[MindBehavior]:
    """Yield MIND behavior rows while preserving the unspecified local clock."""
    for line_number, fields in _rows(path, expected_columns=5):
        impression_id, user_id, raw_time, raw_history, raw_candidates = fields
        try:
            local_timestamp = datetime.strptime(raw_time, MIND_TIME_FORMAT)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} invalid timestamp {raw_time!r}") from error
        labeled = tuple(
            _parse_candidate(value, path=path, line_number=line_number)
            for value in raw_candidates.split()
        )
        yield MindBehavior(
            impression_id=impression_id,
            user_id=user_id,
            local_timestamp=local_timestamp,
            history_item_ids=tuple(raw_history.split()),
            candidate_item_ids=tuple(item_id for item_id, _ in labeled),
            labels=tuple(label for _, label in labeled),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _news_inventory(path: Path) -> tuple[int, set[str]]:
    count = 0
    news_ids: set[str] = set()
    for news in iter_mind_news(path):
        count += 1
        news_ids.add(news.news_id)
    return count, news_ids


def audit_mind_split(split_directory: Path) -> MindAudit:
    """Validate a MIND split and return compact provenance and distribution statistics."""
    behaviors_path = split_directory / "behaviors.tsv"
    news_path = split_directory / "news.tsv"
    users: set[str] = set()
    candidates: set[str] = set()
    history_items: set[str] = set()
    behavior_count = 0
    positive_count = 0
    negative_count = 0
    empty_history_count = 0
    history_total = 0
    candidate_total = 0
    min_time: datetime | None = None
    max_time: datetime | None = None

    for behavior in iter_mind_behaviors(behaviors_path):
        behavior_count += 1
        users.add(behavior.user_id)
        candidates.update(behavior.candidate_item_ids)
        history_items.update(behavior.history_item_ids)
        positive_count += sum(behavior.labels)
        negative_count += len(behavior.labels) - sum(behavior.labels)
        empty_history_count += not behavior.history_item_ids
        history_total += len(behavior.history_item_ids)
        candidate_total += len(behavior.candidate_item_ids)
        min_time = (
            behavior.local_timestamp
            if min_time is None
            else min(min_time, behavior.local_timestamp)
        )
        max_time = (
            behavior.local_timestamp
            if max_time is None
            else max(max_time, behavior.local_timestamp)
        )

    if behavior_count == 0 or min_time is None or max_time is None:
        raise ValueError(f"{behaviors_path} contains no behaviors")

    news_count, news_ids = _news_inventory(news_path)

    return MindAudit(
        split=split_directory.name,
        behaviors_sha256=sha256_file(behaviors_path),
        news_sha256=sha256_file(news_path),
        behavior_count=behavior_count,
        news_count=news_count,
        duplicate_news_id_count=news_count - len(news_ids),
        unique_user_count=len(users),
        unique_candidate_count=len(candidates),
        unique_history_item_count=len(history_items),
        missing_candidate_news_count=len(candidates.difference(news_ids)),
        missing_history_news_count=len(history_items.difference(news_ids)),
        positive_count=positive_count,
        negative_count=negative_count,
        empty_history_count=empty_history_count,
        min_local_timestamp=min_time.isoformat(),
        max_local_timestamp=max_time.isoformat(),
        mean_history_length=round(history_total / behavior_count, 6),
        mean_candidate_count=round(candidate_total / behavior_count, 6),
    )


def _write_json(audits: list[MindAudit], handle: TextIO) -> None:
    payload = {
        "audit_schema_version": 1,
        "splits": [asdict(audit) for audit in audits],
    }
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize extracted MIND splits.")
    parser.add_argument("split_directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audits = [audit_mind_split(path) for path in args.split_directories]
    if args.output is None:
        import sys

        _write_json(audits, sys.stdout)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        _write_json(audits, handle)


if __name__ == "__main__":
    main()
