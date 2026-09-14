"""Leakage-aware query and candidate construction for MIND retrieval."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recforge.data.mind import iter_mind_behaviors


@dataclass(frozen=True, slots=True)
class MindRetrievalExample:
    query_id: str
    user_id: str
    local_timestamp: datetime
    history_item_ids: tuple[str, ...]
    positive_item_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.local_timestamp.tzinfo is not None:
            raise ValueError("MIND local_timestamp must remain timezone-naive")
        if not self.positive_item_ids:
            raise ValueError("positive_item_ids must not be empty")


@dataclass(frozen=True, slots=True)
class TemporalCatalogIndex:
    """Items sorted by their first label-independent observation time."""

    observed_at: tuple[datetime, ...]
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.observed_at) != len(self.item_ids):
            raise ValueError("observed_at and item_ids must have equal length")
        if any(timestamp.tzinfo is not None for timestamp in self.observed_at):
            raise ValueError("MIND observation times must remain timezone-naive")
        if tuple(zip(self.observed_at, self.item_ids, strict=True)) != tuple(
            sorted(zip(self.observed_at, self.item_ids, strict=True))
        ):
            raise ValueError("catalog entries must be sorted by timestamp and item ID")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("catalog item IDs must be unique")

    @classmethod
    def from_behavior_files(cls, behavior_paths: Iterable[Path]) -> TemporalCatalogIndex:
        first_observed: dict[str, datetime] = {}
        for path in behavior_paths:
            for behavior in iter_mind_behaviors(path):
                for item_id in (*behavior.history_item_ids, *behavior.candidate_item_ids):
                    current = first_observed.get(item_id)
                    if current is None or behavior.local_timestamp < current:
                        first_observed[item_id] = behavior.local_timestamp
        ordered = sorted((timestamp, item_id) for item_id, timestamp in first_observed.items())
        return cls(
            observed_at=tuple(timestamp for timestamp, _ in ordered),
            item_ids=tuple(item_id for _, item_id in ordered),
        )

    def eligible_at(self, local_timestamp: datetime) -> tuple[str, ...]:
        if local_timestamp.tzinfo is not None:
            raise ValueError("MIND local_timestamp must remain timezone-naive")
        end = bisect_right(self.observed_at, local_timestamp)
        return self.item_ids[:end]

    def candidates_for(
        self, example: MindRetrievalExample, *, exclude_history: bool = True
    ) -> tuple[str, ...]:
        eligible = self.eligible_at(example.local_timestamp)
        eligible_set = set(eligible)
        missing_positives = example.positive_item_ids.difference(eligible_set)
        if missing_positives:
            raise ValueError(
                f"query {example.query_id} has positives outside the temporal catalog: "
                f"{sorted(missing_positives)}"
            )
        if not exclude_history:
            return eligible
        excluded = set(example.history_item_ids).difference(example.positive_item_ids)
        return tuple(item_id for item_id in eligible if item_id not in excluded)


def iter_positive_retrieval_examples(
    behavior_path: Path, *, namespace: str
) -> Iterator[MindRetrievalExample]:
    """Yield one retrieval query per impression containing at least one click."""
    if not namespace:
        raise ValueError("namespace must not be empty")
    for behavior in iter_mind_behaviors(behavior_path):
        positives = frozenset(
            item_id
            for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        )
        if not positives:
            continue
        yield MindRetrievalExample(
            query_id=f"{namespace}:{behavior.impression_id}",
            user_id=behavior.user_id,
            local_timestamp=behavior.local_timestamp,
            history_item_ids=behavior.history_item_ids,
            positive_item_ids=positives,
        )
