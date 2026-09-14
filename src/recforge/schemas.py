"""Typed, validated records used by evaluation and baseline code."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def require_aware(value: datetime, field_name: str) -> None:
    """Reject ambiguous naive datetimes at data boundaries."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    available_at: datetime
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must not be empty")
        require_aware(self.available_at, "available_at")


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.user_id or not self.item_id:
            raise ValueError("user_id and item_id must not be empty")
        require_aware(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    query_id: str
    user_id: str
    timestamp: datetime
    history: tuple[Interaction, ...]
    positive_item_ids: frozenset[str]
    candidate_item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id or not self.user_id:
            raise ValueError("query_id and user_id must not be empty")
        require_aware(self.timestamp, "timestamp")
        if not self.positive_item_ids:
            raise ValueError("positive_item_ids must not be empty")
        if not self.candidate_item_ids:
            raise ValueError("candidate_item_ids must not be empty")
        if len(set(self.candidate_item_ids)) != len(self.candidate_item_ids):
            raise ValueError("candidate_item_ids must be unique")
        if not self.positive_item_ids.issubset(self.candidate_item_ids):
            raise ValueError("all positive items must be candidates")


@dataclass(frozen=True, slots=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime
    candidate_item_ids: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.impression_id or not self.user_id:
            raise ValueError("impression_id and user_id must not be empty")
        require_aware(self.timestamp, "timestamp")
        if len(self.candidate_item_ids) != len(self.labels):
            raise ValueError("candidate_item_ids and labels must have equal length")
        if not self.candidate_item_ids:
            raise ValueError("an impression must have candidates")
        if len(set(self.candidate_item_ids)) != len(self.candidate_item_ids):
            raise ValueError("candidate_item_ids must be unique")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("labels must be binary")
