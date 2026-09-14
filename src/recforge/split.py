"""Temporal partitioning and leakage checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from recforge.schemas import Interaction, Item, RetrievalQuery, require_aware


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]


def temporal_split(
    interactions: Sequence[Interaction], train_end: datetime, validation_end: datetime
) -> TemporalSplit:
    """Split interactions into half-open intervals using global UTC-aware cutoffs."""
    require_aware(train_end, "train_end")
    require_aware(validation_end, "validation_end")
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    ordered = sorted(
        interactions, key=lambda event: (event.timestamp, event.user_id, event.item_id)
    )
    return TemporalSplit(
        train=tuple(event for event in ordered if event.timestamp < train_end),
        validation=tuple(
            event for event in ordered if train_end <= event.timestamp < validation_end
        ),
        test=tuple(event for event in ordered if event.timestamp >= validation_end),
    )


def validate_retrieval_query(query: RetrievalQuery, items: Mapping[str, Item]) -> None:
    """Raise when query context or candidates use information unavailable at query time."""
    for event in query.history:
        if event.user_id != query.user_id:
            raise ValueError(f"query {query.query_id} contains history from another user")
        if event.timestamp >= query.timestamp:
            raise ValueError(f"query {query.query_id} contains future or simultaneous history")

    missing = set(query.candidate_item_ids).difference(items)
    if missing:
        raise ValueError(f"query {query.query_id} references unknown items: {sorted(missing)}")

    unavailable = [
        item_id
        for item_id in query.candidate_item_ids
        if items[item_id].available_at > query.timestamp
    ]
    if unavailable:
        raise ValueError(
            f"query {query.query_id} contains items unavailable at query time: {unavailable}"
        )
