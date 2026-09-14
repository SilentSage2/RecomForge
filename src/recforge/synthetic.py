"""Small deterministic data designed to expose temporal and metric bugs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from recforge.schemas import Interaction, Item, RetrievalQuery


def _utc(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    items: tuple[Item, ...]
    training_interactions: tuple[Interaction, ...]
    queries: tuple[RetrievalQuery, ...]


def build_synthetic_dataset() -> SyntheticDataset:
    """Return a stable fixture with cold, tied, recent, and future item cases."""
    items = (
        Item("article-a", _utc(1), "science"),
        Item("article-b", _utc(1), "sports"),
        Item("article-c", _utc(2), "science"),
        Item("article-d", _utc(3), "business"),
        Item("article-cold", _utc(4), "science"),
        Item("article-future", _utc(7), "culture"),
    )
    interactions = (
        Interaction("user-1", "article-a", _utc(2, 8)),
        Interaction("user-2", "article-a", _utc(2, 9)),
        Interaction("user-3", "article-b", _utc(2, 10)),
        Interaction("user-1", "article-c", _utc(4, 8)),
        Interaction("user-2", "article-d", _utc(4, 9)),
    )
    eligible = ("article-a", "article-b", "article-c", "article-d", "article-cold")
    queries = (
        RetrievalQuery(
            query_id="query-1",
            user_id="user-1",
            timestamp=_utc(5),
            history=(interactions[0], interactions[3]),
            positive_item_ids=frozenset({"article-d"}),
            candidate_item_ids=eligible,
        ),
        RetrievalQuery(
            query_id="query-2",
            user_id="user-2",
            timestamp=_utc(6),
            history=(interactions[1], interactions[4]),
            positive_item_ids=frozenset({"article-cold"}),
            candidate_item_ids=eligible,
        ),
    )
    return SyntheticDataset(items, interactions, queries)
