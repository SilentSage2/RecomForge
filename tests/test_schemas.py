from datetime import UTC, datetime

import pytest

from recforge.schemas import Impression, Interaction, Item, RetrievalQuery

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Item("item", datetime(2026, 1, 1))


def test_retrieval_query_requires_positive_candidate() -> None:
    with pytest.raises(ValueError, match="positive items"):
        RetrievalQuery("q", "u", NOW, (), frozenset({"missing"}), ("item",))


def test_impression_requires_binary_aligned_labels() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Impression("i", "u", NOW, ("a", "b"), (1,))
    with pytest.raises(ValueError, match="binary"):
        Impression("i", "u", NOW, ("a",), (2,))


def test_interaction_requires_ids() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Interaction("", "item", NOW)
