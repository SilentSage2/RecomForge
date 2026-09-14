from datetime import UTC, datetime, timedelta

import pytest

from recforge.baselines import global_popularity, time_decayed_popularity
from recforge.schemas import Interaction


def _time(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def test_global_popularity_is_deterministic_for_ties_and_cold_items() -> None:
    interactions = [
        Interaction("u1", "b", _time(1)),
        Interaction("u2", "a", _time(1)),
    ]
    assert global_popularity(interactions, ["cold", "b", "a"], 3) == ("a", "b", "cold")


def test_time_decay_can_reverse_global_popularity() -> None:
    interactions = [
        Interaction("u1", "old", _time(1)),
        Interaction("u2", "old", _time(1)),
        Interaction("u3", "recent", _time(4)),
    ]
    ranking = time_decayed_popularity(
        interactions,
        ["old", "recent"],
        as_of=_time(5),
        half_life=timedelta(days=1),
        k=2,
    )
    assert ranking == ("recent", "old")


def test_time_decay_rejects_future_data() -> None:
    interactions = [Interaction("u", "item", _time(5))]
    with pytest.raises(ValueError, match="at or after"):
        time_decayed_popularity(
            interactions,
            ["item"],
            as_of=_time(5),
            half_life=timedelta(days=1),
            k=1,
        )
