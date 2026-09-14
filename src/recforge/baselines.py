"""Auditable non-personalized baselines for temporal recommendation experiments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from recforge.schemas import Interaction, require_aware


def _rank(scores: dict[str, float], candidates: Iterable[str], k: int) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    unique_candidates = set(candidates)
    return tuple(
        sorted(unique_candidates, key=lambda item_id: (-scores.get(item_id, 0.0), item_id))[:k]
    )


def global_popularity(
    interactions: Sequence[Interaction], candidates: Iterable[str], k: int
) -> tuple[str, ...]:
    """Rank candidates by interaction count with deterministic item-ID tie breaking."""
    counts = Counter(event.item_id for event in interactions)
    return _rank({item_id: float(count) for item_id, count in counts.items()}, candidates, k)


def time_decayed_popularity(
    interactions: Sequence[Interaction],
    candidates: Iterable[str],
    *,
    as_of: datetime,
    half_life: timedelta,
    k: int,
) -> tuple[str, ...]:
    """Rank by exponentially decayed counts using only events strictly before ``as_of``."""
    require_aware(as_of, "as_of")
    half_life_seconds = half_life.total_seconds()
    if half_life_seconds <= 0:
        raise ValueError("half_life must be positive")
    scores: dict[str, float] = {}
    for event in interactions:
        age_seconds = (as_of - event.timestamp).total_seconds()
        if age_seconds <= 0:
            raise ValueError("time-decayed popularity received an event at or after as_of")
        weight = math.exp(-math.log(2.0) * age_seconds / half_life_seconds)
        scores[event.item_id] = scores.get(event.item_id, 0.0) + weight
    return _rank(scores, candidates, k)
