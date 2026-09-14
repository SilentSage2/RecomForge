"""Official MIND impression-ranking prediction format utilities."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path


def ranks_from_scores(scores: Sequence[float]) -> tuple[int, ...]:
    """Convert candidate-aligned scores to one-based ranks with stable tie breaking."""
    if not scores:
        raise ValueError("scores must not be empty")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    ranks = [0] * len(scores)
    for rank, candidate_index in enumerate(order, start=1):
        ranks[candidate_index] = rank
    return tuple(ranks)


def format_prediction_line(impression_id: str, ranks: Sequence[int]) -> str:
    if not impression_id or any(character.isspace() for character in impression_id):
        raise ValueError("impression_id must be non-empty and contain no whitespace")
    if not ranks:
        raise ValueError("ranks must not be empty")
    expected = set(range(1, len(ranks) + 1))
    if set(ranks) != expected or len(ranks) != len(expected):
        raise ValueError("ranks must be a permutation from 1 through candidate count")
    return f"{impression_id} {json.dumps(list(ranks), separators=(',', ':'))}\n"


def write_prediction_file(path: Path, predictions: Iterable[tuple[str, Sequence[float]]]) -> None:
    """Write candidate-aligned scores in the official prediction.txt rank format."""
    with path.open("x", encoding="utf-8") as handle:
        for impression_id, scores in predictions:
            handle.write(format_prediction_line(impression_id, ranks_from_scores(scores)))


def _expected_impressions(behavior_path: Path) -> Iterable[tuple[str, int]]:
    with behavior_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{behavior_path}:{line_number} expected 5 tab-separated columns")
            candidates = fields[4].split()
            if not candidates:
                raise ValueError(f"{behavior_path}:{line_number} has no candidates")
            yield fields[0], len(candidates)


def validate_prediction_file(behavior_path: Path, prediction_path: Path) -> int:
    """Require exact impression order, row count, candidate count, and rank permutations."""
    expected = list(_expected_impressions(behavior_path))
    with prediction_path.open(encoding="utf-8") as handle:
        prediction_lines = list(handle)
    if len(prediction_lines) != len(expected):
        raise ValueError(
            f"prediction row count {len(prediction_lines)} != expected {len(expected)}"
        )
    for line_number, (line, (expected_id, candidate_count)) in enumerate(
        zip(prediction_lines, expected, strict=True), start=1
    ):
        try:
            impression_id, raw_ranks = line.rstrip("\n").split(maxsplit=1)
            ranks = json.loads(raw_ranks)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"prediction line {line_number} has invalid format") from error
        if impression_id != expected_id:
            raise ValueError(
                f"prediction line {line_number} impression {impression_id!r} "
                f"!= expected {expected_id!r}"
            )
        if not isinstance(ranks, list) or len(ranks) != candidate_count:
            raise ValueError(f"prediction line {line_number} must contain {candidate_count} ranks")
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
            raise ValueError(f"prediction line {line_number} ranks must be integers")
        if set(ranks) != set(range(1, candidate_count + 1)):
            raise ValueError(
                f"prediction line {line_number} ranks must form a complete permutation"
            )
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an official-format MIND prediction.")
    parser.add_argument("--behaviors", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()
    count = validate_prediction_file(args.behaviors, args.prediction)
    print(json.dumps({"status": "valid", "impression_count": count}, indent=2))


if __name__ == "__main__":
    main()
