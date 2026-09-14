"""Leakage-safe popularity baselines for the official logged-candidate protocol."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from recforge.data.mind import iter_mind_behaviors, sha256_file
from recforge.data.submission import order_from_scores, write_prediction_file
from recforge.evaluation import positive_target_popularity, time_decayed_target_popularity
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.tracking import JsonValue, record_run


@dataclass(frozen=True, slots=True)
class LoggedPopularityConfig:
    train_behaviors: str
    eval_behaviors: str
    output_root: str = "runs"
    half_life_hours: float = 72.0
    seed: int = 2027

    def __post_init__(self) -> None:
        if self.half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")


def rank_scores(
    candidate_item_ids: tuple[str, ...], item_scores: Mapping[str, float]
) -> list[float]:
    """Return unique candidate-aligned scores with item-ID tie breaking."""
    order = sorted(
        range(len(candidate_item_ids)),
        key=lambda index: (
            -item_scores.get(candidate_item_ids[index], 0.0),
            candidate_item_ids[index],
            index,
        ),
    )
    ranks = [0] * len(order)
    for rank, candidate_index in enumerate(order, start=1):
        ranks[candidate_index] = rank
    return [-float(rank) for rank in ranks]


def popularity_predictions(
    behavior_path: Path, item_scores: Mapping[str, float]
) -> list[tuple[str, list[float]]]:
    """Score logged candidates without consulting evaluation labels."""
    return [
        (behavior.impression_id, rank_scores(behavior.candidate_item_ids, item_scores))
        for behavior in iter_mind_behaviors(behavior_path)
    ]


def evaluate_logged_predictions(
    behavior_path: Path, predictions: list[tuple[str, list[float]]]
) -> dict[str, float | int]:
    behaviors = list(iter_mind_behaviors(behavior_path))
    if len(behaviors) != len(predictions) or not behaviors:
        raise ValueError("prediction and behavior rows must be equal and non-empty")
    totals = {"auc": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    for behavior, (impression_id, scores) in zip(behaviors, predictions, strict=True):
        if impression_id != behavior.impression_id:
            raise ValueError("prediction order does not match behavior order")
        positives = {
            item
            for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        }
        if not positives or 0 not in behavior.labels:
            raise ValueError(f"impression {impression_id} lacks positive and negative labels")
        ranked = [behavior.candidate_item_ids[index] for index in order_from_scores(scores)]
        totals["auc"] += binary_auc(behavior.labels, scores)
        totals["mrr"] += mean_reciprocal_rank_at_k(ranked, positives, len(ranked))
        totals["ndcg@5"] += ndcg_at_k(ranked, positives, 5)
        totals["ndcg@10"] += ndcg_at_k(ranked, positives, 10)
    count = len(behaviors)
    return {"query_count": count, **{key: value / count for key, value in totals.items()}}


def run_logged_popularity(config: LoggedPopularityConfig, repository_root: Path) -> Path:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    train_path = repository_root / config.train_behaviors
    eval_path = repository_root / config.eval_behaviors
    global_scores: Counter[str] = positive_target_popularity(train_path)
    training_behaviors = list(iter_mind_behaviors(train_path))
    if not training_behaviors:
        raise ValueError("training behaviors must not be empty")
    reference_time = max(behavior.local_timestamp for behavior in training_behaviors)
    decayed_scores = time_decayed_target_popularity(
        train_path,
        reference_time=reference_time,
        half_life_hours=config.half_life_hours,
    )
    global_predictions = popularity_predictions(eval_path, global_scores)
    decayed_predictions = popularity_predictions(eval_path, decayed_scores)
    metrics: dict[str, JsonValue] = {
        "protocol": "mind_official_impression",
        "training_positive_count": sum(global_scores.values()),
        "training_clicked_item_count": len(global_scores),
        "reference_time": reference_time.isoformat(),
        "global_popularity": cast(
            dict[str, JsonValue], evaluate_logged_predictions(eval_path, global_predictions)
        ),
        "time_decayed_popularity": cast(
            dict[str, JsonValue], evaluate_logged_predictions(eval_path, decayed_predictions)
        ),
    }
    finished_at = datetime.now(UTC)
    run_directory = record_run(
        output_root=repository_root / config.output_root,
        repository_root=repository_root,
        experiment="mind-logged-popularity",
        metrics=metrics,
        config=cast(dict[str, JsonValue], asdict(config)),
        dataset_fingerprints={
            "train_behaviors": sha256_file(train_path),
            "eval_behaviors": sha256_file(eval_path),
        },
        seed=config.seed,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.perf_counter() - started,
    )
    write_prediction_file(run_directory / "global-prediction.txt", global_predictions)
    write_prediction_file(run_directory / "time-decayed-prediction.txt", decayed_predictions)
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate logged-candidate popularity baselines.")
    parser.add_argument("--train-behaviors", type=str, required=True)
    parser.add_argument("--eval-behaviors", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--half-life-hours", type=float, default=72.0)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = LoggedPopularityConfig(
        train_behaviors=args.train_behaviors,
        eval_behaviors=args.eval_behaviors,
        output_root=args.output_root,
        half_life_hours=args.half_life_hours,
    )
    run_directory = run_logged_popularity(config, args.repository_root.resolve())
    print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
