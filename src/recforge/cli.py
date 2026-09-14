"""Synthetic smoke evaluation for the R0 baseline pipeline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean

from recforge.baselines import global_popularity, time_decayed_popularity
from recforge.metrics import catalog_coverage_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from recforge.split import validate_retrieval_query
from recforge.synthetic import build_synthetic_dataset
from recforge.tracking import JsonValue, record_run


def _evaluate(
    rankings: list[tuple[str, ...]], positives: list[frozenset[str]]
) -> dict[str, JsonValue]:
    return {
        "recall@3": fmean(
            recall_at_k(ranked, relevant, 3)
            for ranked, relevant in zip(rankings, positives, strict=True)
        ),
        "mrr@3": fmean(
            reciprocal_rank_at_k(ranked, relevant, 3)
            for ranked, relevant in zip(rankings, positives, strict=True)
        ),
        "ndcg@3": fmean(
            ndcg_at_k(ranked, relevant, 3)
            for ranked, relevant in zip(rankings, positives, strict=True)
        ),
    }


def run_smoke() -> dict[str, JsonValue]:
    """Run the deterministic synthetic baseline comparison."""
    dataset = build_synthetic_dataset()
    item_map = {item.item_id: item for item in dataset.items}
    positives = [query.positive_item_ids for query in dataset.queries]
    candidates = dataset.queries[0].candidate_item_ids

    global_rankings: list[tuple[str, ...]] = []
    decayed_rankings: list[tuple[str, ...]] = []
    for query in dataset.queries:
        validate_retrieval_query(query, item_map)
        visible = tuple(
            event for event in dataset.training_interactions if event.timestamp < query.timestamp
        )
        global_rankings.append(global_popularity(visible, query.candidate_item_ids, k=3))
        decayed_rankings.append(
            time_decayed_popularity(
                visible,
                query.candidate_item_ids,
                as_of=query.timestamp,
                half_life=timedelta(days=1),
                k=3,
            )
        )

    return {
        "dataset": "synthetic-v1",
        "global_popularity": _evaluate(global_rankings, positives),
        "time_decayed_popularity": _evaluate(decayed_rankings, positives),
        "coverage@3": {
            "global_popularity": catalog_coverage_at_k(global_rankings, candidates, 3),
            "time_decayed_popularity": catalog_coverage_at_k(decayed_rankings, candidates, 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic RecForge R0 smoke experiment."
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    start_clock = time.perf_counter()
    metrics = run_smoke()
    finished_at = datetime.now(UTC)
    run_directory = record_run(
        output_root=args.output_root,
        repository_root=Path.cwd(),
        experiment="synthetic-baselines",
        metrics=metrics,
        config={"top_k": 3, "time_decay_half_life_days": 1.0},
        dataset_fingerprints={"synthetic": "synthetic-v1"},
        seed=args.seed,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.perf_counter() - start_clock,
    )
    print(
        json.dumps(
            {"metrics": metrics, "run_directory": str(run_directory)}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
