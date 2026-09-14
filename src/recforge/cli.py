"""Synthetic smoke evaluation for the R0 baseline pipeline."""

from __future__ import annotations

import json
from datetime import timedelta
from statistics import fmean

from recforge.baselines import global_popularity, time_decayed_popularity
from recforge.metrics import catalog_coverage_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from recforge.split import validate_retrieval_query
from recforge.synthetic import build_synthetic_dataset


def _evaluate(rankings: list[tuple[str, ...]], positives: list[frozenset[str]]) -> dict[str, float]:
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


def main() -> None:
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

    output = {
        "dataset": "synthetic-v1",
        "global_popularity": _evaluate(global_rankings, positives),
        "time_decayed_popularity": _evaluate(decayed_rankings, positives),
        "coverage@3": {
            "global_popularity": catalog_coverage_at_k(global_rankings, candidates, 3),
            "time_decayed_popularity": catalog_coverage_at_k(decayed_rankings, candidates, 3),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
