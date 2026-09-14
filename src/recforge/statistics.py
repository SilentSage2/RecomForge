"""Paired uncertainty estimates for locked MIND impression predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from recforge.data.mind import iter_mind_behaviors, sha256_file
from recforge.data.submission import validate_prediction_file
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.tracking import canonical_json_bytes

_METRIC_NAMES = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _read_prediction_ranks(path: Path) -> dict[str, tuple[int, ...]]:
    predictions: dict[str, tuple[int, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            impression_id, raw_ranks = line.rstrip("\n").split(maxsplit=1)
            parsed: Any = json.loads(raw_ranks)
            if not isinstance(parsed, list) or any(
                isinstance(rank, bool) or not isinstance(rank, int) for rank in parsed
            ):
                raise ValueError(f"prediction line {line_number} has invalid ranks")
            if impression_id in predictions:
                raise ValueError(f"duplicate prediction impression {impression_id!r}")
            predictions[impression_id] = tuple(parsed)
    return predictions


def per_impression_metrics(behavior_path: Path, prediction_path: Path) -> NDArray[np.float64]:
    """Return official metrics in behavior order after strict structural validation."""
    validate_prediction_file(behavior_path, prediction_path)
    predictions = _read_prediction_ranks(prediction_path)
    rows: list[tuple[float, float, float, float]] = []
    for behavior in iter_mind_behaviors(behavior_path):
        ranks = predictions[behavior.impression_id]
        ranked_indices = sorted(range(len(ranks)), key=ranks.__getitem__)
        ranked_items = [behavior.candidate_item_ids[index] for index in ranked_indices]
        positives = {
            item
            for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        }
        if not positives or 0 not in behavior.labels:
            raise ValueError(
                f"impression {behavior.impression_id} lacks both positive and negative labels"
            )
        scores = [-float(rank) for rank in ranks]
        rows.append(
            (
                binary_auc(behavior.labels, scores),
                mean_reciprocal_rank_at_k(ranked_items, positives, len(ranked_items)),
                ndcg_at_k(ranked_items, positives, 5),
                ndcg_at_k(ranked_items, positives, 10),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def paired_bootstrap(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
    resample_batch_size: int = 16,
) -> dict[str, dict[str, float]]:
    """Bootstrap paired impression-level mean differences (candidate minus baseline)."""
    if baseline.shape != candidate.shape or baseline.ndim != 2:
        raise ValueError("baseline and candidate must have equal [samples, metrics] shapes")
    if baseline.shape[0] < 2 or baseline.shape[1] != len(_METRIC_NAMES):
        raise ValueError("paired metrics have an invalid shape")
    if resamples <= 0 or resample_batch_size <= 0:
        raise ValueError("bootstrap sizes must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, differences.shape[1]), dtype=np.float64)
    for start in range(0, resamples, resample_batch_size):
        count = min(resample_batch_size, resamples - start)
        indices = rng.integers(0, differences.shape[0], size=(count, differences.shape[0]))
        draws[start : start + count] = differences[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    results: dict[str, dict[str, float]] = {}
    for index, metric in enumerate(_METRIC_NAMES):
        results[metric] = {
            "baseline_mean": float(baseline[:, index].mean()),
            "candidate_mean": float(candidate[:, index].mean()),
            "mean_difference": float(differences[:, index].mean()),
            "bootstrap_standard_error": float(draws[:, index].std(ddof=1)),
            "confidence_lower": float(np.quantile(draws[:, index], alpha)),
            "confidence_upper": float(np.quantile(draws[:, index], 1 - alpha)),
            "bootstrap_probability_improvement": float((draws[:, index] > 0).mean()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap for two MIND predictions.")
    parser.add_argument("--behaviors", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    baseline = per_impression_metrics(args.behaviors, args.baseline)
    candidate = per_impression_metrics(args.behaviors, args.candidate)
    payload = {
        "schema_version": 1,
        "method": "paired_nonparametric_impression_bootstrap",
        "confidence": 0.95,
        "resamples": args.resamples,
        "seed": args.seed,
        "sample_count": int(baseline.shape[0]),
        "input_sha256": {
            "behaviors": sha256_file(args.behaviors),
            "baseline_prediction": sha256_file(args.baseline),
            "candidate_prediction": sha256_file(args.candidate),
        },
        "metrics": paired_bootstrap(baseline, candidate, resamples=args.resamples, seed=args.seed),
    }
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


if __name__ == "__main__":
    main()
