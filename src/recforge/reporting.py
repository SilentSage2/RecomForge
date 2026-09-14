"""Validated aggregation of full-data L1 seed-level comparison artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

from recforge.data.mind import sha256_file
from recforge.tracking import JsonValue, canonical_json_bytes

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _load_object(path: Path) -> dict[str, Any]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _number(mapping: dict[str, Any], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be numeric")
    return float(value)


def _summary(values: list[float]) -> dict[str, JsonValue]:
    if len(values) < 2:
        raise ValueError("at least two seed values are required")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "values": cast(list[JsonValue], values),
    }


def aggregate_l1_comparisons(
    comparison_paths: list[Path], *, repository_root: Path
) -> dict[str, JsonValue]:
    """Aggregate compatible seed files and their paired-bootstrap effect estimates."""
    if len(comparison_paths) < 2:
        raise ValueError("at least two comparison files are required")
    comparisons = [_load_object(path) for path in comparison_paths]
    seeds = [int(item["seed"]) for item in comparisons]
    if len(set(seeds)) != len(seeds):
        raise ValueError("comparison seeds must be unique")

    invariant_keys = ("protocol", "train_examples_per_variant", "dev_impressions")
    for key in invariant_keys:
        values = {item.get(key) for item in comparisons}
        if len(values) != 1:
            raise ValueError(f"comparison {key} values do not match")

    first_variants = comparisons[0].get("variants")
    if not isinstance(first_variants, dict):
        raise ValueError("comparison variants must be an object")
    variant_names = sorted(first_variants)
    if not variant_names:
        raise ValueError("comparison variants must not be empty")
    for item in comparisons[1:]:
        variants = item.get("variants")
        if not isinstance(variants, dict) or sorted(variants) != variant_names:
            raise ValueError("comparison variant sets do not match")

    variant_summary: dict[str, JsonValue] = {}
    for variant_name in variant_names:
        durations: list[float] = []
        metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        parameter_counts: set[int] = set()
        for item in comparisons:
            variant = cast(dict[str, Any], cast(dict[str, Any], item["variants"])[variant_name])
            durations.append(_number(variant, "duration_seconds", context=variant_name))
            parameters = variant.get("parameters")
            if not isinstance(parameters, int) or isinstance(parameters, bool):
                raise ValueError(f"{variant_name}.parameters must be an integer")
            parameter_counts.add(parameters)
            metrics = variant.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"{variant_name}.metrics must be an object")
            for metric in METRICS:
                metric_values[metric].append(
                    _number(metrics, metric, context=f"{variant_name}.metrics")
                )
        if len(parameter_counts) != 1:
            raise ValueError(f"{variant_name} parameter counts do not match")
        variant_summary[variant_name] = {
            "parameters": next(iter(parameter_counts)),
            "duration_seconds": _summary(durations),
            "metrics": {metric: _summary(metric_values[metric]) for metric in METRICS},
        }

    effect_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    bootstrap_hashes: dict[str, JsonValue] = {}
    for seed, item in zip(seeds, comparisons, strict=True):
        bootstrap_reference = item.get("paired_bootstrap")
        if not isinstance(bootstrap_reference, str):
            raise ValueError("paired_bootstrap must be a repository-relative path")
        bootstrap_path = repository_root / bootstrap_reference
        bootstrap = _load_object(bootstrap_path)
        if bootstrap.get("seed") != seed:
            raise ValueError("paired bootstrap seed does not match comparison seed")
        bootstrap_metrics = bootstrap.get("metrics")
        if not isinstance(bootstrap_metrics, dict):
            raise ValueError("paired bootstrap metrics must be an object")
        for metric in METRICS:
            metric_payload = bootstrap_metrics.get(metric)
            if not isinstance(metric_payload, dict):
                raise ValueError(f"paired bootstrap metric {metric} is missing")
            effect_values[metric].append(
                _number(metric_payload, "mean_difference", context=f"bootstrap.{metric}")
            )
        bootstrap_hashes[str(seed)] = sha256_file(bootstrap_path)

    comparison_hashes: dict[str, JsonValue] = {
        str(seed): sha256_file(path) for seed, path in zip(seeds, comparison_paths, strict=True)
    }
    input_hashes: dict[str, JsonValue] = {
        "comparisons": comparison_hashes,
        "paired_bootstraps": bootstrap_hashes,
    }
    return {
        "schema_version": 1,
        "status": "full_data_three_seed_aggregate",
        "protocol": cast(JsonValue, comparisons[0]["protocol"]),
        "seeds": cast(list[JsonValue], seeds),
        "seed_count": len(seeds),
        "train_examples_per_variant_per_seed": int(comparisons[0]["train_examples_per_variant"]),
        "dev_impressions_per_seed": int(comparisons[0]["dev_impressions"]),
        "input_sha256": input_hashes,
        "variants": variant_summary,
        "paired_effects_attention_minus_mean": {
            metric: _summary(effect_values[metric]) for metric in METRICS
        },
        "all_seed_effects_positive": {
            metric: all(value > 0 for value in effect_values[metric]) for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compatible L1 seed comparisons.")
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_l1_comparisons(
        args.comparisons, repository_root=args.repository_root.resolve()
    )
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


if __name__ == "__main__":
    main()
