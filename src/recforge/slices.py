"""Registered paired failure slices for official MIND ranking predictions."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from recforge.data.mind import (
    MindBehavior,
    MindNews,
    iter_mind_behaviors,
    iter_mind_news,
    sha256_file,
)
from recforge.data.text import tokenize_title
from recforge.statistics import per_impression_metrics
from recforge.tracking import canonical_json_bytes, collect_git_state

_METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _quantile_cutoffs(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("quantile values must not be empty")
    result = np.quantile(np.asarray(values), (1 / 3, 2 / 3), method="nearest")
    return float(result[0]), float(result[1])


def _three_way(value: float, cutoffs: tuple[float, float]) -> str:
    if value <= cutoffs[0]:
        return "low"
    if value <= cutoffs[1]:
        return "middle"
    return "high"


def _frequency_band(value: int, cutoffs: tuple[float, float]) -> str:
    if value == 0:
        return "cold"
    if value <= cutoffs[0]:
        return "tail"
    if value <= cutoffs[1]:
        return "middle"
    return "head"


def _load_news(paths: list[Path]) -> dict[str, MindNews]:
    result: dict[str, MindNews] = {}
    for source in paths:
        for news in iter_mind_news(source):
            existing = result.get(news.news_id)
            if existing is not None and existing != news:
                raise ValueError(f"conflicting news metadata for {news.news_id!r}")
            result[news.news_id] = news
    return result


def _publisher(news: MindNews) -> str:
    return urlparse(news.url).hostname or "<unknown>"


def _history_slice(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 9:
        return "1-9"
    if value <= 24:
        return "10-24"
    if value <= 49:
        return "25-49"
    return "50+"


def _candidate_count_slice(value: int) -> str:
    if value <= 9:
        return "2-9"
    if value <= 19:
        return "10-19"
    if value <= 49:
        return "20-49"
    return "50+"


def _position_slice(value: int) -> str:
    if value <= 4:
        return "1-4"
    if value <= 9:
        return "5-9"
    if value <= 19:
        return "10-19"
    return "20+"


def _age_slice(age_hours: float | None) -> str:
    if age_hours is None:
        return "cold"
    if age_hours <= 6:
        return "0-6h"
    if age_hours <= 24:
        return "6-24h"
    if age_hours <= 72:
        return "1-3d"
    return "3d+"


def _first_seen_train(train_behaviors: Path) -> dict[str, datetime]:
    first_seen: dict[str, datetime] = {}
    for behavior in iter_mind_behaviors(train_behaviors):
        for item_id in behavior.candidate_item_ids:
            prior = first_seen.get(item_id)
            if prior is None or behavior.local_timestamp < prior:
                first_seen[item_id] = behavior.local_timestamp
    return first_seen


def _positive_age_slices(
    dev_behaviors: tuple[MindBehavior, ...], first_seen: dict[str, datetime]
) -> list[str]:
    labels: list[str] = []
    index = 0
    while index < len(dev_behaviors):
        timestamp = dev_behaviors[index].local_timestamp
        end = index
        while end < len(dev_behaviors) and dev_behaviors[end].local_timestamp == timestamp:
            end += 1
        for behavior in dev_behaviors[index:end]:
            positive_ids = [
                item
                for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
                if label == 1
            ]
            known_ages = [
                (timestamp - first_seen[item]).total_seconds() / 3600
                for item in positive_ids
                if item in first_seen and first_seen[item] < timestamp
            ]
            labels.append(_age_slice(min(known_ages) if known_ages else None))
        for behavior in dev_behaviors[index:end]:
            for item_id in behavior.candidate_item_ids:
                first_seen.setdefault(item_id, timestamp)
        index = end
    return labels


def _bootstrap_slice(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
    family_slice_count: int,
) -> dict[str, Any]:
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(_METRICS)), dtype=np.float64)
    for start in range(0, resamples, 32):
        count = min(32, resamples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        draws[start : start + count] = differences[indices].mean(axis=1)
    family_alpha = 0.05 / family_slice_count
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(_METRICS):
        result[metric] = {
            "baseline_mean": float(baseline[:, metric_index].mean()),
            "candidate_mean": float(candidate[:, metric_index].mean()),
            "mean_difference": float(differences[:, metric_index].mean()),
            "confidence_95": [
                float(np.quantile(draws[:, metric_index], 0.025)),
                float(np.quantile(draws[:, metric_index], 0.975)),
            ],
            "confidence_familywise_bonferroni": [
                float(np.quantile(draws[:, metric_index], family_alpha / 2)),
                float(np.quantile(draws[:, metric_index], 1 - family_alpha / 2)),
            ],
            "bootstrap_probability_improvement": float((draws[:, metric_index] > 0).mean()),
        }
    return result


def _stable_seed(base_seed: int, family: str, label: str) -> int:
    digest = hashlib.blake2b(f"{family}:{label}".encode(), digest_size=4).digest()
    return base_seed + int.from_bytes(digest, "little")


def _evaluate_family(
    family: str,
    labels: list[str],
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    unique_labels = sorted(set(labels))
    slices: dict[str, Any] = {}
    label_array = np.asarray(labels)
    for label in unique_labels:
        selected = label_array == label
        sample_count = int(selected.sum())
        if sample_count < 2:
            slices[label] = {"sample_count": sample_count, "status": "insufficient"}
            continue
        slices[label] = {
            "sample_count": sample_count,
            "exploratory_below_200": sample_count < 200,
            "metrics": _bootstrap_slice(
                baseline[selected],
                candidate[selected],
                resamples=resamples,
                seed=_stable_seed(seed, family, label),
                family_slice_count=len(unique_labels),
            ),
        }
    return {
        "slice_count": len(unique_labels),
        "familywise_method": "within_family_bonferroni_percentile_bootstrap",
        "slices": slices,
    }


def build_slice_labels(
    train_behaviors_path: Path,
    train_news_path: Path,
    dev_behaviors: tuple[MindBehavior, ...],
    dev_news_path: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    train_news = _load_news([train_news_path])
    all_news = _load_news([train_news_path, dev_news_path])
    item_exposures: Counter[str] = Counter()
    publisher_exposures: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    for news in train_news.values():
        token_counts.update(tokenize_title(news.title))
    for behavior in iter_mind_behaviors(train_behaviors_path):
        item_exposures.update(behavior.candidate_item_ids)
        for item_id in behavior.candidate_item_ids:
            publisher_exposures[_publisher(all_news[item_id])] += 1

    item_cutoffs = _quantile_cutoffs([float(value) for value in item_exposures.values()])
    publisher_cutoffs = _quantile_cutoffs([float(value) for value in publisher_exposures.values()])
    candidate_popularity = [
        float(np.mean([math.log1p(item_exposures[item]) for item in behavior.candidate_item_ids]))
        for behavior in dev_behaviors
    ]
    candidate_popularity_cutoffs = _quantile_cutoffs(candidate_popularity)
    age_labels = _positive_age_slices(dev_behaviors, _first_seen_train(train_behaviors_path))

    labels: dict[str, list[str]] = {
        "history_length": [],
        "candidate_set_size": [],
        "positive_exposure_age": age_labels,
        "positive_item_popularity": [],
        "candidate_set_popularity": [],
        "positive_publisher_popularity": [],
        "positive_title_rarity": [],
        "first_positive_logged_position": [],
    }
    for behavior, candidate_popularity_value in zip(
        dev_behaviors, candidate_popularity, strict=True
    ):
        positive_positions = [
            index for index, label in enumerate(behavior.labels, start=1) if label == 1
        ]
        positive_ids = [
            item
            for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        ]
        if not positive_ids:
            raise ValueError(f"impression {behavior.impression_id} has no positive")
        item_popularity = max(item_exposures[item] for item in positive_ids)
        publisher_popularity = max(
            publisher_exposures[_publisher(all_news[item])] for item in positive_ids
        )
        if any(item not in train_news for item in positive_ids):
            rarity = "dev_only"
        else:
            rare_fractions = []
            for item in positive_ids:
                tokens = tokenize_title(train_news[item].title)
                rare_fractions.append(
                    sum(token_counts[token] <= 5 for token in tokens) / len(tokens)
                    if tokens
                    else 0.0
                )
            rarity = "seen_high_rare" if max(rare_fractions) >= 0.5 else "seen_low_rare"
        labels["history_length"].append(_history_slice(len(behavior.history_item_ids)))
        labels["candidate_set_size"].append(
            _candidate_count_slice(len(behavior.candidate_item_ids))
        )
        labels["positive_item_popularity"].append(_frequency_band(item_popularity, item_cutoffs))
        labels["candidate_set_popularity"].append(
            _three_way(candidate_popularity_value, candidate_popularity_cutoffs)
        )
        labels["positive_publisher_popularity"].append(
            _frequency_band(publisher_popularity, publisher_cutoffs)
        )
        labels["positive_title_rarity"].append(rarity)
        labels["first_positive_logged_position"].append(_position_slice(min(positive_positions)))
    metadata = {
        "item_exposure_tertile_cutoffs": list(item_cutoffs),
        "publisher_exposure_tertile_cutoffs": list(publisher_cutoffs),
        "candidate_set_mean_log1p_exposure_tertile_cutoffs": list(candidate_popularity_cutoffs),
        "rare_token_training_frequency_max": 5,
        "high_rare_fraction_min": 0.5,
        "recency_interpretation": "logged_candidate_exposure_age_proxy",
    }
    return labels, metadata


def evaluate_slices(
    *,
    train_behaviors_path: Path,
    train_news_path: Path,
    dev_behaviors_path: Path,
    dev_news_path: Path,
    baseline_prediction_path: Path,
    candidate_prediction_path: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    behaviors = tuple(iter_mind_behaviors(dev_behaviors_path))
    baseline = per_impression_metrics(dev_behaviors_path, baseline_prediction_path)
    candidate = per_impression_metrics(dev_behaviors_path, candidate_prediction_path)
    if baseline.shape[0] != len(behaviors) or candidate.shape != baseline.shape:
        raise ValueError("prediction metrics and behaviors are misaligned")
    labels, thresholds = build_slice_labels(
        train_behaviors_path, train_news_path, behaviors, dev_news_path
    )
    families = {
        family: _evaluate_family(
            family,
            family_labels,
            baseline,
            candidate,
            resamples=resamples,
            seed=seed,
        )
        for family, family_labels in labels.items()
    }
    return {
        "schema_version": 1,
        "protocol": "registered_l2_failure_slices",
        "comparison": "frozen_minilm_projection_minus_l1_title_attention",
        "sample_count": len(behaviors),
        "metrics": list(_METRICS),
        "resamples": resamples,
        "seed": seed,
        "multiplicity_note": (
            "Interpret direction only when retained by the within-family Bonferroni interval; "
            "slice attributes are correlated and observational."
        ),
        "thresholds": thresholds,
        "input_sha256": {
            "train_behaviors": sha256_file(train_behaviors_path),
            "train_news": sha256_file(train_news_path),
            "dev_behaviors": sha256_file(dev_behaviors_path),
            "dev_news": sha256_file(dev_news_path),
            "baseline_prediction": sha256_file(baseline_prediction_path),
            "candidate_prediction": sha256_file(candidate_prediction_path),
        },
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate registered paired MIND slices.")
    parser.add_argument("--train-behaviors", type=Path, required=True)
    parser.add_argument("--train-news", type=Path, required=True)
    parser.add_argument("--dev-behaviors", type=Path, required=True)
    parser.add_argument("--dev-news", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    payload = evaluate_slices(
        train_behaviors_path=args.train_behaviors,
        train_news_path=args.train_news,
        dev_behaviors_path=args.dev_behaviors,
        dev_news_path=args.dev_news,
        baseline_prediction_path=args.baseline,
        candidate_prediction_path=args.candidate,
        resamples=args.resamples,
        seed=args.seed,
    )
    specification = args.repository_root / "docs/L2_FAILURE_SLICE_SPEC.md"
    payload["provenance"] = {
        "git": asdict(collect_git_state(args.repository_root)),
        "specification_sha256": sha256_file(specification),
    }
    args.output.write_bytes(canonical_json_bytes(payload))


if __name__ == "__main__":
    main()
