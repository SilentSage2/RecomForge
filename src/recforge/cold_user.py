"""Registered candidate-only fallback for empty-history MIND impressions."""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from recforge.calibration import fit_monotonic_platt, probability_metrics
from recforge.checkpoints import serialize_state_dict
from recforge.data.features import ItemFeatureTable
from recforge.data.mind import (
    MindBehavior,
    MindNews,
    iter_mind_behaviors,
    iter_mind_news,
    sha256_file,
)
from recforge.data.pretrained import load_pretrained_artifact
from recforge.data.submission import (
    order_from_scores,
    ranks_from_scores,
    validate_prediction_file,
    write_prediction_file,
)
from recforge.evaluation import positive_target_popularity, time_decayed_target_popularity
from recforge.logged_baselines import rank_scores
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.models.propensity import CandidatePropensityHead
from recforge.statistics import paired_bootstrap
from recforge.tracking import JsonValue, record_run, sha256_bytes

_METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


@dataclass(frozen=True, slots=True)
class ColdUserConfig:
    feature_artifact: str
    fit_behaviors: str
    calibration_behaviors: str
    dev_behaviors: str
    train_news: str
    dev_news: str
    base_prediction: str
    output_root: str = "runs"
    seed: int = 2027
    epochs: int = 3
    batch_size: int = 8192
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    half_life_hours: float = 72.0
    bootstrap_resamples: int = 5000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.bootstrap_resamples <= 0:
            raise ValueError("training and bootstrap sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.half_life_hours <= 0:
            raise ValueError("optimizer rates and half life are invalid")
        if self.device != "cpu":
            raise ValueError("the registered cold-user experiment uses CPU")


@dataclass(frozen=True, slots=True)
class PropensityDataset:
    item_rows: NDArray[np.int64]
    labels: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.item_rows.ndim != 1 or self.labels.shape != self.item_rows.shape:
            raise ValueError("propensity rows and labels must be equal vectors")
        if len(self.labels) == 0 or not np.isin(self.labels, (0.0, 1.0)).all():
            raise ValueError("propensity labels must be nonempty and binary")


def _peak_resident_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def build_propensity_dataset(behavior_path: Path, table: ItemFeatureTable) -> PropensityDataset:
    example_count = sum(
        len(behavior.candidate_item_ids) for behavior in iter_mind_behaviors(behavior_path)
    )
    rows = np.empty(example_count, dtype=np.int64)
    labels = np.empty(example_count, dtype=np.float32)
    row_by_item = table.row_by_item_id()
    offset = 0
    for behavior in iter_mind_behaviors(behavior_path):
        count = len(behavior.candidate_item_ids)
        try:
            rows[offset : offset + count] = [
                row_by_item[item] for item in behavior.candidate_item_ids
            ]
        except KeyError as error:
            raise ValueError(f"candidate {error.args[0]!r} has no frozen features") from error
        labels[offset : offset + count] = behavior.labels
        offset += count
    return PropensityDataset(rows, labels)


def train_propensity_head(
    config: ColdUserConfig,
    dataset: PropensityDataset,
    table: ItemFeatureTable,
    *,
    progress: bool = False,
) -> tuple[CandidatePropensityHead, list[float], list[float], float]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = CandidatePropensityHead(table.dimension)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    positive_count = float(dataset.labels.sum())
    negative_count = len(dataset.labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("propensity training requires both classes")
    positive_weight = negative_count / positive_count
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    losses: list[float] = []
    epoch_seconds: list[float] = []
    for epoch in range(config.epochs):
        started = time.perf_counter()
        order = np.random.default_rng(config.seed + epoch).permutation(len(dataset.labels))
        loss_total = 0.0
        model.train()
        for start in range(0, len(order), config.batch_size):
            selected = order[start : start + config.batch_size]
            feature_batch = torch.tensor(
                np.asarray(table.features[dataset.item_rows[selected]]), dtype=torch.float32
            )
            label_batch = torch.from_numpy(dataset.labels[selected])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(selected)
        losses.append(loss_total / len(order))
        epoch_seconds.append(time.perf_counter() - started)
        if progress:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss": losses[-1],
                        "seconds": epoch_seconds[-1],
                    }
                ),
                flush=True,
            )
    return model, losses, epoch_seconds, positive_weight


def score_all_items(
    model: CandidatePropensityHead, table: ItemFeatureTable, *, batch_size: int = 8192
) -> NDArray[np.float64]:
    chunks: list[NDArray[np.float64]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(table.item_ids), batch_size):
            features = torch.tensor(
                np.asarray(table.features[start : start + batch_size]), dtype=torch.float32
            )
            chunks.append(model(features).numpy().astype(np.float64))
    return np.concatenate(chunks)


def _read_prediction_ranks(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    result: list[tuple[str, tuple[int, ...]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            impression_id, raw_ranks = line.rstrip("\n").split(maxsplit=1)
            ranks: Any = json.loads(raw_ranks)
            if not isinstance(ranks, list) or not all(
                isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks
            ):
                raise ValueError("base prediction has invalid ranks")
            result.append((impression_id, tuple(ranks)))
    return result


def _metric_row(behavior: MindBehavior, scores: list[float]) -> NDArray[np.float64]:
    positives = {
        item
        for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
        if label == 1
    }
    ranked = [behavior.candidate_item_ids[index] for index in order_from_scores(scores)]
    return np.asarray(
        [
            binary_auc(behavior.labels, scores),
            mean_reciprocal_rank_at_k(ranked, positives, len(ranked)),
            ndcg_at_k(ranked, positives, 5),
            ndcg_at_k(ranked, positives, 10),
        ],
        dtype=np.float64,
    )


def _summarize_rows(rows: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        **{metric: float(value) for metric, value in zip(_METRICS, rows.mean(axis=0), strict=True)},
    }


def _coverage(behaviors: list[MindBehavior], predictions: list[list[float]], *, k: int) -> float:
    catalog = {item for behavior in behaviors for item in behavior.candidate_item_ids}
    recommended = {
        behavior.candidate_item_ids[index]
        for behavior, scores in zip(behaviors, predictions, strict=True)
        for index in order_from_scores(scores)[:k]
    }
    return len(recommended) / len(catalog)


def _load_news(paths: list[Path]) -> dict[str, MindNews]:
    result: dict[str, MindNews] = {}
    for path in paths:
        for news in iter_mind_news(path):
            result.setdefault(news.news_id, news)
    return result


def _qualitative_cases(
    behaviors: list[MindBehavior],
    head_scores: list[list[float]],
    popularity_scores: list[list[float]],
    news: dict[str, MindNews],
) -> dict[str, list[dict[str, Any]]]:
    cases: list[tuple[float, str, dict[str, Any]]] = []
    for behavior, head, popularity in zip(behaviors, head_scores, popularity_scores, strict=True):
        positives = [
            item
            for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
            if label == 1
        ]
        head_ranks = ranks_from_scores(head)
        popularity_ranks = ranks_from_scores(popularity)
        head_mrr = _metric_row(behavior, head)[1]
        popularity_mrr = _metric_row(behavior, popularity)[1]
        payload = {
            "impression_id": behavior.impression_id,
            "candidate_count": len(behavior.candidate_item_ids),
            "positive_categories": sorted({news[item].category for item in positives}),
            "head_positive_ranks": [
                head_ranks[index] for index, label in enumerate(behavior.labels) if label == 1
            ],
            "popularity_positive_ranks": [
                popularity_ranks[index] for index, label in enumerate(behavior.labels) if label == 1
            ],
            "mrr_difference": float(head_mrr - popularity_mrr),
        }
        cases.append((float(head_mrr - popularity_mrr), behavior.impression_id, payload))
    ordered = sorted(cases, key=lambda item: (item[0], item[1]))
    gains = sorted(cases, key=lambda item: (-item[0], item[1]))
    return {
        "largest_regret": [item[2] for item in ordered[:5]],
        "largest_gain": [item[2] for item in gains[:5]],
    }


def run_cold_user_experiment(config: ColdUserConfig, repository_root: Path) -> Path:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    artifact_directory = repository_root / config.feature_artifact
    fit_path = repository_root / config.fit_behaviors
    calibration_path = repository_root / config.calibration_behaviors
    dev_path = repository_root / config.dev_behaviors
    base_prediction_path = repository_root / config.base_prediction
    table, artifact_manifest = load_pretrained_artifact(artifact_directory)
    dataset = build_propensity_dataset(fit_path, table)
    model, losses, epoch_seconds, positive_weight = train_propensity_head(
        config, dataset, table, progress=True
    )
    score_started = time.perf_counter()
    item_scores = score_all_items(model, table)
    item_score_seconds = time.perf_counter() - score_started
    row_by_item = table.row_by_item_id()
    validate_prediction_file(dev_path, base_prediction_path)
    base_ranks = _read_prediction_ranks(base_prediction_path)
    dev_behaviors = list(iter_mind_behaviors(dev_path))
    if len(base_ranks) != len(dev_behaviors):
        raise ValueError("base predictions and dev behaviors differ")

    reference_time = max(behavior.local_timestamp for behavior in iter_mind_behaviors(fit_path))
    global_counts: Counter[str] = positive_target_popularity(fit_path)
    decayed_counts = time_decayed_target_popularity(
        fit_path, reference_time=reference_time, half_life_hours=config.half_life_hours
    )
    empty_behaviors: list[MindBehavior] = []
    tie_scores: list[list[float]] = []
    head_rank_scores: list[list[float]] = []
    head_raw_scores: list[list[float]] = []
    global_scores: list[list[float]] = []
    decayed_scores: list[list[float]] = []
    hybrid_predictions: list[tuple[str, list[float]]] = []
    base_predictions: list[list[float]] = []
    nonempty_identity = True
    hybrid_started = time.perf_counter()
    for behavior, (base_id, ranks) in zip(dev_behaviors, base_ranks, strict=True):
        if behavior.impression_id != base_id:
            raise ValueError("base prediction order differs from dev behaviors")
        base_score = [-float(rank) for rank in ranks]
        base_predictions.append(base_score)
        if behavior.history_item_ids:
            hybrid_predictions.append((behavior.impression_id, base_score))
            continue
        raw = [float(item_scores[row_by_item[item]]) for item in behavior.candidate_item_ids]
        resolved_head = rank_scores(
            behavior.candidate_item_ids,
            {item: score for item, score in zip(behavior.candidate_item_ids, raw, strict=True)},
        )
        empty_behaviors.append(behavior)
        tie_scores.append([0.0] * len(raw))
        head_raw_scores.append(raw)
        head_rank_scores.append(resolved_head)
        global_scores.append(rank_scores(behavior.candidate_item_ids, global_counts))
        decayed_scores.append(rank_scores(behavior.candidate_item_ids, decayed_counts))
        hybrid_predictions.append((behavior.impression_id, resolved_head))
    hybrid_seconds = time.perf_counter() - hybrid_started
    for behavior, (_, hybrid), base in zip(
        dev_behaviors, hybrid_predictions, base_predictions, strict=True
    ):
        if behavior.history_item_ids and ranks_from_scores(hybrid) != ranks_from_scores(base):
            nonempty_identity = False

    def rows(predictions: list[list[float]]) -> NDArray[np.float64]:
        return np.stack(
            [
                _metric_row(behavior, scores)
                for behavior, scores in zip(empty_behaviors, predictions, strict=True)
            ]
        )

    tie_rows = rows(tie_scores)
    head_rows = rows(head_rank_scores)
    global_rows = rows(global_scores)
    decayed_rows = rows(decayed_scores)
    popularity_name, popularity_rows, popularity_predictions = max(
        (
            ("global", global_rows, global_scores),
            ("time_decayed_72h", decayed_rows, decayed_scores),
        ),
        key=lambda item: float(item[1][:, 0].mean()),
    )
    full_base_rows = np.stack(
        [
            _metric_row(behavior, scores)
            for behavior, scores in zip(dev_behaviors, base_predictions, strict=True)
        ]
    )
    full_hybrid_rows = np.stack(
        [
            _metric_row(behavior, scores)
            for behavior, (_, scores) in zip(dev_behaviors, hybrid_predictions, strict=True)
        ]
    )

    calibration_raw = [
        [float(item_scores[row_by_item[item]]) for item in behavior.candidate_item_ids]
        for behavior in iter_mind_behaviors(calibration_path)
    ]
    calibration_labels = np.asarray(
        [label for behavior in iter_mind_behaviors(calibration_path) for label in behavior.labels],
        dtype=np.float64,
    )
    calibration_flat = np.asarray(
        [score for impression in calibration_raw for score in impression], dtype=np.float64
    )
    scale, bias = fit_monotonic_platt(calibration_flat, calibration_labels)
    dev_raw = np.asarray(
        [
            float(item_scores[row_by_item[item]])
            for behavior in dev_behaviors
            for item in behavior.candidate_item_ids
        ],
        dtype=np.float64,
    )
    dev_labels = np.asarray(
        [label for behavior in dev_behaviors for label in behavior.labels], dtype=np.float64
    )
    empty_raw = np.asarray(
        [score for impression in head_raw_scores for score in impression], dtype=np.float64
    )
    empty_labels = np.asarray(
        [label for behavior in empty_behaviors for label in behavior.labels], dtype=np.float64
    )
    prevalence = float(calibration_labels.mean())

    def sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-np.clip(values, -50, 50)))

    def calibration_summary(
        scores: NDArray[np.float64], labels: NDArray[np.float64]
    ) -> dict[str, Any]:
        return {
            "candidate_count": len(scores),
            "positive_prevalence": float(labels.mean()),
            "raw": probability_metrics(labels, sigmoid(scores)),
            "calibrated": probability_metrics(labels, sigmoid(scale * scores + bias)),
            "calibration_prevalence_constant": probability_metrics(
                labels, np.full_like(scores, prevalence)
            ),
        }

    checkpoint = serialize_state_dict(model)
    metrics: dict[str, JsonValue] = {
        "protocol": "mind_empty_history_candidate_propensity",
        "fit_candidate_exposures": len(dataset.labels),
        "fit_positive_weight": positive_weight,
        "epoch_losses": cast(list[JsonValue], losses),
        "epoch_seconds": cast(list[JsonValue], epoch_seconds),
        "item_score_precompute_seconds": item_score_seconds,
        "hybrid_construction_seconds": hybrid_seconds,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_resident_memory_bytes": _peak_resident_memory_bytes(),
        "checkpoint_sha256": sha256_bytes(checkpoint),
        "nonempty_rank_identity": nonempty_identity,
        "empty_history": cast(
            dict[str, JsonValue],
            {
                "impression_count": len(empty_behaviors),
                "displayed_catalog_size": len(
                    {item for behavior in empty_behaviors for item in behavior.candidate_item_ids}
                ),
                "original_tie": {
                    "metrics": _summarize_rows(tie_rows),
                    "coverage@1": _coverage(empty_behaviors, tie_scores, k=1),
                    "coverage@5": _coverage(empty_behaviors, tie_scores, k=5),
                    "tied_score_impression_rate": 1.0,
                },
                "global_popularity": {
                    "metrics": _summarize_rows(global_rows),
                    "coverage@1": _coverage(empty_behaviors, global_scores, k=1),
                    "coverage@5": _coverage(empty_behaviors, global_scores, k=5),
                    "tied_score_impression_rate": sum(
                        len({global_counts[item] for item in behavior.candidate_item_ids})
                        != len(behavior.candidate_item_ids)
                        for behavior in empty_behaviors
                    )
                    / len(empty_behaviors),
                },
                "time_decayed_popularity_72h": {
                    "metrics": _summarize_rows(decayed_rows),
                    "coverage@1": _coverage(empty_behaviors, decayed_scores, k=1),
                    "coverage@5": _coverage(empty_behaviors, decayed_scores, k=5),
                    "tied_score_impression_rate": sum(
                        len({decayed_counts.get(item, 0.0) for item in behavior.candidate_item_ids})
                        != len(behavior.candidate_item_ids)
                        for behavior in empty_behaviors
                    )
                    / len(empty_behaviors),
                },
                "candidate_head": {
                    "metrics": _summarize_rows(head_rows),
                    "coverage@1": _coverage(empty_behaviors, head_rank_scores, k=1),
                    "coverage@5": _coverage(empty_behaviors, head_rank_scores, k=5),
                    "tied_score_impression_rate": sum(
                        len(set(scores)) != len(scores) for scores in head_raw_scores
                    )
                    / len(head_raw_scores),
                },
                "paired_bootstrap_head_minus_tie": paired_bootstrap(
                    tie_rows,
                    head_rows,
                    resamples=config.bootstrap_resamples,
                    seed=config.seed,
                ),
                "stronger_popularity_baseline": popularity_name,
                "paired_bootstrap_head_minus_stronger_popularity": paired_bootstrap(
                    popularity_rows,
                    head_rows,
                    resamples=config.bootstrap_resamples,
                    seed=config.seed,
                ),
            },
        ),
        "full_dev": cast(
            dict[str, JsonValue],
            {
                "original": _summarize_rows(full_base_rows),
                "hybrid": _summarize_rows(full_hybrid_rows),
                "effect": dict(
                    (
                        metric,
                        float(value),
                    )
                    for metric, value in zip(
                        _METRICS, (full_hybrid_rows - full_base_rows).mean(axis=0), strict=True
                    )
                ),
                "paired_bootstrap_hybrid_minus_original": paired_bootstrap(
                    full_base_rows,
                    full_hybrid_rows,
                    resamples=config.bootstrap_resamples,
                    seed=config.seed,
                ),
            },
        ),
        "calibration": cast(
            dict[str, JsonValue],
            {
                "positive_scale": scale,
                "bias": bias,
                "dev": calibration_summary(dev_raw, dev_labels),
                "empty_history_dev": calibration_summary(empty_raw, empty_labels),
                "rank_preserving": scale > 0,
            },
        ),
        "qualitative_cases": cast(
            dict[str, JsonValue],
            _qualitative_cases(
                empty_behaviors,
                head_rank_scores,
                popularity_predictions,
                _load_news(
                    [repository_root / config.train_news, repository_root / config.dev_news]
                ),
            ),
        ),
    }
    fingerprints = {
        "fit_behaviors": sha256_file(fit_path),
        "calibration_behaviors": sha256_file(calibration_path),
        "dev_behaviors": sha256_file(dev_path),
        "feature_manifest": sha256_file(artifact_directory / "manifest.json"),
        "features": cast(str, artifact_manifest["feature_sha256"]),
        "base_prediction": sha256_file(base_prediction_path),
        "experiment_spec": sha256_file(repository_root / "docs/L3_COLD_USER_SPEC.md"),
    }
    run_directory = record_run(
        output_root=repository_root / config.output_root,
        repository_root=repository_root,
        experiment="mind-cold-user-propensity",
        metrics=metrics,
        config=cast(dict[str, JsonValue], asdict(config)),
        dataset_fingerprints=fingerprints,
        seed=config.seed,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_seconds=time.perf_counter() - started,
    )
    (run_directory / "model.pt").write_bytes(checkpoint)
    write_prediction_file(run_directory / "hybrid-dev-prediction.txt", hybrid_predictions)
    return run_directory


def _load_config(path: Path) -> ColdUserConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cold-user config must be a JSON object")
    return ColdUserConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the registered cold-user fallback.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    run_directory = run_cold_user_experiment(config, args.repository_root.resolve())
    print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
