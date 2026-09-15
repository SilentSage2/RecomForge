"""Training-only temporal split and monotonic score calibration for L2."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from recforge.data.mind import MIND_TIME_FORMAT, iter_mind_behaviors, sha256_file
from recforge.data.pretrained import load_pretrained_artifact
from recforge.data.submission import ranks_from_scores
from recforge.l2_experiment import L2ExperimentConfig, evaluate_frozen_ranker
from recforge.models.pretrained_ranker import FrozenFeatureRanker
from recforge.tracking import canonical_json_bytes, collect_git_state


def _peak_resident_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def split_calibration_behaviors(
    source: Path, output_directory: Path, *, cutoff: datetime, repository_root: Path
) -> dict[str, Any]:
    """Partition exact source rows by timestamp while preserving within-split order."""
    if cutoff.tzinfo is not None:
        raise ValueError("cutoff must use the MIND timezone-naive local clock")
    output_directory.mkdir(parents=True, exist_ok=False)
    fit_path = output_directory / "fit.tsv"
    calibration_path = output_directory / "calibration.tsv"
    fit_count = calibration_count = inversions = 0
    last_timestamp: datetime | None = None
    with (
        source.open(encoding="utf-8") as source_handle,
        fit_path.open("x", encoding="utf-8") as fit_handle,
        calibration_path.open("x", encoding="utf-8") as calibration_handle,
    ):
        for line_number, line in enumerate(source_handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{source}:{line_number} expected five columns")
            try:
                timestamp = datetime.strptime(fields[2], MIND_TIME_FORMAT)
            except ValueError as error:
                raise ValueError(f"{source}:{line_number} has an invalid timestamp") from error
            inversions += last_timestamp is not None and timestamp < last_timestamp
            last_timestamp = timestamp
            if timestamp < cutoff:
                fit_handle.write(line)
                fit_count += 1
            else:
                calibration_handle.write(line)
                calibration_count += 1
    if fit_count == 0 or calibration_count == 0:
        raise ValueError("temporal calibration split produced an empty partition")
    manifest = {
        "schema_version": 1,
        "protocol": "training_only_temporal_calibration_split",
        "cutoff_local_exclusive_for_fit": cutoff.isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "fit": {
            "path": str(fit_path),
            "impression_count": fit_count,
            "sha256": sha256_file(fit_path),
        },
        "calibration": {
            "path": str(calibration_path),
            "impression_count": calibration_count,
            "sha256": sha256_file(calibration_path),
        },
        "source_timestamp_inversion_count": inversions,
        "git": asdict(collect_git_state(repository_root)),
    }
    (output_directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def probability_metrics(
    labels: NDArray[np.float64], probabilities: NDArray[np.float64], *, bins: int = 15
) -> dict[str, float]:
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be equal vectors")
    if len(labels) == 0 or bins <= 0:
        raise ValueError("probability metrics require samples and positive bin count")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    brier = np.mean(np.square(probabilities - labels))
    bin_ids = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        selected = bin_ids == bin_index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probabilities[selected].mean() - labels[selected].mean())
            )
    return {"nll": float(nll), "brier": float(brier), "ece_15_equal_width": ece}


def fit_monotonic_platt(
    scores: NDArray[np.float64], labels: NDArray[np.float64]
) -> tuple[float, float]:
    if scores.shape != labels.shape or scores.ndim != 1 or len(scores) == 0:
        raise ValueError("calibration scores and labels must be equal nonempty vectors")
    score_tensor = torch.from_numpy(scores)
    label_tensor = torch.from_numpy(labels)
    log_scale = torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float64, requires_grad=True)
    prevalence = float(labels.mean())
    if not 0 < prevalence < 1:
        raise ValueError("calibration labels require both classes")
    bias = torch.tensor(
        math.log(prevalence / (1 - prevalence)), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], max_iter=100, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(log_scale) + 1e-8
        logits = scale * score_tensor + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return float(torch.nn.functional.softplus(log_scale).detach() + 1e-8), float(bias.detach())


def _score_arrays(
    predictions: list[tuple[str, list[float]]], behavior_path: Path
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scores: list[float] = []
    labels: list[float] = []
    behaviors = iter_mind_behaviors(behavior_path)
    for (impression_id, impression_scores), behavior in zip(predictions, behaviors, strict=True):
        if impression_id != behavior.impression_id:
            raise ValueError("score predictions and behaviors are misaligned")
        if len(impression_scores) != len(behavior.labels):
            raise ValueError("score and label candidate counts differ")
        scores.extend(impression_scores)
        labels.extend(float(label) for label in behavior.labels)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1 + negative_exp)
    return result


def evaluate_calibration(
    *,
    config_path: Path,
    checkpoint_path: Path,
    feature_artifact: Path,
    calibration_behaviors: Path,
    dev_behaviors: Path,
    repository_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("experiment config must be a JSON object")
    config = L2ExperimentConfig(**raw_config)
    if config.adapter_rank is not None:
        raise ValueError("registered calibration model must not use an adapter")
    table, artifact_manifest = load_pretrained_artifact(feature_artifact)
    model = FrozenFeatureRanker(
        input_dim=table.dimension,
        embedding_dim=config.embedding_dim,
        temperature=config.temperature,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    calibration_ranking, calibration_predictions = evaluate_frozen_ranker(
        model,
        table,
        behavior_path=calibration_behaviors,
        max_history_items=config.max_history_items,
        device=torch.device("cpu"),
        max_impressions=None,
        batch_size=config.eval_batch_size,
    )
    dev_ranking, dev_predictions = evaluate_frozen_ranker(
        model,
        table,
        behavior_path=dev_behaviors,
        max_history_items=config.max_history_items,
        device=torch.device("cpu"),
        max_impressions=None,
        batch_size=config.eval_batch_size,
    )
    calibration_scores, calibration_labels = _score_arrays(
        calibration_predictions, calibration_behaviors
    )
    dev_scores, dev_labels = _score_arrays(dev_predictions, dev_behaviors)
    scale, bias = fit_monotonic_platt(calibration_scores, calibration_labels)
    prevalence = float(calibration_labels.mean())

    def summarize(scores: NDArray[np.float64], labels: NDArray[np.float64]) -> dict[str, Any]:
        return {
            "candidate_count": len(scores),
            "positive_prevalence": float(labels.mean()),
            "raw_sigmoid": probability_metrics(labels, _sigmoid(scores)),
            "calibrated": probability_metrics(labels, _sigmoid(scale * scores + bias)),
            "calibration_prevalence_constant": probability_metrics(
                labels, np.full_like(scores, prevalence)
            ),
        }

    rank_identity = all(
        ranks_from_scores(scores) == ranks_from_scores([scale * value + bias for value in scores])
        for _, scores in dev_predictions
    )
    return {
        "schema_version": 1,
        "protocol": "training_only_monotonic_platt_calibration",
        "calibrator": {"positive_scale": scale, "bias": bias},
        "rank_identity_on_dev": rank_identity,
        "calibration_partition": {
            "impression_count": len(calibration_predictions),
            "ranking": calibration_ranking,
            "probability": summarize(calibration_scores, calibration_labels),
        },
        "dev": {
            "impression_count": len(dev_predictions),
            "ranking": dev_ranking,
            "probability": summarize(dev_scores, dev_labels),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": time.perf_counter() - started,
        "peak_resident_memory_bytes": _peak_resident_memory_bytes(),
        "input_sha256": {
            "config": sha256_file(config_path),
            "checkpoint": sha256_file(checkpoint_path),
            "feature_manifest": sha256_file(feature_artifact / "manifest.json"),
            "features": cast(str, artifact_manifest["feature_sha256"]),
            "calibration_behaviors": sha256_file(calibration_behaviors),
            "dev_behaviors": sha256_file(dev_behaviors),
        },
        "provenance": {
            "git": asdict(collect_git_state(repository_root)),
            "specification_sha256": sha256_file(repository_root / "docs/L2_CALIBRATION_SPEC.md"),
        },
        "limitations": [
            "Logged click labels inherit exposure and position bias.",
            "This auxiliary ranker excludes the calibration day and is not the headline L2 run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or evaluate L2 calibration artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--source", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--cutoff", default="2019-11-14T00:00:00")
    split_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--calibration-behaviors", type=Path, required=True)
    evaluate_parser.add_argument("--dev-behaviors", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "split":
        split_calibration_behaviors(
            args.source,
            args.output,
            cutoff=datetime.fromisoformat(args.cutoff),
            repository_root=args.repository_root,
        )
        return
    payload = evaluate_calibration(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        feature_artifact=args.features,
        calibration_behaviors=args.calibration_behaviors,
        dev_behaviors=args.dev_behaviors,
        repository_root=args.repository_root,
    )
    args.output.write_bytes(canonical_json_bytes(payload))


if __name__ == "__main__":
    main()
