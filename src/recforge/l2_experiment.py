"""Train and evaluate the L2 ranker over frozen pretrained title features."""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from recforge.checkpoints import load_training_state, save_training_state, serialize_state_dict
from recforge.data.mind import MindBehavior, iter_mind_behaviors, sha256_file
from recforge.data.pretrained import load_pretrained_artifact
from recforge.data.ranking import (
    FeatureRankingBatch,
    iter_feature_ranking_batches,
    load_ranking_examples,
)
from recforge.data.submission import order_from_scores, ranks_from_scores, write_prediction_file
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.models.nrms import sampled_softmax_loss
from recforge.models.pretrained_ranker import FrozenFeatureRanker
from recforge.tracking import JsonValue, record_run, sha256_bytes


@dataclass(frozen=True, slots=True)
class L2ExperimentConfig:
    feature_artifact: str
    train_behaviors: str
    eval_behaviors: str
    output_root: str = "runs"
    seed: int = 2027
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 64
    temperature: float = 0.07
    negative_count: int = 4
    max_history_items: int = 50
    max_train_examples: int | None = None
    max_eval_impressions: int | None = None
    eval_batch_size: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("epochs and batch sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer rates are invalid")
        if self.embedding_dim <= 0 or self.temperature <= 0:
            raise ValueError("embedding_dim and temperature must be positive")
        if self.negative_count <= 0 or self.max_history_items <= 0:
            raise ValueError("negative_count and max_history_items must be positive")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive")
        if self.max_eval_impressions is not None and self.max_eval_impressions <= 0:
            raise ValueError("max_eval_impressions must be positive")
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _peak_resident_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _batch_tensors(
    batch: FeatureRankingBatch, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(batch.history_features).to(device),
        torch.from_numpy(batch.history_item_mask).to(device),
        torch.from_numpy(batch.candidate_features).to(device),
    )


def train_frozen_ranker(
    config: L2ExperimentConfig,
    table: Any,
    *,
    behavior_path: Path,
    device: torch.device,
    progress: bool = False,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    resume_metadata: dict[str, str] | None = None,
) -> tuple[FrozenFeatureRanker, list[float], int, list[float], int]:
    _seed_everything(config.seed)
    model = FrozenFeatureRanker(
        input_dim=table.dimension,
        embedding_dim=config.embedding_dim,
        temperature=config.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    epoch_losses: list[float] = []
    epoch_seconds: list[float] = []
    trained_examples = 0
    start_epoch = 0
    metadata = resume_metadata or {}

    if resume_from is not None:
        state = load_training_state(resume_from, device=device)
        if state.get("schema_version") != 1 or state.get("metadata") != metadata:
            raise ValueError("training checkpoint is incompatible with this run")
        if state.get("seed") != config.seed:
            raise ValueError("training checkpoint seed does not match this run")
        start_epoch = int(state["completed_epochs"])
        if not 0 < start_epoch < config.epochs:
            raise ValueError("training checkpoint has no remaining epochs")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch_losses = [float(value) for value in state["epoch_losses"]]
        epoch_seconds = [float(value) for value in state["epoch_seconds"]]
        trained_examples = int(state["trained_examples"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())

    model.train()
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        examples = load_ranking_examples(
            behavior_path,
            negative_count=config.negative_count,
            seed=config.seed,
            epoch=epoch,
            max_examples=config.max_train_examples,
        )
        if not examples:
            raise ValueError("training produced no impression-local examples")
        loss_total = 0.0
        epoch_examples = 0
        for batch in iter_feature_ranking_batches(
            examples,
            table,
            batch_size=config.batch_size,
            max_history_items=config.max_history_items,
            seed=config.seed,
            epoch=epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            scores = model(*_batch_tensors(batch, device))
            targets = torch.from_numpy(batch.target_indices).to(device)
            loss = sampled_softmax_loss(scores, targets)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = len(batch.query_ids)
            loss_total += float(loss.detach().cpu()) * count
            epoch_examples += count
        epoch_losses.append(loss_total / epoch_examples)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        trained_examples += epoch_examples
        if checkpoint_path is not None:
            save_training_state(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "metadata": metadata,
                    "seed": config.seed,
                    "completed_epochs": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch_losses": epoch_losses,
                    "epoch_seconds": epoch_seconds,
                    "trained_examples": trained_examples,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            )
        if progress:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "loss": epoch_losses[-1],
                        "seconds": epoch_seconds[-1],
                    }
                ),
                flush=True,
            )
    return model, epoch_losses, trained_examples, epoch_seconds, start_epoch


def _encode_all_items(
    model: FrozenFeatureRanker, features: np.ndarray, device: torch.device, *, batch_size: int
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(
                np.asarray(features[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            chunks.append(model.encode_items(batch))
    return torch.cat(chunks)


def evaluate_frozen_ranker(
    model: FrozenFeatureRanker,
    table: Any,
    *,
    behavior_path: Path,
    max_history_items: int,
    device: torch.device,
    max_impressions: int | None,
    batch_size: int,
) -> tuple[dict[str, float | int], list[tuple[str, list[float]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    row_by_item = table.row_by_item_id()
    item_embeddings = _encode_all_items(model, table.features, device, batch_size=4096)
    auc_total = mrr_total = ndcg_5_total = ndcg_10_total = 0.0
    query_count = skipped_auc = tied_score_impressions = 0
    predictions: list[tuple[str, list[float]]] = []
    pending: list[MindBehavior] = []

    def evaluate_pending() -> None:
        nonlocal auc_total, mrr_total, ndcg_5_total, ndcg_10_total
        nonlocal query_count, skipped_auc, tied_score_impressions
        if not pending:
            return
        histories = torch.zeros(
            (len(pending), max_history_items, model.embedding_dim),
            dtype=item_embeddings.dtype,
            device=device,
        )
        history_mask = torch.zeros(
            (len(pending), max_history_items), dtype=torch.bool, device=device
        )
        max_candidates = max(len(item.candidate_item_ids) for item in pending)
        candidates = torch.zeros(
            (len(pending), max_candidates, model.embedding_dim),
            dtype=item_embeddings.dtype,
            device=device,
        )
        for row, behavior in enumerate(pending):
            history_rows = [
                row_by_item[item] for item in behavior.history_item_ids if item in row_by_item
            ][-max_history_items:]
            if history_rows:
                histories[row, : len(history_rows)] = item_embeddings[history_rows]
                history_mask[row, : len(history_rows)] = True
            try:
                candidate_rows = [row_by_item[item] for item in behavior.candidate_item_ids]
            except KeyError as error:
                raise ValueError(
                    f"impression {behavior.impression_id} candidate {error.args[0]!r} "
                    "has no pretrained features"
                ) from error
            candidates[row, : len(candidate_rows)] = item_embeddings[candidate_rows]
        weights = history_mask.to(histories.dtype).unsqueeze(-1)
        users = (histories * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        users = torch.nn.functional.normalize(users, dim=-1)
        scores_batch = torch.einsum("bd,bcd->bc", users, candidates) / model.temperature
        for row, behavior in enumerate(pending):
            scores = cast(
                list[float],
                scores_batch[row, : len(behavior.candidate_item_ids)].cpu().tolist(),
            )
            positives = {
                item
                for item, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
                if label == 1
            }
            ranks = ranks_from_scores(scores)
            ranked = [behavior.candidate_item_ids[index] for index in order_from_scores(scores)]
            tied_score_impressions += len(set(scores)) != len(scores)
            if 0 in behavior.labels and 1 in behavior.labels:
                auc_total += binary_auc(behavior.labels, [-float(rank) for rank in ranks])
            else:
                skipped_auc += 1
            mrr_total += mean_reciprocal_rank_at_k(ranked, positives, len(ranked))
            ndcg_5_total += ndcg_at_k(ranked, positives, 5)
            ndcg_10_total += ndcg_at_k(ranked, positives, 10)
            predictions.append((behavior.impression_id, scores))
            query_count += 1
        pending.clear()

    with torch.no_grad():
        for behavior in iter_mind_behaviors(behavior_path):
            if not any(behavior.labels):
                continue
            if max_impressions is not None and query_count + len(pending) >= max_impressions:
                break
            pending.append(behavior)
            if len(pending) == batch_size:
                evaluate_pending()
        evaluate_pending()
    if query_count == 0 or query_count == skipped_auc:
        raise ValueError("evaluation produced no valid impressions")
    return (
        {
            "query_count": query_count,
            "auc_query_count": query_count - skipped_auc,
            "tied_score_impression_count": tied_score_impressions,
            "auc": auc_total / (query_count - skipped_auc),
            "mrr": mrr_total / query_count,
            "ndcg@5": ndcg_5_total / query_count,
            "ndcg@10": ndcg_10_total / query_count,
        },
        predictions,
    )


def run_l2_experiment(
    config: L2ExperimentConfig,
    repository_root: Path,
    *,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    device = _resolve_device(config.device)
    artifact_directory = repository_root / config.feature_artifact
    table, artifact_manifest = load_pretrained_artifact(artifact_directory)
    train_path = repository_root / config.train_behaviors
    eval_path = repository_root / config.eval_behaviors
    fingerprints = {
        "train_behaviors": sha256_file(train_path),
        "eval_behaviors": sha256_file(eval_path),
        "feature_manifest": sha256_file(artifact_directory / "manifest.json"),
        "features": cast(str, artifact_manifest["feature_sha256"]),
    }
    resume_metadata = {
        "config": sha256_bytes(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
        ),
        **fingerprints,
    }
    model, losses, trained_examples, epoch_seconds, resumed_from_epoch = train_frozen_ranker(
        config,
        table,
        behavior_path=train_path,
        device=device,
        progress=True,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        resume_metadata=resume_metadata,
    )
    evaluation, predictions = evaluate_frozen_ranker(
        model,
        table,
        behavior_path=eval_path,
        max_history_items=config.max_history_items,
        device=device,
        max_impressions=config.max_eval_impressions,
        batch_size=config.eval_batch_size,
    )
    checkpoint_bytes = serialize_state_dict(model)
    try:
        metrics: dict[str, JsonValue] = {
            "protocol": "mind_official_impression",
            "device": str(device),
            "trained_examples": trained_examples,
            "epoch_losses": cast(list[JsonValue], losses),
            "epoch_seconds": cast(list[JsonValue], epoch_seconds),
            "evaluation": cast(dict[str, JsonValue], evaluation),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
            "resumed_from_epoch": resumed_from_epoch,
            "peak_resident_memory_bytes": _peak_resident_memory_bytes(),
            "pretrained_feature_artifact_bytes": cast(
                int, artifact_manifest["encoding"]["artifact_bytes"]
            ),
            "pretrained_encoding_seconds": cast(
                float, artifact_manifest["encoding"]["seconds"]
            ),
        }
        serialized_config = cast(dict[str, JsonValue], asdict(config))
        serialized_config["resolved_device"] = str(device)
        run_directory = record_run(
            output_root=repository_root / config.output_root,
            repository_root=repository_root,
            experiment="mind-frozen-title-l2",
            metrics=metrics,
            config=serialized_config,
            dataset_fingerprints=fingerprints,
            seed=config.seed,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_seconds=time.perf_counter() - started,
        )
        (run_directory / "model.pt").write_bytes(checkpoint_bytes)
        write_prediction_file(run_directory / "dev-prediction.txt", predictions)
        return run_directory
    finally:
        checkpoint_bytes = b""


def _load_config(path: Path) -> L2ExperimentConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return L2ExperimentConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the frozen-feature L2 ranker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    repository_root = args.repository_root.resolve()
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is None or args.checkpoint_path.is_absolute()
        else repository_root / args.checkpoint_path
    )
    resume_from = (
        args.resume_from
        if args.resume_from is None or args.resume_from.is_absolute()
        else repository_root / args.resume_from
    )
    run_directory = run_l2_experiment(
        config,
        repository_root,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )
    print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
