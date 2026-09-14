"""Train and evaluate the frozen L1 NRMS-style official-ranking baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from recforge.data.mind import iter_mind_behaviors, sha256_file
from recforge.data.ranking import (
    RankingBatch,
    RankingExample,
    TitleTable,
    build_title_table,
    iter_ranking_batches,
    load_ranking_examples,
)
from recforge.data.submission import order_from_scores, write_prediction_file
from recforge.data.text import load_vocabulary_artifact
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.models.nrms import NRMSRanker, sampled_softmax_loss
from recforge.tracking import JsonValue, record_run


@dataclass(frozen=True, slots=True)
class L1ExperimentConfig:
    vocabulary_artifact: str
    news_paths: list[str]
    train_behaviors: str
    eval_behaviors: str
    output_root: str = "runs"
    seed: int = 2027
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 128
    attention_heads: int = 8
    attention_hidden_dim: int = 128
    negative_count: int = 4
    max_history_items: int = 50
    max_train_examples: int | None = None
    max_eval_impressions: int | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.news_paths:
            raise ValueError("news_paths must not be empty")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer rates are invalid")
        if self.embedding_dim <= 0 or self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if self.attention_hidden_dim <= 0 or self.negative_count <= 0:
            raise ValueError("attention_hidden_dim and negative_count must be positive")
        if self.max_history_items <= 0:
            raise ValueError("max_history_items must be positive")
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


def _batch_tensors(batch: RankingBatch, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(batch.history_token_ids).to(device),
        torch.from_numpy(batch.history_token_mask).to(device),
        torch.from_numpy(batch.history_item_mask).to(device),
        torch.from_numpy(batch.candidate_token_ids).to(device),
        torch.from_numpy(batch.candidate_token_mask).to(device),
    )


def train_nrms(
    config: L1ExperimentConfig,
    table: TitleTable,
    *,
    behavior_path: Path,
    vocabulary_size: int,
    device: torch.device,
) -> tuple[NRMSRanker, list[float], int]:
    _seed_everything(config.seed)
    model = NRMSRanker(
        vocabulary_size,
        config.embedding_dim,
        config.attention_heads,
        config.attention_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    epoch_losses: list[float] = []
    trained_examples = 0

    model.train()
    for epoch in range(config.epochs):
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
        for batch in iter_ranking_batches(
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
        trained_examples += epoch_examples
    return model, epoch_losses, trained_examples


def evaluate_nrms(
    model: NRMSRanker,
    table: TitleTable,
    *,
    behavior_path: Path,
    max_history_items: int,
    device: torch.device,
    max_impressions: int | None,
) -> tuple[dict[str, float | int], list[tuple[str, list[float]]]]:
    auc_total = mrr_total = ndcg_5_total = ndcg_10_total = 0.0
    query_count = skipped_auc = 0
    predictions: list[tuple[str, list[float]]] = []
    model.eval()
    with torch.no_grad():
        for behavior in iter_mind_behaviors(behavior_path):
            positives = {
                item_id
                for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
                if label == 1
            }
            if not positives:
                continue
            if max_impressions is not None and query_count >= max_impressions:
                break
            example = RankingExample(
                query_id=f"eval:{behavior.impression_id}",
                history_item_ids=behavior.history_item_ids,
                candidate_item_ids=behavior.candidate_item_ids,
            )
            batch = next(
                iter_ranking_batches(
                    (example,),
                    table,
                    batch_size=1,
                    max_history_items=max_history_items,
                    seed=0,
                    epoch=0,
                    shuffle=False,
                )
            )
            raw_scores = model(*_batch_tensors(batch, device)).squeeze(0).cpu().tolist()
            scores = cast(list[float], raw_scores)
            ranked = [behavior.candidate_item_ids[index] for index in order_from_scores(scores)]
            if 0 in behavior.labels and 1 in behavior.labels:
                auc_total += binary_auc(behavior.labels, scores)
            else:
                skipped_auc += 1
            mrr_total += mean_reciprocal_rank_at_k(ranked, positives, len(ranked))
            ndcg_5_total += ndcg_at_k(ranked, positives, 5)
            ndcg_10_total += ndcg_at_k(ranked, positives, 10)
            predictions.append((behavior.impression_id, scores))
            query_count += 1
    if query_count == 0 or query_count == skipped_auc:
        raise ValueError("evaluation produced no valid impressions")
    return (
        {
            "query_count": query_count,
            "auc_query_count": query_count - skipped_auc,
            "auc": auc_total / (query_count - skipped_auc),
            "mrr": mrr_total / query_count,
            "ndcg@5": ndcg_5_total / query_count,
            "ndcg@10": ndcg_10_total / query_count,
        },
        predictions,
    )


def run_l1_experiment(config: L1ExperimentConfig, repository_root: Path) -> Path:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    device = _resolve_device(config.device)
    vocabulary_directory = repository_root / config.vocabulary_artifact
    vocabulary, vocabulary_config = load_vocabulary_artifact(vocabulary_directory)
    news_paths = [repository_root / value for value in config.news_paths]
    table = build_title_table(
        news_paths, vocabulary, max_title_tokens=vocabulary_config.max_title_tokens
    )
    train_path = repository_root / config.train_behaviors
    eval_path = repository_root / config.eval_behaviors
    model, losses, trained_examples = train_nrms(
        config,
        table,
        behavior_path=train_path,
        vocabulary_size=len(vocabulary.tokens),
        device=device,
    )
    evaluation, predictions = evaluate_nrms(
        model,
        table,
        behavior_path=eval_path,
        max_history_items=config.max_history_items,
        device=device,
        max_impressions=config.max_eval_impressions,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        checkpoint_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), checkpoint_path)
        metrics: dict[str, JsonValue] = {
            "protocol": "mind_official_impression",
            "device": str(device),
            "trained_examples": trained_examples,
            "epoch_losses": cast(list[JsonValue], losses),
            "evaluation": cast(dict[str, JsonValue], evaluation),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        serialized_config = cast(dict[str, JsonValue], asdict(config))
        serialized_config["resolved_device"] = str(device)
        fingerprints = {
            "train_behaviors": sha256_file(train_path),
            "eval_behaviors": sha256_file(eval_path),
            "vocabulary_manifest": sha256_file(vocabulary_directory / "manifest.json"),
        }
        fingerprints.update(
            {f"news_{index}": sha256_file(path) for index, path in enumerate(news_paths)}
        )
        finished_at = datetime.now(UTC)
        run_directory = record_run(
            output_root=repository_root / config.output_root,
            repository_root=repository_root,
            experiment="mind-nrms-l1",
            metrics=metrics,
            config=serialized_config,
            dataset_fingerprints=fingerprints,
            seed=config.seed,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - started,
        )
        os.replace(checkpoint_path, run_directory / "model.pt")
        write_prediction_file(run_directory / "dev-prediction.txt", predictions)
        return run_directory
    finally:
        checkpoint_path.unlink(missing_ok=True)


def _load_config(path: Path) -> L1ExperimentConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return L1ExperimentConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the L1 NRMS baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    run_directory = run_l1_experiment(config, args.repository_root.resolve())
    print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
