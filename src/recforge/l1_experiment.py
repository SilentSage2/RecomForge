"""Train and evaluate the frozen L1 NRMS-style official-ranking baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from recforge.checkpoints import serialize_state_dict
from recforge.data.mind import MindBehavior, iter_mind_behaviors, sha256_file
from recforge.data.ranking import (
    RankingBatch,
    RankingExample,
    TitleTable,
    build_title_table,
    iter_ranking_batches,
    load_ranking_examples,
)
from recforge.data.submission import order_from_scores, ranks_from_scores, write_prediction_file
from recforge.data.text import load_vocabulary_artifact
from recforge.metrics import binary_auc, mean_reciprocal_rank_at_k, ndcg_at_k
from recforge.models.nrms import NRMSRanker, sampled_softmax_loss
from recforge.tracking import JsonValue, record_run, sha256_bytes


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
    title_encoder_mode: str = "attention"
    history_encoder_mode: str = "attention"
    deduplicate_titles: bool = False
    negative_count: int = 4
    max_history_items: int = 50
    max_train_examples: int | None = None
    max_eval_impressions: int | None = None
    eval_batch_size: int = 64
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
        if self.title_encoder_mode not in {"attention", "mean"}:
            raise ValueError("title_encoder_mode must be attention or mean")
        if self.history_encoder_mode not in {"attention", "mean"}:
            raise ValueError("history_encoder_mode must be attention or mean")
        if self.max_history_items <= 0:
            raise ValueError("max_history_items must be positive")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive")
        if self.max_eval_impressions is not None and self.max_eval_impressions <= 0:
            raise ValueError("max_eval_impressions must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
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
        config.title_encoder_mode,
        config.history_encoder_mode,
        config.deduplicate_titles,
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
    query_count = skipped_auc = tied_score_impressions = 0
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


def _encode_title_table(
    model: NRMSRanker, table: TitleTable, device: torch.device, *, batch_size: int = 4096
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(table.item_ids), batch_size):
            token_ids = torch.from_numpy(
                np.asarray(table.token_ids[start : start + batch_size])
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(table.attention_mask[start : start + batch_size])
            ).to(device)
            chunks.append(model.title_encoder(token_ids, token_mask))
    return torch.cat(chunks)


def evaluate_nrms_cached(
    model: NRMSRanker,
    table: TitleTable,
    *,
    behavior_path: Path,
    max_history_items: int,
    device: torch.device,
    max_impressions: int | None,
    batch_size: int,
) -> tuple[dict[str, float | int], list[tuple[str, list[float]]]]:
    """Evaluate in batches after encoding every news title exactly once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    row_by_item = table.row_by_item_id()
    auc_total = mrr_total = ndcg_5_total = ndcg_10_total = 0.0
    query_count = skipped_auc = tied_score_impressions = 0
    predictions: list[tuple[str, list[float]]] = []
    model.eval()
    title_embeddings = _encode_title_table(model, table, device)
    pending: list[MindBehavior] = []

    def evaluate_pending() -> None:
        nonlocal auc_total, mrr_total, ndcg_5_total, ndcg_10_total
        nonlocal query_count, skipped_auc, tied_score_impressions
        if not pending:
            return
        embedding_dim = title_embeddings.shape[1]
        max_candidates = max(len(behavior.candidate_item_ids) for behavior in pending)
        histories = torch.zeros(
            (len(pending), max_history_items, embedding_dim),
            dtype=title_embeddings.dtype,
            device=device,
        )
        history_mask = torch.zeros(
            (len(pending), max_history_items), dtype=torch.bool, device=device
        )
        candidates = torch.zeros(
            (len(pending), max_candidates, embedding_dim),
            dtype=title_embeddings.dtype,
            device=device,
        )
        for row, behavior in enumerate(pending):
            history_rows = [
                row_by_item[item] for item in behavior.history_item_ids if item in row_by_item
            ][-max_history_items:]
            if history_rows:
                count = len(history_rows)
                histories[row, :count] = title_embeddings[history_rows]
                history_mask[row, :count] = True
            try:
                candidate_rows = [row_by_item[item] for item in behavior.candidate_item_ids]
            except KeyError as error:
                raise ValueError(
                    f"impression {behavior.impression_id} candidate {error.args[0]!r} has no title"
                ) from error
            candidates[row, : len(candidate_rows)] = title_embeddings[candidate_rows]
        with torch.no_grad():
            users = model.user_encoder(histories, history_mask)
            batch_scores = torch.einsum("bd,bcd->bc", users, candidates)
        for row, behavior in enumerate(pending):
            scores = cast(
                list[float],
                batch_scores[row, : len(behavior.candidate_item_ids)].cpu().tolist(),
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
    evaluation, predictions = evaluate_nrms_cached(
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
            "evaluation": cast(dict[str, JsonValue], evaluation),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
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
        (run_directory / "model.pt").write_bytes(checkpoint_bytes)
        write_prediction_file(run_directory / "dev-prediction.txt", predictions)
        return run_directory
    finally:
        checkpoint_bytes = b""


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
