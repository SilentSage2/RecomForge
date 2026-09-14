"""Train and evaluate the R1 MIND two-tower baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from recforge.data.features import (
    ItemFeatureTable,
    aggregate_history_features,
    load_feature_artifact,
)
from recforge.data.mind import iter_mind_behaviors, sha256_file
from recforge.data.protocol import TemporalCatalogIndex
from recforge.data.training import iter_feature_batches, load_training_examples
from recforge.evaluation import evaluate_temporal_corpus, positive_target_popularity
from recforge.metrics import binary_auc, ndcg_at_k, reciprocal_rank_at_k
from recforge.models.two_tower import TwoTowerRetriever, in_batch_softmax_loss
from recforge.tracking import JsonValue, record_run


@dataclass(frozen=True, slots=True)
class R1ExperimentConfig:
    feature_artifact: str
    train_behaviors: str
    eval_behaviors: str
    output_root: str = "runs"
    seed: int = 2027
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    embedding_dim: int = 64
    hidden_dim: int = 128
    temperature: float = 0.07
    symmetric_loss: bool = False
    device: str = "auto"
    max_train_queries: int | None = None
    max_eval_queries: int | None = None
    run_temporal_corpus_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 1:
            raise ValueError("epochs must be positive and batch_size must exceed one")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.temperature <= 0:
            raise ValueError("optimizer rates and temperature are invalid")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")
        if self.max_train_queries is not None and self.max_train_queries < 2:
            raise ValueError("max_train_queries must be at least two")
        if self.max_eval_queries is not None and self.max_eval_queries <= 0:
            raise ValueError("max_eval_queries must be positive")


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


def train_model(
    config: R1ExperimentConfig,
    table: ItemFeatureTable,
    *,
    behavior_path: Path,
    max_history_items: int,
    device: torch.device,
) -> tuple[TwoTowerRetriever, list[float], int]:
    _seed_everything(config.seed)
    examples = load_training_examples(behavior_path)
    if config.max_train_queries is not None:
        examples = examples[: config.max_train_queries]
    model = TwoTowerRetriever(
        table.dimension,
        table.dimension,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    epoch_losses: list[float] = []
    trained_pairs = 0

    model.train()
    for epoch in range(config.epochs):
        loss_total = 0.0
        pair_count = 0
        for batch in iter_feature_batches(
            examples,
            table,
            batch_size=config.batch_size,
            max_history_items=max_history_items,
            seed=config.seed,
            epoch=epoch,
        ):
            users = torch.from_numpy(batch.user_features).to(device)
            items = torch.from_numpy(batch.positive_item_features).to(device)
            item_ids = torch.from_numpy(batch.positive_item_rows).to(device)
            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model(users, items)
            loss = in_batch_softmax_loss(
                user_embeddings,
                item_embeddings,
                temperature=config.temperature,
                symmetric=config.symmetric_loss,
                positive_item_ids=item_ids,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            batch_pairs = len(batch.query_ids)
            loss_total += float(loss.detach().cpu()) * batch_pairs
            pair_count += batch_pairs
        if pair_count == 0:
            raise ValueError("training produced no complete batches")
        epoch_losses.append(loss_total / pair_count)
        trained_pairs += pair_count
    return model, epoch_losses, trained_pairs


def evaluate_logged_impressions(
    model: TwoTowerRetriever,
    table: ItemFeatureTable,
    *,
    behavior_path: Path,
    max_history_items: int,
    device: torch.device,
    max_queries: int | None,
) -> dict[str, float | int]:
    row_by_item_id = table.row_by_item_id()
    auc_total = 0.0
    mrr_total = 0.0
    ndcg_5_total = 0.0
    ndcg_10_total = 0.0
    query_count = 0
    skipped_auc = 0

    model.eval()
    with torch.no_grad():
        encoded_chunks = []
        for start in range(0, len(table.item_ids), 4096):
            feature_chunk = np.array(table.features[start : start + 4096], copy=True)
            encoded_chunks.append(model.encode_items(torch.from_numpy(feature_chunk).to(device)))
        item_embeddings = torch.cat(encoded_chunks)
        for behavior in iter_mind_behaviors(behavior_path):
            positives = {
                item_id
                for item_id, label in zip(behavior.candidate_item_ids, behavior.labels, strict=True)
                if label == 1
            }
            if not positives:
                continue
            if max_queries is not None and query_count >= max_queries:
                break
            try:
                candidate_rows = [row_by_item_id[item] for item in behavior.candidate_item_ids]
            except KeyError as error:
                raise ValueError(
                    f"evaluation candidate {error.args[0]!r} has no features"
                ) from error
            user_features = aggregate_history_features(
                behavior.history_item_ids,
                table,
                max_history_items=max_history_items,
                row_by_item_id=row_by_item_id,
            )
            user_embedding = model.encode_users(
                torch.from_numpy(user_features).unsqueeze(0).to(device)
            )
            scores_tensor = user_embedding @ item_embeddings[candidate_rows].transpose(0, 1)
            scores = cast(list[float], scores_tensor.squeeze(0).cpu().tolist())
            ranked = [
                item_id
                for _, item_id in sorted(
                    zip(scores, behavior.candidate_item_ids, strict=True), reverse=True
                )
            ]
            if 0 in behavior.labels and 1 in behavior.labels:
                auc_total += binary_auc(behavior.labels, scores)
            else:
                skipped_auc += 1
            mrr_total += reciprocal_rank_at_k(ranked, positives, len(ranked))
            ndcg_5_total += ndcg_at_k(ranked, positives, 5)
            ndcg_10_total += ndcg_at_k(ranked, positives, 10)
            query_count += 1

    if query_count == 0 or query_count == skipped_auc:
        raise ValueError("evaluation produced no valid queries")
    return {
        "query_count": query_count,
        "auc_query_count": query_count - skipped_auc,
        "auc": auc_total / (query_count - skipped_auc),
        "mrr": mrr_total / query_count,
        "ndcg@5": ndcg_5_total / query_count,
        "ndcg@10": ndcg_10_total / query_count,
    }


def run_experiment(config: R1ExperimentConfig, repository_root: Path) -> Path:
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    device = _resolve_device(config.device)
    feature_directory = repository_root / config.feature_artifact
    train_path = repository_root / config.train_behaviors
    eval_path = repository_root / config.eval_behaviors
    table, feature_config = load_feature_artifact(feature_directory)
    model, epoch_losses, trained_pairs = train_model(
        config,
        table,
        behavior_path=train_path,
        max_history_items=feature_config.max_history_items,
        device=device,
    )
    evaluation = evaluate_logged_impressions(
        model,
        table,
        behavior_path=eval_path,
        max_history_items=feature_config.max_history_items,
        device=device,
        max_queries=config.max_eval_queries,
    )
    corpus_evaluation: dict[str, float | int] | None = None
    if config.run_temporal_corpus_evaluation:
        catalog = TemporalCatalogIndex.from_behavior_files([train_path, eval_path])
        corpus_evaluation = evaluate_temporal_corpus(
            model,
            table,
            catalog=catalog,
            behavior_path=eval_path,
            training_popularity=positive_target_popularity(train_path),
            max_history_items=feature_config.max_history_items,
            device=device,
            max_queries=config.max_eval_queries,
        )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        checkpoint_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        finished_at = datetime.now(UTC)
        metrics: dict[str, JsonValue] = {
            "protocol": "logged_impression",
            "device": str(device),
            "trained_pairs": trained_pairs,
            "epoch_losses": cast(list[JsonValue], epoch_losses),
            "evaluation": cast(dict[str, JsonValue], evaluation),
            "temporal_corpus_evaluation": cast(dict[str, JsonValue] | None, corpus_evaluation),
            "checkpoint_sha256": checkpoint_sha256,
        }
        serialized_config = cast(dict[str, JsonValue], asdict(config))
        serialized_config["resolved_device"] = str(device)
        run_directory = record_run(
            output_root=repository_root / config.output_root,
            repository_root=repository_root,
            experiment="mind-two-tower",
            metrics=metrics,
            config=serialized_config,
            dataset_fingerprints={
                "train_behaviors": sha256_file(train_path),
                "eval_behaviors": sha256_file(eval_path),
                "feature_manifest": sha256_file(feature_directory / "manifest.json"),
            },
            seed=config.seed,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - start,
        )
        os.replace(checkpoint_path, run_directory / "model.pt")
        return run_directory
    finally:
        checkpoint_path.unlink(missing_ok=True)


def _load_config(path: Path) -> R1ExperimentConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return R1ExperimentConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the R1 two-tower model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_directory = run_experiment(_load_config(args.config), args.repository_root.resolve())
    print(json.dumps({"run_directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
