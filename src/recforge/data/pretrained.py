"""Fingerprintable frozen-transformer title feature artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import resource
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from recforge.data.features import ItemFeatureTable
from recforge.data.mind import MindNews, iter_mind_news, sha256_file


@dataclass(frozen=True, slots=True)
class PretrainedFeatureConfig:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "826711e54e001c83835913827a843d8dd0a1def9"
    dimension: int = 384
    max_tokens: int = 32
    batch_size: int = 64
    normalize: bool = True
    dtype: str = "float32"
    max_items: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("model_id and revision must not be empty")
        if self.dimension <= 0 or self.max_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("feature dimensions and batch size must be positive")
        if self.dtype != "float32":
            raise ValueError("only float32 artifacts are supported")
        if self.max_items is not None and self.max_items <= 0:
            raise ValueError("max_items must be positive")


def _peak_resident_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _collect_titles(
    news_paths: list[Path], *, max_items: int | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not news_paths:
        raise ValueError("at least one news file is required")
    news_by_id: dict[str, MindNews] = {}
    for path in news_paths:
        for news in iter_mind_news(path):
            existing = news_by_id.get(news.news_id)
            if existing is not None and existing != news:
                raise ValueError(f"conflicting metadata for item {news.news_id!r}")
            news_by_id[news.news_id] = news
    item_ids = tuple(sorted(news_by_id))
    if max_items is not None:
        item_ids = item_ids[:max_items]
    if not item_ids:
        raise ValueError("news inputs contain no items")
    return item_ids, tuple(news_by_id[item_id].title for item_id in item_ids)


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool unmasked token states and return zeros for an empty sequence."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, tokens, dimension]")
    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask shape must match the first two hidden-state axes")
    weights = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def encode_titles(
    titles: tuple[str, ...],
    *,
    tokenizer: Any,
    model: Any,
    config: PretrainedFeatureConfig,
    device: torch.device,
) -> NDArray[np.float32]:
    """Encode title batches with explicit pooling and normalization."""
    chunks: list[NDArray[np.float32]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(titles), config.batch_size):
            batch = titles[start : start + config.batch_size]
            encoded = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=config.max_tokens,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = masked_mean_pool(output.last_hidden_state, encoded["attention_mask"])
            if config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    features = np.concatenate(chunks, axis=0)
    if features.shape != (len(titles), config.dimension):
        raise ValueError(
            f"encoded feature shape {features.shape} != {(len(titles), config.dimension)}"
        )
    if not np.isfinite(features).all():
        raise ValueError("encoded features contain non-finite values")
    return features


def write_pretrained_artifact(
    output_directory: Path,
    table: ItemFeatureTable,
    *,
    titles: tuple[str, ...],
    config: PretrainedFeatureConfig,
    news_paths: list[Path],
    encoding_seconds: float,
    device: str,
    dependencies: dict[str, str],
) -> None:
    """Write an immutable artifact, publishing its manifest last."""
    if len(titles) != len(table.item_ids):
        raise ValueError("titles and item feature rows must have equal length")
    output_directory.mkdir(parents=True, exist_ok=False)
    item_ids_path = output_directory / "item_ids.json"
    features_path = output_directory / "features.npy"
    title_hashes_path = output_directory / "title_hashes.json"
    item_ids_path.write_text(json.dumps(table.item_ids) + "\n", encoding="utf-8")
    title_hashes = [hashlib.sha256(title.encode()).hexdigest() for title in titles]
    title_hashes_path.write_text(json.dumps(title_hashes) + "\n", encoding="utf-8")
    np.save(features_path, table.features, allow_pickle=False)
    manifest = {
        "artifact_schema_version": 1,
        "config": asdict(config),
        "model": {
            "id": config.model_id,
            "revision": config.revision,
            "trust_remote_code": False,
            "pooling": "attention_masked_mean",
            "normalized": config.normalize,
        },
        "item_count": len(table.item_ids),
        "feature_sha256": sha256_file(features_path),
        "item_ids_sha256": sha256_file(item_ids_path),
        "title_hashes_sha256": sha256_file(title_hashes_path),
        "news_inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in news_paths],
        "encoding": {
            "device": device,
            "seconds": encoding_seconds,
            "items_per_second": len(table.item_ids) / encoding_seconds,
            "peak_resident_memory_bytes": _peak_resident_memory_bytes(),
            "artifact_bytes": features_path.stat().st_size,
        },
        "dependencies": dependencies,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_pretrained_artifact(artifact_directory: Path) -> tuple[ItemFeatureTable, dict[str, Any]]:
    manifest_path = artifact_directory / "manifest.json"
    item_ids_path = artifact_directory / "item_ids.json"
    title_hashes_path = artifact_directory / "title_hashes.json"
    features_path = artifact_directory / "features.npy"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != 1:
        raise ValueError("unsupported pretrained feature artifact schema")
    for path, key in (
        (item_ids_path, "item_ids_sha256"),
        (title_hashes_path, "title_hashes_sha256"),
        (features_path, "feature_sha256"),
    ):
        if sha256_file(path) != manifest.get(key):
            raise ValueError(f"{key} fingerprint mismatch")
    raw_item_ids: Any = json.loads(item_ids_path.read_text(encoding="utf-8"))
    raw_title_hashes: Any = json.loads(title_hashes_path.read_text(encoding="utf-8"))
    if not isinstance(raw_item_ids, list) or not all(
        isinstance(item, str) for item in raw_item_ids
    ):
        raise ValueError("item_ids.json must contain strings")
    if not isinstance(raw_title_hashes, list) or not all(
        isinstance(item, str) and len(item) == 64 for item in raw_title_hashes
    ):
        raise ValueError("title_hashes.json must contain SHA-256 strings")
    if len(raw_title_hashes) != len(raw_item_ids):
        raise ValueError("title hashes and item IDs must have equal length")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    table = ItemFeatureTable(tuple(raw_item_ids), features)
    config = PretrainedFeatureConfig(**manifest["config"])
    if len(table.item_ids) != manifest.get("item_count") or table.dimension != config.dimension:
        raise ValueError("pretrained feature artifact shape metadata mismatch")
    return table, manifest


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, or mps")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen pretrained MIND title features.")
    parser.add_argument("--news", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    config = PretrainedFeatureConfig(batch_size=args.batch_size, max_items=args.max_items)
    device = _resolve_device(args.device)
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id, revision=config.revision, trust_remote_code=False
    )
    model = transformers.AutoModel.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    item_ids, titles = _collect_titles(args.news, max_items=config.max_items)
    started = time.perf_counter()
    features = encode_titles(titles, tokenizer=tokenizer, model=model, config=config, device=device)
    encoding_seconds = time.perf_counter() - started
    write_pretrained_artifact(
        args.output,
        ItemFeatureTable(item_ids, features),
        titles=titles,
        config=config,
        news_paths=args.news,
        encoding_seconds=encoding_seconds,
        device=str(device),
        dependencies={
            "numpy": version("numpy"),
            "safetensors": version("safetensors"),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
    )
    print(json.dumps({"output": str(args.output), "items": len(item_ids)}, indent=2))


if __name__ == "__main__":
    main()
