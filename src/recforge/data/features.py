"""Deterministic, leakage-resistant content features for MIND retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from recforge.data.mind import MindNews, iter_mind_news, sha256_file

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class HashingFeatureConfig:
    dimension: int = 512
    max_history_items: int = 50
    include_abstract: bool = False

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("feature dimension must be positive")
        if self.max_history_items <= 0:
            raise ValueError("max_history_items must be positive")


@dataclass(frozen=True, slots=True)
class ItemFeatureTable:
    item_ids: tuple[str, ...]
    features: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("item features must have shape [items, dimension]")
        if self.features.shape[0] != len(self.item_ids):
            raise ValueError("item IDs and feature rows must have equal length")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item IDs must be unique")
        if self.features.dtype != np.float32:
            raise ValueError("item features must use float32")

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    def row_by_item_id(self) -> dict[str, int]:
        return {item_id: row for row, item_id in enumerate(self.item_ids)}


def _feature_tokens(news: MindNews, config: HashingFeatureConfig) -> list[str]:
    tokens = [f"category={news.category.lower()}", f"subcategory={news.subcategory.lower()}"]
    tokens.extend(f"title={token}" for token in _TOKEN_PATTERN.findall(news.title.lower()))
    if config.include_abstract:
        tokens.extend(
            f"abstract={token}" for token in _TOKEN_PATTERN.findall(news.abstract.lower())
        )
    return tokens


def _hash_token(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"recforge").digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    index = value % dimension
    sign = 1.0 if value & (1 << 63) else -1.0
    return index, sign


def hash_item_features(news: MindNews, config: HashingFeatureConfig) -> NDArray[np.float32]:
    """Create a signed hashed bag of metadata and text, independent of corpus vocabulary."""
    features = np.zeros(config.dimension, dtype=np.float32)
    for token in _feature_tokens(news, config):
        index, sign = _hash_token(token, config.dimension)
        features[index] += sign
    norm = float(np.linalg.norm(features))
    if norm > 0:
        features /= norm
    return features


def build_item_feature_table(
    news_paths: list[Path], config: HashingFeatureConfig
) -> ItemFeatureTable:
    if not news_paths:
        raise ValueError("at least one news file is required")
    news_by_id: dict[str, MindNews] = {}
    for path in news_paths:
        for news in iter_mind_news(path):
            existing = news_by_id.get(news.news_id)
            if existing is not None and existing != news:
                raise ValueError(f"conflicting metadata for item {news.news_id!r}")
            news_by_id[news.news_id] = news
    if not news_by_id:
        raise ValueError("news inputs contain no items")

    item_ids = tuple(sorted(news_by_id))
    features = np.stack(
        [hash_item_features(news_by_id[item_id], config) for item_id in item_ids]
    ).astype(np.float32, copy=False)
    return ItemFeatureTable(item_ids=item_ids, features=features)


def aggregate_history_features(
    history_item_ids: tuple[str, ...],
    table: ItemFeatureTable,
    *,
    max_history_items: int,
) -> NDArray[np.float32]:
    """Mean-pool the most recent known pre-query items and L2-normalize the result."""
    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")
    row_by_item_id = table.row_by_item_id()
    rows = [
        row_by_item_id[item_id]
        for item_id in history_item_ids[-max_history_items:]
        if item_id in row_by_item_id
    ]
    if not rows:
        return np.zeros(table.dimension, dtype=np.float32)
    features = np.asarray(table.features[rows].mean(axis=0), dtype=np.float32)
    norm = float(np.linalg.norm(features))
    if norm > 0:
        features /= norm
    return features


def write_feature_artifact(
    output_directory: Path,
    table: ItemFeatureTable,
    config: HashingFeatureConfig,
    news_paths: list[Path],
) -> None:
    """Write an immutable local feature artifact, with the manifest written last."""
    output_directory.mkdir(parents=True, exist_ok=False)
    item_ids_path = output_directory / "item_ids.json"
    features_path = output_directory / "features.npy"
    item_ids_path.write_text(json.dumps(table.item_ids) + "\n", encoding="utf-8")
    np.save(features_path, table.features, allow_pickle=False)
    manifest = {
        "artifact_schema_version": 1,
        "config": asdict(config),
        "feature_sha256": sha256_file(features_path),
        "item_count": len(table.item_ids),
        "item_ids_sha256": sha256_file(item_ids_path),
        "news_inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in news_paths],
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_feature_artifact(
    artifact_directory: Path,
) -> tuple[ItemFeatureTable, HashingFeatureConfig]:
    """Load a memory-mapped feature artifact after validating its fingerprints."""
    manifest_path = artifact_directory / "manifest.json"
    item_ids_path = artifact_directory / "item_ids.json"
    features_path = artifact_directory / "features.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != 1:
        raise ValueError("unsupported feature artifact schema")
    if sha256_file(item_ids_path) != manifest.get("item_ids_sha256"):
        raise ValueError("item ID fingerprint mismatch")
    if sha256_file(features_path) != manifest.get("feature_sha256"):
        raise ValueError("feature matrix fingerprint mismatch")

    config = HashingFeatureConfig(**manifest["config"])
    raw_item_ids = json.loads(item_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw_item_ids, list) or not all(
        isinstance(item_id, str) for item_id in raw_item_ids
    ):
        raise ValueError("item_ids.json must contain a list of strings")
    item_ids = tuple(raw_item_ids)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    table = ItemFeatureTable(item_ids=item_ids, features=features)
    if len(item_ids) != manifest.get("item_count"):
        raise ValueError("feature artifact item count mismatch")
    if table.dimension != config.dimension:
        raise ValueError("feature artifact dimension mismatch")
    return table, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic MIND item features.")
    parser.add_argument("--news", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--max-history-items", type=int, default=50)
    parser.add_argument("--include-abstract", action="store_true")
    args = parser.parse_args()
    config = HashingFeatureConfig(
        dimension=args.dimension,
        max_history_items=args.max_history_items,
        include_abstract=args.include_abstract,
    )
    table = build_item_feature_table(args.news, config)
    write_feature_artifact(args.output, table, config, args.news)


if __name__ == "__main__":
    main()
