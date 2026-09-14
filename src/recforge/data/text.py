"""Deterministic, training-only title vocabulary artifacts for L1 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from recforge.data.mind import iter_mind_news, sha256_file

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class VocabularyConfig:
    max_vocab_size: int = 30_000
    min_frequency: int = 2
    max_title_tokens: int = 30

    def __post_init__(self) -> None:
        if self.max_vocab_size < 2:
            raise ValueError("max_vocab_size must reserve padding and unknown tokens")
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if self.max_title_tokens <= 0:
            raise ValueError("max_title_tokens must be positive")


@dataclass(frozen=True, slots=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or self.tokens[:2] != (PAD_TOKEN, UNKNOWN_TOKEN):
            raise ValueError("vocabulary must begin with padding and unknown tokens")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}


@dataclass(frozen=True, slots=True)
class EncodedTitle:
    token_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]


def tokenize_title(title: str) -> tuple[str, ...]:
    """Apply the declared lowercase Unicode-word title tokenizer."""
    return tuple(_TOKEN_PATTERN.findall(title.lower()))


def build_training_vocabulary(train_news_paths: list[Path], config: VocabularyConfig) -> Vocabulary:
    """Build a vocabulary only from explicitly declared training news files."""
    if not train_news_paths:
        raise ValueError("at least one training news file is required")
    counts: Counter[str] = Counter()
    for path in train_news_paths:
        for news in iter_mind_news(path):
            counts.update(tokenize_title(news.title))
    eligible = (item for item in counts.items() if item[1] >= config.min_frequency)
    ordered = sorted(eligible, key=lambda item: (-item[1], item[0]))
    learned = tuple(token for token, _ in ordered[: config.max_vocab_size - 2])
    return Vocabulary((PAD_TOKEN, UNKNOWN_TOKEN, *learned))


def encode_title(title: str, vocabulary: Vocabulary, *, max_title_tokens: int) -> EncodedTitle:
    if max_title_tokens <= 0:
        raise ValueError("max_title_tokens must be positive")
    mapping = vocabulary.token_to_id()
    tokens = tokenize_title(title)[:max_title_tokens]
    token_ids = [mapping.get(token, 1) for token in tokens]
    mask = [True] * len(token_ids)
    padding = max_title_tokens - len(token_ids)
    token_ids.extend([0] * padding)
    mask.extend([False] * padding)
    return EncodedTitle(tuple(token_ids), tuple(mask))


def _vocabulary_digest(tokens: tuple[str, ...]) -> str:
    payload = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_vocabulary_artifact(
    output_directory: Path,
    vocabulary: Vocabulary,
    config: VocabularyConfig,
    train_news_paths: list[Path],
) -> None:
    """Write an immutable artifact whose manifest declares training-only provenance."""
    output_directory.mkdir(parents=True, exist_ok=False)
    vocabulary_path = output_directory / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(vocabulary.tokens, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifact_schema_version": 1,
        "config": asdict(config),
        "provenance_role": "training_news_only",
        "train_news_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in train_news_paths
        ],
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "semantic_vocabulary_sha256": _vocabulary_digest(vocabulary.tokens),
        "vocabulary_size": len(vocabulary.tokens),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_vocabulary_artifact(
    artifact_directory: Path,
) -> tuple[Vocabulary, VocabularyConfig]:
    manifest = json.loads((artifact_directory / "manifest.json").read_text(encoding="utf-8"))
    vocabulary_path = artifact_directory / "vocabulary.json"
    if manifest.get("artifact_schema_version") != 1:
        raise ValueError("unsupported vocabulary artifact schema")
    if manifest.get("provenance_role") != "training_news_only":
        raise ValueError("vocabulary artifact lacks training-only provenance")
    if sha256_file(vocabulary_path) != manifest.get("vocabulary_sha256"):
        raise ValueError("vocabulary fingerprint mismatch")
    raw_tokens = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if not isinstance(raw_tokens, list) or not all(isinstance(token, str) for token in raw_tokens):
        raise ValueError("vocabulary.json must contain a list of strings")
    vocabulary = Vocabulary(tuple(raw_tokens))
    if _vocabulary_digest(vocabulary.tokens) != manifest.get("semantic_vocabulary_sha256"):
        raise ValueError("semantic vocabulary fingerprint mismatch")
    if len(vocabulary.tokens) != manifest.get("vocabulary_size"):
        raise ValueError("vocabulary size mismatch")
    return vocabulary, VocabularyConfig(**manifest["config"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training-only MIND title vocabulary.")
    parser.add_argument("--train-news", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-vocab-size", type=int, default=30_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-title-tokens", type=int, default=30)
    args = parser.parse_args()
    config = VocabularyConfig(
        max_vocab_size=args.max_vocab_size,
        min_frequency=args.min_frequency,
        max_title_tokens=args.max_title_tokens,
    )
    vocabulary = build_training_vocabulary(args.train_news, config)
    write_vocabulary_artifact(args.output, vocabulary, config, args.train_news)


if __name__ == "__main__":
    main()
