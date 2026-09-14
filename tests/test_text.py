import json
from pathlib import Path

import pytest

from recforge.data.text import (
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    VocabularyConfig,
    build_training_vocabulary,
    encode_title,
    load_vocabulary_artifact,
    tokenize_title,
    write_vocabulary_artifact,
)


def _write_news(path: Path, titles: list[str]) -> Path:
    rows = [f"N{index}\tcat\tsub\t{title}\tabs\turl\t[]\t[]" for index, title in enumerate(titles)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_tokenizer_lowercases_and_removes_punctuation() -> None:
    assert tokenize_title("AI's New Era—2027!") == ("ai", "s", "new", "era", "2027")


def test_vocabulary_uses_frequency_then_lexical_tie_breaking(tmp_path: Path) -> None:
    train = _write_news(tmp_path / "train.tsv", ["zeta beta beta", "alpha zeta rare"])
    vocabulary = build_training_vocabulary(
        [train], VocabularyConfig(max_vocab_size=5, min_frequency=1)
    )
    assert vocabulary.tokens == (PAD_TOKEN, UNKNOWN_TOKEN, "beta", "zeta", "alpha")


def test_dev_only_tokens_are_unknown_and_titles_are_truncated_and_padded(tmp_path: Path) -> None:
    train = _write_news(tmp_path / "train.tsv", ["known token"])
    _write_news(tmp_path / "dev.tsv", ["validation secret"])
    vocabulary = build_training_vocabulary(
        [train], VocabularyConfig(max_vocab_size=10, min_frequency=1)
    )

    encoded = encode_title("known validation extra", vocabulary, max_title_tokens=2)
    assert encoded.token_ids == (vocabulary.token_to_id()["known"], 1)
    assert encoded.attention_mask == (True, True)
    padded = encode_title("known", vocabulary, max_title_tokens=3)
    assert padded.token_ids[-2:] == (0, 0)
    assert padded.attention_mask == (True, False, False)


def test_vocabulary_artifact_is_immutable_and_detects_tampering(tmp_path: Path) -> None:
    train = _write_news(tmp_path / "train.tsv", ["alpha beta", "alpha gamma"])
    config = VocabularyConfig(max_vocab_size=10, min_frequency=1, max_title_tokens=4)
    vocabulary = build_training_vocabulary([train], config)
    output = tmp_path / "vocabulary"
    write_vocabulary_artifact(output, vocabulary, config, [train])

    restored, restored_config = load_vocabulary_artifact(output)
    assert restored == vocabulary
    assert restored_config == config
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_role"] == "training_news_only"
    assert len(manifest["semantic_vocabulary_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_vocabulary_artifact(output, vocabulary, config, [train])

    (output / "vocabulary.json").write_text('["<pad>", "<unk>", "changed"]\n')
    with pytest.raises(ValueError, match="fingerprint"):
        load_vocabulary_artifact(output)
