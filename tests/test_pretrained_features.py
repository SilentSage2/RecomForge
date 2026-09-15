from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recforge.data.features import ItemFeatureTable
from recforge.data.pretrained import (
    PretrainedFeatureConfig,
    encode_titles,
    load_pretrained_artifact,
    masked_mean_pool,
    write_pretrained_artifact,
)


class _Tokenizer:
    def __call__(self, titles: list[str], **_: object) -> dict[str, torch.Tensor]:
        lengths = [min(len(title.split()), 3) for title in titles]
        ids = torch.zeros((len(titles), 3), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _Model:
    def eval(self) -> "_Model":
        return self

    def __call__(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def test_masked_mean_pool_ignores_padding_and_handles_empty() -> None:
    hidden = torch.tensor([[[1.0], [3.0]], [[4.0], [8.0]]])
    mask = torch.tensor([[1, 0], [0, 0]])
    assert torch.equal(masked_mean_pool(hidden, mask), torch.tensor([[1.0], [0.0]]))


def test_encode_titles_normalizes_and_validates_dimension() -> None:
    config = PretrainedFeatureConfig(dimension=4, batch_size=1)
    features = encode_titles(
        ("one two", "one two three"),
        tokenizer=_Tokenizer(),
        model=_Model(),
        config=config,
        device=torch.device("cpu"),
    )
    assert features.shape == (2, 4)
    assert features.dtype == np.float32
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)


def test_pretrained_artifact_detects_feature_tampering(tmp_path: Path) -> None:
    news = tmp_path / "news.tsv"
    news.write_text("N1\tcat\tsub\tTitle\tabs\turl\t[]\t[]\n", encoding="utf-8")
    config = PretrainedFeatureConfig(dimension=4)
    table = ItemFeatureTable(("N1",), np.ones((1, 4), dtype=np.float32))
    output = tmp_path / "artifact"
    write_pretrained_artifact(
        output,
        table,
        titles=("Title",),
        config=config,
        news_paths=[news],
        encoding_seconds=1.0,
        device="cpu",
        dependencies={"transformers": "test"},
    )
    restored, manifest = load_pretrained_artifact(output)
    assert restored.item_ids == ("N1",)
    assert manifest["model"]["revision"] == config.revision
    np.save(output / "features.npy", np.zeros((1, 4), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="fingerprint"):
        load_pretrained_artifact(output)
