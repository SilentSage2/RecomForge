from pathlib import Path

import torch

from recforge.data.ranking import build_title_table
from recforge.data.text import Vocabulary
from recforge.l1_experiment import (
    L1ExperimentConfig,
    evaluate_nrms,
    evaluate_nrms_cached,
    train_nrms,
)


def _write_news(path: Path) -> Path:
    path.write_text(
        "N0\tcat\tsub\tzero common\tabs\turl\t[]\t[]\n"
        "N1\tcat\tsub\tone common\tabs\turl\t[]\t[]\n"
        "N2\tcat\tsub\ttwo common\tabs\turl\t[]\t[]\n",
        encoding="utf-8",
    )
    return path


def _write_behaviors(path: Path) -> Path:
    path.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN0\tN0-1 N1-0 N2-0\n"
        "2\tU2\t11/11/2019 9:01:00 AM\tN1\tN1-1 N0-0 N2-0\n"
        "3\tU3\t11/11/2019 9:02:00 AM\tN2\tN2-1 N0-0 N1-0\n",
        encoding="utf-8",
    )
    return path


def test_tiny_l1_train_and_official_evaluation(tmp_path: Path) -> None:
    news = _write_news(tmp_path / "news.tsv")
    behaviors = _write_behaviors(tmp_path / "behaviors.tsv")
    vocabulary = Vocabulary(("<pad>", "<unk>", "zero", "one", "two", "common"))
    table = build_title_table([news], vocabulary, max_title_tokens=3)
    config = L1ExperimentConfig(
        vocabulary_artifact="vocab",
        news_paths=["news"],
        train_behaviors="train",
        eval_behaviors="eval",
        seed=7,
        epochs=2,
        batch_size=2,
        embedding_dim=8,
        attention_heads=2,
        attention_hidden_dim=4,
        negative_count=2,
        max_history_items=2,
        device="cpu",
    )
    model, losses, trained, epoch_seconds = train_nrms(
        config,
        table,
        behavior_path=behaviors,
        vocabulary_size=len(vocabulary.tokens),
        device=torch.device("cpu"),
    )
    metrics, predictions = evaluate_nrms(
        model,
        table,
        behavior_path=behaviors,
        max_history_items=2,
        device=torch.device("cpu"),
        max_impressions=None,
    )

    assert trained == 6
    assert len(losses) == 2
    assert len(epoch_seconds) == 2
    assert all(duration > 0 for duration in epoch_seconds)
    assert metrics["query_count"] == 3
    assert 0.0 <= metrics["auc"] <= 1.0
    assert len(predictions) == 3

    cached_metrics, cached_predictions = evaluate_nrms_cached(
        model,
        table,
        behavior_path=behaviors,
        max_history_items=2,
        device=torch.device("cpu"),
        max_impressions=None,
        batch_size=2,
    )
    assert cached_metrics == metrics
    assert [item[0] for item in cached_predictions] == [item[0] for item in predictions]
    for (_, cached), (_, reference) in zip(cached_predictions, predictions, strict=True):
        assert torch.allclose(torch.tensor(cached), torch.tensor(reference), atol=1e-6)
