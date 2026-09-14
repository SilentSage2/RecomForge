import json
from pathlib import Path

import pytest

from recforge.reporting import aggregate_l1_comparisons


def _write_seed(root: Path, seed: int, baseline: float, candidate: float) -> Path:
    bootstrap_path = root / f"bootstrap-{seed}.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "metrics": {
                    metric: {"mean_difference": candidate - baseline}
                    for metric in ("auc", "mrr", "ndcg@5", "ndcg@10")
                },
            }
        ),
        encoding="utf-8",
    )
    comparison_path = root / f"comparison-{seed}.json"
    comparison_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "protocol": "fixture",
                "train_examples_per_variant": 10,
                "dev_impressions": 5,
                "paired_bootstrap": bootstrap_path.name,
                "variants": {
                    "mean_mean": {
                        "duration_seconds": 2.0,
                        "parameters": 10,
                        "metrics": {
                            metric: baseline for metric in ("auc", "mrr", "ndcg@5", "ndcg@10")
                        },
                    },
                    "title_attention_history_mean": {
                        "duration_seconds": 4.0,
                        "parameters": 12,
                        "metrics": {
                            metric: candidate for metric in ("auc", "mrr", "ndcg@5", "ndcg@10")
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return comparison_path


def test_aggregate_l1_comparisons_reports_seed_variance(tmp_path: Path) -> None:
    paths = [_write_seed(tmp_path, 1, 0.5, 0.6), _write_seed(tmp_path, 2, 0.6, 0.8)]
    result = aggregate_l1_comparisons(paths, repository_root=tmp_path)

    assert result["seed_count"] == 2
    variants = result["variants"]
    assert isinstance(variants, dict)
    attention = variants["title_attention_history_mean"]
    assert isinstance(attention, dict)
    metrics = attention["metrics"]
    assert isinstance(metrics, dict)
    auc = metrics["auc"]
    assert isinstance(auc, dict)
    assert auc["mean"] == pytest.approx(0.7)
    effects = result["paired_effects_attention_minus_mean"]
    assert isinstance(effects, dict)
    auc_effect = effects["auc"]
    assert isinstance(auc_effect, dict)
    assert auc_effect["mean"] == pytest.approx(0.15)


def test_aggregate_l1_comparisons_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = _write_seed(tmp_path, 1, 0.5, 0.6)
    with pytest.raises(ValueError, match="unique"):
        aggregate_l1_comparisons([path, path], repository_root=tmp_path)
