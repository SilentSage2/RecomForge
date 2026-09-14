import math

import pytest

from recforge.metrics import (
    binary_auc,
    catalog_coverage_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hand_computed_retrieval_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert reciprocal_rank_at_k(ranked, relevant, 3) == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected)


def test_duplicate_results_do_not_receive_duplicate_credit() -> None:
    ranked = ["a", "a", "b"]
    assert recall_at_k(ranked, {"a", "b"}, 2) == 1.0
    assert ndcg_at_k(ranked, {"a", "b"}, 2) == 1.0


def test_catalog_coverage_checks_catalog_membership() -> None:
    assert catalog_coverage_at_k([["a", "b"], ["b", "c"]], {"a", "b", "c", "d"}, 2) == 0.75
    with pytest.raises(ValueError, match="outside the catalog"):
        catalog_coverage_at_k([["unknown"]], {"a"}, 1)


def test_pairwise_auc_gives_half_credit_to_ties() -> None:
    assert binary_auc([1, 0, 1, 0], [0.9, 0.1, 0.5, 0.5]) == pytest.approx(0.875)


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k, ndcg_at_k])
def test_metrics_reject_empty_relevance(metric: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        metric(["a"], set(), 1)  # type: ignore[operator]
