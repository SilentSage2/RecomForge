from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from recforge.data.features import ItemFeatureTable
from recforge.data.protocol import TemporalCatalogIndex
from recforge.evaluation import evaluate_temporal_corpus
from recforge.models.two_tower import TwoTowerRetriever


def test_temporal_corpus_evaluation_returns_bounded_metrics(tmp_path: Path) -> None:
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(
        "\n".join(
            [
                "1\tU1\t11/11/2019 9:00:00 AM\ti0\ti0-1 i1-0",
                "2\tU2\t11/11/2019 9:01:00 AM\ti1\ti1-1 i2-0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    table = ItemFeatureTable(
        item_ids=("i0", "i1", "i2"),
        features=np.eye(3, dtype=np.float32),
    )
    start = datetime(2019, 11, 11, 9)
    catalog = TemporalCatalogIndex(
        observed_at=(start, start, start + timedelta(minutes=1)),
        item_ids=("i0", "i1", "i2"),
    )
    model = TwoTowerRetriever(3, 3, embedding_dim=2, hidden_dim=4)

    metrics = evaluate_temporal_corpus(
        model,
        table,
        catalog=catalog,
        behavior_path=behaviors,
        training_popularity=Counter({"i0": 2, "i1": 1}),
        max_history_items=5,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert metrics["query_count"] == 2
    assert 0.0 <= metrics["recall@20"] <= 1.0
    assert 0.0 <= metrics["coverage@100"] <= 1.0
    assert metrics["exact_search_ms_per_query"] >= 0.0
