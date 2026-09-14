"""Minimal two-tower retrieval model and contrastive training objective."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProjectionTower(nn.Module):
    """Project precomputed features into a normalized retrieval space."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if min(input_dim, embedding_dim, hidden_dim) <= 0:
            raise ValueError("tower dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("tower features must have shape [batch, feature_dim]")
        return F.normalize(self.network(features), dim=-1)


class TwoTowerRetriever(nn.Module):
    """Independent user and item encoders with dot-product retrieval scores."""

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_tower = ProjectionTower(user_input_dim, embedding_dim, hidden_dim)
        self.item_tower = ProjectionTower(item_input_dim, embedding_dim, hidden_dim)

    def encode_users(self, user_features: Tensor) -> Tensor:
        return cast(Tensor, self.user_tower(user_features))

    def encode_items(self, item_features: Tensor) -> Tensor:
        return cast(Tensor, self.item_tower(item_features))

    def forward(self, user_features: Tensor, item_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_users(user_features), self.encode_items(item_features)


def in_batch_softmax_loss(
    user_embeddings: Tensor,
    positive_item_embeddings: Tensor,
    *,
    temperature: float = 0.07,
    symmetric: bool = False,
    positive_item_ids: Tensor | None = None,
) -> Tensor:
    """Treat paired batch items as positives and all off-diagonal items as negatives."""
    if user_embeddings.ndim != 2 or positive_item_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embedding_dim]")
    if user_embeddings.shape != positive_item_embeddings.shape:
        raise ValueError("user and positive item embeddings must have equal shapes")
    if user_embeddings.shape[0] < 2:
        raise ValueError("in-batch negatives require at least two pairs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if positive_item_ids is not None and (
        positive_item_ids.ndim != 1 or positive_item_ids.shape[0] != user_embeddings.shape[0]
    ):
        raise ValueError("positive_item_ids must have shape [batch]")

    logits = user_embeddings @ positive_item_embeddings.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    if positive_item_ids is not None:
        same_item = positive_item_ids[:, None] == positive_item_ids[None, :]
        off_diagonal = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_item & off_diagonal, float("-inf"))
    user_to_item: Tensor = F.cross_entropy(logits, labels)
    if not symmetric:
        return user_to_item
    item_to_user: Tensor = F.cross_entropy(logits.transpose(0, 1), labels)
    return (user_to_item + item_to_user) / 2
