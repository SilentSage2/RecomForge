"""Minimal ranker over frozen pretrained item representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenFeatureRanker(nn.Module):
    """Share one learned projection across clicked and candidate news."""

    def __init__(
        self, input_dim: int = 384, embedding_dim: int = 64, temperature: float = 0.07
    ) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, embedding_dim)

    def encode_items(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("item feature dimension does not match the ranker")
        return F.normalize(self.projection(self.normalization(features)), dim=-1)

    def forward(
        self,
        history_features: Tensor,
        history_mask: Tensor,
        candidate_features: Tensor,
    ) -> Tensor:
        if history_features.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("history and candidate features must be rank-three")
        if history_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("history and candidate batch sizes must match")
        if history_mask.shape != history_features.shape[:2] or history_mask.dtype != torch.bool:
            raise ValueError("history mask must be boolean with shape [batch, history]")
        clicked = self.encode_items(history_features)
        candidates = self.encode_items(candidate_features)
        weights = history_mask.to(clicked.dtype).unsqueeze(-1)
        user = (clicked * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        user = F.normalize(user, dim=-1)
        return torch.einsum("bd,bcd->bc", user, candidates) / self.temperature
