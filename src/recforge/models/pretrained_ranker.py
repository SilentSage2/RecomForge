"""Minimal ranker over frozen pretrained item representations."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualFeatureAdapter(nn.Module):
    """Identity-initialized low-rank correction for frozen input features."""

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if input_dim <= 0 or rank <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.normalization = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, input_dim)
        self.scale = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        correction = cast(Tensor, self.up(F.gelu(self.down(self.normalization(features)))))
        return features + self.scale * correction


class FrozenFeatureRanker(nn.Module):
    """Share one learned projection across clicked and candidate news."""

    def __init__(
        self,
        input_dim: int = 384,
        embedding_dim: int = 64,
        temperature: float = 0.07,
        adapter_rank: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if adapter_rank is not None and adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive when provided")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.adapter = (
            ResidualFeatureAdapter(input_dim, adapter_rank) if adapter_rank is not None else None
        )
        self.normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, embedding_dim)

    def encode_items(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError("item feature dimension does not match the ranker")
        if self.adapter is not None:
            features = self.adapter(features)
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
