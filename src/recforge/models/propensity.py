"""Candidate-only propensity head for cold-user fallback ranking."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class CandidatePropensityHead(nn.Module):
    """Score a frozen item representation with one affine scalar."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape [batch, input_dim]")
        return cast(Tensor, self.linear(features).squeeze(1))
