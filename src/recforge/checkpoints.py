"""Deterministic model-state serialization for reproducible checkpoint hashes."""

from __future__ import annotations

from io import BytesIO

import torch
from torch import nn


def serialize_state_dict(model: nn.Module) -> bytes:
    """Serialize through a fixed in-memory archive name instead of a temp basename."""
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()
