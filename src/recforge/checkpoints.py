"""Deterministic model-state serialization for reproducible checkpoint hashes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from torch import nn


def serialize_state_dict(model: nn.Module) -> bytes:
    """Serialize through a fixed in-memory archive name instead of a temp basename."""
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def save_training_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically persist trusted, local training state for interruption recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    buffer = BytesIO()
    torch.save(state, buffer)
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def load_training_state(path: Path, *, device: torch.device) -> dict[str, Any]:
    """Load a training state produced locally by :func:`save_training_state`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must contain a mapping")
    return payload
