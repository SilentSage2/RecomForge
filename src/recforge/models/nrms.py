"""Compact masked NRMS-style title and user encoders for official MIND ranking."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_sequence(features: Tensor, mask: Tensor) -> None:
    if features.ndim != 3:
        raise ValueError("sequence features must have shape [batch, length, dimension]")
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("sequence mask must be boolean with shape [batch, length]")


def _safe_attention_mask(mask: Tensor) -> Tensor:
    """Give PyTorch attention one finite key while preserving the original pool mask."""
    safe = mask.clone()
    empty = ~safe.any(dim=1)
    if bool(empty.any()):
        safe[empty, 0] = True
    return safe


def _masked_mean(features: Tensor, mask: Tensor) -> Tensor:
    _validate_sequence(features, mask)
    weights = mask.to(features.dtype).unsqueeze(-1)
    return torch.sum(features * weights, dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MaskedAdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        _validate_sequence(features, mask)
        logits = self.score(torch.tanh(self.projection(features))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class TitleEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        attention_heads: int = 8,
        attention_hidden_dim: int = 128,
        encoder_mode: str = "attention",
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must include padding and unknown tokens")
        if embedding_dim <= 0 or embedding_dim % attention_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by attention_heads")
        if encoder_mode not in {"attention", "mean"}:
            raise ValueError("title encoder_mode must be attention or mean")
        self.encoder_mode = encoder_mode
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.self_attention: nn.MultiheadAttention | None = None
        self.normalization: nn.LayerNorm | None = None
        self.pooling: MaskedAdditiveAttention | None = None
        if encoder_mode == "attention":
            self.self_attention = nn.MultiheadAttention(
                embedding_dim, attention_heads, batch_first=True
            )
            self.normalization = nn.LayerNorm(embedding_dim)
            self.pooling = MaskedAdditiveAttention(embedding_dim, attention_hidden_dim)

    def forward(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("title token IDs must be int64 with shape [batch, tokens]")
        if token_mask.shape != token_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("title token mask must be boolean and match token IDs")
        embedded = self.embedding(token_ids)
        if self.encoder_mode == "mean":
            return _masked_mean(embedded, token_mask)
        assert self.self_attention is not None
        assert self.normalization is not None
        assert self.pooling is not None
        safe_mask = _safe_attention_mask(token_mask)
        contextual, _ = self.self_attention(
            embedded,
            embedded,
            embedded,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        contextual = self.normalization(embedded + contextual)
        pooled: Tensor = self.pooling(contextual, token_mask)
        return pooled


class UserEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        attention_heads: int = 8,
        attention_hidden_dim: int = 128,
        encoder_mode: str = "attention",
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % attention_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by attention_heads")
        if encoder_mode not in {"attention", "mean"}:
            raise ValueError("user encoder_mode must be attention or mean")
        self.encoder_mode = encoder_mode
        self.self_attention: nn.MultiheadAttention | None = None
        self.normalization: nn.LayerNorm | None = None
        self.pooling: MaskedAdditiveAttention | None = None
        if encoder_mode == "attention":
            self.self_attention = nn.MultiheadAttention(
                embedding_dim, attention_heads, batch_first=True
            )
            self.normalization = nn.LayerNorm(embedding_dim)
            self.pooling = MaskedAdditiveAttention(embedding_dim, attention_hidden_dim)

    def forward(self, clicked_news: Tensor, history_mask: Tensor) -> Tensor:
        _validate_sequence(clicked_news, history_mask)
        if self.encoder_mode == "mean":
            return _masked_mean(clicked_news, history_mask)
        assert self.self_attention is not None
        assert self.normalization is not None
        assert self.pooling is not None
        safe_mask = _safe_attention_mask(history_mask)
        contextual, _ = self.self_attention(
            clicked_news,
            clicked_news,
            clicked_news,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        contextual = self.normalization(clicked_news + contextual)
        pooled: Tensor = self.pooling(contextual, history_mask)
        return pooled


class NRMSRanker(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        attention_heads: int = 8,
        attention_hidden_dim: int = 128,
        title_encoder_mode: str = "attention",
        history_encoder_mode: str = "attention",
        deduplicate_titles: bool = False,
    ) -> None:
        super().__init__()
        self.title_encoder = TitleEncoder(
            vocabulary_size,
            embedding_dim,
            attention_heads,
            attention_hidden_dim,
            title_encoder_mode,
        )
        self.user_encoder = UserEncoder(
            embedding_dim,
            attention_heads,
            attention_hidden_dim,
            history_encoder_mode,
        )
        self.deduplicate_titles = deduplicate_titles

    def _encode_flat_titles(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        if not self.deduplicate_titles:
            encoded: Tensor = self.title_encoder(token_ids, token_mask)
            return encoded
        width = token_ids.shape[1]
        keys = torch.cat((token_ids, token_mask.to(token_ids.dtype)), dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        unique_embeddings = self.title_encoder(
            unique_keys[:, :width], unique_keys[:, width:].to(torch.bool)
        )
        restored: Tensor = unique_embeddings[inverse]
        return restored

    def forward(
        self,
        history_token_ids: Tensor,
        history_token_mask: Tensor,
        history_item_mask: Tensor,
        candidate_token_ids: Tensor,
        candidate_token_mask: Tensor,
    ) -> Tensor:
        if history_token_ids.ndim != 3 or candidate_token_ids.ndim != 3:
            raise ValueError("history and candidate title IDs must be rank-three")
        batch_size, history_length, title_length = history_token_ids.shape
        candidate_batch, candidate_count, candidate_title_length = candidate_token_ids.shape
        if candidate_batch != batch_size or candidate_title_length != title_length:
            raise ValueError("history and candidate title shapes are incompatible")
        if history_token_mask.shape != history_token_ids.shape:
            raise ValueError("history token mask shape mismatch")
        if candidate_token_mask.shape != candidate_token_ids.shape:
            raise ValueError("candidate token mask shape mismatch")
        if history_item_mask.shape != (batch_size, history_length):
            raise ValueError("history item mask shape mismatch")

        history_flat_ids = history_token_ids.reshape(batch_size * history_length, title_length)
        history_flat_mask = history_token_mask.reshape(batch_size * history_length, title_length)
        candidate_flat_ids = candidate_token_ids.reshape(batch_size * candidate_count, title_length)
        candidate_flat_mask = candidate_token_mask.reshape(
            batch_size * candidate_count, title_length
        )
        if self.deduplicate_titles:
            combined_count = history_flat_ids.shape[0]
            combined = self._encode_flat_titles(
                torch.cat((history_flat_ids, candidate_flat_ids), dim=0),
                torch.cat((history_flat_mask, candidate_flat_mask), dim=0),
            )
            clicked_flat = combined[:combined_count]
            candidate_flat = combined[combined_count:]
        else:
            clicked_flat = self._encode_flat_titles(history_flat_ids, history_flat_mask)
            candidate_flat = self._encode_flat_titles(candidate_flat_ids, candidate_flat_mask)
        clicked = clicked_flat.reshape(batch_size, history_length, -1)
        user = self.user_encoder(clicked, history_item_mask)
        candidates = candidate_flat.reshape(batch_size, candidate_count, -1)
        return torch.einsum("bd,bcd->bc", user, candidates)


def sampled_softmax_loss(scores: Tensor, target_indices: Tensor) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, candidates]")
    if target_indices.shape != (scores.shape[0],) or target_indices.dtype != torch.long:
        raise ValueError("target_indices must be int64 with shape [batch]")
    return F.cross_entropy(scores, target_indices)
