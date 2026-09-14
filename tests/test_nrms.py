import pytest
import torch

from recforge.models.nrms import MaskedAdditiveAttention, NRMSRanker, sampled_softmax_loss


def _inputs() -> tuple[torch.Tensor, ...]:
    history_ids = torch.tensor([[[2, 3, 0], [4, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    history_tokens = history_ids != 0
    history_items = history_tokens.any(dim=2)
    candidate_ids = torch.tensor([[[2, 3, 0], [4, 0, 0]], [[3, 0, 0], [2, 4, 0]]])
    candidate_tokens = candidate_ids != 0
    return history_ids, history_tokens, history_items, candidate_ids, candidate_tokens


def test_masked_additive_attention_ignores_padding_and_handles_empty_rows() -> None:
    torch.manual_seed(2)
    attention = MaskedAdditiveAttention(4, 3)
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    expected = attention(features, mask)
    changed = features.clone()
    changed[0, 2] = 1_000
    actual = attention(changed, mask)

    assert torch.allclose(actual[0], expected[0])
    assert torch.equal(actual[1], torch.zeros(4))


def test_nrms_scores_are_finite_with_empty_history_and_padding_invariant() -> None:
    torch.manual_seed(3)
    model = NRMSRanker(8, embedding_dim=8, attention_heads=2, attention_hidden_dim=4).eval()
    inputs = _inputs()
    with torch.no_grad():
        expected = model(*inputs)
        changed_history = inputs[0].clone()
        changed_history[~inputs[1]] = 7
        changed_candidates = inputs[3].clone()
        changed_candidates[~inputs[4]] = 7
        actual = model(
            changed_history,
            inputs[1],
            inputs[2],
            changed_candidates,
            inputs[4],
        )

    assert expected.shape == (2, 2)
    assert torch.isfinite(expected).all()
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.equal(expected[1], torch.zeros(2))


def test_nrms_backward_and_parameter_budget() -> None:
    torch.manual_seed(5)
    model = NRMSRanker(19_032, embedding_dim=128, attention_heads=8)
    small_model = NRMSRanker(8, embedding_dim=8, attention_heads=2, attention_hidden_dim=4)
    scores = small_model(*_inputs())
    loss = sampled_softmax_loss(scores, torch.zeros(2, dtype=torch.long))
    loss.backward()  # type: ignore[no-untyped-call]

    assert loss.isfinite()
    assert all(parameter.grad is not None for parameter in small_model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < 5_000_000


def test_sampled_softmax_validates_targets() -> None:
    with pytest.raises(ValueError, match="target_indices"):
        sampled_softmax_loss(torch.ones(2, 3), torch.zeros((2, 1), dtype=torch.long))


def test_mean_pooling_baseline_is_padding_invariant_and_smaller() -> None:
    torch.manual_seed(13)
    mean_model = NRMSRanker(
        8,
        embedding_dim=8,
        attention_heads=2,
        attention_hidden_dim=4,
        title_encoder_mode="mean",
        history_encoder_mode="mean",
    ).eval()
    attention_model = NRMSRanker(8, embedding_dim=8, attention_heads=2, attention_hidden_dim=4)
    inputs = _inputs()
    with torch.no_grad():
        expected = mean_model(*inputs)
        changed_history = inputs[0].clone()
        changed_history[~inputs[1]] = 7
        changed_candidates = inputs[3].clone()
        changed_candidates[~inputs[4]] = 7
        actual = mean_model(changed_history, inputs[1], inputs[2], changed_candidates, inputs[4])

    assert torch.allclose(actual, expected)
    assert sum(parameter.numel() for parameter in mean_model.parameters()) < sum(
        parameter.numel() for parameter in attention_model.parameters()
    )


@pytest.mark.parametrize("mode", ["invalid", "transformer"])
def test_nrms_rejects_unknown_encoder_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="encoder_mode"):
        NRMSRanker(8, 8, 2, 4, title_encoder_mode=mode)
    with pytest.raises(ValueError, match="encoder_mode"):
        NRMSRanker(8, 8, 2, 4, history_encoder_mode=mode)


def test_batch_title_deduplication_preserves_scores_and_gradients() -> None:
    torch.manual_seed(17)
    reference = NRMSRanker(8, 8, 2, 4)
    deduplicated = NRMSRanker(8, 8, 2, 4, deduplicate_titles=True)
    deduplicated.load_state_dict(reference.state_dict())
    reference_scores = reference(*_inputs())
    deduplicated_scores = deduplicated(*_inputs())
    reference_loss = sampled_softmax_loss(reference_scores, torch.zeros(2, dtype=torch.long))
    deduplicated_loss = sampled_softmax_loss(deduplicated_scores, torch.zeros(2, dtype=torch.long))
    reference_loss.backward()  # type: ignore[no-untyped-call]
    deduplicated_loss.backward()  # type: ignore[no-untyped-call]

    assert torch.allclose(deduplicated_scores, reference_scores, atol=1e-6)
    assert torch.allclose(deduplicated_loss, reference_loss, atol=1e-6)
    for expected, actual in zip(reference.parameters(), deduplicated.parameters(), strict=True):
        assert expected.grad is not None and actual.grad is not None
        assert torch.allclose(actual.grad, expected.grad, atol=1e-5)
