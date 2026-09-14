from io import BytesIO

import pytest
import torch

from recforge.models.two_tower import TwoTowerRetriever, in_batch_softmax_loss


def test_two_tower_outputs_normalized_embeddings_and_gradients() -> None:
    torch.manual_seed(7)
    model = TwoTowerRetriever(5, 7, embedding_dim=3, hidden_dim=8)
    users, items = model(torch.randn(4, 5), torch.randn(4, 7))
    loss = in_batch_softmax_loss(users, items, symmetric=True)
    loss.backward()  # type: ignore[no-untyped-call]

    assert users.shape == (4, 3)
    assert items.shape == (4, 3)
    assert torch.allclose(users.norm(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(items.norm(dim=1), torch.ones(4), atol=1e-6)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_two_tower_state_dict_round_trip() -> None:
    torch.manual_seed(11)
    model = TwoTowerRetriever(3, 4, embedding_dim=2, hidden_dim=5).eval()
    user_features = torch.randn(2, 3)
    item_features = torch.randn(2, 4)
    expected = model(user_features, item_features)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)

    restored = TwoTowerRetriever(3, 4, embedding_dim=2, hidden_dim=5).eval()
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(user_features, item_features)

    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])


def test_two_tower_overfits_tiny_paired_batch() -> None:
    torch.manual_seed(19)
    model = TwoTowerRetriever(6, 6, embedding_dim=8, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    features = torch.eye(6)

    with torch.no_grad():
        initial = in_batch_softmax_loss(*model(features, features)).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = in_batch_softmax_loss(*model(features, features))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    assert loss.item() < 0.02
    assert loss.item() < initial / 20


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_in_batch_loss_rejects_nonpositive_temperature(temperature: float) -> None:
    embeddings = torch.eye(2)
    with pytest.raises(ValueError, match="temperature"):
        in_batch_softmax_loss(embeddings, embeddings, temperature=temperature)


def test_in_batch_loss_requires_a_negative() -> None:
    with pytest.raises(ValueError, match="at least two"):
        in_batch_softmax_loss(torch.ones(1, 2), torch.ones(1, 2))
