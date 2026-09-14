from io import BytesIO

import torch

from recforge.checkpoints import serialize_state_dict
from recforge.models.nrms import NRMSRanker


def test_state_dict_serialization_is_byte_deterministic_and_loadable() -> None:
    torch.manual_seed(11)
    model = NRMSRanker(8, embedding_dim=8, attention_heads=2, attention_hidden_dim=4)
    first = serialize_state_dict(model)
    second = serialize_state_dict(model)

    assert first == second
    restored = NRMSRanker(8, embedding_dim=8, attention_heads=2, attention_hidden_dim=4)
    restored.load_state_dict(torch.load(BytesIO(first), weights_only=True))
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(model.parameters(), restored.parameters(), strict=True)
    )
