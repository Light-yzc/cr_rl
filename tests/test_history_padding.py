import torch

from cr_rl.models.policy import HistoryEncoder, ModelConfig


def test_left_padding_is_finite_and_cannot_change_valid_context():
    torch.manual_seed(12)
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        context_len=4,
        header_dim=5,
        entity_feature_dim=6,
    )
    model = HistoryEncoder(cfg).eval()
    header = torch.randn(2, 4, 5)
    entities = torch.randn(2, 4, 3, 6)
    entity_mask = torch.ones(2, 4, 3, dtype=torch.bool)
    time_mask = torch.tensor([[False, False, False, True], [False, True, True, True]])
    with torch.inference_mode():
        context = model.encode_history(header, entities, entity_mask, time_mask)[0]
        assert torch.isfinite(context).all()
        changed_header, changed_entities = header.clone(), entities.clone()
        changed_header[~time_mask] = 1000
        changed_entities[~time_mask] = -1000
        changed = model.encode_history(changed_header, changed_entities, entity_mask, time_mask)[0]
        torch.testing.assert_close(context, changed)
    model.train()
    training = model.encode_history(header, entities, entity_mask, time_mask)[0]
    torch.testing.assert_close(context, training, atol=1e-5, rtol=1e-5)
    training.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
