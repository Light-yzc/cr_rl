from dataclasses import replace

import torch

from cr_rl.models import ActorPolicy, ModelConfig


def test_frame_precision_is_saved_and_keeps_encoder_gradients():
    base = ModelConfig(entity_dim=16, temporal_dim=16, entity_heads=4, temporal_heads=4)
    model = ActorPolicy(replace(base, frame_encoder_fp32=True))
    header = torch.randn(2, base.header_dim)
    entities = torch.randn(2, 4, base.entity_feature_dim)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        frame, tokens = model.encode_frames(header, entities, mask)
    assert frame.dtype == tokens.dtype == torch.float32
    frame.square().mean().backward()
    assert torch.isfinite(model.frame.entity_in.weight.grad).all()
    assert ModelConfig(**model.cfg.__dict__).frame_encoder_fp32
    # Precision changes have no parameter-shape migration.
    ActorPolicy(base).load_state_dict(model.state_dict(), strict=True)
