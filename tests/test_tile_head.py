from dataclasses import replace

import torch
from torch import nn

from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.models.policy import spatial_features
from cr_rl.training.bc_optimizer import (
    carry_bc_optimizer,
    make_bc_optimizer,
    optimizer_parameter_names,
)


def test_query_identity_survives_one_entity_and_head_stays_fp32_under_autocast():
    torch.manual_seed(17)
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=4,
        temporal_dim=16,
        temporal_heads=4,
        temporal_layers=1,
        header_dim=4,
        entity_feature_dim=3,
        context_len=2,
    )
    legacy = ActorCritic(cfg)
    fixed = ActorCritic(replace(cfg, tile_query_residual=True, tile_head_fp32=True))
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    assert list(legacy.state_dict()) == list(fixed.state_dict())
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in fixed.parameters())
    context = torch.randn(2, 16)
    memory = torch.randn(2, 1, 16)
    mask = torch.ones(2, 1, dtype=torch.bool)
    cards = torch.randn(2, 4, 16)
    old = legacy.placement_logits(context, memory, mask, cards)
    torch.testing.assert_close(old, old[..., :1].expand_as(old), atol=0, rtol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = fixed.placement_logits(context, memory, mask, cards)
    assert new.dtype == torch.float32
    assert float(new.detach().std(-1).min()) > 0
    torch.nn.functional.cross_entropy(new[:, 0], torch.tensor([7, 12])).backward()
    assert fixed.tile_queries.grad.norm() > 0


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_queries = nn.Parameter(torch.randn(2, 3))
        self.backbone = nn.Linear(3, 3)
        self.tile_actor = nn.Linear(3, 4)


def test_spatial_basis_is_shared_for_entity_and_grid_coordinates_and_has_no_state_keys():
    coords = torch.tensor([[0.5 / 18, 0.5 / 32], [1.5 / 18, 0.5 / 32]])
    features = spatial_features(coords, 32)
    torch.testing.assert_close(features[:, :2], coords)
    assert not torch.equal(features[0], features[1])
    cfg = ModelConfig(
        entity_dim=32,
        entity_heads=4,
        temporal_dim=32,
        temporal_heads=4,
        temporal_layers=1,
        entity_feature_dim=6,
        context_len=2,
        spatial_fourier=True,
    )
    model = ActorCritic(cfg)
    torch.testing.assert_close(model.tile_grid_features[:2], features)
    assert "tile_grid_features" not in model.state_dict()


def test_legacy_optimizer_moments_survive_parameter_group_split():
    model = SmallPolicy()
    original = make_bc_optimizer(model, 3e-5)
    sum(p.square().mean() for p in model.parameters()).backward()
    original.step()
    state = original.state_dict()
    split = make_bc_optimizer(model, 3e-5, 3e-4)
    carry_bc_optimizer(model, split, state)
    assert [g["lr"] for g in split.param_groups] == [3e-5, 3e-4]
    for parameter in model.parameters():
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(split.state[parameter][key], original.state[parameter][key])
    restored = make_bc_optimizer(model, 3e-5, 3e-4)
    carry_bc_optimizer(model, restored, split.state_dict(), optimizer_parameter_names(model, split))
    assert [g["lr"] for g in restored.param_groups] == [3e-5, 3e-4]
    for parameter in model.parameters():
        torch.testing.assert_close(
            restored.state[parameter]["exp_avg"], split.state[parameter]["exp_avg"]
        )
