from types import SimpleNamespace

import pytest
import torch

from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig


def configuration(pooling="dynamic4"):
    schema = SimpleNamespace(
        header_fields=(
            "elixir",
            "hand_card",
            "next_card",
            "deck",
            "enemy_seen",
            "pending_card",
            "ability_champion",
            "enemy_play_card",
        ),
        entity_fields=(
            "id",
            "kind",
            "side",
            "row",
            "x",
            "y",
            "state",
            "flags",
            "buff_0",
            "buff_1",
            "buff_2",
        ),
    )
    return ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=2,
        context_len=3,
        header_dim=8,
        entity_feature_dim=11,
    ).for_v2(schema, pooling)


def inputs():
    header = torch.zeros(2, 3, 8)
    entities = torch.zeros(2, 3, 4, 11)
    entities[..., 3] = 1 / 1466
    entities[..., 7] = 1 / 65535
    entities[..., 8:] = -1
    mask = torch.tensor([True, True, False, False]).expand(2, 3, 4).clone()
    legal = torch.ones(2, 2307, dtype=torch.bool)
    time_mask = torch.tensor([[False, False, True], [True, True, True]])
    return header, entities, mask, legal, time_mask


@pytest.mark.parametrize("pooling", ["mean", "static1", "static4", "dynamic4"])
def test_v2_pooling_ablation_masks_and_discrete_gradients(pooling):
    torch.manual_seed(42)
    model = ActorCritic(configuration(pooling))
    values = inputs()
    logits, value = model(*values)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones(2))
    (-logits[:, 7].mean() + value.square().mean()).backward()
    assert model.frame.category_embeddings["row"].weight.grad[2].norm() > 0
    assert model.frame.flag_projection.weight.grad[:, 0].norm() > 0
    assert model.card_embedding.weight.grad[1].norm() > 0
    if pooling == "dynamic4":
        assert model.frame.dynamic_queries[-1].weight.grad.norm() > 0
    model.eval()
    with torch.inference_mode():
        reference = model(*values)[0]
        changed = values[1].clone()
        changed[:, :, 2:] = 1000
        result = model(values[0], changed, *values[2:])[0]
        torch.testing.assert_close(reference, result, atol=2e-6, rtol=2e-6)


def test_v2_deck_and_next_card_affect_each_frame_before_temporal_encoding():
    model = ActorCritic(configuration()).eval()
    header, entities, mask, *_ = inputs()
    flat = (header.flatten(0, 1), entities.flatten(0, 1), mask.flatten(0, 1))
    base = model.encode_frames(*flat)[0]
    for index in (2, 3, 4, 5, 6, 7):
        changed = flat[0].clone()
        changed[:, index] = 19 / 1466
        altered = model.encode_frames(changed, *flat[1:])[0]
        assert not torch.allclose(base, altered)


def test_v2_critic_export_and_serialized_configuration_preserve_representation():
    cfg = configuration()
    actor = ActorPolicy.from_shared(ActorCritic(cfg))
    critic = HistoryValueNetwork(cfg)
    copied = critic.initialize_encoder(actor)
    assert copied["encoder_parameters_copied"] > 0
    assert not {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}
    serialized = {
        **cfg.__dict__,
        "entity_fields": list(cfg.entity_fields),
        "header_fields": list(cfg.header_fields),
    }
    assert ModelConfig(**serialized) == cfg
    assert (
        ModelConfig(header_dim=8, entity_feature_dim=11)
        .inherit_representation(cfg)
        .representation_version
        == 2
    )
    restored = ActorPolicy(ModelConfig(**serialized))
    restored.load_state_dict(actor.state_dict(), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(actor.eval()(*inputs())[0], restored.eval()(*inputs())[0])
        assert torch.isfinite(critic.eval()(*inputs())).all()
