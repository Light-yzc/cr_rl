from dataclasses import replace

import numpy as np
import pytest
import torch
from test_representation_v2 import configuration, inputs

from cr_rl.data.bc_sampling import epoch_rows
from cr_rl.models import ActorCritic
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.bc_optimizer import (
    carry_bc_optimizer,
    make_bc_optimizer,
    optimizer_parameter_names,
)
from cr_rl.training.readout_upgrade import load_upgraded_readouts


def test_windowed_mixing_keeps_coverage_and_bounds_the_working_set():
    offsets = np.arange(0, 801, 100)
    plays = np.arange(1, 800, 2)
    waits = np.arange(0, 800, 2)

    def sample(window):
        return epoch_rows(
            plays, waits, np.random.default_rng(8), 0.5, offsets=offsets, shard_mix_window=window
        )

    legacy, mixed = sample(1), sample(4)
    np.testing.assert_array_equal(np.sort(legacy), np.sort(mixed))
    np.testing.assert_array_equal(mixed, sample(4))
    # Equal-size shards give exactly 400 draws per four-shard window here.
    # Wait sampling is without replacement, so all eight shards are complete.
    for part in (mixed[:400], mixed[400:]):
        assert len(np.unique(part // 100)) == 4
        assert np.count_nonzero(np.diff(part // 100)) > 200
    assert not np.intersect1d(mixed[:400] // 100, mixed[400:] // 100).size
    np.testing.assert_array_equal(np.sort(mixed[np.isin(mixed, plays)]), plays)


def test_natural_timing_optimum_is_target_prior_without_diluting_conditional_choice():
    prior = 0.03
    u = torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32, requires_grad=True)
    slots = torch.stack(
        (u * 0, *(u - np.log(4) for _ in range(4)), u * 0 - 1e9, u * 0 - 1e9)
    ).log_softmax(-1)
    joint = torch.cat(
        (slots[:1], (slots[1:5, None].expand(4, 576) - np.log(576)).flatten(), slots[5:])
    ).expand(2, -1)
    loss, _ = hierarchical_bc_loss(
        joint,
        torch.tensor([0, 1]),
        torch.ones_like(joint, dtype=torch.bool),
        conditional_card_loss=True,
        position_coef=0,
        timing_prior=prior,
        sampled_action_prior=0.5,
    )
    loss.backward()
    assert abs(u.grad.item()) < 1e-6
    with pytest.raises(ValueError, match="priors"):
        hierarchical_bc_loss(
            joint,
            torch.tensor([0, 1]),
            torch.ones_like(joint, dtype=torch.bool),
            conditional_card_loss=True,
            timing_prior=0.03,
        )


def test_natural_prior_does_not_weaken_selected_card_relative_to_other_cards():
    gradients = []
    for prior in (None, 0.001):
        logits = torch.zeros(7, requires_grad=True)
        slots = logits.log_softmax(-1)
        joint = torch.cat(
            (slots[:1], (slots[1:5, None].expand(4, 576) - np.log(576)).flatten(), slots[5:])
        ).expand(2, -1)
        loss, _ = hierarchical_bc_loss(
            joint,
            torch.tensor([0, 1]),
            torch.ones_like(joint, dtype=torch.bool),
            conditional_card_loss=True,
            position_coef=0,
            timing_prior=prior,
            sampled_action_prior=0.5,
        )
        loss.backward()
        gradients.append(logits.grad[1] - logits.grad[2])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert gradients[1] < -0.9


def test_readout_migration_preserves_backbone_and_optimizer_and_opens_spatial_gradients():
    torch.manual_seed(3)
    old = ActorCritic(configuration())
    upgraded = ActorCritic(replace(old.cfg, readout_version=2))
    original_optimizer = make_bc_optimizer(old, 3e-5, 3e-4)
    (-old(*inputs())[0][:, 1].mean()).backward()
    original_optimizer.step()
    report = load_upgraded_readouts(upgraded, old.state_dict())
    assert report["new_tensors"]
    for name, value in old.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[name], value, atol=0, rtol=0)
    old.eval()
    upgraded.eval()
    with torch.no_grad():
        torch.testing.assert_close(old(*inputs())[0], upgraded(*inputs())[0], atol=3e-6, rtol=3e-6)
    optimizer = make_bc_optimizer(upgraded, 3e-5, 3e-4)
    carry_bc_optimizer(
        upgraded,
        optimizer,
        original_optimizer.state_dict(),
        optimizer_parameter_names(old, original_optimizer),
        allow_new=True,
    )
    assert [g["lr"] for g in optimizer.param_groups] == [3e-5, 3e-4]
    for name, p in old.named_parameters():
        target = dict(upgraded.named_parameters())[name]
        if p in original_optimizer.state:
            torch.testing.assert_close(
                optimizer.state[target]["exp_avg"], original_optimizer.state[p]["exp_avg"]
            )
    loss = -upgraded(*inputs())[0][:, 7].mean()
    loss.backward()
    assert upgraded.tile_spatial_gate.grad.abs() > 0
    assert upgraded.ordered_card_context["hand_card"].weight.grad.norm() > 0
    assert upgraded.frame.spatial_projection.weight.grad.norm() > 0
    bad = dict(old.state_dict())
    bad.pop("wait_actor.weight")
    with pytest.raises(ValueError, match="migration keys"):
        load_upgraded_readouts(upgraded, bad)


def test_ordered_card_residual_can_distinguish_event_card_permutations():
    cfg = configuration()
    # Use two slots for hand and pending records, while retaining all header groups.
    fields = (*cfg.header_fields, "hand_card", "pending_card", "enemy_play_card")
    cfg = replace(cfg, header_dim=len(fields), header_fields=fields, readout_version=2)
    model = ActorCritic(cfg).eval()
    header = torch.zeros(1, len(fields))
    indices = [i for i, name in enumerate(fields) if name == "pending_card"]
    header[:, indices] = torch.tensor([5, 9]) / cfg.vocab_size
    other = header.clone()
    other[:, indices] = header[:, indices[::-1]]
    with torch.no_grad():
        model.ordered_card_context["pending_card"].weight.normal_(std=0.1)
        entities = torch.zeros(1, 2, cfg.entity_feature_dim)
        mask = torch.ones(1, 2, dtype=torch.bool)
        a = model.encode_frames(header, entities, mask)[0]
        b = model.encode_frames(other, entities, mask)[0]
    assert not torch.allclose(a, b)
