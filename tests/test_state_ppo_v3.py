"""Unit and integration tests for full-feature architecture-v3 PPO."""
from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_representation_v2 import configuration

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import StateActorCritic, StateValueNetwork
from cr_rl.models.state_temporal import StateTemporalConfig
from cr_rl.training.adaptive_kl import (
    AdaptiveActorLearningRate,
    AdaptiveActorLRConfig,
    restore_actor_lr_controller,
)
from cr_rl.training.episode_trace import EpisodeTrace
from cr_rl.training.fp16 import configure_optimizer_amp
from cr_rl.training.host_memory import streaming_torch_save
from cr_rl.training.state_ppo import (
    StatePPOConfig,
    burn_in_memory,
    evaluate_critic_segments,
    optimize_state_separate,
    prepare_state_rollout_targets,
)

HEADER_FIELDS = (
    "entity_count",
    "elixir",
    "hand_card",
    "hand_card",
    "hand_card",
    "hand_card",
    "pending_x",
    "pending_y",
)
ENTITY_FIELDS = (
    "id",
    "kind",
    "side",
    "row",
    "x",
    "y",
    "hp",
    "state",
    "flags",
    "buff_0",
    "buff_1",
)


def make_schema_and_adapter(max_entities: int = 4, entity_bucket: int | None = None):
    schema = SimpleNamespace(
        header_fields=HEADER_FIELDS,
        entity_fields=ENTITY_FIELDS,
        header_ints=len(HEADER_FIELDS),
        entity_ints=len(ENTITY_FIELDS),
        max_entities=max_entities,
        seat_ints=len(HEADER_FIELDS) + max_entities * len(ENTITY_FIELDS),
        header={
            name: tuple(i for i, field in enumerate(HEADER_FIELDS) if field == name)
            for name in set(HEADER_FIELDS)
        },
    )
    adapter = CrForgeObservationAdapter(schema, 1000, entity_bucket=entity_bucket)
    return schema, adapter


def make_state_models(seed: int = 17, *, init_encoder: bool = True):
    torch.manual_seed(seed)
    cfg = replace(
        configuration(),
        readout_version=2,
        header_dim=len(HEADER_FIELDS),
        entity_feature_dim=len(ENTITY_FIELDS),
        max_entities=4,
    )
    temporal = StateTemporalConfig(
        width=32,
        heads=4,
        kv_heads=2,
        head_dim=8,
        pattern=("kda", "gqa"),
        ffn_multiplier=2,
        chunk_size=4,
        cache_initial_capacity=8,
    )
    actor = StateActorCritic(cfg, temporal, decision_ticks=8, frame_microbatch=4)
    critic = StateValueNetwork(copy.deepcopy(cfg), temporal, decision_ticks=8, frame_microbatch=4)
    report = critic.initialize_encoder(actor) if init_encoder else None
    behavior = copy.deepcopy(actor).eval()
    for p in behavior.parameters():
        p.requires_grad_(False)
    schema, adapter = make_schema_and_adapter(max_entities=4)
    return schema, adapter, actor, critic, behavior, report


def make_synthetic_segment(
    schema,
    adapter,
    actor: StateActorCritic,
    length: int = 10,
    *,
    seed: int = 0,
    prefix_len: int = 0,
    terminated_last: bool = True,
    truncated_last: bool = False,
    forced_wait_steps: tuple[int, ...] = (),
    invalid_steps: tuple[int, ...] = (),
    reward_value: float = 0.25,
    terminal_reward: float = 1.0,
    bootstrap_tail: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    total_len = prefix_len + length + (1 if bootstrap_tail else 0)
    raw = np.zeros((total_len, schema.seat_ints), dtype=np.int32)
    raw[:, 0] = 2
    raw[:, 1] = rng.integers(10000, 90000, size=total_len)
    raw[:, 2:6] = rng.integers(1, 20, size=(total_len, 4))
    ent_base = schema.header_ints
    for e in range(2):
        off = ent_base + e * schema.entity_ints
        raw[:, off] = e + 1
        raw[:, off + 1] = rng.integers(1, 20, size=total_len)
        raw[:, off + 3] = rng.integers(100, 500, size=total_len)
        raw[:, off + 6] = rng.integers(100, 800, size=total_len)

    prefix_raw = raw[:prefix_len].copy()
    seg_raw = raw[prefix_len : prefix_len + length].copy()
    tail_raw = raw[prefix_len + length].copy() if bootstrap_tail else None

    legal_all = np.zeros((prefix_len + length, actor.cfg.action_count), dtype=bool)
    legal_all[:, 0] = True
    legal_all[:, 1:6] = True
    for s in forced_wait_steps:
        if 0 <= prefix_len + s < len(legal_all):
            legal_all[prefix_len + s, 1:] = False

    hand_idx = np.asarray(schema.header["hand_card"], dtype=np.int64)
    actor_eval = copy.deepcopy(actor).eval()
    actions = np.zeros(length, dtype=np.int64)
    old_logp = np.zeros(length, dtype=np.float32)
    h0_actor = None
    h0_behavior = None
    with torch.no_grad():
        state = actor_eval.initial_state(1)
        if prefix_len > 0:
            ph, pe, pm = adapter(prefix_raw[None])
            _, _, state = actor_eval.forward_state(
                torch.as_tensor(ph),
                torch.as_tensor(pe),
                torch.as_tensor(pm),
                legal_action_mask=torch.as_tensor(legal_all[:prefix_len][None]),
                state=state,
                hand_card_ids=torch.as_tensor(prefix_raw[None, :, hand_idx], dtype=torch.long),
            )
            h0_actor = state
            h0_behavior = state
        sh, se, sm = adapter(seg_raw[None])
        seg_legal = legal_all[prefix_len : prefix_len + length]
        logp_t, _, _ = actor_eval.forward_state(
            torch.as_tensor(sh),
            torch.as_tensor(se),
            torch.as_tensor(sm),
            legal_action_mask=torch.as_tensor(seg_legal[None]),
            state=state,
            hand_card_ids=torch.as_tensor(seg_raw[None, :, hand_idx], dtype=torch.long),
        )
        probs = logp_t[0].exp().numpy()
        for t in range(length):
            legal_choices = np.flatnonzero(seg_legal[t])
            chosen = int(legal_choices[-1] if len(legal_choices) > 1 and t % 2 == 0 else 0)
            actions[t] = chosen
            old_logp[t] = float(logp_t[0, t, chosen])
            assert np.isfinite(probs[t, chosen])

    rewards = np.full(length, reward_value, dtype=np.float32)
    terminated = np.zeros(length, dtype=bool)
    truncated = np.zeros(length, dtype=bool)
    dones = np.zeros(length, dtype=bool)
    valid = np.ones(length, dtype=bool)
    if terminated_last and length > 0:
        terminated[-1] = True
        dones[-1] = True
        rewards[-1] += terminal_reward
    if truncated_last and length > 0:
        truncated[-1] = True
        dones[-1] = True
    for s in invalid_steps:
        valid[s] = False

    return {
        "episode_id": seed + 1,
        "raw_obs": seg_raw,
        "prefix_raw": prefix_raw,
        "bootstrap_raw": tail_raw,
        "legal_masks": legal_all[prefix_len : prefix_len + length].copy(),
        "actions": actions,
        "old_log_probs": old_logp,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "dones": dones,
        "valid": valid,
        "learner_mask": np.ones(length, dtype=bool),
        "h0_actor": h0_actor,
        "h0_behavior": h0_behavior,
        "has_complete_prefix": prefix_len == 0,
        "old_values": np.zeros(length, dtype=np.float32),
        "bootstrap_value": 0.0,
        "h0_critic": None,
    }


def _cfg(**overrides) -> StatePPOConfig:
    base = dict(
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        precision="fp32",
        max_initial_logp_error=None,
        epochs=2,
        critic_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        critic_gae_lambda=None,
        reference_kl_coef=0.01,
        target_kl=0.5,
        continue_after_kl_reject=False,
        actor_choice_only=True,
        tbptt_chunk=4,
        actor_batch_size=2,
        critic_batch_size=2,
        actor_effective_batch_size=64,
        critic_effective_batch_size=64,
        critic_max_grad_norm=1.0,
        cache_reference=False,
        defer_scalar_metrics=False,
    )
    base.update(overrides)
    return StatePPOConfig(**base)


def test_state_value_network_separation_and_evaluate_segments():
    schema, adapter, actor, critic, _, report = make_state_models(seed=11)
    assert report is not None and report["encoder_parameters_copied"] > 0
    assert not ({p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()})
    assert not any(k.startswith(("card_head.", "tile_head.")) for k in critic.state_dict())

    fresh_seg = make_synthetic_segment(schema, adapter, actor, length=8, seed=1, prefix_len=0)
    cont_seg = make_synthetic_segment(
        schema, adapter, actor, length=6, seed=2, prefix_len=4, terminated_last=False, bootstrap_tail=True
    )
    h0_list, elapsed = evaluate_critic_segments(
        critic, adapter, [fresh_seg, cont_seg], device="cpu", precision="fp32", batch_size=2, chunk_size=3
    )
    assert elapsed >= 0.0
    assert len(h0_list) == 2
    assert fresh_seg["h0_critic"] is None
    assert cont_seg["h0_critic"] is not None
    assert cont_seg["h0_critic"].temporal.positions.tolist() == [4]
    assert cont_seg["old_values"].shape == (6,)
    assert np.isfinite(cont_seg["bootstrap_value"]) and cont_seg["bootstrap_value"] != 0.0


def test_actor_choice_only_mask_advantage_norm_and_separate_critic_lambda():
    schema, adapter, actor, critic, _, _ = make_state_models(seed=21)
    seg_a = make_synthetic_segment(
        schema,
        adapter,
        actor,
        length=8,
        seed=3,
        forced_wait_steps=(0, 1, 2),
        invalid_steps=(7,),
    )
    seg_b = make_synthetic_segment(
        schema,
        adapter,
        actor,
        length=8,
        seed=4,
        forced_wait_steps=(1, 3, 5),
    )
    evaluate_critic_segments(critic, adapter, [seg_a, seg_b], device="cpu", precision="fp32")
    info_a = prepare_state_rollout_targets(
        [seg_a, seg_b],
        gamma=0.99,
        gae_lambda=0.9,
        critic_gae_lambda=0.99,
        choice_only=True,
        critic_warmup=False,
    )
    assert not seg_a["actor_mask"][[0, 1, 2, 7]].any()
    assert seg_a["actor_mask"][[3, 4, 5, 6]].all()
    assert seg_a["critic_mask"][:7].all() and not seg_a["critic_mask"][7]
    selected_adv = np.concatenate([seg_a["advantages"][seg_a["actor_mask"]], seg_b["advantages"][seg_b["actor_mask"]]])
    assert len(selected_adv) == info_a["actor_eligible_count"]
    np.testing.assert_allclose(selected_adv.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(selected_adv.std(), 1.0, atol=1e-5)

    # Distinct critic_gae_lambda changes critic returns while keeping actor advantages on actor gae_lambda.
    seg_a_alt = copy.deepcopy(seg_a)
    seg_b_alt = copy.deepcopy(seg_b)
    prepare_state_rollout_targets(
        [seg_a_alt, seg_b_alt], gamma=0.99, gae_lambda=0.9, critic_gae_lambda=0.5, choice_only=True
    )
    np.testing.assert_allclose(seg_a_alt["advantages"], seg_a["advantages"], atol=1e-6)
    assert not np.allclose(seg_a_alt["returns"], seg_a["returns"])


def test_critic_warmup_uses_complete_mc_targets_only_and_freezes_actor():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=31)
    complete = make_synthetic_segment(
        schema, adapter, actor, length=6, seed=5, prefix_len=0, terminated_last=True, reward_value=0.5, terminal_reward=1.0
    )
    truncated = make_synthetic_segment(
        schema, adapter, actor, length=6, seed=6, prefix_len=0, terminated_last=False, truncated_last=True
    )
    continuing = make_synthetic_segment(
        schema, adapter, actor, length=6, seed=7, prefix_len=3, terminated_last=True
    )
    faulted = make_synthetic_segment(
        schema, adapter, actor, length=6, seed=8, prefix_len=0, terminated_last=True, invalid_steps=(2,)
    )
    segs = [complete, truncated, continuing, faulted]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    ctrl = AdaptiveActorLearningRate(AdaptiveActorLRConfig(min_lr=1e-4, max_lr=1e-2))
    ctrl.begin_update(actor_opt)

    out = optimize_state_separate(
        actor,
        critic,
        behavior,
        actor_opt,
        critic_opt,
        adapter,
        segs,
        _cfg(gamma=0.9),
        rng=np.random.default_rng(1),
        actor_device="cpu",
        critic_device="cpu",
        actor_enabled=False,
        lr_controller=ctrl,
    )
    assert out["actor_steps"] == 0
    assert out["critic_steps"] > 0
    assert out["warmup_mc_games"] == 1
    assert out["actor_lr_event"]["reason"] == "actor_disabled"
    assert complete["critic_mask"].all()
    assert not truncated["critic_mask"].any()
    assert not continuing["critic_mask"].any()
    assert not faulted["critic_mask"].any()
    for b, a in zip(actor_before, actor.parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=0, atol=0)
    assert any(not torch.equal(b, c) for b, c in zip(critic_before, critic.parameters(), strict=True))


def test_factorized_entropy_kl_stop_and_continue_after_reject():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=41)
    segs = [make_synthetic_segment(schema, adapter, actor, length=8, seed=10 + i) for i in range(4)]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    # Force measurable divergence against old_logp by perturbing actor weights first.
    with torch.no_grad():
        for p in actor.slot_actor.parameters():
            p.add_(0.25)

    actor_a = copy.deepcopy(actor)
    critic_a = copy.deepcopy(critic)
    out_stop = optimize_state_separate(
        actor_a,
        critic_a,
        behavior,
        torch.optim.AdamW(actor_a.parameters(), lr=1e-3),
        torch.optim.AdamW(critic_a.parameters(), lr=1e-3),
        adapter,
        copy.deepcopy(segs),
        _cfg(
            epochs=2,
            actor_batch_size=1,
            actor_effective_batch_size=4,
            card_entropy_coef=0.02,
            tile_entropy_coef=0.0,
            target_card_kl=1e-6,
            target_tile_kl=1e-6,
            continue_after_kl_reject=False,
        ),
        rng=np.random.default_rng(2),
        actor_device="cpu",
    )
    assert out_stop["kl_stops"] == 1
    assert out_stop["rejected_groups"] == 1
    assert out_stop["stopped_early"] is True
    assert out_stop["actor_steps"] == 0

    actor_b = copy.deepcopy(actor)
    critic_b = copy.deepcopy(critic)
    out_cont = optimize_state_separate(
        actor_b,
        critic_b,
        behavior,
        torch.optim.AdamW(actor_b.parameters(), lr=1e-3),
        torch.optim.AdamW(critic_b.parameters(), lr=1e-3),
        adapter,
        copy.deepcopy(segs),
        _cfg(
            epochs=1,
            actor_batch_size=1,
            actor_effective_batch_size=4,
            card_entropy_coef=0.0,
            tile_entropy_coef=0.02,
            target_card_kl=1e-6,
            target_tile_kl=1e-6,
            continue_after_kl_reject=True,
        ),
        rng=np.random.default_rng(2),
        actor_device="cpu",
    )
    assert out_cont["stopped_early"] is False
    assert out_cont["rejected_groups"] >= 2


def test_adaptive_actor_lr_roundtrip_and_optimize_integration():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=51)
    segs = [make_synthetic_segment(schema, adapter, actor, length=10, seed=20 + i) for i in range(4)]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=5e-5)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    ctrl = AdaptiveActorLearningRate(
        AdaptiveActorLRConfig(
            min_lr=1e-5,
            max_lr=2e-4,
            min_groups=2,
            min_samples=4,
            low_updates_before_increase=1,
        )
    )
    ctrl.begin_update(actor_opt)
    out = optimize_state_separate(
        actor,
        critic,
        behavior,
        actor_opt,
        critic_opt,
        adapter,
        segs,
        _cfg(
            epochs=1,
            actor_batch_size=2,
            actor_effective_batch_size=4,
            target_card_kl=0.5,
            target_tile_kl=0.5,
        ),
        rng=np.random.default_rng(3),
        actor_device="cpu",
        lr_controller=ctrl,
    )
    assert out["actor_lr_event"] is not None
    assert out["actor_lr_event"]["enabled"] is True
    assert out["actor_lr_event"]["groups_observed"] >= 2

    saved = ctrl.state_dict()
    restored = restore_actor_lr_controller(actor_opt, saved, enabled=True, min_lr=1e-5, max_lr=2e-4)
    assert restored is not None
    assert restored.state_dict() == saved


def test_tbptt_cross_chunk_continuity_across_effective_batch_splits():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=61)
    seg_a = make_synthetic_segment(schema, adapter, actor, length=12, seed=30)
    seg_b = make_synthetic_segment(schema, adapter, actor, length=12, seed=31)
    evaluate_critic_segments(critic, adapter, [seg_a, seg_b], device="cpu", precision="fp32", chunk_size=4)

    # Set lr=0 so weights stay frozen across optimizer steps while effective-batch splits each TBPTT chunk!
    # If hidden states were mistakenly reset on each optimizer group, the KL/losses would diverge from full T=12!
    actor_split = copy.deepcopy(actor)
    critic_split = copy.deepcopy(critic)
    out_split = optimize_state_separate(
        actor_split,
        critic_split,
        behavior,
        torch.optim.AdamW(actor_split.parameters(), lr=0.0),
        torch.optim.AdamW(critic_split.parameters(), lr=0.0),
        adapter,
        copy.deepcopy([seg_a, seg_b]),
        _cfg(
            epochs=1,
            critic_epochs=1,
            tbptt_chunk=4,
            actor_batch_size=2,
            critic_batch_size=2,
            actor_effective_batch_size=1,
            critic_effective_batch_size=1,
            reference_kl_coef=0.0,
        ),
        rng=np.random.default_rng(7),
        actor_device="cpu",
    )

    actor_whole = copy.deepcopy(actor)
    critic_whole = copy.deepcopy(critic)
    out_whole = optimize_state_separate(
        actor_whole,
        critic_whole,
        behavior,
        torch.optim.AdamW(actor_whole.parameters(), lr=0.0),
        torch.optim.AdamW(critic_whole.parameters(), lr=0.0),
        adapter,
        copy.deepcopy([seg_a, seg_b]),
        _cfg(
            epochs=1,
            critic_epochs=1,
            tbptt_chunk=12,
            actor_batch_size=2,
            critic_batch_size=2,
            actor_effective_batch_size=1024,
            critic_effective_batch_size=1024,
            reference_kl_coef=0.0,
        ),
        rng=np.random.default_rng(7),
        actor_device="cpu",
    )
    assert out_split["actor_steps"] == 3
    assert out_split["critic_steps"] == 3
    assert out_whole["actor_steps"] == 1
    assert out_split["metrics"]["policy_loss"] == pytest.approx(out_whole["metrics"]["policy_loss"], abs=1e-5)
    assert out_split["metrics"]["approx_kl"] == pytest.approx(out_whole["metrics"]["approx_kl"], abs=1e-6)
    assert out_split["metrics"]["value_loss"] == pytest.approx(out_whole["metrics"]["value_loss"], abs=1e-5)


def test_cache_reference_parity_with_inline_behavior():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=71)
    with torch.no_grad():
        for p in behavior.slot_actor.parameters():
            p.add_(0.1)
    segs = [
        make_synthetic_segment(schema, adapter, actor, length=10, seed=40 + i, prefix_len=3)
        for i in range(2)
    ]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    actor_a, critic_a = copy.deepcopy(actor), copy.deepcopy(critic)
    out_inline = optimize_state_separate(
        actor_a,
        critic_a,
        behavior,
        torch.optim.AdamW(actor_a.parameters(), lr=1e-3),
        torch.optim.AdamW(critic_a.parameters(), lr=1e-3),
        adapter,
        copy.deepcopy(segs),
        _cfg(epochs=1, tbptt_chunk=5, reference_kl_coef=0.1, cache_reference=False),
        rng=np.random.default_rng(9),
        actor_device="cpu",
    )

    actor_b, critic_b = copy.deepcopy(actor), copy.deepcopy(critic)
    out_cached = optimize_state_separate(
        actor_b,
        critic_b,
        behavior,
        torch.optim.AdamW(actor_b.parameters(), lr=1e-3),
        torch.optim.AdamW(critic_b.parameters(), lr=1e-3),
        adapter,
        copy.deepcopy(segs),
        _cfg(epochs=1, tbptt_chunk=5, reference_kl_coef=0.1, cache_reference=True),
        rng=np.random.default_rng(9),
        actor_device="cpu",
    )
    assert out_cached["metrics"]["reference_kl"] > 0.0
    assert out_cached["metrics"]["reference_kl"] == pytest.approx(out_inline["metrics"]["reference_kl"], rel=1e-3, abs=1e-4)
    assert out_cached["metrics"]["policy_loss"] == pytest.approx(out_inline["metrics"]["policy_loss"], rel=1e-3, abs=1e-4)
    for pa, pb in zip(actor_a.parameters(), actor_b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=1.5e-3, rtol=1e-2)


def test_amp_overflow_skips_mark_weights_updated(monkeypatch):
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=81)
    seg = make_synthetic_segment(schema, adapter, actor, length=6, seed=50)
    evaluate_critic_segments(critic, adapter, [seg], device="cpu", precision="fp32")

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    actor_opt.amp_scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    critic_opt.amp_scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    gen_a0, gen_c0 = actor.memory_generation, critic.memory_generation
    weights_a0 = [p.detach().clone() for p in actor.parameters()]

    monkeypatch.setattr("cr_rl.training.state_ppo._optimizer_step", lambda *args, **kwargs: None)
    out = optimize_state_separate(
        actor,
        critic,
        behavior,
        actor_opt,
        critic_opt,
        adapter,
        [seg],
        _cfg(epochs=1, critic_epochs=1),
        rng=np.random.default_rng(11),
        actor_device="cpu",
    )
    assert out["actor_steps"] == 0 and out["critic_steps"] == 0
    assert out["actor_overflow_steps"] > 0 and out["critic_overflow_steps"] > 0
    assert actor.memory_generation == gen_a0
    assert critic.memory_generation == gen_c0
    for before, after in zip(weights_a0, actor.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_episode_trace_cross_update_burn_in_and_truncation_bootstrap():
    schema, adapter, actor, critic, _, _ = make_state_models(seed=91)
    critic.eval()
    rng = np.random.default_rng(77)
    raw = np.zeros((2, 9, schema.seat_ints), dtype=np.int32)
    raw[..., 0] = 2
    raw[..., 1] = rng.integers(10000, 90000, size=(2, 9))
    raw[..., 2:6] = rng.integers(1, 20, size=(2, 9, 4))

    trace = EpisodeTrace(episode_id=101, seed=77)
    for t in range(8):
        trace.append(raw[:, t], np.ones((2, 2), dtype=bool), np.zeros(2, dtype=np.float32))
    prefix_view = np.stack(trace.observations[:4])[:, 0]
    suffix_view = np.stack(trace.observations[4:8])[:, 0]
    tail_raw = raw[0, 8]

    # Full continuous forward [0:9] for reference values.
    full_obs = raw[0, :9][None]
    fh, fe, fm = adapter(full_obs)
    with torch.inference_mode():
        full_vals, _ = critic.forward_state(torch.as_tensor(fh), torch.as_tensor(fe), torch.as_tensor(fm))
        expected_suffix_vals = full_vals[0, 4:8].numpy()
        expected_tail_val = float(full_vals[0, 8])

    cont_seg = make_synthetic_segment(schema, adapter, actor, length=4, seed=60)
    cont_seg["prefix_raw"] = prefix_view.copy()
    cont_seg["raw_obs"] = suffix_view.copy()
    cont_seg["bootstrap_raw"] = tail_raw.copy()
    cont_seg["terminated"][:] = False
    cont_seg["truncated"][:] = False
    cont_seg["truncated"][-1] = True
    cont_seg["dones"][-1] = True

    evaluate_critic_segments(critic, adapter, [cont_seg], device="cpu", precision="fp32", chunk_size=2)
    np.testing.assert_allclose(cont_seg["old_values"], expected_suffix_vals, atol=1e-5)
    assert cont_seg["bootstrap_value"] == pytest.approx(expected_tail_val, abs=1e-5)

    # Faulted segment stays excluded by prepare_state_rollout_targets.
    cont_seg["valid"][:] = False
    info = prepare_state_rollout_targets([cont_seg], gamma=0.99, gae_lambda=0.95)
    assert info["actor_eligible_count"] == 0
    assert info["critic_eligible_count"] == 0


def test_v3_cli_validation_and_continue_script_support(tmp_path):
    dummy_bc = tmp_path / "dummy_bc.pt"
    _, _, actor, _, _, _ = make_state_models(seed=101)
    torch.save(
        {
            "format": "crrl-state-bc-weights-v1",
            "model_architecture_version": 3,
            "decision_ticks": 8,
            "config": actor.cfg.__dict__,
            "temporal_config": actor.temporal_cfg.__dict__,
            "model": actor.state_dict(),
        },
        dummy_bc,
    )

    from scripts.continue_ppo_training import continuation_command

    resumed_cmd = continuation_command(
        [
            "python",
            "scripts/train_selfplay.py",
            "--v3",
            "--resume",
            str(tmp_path / "old" / "ppo_latest.pt"),
            "--updates",
            "8",
            "--batch-size",
            "8",
            "--actor-batch-size",
            "8",
            "--out",
            str(tmp_path / "old"),
            "--backup-dir",
            str(tmp_path / "bak"),
            "--progress-jsonl",
            str(tmp_path / "p.jsonl"),
        ],
        tmp_path / "new_out",
        tmp_path / "new_bak",
        4,
        actor_effective_batch_size=64,
        gae_lambda=0.985,
        critic_gae_lambda=0.992,
    )
    assert "--v3" in resumed_cmd and "--critic-gae-lambda" in resumed_cmd

    # Verify scripts/train_selfplay.py accepts full v3 options cleanly via parser (stop before run()).
    probe = (
        "import sys; sys.path.insert(0, 'scripts'); import train_selfplay as ts; "
        "seen = {}; "
        "sys.modules['cr_rl.training.state_selfplay'] = type('M', (), {'run': staticmethod(lambda a: seen.update(vars(a)))})(); "
        "sys.argv = ['train_selfplay.py', '--v3', '--device', 'cpu', '--init-from-bc', "
        f"'{dummy_bc}', '--out', '{tmp_path / 'out'}', '--envs', '2', '--steps', '8', '--updates', '1', "
        "'--batch-size', '2', '--actor-batch-size', '2', '--critic-batch-size', '4', "
        "'--actor-effective-batch-size', '16', '--critic-effective-batch-size', '32', "
        "'--actor-choice-only', '--card-entropy-coef', '0.001', '--tile-entropy-coef', '0.001', "
        "'--adaptive-actor-lr', '--target-card-kl', '0.015', '--target-tile-kl', '0.06', "
        "'--critic-gae-lambda', '0.992', '--min-training-games', '4', '--role-league', "
        "'--critic-peer-device', 'cpu', '--opponent-device', 'cpu']; "
        "ts.main(); "
        "assert seen['actor_choice_only'] is True; "
        "assert seen['current_opponent_fraction'] == 0.30; "
        "assert seen['historical_opponent_fraction'] == 0.50; "
        "assert seen['anchor_opponent_fraction'] == 0.20"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    # v3 allows negative tile_entropy_coef + explicit --allow-reward-migration + custom role mixture;
    # v2 still rejects negative tile_entropy_coef.
    probe_v3_neg = (
        "import sys; sys.path.insert(0, 'scripts'); import train_selfplay as ts; "
        "seen = {}; "
        "sys.modules['cr_rl.training.state_selfplay'] = type('M', (), {'run': staticmethod(lambda a: seen.update(vars(a)))})(); "
        "sys.argv = ['train_selfplay.py', '--v3', '--device', 'cpu', '--init-from-bc', "
        f"'{dummy_bc}', '--out', '{tmp_path / 'out_neg'}', '--envs', '2', '--steps', '8', '--updates', '1', "
        "'--card-entropy-coef', '0.0003', '--tile-entropy-coef', '-0.0003', '--anchor-kl-coef', '0.005', '--allow-reward-migration', "
        "'--role-league', '--current-opponent-fraction', '0.25', "
        "'--historical-opponent-fraction', '0.30', '--anchor-opponent-fraction', '0.45']; "
        "ts.main(); "
        "assert seen['tile_entropy_coef'] == -0.0003; "
        "assert seen['anchor_kl_coef'] == 0.005; "
        "assert seen['allow_reward_migration'] is True; "
        "assert seen['anchor_opponent_fraction'] == 0.45"
    )
    proc_neg = subprocess.run([sys.executable, "-c", probe_v3_neg], capture_output=True, text=True, check=False)
    assert proc_neg.returncode == 0, proc_neg.stderr + proc_neg.stdout

    probe_v2_neg = (
        "import sys; sys.path.insert(0, 'scripts'); import train_selfplay as ts; "
        "sys.argv = ['train_selfplay.py', '--v2', '--device', 'cpu', '--init-from-bc', "
        f"'{dummy_bc}', '--out', '{tmp_path / 'out_v2'}', '--card-entropy-coef', '0.001', '--tile-entropy-coef', '-0.001']; "
        "ts.main()"
    )
    proc_v2_neg = subprocess.run([sys.executable, "-c", probe_v2_neg], capture_output=True, text=True, check=False)
    assert proc_v2_neg.returncode != 0

    probe_anc_neg = (
        "import sys; sys.path.insert(0, 'scripts'); import train_selfplay as ts; "
        "sys.argv = ['train_selfplay.py', '--v3', '--device', 'cpu', '--init-from-bc', "
        f"'{dummy_bc}', '--out', '{tmp_path / 'out_anc_neg'}', '--anchor-kl-coef', '-0.01']; "
        "ts.main()"
    )
    proc_anc_neg = subprocess.run([sys.executable, "-c", probe_anc_neg], capture_output=True, text=True, check=False)
    assert proc_anc_neg.returncode != 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 CUDA GPUs")
def test_gpu_concurrent_critic_and_peer_critic_sharded_parity():
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=111)
    actor_dev = torch.device("cuda:0")
    critic_dev = torch.device("cuda:1")
    peer_dev = torch.device("cuda:2" if torch.cuda.device_count() >= 3 else "cuda:0")

    segs = [make_synthetic_segment(schema, adapter, actor, length=10, seed=70 + i, prefix_len=2) for i in range(4)]

    # Reference single-GPU evaluation.
    critic_ref = copy.deepcopy(critic).to(critic_dev)
    ref_segs = copy.deepcopy(segs)
    evaluate_critic_segments(
        critic_ref, adapter, ref_segs, device=critic_dev, precision="fp16", batch_size=2, chunk_size=4
    )

    actor_gpu = actor.to(actor_dev)
    behavior_gpu = behavior.to(actor_dev)
    critic_gpu = copy.deepcopy(critic).to(critic_dev)
    peer_critic = copy.deepcopy(critic).to(peer_dev)

    evaluate_critic_segments(
        critic_gpu,
        adapter,
        segs,
        device=critic_dev,
        precision="fp16",
        batch_size=2,
        chunk_size=4,
        peer_critic=peer_critic,
        peer_device=peer_dev,
    )
    for r, s in zip(ref_segs, segs, strict=True):
        np.testing.assert_allclose(r["old_values"], s["old_values"], atol=5e-3, rtol=5e-3)

    actor_opt = torch.optim.AdamW(actor_gpu.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic_gpu.parameters(), lr=1e-4)
    configure_optimizer_amp(actor_opt, actor_dev)
    configure_optimizer_amp(critic_opt, critic_dev)
    gen_c0 = critic_gpu.memory_generation

    out = optimize_state_separate(
        actor_gpu,
        critic_gpu,
        behavior_gpu,
        actor_opt,
        critic_opt,
        adapter,
        segs,
        _cfg(
            precision="fp16",
            epochs=1,
            critic_epochs=2,
            tbptt_chunk=5,
            actor_batch_size=2,
            critic_batch_size=4,
            actor_effective_batch_size=8,
            critic_effective_batch_size=16,
            cache_reference=True,
        ),
        rng=np.random.default_rng(19),
        actor_device=actor_dev,
        critic_device=critic_dev,
        peer_critic=peer_critic,
        peer_device=peer_dev,
    )
    assert out["actor_steps"] > 0
    assert out["critic_steps"] > 0
    assert critic_gpu.memory_generation > gen_c0
    assert np.isfinite(out["metrics"]["loss"])
    assert np.isfinite(out["metrics"]["value_loss"])
    for p_main, p_peer in zip(critic_gpu.parameters(), peer_critic.parameters(), strict=True):
        torch.testing.assert_close(p_main.cpu(), p_peer.cpu(), atol=0, rtol=0)


def test_streaming_torch_save_cpu_tagged_roundtrip(tmp_path):
    target_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    linear = torch.nn.Linear(16, 8).to(target_dev)
    opt = torch.optim.AdamW(linear.parameters(), lr=1e-3, fused=target_dev.type == "cuda")
    linear(torch.randn(2, 16, device=target_dev)).sum().backward()
    opt.step()

    sd = linear.state_dict()
    opt_sd = opt.state_dict()
    ckpt_path = tmp_path / "streamed_ckpt.pt"
    streaming_torch_save(
        {
            "format": "crrl-state-ppo-v1",
            "model": sd,
            "actor": sd,
            "optimizer": opt_sd,
            "scalar": torch.tensor(7),
        },
        ckpt_path,
        fsync=False,
    )

    loaded_default = torch.load(ckpt_path, weights_only=False)
    loaded_mmap = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    loaded_safe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert loaded_default["model"]["weight"].device.type == "cpu"
    assert loaded_default["optimizer"]["state"][0]["exp_avg"].device.type == "cpu"
    torch.testing.assert_close(loaded_default["model"]["weight"], linear.weight.detach().cpu(), atol=0, rtol=0)
    torch.testing.assert_close(loaded_mmap["actor"]["bias"], linear.bias.detach().cpu(), atol=0, rtol=0)
    torch.testing.assert_close(
        loaded_safe["optimizer"]["state"][0]["exp_avg_sq"],
        opt.state[linear.weight]["exp_avg_sq"].detach().cpu(),
        atol=0,
        rtol=0,
    )
    assert int(loaded_safe["scalar"].item()) == 7


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 CUDA GPUs")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("cache_reference", [True, False])
def test_gpu_peer_actor_sharded_handoff_parity(precision, cache_reference, monkeypatch):
    monkeypatch.setenv("CRRL_AMP_DTYPE", precision)
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=131)
    actor_dev = torch.device("cuda:0")
    critic_dev = torch.device("cuda:1")

    segs_base = [
        make_synthetic_segment(schema, adapter, actor, length=11, seed=90 + i, prefix_len=2)
        for i in range(6)
    ]
    evaluate_critic_segments(critic, adapter, segs_base, device=torch.device("cpu"), precision="fp32")

    # Baseline single-GPU actor + separate critic on cuda:1 (no peer_actor).
    actor_ref = copy.deepcopy(actor).to(actor_dev)
    behavior_ref = copy.deepcopy(behavior).to(actor_dev).eval().requires_grad_(False)
    critic_ref = copy.deepcopy(critic).to(critic_dev)
    opt_a_ref = torch.optim.AdamW(actor_ref.parameters(), lr=1e-4)
    opt_c_ref = torch.optim.AdamW(critic_ref.parameters(), lr=1e-4)
    configure_optimizer_amp(opt_a_ref, actor_dev)
    configure_optimizer_amp(opt_c_ref, critic_dev)
    segs_ref = copy.deepcopy(segs_base)
    cfg = _cfg(
        precision=precision,
        epochs=2,
        critic_epochs=1,
        tbptt_chunk=4,
        actor_batch_size=4,
        critic_batch_size=4,
        actor_effective_batch_size=12,
        critic_effective_batch_size=16,
        reference_kl_coef=0.05,
        cache_reference=cache_reference,
    )
    out_ref = optimize_state_separate(
        actor_ref,
        critic_ref,
        behavior_ref,
        opt_a_ref,
        opt_c_ref,
        adapter,
        segs_ref,
        cfg,
        rng=np.random.default_rng(77),
        actor_device=actor_dev,
        critic_device=critic_dev,
    )

    # Sharded PeerActor on cuda:1 activated via critic_done_event handoff.
    actor_sharded = copy.deepcopy(actor).to(actor_dev)
    peer_actor = copy.deepcopy(actor).to(critic_dev)
    behavior_sharded = copy.deepcopy(behavior).to(actor_dev).eval().requires_grad_(False)
    peer_behavior = copy.deepcopy(behavior).to(critic_dev).eval().requires_grad_(False)
    critic_sharded = copy.deepcopy(critic).to(critic_dev)
    opt_a_sharded = torch.optim.AdamW(actor_sharded.parameters(), lr=1e-4)
    opt_c_sharded = torch.optim.AdamW(critic_sharded.parameters(), lr=1e-4)
    configure_optimizer_amp(opt_a_sharded, actor_dev)
    configure_optimizer_amp(opt_c_sharded, critic_dev)
    segs_sharded = copy.deepcopy(segs_base)
    out_sharded = optimize_state_separate(
        actor_sharded,
        critic_sharded,
        behavior_sharded,
        opt_a_sharded,
        opt_c_sharded,
        adapter,
        segs_sharded,
        cfg,
        rng=np.random.default_rng(77),
        actor_device=actor_dev,
        critic_device=critic_dev,
        peer_actor=peer_actor,
        peer_actor_device=critic_dev,
        peer_behavior=peer_behavior,
    )

    assert out_sharded["actor_steps"] == out_ref["actor_steps"] > 0
    assert out_sharded["actor_samples"] == out_ref["actor_samples"] > 0
    metric_tol = 1e-6 if precision == "fp32" else 2e-3
    grad_tol = 1e-5 if precision == "fp32" else 5e-2
    assert abs(out_sharded["metrics"]["policy_loss"] - out_ref["metrics"]["policy_loss"]) < metric_tol
    assert abs(out_sharded["metrics"]["entropy"] - out_ref["metrics"]["entropy"]) < metric_tol
    assert abs(out_sharded["metrics"]["reference_kl"] - out_ref["metrics"]["reference_kl"]) < metric_tol
    assert abs(out_sharded["metrics"]["actor_grad_norm"] - out_ref["metrics"]["actor_grad_norm"]) < grad_tol
    for p_ref, p_sharded in zip(actor_ref.parameters(), actor_sharded.parameters(), strict=True):
        # Shift-invariant softmax/LayerNorm biases have ~0 analytic grad;
        # AdamW amplifies tiny rounding by ~lr.
        atol = (
            (1e-5 if p_ref.ndim >= 2 else 5e-4)
            if precision == "fp32"
            else (5e-4 if p_ref.ndim >= 2 else 1e-3)
        )
        rtol = 1e-4 if precision == "fp32" else 1e-3
        torch.testing.assert_close(p_ref, p_sharded, atol=atol, rtol=rtol)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 CUDA GPUs")
def test_gpu_peer_actor_mid_batch_handoff_parity():
    import threading
    from cr_rl.training.state_ppo import _run_actor_optimization, prepare_state_rollout_targets

    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=149)
    actor_dev = torch.device("cuda:0")
    peer_dev = torch.device("cuda:1")
    segs_base = [
        make_synthetic_segment(schema, adapter, actor, length=12, seed=110 + i, prefix_len=2)
        for i in range(4)
    ]
    evaluate_critic_segments(critic, adapter, segs_base, device=torch.device("cpu"), precision="fp32")

    cfg = _cfg(
        precision="fp16",
        epochs=1,
        tbptt_chunk=3,
        actor_batch_size=4,
        actor_effective_batch_size=4,
        reference_kl_coef=0.04,
        cache_reference=True,
    )

    # Single-GPU reference
    segs_ref = copy.deepcopy(segs_base)
    prepare_state_rollout_targets(segs_ref, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    actor_ref = copy.deepcopy(actor).to(actor_dev)
    behavior_ref = copy.deepcopy(behavior).to(actor_dev).eval().requires_grad_(False)
    opt_ref = torch.optim.AdamW(actor_ref.parameters(), lr=1e-4)
    configure_optimizer_amp(opt_ref, actor_dev)
    out_ref = _run_actor_optimization(
        actor_ref,
        behavior_ref,
        opt_ref,
        adapter,
        segs_ref,
        cfg,
        device=actor_dev,
        rng=np.random.default_rng(55),
    )

    # Mid-batch event flip: trigger peer_ready_event right after Group 0 runs unsharded!
    segs_sh = copy.deepcopy(segs_base)
    prepare_state_rollout_targets(segs_sh, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    actor_sh = copy.deepcopy(actor).to(actor_dev)
    peer_actor = copy.deepcopy(actor).to(peer_dev)
    behavior_sh = copy.deepcopy(behavior).to(actor_dev).eval().requires_grad_(False)
    opt_sh = torch.optim.AdamW(actor_sh.parameters(), lr=1e-4)
    configure_optimizer_amp(opt_sh, actor_dev)
    ready_evt = threading.Event()
    calls = {"n": 0}

    class FlipOnStepController:
        def observe(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                ready_evt.set()

        def finish_update(self, *args, **kwargs):
            return None

    out_sh = _run_actor_optimization(
        actor_sh,
        behavior_sh,
        opt_sh,
        adapter,
        segs_sh,
        cfg,
        device=actor_dev,
        rng=np.random.default_rng(55),
        lr_controller=FlipOnStepController(),
        peer_actor=peer_actor,
        peer_device=peer_dev,
        peer_ready_event=ready_evt,
    )
    assert ready_evt.is_set()
    assert out_sh["actor_steps"] == out_ref["actor_steps"] >= 2
    assert out_sh["actor_samples"] == out_ref["actor_samples"]
    assert abs(out_sh["metrics"]["policy_loss"] - out_ref["metrics"]["policy_loss"]) < 2e-3
    for p_ref, p_sh in zip(actor_ref.parameters(), actor_sh.parameters(), strict=True):
        torch.testing.assert_close(p_ref, p_sh, atol=2e-4, rtol=1e-3)


def test_streaming_torch_load_and_placement_extra_state_finite_guard(tmp_path):
    """Streaming load keeps tiny CPU state, sends weights to GPU, skips _extra_state."""
    import json
    from dataclasses import replace

    from test_representation_v2 import configuration

    from cr_rl.models.learned_placement import PlacementAdapter, definition_for
    from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
    from cr_rl.opponents.bootstrap import seed_history
    from cr_rl.opponents.league import OpponentEntry, OpponentPool
    from cr_rl.training.host_memory import streaming_torch_load, streaming_torch_save
    from cr_rl.training.persistence import file_sha256

    cfg = replace(
        configuration(),
        readout_version=2,
        tile_query_residual=True,
        spatial_fourier=True,
        tile_head_fp32=True,
        vocab_size=16,
    )
    temporal = StateTemporalConfig(
        width=16,
        heads=2,
        kv_heads=1,
        head_dim=8,
        pattern=("kda", "gqa"),
        ffn_multiplier=1.0,
        chunk_size=2,
    )
    model = StateActorCritic(cfg, temporal).eval()
    adapter = PlacementAdapter(model, definition_for(model, "card_prior_cnn_001", cnn_channels=32))
    model.add_module("learned_placement", adapter)
    model.learned_placement_definition = adapter.definition
    assert isinstance(model.state_dict()["learned_placement._extra_state"], dict)

    ckpt_path = tmp_path / "v3_placement_actor.pt"
    large_tensor = torch.randn(256, 128, dtype=torch.float32)  # 128 KiB > 64 KiB threshold
    small_rng = torch.get_rng_state()
    payload = {
        **model.checkpoint_config(),
        "model": model.state_dict(),
        "updates": 4,
        "backend_commit": "abc",
        "large_tensor": large_tensor,
        "small_rng": small_rng,
    }
    streaming_torch_save(payload, ckpt_path)

    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    loaded = streaming_torch_load(ckpt_path, target_device=target)
    assert isinstance(loaded["model"]["learned_placement._extra_state"], dict)
    assert loaded["small_rng"].device.type == "cpu"
    torch.testing.assert_close(loaded["small_rng"], small_rng)
    assert loaded["large_tensor"].device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    torch.testing.assert_close(loaded["large_tensor"].cpu(), large_tensor)

    # Restore StateActorCritic cleanly from streaming_torch_load output
    restored = state_model_from_checkpoint(loaded, target).eval()
    assert restored.checkpoint_config() == model.checkpoint_config()

    # seed_history finite check must skip PlacementAdapter's non-tensor _extra_state dict
    manifest = tmp_path / "history_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "checkpoint": str(ckpt_path),
                        "generation": 4,
                        "sha256": file_sha256(ckpt_path),
                    }
                ]
            }
        )
    )
    pool = OpponentPool([OpponentEntry(0, "anchor.pt", 0, role="anchor")], max_entries=4)
    seed_history(
        pool,
        manifest,
        tmp_path,
        model.cfg.__dict__,
        {"backend_commit": "abc"},
        temporal_config=model.checkpoint_config()["temporal_config"],
    )
    assert pool.entries[-1].role == "historical"
    assert pool.entries[-1].generation == 4


def test_opponent_cache_eviction_and_non_continuing_burn_in(tmp_path):
    """Frozen LRU reload rebinds lane memories and dirty empty lanes advance generation."""
    from collections import OrderedDict

    from cr_rl.opponents.league import OpponentEntry
    from cr_rl.training.state_selfplay import PersistentStateSelfplay

    schema, adapter, actor, _, behavior, _ = make_state_models(seed=29)
    actor.eval()
    device = torch.device("cpu")

    def _save_frozen(path, model):
        streaming_torch_save(
            {
                **model.checkpoint_config(),
                "model": model.state_dict(),
                "updates": 1,
                "backend_commit": "abc",
            },
            path,
        )

    ckpt_a = tmp_path / "opp_a.pt"
    ckpt_b = tmp_path / "opp_b.pt"
    _save_frozen(ckpt_a, copy.deepcopy(actor).eval())
    other = copy.deepcopy(actor).eval()
    with torch.no_grad():
        for p in other.parameters():
            p.mul_(0.5)
    _save_frozen(ckpt_b, other)

    trainer = PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.args = SimpleNamespace(envs=2, batch_size=4, actor_batch_size=4, tbptt_chunk=4)
    trainer.schema = schema
    trainer.adapter = adapter
    trainer.actor = actor
    trainer.behavior_model = behavior
    trainer.device = device
    trainer.opponent_device = device
    trainer.precision = "fp32"
    trainer.rollout_opponent_cache_size = 1
    trainer.training_opponent_cache_size = 1
    trainer.opponent_cache = OrderedDict()
    trainer.actor_diverged = False
    trainer.actor_weights_dirty = False
    trainer.learner_seats = np.array([0, 0], dtype=np.int32)

    entry_a = OpponentEntry(1, str(ckpt_a), 1, role="historical")
    entry_b = OpponentEntry(2, str(ckpt_b), 2, role="historical")
    key_a, mdl_a_first = trainer._load_opponent_model(entry_a)

    trainer.opponents = [entry_a, None]
    trainer.lane_opp_keys = [key_a, "current"]
    trainer.lane_actor_mem = actor.initial_state(2, device=device)
    trainer.lane_opp_mem = [
        mdl_a_first.initial_state(1, device=device),
        actor.initial_state(1, device=device),
    ]

    # Step opponent A once before eviction to populate non-zero temporal state.
    raw = np.zeros((1, schema.seat_ints), dtype=np.int32)
    raw[0, 0] = 2
    raw[0, 1] = 30000
    h, e, em = adapter(raw)
    lt = torch.ones((1, actor.cfg.action_count), dtype=torch.bool)
    hd = torch.zeros((1, 4), dtype=torch.long)
    logp_ref_step, _, mem_after_1 = mdl_a_first.step(
        torch.as_tensor(h),
        torch.as_tensor(e),
        torch.as_tensor(em),
        legal_action_mask=lt,
        state=trainer.lane_opp_mem[0],
        hand_card_ids=hd,
        compute_value=False,
    )
    trainer.lane_opp_mem[0] = mem_after_1
    raw[0, 1] = 40000
    h2, e2, em2 = adapter(raw)
    with torch.no_grad():
        logp_ref, _, _ = mdl_a_first.step(
            torch.as_tensor(h2),
            torch.as_tensor(e2),
            torch.as_tensor(em2),
            legal_action_mask=lt,
            state=mem_after_1,
            hand_card_ids=hd,
            compute_value=False,
        )

    # Evict opponent A via loading opponent B (cache size=1), then reload opponent A.
    trainer._load_opponent_model(entry_b)
    assert key_a not in trainer.opponent_cache
    key_a_reload, mdl_a_reloaded = trainer._load_opponent_model(entry_a)
    assert key_a_reload == key_a
    assert id(mdl_a_reloaded) != id(mdl_a_first)

    aligned = trainer._align_frozen_opponent_memories(key_a, mdl_a_reloaded, [0])
    assert aligned.owner == id(mdl_a_reloaded)
    with torch.no_grad():
        logp_reloaded, _, _ = mdl_a_reloaded.step(
            torch.as_tensor(h2),
            torch.as_tensor(e2),
            torch.as_tensor(em2),
            legal_action_mask=lt,
            state=aligned,
            hand_card_ids=hd,
            compute_value=False,
        )
    torch.testing.assert_close(logp_reloaded, logp_ref)

    # Mismatched lane opponent key must still fail loudly instead of silently rebinding.
    trainer.lane_opp_keys[0] = str(ckpt_b.resolve())
    with pytest.raises(ValueError, match="Opponent memory key mismatch"):
        trainer._align_frozen_opponent_memories(key_a, mdl_a_reloaded, [0])

    # Non-continuing lane 0 ("current") after actor weight update must advance memory_generation.
    trainer.opponents = [None, None]
    trainer.lane_opp_keys = ["current", "current"]
    t_empty = EpisodeTrace(1, 0)
    t_cont = EpisodeTrace(2, 1)
    dual_raw = np.zeros((2, schema.seat_ints), dtype=np.int32)
    dual_raw[:, 0] = 2
    dual_raw[:, 1] = 25000
    dual_mask = np.ones((2, actor.cfg.action_count), dtype=bool)
    t_cont.observations.append(dual_raw)
    t_cont.masks.append(dual_mask)
    trainer.traces = [t_empty, t_cont]
    actor.mark_weights_updated()
    trainer.actor_weights_dirty = True

    trainer._burn_in_continuing_lanes()
    assert trainer.lane_actor_mem.generation == actor.memory_generation
    assert trainer.lane_opp_mem[0].generation == actor.memory_generation
    assert trainer.lane_opp_mem[1].generation == actor.memory_generation
    assert not trainer.actor_weights_dirty


def test_behavior_model_sync_and_per_update_kl_and_validation_preservation(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import subprocess
    import cr_rl.training.state_selfplay as ssp_mod
    from cr_rl.training.state_selfplay import (
        OpponentPool,
        PersistentStateSelfplay,
        _clone_actor_to_device,
    )
    from cr_rl.training.ppo import factorized_reverse_kl

    schema, adapter, actor, critic, anchor, _ = make_state_models(seed=77)
    anchor_snapshot = {k: v.clone() for k, v in anchor.state_dict().items()}
    behavior = _clone_actor_to_device(actor, torch.device("cpu"))

    # Shift actor far past target_card_kl relative to Update-0 anchor, then sync behavior_model.
    with torch.no_grad():
        actor.wait_actor.bias.add_(2.0)
    segs = [make_synthetic_segment(schema, adapter, actor, length=10, seed=90 + i) for i in range(4)]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    # Verify actor actually moved > 0.015 card_kl away from Update-0 anchor.
    with torch.no_grad():
        raw_t = segs[0]["raw_obs"]
        h, e, em = adapter(raw_t[None])
        hand_idx = np.asarray(schema.header["hand_card"], dtype=np.int64)
        hd = torch.as_tensor(raw_t[None, :, hand_idx], dtype=torch.long)
        lt = torch.as_tensor(segs[0]["legal_masks"]).unsqueeze(0)
        lp_actor, _, _ = actor.forward_state(
            torch.as_tensor(h),
            torch.as_tensor(e),
            torch.as_tensor(em),
            lt,
            hand_card_ids=hd,
            compute_value=False,
        )
        lp_anchor, _, _ = anchor.forward_state(
            torch.as_tensor(h),
            torch.as_tensor(e),
            torch.as_tensor(em),
            lt,
            hand_card_ids=hd,
            compute_value=False,
        )
        card_kl_vs_anchor, _ = factorized_reverse_kl(
            lp_actor[0][segs[0]["valid"]].float(),
            lp_anchor[0][segs[0]["valid"]].float(),
        )
        assert float(card_kl_vs_anchor.mean()) > 0.015

    trainer = PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.device = torch.device("cpu")
    trainer.opponent_device = torch.device("cpu")
    trainer.actor = actor
    trainer.behavior_model = behavior
    trainer.anchor_model = anchor
    trainer.actor_diverged = True
    trainer._sync_behavior_model()

    assert trainer.actor_diverged is False
    for k, v in anchor_snapshot.items():
        torch.testing.assert_close(trainer.anchor_model.state_dict()[k], v)
    for k, v in actor.state_dict().items():
        torch.testing.assert_close(trainer.behavior_model.state_dict()[k], v)

    # With synced behavior_model (pi_old), target_card_kl=0.015 optimizes cleanly instead of stalling.
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    out = optimize_state_separate(
        actor,
        critic,
        trainer.behavior_model,
        actor_opt,
        critic_opt,
        adapter,
        segs,
        _cfg(epochs=1, actor_batch_size=2, actor_effective_batch_size=4, target_card_kl=0.015, target_tile_kl=0.06),
        rng=np.random.default_rng(7),
        actor_device="cpu",
    )
    assert out["actor_steps"] > 0

    # Validation preserves wins/games/win_rate and emits validation progress event on exit 1 after saving evaluation.json.
    trainer.out = tmp_path / "run"
    trainer.out.mkdir()
    trainer.update = 32
    trainer.rank = 0
    trainer.precision = "fp16"
    trainer.telemetry = None
    trainer.tensorboard = None
    trainer.anchor_path = tmp_path / "anchor.pt"
    trainer.anchor_path.write_bytes(b"anchor")
    trainer.pool = OpponentPool(max_entries=4)
    trainer.args = SimpleNamespace(
        validation_decks=str(tmp_path / "decks.yaml"),
        validation_pairs=3,
        progress_jsonl=str(trainer.out / "progress.jsonl"),
    )

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, check=False):
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "wins": 2,
                    "games": 6,
                    "invalid_games": 0,
                    "candidate_inference_budget_passed": False,
                    "latency": {
                        "candidate_end_to_end": {
                            "p50_ms": 9.1,
                            "p99_ms": 10.0,
                            "max_ms": 676.9,
                            "violations": 2,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ssp_mod.subprocess, "run", fake_run)
    trainer._run_validation(tmp_path / "ckpt.pt")

    summary = json.loads((trainer.out / "validation/update_00000032/summary.json").read_text())
    assert summary["bc"]["wins"] == 2
    assert summary["bc"]["games"] == 6
    assert summary["bc"]["win_rate"] == pytest.approx(2 / 6)
    assert summary["bc"]["budget_passed"] is False
    assert summary["bc"]["p99_ms"] == 10.0
    assert "warning" in summary["bc"]

    events = [json.loads(line) for line in (trainer.out / "progress.jsonl").read_text().splitlines()]
    val_events = [e for e in events if e.get("phase") == "validation"]
    assert len(val_events) == 1 and val_events[0]["update"] == 32
    assert val_events[0]["results"]["bc"]["win_rate"] == pytest.approx(2 / 6)


def test_opponent_pool_defers_admission_and_pruning_when_live_matches_fill_capacity(tmp_path, monkeypatch):
    import json

    import cr_rl.training.state_selfplay as ssp_mod
    from cr_rl.opponents.admission import SnapshotAdmission
    from cr_rl.opponents.league import OpponentEntry, OpponentPool

    trainer = ssp_mod.PersistentStateSelfplay.__new__(ssp_mod.PersistentStateSelfplay)
    trainer.out = tmp_path / "run"
    trainer.out.mkdir()
    trainer.rank = 0
    trainer.args = SimpleNamespace(keep_checkpoints=2,keep_rollouts=2)
    storage_events=[]
    trainer.emit=lambda **event:storage_events.append(event)
    monkeypatch.setattr(ssp_mod.shutil, "disk_usage",lambda _:SimpleNamespace(free=5*1024**3))
    trainer.update = 80
    trainer.samples = 737_280
    trainer.role_league = True
    trainer.actor = SimpleNamespace(state_dict=lambda: {"w": torch.ones(2)})
    saved_ids = []

    def fake_save_opponent(policy_id: int, actor_cpu=None):
        saved_ids.append(policy_id)
        p = trainer.out / "opponents" / f"opponent_{policy_id:06d}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"pt")
        return p

    trainer.save_opponent = fake_save_opponent
    entries = [
        OpponentEntry(0, "opp0.pt", 0, role="anchor"),
        OpponentEntry(1073741827, "opp1.pt", 3, role="historical"),
        OpponentEntry(1073741831, "opp2.pt", 7, role="historical"),
    ]
    trainer.pool = OpponentPool(entries=list(entries), max_entries=3)
    trainer.pool.save(trainer.out / "opponent_pool.json")
    trainer.opponents = [entries[0], entries[1], entries[2], entries[1]]
    trainer.admission = SnapshotAdmission(
        purpose="history", interval_samples=25_000, last_admitted_samples=709_632
    )

    # All 3 pool entries are live across concurrent envs -> defer cleanly without crashing.
    trainer._update_opponent_pool(actor_steps=20)
    assert len(trainer.pool.entries) == 3
    assert saved_ids == []
    assert any(e["phase"]=="storage_headroom" for e in storage_events)
    assert trainer.admission.ready(trainer.samples)

    # Once an env rotates off one historical opponent, admission + pruning succeed cleanly.
    trainer.update = 81
    trainer.samples += 9216
    trainer.opponents = [entries[0], entries[2], None, entries[0]]
    trainer._update_opponent_pool(actor_steps=20)
    assert len(saved_ids) == 1
    assert len(trainer.pool.entries) == 3
    assert {e.policy_id for e in trainer.pool.entries} == {0, entries[2].policy_id, saved_ids[0]}
    assert not trainer.admission.ready(trainer.samples)

    # Tightening max_entries below currently live opponents also skips pruning cleanly until freed.
    trainer.pool.max_entries = 2
    trainer.opponents = [entries[0], entries[2], trainer.pool.entries[-1]]
    trainer._update_opponent_pool(actor_steps=0)
    assert len(trainer.pool.entries) == 3



def test_frozen_opponent_cache_preserves_pinned_and_bounds_active_set(tmp_path):
    """Trimming evicts inactive models first and _reset_match caps active distinct keys to cache."""
    from collections import OrderedDict
    from pathlib import Path
    from types import SimpleNamespace
    from cr_rl.opponents.league import OpponentEntry, OpponentPool
    from cr_rl.opponents.role_sampling import RoleMixture
    from cr_rl.training.state_selfplay import PersistentStateSelfplay, _prepare_frozen_opponent

    schema, adapter, actor, _, _, _ = make_state_models(seed=51)
    trainer = PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.actor = actor
    trainer.opponent_device = torch.device("cpu")
    trainer.precision = "fp32"
    trainer.rollout_opponent_cache_size = 2
    trainer.opponent_cache = OrderedDict()

    entries = [OpponentEntry(0, "anchor.pt", 0, role="anchor")]
    for idx in range(1, 5):
        ckpt = tmp_path / f"opp_{idx}.pt"
        ckpt.write_bytes(b"stub")
        entries.append(OpponentEntry(1 << 30 | idx, str(ckpt), idx, role="historical"))
        trainer.opponent_cache[str(ckpt.resolve())] = copy.deepcopy(actor).eval()

    # Lanes hold opp_2 and opp_3 (out of 4 cached); trimming to 2 must keep both live lanes.
    trainer.opponents = [entries[2], entries[3], entries[0]]
    trainer._trim_opponent_cache(2)
    assert set(trainer.opponent_cache.keys()) == {
        str(Path(entries[2].checkpoint).resolve()),
        str(Path(entries[3].checkpoint).resolve()),
    }

    # On CUDA + fp16, frozen opponent trunk is cast to fp16 while learned_placement stays fp32.
    if torch.cuda.is_available():
        from cr_rl.models.learned_placement import PlacementAdapter, definition_for

        cuda_actor = copy.deepcopy(actor).cuda()
        pa = PlacementAdapter(cuda_actor, definition_for(cuda_actor, "card_prior_cnn_001", cnn_channels=32))
        cuda_actor.add_module("learned_placement", pa.cuda())
        cuda_actor = _prepare_frozen_opponent(
            cuda_actor, device=torch.device("cuda:0"), precision="fp16"
        )
        assert cuda_actor.to_temporal.weight.dtype == torch.float16
        assert next(cuda_actor.learned_placement.parameters()).dtype == torch.float32
        assert all(not p.requires_grad for p in cuda_actor.parameters())

    # _reset_match must not draw a 3rd distinct historical opponent when 2 other lanes already hit cap=2.
    pool = OpponentPool(entries, max_entries=8)
    trainer.pool = pool
    trainer.role_mixture = RoleMixture(current=0.0, historical=1.0, anchor=0.0)
    trainer.role_draws = {"current": 0, "historical": 0, "anchor": 0}
    trainer.rng = np.random.default_rng(123)
    trainer.update = 10
    trainer.args = SimpleNamespace(envs=3)
    trainer.device = torch.device("cpu")
    trainer.schema = schema
    trainer.decks = [[]]
    trainer.forms = [[]]
    trainer.deck_names = ["d0"]
    trainer.next_episode = 1
    trainer.episode_ids = np.zeros(3, dtype=np.int64)
    trainer.seeds = np.zeros(3, dtype=np.int64)
    trainer.learner_seats = np.zeros(3, dtype=np.int32)
    trainer.match_decks = [("d0", "d0")] * 3
    trainer.traces = [None] * 3
    trainer.lane_actor_mem = actor.initial_state(3, device=torch.device("cpu"))
    trainer.lane_opp_keys = [None] * 3
    trainer.lane_opp_mem = [None] * 3
    trainer.anchor_model = actor
    trainer.env = SimpleNamespace(vector=SimpleNamespace(reset_env=lambda *a, **kw: None))
    trainer.opponents = [entries[1], entries[2], entries[3]]  # env 0 resets while 1 & 2 hold opp_2, opp_3

    for _ in range(20):
        trainer._reset_match(0)
        assert trainer.opponents[0].policy_id in {entries[2].policy_id, entries[3].policy_id}


def test_validation_prefers_freer_cuda_and_load_releases_ckpt_before_half(tmp_path, monkeypatch):
    import contextlib
    import weakref
    from collections import OrderedDict

    import cr_rl.training.state_selfplay as ssp_mod
    from cr_rl.opponents.league import OpponentEntry, OpponentPool

    trainer = ssp_mod.PersistentStateSelfplay.__new__(ssp_mod.PersistentStateSelfplay)
    trainer.device = torch.device("cuda:0")
    trainer.critic_device = torch.device("cuda:1")
    trainer.opponent_device = torch.device("cuda:1")

    monkeypatch.setattr(ssp_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        ssp_mod.torch.cuda,
        "device",
        lambda _dev: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        ssp_mod.torch.cuda,
        "mem_get_info",
        lambda dev: (7_000_000_000, 16_000_000_000) if dev.index == 0 else (500_000_000, 16_000_000_000),
    )
    assert trainer._validation_device() == torch.device("cuda:0")

    # Failed zero-game validation must not pollute raw tensorboard with a fake 0.0 win rate.
    tb_calls = []
    trainer.out = tmp_path / "run"
    trainer.out.mkdir()
    trainer.update = 96
    trainer.rank = 0
    trainer.precision = "fp16"
    trainer.telemetry = None
    trainer.anchor_path = tmp_path / "anchor.pt"
    trainer.anchor_path.write_bytes(b"anchor")
    trainer.pool = OpponentPool(max_entries=4)
    trainer.tensorboard = SimpleNamespace(
        add_scalar=lambda tag, val, step: tb_calls.append((tag, val, step)),
        flush=lambda: None,
    )
    trainer.args = SimpleNamespace(
        validation_decks=str(tmp_path / "decks.yaml"),
        validation_pairs=2,
        progress_jsonl=str(trainer.out / "progress.jsonl"),
    )
    seen_devices = []

    def fail_run(cmd, **_kw):
        seen_devices.append(cmd[cmd.index("--device") + 1])
        raise ssp_mod.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ssp_mod.subprocess, "run", fail_run)
    trainer._run_validation(tmp_path / "ckpt.pt")
    assert seen_devices == ["cuda:0"]
    assert tb_calls == []

    # _load_opponent_model must pre-trim inactive entries and drop ckpt before _prepare_frozen_opponent.
    trainer.opponent_device = torch.device("cpu")
    trainer.rollout_opponent_cache_size = 2
    trainer.actor = SimpleNamespace(frame_microbatch=16)
    opp_a = tmp_path / "opp_a.pt"
    opp_b = tmp_path / "opp_b.pt"
    opp_c = tmp_path / "opp_c.pt"
    for p in (opp_a, opp_b, opp_c):
        p.write_bytes(b"stub")
    entry_a = OpponentEntry(1 << 30 | 1, str(opp_a), 1, role="historical")
    entry_b = OpponentEntry(1 << 30 | 2, str(opp_b), 2, role="historical")
    entry_c = OpponentEntry(1 << 30 | 3, str(opp_c), 3, role="historical")
    trainer.opponents = [entry_b]  # entry_b is live; entry_a is inactive and eligible for pre-trim
    trainer.opponent_cache = OrderedDict(
        {
            str(opp_a.resolve()): torch.nn.Linear(2, 2),
            str(opp_b.resolve()): torch.nn.Linear(2, 2),
        }
    )

    class _TrackedDict(dict):
        pass

    ckpt_ref = None
    observed = {}

    def fake_load(_path, target_device=None):
        nonlocal ckpt_ref
        observed["cache_len_before_load"] = len(trainer.opponent_cache)
        observed["keys_before_load"] = set(trainer.opponent_cache.keys())
        payload = _TrackedDict({"model_architecture_version": 3})
        ckpt_ref = weakref.ref(payload)
        return payload

    def fake_from_ckpt(_ckpt, _dev):
        return torch.nn.Linear(2, 2)

    def fake_prepare(model, *, device, precision):
        observed["ckpt_alive_during_prepare"] = ckpt_ref() is not None
        return model

    monkeypatch.setattr(ssp_mod, "streaming_torch_load", fake_load)
    monkeypatch.setattr(ssp_mod, "state_model_from_checkpoint", fake_from_ckpt)
    monkeypatch.setattr(ssp_mod, "_prepare_frozen_opponent", fake_prepare)
    trainer._load_opponent_model(entry_c)

    assert observed["cache_len_before_load"] == 1
    assert observed["keys_before_load"] == {str(opp_b.resolve())}
    assert observed["ckpt_alive_during_prepare"] is False
    assert set(trainer.opponent_cache.keys()) == {str(opp_b.resolve()), str(opp_c.resolve())}


def test_level_profiles_compatible_allows_env_scaling_only():
    from cr_rl.training.state_selfplay import _level_profiles_compatible

    base = {
        "mode": "per_environment_global",
        "card_levels": [11] * 18,
        "tower_levels": [11] * 18,
        "backend_semantics": "one global card/tower level per environment lane; both seats share it",
        "jitter": None,
    }
    scaled = {**base, "card_levels": [11] * 24, "tower_levels": [11] * 24}
    assert _level_profiles_compatible(base, scaled) is True

    mix18 = {**base, "card_levels": [11, 12] * 9, "tower_levels": [11, 12] * 9}
    mix20 = {**base, "card_levels": [11, 12] * 10, "tower_levels": [11, 12] * 10}
    assert _level_profiles_compatible(mix18, mix20) is True

    diff_lvl = {**base, "card_levels": [12] * 24, "tower_levels": [12] * 24}
    assert _level_profiles_compatible(base, diff_lvl) is False
    assert _level_profiles_compatible(base, mix20) is False

    diff_jitter = {**scaled, "jitter": {"probability": 0.1, "amount": 1, "seed": 7}}
    assert _level_profiles_compatible(base, diff_jitter) is False


def test_opponent_load_and_evict_syncs_and_guards_opponent_cuda_device(tmp_path, monkeypatch):
    from collections import OrderedDict
    from contextlib import contextmanager
    import cr_rl.training.state_selfplay as ssp

    events: list[tuple[str, object]] = []
    active_cuda: list[torch.device] = [torch.device("cuda:0")]

    @contextmanager
    def fake_cuda_device(dev):
        d = torch.device(dev)
        events.append(("enter", d, len(active_cuda)))
        active_cuda.append(d)
        try:
            yield
        finally:
            active_cuda.pop()
            events.append(("exit", d))

    monkeypatch.setattr(torch.cuda, "device", fake_cuda_device)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda dev=None: events.append(("sync", None if dev is None else torch.device(dev), active_cuda[-1])),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: events.append(("empty_cache", active_cuda[-1])),
    )

    dummy_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        ssp,
        "streaming_torch_load",
        lambda path, target_device=None: (
            events.append(("load", torch.device(target_device), active_cuda[-1])),
            {"model_architecture_version": 3},
        )[1],
    )
    monkeypatch.setattr(
        ssp,
        "state_model_from_checkpoint",
        lambda ckpt, dev: (
            events.append(("build", torch.device(dev), active_cuda[-1])),
            copy.deepcopy(dummy_model),
        )[1],
    )

    trainer = ssp.PersistentStateSelfplay.__new__(ssp.PersistentStateSelfplay)
    trainer.device = torch.device("cuda:0")
    trainer.opponent_device = torch.device("cuda:1")
    trainer.precision = "fp16"
    trainer.actor = SimpleNamespace(frame_microbatch=16)
    trainer.rollout_opponent_cache_size = 1
    trainer.lane_opp_models = [None]
    trainer.opponent_cache = OrderedDict({"old_key": copy.deepcopy(dummy_model)})

    ckpt_path = tmp_path / "opp_new.pt"
    ckpt_path.write_bytes(b"placeholder")
    entry = SimpleNamespace(role="historical", policy_id=1, checkpoint=str(ckpt_path))
    key, loaded = trainer._load_opponent_model(entry)

    assert key == str(ckpt_path.resolve())
    assert list(trainer.opponent_cache.keys()) == [key]
    assert loaded.training is False
    # All opponent loads, builds, syncs, and cache flushes happen inside cuda:1 guard.
    assert ("load", torch.device("cuda:1"), torch.device("cuda:1")) in events
    assert ("build", torch.device("cuda:1"), torch.device("cuda:1")) in events
    assert all(ev[1] == torch.device("cuda:1") for ev in events if ev[0] == "empty_cache")
    assert all(ev[1] == torch.device("cuda:1") and ev[2] == torch.device("cuda:1") for ev in events if ev[0] == "sync")
    # Every empty_cache is immediately preceded by synchronize on cuda:1.
    for idx, ev in enumerate(events):
        if ev[0] == "empty_cache":
            assert idx > 0 and events[idx - 1][:2] == ("sync", torch.device("cuda:1"))
    assert active_cuda == [torch.device("cuda:0")]


def test_anchor_kl_and_non_resume_critic_warmstart(tmp_path):
    from dataclasses import asdict
    from cr_rl.training.state_selfplay import PersistentStateSelfplay

    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=211)
    anchor = copy.deepcopy(actor)
    anchor_snap = {k: v.clone() for k, v in anchor.state_dict().items()}

    # Diverge actor away from frozen BC anchor while keeping behavior == actor at rollout time.
    with torch.no_grad():
        actor.wait_actor.bias.add_(1.5)
    behavior = copy.deepcopy(actor)

    segs = [make_synthetic_segment(schema, adapter, actor, length=10, seed=300 + i) for i in range(4)]
    evaluate_critic_segments(critic, adapter, segs, device="cpu", precision="fp32")

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    out = optimize_state_separate(
        actor,
        critic,
        behavior,
        actor_opt,
        critic_opt,
        adapter,
        segs,
        _cfg(epochs=1, actor_batch_size=2, actor_effective_batch_size=4, anchor_kl_coef=0.5),
        rng=np.random.default_rng(19),
        actor_device="cpu",
        anchor=anchor,
        anchor_device="cpu",
    )
    assert out["actor_steps"] > 0
    assert out["metrics"]["anchor_kl"] > 0.0
    assert np.isfinite(out["metrics"]["loss"])
    for k, v in anchor_snap.items():
        torch.testing.assert_close(anchor.state_dict()[k], v)

    # Non-resume initialization loads pre-trained critic weights when present in checkpoint.
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(0.25)
    critic_snap = {k: v.clone() for k, v in critic.state_dict().items()}
    ckpt = {
        "model_architecture_version": 3,
        "decision_ticks": 8,
        "config": asdict(anchor.cfg),
        "temporal_config": asdict(anchor.temporal_cfg),
        "model": anchor.state_dict(),
        "critic_config": asdict(critic.cfg),
        "critic_temporal_config": asdict(critic.temporal_cfg),
        "critic": critic.state_dict(),
        "critic_mode": "independent",
    }
    init_path = tmp_path / "warm_critic.pt"
    streaming_torch_save(ckpt, init_path)

    trainer = PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.world_size = 1
    trainer.rank = 0
    trainer.device = torch.device("cpu")
    trainer.critic_device = torch.device("cpu")
    trainer.critic_peer_device = None
    trainer.opponent_device = torch.device("cpu")
    trainer.dual_rollout_device = None
    trainer.precision = "fp32"
    trainer.critic_mode = "independent"
    trainer.init_source = init_path
    trainer.args = SimpleNamespace(
        resume=None,
        init_from_bc=str(init_path),
        bc_anchor=str(init_path),
        learning_rate=1e-4,
        critic_learning_rate=1e-4,
        weight_decay=0.0,
        frame_microbatch=8,
        balanced_critic=False,
        seed=5,
        adaptive_actor_lr=False,
    )
    loaded = torch.load(init_path, map_location="cpu", weights_only=False)
    trainer._init_models_and_optimizers(trainer.args, loaded)
    for k, v in critic_snap.items():
        torch.testing.assert_close(trainer.critic.state_dict()[k], v)
    assert trainer.update == 0
    assert trainer.critic_optimizer.state == {}


def test_grad_scaler_underflow_fix_and_parallel_validation(tmp_path, monkeypatch):
    import contextlib
    import json
    from pathlib import Path

    import cr_rl.training.fp16 as fp16_mod
    import cr_rl.training.state_selfplay as ssp_mod
    from cr_rl.opponents.league import OpponentEntry, OpponentPool
    from cr_rl.training.separate import _optimizer_step

    real_grad_scaler = torch.amp.GradScaler
    monkeypatch.setattr(
        fp16_mod.torch.amp,
        "GradScaler",
        lambda _dev, **kw: real_grad_scaler("cpu", **kw),
    )
    monkeypatch.setenv("CRRL_AMP_DTYPE", "fp16")

    stale_zero = {
        "scale": 0.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 8,
    }
    param_a = torch.nn.Linear(2, 1, bias=False)
    opt_a = torch.optim.AdamW(param_a.parameters(), lr=1e-3)
    configure_optimizer_amp(opt_a, "cuda:0", stale_zero)
    assert opt_a.amp_scaler.get_scale() == 128.0
    assert opt_a.amp_scaler.get_growth_interval() == 100
    assert opt_a.amp_scaler.state_dict()["_growth_tracker"] == 0

    scaler_b = ssp_mod._make_scaler(torch.device("cuda:0"), "fp16", stale_zero)
    assert scaler_b.get_scale() == 128.0
    assert scaler_b.get_growth_interval() == 100

    # Overflow backoff must clamp scale at >= 1.0 and recover cleanly on valid step.
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    opt.amp_scaler = real_grad_scaler("cpu", init_scale=2.0, growth_interval=100)
    scaler = opt.amp_scaler
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    model.weight.grad.fill_(float("inf"))
    assert _optimizer_step(model, opt, 1, 0.5) is None
    assert scaler.get_scale() == 1.0
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    model.weight.grad.fill_(float("inf"))
    assert _optimizer_step(model, opt, 1, 0.5) is None
    assert scaler.get_scale() == 1.0
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    assert _optimizer_step(model, opt, 1, 0.5) == pytest.approx(1.0)
    assert int(opt.state[model.weight]["step"]) == 1

    # Parallel dual-GPU validation dispatches vs_bc and vs_history onto separate devices via Popen.
    trainer = ssp_mod.PersistentStateSelfplay.__new__(ssp_mod.PersistentStateSelfplay)
    trainer.device = torch.device("cuda:0")
    trainer.critic_device = torch.device("cuda:1")
    trainer.opponent_device = torch.device("cuda:1")
    trainer.out = tmp_path / "run_val"
    trainer.out.mkdir()
    trainer.update = 64
    trainer.rank = 0
    trainer.precision = "fp16"
    trainer.telemetry = None
    trainer.tensorboard = None
    trainer.anchor_path = tmp_path / "anchor.pt"
    trainer.anchor_path.write_bytes(b"anchor")
    hist_path = tmp_path / "hist.pt"
    hist_path.write_bytes(b"hist")
    trainer.pool = OpponentPool(max_entries=4)
    trainer.pool.entries.append(OpponentEntry(100, str(hist_path), 10, role="historical"))
    trainer.args = SimpleNamespace(
        validation_decks=str(tmp_path / "decks.yaml"),
        validation_pairs=2,
        progress_jsonl=str(trainer.out / "progress.jsonl"),
    )
    monkeypatch.setattr(ssp_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ssp_mod.torch.cuda, "device", lambda _dev: contextlib.nullcontext())
    monkeypatch.setattr(
        ssp_mod.torch.cuda,
        "mem_get_info",
        lambda dev: (9_000_000_000, 16_000_000_000) if dev.index == 0 else (6_000_000_000, 16_000_000_000),
    )

    events = []
    launched_devices = []
    launched_cmds = []
    eval_wins = 3

    class FakeProc:
        def __init__(self, cmd):
            self.cmd = list(cmd)
            self._wins = eval_wins
            self._done = False
            dev = cmd[cmd.index("--device") + 1]
            launched_devices.append(dev)
            launched_cmds.append(self.cmd)
            events.append(("start", dev))

        def poll(self):
            return 0 if self._done else None

        def kill(self):
            self._done = True

        def wait(self):
            events.append(("wait", self.cmd[self.cmd.index("--device") + 1]))
            out_dir = Path(self.cmd[self.cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation.json").write_text(
                json.dumps({"wins": self._wins, "games": 4, "invalid_games": 0}),
                encoding="utf-8",
            )
            self._done = True
            return 0

    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    _atomic_write(trainer.out / "ppo_latest.pt", b"ppo-v1")
    actor_export = trainer.out / "actor_export.pt"
    _atomic_write(actor_export, b"actor-v1")

    monkeypatch.setattr(
        ssp_mod.subprocess,
        "Popen",
        lambda cmd, cwd=None, stdout=None, stderr=None: FakeProc(cmd),
    )
    trainer._run_validation(actor_export)
    assert set(launched_devices) == {"cuda:0", "cuda:1"}
    assert all(str(actor_export) in cmd for cmd in launched_cmds)
    # Both processes must be spawned before waiting on the first one.
    assert [ev[0] for ev in events[:2]] == ["start", "start"]
    summary = json.loads((trainer.out / "validation/update_00000064/summary.json").read_text())
    assert summary["bc"]["win_rate"] == pytest.approx(0.75)
    assert summary["history"]["win_rate"] == pytest.approx(0.75)
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-v1"
    assert (trainer.out / "actor_best_val.pt").read_bytes() == b"actor-v1"
    best_meta = json.loads((trainer.out / "best_validation.json").read_text())
    assert best_meta["update"] == 64
    assert best_meta["bc_win_rate"] == pytest.approx(0.75)

    # Lower validation win rate must not overwrite best_val snapshots.
    trainer.update = 96
    eval_wins = 2
    _atomic_write(trainer.out / "ppo_latest.pt", b"ppo-v2")
    _atomic_write(actor_export, b"actor-v2")
    trainer._run_validation(actor_export)
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-v1"
    assert (trainer.out / "actor_best_val.pt").read_bytes() == b"actor-v1"
    assert json.loads((trainer.out / "best_validation.json").read_text())["update"] == 64

    # Higher validation win rate updates best_val snapshots.
    trainer.update = 128
    eval_wins = 4
    _atomic_write(trainer.out / "ppo_latest.pt", b"ppo-v3")
    _atomic_write(actor_export, b"actor-v3")
    trainer._run_validation(actor_export)
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-v3"
    assert (trainer.out / "actor_best_val.pt").read_bytes() == b"actor-v3"
    best_meta = json.loads((trainer.out / "best_validation.json").read_text())
    assert best_meta["update"] == 128
    assert best_meta["bc_win_rate"] == pytest.approx(1.0)


def test_training_trajectory_win_rates_tensorboard_and_best_snapshot(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    import json
    import cr_rl.training.state_selfplay as ssp_mod
    from cr_rl.monitoring.tensorboard_bridge import summary_scalars
    from cr_rl.opponents.league import OpponentEntry, OpponentPool

    trainer = ssp_mod.PersistentStateSelfplay.__new__(ssp_mod.PersistentStateSelfplay)
    trainer.out = tmp_path / "run_traj"
    trainer.out.mkdir()
    trainer.update = 16
    trainer.pool = OpponentPool(max_entries=4)
    anc_entry = OpponentEntry(
        0, str(tmp_path / "anc.pt"), 0, role="anchor", games=120, score_sum=72.0, ema_score=0.62
    )
    hist_entry = OpponentEntry(
        100, str(tmp_path / "hist.pt"), 10, role="historical", games=50, score_sum=27.5, ema_score=0.55
    )
    trainer.pool.entries.extend([anc_entry, hist_entry])

    rollout_out = {
        "terminal_games": 10,
        "terminal_wins": 6,
        "role_terminal_stats": {
            "games": {"anchor": 4, "historical": 4, "current": 2},
            "wins": {"anchor": 3, "historical": 2, "current": 1},
            "score_sums": {"anchor": 3.0, "historical": 2.0, "current": 1.0},
        },
    }
    twr = trainer._compute_training_win_rates(rollout_out)
    assert twr["overall_win_rate"] == pytest.approx(0.6)
    assert twr["anchor_win_rate"] == pytest.approx(0.75)
    assert twr["historical_win_rate"] == pytest.approx(0.5)
    assert twr["current_win_rate"] == pytest.approx(0.5)
    assert twr["anchor_pool_games"] == 120
    assert twr["anchor_pool_ema_win_rate"] == pytest.approx(0.62)
    assert twr["anchor_pool_cumulative_win_rate"] == pytest.approx(0.6)
    assert twr["historical_pool_games"] == 50
    assert twr["historical_pool_mean_ema_win_rate"] == pytest.approx(0.55)
    assert twr["historical_pool_cumulative_win_rate"] == pytest.approx(0.55)

    recorded = []
    trainer.tensorboard = SimpleNamespace(
        add_scalar=lambda tag, val, step: recorded.append((tag, float(val), int(step))),
        flush=lambda: None,
    )
    summary = {
        "metrics": {"entropy": 0.9},
        "decision_coverage": {"wait_when_play_legal": 0.8},
        "training_win_rates": twr,
    }
    trainer._write_tensorboard(summary, {"rows": 1, "entropy_sum": 0.9})
    tags = {tag: (val, step) for tag, val, step in recorded}
    assert tags["train_traj/overall_win_rate"] == (pytest.approx(0.6), 16)
    assert tags["train_traj/anchor_pool_ema_win_rate"] == (pytest.approx(0.62), 16)

    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    _atomic_write(trainer.out / "ppo_latest.pt", b"ppo-16")
    _atomic_write(trainer.out / "actor_export.pt", b"actor-16")
    trainer._maybe_save_best_train_traj(twr)
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-16"
    assert (trainer.out / "actor_best_val.pt").read_bytes() == b"actor-16"
    assert json.loads((trainer.out / "best_train_traj.json").read_text())["update"] == 16

    trainer.update = 32
    _atomic_write(trainer.out / "ppo_latest.pt", b"ppo-32")
    _atomic_write(trainer.out / "actor_export.pt", b"actor-32")
    trainer._maybe_save_best_train_traj({**twr, "anchor_pool_ema_win_rate": 0.58})
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-16"
    assert json.loads((trainer.out / "best_train_traj.json").read_text())["update"] == 16

    trainer.update = 48
    trainer._maybe_save_best_train_traj({**twr, "anchor_pool_ema_win_rate": 0.68})
    assert (trainer.out / "ppo_best_val.pt").read_bytes() == b"ppo-32"
    assert json.loads((trainer.out / "best_train_traj.json").read_text())["update"] == 48

    bridged = summary_scalars(
        {
            "update": 16,
            "seconds": 80.0,
            "valid_learner_samples": 9216,
            "completed_training_games": 20,
            "terminal_games": 10,
            "terminal_seat_wins": 6,
            "training_win_rates": twr,
        }
    )
    assert bridged["train_traj/anchor_pool_ema_win_rate"] == pytest.approx(0.62)
    bridged_legacy = summary_scalars(
        {
            "update": 10,
            "seconds": 80.0,
            "valid_learner_samples": 9216,
            "completed_training_games": 20,
            "terminal_games": 10,
            "terminal_seat_wins": 7,
        }
    )
    assert bridged_legacy["train_traj/overall_win_rate"] == pytest.approx(0.7)

    # Verify deck-balanced skill advantage vs BC computation.
    trainer._record_deck_vs_bc("deck_weak", "deck_strong", 1.0)
    trainer._record_deck_vs_bc("deck_strong", "deck_weak", 1.0)
    # Learner won both: deck_weak L=1.0, BC=0.0 (+1.0); deck_strong L=1.0, BC=0.0 (+1.0).
    twr_bal = trainer._compute_training_win_rates(rollout_out)
    assert twr_bal["skill_advantage_vs_bc"] == pytest.approx(1.0)
    assert twr_bal["deck_balanced_vs_bc_win_rate"] == pytest.approx(1.0)
    assert twr_bal["skill_advantage_vs_bc_decks"] == 2


@pytest.mark.parametrize("full_episode", [False, True])
def test_collect_current_both_seats_without_duplicating_frozen_lane(full_episode):
    from types import SimpleNamespace
    from cr_rl.training.state_selfplay import PersistentStateSelfplay
    from cr_rl.training.episode_trace import EpisodeTrace
    from cr_rl.training.rewards import RewardConfig

    schema,adapter,actor,_,_,_ = make_state_models(seed=731)
    schema.header["tick"] = [0]
    trainer=PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.args=SimpleNamespace(envs=2,steps=3,actor_batch_size=2,batch_size=2,
        reference_kl_coef=0.,anchor_kl_coef=0.,tbptt_chunk=128,full_episode_bptt=full_episode)
    trainer.actor=actor;trainer.critic=None;trainer.schema=schema;trainer.adapter=adapter
    trainer.device=torch.device('cpu');trainer.opponent_device=torch.device('cpu')
    trainer.actor_diverged=False;trainer._publisher=None;trainer.training_opponent_cache_size=3
    trainer.behavior_model=actor;trainer.anchor_model=actor
    trainer.opponents=[None,SimpleNamespace(role='anchor')]
    trainer.learner_seats=np.array([0,1]);trainer.episode_ids=np.array([10,11])
    trainer.match_decks=[('a','b'),('c','d')]
    trainer.traces=[EpisodeTrace(10,1),EpisodeTrace(11,2)]
    trainer.lane_opp_mem=[actor.initial_state(1),actor.initial_state(1)]
    trainer.lane_actor_mem=actor.initial_state(2)
    trainer._burn_in_continuing_lanes=lambda: ([np.zeros((0,schema.seat_ints),dtype=np.int32)]*2,[None]*2,[None]*2)
    trainer._trim_opponent_cache=lambda *a,**kw:None
    trainer._evaluate_shared_values=lambda segments:0.
    trainer.game_counter=SimpleNamespace(record=lambda rows:0)
    trainer.rewards=RewardConfig();trainer.gamma=.9995
    raw=np.zeros((2,2,schema.seat_ints),dtype=np.int32)
    masks=np.zeros((2,2,2307),dtype=bool);masks[:,:,0]=True
    env=SimpleNamespace(observations=raw,masks=masks,rewards=np.zeros((2,2)),dones=np.zeros(2,dtype=bool))
    trainer.update=17
    resets=[]
    trainer._reset_match=lambda e:resets.append(e)
    step_count=[0]
    def step(actions):
        step_count[0]+=1
        env.observations[:,:,schema.header['tick'][0]]=step_count[0]*8
        if full_episode:
            env.dones[:]=np.array([step_count[0]>=2,step_count[0]>=3])
    env.step=step
    env.vector=SimpleNamespace(info=lambda i:{'episode_step':step_count[0],'tick':step_count[0]*8,'fault':0,'truncated':True})
    trainer.env=env
    trainer._step_models=lambda *a,**kw:(np.zeros((2,2),dtype=np.int64),np.array([[10.,20.],[30.,40.]]),None)
    result=trainer.collect()
    segs=result['segments']
    assert len(segs)==3
    assert sum(len(s['raw_obs']) for s in segs)==(7 if full_episode else 9)  # 2 primary + 1 current opponent
    by_deck={s['own_deck']:s for s in segs}
    for name,logp in [('a',10),('b',20),('c',40)]:
        length=2 if full_episode and name in ('a','b') else 3
        np.testing.assert_array_equal(by_deck[name]['old_log_probs'],[logp]*length)
        np.testing.assert_array_equal(by_deck[name]['raw_obs'][:,schema.header['tick'][0]],np.arange(length)*8)
    if full_episode:
        assert resets==[0,1]
        assert all(not t.observations for t in trainer.traces)
        assert all(s['full_episode'] and s['behavior_version']==17 for s in segs)
        assert all(s['has_complete_prefix'] and s['h0_actor'] is None for s in segs)
    assert by_deck['b']['opp_deck']=='a'


def test_historical_cohort_drains_without_switching_live_opponents(tmp_path):
    from types import SimpleNamespace
    from pathlib import Path
    from cr_rl.opponents.league import OpponentPool,OpponentEntry
    from cr_rl.training.state_selfplay import PersistentStateSelfplay
    trainer=PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    entries=[OpponentEntry(100+i,str(tmp_path/f'{i}.pt'),i,role='historical') for i in range(5)]
    trainer.pool=OpponentPool(entries)
    trainer.args=SimpleNamespace(historical_cohort_size=2,historical_cohort_updates=2)
    trainer.rng=np.random.default_rng(93);trainer.update=0
    trainer._cohort_update=-1;trainer.active_historical_keys=None;trainer.opponents=[None]*4
    trainer._refresh_historical_cohort()
    assert len(trainer.active_historical_keys)==2
    selected=[e for e in entries if str(Path(e.checkpoint).resolve()) in trainer.active_historical_keys]
    trainer.opponents=selected*2
    before=list(trainer.opponents)
    trainer.update=2
    trainer._refresh_historical_cohort()
    retired=trainer._retiring_historical_key
    assert trainer.opponents==before  # No mid-match opponent replacement.
    assert len(trainer.active_historical_keys)==1
    assert retired not in trainer.active_historical_keys
    trainer.opponents=[e for e in before if str(Path(e.checkpoint).resolve())!=retired]
    trainer._refresh_historical_cohort()
    assert len(trainer.active_historical_keys)==2
    assert retired not in trainer.active_historical_keys
    live={str(Path(e.checkpoint).resolve()) for e in trainer.opponents}
    assert len(live|trainer.active_historical_keys)==2
