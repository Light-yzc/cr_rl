"""Full-game gradients, scheduling and collection provenance regressions."""
import copy
import numpy as np
import torch
import pytest
from cr_rl.models.state_temporal import StateTemporalConfig, StateTemporalCore
from cr_rl.training.state_ppo import _build_microbatch_groups


@pytest.mark.parametrize("nested", [False, True])
def test_checkpointed_full_history_gradient_reaches_before_128_steps(nested):
    torch.manual_seed(12)
    cfg = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
        pattern=('kda', 'gqa'), chunk_size=16, cache_initial_capacity=16)
    reference = StateTemporalCore(cfg).train()
    checked = copy.deepcopy(reference)
    checked.gradient_checkpointing = True
    checked.checkpoint_kda_chunks = nested
    x = torch.randn(1, 161, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    weights = torch.linspace(0.5, 1.5, 16)
    (reference(x)[:, -1].square() * weights).sum().backward()
    (checked(y)[:, -1].square() * weights).sum().backward()
    assert y.grad[:, :16].abs().sum() > 1e-7
    torch.testing.assert_close(y.grad, x.grad, atol=1e-6, rtol=2e-4)
    for a, b in zip(reference.parameters(), checked.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)


def test_full_episode_groups_never_split_graph_for_an_optimizer_step():
    segs = [dict(raw_obs=np.zeros((n, 1)), actor_mask=np.ones(n, bool)) for n in (181, 267, 33)]
    groups = _build_microbatch_groups(segs, mask_key='actor_mask', batch_segments=2,
        tbptt_chunk=0, effective_batch_samples=64, rng=np.random.default_rng(1))
    seen = []
    for group in groups:
        for batch, start, end, effective_max, count in group:
            assert start == 0 and end == effective_max
            assert end == max(len(s['raw_obs']) for s in batch)
            seen.extend(id(s) for s in batch)
    assert sorted(seen) == sorted(id(s) for s in segs)


def test_full_episode_ppo_updates_actor_and_critic_with_all_guards():
    from test_state_ppo_v3 import make_state_models, make_synthetic_segment, _cfg
    from cr_rl.training.state_ppo import evaluate_critic_segments, optimize_state_separate
    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=81)
    actor.gradient_checkpointing = critic.gradient_checkpointing = True
    actor.ppo_actor_fp32 = behavior.ppo_actor_fp32 = True
    segments = [make_synthetic_segment(schema, adapter, actor, length=21, seed=82,
        prefix_len=0, forced_wait_steps=tuple(range(8)))]
    for seg in segments:
        seg.update(full_episode=True, behavior_version=7)
    evaluate_critic_segments(critic, adapter, segments, device='cpu', precision='fp32')
    result = optimize_state_separate(actor, critic, behavior,
        torch.optim.AdamW(actor.parameters(), lr=1e-5),
        torch.optim.AdamW(critic.parameters(), lr=1e-5), adapter, segments,
        _cfg(full_episode_bptt=True, tbptt_chunk=4, epochs=1, critic_epochs=1,
             max_initial_logp_error=1e-4, anchor_kl_coef=0.002, cache_reference=True),
        rng=np.random.default_rng(1), actor_device='cpu', critic_device='cpu', anchor=behavior)
    assert result['actor_steps'] > 0 and result['critic_steps'] > 0
    assert len(result['initial_logp_errors']) >= 2  # frozen rollout pass and pre-step actor pass
    assert max(result['initial_logp_errors']) < 1e-4
    assert all(torch.isfinite(p).all() for model in (actor, critic) for p in model.parameters())


def test_live_segment_variable_ticks_runs_one_ppo_update():
    from test_state_ppo_v3 import make_state_models, make_synthetic_segment, _cfg
    from cr_rl.client.rl_recording import as_ppo_segment
    from cr_rl.training.state_ppo import evaluate_critic_segments, optimize_state_separate

    schema, adapter, actor, critic, behavior, _ = make_state_models(seed=83)
    episode = make_synthetic_segment(schema, adapter, actor, length=12, seed=84, prefix_len=0)
    episode['old_logp'] = episode.pop('old_log_probs')
    episode.update(step_ticks=np.resize([7, 8, 11], 12), behavior_version='smoke', session_id='synthetic')
    episode['rewards'][:] = 0
    episode['rewards'][-1] = -1
    episode['terminated'][-1] = True
    episode['policy_loss_mask'] = np.ones(12, dtype=bool)
    episode['policy_loss_mask'][0] = False
    seg = as_ppo_segment(episode)
    evaluate_critic_segments(critic, adapter, [seg], device='cpu', precision='fp32')
    result = optimize_state_separate(actor, critic, behavior,
        torch.optim.AdamW(actor.parameters(), lr=1e-5),
        torch.optim.AdamW(critic.parameters(), lr=1e-5), adapter, [seg],
        _cfg(full_episode_bptt=True, epochs=1, critic_epochs=1, max_initial_logp_error=1e-4),
        rng=np.random.default_rng(2), actor_device='cpu')
    assert result['actor_steps'] > 0 and result['critic_steps'] > 0
    assert np.isfinite(seg['advantages']).all()
    assert not seg['actor_mask'][0] and seg['critic_mask'][0]
    assert seg['returns'][-1] == pytest.approx(-1)
    print(f"live-format smoke: actor_steps={result['actor_steps']}, critic_steps={result['critic_steps']}")


def test_full_episode_optimizer_rejects_mixed_policy_versions_and_partial_games():
    import pytest
    from cr_rl.training.state_ppo import StatePPOConfig, optimize_state_separate
    base = dict(full_episode=True, has_complete_prefix=True, behavior_version=3,
        h0_actor=None, raw_obs=np.zeros((2, 1)), terminated=np.array([False, True]),
        truncated=np.zeros(2, bool))
    def run(segments):
        optimize_state_separate(None, None, None, None, None, None, segments,
            StatePPOConfig(full_episode_bptt=True), rng=np.random.default_rng(1), actor_device='cpu')
    with pytest.raises(ValueError, match='versions'):
        run([base, dict(base, behavior_version=4)])
    with pytest.raises(ValueError, match='complete'):
        run([dict(base, has_complete_prefix=False)])
    with pytest.raises(ValueError, match='boundary'):
        run([dict(base, terminated=np.zeros(2, bool))])


def test_single_episode_items_can_use_both_gpus_without_splitting_time():
    from cr_rl.training.state_ppo import _shard_group_items
    segments = [dict(raw_obs=np.zeros((160, 1))) for _ in range(4)]
    group = [([s], 0, 160, 160, 160) for s in segments]
    main, peer = _shard_group_items(group, {id(s):160 for s in segments}, can_shard=True)
    assert len(main) == len(peer) == 2
    assert {id(item[0][0]) for item in main}.isdisjoint(id(item[0][0]) for item in peer)
    assert all(start == 0 and end == 160 for _, start, end, _ in main + peer)
