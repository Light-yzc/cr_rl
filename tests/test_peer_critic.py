import copy

import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.fp16 import configure_optimizer_amp
from cr_rl.training.peer_critic import PeerCritic
from cr_rl.training.separate import _optimizer_step, bounded_microbatches


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("peer_microbatch", [1, 8])
def test_peer_preserves_weighted_tail_updates_and_dynamic_handoff(
    monkeypatch, precision, peer_microbatch
):
    monkeypatch.setenv("CRRL_AMP_DTYPE", precision)
    torch.manual_seed(23)
    _, _, source, inputs = setup_models()
    inputs = tuple(t.repeat((3,) + (1,) * (t.ndim - 1))[:9].to("cuda:0") for t in inputs)
    targets = torch.linspace(-0.7, 0.9, 9, device="cuda:1")
    baseline = copy.deepcopy(source).to("cuda:1")
    primary = source.to("cuda:1")
    reference_opt = torch.optim.AdamW(baseline.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(primary.parameters(), lr=1e-4)
    configure_optimizer_amp(reference_opt, "cuda:1")
    configure_optimizer_amp(optimizer, "cuda:1")
    peer = PeerCritic(primary, "cuda:0", peer_microbatch=peer_microbatch)

    def make_batch(rows):
        return tuple(t[rows] for t in inputs)

    try:
        for _ in range(2):
            peer.begin_update(True, optimizer)
            # One primary-only group followed by an odd tail split 3 + 2.
            for rows in (np.arange(4), np.arange(4, 9)):
                for chunk in (
                    (rows[:3], *np.array_split(rows[3:], 2 if peer_microbatch == 1 else 1))
                    if peer.ready
                    else (rows,)
                ):
                    if len(chunk):
                        peer.shard(baseline, "cuda:1", make_batch, chunk, targets, reference_opt)
                peer.loss_and_backward(make_batch, rows, targets, optimizer)
                peer.reduce_gradients()
                _optimizer_step(baseline, reference_opt, len(rows), 0.5)
                _optimizer_step(primary, optimizer, len(rows), 0.5)
                peer.primary_updated()
                peer.actor_done.set()
            assert peer.rows == 2
            assert peer.primary_rows == 7
            assert peer.gradient_bytes > 0 and peer.weight_bytes > 0
            for left, right in zip(baseline.parameters(), primary.parameters(), strict=True):
                torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
        peer.begin_update(False, optimizer)
        assert peer.ready
        peer.ensure_synced()
        for left, right in zip(primary.parameters(), peer.replica.parameters(), strict=True):
            torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    finally:
        peer.close()


def test_dynamic_parallel_chunks_keep_order_and_effective_groups():
    ready = [False]
    chunks = bounded_microbatches(
        np.arange(139), 16, 64, lambda _: None, 1024, parallel_factor=lambda: 2 if ready[0] else 1
    )
    result = []
    for rows, _last in chunks:
        result.append(rows)
        ready[0] = True
    assert [len(rows) for rows in result] == [16, 32, 16, 32, 32, 11]
    np.testing.assert_array_equal(np.concatenate(result), np.arange(139))
    assert _last


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("actor_enabled", [False, True])
def test_full_optimizer_handoff_completes(actor_enabled, monkeypatch):
    from cr_rl.training.separate import forward_inputs, optimize_separate

    monkeypatch.setenv("CRRL_AMP_DTYPE", "fp16")
    _, actor, critic, inputs = setup_models()
    actor, critic = actor.to("cuda:0"), critic.to("cuda:1")
    inputs = tuple(t.to("cuda:0") for t in inputs)
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    configure_optimizer_amp(actor_opt, "cuda:0")
    configure_optimizer_amp(critic_opt, "cuda:1")
    peer = PeerCritic(critic, "cuda:0")
    try:
        result = optimize_separate(
            actor,
            critic,
            behavior,
            actor_opt,
            critic_opt,
            lambda rows: tuple(t[rows] for t in inputs),
            np.arange(4),
            np.arange(4),
            old[:, 0],
            torch.zeros(4, dtype=torch.long, device="cuda:0"),
            torch.ones(4, device="cuda:0"),
            torch.ones(4, device="cuda:0"),
            rng=np.random.default_rng(18),
            device="cuda:0",
            critic_device="cuda:1",
            batch_size=2,
            accumulation=2,
            actor_epochs=1,
            critic_epochs=2,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=1,
            max_grad_norm=0.5,
            actor_enabled=actor_enabled,
            critic_peer=peer,
        )
        assert result["critic_steps"] == 2
        assert result["actor_steps"] == int(actor_enabled)
        assert peer.ready
        if not actor_enabled:
            assert result["critic_parallel"]["peer_rows"] == 4
    finally:
        peer.close()


@pytest.mark.parametrize("entities", [48, 64, 96, 128])
def test_peer_bounds_dense_activations_and_preserves_weighted_loss(entities):
    peer = object.__new__(PeerCritic)
    peer.peer_microbatch = 16
    peer.replica, peer.peer_device = None, "cpu"
    sizes = []

    def make_batch(rows):
        return rows

    make_batch.limits = np.full(16, entities)

    def shard(model, device, batch, rows, targets, optimizer):
        sizes.append(rows.copy())
        return targets[rows].mean()

    peer.shard = shard
    targets = torch.arange(16, dtype=torch.float32)
    loss = peer.replica_backward(make_batch, np.arange(16), targets, None)
    assert all(len(rows) <= 16 and len(rows) * entities <= 768 for rows in sizes)
    np.testing.assert_array_equal(np.concatenate(sizes), np.arange(16))
    assert loss == targets.mean()
