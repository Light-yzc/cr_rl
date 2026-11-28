from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.training.colab_pipeline import ppo_command
from cr_rl.training.live_awbc import amp as awbc_amp
from cr_rl.training.replay_starts import ReplayStartPool, _find_counter_cutpoints
from cr_rl.training.state_ppo import _amp_context as ppo_amp_context, prepare_state_rollout_targets
from cr_rl.training.state_selfplay import _amp_context as sp_amp_context, _finalize_segment, _new_segment


def _make_synthetic_bc_snapshot(root: Path, *, num_train_sparky: int = 2, num_train_general: int = 2):
    """Build a minimal valid dual-seat crrl-bc-shards-v1 snapshot for unit tests."""
    (root / "games").mkdir(parents=True, exist_ok=True)
    vocab = [
        "Giant", "ZapMachine", "Rage", "MiniPekka", "DarkPrince", "Wizard", "MinionHorde", "Arrows",
        "Knight", "Archers", "Fireball", "Zap", "HogRider", "Cannon", "Skeletons", "IceSpirit",
    ]
    # Compact synthetic header layout covering deck(8), deck_slot(8), hand_card(4), queue(4)
    header = {
        "deck": tuple(range(0, 8)),
        "deck_slot": tuple(range(8, 16)),
        "hand_card": tuple(range(16, 20)),
        "queue": tuple(range(20, 24)),
    }
    seat_ints = 24
    schema_obj = SimpleNamespace(header=header, seat_ints=seat_ints)

    replays = []
    shards = []
    assignments = {}

    def _add_game(idx: int, cohort: str, partition: str, pref_seat: int, order0, order1):
        T = 120
        obs = np.zeros((2 * T, seat_ints), dtype=np.int32)
        acts = np.zeros(2 * T, dtype=np.int32)
        ticks = np.zeros(2 * T, dtype=np.int32)
        seats = np.zeros(2 * T, dtype=np.int32)
        flags = np.zeros(2 * T, dtype=np.int32)
        masks = np.zeros((2 * T, 10), dtype=np.uint8)
        masks[:, 0] = 1

        d0 = list(range(0, 8))
        d1 = list(range(8, 16))
        for s, (d_ids, order) in enumerate(((d0, order0), (d1, order1))):
            sl = slice(s * T, (s + 1) * T)
            seats[sl] = s
            ticks[sl] = np.arange(T, dtype=np.int32) * 8
            obs[sl, 0:8] = d_ids
            obs[sl, 8:16] = [1, 0, 0, 0, 0, 0, 0, 0]
            obs[sl, 16:20] = [d_ids[i] for i in order[:4]]
            obs[sl, 20:24] = [d_ids[i] for i in order[4:]]
        # Seat 0 plays at steps 20, 50; Seat 1 plays at steps 38, 70
        acts[20] = 5
        acts[50] = 7
        acts[T + 38] = 6
        acts[T + 70] = 8

        rel = f"games/{idx:06d}.npz"
        np.savez_compressed(
            root / rel,
            observations=obs,
            masks=masks,
            actions=acts,
            ticks=ticks,
            seats=seats,
            flags=flags,
            replay_ids=np.zeros(2 * T, dtype=np.int32),
        )
        ident = f"sig_{idx:04d}"
        replays.append({
            "canonical_game_signature": ident,
            "decks": [[vocab[i] for i in d0], [vocab[i] for i in d1]],
            "cohort": cohort,
        })
        shards.append({
            "path": rel,
            "replay_offset": idx,
            "selected_seats": [pref_seat],
            "cohort": cohort,
        })
        assignments[ident] = partition

    idx = 0
    for k in range(num_train_sparky):
        _add_game(idx, "sparky", "train", 0, [3, 1, 0, 2, 7, 5, 4, 6], [7, 6, 5, 4, 3, 2, 1, 0])
        idx += 1
    for k in range(num_train_general):
        _add_game(idx, "general", "train", 1, [0, 2, 4, 6, 1, 3, 5, 7], [1, 3, 5, 7, 0, 2, 4, 6])
        idx += 1
    # Add a validation game that must NEVER be loaded when train_only=True
    _add_game(idx, "sparky", "validation", 0, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7])

    (root / "dataset_metadata.json").write_text(json.dumps({
        "decision_ticks": 8,
        "schema": {"header": {k: list(v) for k, v in header.items()}, "seat_ints": seat_ints},
        "vocabulary": {"names": vocab},
        "replays": replays,
    }))
    (root / "dataset.manifest.json").write_text(json.dumps({
        "snapshot_id": "test_snap_001",
        "metadata": "dataset_metadata.json",
        "format": "crrl-bc-shards-v1",
        "shards": shards,
        "decision_ticks": 8,
    }))
    (root / "split_registry.json").write_text(json.dumps({
        "snapshot_id": "test_snap_001",
        "assignments": assignments,
    }))
    return schema_obj


def test_replay_start_pool_excludes_validation_and_samples_counter_cuts(tmp_path):
    schema = _make_synthetic_bc_snapshot(tmp_path)
    pool = ReplayStartPool(
        tmp_path,
        schema=schema,
        min_steps=35,
        max_steps=85,
        min_tail_steps=20,
        counter_prob=1.0,
        sparky_prob=0.5,
        train_only=True,
    )
    assert pool.size == 4
    assert all(ep.identity != "sig_0004" for ep in pool.episodes)

    rng = np.random.default_rng(123)
    for _ in range(30):
        plan = pool.sample(rng)
        assert plan.is_counter_cut
        assert 35 <= plan.cut_step <= 85
        assert len(plan.actions0) == plan.cut_step
        assert len(plan.actions1) == plan.cut_step
        assert sorted(plan.hand_order0) == list(range(8))
        assert sorted(plan.hand_order1) == list(range(8))
        # Opponent plays at 38, 70 (for seat 0) or 20, 50 (for seat 1)
        if plan.learner_seat == 0:
            assert plan.cut_step in (39, 40, 41, 71, 72, 73)
        else:
            assert plan.cut_step in (51, 52, 53)


def test_prefix_critic_and_actor_masks_in_gae():
    seg = _new_segment(1, "d0", "d1", prefix_raw=np.zeros((0, 4), dtype=np.int32), h0_actor=None, h0_behavior=None)
    T = 10
    for t in range(T):
        if t < 4:
            seg.setdefault("_actor_excluded", []).append(t)
            seg.setdefault("_critic_excluded", []).append(t)
        seg["_raw_obs"].append(np.zeros(4, dtype=np.int32))
        mask = np.zeros(5, dtype=bool)
        mask[0] = True
        mask[1] = True
        seg["_legal_masks"].append(mask)
        seg["_actions"].append(0)
        seg["_old_log_probs"].append(-0.5)
        seg["_rewards"].append(0.1 if t >= 4 else 0.0)
        seg["_raw_rewards"].append(0.0)
        seg["_reward_hp"].append(0.0)
        seg["_reward_crown"].append(0.0)
        seg["_reward_overflow"].append(0.0)
        seg["_reward_army"].append(0.0)
        seg["_terminated"].append(t == T - 1)
        seg["_truncated"].append(False)
        seg["_valid"].append(True)
    fin = _finalize_segment(seg)
    fin["old_values"] = np.linspace(0.0, 0.9, T, dtype=np.float32)

    stats = prepare_state_rollout_targets([fin], gamma=0.99, gae_lambda=0.95)
    assert fin["actor_mask"][:4].sum() == 0
    assert fin["actor_mask"][4:].all()
    assert fin["critic_mask"][:4].sum() == 0
    assert fin["critic_mask"][4:].all()
    assert stats["actor_eligible_count"] == 6
    assert stats["critic_eligible_count"] == 6


def test_bf16_amp_helpers_and_colab_pipeline_forwarding(tmp_path):
    cpu = torch.device("cpu")
    with ppo_amp_context(cpu, "bf16"), sp_amp_context(cpu, "bf16"), awbc_amp(cpu, "bf16"):
        assert torch.ones(2).sum().item() == 2.0

    prep = tmp_path / "prepared"
    prep.mkdir(parents=True)
    (prep / "source_metadata.json").write_text(json.dumps({"training_args": {}}))
    cfg = {
        "run_name": "test_run",
        "precision": "bf16",
        "ppo_only": True,
        "backup": False,
        "wandb": False,
        "replay_start_dataset": "/content/crrl_sparky_plan_b",
        "replay_start_prob": 0.35,
        "replay_start_max_steps": 560,
        "replay_start_counter_prob": 0.60,
        "replay_start_late_prob": 0.55,
        "replay_start_late_min_steps": 260,
        "validation_history": "/content/crrl_training/prepared/validation_history.pt",
    }
    cmd = ppo_command(tmp_path, cfg)
    assert cmd[cmd.index("--precision") + 1] == "bf16"
    assert cmd[cmd.index("--replay-start-dataset") + 1] == "/content/crrl_sparky_plan_b"
    assert cmd[cmd.index("--replay-start-prob") + 1] == "0.35"
    assert cmd[cmd.index("--replay-start-max-steps") + 1] == "560"
    assert cmd[cmd.index("--replay-start-counter-prob") + 1] == "0.6"
    assert cmd[cmd.index("--replay-start-late-prob") + 1] == "0.55"
    assert cmd[cmd.index("--replay-start-late-min-steps") + 1] == "260"
    assert cmd[cmd.index("--validation-history") + 1] == "/content/crrl_training/prepared/validation_history.pt"


def test_replay_start_pool_late_game_weighted_sampling(tmp_path):
    """Verify max_steps=560 and late_game_prob=0.55 concentrate >55% of cuts in t>=260 even with short replays mixed in."""
    schema = _make_synthetic_bc_snapshot(tmp_path)
    # Overwrite 000000.npz (sparky) and 000002.npz (general) as full 500-step double-elixir/overtime games
    for sh_name in ("000000.npz", "000002.npz"):
        p = tmp_path / "games" / sh_name
        with np.load(p) as old:
            old_obs = old["observations"]
            obs0 = old_obs[0].copy()
            obs1 = old_obs[len(old_obs) // 2].copy()
        T = 500
        obs = np.zeros((2 * T, obs0.shape[0]), dtype=np.int32)
        acts = np.zeros(2 * T, dtype=np.int32)
        ticks = np.zeros(2 * T, dtype=np.int32)
        seats = np.zeros(2 * T, dtype=np.int32)
        for s, row_template in ((0, obs0), (1, obs1)):
            sl = slice(s * T, (s + 1) * T)
            seats[sl] = s
            ticks[sl] = np.arange(T, dtype=np.int32) * 8
            obs[sl] = row_template
        # Add late-game plays in double elixir (t=310, 380) and overtime (t=455)
        for play_t in (40, 180, 310, 380, 455):
            acts[play_t] = 12
            acts[T + play_t] = 25
        np.savez_compressed(p, observations=obs, actions=acts, ticks=ticks, seats=seats)

    pool = ReplayStartPool(
        tmp_path,
        schema=schema,
        min_steps=35,
        max_steps=560,
        min_tail_steps=30,
        counter_prob=0.60,
        sparky_prob=0.60,
        late_game_prob=0.55,
        late_min_steps=260,
        train_only=True,
    )
    summary = pool.summary()
    assert summary["episodes"] == 4
    assert summary["late_episodes"] == 2
    assert summary["max_steps"] == 560
    assert summary["late_game_prob"] == 0.55

    rng = np.random.default_rng(2026)
    plans = [pool.sample(rng) for _ in range(400)]
    late_rate = sum(1 for pl in plans if pl.cut_step >= 260 and pl.is_late_cut) / len(plans)
    overtime_or_deep_2x = sum(1 for pl in plans if pl.cut_step >= 310)
    assert late_rate >= 0.55, f"Expected >=55% late cuts (t>=260), got {late_rate:.3f}"
    assert overtime_or_deep_2x > 80
    assert max(pl.cut_step for pl in plans) <= 470  # T=500 - min_tail_steps=30


def test_crforge_set_hand_order_exact_opening():
    tdir = Path.home() / ".crforge/tables/16.635.3"
    if not tdir.is_dir():
        return
    from cr_rl.envs.crforge import CrForgeVector

    env = CrForgeVector(count=1, threads=1, independent=True, decision_ticks=8)
    schema = env.vector.schema
    vocab = env.vector.metadata["vocabulary"]["names"]
    d0 = ["Giant", "ZapMachine", "Rage", "MiniPekka", "DarkPrince", "Wizard", "MinionHorde", "Arrows"]
    d1 = ["Knight", "Archer", "Fireball", "Zap", "HogRider", "Cannon", "Skeletons", "IceSpirits"]
    order0 = [5, 2, 7, 0, 4, 1, 6, 3]
    order1 = [7, 6, 5, 4, 3, 2, 1, 0]
    env.vector.reset_env(0, 42, d0, d1, [0] * 8, [0] * 8, hand_order=(order0, order1))

    obs0 = env.observations[0, 0]
    obs1 = env.observations[0, 1]
    hand0 = [vocab[int(i)] for i in obs0[list(schema.header["hand_card"])]]
    queue0 = [vocab[int(i)] for i in obs0[list(schema.header["queue"])]]
    hand1 = [vocab[int(i)] for i in obs1[list(schema.header["hand_card"])]]
    queue1 = [vocab[int(i)] for i in obs1[list(schema.header["queue"])]]
    assert hand0 == [d0[i] for i in order0[:4]]
    assert queue0 == [d0[i] for i in order0[4:]]
    assert hand1 == [d1[i] for i in order1[:4]]
    assert queue1 == [d1[i] for i in order1[4:]]
    env.close()
