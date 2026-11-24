"""Production architecture-v3 recurrent PPO trainer with multi-GPU concurrency.

Preserves 100% of v2's RL methodology while exploiting explicit episode-state
(v3) recurrence and hardware parallelism:
- Continuous cross-update episodes with recurrent burn-in on weight updates
- Independent `StateValueNetwork` critic (`--critic-mode independent`) trained
  concurrently on a separate GPU (and optional NVLink peer via `--balanced-critic`
  or `--critic-peer-device`) with post-rollout batched value evaluation
- Per-lane role-league opponent sampling (`current` / `historical` / `anchor`)
  using `RoleMixture`, `OpponentPool`, `SnapshotAdmission`, `DiversityRetention`
- Per-model batched GPU rollout inference with optional `--opponent-device`
  parallel worker overlap and `--dual-rollout` actor replica splitting
- Choice-only Actor masking, factorized entropy & factorized KL limits,
  adaptive Actor LR controller, reference KL (with optional cache), MC warmup
- Low-RAM checkpointing, optional `TrajectoryBuffer` archival, `TrainingTelemetry`,
  TensorBoard + `metrics.jsonl`/`progress.jsonl`, and async backup publication.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from fractions import Fraction
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.trajectory import TrajectoryBuffer
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models.state_policy import (
    PolicyMemory,
    StateValueNetwork,
    clone_memory,
    rebind_memory,
    reserve_step_memory,
    slice_memory,
    stack_memories,
    state_model_from_checkpoint,
    state_value_model_from_checkpoint,
)
from cr_rl.opponents.admission import SnapshotAdmission
from cr_rl.opponents.bootstrap import seed_history
from cr_rl.opponents.league import OpponentEntry, OpponentPool
from cr_rl.opponents.role_sampling import (
    RoleMixture,
    frozen_policy_version,
    next_history_snapshot_id,
)
from cr_rl.training.adaptive_kl import restore_actor_lr_controller
from cr_rl.training.awbc_playbuffer import PlayBuffer, read_game, run_phase
from cr_rl.training.deck_augmentation import deck_probabilities
from cr_rl.training.episode_trace import EpisodeTrace, on_policy_seats
from cr_rl.training.fp16 import build_grad_scaler
from cr_rl.training.game_budget import TrainingGameCounter
from cr_rl.training.host_memory import (
    cache_frozen_checkpoint,
    detach_optimizer_counters,
    release_host_memory,
    streaming_torch_load,
    streaming_torch_save,
)
from cr_rl.training.persistence import (
    atomic_json,
    copy_verified,
    prune_backup,
    prune_local_opponents,
    publish_update,
)
from cr_rl.training.resume_options import override_resume_rates
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.state_ppo import (
    StatePPOConfig,
    burn_in_memory,
    ensure_cuda_linalg_initialized,
    evaluate_critic_segments,
    optimize_state_separate,
    state_frame_cache,
)
from cr_rl.training.telemetry import TrainingTelemetry


def _resolve_device(requested: str, rank: int, world_size: int) -> torch.device:
    if requested == "auto":
        return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and world_size > 1:
        return torch.device(f"cuda:{rank}")
    return torch.device(requested)


def _auto_peer_device(
    actor_device: torch.device,
    critic_device: torch.device,
) -> torch.device | None:
    """Pick an idle GPU for balanced NVLink critic training if one is available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    candidates = [
        torch.device(f"cuda:{i}")
        for i in range(torch.cuda.device_count())
        if torch.device(f"cuda:{i}") not in {actor_device, critic_device}
    ]
    if not candidates:
        return None
    c_idx = critic_device.index or 0
    for cand in candidates:
        if hasattr(torch.cuda, "can_device_access_peer") and torch.cuda.can_device_access_peer(
            c_idx, cand.index or 0
        ):
            return cand
    return candidates[0]


def _make_scaler(
    device: torch.device,
    precision: str,
    saved_state: dict | None = None,
) -> torch.amp.GradScaler:
    enabled = device.type == "cuda" and precision == "fp16"
    return build_grad_scaler(enabled, saved_state)


def _amp_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def restore_actor_optimizer(optimizer, state, actor):
    """Insert empty gate moments by parameter name, preserving placement moments."""
    groups = state["param_groups"]
    names = [name for name,_ in actor.named_parameters()]
    old_ids = groups[0]["params"]
    gate_count = sum(name.startswith("wait_hand_gate.") for name in names)
    if len(groups) == 1 and len(old_ids) == len(names)-gate_count:
        ids = iter(old_ids)
        next_id = max(old_ids, default=-1)+1
        migrated_ids = []
        for name in names:
            if name.startswith("wait_hand_gate."):
                migrated_ids.append(next_id)
                next_id += 1
            else:
                migrated_ids.append(next(ids))
        state = {**state, "param_groups": [{**groups[0], "params": migrated_ids}]}
    # Catch bad name/order migrations before calling fused AdamW.
    for group, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
        for parameter, key in zip(group["params"], saved["params"], strict=True):
            moments = state["state"].get(key, {})
            for field in ("exp_avg", "exp_avg_sq"):
                if field in moments and moments[field].shape != parameter.shape:
                    raise ValueError(f"Actor optimizer moment shape mismatch for {key}: {field}")
    optimizer.load_state_dict(state)


def _model_state_refs(model) -> dict[str, Any]:
    raw = model.module if hasattr(model, "module") else model
    return raw.state_dict()


def _cuda_device_ctx(device: torch.device | None):
    """Return a CUDA device guard context for CUDA targets, or a no-op context on CPU."""
    return torch.cuda.device(device) if device is not None and device.type == "cuda" else nullcontext()


def _sync_cuda_device(device: torch.device | None) -> None:
    """Synchronize a CUDA device stream if `device` is a CUDA device."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cuda_cache_on(device: torch.device | None) -> None:
    """Synchronize and flush the CUDA caching allocator within `device`'s context."""
    if device is not None and device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _copy_module_weights(
    src: torch.nn.Module,
    dst: torch.nn.Module,
    *,
    sync_device: torch.device | None = None,
) -> None:
    """Copy parameters and buffers from `src` to `dst` in place without extra allocations."""
    src_first = next(src.parameters(), None)
    dst_first = next(dst.parameters(), None)
    src_device = src_first.device if src_first is not None else None
    dst_device = dst_first.device if dst_first is not None else sync_device
    if src_device is not None and src_device != dst_device:
        _sync_cuda_device(src_device)
    with _cuda_device_ctx(dst_device), torch.no_grad():
        for dst_p, src_p in zip(dst.parameters(), src.parameters(), strict=True):
            dst_p.data.copy_(src_p.data, non_blocking=True)
        for dst_b, src_b in zip(dst.buffers(), src.buffers(), strict=True):
            dst_b.data.copy_(src_b.data, non_blocking=True)
    _sync_cuda_device(sync_device or dst_device)


def _clone_actor_to_device(actor, target_device: torch.device, *, trainable: bool = False):
    """Instantiate a replica on `target_device` and copy weights over NVLink/GPU."""
    from cr_rl.models.state_policy import StateActorCritic

    with _cuda_device_ctx(target_device), torch.random.fork_rng(devices=[]), torch.device(target_device):
        clone = StateActorCritic(
            copy.deepcopy(actor.cfg),
            actor.temporal_cfg,
            decision_ticks=8,
            frame_microbatch=actor.frame_microbatch,
        )
        placement_definition = getattr(actor, "learned_placement_definition", None)
        if placement_definition is not None:
            from cr_rl.models.learned_placement import PlacementAdapter

            clone.add_module(
                "learned_placement",
                PlacementAdapter(clone, placement_definition),
            )
            clone.learned_placement_definition = dict(placement_definition)
    clone.ppo_actor_fp32 = getattr(actor, "ppo_actor_fp32", False)
    clone.eval()
    for p in clone.parameters():
        p.requires_grad_(bool(trainable))
    _copy_module_weights(actor, clone, sync_device=target_device)
    return clone


def _prepare_frozen_opponent(
    model: torch.nn.Module,
    *,
    device: torch.device,
    precision: str,
) -> torch.nn.Module:
    """Configure a frozen opponent for inference and halve CUDA trunk VRAM in fp16 runs."""
    with _cuda_device_ctx(device):
        model.ppo_actor_fp32 = False
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if device.type == "cuda" and precision == "fp16":
            model.half()
            # PlacementAdapter._validate requires its small (~11 MiB) decoder parameters to stay FP32.
            if hasattr(model, "learned_placement"):
                model.learned_placement.float()
        _sync_cuda_device(device)
    return model


def _extract_decks(deck_path: Path) -> tuple[dict, list[str], list[list[int]], list[list[int]]]:
    league = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
    decks_map = dict(league["decks"])
    if not decks_map:
        raise ValueError(f"Deck manifest has no decks: {deck_path}")
    names = list(decks_map)
    cards = [list(decks_map[name]) for name in names]
    slots_map = league.get("slots", {})
    slots = [list(slots_map.get(name, [0] * 8)) for name in names]
    return league, names, cards, slots


def _batched_sample(
    logp: torch.Tensor,
    legal: torch.Tensor,
    *,
    collect_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Sample masked legal actions on GPU and return small CPU numpy arrays."""
    probs = torch.nan_to_num(logp.float().exp(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs * legal.float()
    denom = probs.sum(-1, keepdim=True)
    count = legal.sum(-1, keepdim=True).clamp_min(1)
    safe_probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), legal.float() / count)
    actions = torch.multinomial(safe_probs, 1).squeeze(-1)
    chosen_logp = logp.gather(-1, actions[:, None]).squeeze(-1).float()
    stats = None
    if collect_stats:
        with torch.no_grad():
            ent = -(safe_probs * torch.where(legal, logp.float(), 0.0)).sum(-1)
            max_p = safe_probs.max(-1).values
            play_legal = legal[:, 1:2305].any(-1)
            stats = {
                "rows": int(actions.shape[0]),
                "entropy_sum": float(ent.sum()),
                "max_prob_sum": float(max_p.sum()),
                "play_legal_rows": int(play_legal.sum()),
                "entropy_play_legal_sum": float(ent[play_legal].sum()) if play_legal.any() else 0.0,
            }
    return (
        actions.cpu().numpy().astype(np.int64, copy=False),
        chosen_logp.cpu().numpy().astype(np.float32, copy=False),
        stats,
    )


def _compute_decision_coverage(segments: list[dict], schema) -> dict[str, Any]:
    choice_rows = forced_affordable = high_elixir = requests = high_cost = played = play_legal_cnt = 0
    wait_when_play_legal = total_valid = 0
    request_cost = 0.0
    elixir_idx = schema.header["elixir"][0]
    reserved_idx = schema.header["elixir_reserved"][0]
    cost_idx = list(schema.header["hand_cost"])
    card_idx = list(schema.header["hand_card"])
    pending_idx = list(schema.header["hand_pending"])

    for seg in segments:
        valid = seg["valid"].astype(bool)
        if "policy_loss_mask" in seg:
            valid &= seg["policy_loss_mask"]
        total_valid += int(valid.sum())
        if not valid.any():
            continue
        raw_rows = seg["raw_obs"][valid]
        masks = seg["legal_masks"][valid]
        actions = seg["actions"][valid]
        choice = masks.sum(-1) > 1
        choice_rows += int(choice.sum())
        play_legal = masks[:, 1:2305].any(-1)
        play_legal_cnt += int(play_legal.sum())
        wait_when_play_legal += int((play_legal & (actions == 0)).sum())
        played += int((actions != 0).sum())

        available = raw_rows[:, elixir_idx] - raw_rows[:, reserved_idx]
        costs = raw_rows[:, cost_idx]
        cards = raw_rows[:, card_idx]
        pending = raw_rows[:, pending_idx]
        affordable = (cards >= 0) & (pending == 0) & (costs * 10000 <= available[:, None])
        forced_affordable += int((~choice & affordable.any(-1)).sum())
        high_elixir += int((available >= 50000).sum())

        play = (actions > 0) & (actions < 2305)
        if play.any():
            slots = ((actions[play] - 1) // 576).clip(0, 3)
            chosen_cost = costs[play, slots]
            requests += int(play.sum())
            request_cost += float(chosen_cost.sum())
            high_cost += int((chosen_cost >= 5).sum())

    return {
        "choice_rows": choice_rows,
        "forced_wait_with_affordable_card": forced_affordable,
        "available_elixir_at_least_five_rows": high_elixir,
        "available_elixir_ge5_rows": high_elixir,
        "card_requests": requests,
        "requested_card_cost_mean": (request_cost / requests) if requests else None,
        "at_least_five_cost_requests": high_cost,
        "high_cost_requests": high_cost,
        "play_fraction": float(played / total_valid) if total_valid else 0.0,
        "wait_when_play_legal": (
            float(wait_when_play_legal / play_legal_cnt) if play_legal_cnt else None
        ),
        "scope": "On-policy valid learner transitions",
    }


def _new_segment(
    episode_id: int,
    own_deck: str,
    opp_deck: str,
    *,
    prefix_raw: np.ndarray,
    h0_actor: PolicyMemory | None,
    h0_behavior: PolicyMemory | None,
    h0_anchor: PolicyMemory | None = None,
    opponent_role: str | None = None,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "own_deck": own_deck,
        "opp_deck": opp_deck,
        "opponent_role": opponent_role,
        "prefix_raw": prefix_raw,
        "h0_actor": h0_actor,
        "h0_behavior": h0_behavior,
        "h0_anchor": h0_anchor,
        "h0_critic": None,
        "_raw_obs": [],
        "_legal_masks": [],
        "_actions": [],
        "_old_log_probs": [],
        "_rewards": [],
        "_raw_rewards": [],
        "_reward_hp": [],
        "_reward_crown": [],
        "_reward_overflow": [],
        "_reward_army": [],
        "_terminated": [],
        "_truncated": [],
        "_valid": [],
        "bootstrap_raw": None,
        "bootstrap_value": 0.0,
    }


def _finalize_segment(seg: dict[str, Any]) -> dict[str, Any]:
    raw_obs = np.stack(seg.pop("_raw_obs"))
    terminated = np.asarray(seg.pop("_terminated"), dtype=bool)
    truncated = np.asarray(seg.pop("_truncated"), dtype=bool)
    excluded = seg.pop("_actor_excluded", ())
    if excluded:
        mask = np.ones(len(raw_obs), dtype=bool)
        mask[excluded] = False
        seg["policy_loss_mask"] = mask
    return {
        **seg,
        "raw_obs": raw_obs,
        "legal_masks": np.stack(seg.pop("_legal_masks")),
        "actions": np.asarray(seg.pop("_actions"), dtype=np.int64),
        "old_log_probs": np.asarray(seg.pop("_old_log_probs"), dtype=np.float32),
        "rewards": np.asarray(seg.pop("_rewards"), dtype=np.float32),
        "raw_rewards": np.asarray(seg.pop("_raw_rewards"), dtype=np.float32),
        "reward_components": {
            "hp": np.asarray(seg.pop("_reward_hp"), dtype=np.float32),
            "crown": np.asarray(seg.pop("_reward_crown"), dtype=np.float32),
            "overflow": np.asarray(seg.pop("_reward_overflow"), dtype=np.float32),
            "army": np.asarray(seg.pop("_reward_army"), dtype=np.float32),
        },
        "terminated": terminated,
        "truncated": truncated,
        "dones": np.logical_or(terminated, truncated),
        "has_complete_prefix": len(seg.get("prefix_raw", ())) == 0,
        "valid": np.asarray(seg.pop("_valid"), dtype=bool),
    }


def select_awbc_probes(segments, max_probes=4):
    """Bounded, deterministic coverage of usable games/roles/decks; retain every prefix."""
    candidates = []
    for index, segment in enumerate(segments or []):
        if segment.get('has_complete_prefix') is False and not len(segment.get('prefix_raw', ())):
            continue  # Never evaluate a midgame probe with empty recurrent memory.
        legal = segment['legal_masks']
        keep = np.asarray(segment['valid'], dtype=bool) & (legal.sum(-1) > 1)
        if 'policy_loss_mask' in segment:
            keep &= np.asarray(segment['policy_loss_mask'], dtype=bool)
        if not keep.any():
            continue
        actions = np.asarray(segment['actions'])[keep]
        kinds = frozenset(name for name, present in (
            ('wait', (actions == 0).any()),
            ('deploy', ((actions > 0) & (actions < 2305)).any()),
            ('ability', (actions >= 2305).any())) if present)
        candidates.append(dict(index=index, segment=segment, keep=keep,
            game=segment.get('episode_id', ('segment', index)),
            role=segment.get('opponent_role') or 'unknown',
            decks=(segment.get('own_deck'), segment.get('opp_deck')), kinds=kinds))
    chosen = []
    games, roles, decks, kinds = set(), set(), set(), set()
    while candidates and len(chosen) < max_probes:
        best = max(range(len(candidates)), key=lambda i: (
            candidates[i]['game'] not in games,
            candidates[i]['role'] != 'unknown' and candidates[i]['role'] not in roles,
            candidates[i]['decks'] not in decks,
            len(candidates[i]['kinds'] - kinds), int(candidates[i]['keep'].sum()),
            -candidates[i]['index']))
        row = candidates.pop(best)
        chosen.append(row)
        games.add(row['game']); roles.add(row['role']); decks.add(row['decks'])
        kinds.update(row['kinds'])
    probes = []
    for row in chosen:
        segment, keep = row['segment'], row['keep']
        raw, legal = segment['raw_obs'], segment['legal_masks']
        prefix = segment.get('prefix_raw', raw[:0])
        prefix_masks = segment.get('prefix_masks')
        if len(prefix) and prefix_masks is None:
            # Prefix readouts are not labels; legal masks do not alter recurrence.
            prefix_masks = np.zeros((len(prefix), legal.shape[-1]), dtype=bool)
            prefix_masks[:, 0] = True
        probes.append(dict(raw_obs=np.concatenate([prefix, raw]),
            legal_masks=np.concatenate([prefix_masks, legal]) if len(prefix) else legal,
            actions=np.concatenate([np.zeros(len(prefix), dtype=np.int64), segment['actions']]),
            actor_keep=np.concatenate([np.zeros(len(prefix), dtype=bool), keep]),
            probe_info=dict(episode_id=segment.get('episode_id'), opponent_role=row['role'],
                own_deck=row['decks'][0], opp_deck=row['decks'][1],
                choice_steps=int(keep.sum()), prefix_steps=len(prefix))))
    return probes


def _level_profiles_compatible(saved_lp: dict[str, Any], actual_lp: dict[str, Any]) -> bool:
    """Allow changing ``--envs`` across resumes when lane level distribution & metadata match."""
    if saved_lp == actual_lp:
        return True
    lane_keys = ("card_levels", "tower_levels")
    if {k: v for k, v in saved_lp.items() if k not in lane_keys} != {
        k: v for k, v in actual_lp.items() if k not in lane_keys
    }:
        return False
    saved_cards = saved_lp.get("card_levels")
    saved_towers = saved_lp.get("tower_levels")
    actual_cards = actual_lp.get("card_levels")
    actual_towers = actual_lp.get("tower_levels")
    if (
        not isinstance(saved_cards, list)
        or not isinstance(saved_towers, list)
        or not isinstance(actual_cards, list)
        or not isinstance(actual_towers, list)
        or not saved_cards
        or not actual_cards
        or len(saved_cards) != len(saved_towers)
        or len(actual_cards) != len(actual_towers)
        or len(saved_cards) == len(actual_cards)
    ):
        return False
    saved_dist = {
        pair: Fraction(count, len(saved_cards))
        for pair, count in Counter(zip(saved_cards, saved_towers)).items()
    }
    actual_dist = {
        pair: Fraction(count, len(actual_cards))
        for pair, count in Counter(zip(actual_cards, actual_towers)).items()
    }
    return saved_dist == actual_dist


class PersistentStateSelfplay:
    """Full-feature persistent architecture-v3 self-play PPO trainer."""

    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError("torchrun process group must be initialized before v3 PPO")
        if not getattr(args, "init_from_bc", None) and not getattr(args, "resume", None):
            raise ValueError("v3 PPO requires --init-from-bc or --resume")

        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "opponents").mkdir(exist_ok=True)
        (self.out / "rollouts").mkdir(exist_ok=True)
        (self.out / "checkpoints").mkdir(exist_ok=True)
        (self.out / "timings").mkdir(exist_ok=True)

        self.precision = getattr(args, "precision", None) or (
            "fp16" if getattr(args, "amp_dtype", "auto") in ("auto", "fp16") else "fp32"
        )
        if self.precision == "fp16" and os.environ.get("CRRL_AMP_DTYPE") in (None, "auto"):
            os.environ["CRRL_AMP_DTYPE"] = "fp16"

        self.init_source = Path(args.resume or args.init_from_bc)
        self.device = _resolve_device(getattr(args, "device", "auto"), self.rank, self.world_size)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        checkpoint = streaming_torch_load(self.init_source, target_device=self.device)
        if checkpoint.get("model_architecture_version") != 3:
            raise ValueError("State PPO requires an architecture-v3 checkpoint")
        if checkpoint.get("decision_ticks") != 8:
            raise ValueError("State PPO requires the 0.4s / 8-tick decision grid")

        req_mode = getattr(args, "critic_mode", None)
        self.critic_mode = (
            checkpoint.get("critic_mode", "independent")
            if req_mode in (None, "auto")
            else req_mode
        )
        critic_dev_arg = getattr(args, "critic_device", None)
        if self.critic_mode == "independent" and critic_dev_arg:
            self.critic_device = (
                torch.device(f"cuda:{self.rank}")
                if critic_dev_arg == "cuda" and self.world_size > 1
                else torch.device(critic_dev_arg)
            )
        else:
            self.critic_device = self.device

        peer_arg = getattr(args, "critic_peer_device", None)
        if peer_arg:
            self.critic_peer_device = torch.device(peer_arg)
        elif (
            self.critic_mode == "independent"
            and getattr(args, "balanced_critic", False)
            and self.world_size == 1
        ):
            self.critic_peer_device = _auto_peer_device(self.device, self.critic_device)
        else:
            self.critic_peer_device = None
        if self.critic_peer_device is not None and self.critic_peer_device == self.critic_device:
            raise ValueError("Critic peer device must differ from primary critic device")

        opp_dev_arg = getattr(args, "opponent_device", None)
        self.opponent_device = torch.device(opp_dev_arg) if opp_dev_arg else self.device
        dual_rollout = bool(getattr(args, "dual_rollout", False))
        self.dual_rollout_device: torch.device | None = None
        if dual_rollout:
            for cand in (self.critic_device, self.critic_peer_device, self.opponent_device):
                if cand is not None and cand != self.device:
                    self.dual_rollout_device = cand
                    break
            if self.dual_rollout_device is None:
                raise ValueError("--dual-rollout requires a second CUDA device")

        for dev in (
            self.device,
            self.critic_device,
            self.critic_peer_device,
            self.opponent_device,
            self.dual_rollout_device,
        ):
            ensure_cuda_linalg_initialized(dev)

        self._init_models_and_optimizers(args, checkpoint)
        self._init_env_and_data(args, checkpoint)
        self._init_league_and_tracking(args, checkpoint)
        self.awbc_buffer = None
        if getattr(args, 'awbc_sessions', None):
            if self.world_size != 1 or self.critic is None:
                raise ValueError('AWBC playbuffer requires one PPO process and an independent Critic; dual GPU is supported')
            self.awbc_buffer = PlayBuffer(args.awbc_sessions, checkpoint.get('awbc_playbuffer'),
                                          max_reuse=getattr(args, 'awbc_max_reuse', 2))

        self.expert_bc = None
        if getattr(args, 'expert_bc_dataset', None):
            if self.world_size != 1:
                raise ValueError('Auxiliary BC requires one PPO process')
            if getattr(args, 'expert_bc_plan_b', False):
                from cr_rl.training.expert_bc_plan_b import PlanBExpertBC as expert_bc_type
            else:
                from cr_rl.training.expert_bc import ExpertBC as expert_bc_type
            bc_state = checkpoint.get('expert_bc')
            if getattr(args, 'expert_bc_plan_b', False) and (bc_state or {}).get('bc_plan') != 'sparky_b':
                bc_state = None
            self.expert_bc = expert_bc_type(args.expert_bc_dataset, self.actor,
                seed=args.seed, state=bc_state,
                experimental=getattr(args, 'expert_bc_experimental', False))

        self.macro_search = None
        if getattr(args, 'macro_search', False):
            if self.world_size != 1:
                raise ValueError('Macro card search requires one PPO process')
            from cr_rl.training.macro_card_search import MacroCardSearch
            self.macro_search = MacroCardSearch(
                self.out,
                max_buffer_episodes=int(getattr(args, 'macro_search_max_buffer', 128)),
                seed=args.seed,
                state=checkpoint.get('macro_search'),
            )

        self.lookahead_distiller = None
        if getattr(args, 'lookahead_distill', False):
            if self.world_size != 1:
                raise ValueError('Lookahead distillation requires one PPO process')
            from cr_rl.training.lookahead_distiller import LookaheadDistiller
            self.lookahead_distiller = LookaheadDistiller(
                self.out,
                max_buffer_episodes=int(getattr(args, 'lookahead_distill_max_buffer', 192)),
                seed=args.seed + 17,
                state=checkpoint.get('lookahead_distill'),
            )

        # Drop checkpoint mmaps before rollout starts to keep host RSS lean on 14 GiB hosts.
        detach_optimizer_counters(self.actor_optimizer)
        if self.critic_optimizer is not None:
            detach_optimizer_counters(self.critic_optimizer)
        del checkpoint
        release_host_memory()

    def _init_models_and_optimizers(self, args, checkpoint: dict) -> None:
        self.actor = state_model_from_checkpoint(checkpoint, self.device)
        self.actor.ppo_actor_fp32 = bool(getattr(args, "ppo_actor_fp32", False))
        # The fused eval Transformer differs from the training encoder in both
        # FP32 and FP16. Always use the same attention path in rollout and PPO.
        torch.backends.mha.set_fastpath_enabled(False)
        if os.environ.get("CRRL_ALLOW_TF32") == "1":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        checkpoint.pop("model", None)
        checkpoint.pop("actor", None)
        want_reserve = bool(getattr(args, "reserve_to_wait", False))
        self.reserve_migrated = bool(want_reserve and not self.actor.cfg.reserve_to_wait)
        if self.actor.cfg.reserve_to_wait != want_reserve:
            self.actor.cfg = replace(self.actor.cfg, reserve_to_wait=want_reserve)
        self.actor.gradient_checkpointing = bool(getattr(args, "gradient_checkpointing", False))
        self.actor.temporal_checkpoint_mode = getattr(args, "temporal_checkpoint_mode", "nested")
        self.actor.frame_microbatch = int(getattr(args, "frame_microbatch", 32))
        self.actor.eval()
        self.model = self.actor  # backwards-compat alias

        # Fixed BC anchor opponent for the league pool's 20% anchor role;
        # `behavior_model` separately tracks each rollout's pi_old (v2 parity).
        anchor_arg = getattr(args, "bc_anchor", None)
        saved_anchor = checkpoint.get("training_args", {}).get("bc_anchor") or checkpoint.get(
            "bc_anchor"
        )
        default_anchor = Path("artifacts/ppo_prep_20261113/best.pt")
        if anchor_arg is not None:
            anchor_path = Path(anchor_arg)
        elif saved_anchor is not None and Path(saved_anchor).is_file():
            anchor_path = Path(saved_anchor)
        elif not args.resume:
            anchor_path = self.init_source
        elif default_anchor.is_file():
            anchor_path = default_anchor
        else:
            anchor_path = self.init_source
        self.anchor_path = anchor_path
        if not self.anchor_path.is_file():
            raise FileNotFoundError(f"v3 league requires BC anchor: {self.anchor_path}")
        if not args.resume and self.anchor_path.resolve() == self.init_source.resolve():
            # Clone directly on GPU / NVLink without re-loading or mapping the checkpoint again.
            self.anchor_model = _clone_actor_to_device(self.actor, self.opponent_device)
        else:
            anchor_ckpt = streaming_torch_load(
                self.anchor_path, target_device=self.opponent_device
            )
            if (
                anchor_ckpt.get("model_architecture_version") != 3
                or anchor_ckpt.get("decision_ticks") != 8
            ):
                raise ValueError("BC anchor must be an architecture-v3 8-tick checkpoint")

            self.anchor_model = state_model_from_checkpoint(anchor_ckpt, self.opponent_device)
            self.anchor_model.frame_microbatch = self.actor.frame_microbatch
            del anchor_ckpt
            release_host_memory()
        self.anchor_model = _prepare_frozen_opponent(
            self.anchor_model, device=self.opponent_device, precision=self.precision
        )

        self.behavior_model = _clone_actor_to_device(self.actor, self.device)

        self.peer_actor = None
        self.actor_peer_device: torch.device | None = None
        balanced = bool(getattr(args, "balanced_critic", False))
        can_shard_actor = (
            self.world_size == 1 and self.critic_mode == "independent" and balanced
        )
        if self.dual_rollout_device is not None:
            self.actor_peer_device = self.dual_rollout_device if can_shard_actor else None
            self.peer_actor = _clone_actor_to_device(
                self.actor, self.dual_rollout_device, trainable=can_shard_actor
            )
        elif can_shard_actor and self.critic_device != self.device:
            self.actor_peer_device = self.critic_device
            self.peer_actor = _clone_actor_to_device(
                self.actor, self.actor_peer_device, trainable=True
            )

        if self.peer_actor is not None:
            self.peer_actor.gradient_checkpointing = self.actor.gradient_checkpointing
            self.peer_actor.temporal_checkpoint_mode = self.actor.temporal_checkpoint_mode

        # Independent or Shared Critic
        self.critic = None
        self.peer_critic = None
        self.critic_optimizer = None
        if self.critic_mode == "independent":
            if "critic" in checkpoint and checkpoint["critic"] is not None:
                self.critic = state_value_model_from_checkpoint(checkpoint, self.critic_device)
            else:
                with torch.device(self.critic_device):
                    self.critic = StateValueNetwork(
                        copy.deepcopy(self.actor.cfg),
                        self.actor.temporal_cfg,
                        decision_ticks=8,
                        frame_microbatch=self.actor.frame_microbatch,
                    )
                self.critic.initialize_encoder(self.actor)
            checkpoint.pop("critic", None)
            self.critic.frame_microbatch = self.actor.frame_microbatch
            self.critic.gradient_checkpointing = bool(getattr(args, "gradient_checkpointing", False))
            self.critic.temporal_checkpoint_mode = getattr(args, "temporal_checkpoint_mode", "nested")
            self.critic.eval()

            if self.critic_peer_device is not None:
                with torch.device(self.critic_peer_device):
                    self.peer_critic = StateValueNetwork(
                        copy.deepcopy(self.critic.cfg),
                        self.critic.temporal_cfg,
                        decision_ticks=8,
                        frame_microbatch=self.actor.frame_microbatch,
                    )
                self.peer_critic.gradient_checkpointing = self.critic.gradient_checkpointing
                self.peer_critic.temporal_checkpoint_mode = self.critic.temporal_checkpoint_mode
                self.peer_critic.eval()
                self._sync_peer_critic()

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
            fused=self.device.type == "cuda",
        )
        saved_actor_scaler = checkpoint.get("actor_amp_scaler") or checkpoint.get("scaler")
        if saved_actor_scaler and not self.actor.ppo_actor_fp32 and self.precision == "fp16":
            from cr_rl.training.fp16 import sanitize_grad_scaler_state
            saved_actor_scaler = sanitize_grad_scaler_state(saved_actor_scaler)
            if float(saved_actor_scaler.get("scale", 0.0)) > 4096.0:
                saved_actor_scaler = dict(saved_actor_scaler, scale=4096.0, _growth_tracker=0)
        self.actor_optimizer.amp_scaler = _make_scaler(
            self.device, self.precision, saved_actor_scaler if args.resume else None
        )

        if self.critic is not None:
            critic_lr = getattr(args, "critic_learning_rate", None) or args.learning_rate
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(),
                lr=critic_lr,
                weight_decay=float(getattr(args, "weight_decay", 0.0)),
                fused=self.critic_device.type == "cuda",
            )
            critic_scale_state = checkpoint.get("critic_amp_scaler") if args.resume else None
            scale_cap = getattr(args, "critic_fp16_resume_scale_cap", None)
            if critic_scale_state and scale_cap is not None and self.precision == "fp16":
                from cr_rl.training.fp16 import sanitize_grad_scaler_state
                critic_scale_state = sanitize_grad_scaler_state(critic_scale_state)
                if float(critic_scale_state["scale"]) > scale_cap:
                    critic_scale_state = dict(critic_scale_state,scale=float(scale_cap),_growth_tracker=0)
            self.critic_optimizer.amp_scaler = _make_scaler(
                self.critic_device,
                self.precision,
                critic_scale_state,
            )

        if args.resume:
            opt_state = checkpoint.get("actor_optimizer") or checkpoint.get("optimizer")
            if opt_state:
                restore_actor_optimizer(self.actor_optimizer, opt_state, self.actor)
            del opt_state
            checkpoint.pop("optimizer", None)
            checkpoint.pop("actor_optimizer", None)
            if self.critic_optimizer is not None and checkpoint.get("critic_optimizer"):
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            checkpoint.pop("critic", None)
            checkpoint.pop("critic_optimizer", None)
            resume_actor_lr = getattr(args, "resume_actor_learning_rate", None)
            if resume_actor_lr is None and self.reserve_migrated:
                resume_actor_lr = min(1.5e-6, float(self.actor_optimizer.param_groups[0]["lr"]))
            self.resume_learning_rates = override_resume_rates(
                self.actor_optimizer,
                self.critic_optimizer or self.actor_optimizer,
                resume_actor_lr,
                getattr(args, "resume_critic_learning_rate", None),
            )
        else:
            self.resume_learning_rates = None

        # Drop any remaining heavy weight/optimizer references immediately after GPU loads finish.
        for heavy_key in ("model", "actor", "critic", "optimizer", "actor_optimizer", "critic_optimizer"):
            checkpoint.pop(heavy_key, None)
        release_host_memory()

        self.actor_lr_controller = restore_actor_lr_controller(
            self.actor_optimizer,
            checkpoint.get("adaptive_actor_lr") if args.resume else None,
            enabled=bool(getattr(args, "adaptive_actor_lr", False)),
            min_lr=float(getattr(args, "adaptive_actor_lr_min", 2.5e-6)),
            max_lr=getattr(args, "adaptive_actor_lr_max", None),
            allow_bound_override=bool(
                getattr(args, "override_adaptive_actor_lr", False)
                or getattr(args, "resume_actor_learning_rate", None) is not None
            ),
        )

        # Optional DDP wrappers used strictly during optimization.
        self._actor_opt_model = (
            torch.nn.parallel.DistributedDataParallel(
                self.actor,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if self.world_size > 1
            else self.actor
        )
        self._critic_opt_model = (
            torch.nn.parallel.DistributedDataParallel(
                self.critic,
                device_ids=[self.critic_device.index] if self.critic_device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            if (self.world_size > 1 and self.critic is not None)
            else self.critic
        )

        progress = checkpoint.get("progress", {})
        self.update = int(checkpoint.get("updates", progress.get("update", 0))) if args.resume else 0
        self.start_update = self.update
        self.target_update = self.update + int(getattr(args, "updates", 1))
        self.samples = (
            int(checkpoint.get("learner_samples_total", progress.get("learner_samples_total", 0)))
            if args.resume
            else 0
        )
        self.optimizer_steps_total = (
            int(checkpoint.get("optimizer_steps_total", 0)) if args.resume else 0
        )
        self.critic_steps_total = int(checkpoint.get("critic_steps_total", 0)) if args.resume else 0
        self.actor_weights_dirty = False
        self.actor_diverged = False
        self._deck_vs_bc = self._init_deck_vs_bc_tracker(args, checkpoint)

        if args.resume:
            if checkpoint.get("numpy_rng_state") is not None:
                self.rng = np.random.default_rng()
                self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
            else:
                self.rng = np.random.default_rng(args.seed + self.rank + self.update)
            if checkpoint.get("torch_rng_state") is not None:
                t_rng = checkpoint["torch_rng_state"]
                torch.set_rng_state(t_rng.cpu() if isinstance(t_rng, torch.Tensor) else t_rng)
            if self.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
                try:
                    c_rng = [
                        t.cpu() if isinstance(t, torch.Tensor) else t
                        for t in checkpoint["cuda_rng_state"][:torch.cuda.device_count()]
                    ]
                    torch.cuda.set_rng_state_all(c_rng)
                except RuntimeError:
                    pass
        else:
            self.rng = np.random.default_rng(args.seed + self.rank)

    def _init_env_and_data(self, args, checkpoint: dict) -> None:
        self.gamma = float(args.gamma if getattr(args, 'gamma', None) is not None else .995)
        self.gae_lambda = float(getattr(args, "gae_lambda", None) or 0.95)
        self.critic_gae_lambda = getattr(args, "critic_gae_lambda", None)
        if not getattr(args, "overflow_penalty", None):
            overflow_pen = 0.0
        else:
            overflow_pen = float(args.overflow_penalty)
        tower_shaping = bool(
            True if getattr(args, "tower_shaping", None) is None else args.tower_shaping
        )
        elixir_values=None
        if getattr(args,'army_reward_mode',None)=='elixir_potential':
            if not tower_shaping: raise ValueError('Elixir shaping requires enabled reward shaping')
            from cr_rl.training.army_elixir import load_values
            elixir_values=load_values(args.army_elixir_config)
        self.rewards = RewardConfig(
            enabled=tower_shaping,
            princess=float(getattr(args, "princess_reward", 0.15)),
            king=float(getattr(args, "king_reward", 0.20)),
            crown=float(getattr(args, "crown_reward", 0.15)),
            overflow=overflow_pen,
            army_potential_weight=float(getattr(args, "army_reward", 0.0)),
            army_reward_mode=getattr(args,'army_reward_mode','uniform') or 'uniform',
            army_elixir_beta=float(args.army_elixir_beta if getattr(args,'army_elixir_beta',None) is not None else .004),
            army_elixir_bar_factor=float(getattr(args, 'army_elixir_bar_factor', None) or 0.),
            army_elixir_values=elixir_values,
        )

        self.env = CrForgeVector(
            args.envs,
            int(getattr(args, "env_threads", 1)),
            independent=True,
            arena="Arena_L",
            level_mixture=getattr(args, "level_mixture", None),
            tower_level_mixture=getattr(args, "tower_level_mixture", None),
            level_jitter_prob=getattr(args, "level_jitter_prob", None),
            level_jitter=int(getattr(args, "level_jitter", 2)),
            level_jitter_seed=args.seed + self.rank,
            decision_ticks=8,
        )
        self.env.set_autoreset(False)
        self.army_elixir = None
        self.army_phi = np.zeros((args.envs,2),dtype=np.float64)
        if self.rewards.army_reward_mode=='elixir_potential':
            from cr_rl.training.army_elixir import NativeArmyElixir
            self.army_elixir=NativeArmyElixir(self.env,self.rewards.army_elixir_values)
        self.schema = self.env.vector.schema
        self.hand_indices = np.asarray(self.schema.header["hand_card"], dtype=np.int64)
        self.sparky_curriculum = None
        wait_fraction = getattr(args, "sparky_wait_fraction", 0.0)
        play_fraction = getattr(args, "sparky_play_fraction", 0.0)
        if wait_fraction or play_fraction:
            from cr_rl.training.sparky_curriculum import SparkyCurriculum

            vector = self.env.vector
            self.sparky_curriculum = SparkyCurriculum(
                self.schema, vector.card_id("ZapMachine"), vector.max_elixir,
                wait_fraction, play_fraction, getattr(args, "sparky_wait_max_steps", 3),
            )
        self.adapter = CrForgeObservationAdapter(
            self.schema, self.env.vector.vocabulary_size, entity_bucket=16
        )
        self.level_profile = dict(self.env.level_profile)
        self.level_profile_migrated = False

        deck_path = Path(getattr(args, "decks", None) or "configs/decks_ppo46.yaml")
        self.league, self.deck_names, self.decks, self.forms = _extract_decks(deck_path)
        self.deck_probabilities = deck_probabilities(self.league, self.deck_names)
        focal = [x.strip() for x in (getattr(args, "focal_decks", "") or "").split(",") if x.strip()]
        missing = set(focal) - set(self.deck_names)
        if missing:
            raise ValueError(f"Unknown focal decks: {sorted(missing)}")
        self.focal_deck_indices = [self.deck_names.index(x) for x in focal]
        self.active_historical_keys = None
        self._cohort_update = -1
        self.deck_pool_transitions: list[dict] = list(
            checkpoint.get("deck_pool_transitions", []) if args.resume else []
        )

        if args.resume:
            meta = self.env.metadata()
            for key, actual in (
                ("backend_commit", meta.get("commit")),
                ("data_version", self.env.vector.data_version),
                ("content_sha", self.env.vector.content_sha),
            ):
                saved_v = checkpoint.get(key)
                if saved_v is not None and actual != saved_v:
                    raise ValueError(f"Resume simulator provenance mismatch for {key}")
            saved_reward = checkpoint.get("reward_config")
            if (saved_reward is not None and saved_reward!=self.rewards.metadata()
                    and self.rewards.army_reward_mode=='elixir_potential'
                    and not getattr(args, "allow_reward_migration", False)
                    and Path(args.out).resolve()==self.init_source.resolve().parent):
                raise ValueError('Reward migration requires a separate experiment directory')
            if (
                saved_reward is not None
                and saved_reward != self.rewards.metadata()
                and not getattr(args, "allow_reward_migration", False)
            ):
                raise ValueError("Resume reward configuration differs from checkpoint")
            saved_lp = checkpoint.get("level_profile")
            if saved_lp is not None and not _level_profiles_compatible(saved_lp, self.level_profile):
                if not getattr(args, "allow_level_profile_migration", False):
                    raise ValueError("Resume level_profile differs (pass --allow-level-profile-migration)")
                self.level_profile_migrated = True
            saved_league = checkpoint.get("league_config")
            if saved_league is not None:
                prev_names = list(saved_league.get("decks", {}))
                if set(prev_names) != set(self.deck_names):
                    if not getattr(args, "expand_decks", False):
                        raise ValueError("Resume decks differ (pass --expand-decks to add new decks)")
                    if not set(prev_names).issubset(set(self.deck_names)):
                        raise ValueError("--expand-decks requires retaining all prior checkpoint decks")
                    self.deck_pool_transitions.append(
                        {"update": self.update, "previous": prev_names, "current": self.deck_names}
                    )

        (self.out / "league.yaml").write_text(yaml.safe_dump(self.league), encoding="utf-8")

        self.warmup_updates = int(getattr(args, "critic_warmup_updates", 0) or 0)
        self.warmup_games = int(getattr(args, "critic_warmup_games", 0) or 0)
        self.game_counter = TrainingGameCounter(
            checkpoint.get("training_game_counter") if args.resume else None,
            origin_update=self.update,
        )
        self.next_episode = int(checkpoint.get("next_episode", 0)) if args.resume else 0

        self.episode_ids = np.zeros(args.envs, dtype=np.int64)
        self.seeds = np.zeros(args.envs, dtype=np.int64)
        self.learner_seats = np.zeros(args.envs, dtype=np.int32)
        self.match_decks: list[tuple[str, str]] = [("default", "default")] * args.envs
        self.opponents: list[OpponentEntry | None] = [None] * args.envs
        self.lane_opp_keys: list[str | None] = [None] * args.envs
        self.traces: list[EpisodeTrace | None] = [None] * args.envs
        self.lane_actor_mem = self.actor.initial_state(args.envs, device=self.device)
        self.lane_opp_mem: list[PolicyMemory | None] = [None] * args.envs

    def _init_league_and_tracking(self, args, checkpoint: dict) -> None:
        self.role_league = bool(getattr(args, "role_league", True))
        hist_f = getattr(args, "historical_opponent_fraction", None)
        anc_f = getattr(args, "anchor_opponent_fraction", None)
        cur_f = getattr(args, "current_opponent_fraction", None)
        if hist_f is None and anc_f is None and cur_f is None:
            hist_f, anc_f = (0.50, 0.20) if self.role_league else (0.0, 0.0)
            cur_f = 1.0 - hist_f - anc_f
        elif cur_f is None:
            hist_f = float(0.0 if hist_f is None else hist_f)
            anc_f = float(0.0 if anc_f is None else anc_f)
            cur_f = 1.0 - hist_f - anc_f
        else:
            cur_f = float(cur_f)
            hist_f = float(0.0 if hist_f is None else hist_f)
            anc_f = float(0.0 if anc_f is None else anc_f)
        self.role_mixture = RoleMixture(current=cur_f, historical=hist_f, anchor=anc_f)
        self.role_draws = dict(
            checkpoint.get("role_draws", {"current": 0, "historical": 0, "anchor": 0})
            if args.resume
            else {"current": 0, "historical": 0, "anchor": 0}
        )

        pool_path = self.out / "opponent_pool.json"
        requested_capacity = getattr(args, "pool_capacity", None)
        if args.resume and pool_path.exists():
            self.pool = OpponentPool.load(pool_path)
            if requested_capacity is not None:
                self.pool.max_entries = max(2, int(requested_capacity))
        else:
            self.pool = OpponentPool(
                max_entries=int(8 if requested_capacity is None else requested_capacity),
                ema_decay=0.95,
                uniform_mix=0.1,
            )
            anchor_copy = self.save_opponent(0)
            self.pool.add(OpponentEntry(0, str(anchor_copy), 0, role="anchor"))

        if getattr(args, "history_seed_manifest", None) and not args.resume:
            meta = self.env.metadata()
            seed_history(
                self.pool,
                args.history_seed_manifest,
                self.out / "opponents",
                self.actor.cfg.__dict__,
                {
                    "backend_commit": meta.get("commit"),
                    "data_version": self.env.vector.data_version,
                    "content_sha": self.env.vector.content_sha,
                },
                temporal_config=self.actor.temporal_cfg.__dict__,
            )
        if getattr(args, "league_diversity_manifest", None):
            self.pool.configure_diverse_retention(
                args.league_diversity_manifest, learner_version=self.update
            )
        self.pool.save(pool_path)

        saved_adm = checkpoint.get("admission") if args.resume else None
        if self.role_league and self.role_mixture.historical > 0:
            self.admission = SnapshotAdmission(
                purpose="history",
                interval_samples=int(getattr(args, "history_snapshot_interval", 20_000)),
                last_admitted_samples=int(saved_adm.get("last_admitted_samples", self.samples))
                if saved_adm
                else self.samples,
            )
        else:
            self.admission = SnapshotAdmission(
                purpose="promotion",
                interval_samples=int(getattr(args, "snapshot_interval", 250_000)),
                min_games=int(getattr(args, "snapshot_min_games", 20)),
                min_score=float(getattr(args, "snapshot_min_score", 0.55)),
                last_admitted_samples=int(saved_adm.get("last_admitted_samples", self.samples))
                if saved_adm
                else self.samples,
            )

        self.rollout_opponent_cache_size = max(
            1,
            int(getattr(args, "opponent_cache_size", 4) or 4),
        )
        self.training_opponent_cache_size = max(
            1,
            min(
                self.rollout_opponent_cache_size,
                int(getattr(args, "training_opponent_cache_size", None) or self.rollout_opponent_cache_size),
            ),
        )
        self.opponent_cache_size = self.training_opponent_cache_size
        self.opponent_cache: OrderedDict[str, torch.nn.Module] = OrderedDict()

        # Thread executors for overlapped rollout / async backup publication.
        self._opp_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3-opponent-rollout")
            if self.opponent_device != self.device
            else None
        )
        self._peer_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3-actor-peer")
            if self.dual_rollout_device is not None
            else None
        )
        self._publisher = None
        if getattr(args, "async_publication", False) and getattr(args, "backup_dir", None):
            from cr_rl.training.async_publication import BoundedAsyncPublisher

            self._publisher = BoundedAsyncPublisher(self.out, Path(args.backup_dir),
                checksum=not getattr(args, 'no_backup_hash', False))

        self.tensorboard = None
        if getattr(args, "tensorboard_dir", None) and self.rank == 0:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(str(args.tensorboard_dir))
            self.tensorboard.add_text("run/architecture", "architecture-v3 persistent PPO", 0)

        self.tracking = self.telemetry = None
        if getattr(args, "wandb", False) and self.rank == 0:
            import wandb

            run_file = self.out / "wandb_run.json"
            wb_opts = (
                {"id": json.loads(run_file.read_text())["id"], "resume": "allow"}
                if run_file.exists()
                else {}
            )
            self.tracking = wandb.init(
                project=os.environ.get('WANDB_PROJECT','cr-rl'),
                name=self.out.name,
                job_type="ppo-v3",
                config={
                    "architecture": 3,
                    "decision_ticks": 8,
                    "precision": self.precision,
                    "critic_mode": self.critic_mode,
                    "reward": self.rewards.metadata(),
                },
                **wb_opts,
            )
            atomic_json(run_file, {"id": self.tracking.id, "url": self.tracking.url})
            self.telemetry = TrainingTelemetry(self.tracking, args.steps * args.envs * self.world_size)

        atomic_json(
            self.out / "run_config.json",
            {
                "driver": "persistent-state-v3",
                "architecture": 3,
                "decision_ticks": 8,
                "critic_mode": self.critic_mode,
                "devices": {
                    "actor": str(self.device),
                    "critic": str(self.critic_device),
                    "critic_peer": str(self.critic_peer_device) if self.critic_peer_device else None,
                    "opponent": str(self.opponent_device),
                    "dual_rollout": str(self.dual_rollout_device) if self.dual_rollout_device else None,
                },
                "precision": self.precision,
                "ppo_actor_fp32": self.actor.ppo_actor_fp32,
                "reward": self.rewards.metadata(),
                **({"sparky_curriculum": {
                    "wait_fraction": self.sparky_curriculum.wait_fraction,
                    "play_fraction": self.sparky_curriculum.play_fraction,
                    "wait_max_steps": self.sparky_curriculum.wait_max_steps,
                }} if self.sparky_curriculum is not None else {}),
            },
        )
        for env in range(args.envs):
            self._reset_match(env)

    def _sync_peer_actor(self) -> None:
        if self.peer_actor is None:
            return
        _copy_module_weights(self.actor, self.peer_actor, sync_device=self.dual_rollout_device)
        self.peer_actor.mark_weights_updated()

    def _sync_behavior_model(self) -> None:
        """Keep frozen rollout behavior reference synchronized with pre-update `actor` (v2 parity)."""
        if self.behavior_model is None:
            return
        _copy_module_weights(self.actor, self.behavior_model, sync_device=self.device)
        if hasattr(self.behavior_model, "mark_weights_updated"):
            self.behavior_model.mark_weights_updated()
        self.actor_diverged = False

    def _sync_peer_critic(self) -> None:
        if self.peer_critic is None:
            return
        _copy_module_weights(self.critic, self.peer_critic, sync_device=self.critic_peer_device)

    def _pinned_opponent_keys(self, *, exclude_env: int | None = None) -> set[str]:
        opponents = getattr(self, "opponents", None) or []
        pinned: set[str] = set()
        for idx, entry in enumerate(opponents):
            if idx == exclude_env or entry is None:
                continue
            if getattr(entry, "role", None) == "anchor" and getattr(entry, "policy_id", None) == 0:
                continue
            ckpt = getattr(entry, "checkpoint", None)
            if ckpt is not None:
                pinned.add(str(Path(ckpt).resolve()))
        return pinned

    def _trim_opponent_cache(
        self,
        max_entries: int,
        *,
        protect_key: str | None = None,
        allow_pinned_eviction: bool = True,
    ) -> None:
        limit = max(1, int(max_entries))
        if len(self.opponent_cache) <= limit:
            return
        pinned = self._pinned_opponent_keys()
        if protect_key is not None:
            pinned.add(protect_key)
        evicted_any = False
        predicates = (
            (lambda k: k not in pinned, lambda k: k != protect_key, lambda _k: True)
            if allow_pinned_eviction
            else (lambda k: k not in pinned,)
        )
        # Evict inactive opponents first so live rollout matches never thrash against the LRU limit.
        with _cuda_device_ctx(self.opponent_device):
            _sync_cuda_device(self.opponent_device)
            for eligible in predicates:
                for key in [k for k in self.opponent_cache if eligible(k)]:
                    if len(self.opponent_cache) <= limit:
                        break
                    evicted = self.opponent_cache.pop(key)
                    del evicted
                    evicted_any = True
                if len(self.opponent_cache) <= limit:
                    break

    def _load_opponent_model(self, entry: OpponentEntry | None) -> tuple[str, torch.nn.Module]:
        if entry is None:
            return "current", self.actor
        if entry.role == "anchor" and entry.policy_id == 0:
            return "anchor", self.anchor_model
        key = str(Path(entry.checkpoint).resolve())
        if key in self.opponent_cache:
            self.opponent_cache.move_to_end(key)
            return key, self.opponent_cache[key]
        with _cuda_device_ctx(self.opponent_device):
            _sync_cuda_device(self.opponent_device)
            rollout_cap = max(1, int(getattr(self, "rollout_opponent_cache_size", 1)))
            if rollout_cap > 1 and len(self.opponent_cache) >= rollout_cap:
                self._trim_opponent_cache(rollout_cap - 1, allow_pinned_eviction=False)
            load_path = Path(key)
            disk_cache = os.environ.get("CRRL_OPPONENT_DISK_CACHE")
            if disk_cache:
                load_path = cache_frozen_checkpoint(
                    load_path, Path(disk_cache), max_bytes=6 * 1024**3,
                )
            ckpt = streaming_torch_load(load_path, target_device=self.opponent_device)
            if ckpt.get("model_architecture_version") != 3:
                raise ValueError(f"Opponent checkpoint must be architecture v3: {key}")
            model = state_model_from_checkpoint(ckpt, self.opponent_device)
            del ckpt
            model.frame_microbatch = self.actor.frame_microbatch
            model = _prepare_frozen_opponent(
                model,
                device=self.opponent_device,
                precision=getattr(self, "precision", "fp32"),
            )
            self.opponent_cache[key] = model
            self._trim_opponent_cache(rollout_cap, protect_key=key)
            return key, model

    def _refresh_historical_cohort(self):
        capacity = int(getattr(self.args, "historical_cohort_size", 0))
        interval = int(getattr(self.args, "historical_cohort_updates", 2))
        if capacity <= 0:
            return
        epoch = self.update // interval
        available = {str(Path(e.checkpoint).resolve()): e for e in self.pool.entries if e.role == "historical"}
        alive = {str(Path(e.checkpoint).resolve()) for e in self.opponents
                 if e is not None and e.role == "historical"}
        desired = set(self.active_historical_keys or ()) & available.keys()
        if self._cohort_update != epoch:
            self._retiring_historical_key = str(self.rng.choice(sorted(desired))) if desired else None
            self._cohort_update = epoch
        retiring = getattr(self, "_retiring_historical_key", None)
        desired.discard(retiring)
        # Drain one identity across episode boundaries; never change a live
        # opponent or exceed the resident-cohort cap while rotating the pool.
        occupied = alive | desired
        candidates = [e for key,e in available.items() if key not in occupied and key != retiring]
        count = min(max(0, capacity-len(occupied)), len(candidates))
        if count:
            scores = np.array([self.pool.score_for(e, "multi_deck", "multi_deck", self.update) for e in candidates])
            weights = np.maximum(1-scores, 1e-3)
            weights = (1-self.pool.uniform_mix)*weights/weights.sum()+self.pool.uniform_mix/len(weights)
            for idx in self.rng.choice(len(candidates), size=count, replace=False, p=weights):
                desired.add(str(Path(candidates[int(idx)].checkpoint).resolve()))
        self.active_historical_keys = desired

    def _reset_match(self, env_idx: int) -> None:
        seed = int(self.rng.integers(2**31 - 1))
        if getattr(self, "deck_probabilities", None) is None:
            d0, d1 = (int(x) for x in self.rng.integers(len(self.decks), size=2))
        else:
            d0, d1 = (int(x) for x in self.rng.choice(
                len(self.decks), size=2, p=self.deck_probabilities))
        learner_seat = int(self.rng.integers(0, 2))
        if getattr(self, "focal_deck_indices", None) and self.rng.random() < self.args.focal_deck_prob:
            focal_idx = int(self.rng.choice(self.focal_deck_indices))
            if learner_seat == 0:
                d0 = focal_idx
            else:
                d1 = focal_idx
        self._refresh_historical_cohort()
        self.env.vector.reset_env(
            env_idx, seed, self.decks[d0], self.decks[d1], self.forms[d0], self.forms[d1]
        )
        own_name = self.deck_names[d0 if learner_seat == 0 else d1]
        opp_name = self.deck_names[d1 if learner_seat == 0 else d0]
        other_active = self._pinned_opponent_keys(exclude_env=env_idx)
        rollout_cap = max(1, int(getattr(self, "rollout_opponent_cache_size", 1)))
        sample_kwargs = (
            {"allowed_historical": other_active} if len(other_active) >= rollout_cap else {}
        )
        if getattr(self, "active_historical_keys", None):
            sample_kwargs = {"allowed_historical": self.active_historical_keys}
        entry, _, actual_role = self.role_mixture.sample(
            self.pool, self.rng, own_name, opp_name, self.update, **sample_kwargs
        )
        self.role_draws[actual_role] = int(self.role_draws.get(actual_role, 0)) + 1

        self.episode_ids[env_idx] = self.next_episode
        if getattr(self,'army_elixir',None) is not None:
            self.army_elixir.reset_lane(env_idx)
            self.army_phi[env_idx]=0
        self.next_episode += 1
        self.seeds[env_idx] = seed
        self.learner_seats[env_idx] = learner_seat
        curriculum = getattr(self, "sparky_curriculum", None)
        if curriculum is not None:
            curriculum.reset_lane(
                env_idx, seed, int(self.episode_ids[env_idx]), self.env.observations[env_idx, learner_seat]
            )
        self.match_decks[env_idx] = (own_name, opp_name)
        self.opponents[env_idx] = entry
        self.traces[env_idx] = EpisodeTrace(int(self.episode_ids[env_idx]), seed)

        reset_mask = torch.zeros(self.args.envs, dtype=torch.bool, device=self.device)
        reset_mask[env_idx] = True
        self.lane_actor_mem = PolicyMemory(
            self.lane_actor_mem.temporal.reset(reset_mask),
            id(self.actor),
            self.actor.memory_generation,
        )
        opp_key, opp_model = self._load_opponent_model(entry)
        opp_dev = self.device if entry is None else self.opponent_device
        self.lane_opp_keys[env_idx] = opp_key
        self.lane_opp_mem[env_idx] = opp_model.initial_state(1, device=opp_dev)

    def _burn_in_continuing_lanes(
        self,
    ) -> tuple[list[np.ndarray], list[PolicyMemory | None], list[PolicyMemory | None]]:
        """Rebuild exact recurrent states for continuing matches after Actor updates."""
        envs = self.args.envs
        prefix_raw: list[np.ndarray] = [
            np.zeros((0, self.schema.seat_ints), dtype=np.int32) for _ in range(envs)
        ]
        prefix_masks: list[np.ndarray] = [
            np.zeros((0, self.actor.cfg.action_count), dtype=bool) for _ in range(envs)
        ]
        cont_lanes: list[int] = []
        for env in range(envs):
            trace = self.traces[env]
            if trace is not None and len(trace.observations) > 0:
                seat = int(self.learner_seats[env])
                obs_arr = np.stack(trace.observations)
                mask_arr = np.stack(trace.masks)
                prefix_raw[env] = obs_arr[:, seat]
                prefix_masks[env] = mask_arr[:, seat]
                cont_lanes.append(env)

        h0_behavior: list[PolicyMemory | None] = [None] * envs
        h0_anchor: list[PolicyMemory | None] = [None] * envs
        burn_batch = max(int(getattr(self.args, "actor_batch_size", None) or getattr(self.args, "batch_size", 8) or 8), 8)
        chunk_size = int(getattr(self.args, "tbptt_chunk", 64))
        if not cont_lanes:
            if self.actor_weights_dirty:
                self.lane_actor_mem = self.actor.initial_state(envs, device=self.device)
                for e in range(envs):
                    if self.opponents[e] is None:
                        self.lane_opp_mem[e] = self.actor.initial_state(1, device=self.device)
                        self.lane_opp_keys[e] = "current"
                self.actor_weights_dirty = False
            return prefix_raw, h0_behavior, h0_anchor

        anchor_kl_coef = float(getattr(getattr(self, "args", None), "anchor_kl_coef", 0.0) or 0.0)
        want_anchor_burn = anchor_kl_coef > 0 and getattr(self, "anchor_model", None) is not None

        def _run_anchor_burn_in() -> list[PolicyMemory]:
            with _cuda_device_ctx(self.opponent_device):
                burned_anc = burn_in_memory(
                    self.anchor_model,
                    self.adapter,
                    [prefix_raw[e] for e in cont_lanes],
                    legal_masks=[prefix_masks[e] for e in cont_lanes],
                    device=self.opponent_device,
                    precision=self.precision,
                    batch_size=burn_batch,
                    chunk_size=chunk_size,
                )
                out = [
                    clone_memory(m, device=self.opponent_device, detach=True, trim_gqa=True)
                    for m in burned_anc
                ]
                _sync_cuda_device(self.opponent_device)
                return out

        if self.actor_weights_dirty:
            burned_actor = burn_in_memory(
                self.actor,
                self.adapter,
                [prefix_raw[e] for e in cont_lanes],
                legal_masks=[prefix_masks[e] for e in cont_lanes],
                device=self.device,
                precision=self.precision,
                batch_size=burn_batch,
                chunk_size=chunk_size,
            )
            cont_set = set(cont_lanes)
            per_lane = [self.actor.initial_state(1, device=self.device) for _ in range(envs)]
            for pos, env in enumerate(cont_lanes):
                per_lane[env] = burned_actor[pos]
            self.lane_actor_mem = stack_memories(per_lane, device=self.device)

            for e in range(envs):
                if e not in cont_set and self.opponents[e] is None:
                    self.lane_opp_mem[e] = self.actor.initial_state(1, device=self.device)
                    self.lane_opp_keys[e] = "current"

            cur_opp_cont = [e for e in cont_lanes if self.opponents[e] is None]
            if cur_opp_cont:
                opp_raw = [np.stack(self.traces[e].observations)[:, 1 - self.learner_seats[e]] for e in cur_opp_cont]
                opp_masks = [np.stack(self.traces[e].masks)[:, 1 - self.learner_seats[e]] for e in cur_opp_cont]
                burned_opp = burn_in_memory(
                    self.actor,
                    self.adapter,
                    opp_raw,
                    legal_masks=opp_masks,
                    device=self.device,
                    precision=self.precision,
                    batch_size=burn_batch,
                    chunk_size=chunk_size,
                )
                for pos, env in enumerate(cur_opp_cont):
                    self.lane_opp_mem[env] = burned_opp[pos]
                    self.lane_opp_keys[env] = "current"
            self.actor_weights_dirty = False

        if self.actor_diverged and float(getattr(getattr(self, "args", None), "reference_kl_coef", 0.01)) > 0:
            burned_beh = burn_in_memory(
                self.behavior_model,
                self.adapter,
                [prefix_raw[e] for e in cont_lanes],
                legal_masks=[prefix_masks[e] for e in cont_lanes],
                device=self.device,
                precision=self.precision,
                batch_size=burn_batch,
                chunk_size=chunk_size,
            )
            for pos, env in enumerate(cont_lanes):
                h0_behavior[env] = clone_memory(burned_beh[pos], device=self.device, detach=True, trim_gqa=True)

        if want_anchor_burn:
            anc_list = _run_anchor_burn_in()
            for pos, env in enumerate(cont_lanes):
                h0_anchor[env] = anc_list[pos]

        return prefix_raw, h0_behavior, h0_anchor

    def _align_frozen_opponent_memories(
        self, key: str, model: torch.nn.Module, lane_list: list[int]
    ) -> PolicyMemory:
        """Align continuing frozen-opponent memories after LRU cache eviction/reload."""
        aligned: list[PolicyMemory] = []
        for env_idx in lane_list:
            mem = self.lane_opp_mem[env_idx]
            if mem is None or self.lane_opp_keys[env_idx] != key:
                raise ValueError(
                    f"Opponent memory key mismatch on lane {env_idx}: "
                    f"expected {key!r}, got {self.lane_opp_keys[env_idx]!r}"
                )
            if mem.generation != model.memory_generation:
                raise ValueError(
                    f"Opponent memory generation mismatch on lane {env_idx} for {key!r}: "
                    f"{mem.generation} != {model.memory_generation}"
                )
            if mem.owner != id(model):
                # Reloading an evicted frozen checkpoint yields a new nn.Module instance
                # with identical immutable weights; rebinding restores ownership.
                mem = rebind_memory(mem, model)
                self.lane_opp_mem[env_idx] = mem
            aligned.append(mem)
        return stack_memories(aligned, device=self.opponent_device)

    def _run_model_step_group(
        self,
        model,
        device: torch.device,
        raw_rows: np.ndarray,
        legal_rows: np.ndarray,
        mem: PolicyMemory,
        *,
        collect_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, PolicyMemory, dict | None]:
        header, entities, entity_mask = self.adapter(raw_rows)
        non_blocking = device.type == "cuda"
        ht = torch.as_tensor(header).to(device=device, non_blocking=non_blocking)
        et = torch.as_tensor(entities).to(device=device, non_blocking=non_blocking)
        emt = torch.as_tensor(entity_mask).to(device=device, non_blocking=non_blocking)
        lt = torch.as_tensor(legal_rows, dtype=torch.bool).to(device=device, non_blocking=non_blocking)
        hdt = torch.as_tensor(raw_rows[:, self.hand_indices], dtype=torch.long).to(
            device=device, non_blocking=non_blocking
        )
        # Whole-game waves have an exact bounded prefix. Continuous lanes keep
        # the original allocation-boundary correction for pessimistic bounds.
        if getattr(self.args, "full_episode_bptt", False):
            mem = reserve_step_memory(mem, model.temporal_cfg.cache_initial_capacity)
        with _amp_context(device, self.precision):
            logp, _, next_mem = model.step(
                ht, et, emt, legal_action_mask=lt, state=mem, hand_card_ids=hdt, compute_value=False
            )
        act_np, lp_np, stats = _batched_sample(logp, lt, collect_stats=collect_stats)
        return act_np, lp_np, next_mem, stats

    def _step_models(
        self,
        raw: np.ndarray,
        masks: np.ndarray,
        *,
        collect_stats: bool = False,
        active_lanes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict | None]:
        """Run all active seats per-model in contiguous GPU batches with optional device overlap."""
        envs = self.args.envs
        actions = np.zeros((envs, 2), dtype=np.int64)
        learner_logp = np.zeros(envs, dtype=np.float32)

        rows = np.arange(envs) if active_lanes is None else np.asarray(active_lanes, dtype=np.int64)
        nactive = len(rows)
        cur_opp_lanes = [int(e) for e in rows if self.opponents[e] is None]
        frozen_groups: dict[str, tuple[torch.nn.Module, list[int]]] = {}
        for e in rows:
            entry = self.opponents[e]
            if entry is not None:
                k, mdl = self._load_opponent_model(entry)
                if k not in frozen_groups:
                    frozen_groups[k] = (mdl, [])
                frozen_groups[k][1].append(e)

        self._rollout_max_frozen_groups = max(getattr(self, "_rollout_max_frozen_groups", 0), len(frozen_groups))

        def run_frozen_groups():
            with _cuda_device_ctx(self.opponent_device), torch.inference_mode():
                out = {}
                wave_fmem = getattr(self, "_wave_frozen_mem", None) if (
                    getattr(self.args, "full_episode_bptt", False) and active_lanes is not None
                ) else None
                for key, (mdl, lane_list) in frozen_groups.items():
                    seats = 1 - self.learner_seats[lane_list]
                    f_raw = raw[lane_list, seats]
                    f_masks = masks[lane_list, seats]
                    if wave_fmem is not None:
                        cached = wave_fmem.get(key)
                        if cached is None:
                            mem = self._align_frozen_opponent_memories(key, mdl, lane_list)
                        else:
                            prev_lanes, prev_mem = cached
                            if prev_lanes == lane_list:
                                mem = prev_mem
                            else:
                                p_map = {e: idx for idx, e in enumerate(prev_lanes)}
                                mem = slice_memory(
                                    prev_mem, [p_map[e] for e in lane_list],
                                    device=self.opponent_device, detach=True, trim_gqa=False,
                                )
                        act_np, _, next_mem, _ = self._run_model_step_group(
                            mdl, self.opponent_device, f_raw, f_masks, mem, collect_stats=False
                        )
                        wave_fmem[key] = (list(lane_list), next_mem)
                    else:
                        mem = self._align_frozen_opponent_memories(key, mdl, lane_list)
                        act_np, _, next_mem, _ = self._run_model_step_group(
                            mdl, self.opponent_device, f_raw, f_masks, mem, collect_stats=False
                        )
                        position_bounds = next_mem.temporal.positions.tolist()
                        for pos, e in enumerate(lane_list):
                            self.lane_opp_mem[e] = slice_memory(
                                next_mem, [pos], device=self.opponent_device, detach=True, trim_gqa=True,
                                position_bound=position_bounds[pos],
                            )
                    out[key] = (lane_list, seats, act_np)
                if self.opponent_device != self.device:
                    _sync_cuda_device(self.opponent_device)
                return out

        opp_future = (
            self._opp_executor.submit(run_frozen_groups)
            if (self._opp_executor is not None and self.opponent_device != self.device and frozen_groups)
            else None
        )

        # Learner seats (+ "current" opponent seats) batched together, optionally split for --dual-rollout.
        l_seats = self.learner_seats[rows]
        l_raw = raw[rows, l_seats]
        l_masks = masks[rows, l_seats]
        total_cur = nactive + len(cur_opp_lanes)
        use_wave_cur = (
            bool(getattr(self.args, "full_episode_bptt", False))
            and active_lanes is not None
            and self.dual_rollout_device is None
            and hasattr(self, "_wave_cur_mem_ready")
            and self._wave_cur_mem_ready
        )
        rows_list = rows.tolist() if use_wave_cur else None
        if cur_opp_lanes:
            co_seats = 1 - self.learner_seats[cur_opp_lanes]
            all_raw = np.concatenate([l_raw, raw[cur_opp_lanes, co_seats]], axis=0)
            all_masks = np.concatenate([l_masks, masks[cur_opp_lanes, co_seats]], axis=0)
        else:
            all_raw = l_raw
            all_masks = l_masks

        if use_wave_cur and self._wave_cur_mem is not None:
            if self._wave_active_rows == rows_list and self._wave_cur_opp_lanes == cur_opp_lanes:
                cur_mem = self._wave_cur_mem
            else:
                n_prev = len(self._wave_active_rows)
                r_map = {e: idx for idx, e in enumerate(self._wave_active_rows)}
                co_map = {e: n_prev + idx for idx, e in enumerate(self._wave_cur_opp_lanes)}
                keep_all = [r_map[e] for e in rows_list] + [co_map[e] for e in cur_opp_lanes]
                cur_mem = slice_memory(
                    self._wave_cur_mem, keep_all, device=self.device, detach=True, trim_gqa=False
                )
        else:
            learner_memory = self.lane_actor_mem if nactive == envs else slice_memory(
                self.lane_actor_mem, rows.tolist(), device=self.device, detach=True, trim_gqa=True)
            if cur_opp_lanes:
                cur_mem = stack_memories(
                    [learner_memory] + [self.lane_opp_mem[e] for e in cur_opp_lanes],
                    device=self.device,
                )
            else:
                cur_mem = learner_memory

        peer_act_future = None
        if self.dual_rollout_device == self.opponent_device and frozen_groups:
            split_at = min(total_cur, nactive)
        else:
            split_at = (total_cur + 1) // 2
        if (
            self.dual_rollout_device is not None
            and self.peer_actor is not None
            and self._peer_executor is not None
            and split_at < total_cur
        ):
            peer_mem = slice_memory(
                cur_mem,
                list(range(split_at, total_cur)),
                device=self.dual_rollout_device,
                detach=True,
                trim_gqa=True,
            )
            peer_mem = PolicyMemory(
                peer_mem.temporal, id(self.peer_actor), self.peer_actor.memory_generation
            )
            peer_raw = all_raw[split_at:]
            peer_masks = all_masks[split_at:]

            def run_peer_actor(p_raw=peer_raw, p_masks=peer_masks, p_mem=peer_mem):
                ctx = (
                    torch.cuda.device(self.dual_rollout_device)
                    if self.dual_rollout_device.type == "cuda"
                    else nullcontext()
                )
                with ctx, torch.inference_mode():
                    return self._run_model_step_group(
                        self.peer_actor,
                        self.dual_rollout_device,
                        p_raw,
                        p_masks,
                        p_mem,
                        collect_stats=collect_stats,
                    )

            peer_act_future = self._peer_executor.submit(run_peer_actor)
            primary_mem = slice_memory(
                cur_mem, list(range(split_at)), device=self.device, detach=True, trim_gqa=False
            )
            primary_raw = all_raw[:split_at]
            primary_masks = all_masks[:split_at]
        else:
            primary_mem = cur_mem
            primary_raw = all_raw
            primary_masks = all_masks

        with torch.inference_mode():
            act_p, lp_p, next_p_mem, roll_stats = self._run_model_step_group(
                self.actor,
                self.device,
                primary_raw,
                primary_masks,
                primary_mem,
                collect_stats=collect_stats,
            )

        if peer_act_future is not None:
            act_r, lp_r, next_r_mem, peer_stats = peer_act_future.result()
            if collect_stats and roll_stats is not None and peer_stats is not None:
                for sk in roll_stats:
                    roll_stats[sk] += peer_stats[sk]
            act_np = np.concatenate([act_p, act_r], axis=0)
            lp_np = np.concatenate([lp_p, lp_r], axis=0)
            next_cur_mem = stack_memories([next_p_mem, next_r_mem], device=self.device)
        else:
            act_np, lp_np, next_cur_mem = act_p, lp_p, next_p_mem

        if use_wave_cur:
            self._wave_active_rows = rows_list
            self._wave_cur_opp_lanes = list(cur_opp_lanes)
            self._wave_cur_mem = next_cur_mem
            for pos, e in enumerate(cur_opp_lanes):
                actions[e, 1 - self.learner_seats[e]] = act_np[nactive + pos]
        else:
            next_bounds = next_cur_mem.temporal.positions.tolist() if cur_opp_lanes or nactive != envs else None
            next_learner = slice_memory(next_cur_mem, list(range(nactive)),
                device=self.device, detach=True, trim_gqa=False)
            if nactive == envs:
                self.lane_actor_mem = next_learner
            else:
                row_positions = {int(e): pos for pos, e in enumerate(rows)}
                old_bounds = self.lane_actor_mem.temporal.positions.tolist()
                self.lane_actor_mem = stack_memories([
                    slice_memory(next_learner, [row_positions[e]], device=self.device, detach=True, trim_gqa=True,
                        position_bound=next_bounds[row_positions[e]])
                    if e in row_positions else slice_memory(self.lane_actor_mem, [e], device=self.device, detach=True, trim_gqa=True,
                        position_bound=old_bounds[e])
                    for e in range(envs)], device=self.device)
            for pos, e in enumerate(cur_opp_lanes):
                self.lane_opp_mem[e] = slice_memory(next_cur_mem, [nactive + pos],
                    device=self.device, detach=True, trim_gqa=True, position_bound=next_bounds[nactive + pos])
                actions[e, 1 - self.learner_seats[e]] = act_np[nactive + pos]
        actions[rows, l_seats] = act_np[:nactive]
        learner_logp[rows] = lp_np[:nactive]

        if opp_future is not None:
            frozen_results = opp_future.result()
        elif frozen_groups:
            frozen_results = run_frozen_groups()
        else:
            frozen_results = {}

        for lane_list, seats, f_act in frozen_results.values():
            actions[lane_list, seats] = f_act

        seat_logp = np.zeros((envs, 2), dtype=np.float32)
        seat_logp[rows, l_seats] = learner_logp[rows]
        for pos, e in enumerate(cur_opp_lanes):
            seat_logp[e, 1-self.learner_seats[e]] = lp_np[nactive+pos]
        return actions, seat_logp, roll_stats

    def _evaluate_shared_values(self, segments: list[dict]) -> float:
        """Fallback evaluation for `--critic-mode shared`."""
        began = time.perf_counter()
        self.actor.eval()
        batch_size = max(int(getattr(self.args, "critic_batch_size", None) or self.args.batch_size), 8)
        chunk_size = int(getattr(self.args, "tbptt_chunk", 64))
        order = sorted(range(len(segments)), key=lambda i: len(segments[i]["raw_obs"]), reverse=True)
        with torch.inference_mode():
            for left in range(0, len(order), batch_size):
                group = order[left : left + batch_size]
                seqs, has_tail = [], []
                for idx in group:
                    tail = segments[idx].get("bootstrap_raw")
                    if tail is not None:
                        seqs.append(np.concatenate([segments[idx]["raw_obs"], tail[None]], axis=0))
                        has_tail.append(True)
                    else:
                        seqs.append(segments[idx]["raw_obs"])
                        has_tail.append(False)
                h0s = [
                    segments[idx]["h0_actor"]
                    if segments[idx].get("h0_actor") is not None
                    else self.actor.initial_state(1, device=self.device)
                    for idx in group
                ]
                state = stack_memories(h0s, device=self.device)
                max_len = max(len(x) for x in seqs)
                preds = [[] for _ in group]
                for start in range(0, max_len, chunk_size):
                    end = min(max_len, start + chunk_size)
                    span_max = end - start
                    raw_c = np.zeros((len(group), span_max, seqs[0].shape[-1]), dtype=np.int32)
                    act_m = np.zeros((len(group), span_max), dtype=bool)
                    for pos, sq in enumerate(seqs):
                        if start < len(sq):
                            seg_e = min(len(sq), end)
                            raw_c[pos, : seg_e - start] = sq[start:seg_e]
                            act_m[pos, : seg_e - start] = True
                    h, e, em = self.adapter(raw_c)
                    hand = raw_c[:, :, self.hand_indices]
                    with _amp_context(self.device, self.precision):
                        _, vals, state = self.actor.forward_state(
                            torch.as_tensor(h, device=self.device),
                            torch.as_tensor(e, device=self.device),
                            torch.as_tensor(em, device=self.device),
                            state=state,
                            hand_card_ids=torch.as_tensor(hand, device=self.device),
                            time_mask=torch.as_tensor(act_m, device=self.device),
                            compute_value=True,
                        )
                    state = self.actor.detach_state(state)
                    vals_np = vals.float().cpu().numpy()
                    for pos, sq in enumerate(seqs):
                        if start < len(sq):
                            seg_e = min(len(sq), end)
                            preds[pos].append(vals_np[pos, : seg_e - start])
                for pos, idx in enumerate(group):
                    fv = np.concatenate(preds[pos], axis=0) if preds[pos] else np.zeros(0, np.float32)
                    t_len = len(segments[idx]["raw_obs"])
                    segments[idx]["old_values"] = fv[:t_len].astype(np.float32)
                    if has_tail[pos]:
                        segments[idx]["bootstrap_value"] = float(fv[t_len])
        return time.perf_counter() - began

    def _new_current_segment(self, e):
        if self.opponents[e] is not None:
            return None
        seat = 1-int(self.learner_seats[e])
        trace = self.traces[e]
        prefix = (np.stack(trace.observations)[:, seat].copy() if trace.observations else
                  np.zeros((0, self.schema.seat_ints), dtype=np.int32))
        memory = clone_memory(self.lane_opp_mem[e], device=self.device, detach=True, trim_gqa=True) if len(prefix) else None
        h0_beh = memory if not self.actor_diverged else None
        h0_anc = None
        return _new_segment(int(self.episode_ids[e]), self.match_decks[e][1], self.match_decks[e][0],
            prefix_raw=prefix, h0_actor=memory, h0_behavior=h0_beh, h0_anchor=h0_anc,
            opponent_role='current')

    def collect(self, *, save_rollout_buffer: bool = False) -> dict[str, Any]:
        """Collect continuous rollout trajectories and evaluate critic baseline values."""
        args = self.args
        envs = args.envs
        curriculum = getattr(self, "sparky_curriculum", None)
        if curriculum is not None and save_rollout_buffer:
            raise ValueError("Sparky curriculum cannot write on-policy rollout buffers")
        sparky_stats = curriculum.new_stats() if curriculum is not None else None
        self.actor.eval()
        began = time.perf_counter()
        self._rollout_max_frozen_groups = 0
        army_diagnostics = dict(samples=0,unknown_lineage_count=0,mapped_member_count=0,
            budget_violation_count=0,army_phi_before_sum=0.,army_phi_after_sum=0.,
            army_value_own_sum=0.,army_value_enemy_sum=0.,seconds=0.)

        starts = [int(self.env.vector.info(i)["episode_step"]) for i in range(envs)]
        prefix_raw, h0_beh_list, h0_anc_list = self._burn_in_continuing_lanes()

        open_segs = [
            _new_segment(
                int(self.episode_ids[e]),
                self.match_decks[e][0],
                self.match_decks[e][1],
                prefix_raw=prefix_raw[e],
                h0_actor=(
                    slice_memory(self.lane_actor_mem, [e], device=self.device, detach=True, trim_gqa=True)
                    if len(prefix_raw[e]) > 0
                    else None
                ),
                h0_behavior=(
                    h0_beh_list[e]
                    if self.actor_diverged
                    else (
                        None
                        if len(prefix_raw[e]) == 0
                        else "ALIAS_ACTOR"
                    )
                ),
                h0_anchor=(h0_anc_list[e] if len(prefix_raw[e]) > 0 else None),
                opponent_role=('current' if self.opponents[e] is None else self.opponents[e].role),
            )
            for e in range(envs)
        ]
        for seg in open_segs:
            if seg["h0_behavior"] == "ALIAS_ACTOR":
                seg["h0_behavior"] = seg["h0_actor"]

        other_segs = [self._new_current_segment(e) for e in range(envs)]
        continuing_other = [e for e,s in enumerate(other_segs) if s is not None and len(s["prefix_raw"])]
        for model, dev, key, coef in (
            (self.behavior_model, self.device, "h0_behavior", self.args.reference_kl_coef if self.actor_diverged else 0),
            (self.anchor_model, self.opponent_device, "h0_anchor", self.args.anchor_kl_coef),
        ):
            if not continuing_other or coef <= 0:
                continue
            burned = burn_in_memory(model, self.adapter,
                [other_segs[e]["prefix_raw"] for e in continuing_other],
                legal_masks=[np.stack(self.traces[e].masks)[:,1-int(self.learner_seats[e])] for e in continuing_other],
                device=dev, precision=self.precision, batch_size=max(1,args.actor_batch_size or args.batch_size),
                chunk_size=args.tbptt_chunk)
            for e,memory in zip(continuing_other,burned,strict=True):
                other_segs[e][key] = clone_memory(memory,device=dev,detach=True,trim_gqa=True)
        finished_segs: list[dict] = []
        game_records: list[dict] = []
        faulted_episodes: set[int] = set()
        buf_rows: list[dict] | None = [] if save_rollout_buffer else None
        ep_starts = np.array([len(self.traces[e].observations) == 0 for e in range(envs)], dtype=bool)
        rollout_stat_accum = {
            "rows": 0,
            "entropy_sum": 0.0,
            "max_prob_sum": 0.0,
            "play_legal_rows": 0,
            "entropy_play_legal_sum": 0.0,
        }

        terminal_games = 0
        terminal_wins = 0
        faulted_samples = 0
        role_games = {"anchor": 0, "historical": 0, "current": 0}
        role_wins = {"anchor": 0, "historical": 0, "current": 0}
        role_score_sums = {"anchor": 0.0, "historical": 0.0, "current": 0.0}

        full_episode = bool(getattr(args, "full_episode_bptt", False))
        active = np.ones(envs, dtype=bool)
        if full_episode and (any(len(p) for p in prefix_raw) or any(starts)):
            raise RuntimeError("Full-episode collector cannot continue a previous policy version")
        if full_episode:
            self._wave_cur_mem_ready = True
            self._wave_active_rows = None
            self._wave_cur_opp_lanes = None
            self._wave_cur_mem = None
            self._wave_frozen_mem = {}
        else:
            self._wave_cur_mem_ready = False
            self._wave_frozen_mem = None
        primary_samples = 0
        model_step_seconds = env_step_seconds = 0.0
        step = -1
        while (active.any() if full_episode else step + 1 < args.steps):
            step += 1
            if step % 64 == 0:
                self._flush_async_checkpoint()
                if self._publisher is not None:
                    self._publisher.check()

            raw = self.env.observations.copy()
            masks = self.env.masks.copy()
            collect_step_stats = (step % 64 == 0) or (step == args.steps - 1)
            model_step_began = time.perf_counter()
            if curriculum is not None:
                native_masks = masks
                masks, sparky_choices = curriculum.prepare(
                    raw, masks, active, self.learner_seats, sparky_stats
                )
                collect_step_stats &= masks is native_masks
            actions, old_logp, step_stats = self._step_models(
                raw, masks, collect_stats=collect_step_stats,
                **({"active_lanes": np.flatnonzero(active)} if full_episode else {}),
            )
            excluded = (
                curriculum.record_choices(sparky_choices, actions, raw, sparky_stats)
                if curriculum is not None else ()
            )
            model_step_seconds += time.perf_counter() - model_step_began
            if step_stats is not None:
                for k, v in step_stats.items():
                    rollout_stat_accum[k] += v

            env_step_began = time.perf_counter()
            if full_episode and hasattr(self.env.vector, "step_lanes"):
                self.env.actions[:] = actions
                self.env.vector.step_lanes(np.flatnonzero(active))
            else:
                self.env.step(actions)
            env_step_seconds += time.perf_counter() - env_step_began
            following = self.env.observations.copy()
            next_masks = self.env.masks.copy() if save_rollout_buffer else None
            raw_reward = self.env.rewards.copy().astype(np.float32)
            done = self.env.dones.astype(bool).copy() & active
            infos = [self.env.vector.info(i) for i in range(envs)]
            fault_codes = np.array([int(info.get("fault", 0) or 0) for info in infos], dtype=np.int32)
            faulted = fault_codes != 0
            truncated = done & (
                np.array([bool(info.get("truncated", 0)) for info in infos], dtype=bool) | faulted
            )
            terminated = done & ~truncated
            if curriculum is not None:
                curriculum.record_followups(following, terminated, truncated | faulted, sparky_stats)
            army_values=None
            if getattr(self,'army_elixir',None) is not None:
                army_started=time.perf_counter()
                before_army=self.army_phi.copy()
                after_army,diagnostics=self.army_elixir.poll(np.flatnonzero(active & ~faulted))
                self.army_phi[active & ~faulted]=after_army[active & ~faulted]
                army_values=(before_army,self.army_phi.copy())
                for row in diagnostics:
                    army_diagnostics['samples']+=1
                    for key in ('unknown_lineage_count','mapped_member_count','fallback_low_count',
                            'fallback_high_count','fallback_cap_hit_count','still_unknown_count',
                            'fallback_value_own','fallback_value_enemy'):
                        if key in row: army_diagnostics[key]=army_diagnostics.get(key,0)+row[key]
                    army_diagnostics['budget_violation_count']=max(
                        army_diagnostics['budget_violation_count'],row['budget_violation_count'])
                    army_diagnostics['army_value_own_sum']+=row['army_value_own']
                    army_diagnostics['army_value_enemy_sum']+=row['army_value_enemy']
                army_diagnostics['army_phi_before_sum']+=float(before_army[:,0].sum())*self.rewards.army_elixir_beta
                army_diagnostics['army_phi_after_sum']+=float(np.where(terminated,0,self.army_phi[:,0]).sum())*self.rewards.army_elixir_beta
                army_diagnostics['seconds']+=time.perf_counter()-army_started
            shaped, comp = shaped_rewards(
                raw,
                following,
                actions,
                raw_reward,
                terminated,
                self.schema,
                self.gamma,
                self.rewards,
                faulted=faulted,
                army_values=army_values,
            )

            learner_mask = np.zeros((envs, 2), dtype=np.uint8)
            rows = np.arange(envs)
            learner_mask[rows, self.learner_seats] = 1
            for e in range(envs):
                if self.opponents[e] is None:
                    learner_mask[e, :] = 1
            valid_step = ~faulted & active

            game_records.append(
                {
                    "episode_ids": self.episode_ids.copy(),
                    "episode_start": ep_starts.copy(),
                    "valid": np.repeat(valid_step[:, None], 2, axis=1),
                    "learner_mask": learner_mask,
                    "terminated": terminated.astype(np.uint8),
                    "truncated": truncated.astype(np.uint8),
                    "fault_codes": fault_codes,
                }
            )
            ep_starts[:] = False

            if buf_rows is not None:
                versions = np.full((envs, 2), self.update, dtype=np.int32)
                for e, entry in enumerate(self.opponents):
                    if entry is not None:
                        versions[e, 1 - self.learner_seats[e]] = frozen_policy_version(entry)
                on_policy_seats(versions, self.update, self.learner_seats)
                buf_rows.append(
                    {
                        "obs": raw,
                        "masks": masks,
                        "actions": actions.astype(np.int32),
                        "rewards": raw_reward,
                        "training_rewards": shaped,
                        "next_obs": following,
                        "next_masks": next_masks,
                        "dones": done.astype(np.uint8),
                        "terminated": terminated.astype(np.uint8),
                        "truncated": truncated.astype(np.uint8),
                        "old_log_probs": np.zeros((envs, 2), dtype=np.float32),
                        "old_values": np.zeros((envs, 2), dtype=np.float32),
                        "learner_mask": learner_mask,
                        "policy_ids": versions,
                        "policy_versions": versions.copy(),
                        "opponent_ids": versions[:, ::-1].copy(),
                        "episode_ids": self.episode_ids.copy(),
                        "episode_start": game_records[-1]["episode_start"],
                        "episode_steps": np.array([info["episode_step"] for info in infos]),
                        "ticks": np.array([info["tick"] for info in infos]),
                        "winner": np.array([infos[i]["winner"] if terminated[i] else -1 for i in range(envs)]),
                        "end_reason": np.where(faulted, 3, np.where(truncated, 2, terminated.astype(int))),
                        "fault_codes": fault_codes,
                        "seeds": self.seeds.copy(),
                        "action_consumed": np.ones((envs, 2), dtype=np.uint8),
                        "action_refusals": following[..., self.schema.header["last_refusal"][0]],
                        "run_refusals": following[..., self.schema.header["last_run_refusal"][0]],
                    }
                )
                buf_rows[-1]["old_log_probs"][:] = old_logp

            for e in np.flatnonzero(active):
                primary_samples += 1
                seat = int(self.learner_seats[e])
                trace = self.traces[e]
                # Complete segments already own every learner frame. Raw prefix
                # traces are needed only when carrying games across updates;
                # their multi-lane views otherwise retain entire parked batches.
                if not full_episode:
                    trace.append(raw[e], masks[e], shaped[e], learner_mask[e], copy=False)
                seat_segments = [(seat, open_segs[e])]
                if other_segs[e] is not None:
                    seat_segments.append((1-seat, other_segs[e]))
                for seat, seg in seat_segments:
                    if (e, seat) in excluded:
                        seg.setdefault("_actor_excluded", []).append(len(seg["_raw_obs"]))
                    seg["_raw_obs"].append(raw[e, seat].copy())
                    seg["_legal_masks"].append(masks[e, seat].copy())
                    seg["_actions"].append(int(actions[e, seat]))
                    seg["_old_log_probs"].append(float(old_logp[e, seat]))
                    seg["_rewards"].append(float(shaped[e, seat]))
                    seg["_raw_rewards"].append(float(raw_reward[e, seat]))
                    seg["_reward_hp"].append(float(comp["hp"][e, seat]))
                    seg["_reward_crown"].append(float(comp["crown"][e, seat]))
                    seg["_reward_overflow"].append(float(comp["overflow"][e, seat]))
                    seg["_reward_army"].append(float(comp["army"][e, seat]))
                    seg["_terminated"].append(bool(terminated[e]))
                    seg["_truncated"].append(bool(truncated[e]))
                    seg["_valid"].append(bool(valid_step[e]))
                if faulted[e]:
                    faulted_episodes.add(int(self.episode_ids[e]))

            if done.any():
                for e in np.flatnonzero(done):
                    ei = int(e)
                    seat = int(self.learner_seats[ei])
                    ep_id = int(self.episode_ids[ei])
                    self.traces[ei].finish(terminated=bool(terminated[ei]), faulted=bool(faulted[ei]))
                    if faulted[ei]:
                        open_segs[ei]["_valid"] = [False] * len(open_segs[ei]["_valid"])
                    elif truncated[ei]:
                        open_segs[ei]["bootstrap_raw"] = following[ei, seat].copy()
                    elif terminated[ei]:
                        terminal_games += 1
                        r_term = float(raw_reward[ei, seat])
                        if curriculum is not None:
                            curriculum.record_terminal(ei, r_term, sparky_stats)
                        terminal_wins += int(r_term > 0)
                        score_val = float(np.clip(round(r_term), -1, 1))
                        entry = self.opponents[ei]
                        opp_role = (
                            "current"
                            if entry is None
                            else str(getattr(entry, "role", "historical") or "historical")
                        )
                        if opp_role not in role_games:
                            opp_role = "historical"
                        role_games[opp_role] += 1
                        role_wins[opp_role] += int(r_term > 0)
                        role_score_sums[opp_role] += 1.0 if r_term > 0 else (0.5 if r_term == 0 else 0.0)
                        if opp_role == "anchor":
                            self._record_deck_vs_bc(
                                self.match_decks[ei][0], self.match_decks[ei][1], r_term
                            )
                        if entry is not None:
                            self.pool.record(
                                entry.policy_id,
                                score_val,
                                own_deck=self.match_decks[ei][0],
                                enemy_deck=self.match_decks[ei][1],
                                learner_version=self.update,
                            )
                        if self.admission.purpose == "promotion" and r_term in (-1.0, 0.0, 1.0):
                            self.admission.record(int(r_term))
                        self.emit(
                            phase="episode",
                            update=self.update + 1,
                            learner_samples_total=self.samples + primary_samples * self.world_size,
                            reward=r_term,
                            episode_length=int(infos[ei]["episode_step"]),
                            learner_deck=self.match_decks[ei][0],
                            opponent_deck=self.match_decks[ei][1],
                            opponent_role=opp_role,
                            fault=int(fault_codes[ei]),
                            truncated=int(truncated[ei]),
                        )

                    if other_segs[ei] is not None:
                        other = other_segs[ei]
                        if faulted[ei]:
                            other["_valid"] = [False] * len(other["_valid"])
                        elif truncated[ei]:
                            other["bootstrap_raw"] = following[ei, 1-seat].copy()
                        finished_segs.append(_finalize_segment(other))
                    finished_segs.append(_finalize_segment(open_segs[ei]))
                    if full_episode:
                        active[ei] = False
                        # Closed lanes remain parked until every game in this policy batch ends.
                        open_segs[ei] = None
                        other_segs[ei] = None
                        continue
                    self._reset_match(ei)
                    other_segs[ei] = self._new_current_segment(ei)
                    ep_starts[ei] = True
                    open_segs[ei] = _new_segment(
                        int(self.episode_ids[ei]),
                        self.match_decks[ei][0],
                        self.match_decks[ei][1],
                        prefix_raw=np.zeros((0, self.schema.seat_ints), dtype=np.int32),
                        h0_actor=None,
                        h0_behavior=None,
                        h0_anchor=None,
                        opponent_role=('current' if self.opponents[ei] is None else self.opponents[ei].role),
                    )

        # Finalize still-open tail segments with live post-step observation as bootstrap target.
        tail_obs = self.env.observations.copy()
        for e in range(envs):
            seg = open_segs[e]
            if seg is not None and seg["_raw_obs"]:
                if self.episode_ids[e] in faulted_episodes:
                    seg["_valid"] = [False] * len(seg["_valid"])
                seg["bootstrap_raw"] = tail_obs[e, int(self.learner_seats[e])].copy()
                finished_segs.append(_finalize_segment(seg))
            other = other_segs[e]
            if other is not None and other["_raw_obs"]:
                other["bootstrap_raw"] = tail_obs[e, 1-int(self.learner_seats[e])].copy()
                finished_segs.append(_finalize_segment(other))
            if self.traces[e] is not None:
                self.traces[e].detach_views()

        if full_episode:
            self._wave_cur_mem_ready = False
            self._wave_active_rows = None
            self._wave_cur_opp_lanes = None
            self._wave_cur_mem = None
            self._wave_frozen_mem = None
            for seg in finished_segs:
                seg["full_episode"] = True
                seg["behavior_version"] = self.update
            for e in range(envs):
                self._reset_match(e)

        for seg in finished_segs:
            if seg["episode_id"] in faulted_episodes:
                seg["valid"][:] = False
            faulted_samples += int((~seg["valid"]).sum())

        rollout_step_seconds = time.perf_counter() - began
        self._trim_opponent_cache(self.training_opponent_cache_size)

        if self.critic is not None:
            _, critic_eval_seconds = evaluate_critic_segments(
                self.critic,
                self.adapter,
                finished_segs,
                device=self.critic_device,
                precision=self.precision,
                batch_size=max(int(getattr(args, "critic_batch_size", None) or args.batch_size), 8),
                chunk_size=int(getattr(args, "tbptt_chunk", 64)),
                peer_critic=self.peer_critic,
                peer_device=self.critic_peer_device,
            )
        else:
            critic_eval_seconds = self._evaluate_shared_values(finished_segs)

        # A scheduled auxiliary phase still needs the real recurrent history.
        next_update = self.update + 1
        keep_aux_prefix = (
            (getattr(self, 'awbc_buffer', None) is not None and next_update % args.awbc_interval == 0)
            or (getattr(self, 'expert_bc', None) is not None and next_update % args.expert_bc_interval == 0)
            or self._macro_search_due(next_update)
        )
        for seg in finished_segs:
            if not keep_aux_prefix:
                seg.pop("prefix_raw", None)
            seg.pop("bootstrap_raw", None)

        ends = [int(self.env.vector.info(i)["episode_step"]) for i in range(envs)]
        completed_games_new = self.game_counter.record(game_records)

        return {
            "segments": finished_segs,
            "rollout_seconds": time.perf_counter() - began,
            "rollout_step_seconds": rollout_step_seconds,
            "rollout_model_seconds": model_step_seconds,
            "rollout_env_seconds": env_step_seconds,
            "rollout_bookkeeping_seconds": max(0., rollout_step_seconds-model_step_seconds-env_step_seconds),
            "critic_eval_seconds": critic_eval_seconds,
            "starts": starts,
            "ends": ends,
            "terminal_games": terminal_games,
            "terminal_wins": terminal_wins,
            "role_terminal_stats": {
                "games": role_games,
                "wins": role_wins,
                "score_sums": role_score_sums,
            },
            "faulted_samples": faulted_samples,
            "completed_games_new": completed_games_new,
            "step_policy_stats": rollout_stat_accum,
            "army_elixir_diagnostics": army_diagnostics if getattr(self,'army_elixir',None) is not None else {},
            **({"sparky_curriculum": sparky_stats} if sparky_stats is not None else {}),
            "buffer_rows": buf_rows,
            "max_frozen_groups": self._rollout_max_frozen_groups,
            "secondary_samples": sum(len(s["raw_obs"]) for s in finished_segs)-primary_samples,
            "full_episode_bptt": full_episode,
            "max_episode_frames": max((len(s["raw_obs"]) for s in finished_segs), default=0),
            "deck_sample_counts": dict(Counter({name: sum(len(s["raw_obs"]) for s in finished_segs if s["own_deck"]==name)
                for name in {s["own_deck"] for s in finished_segs}})),
        }

    def _build_ppo_config(self) -> StatePPOConfig:
        args = self.args
        base_batch = int(args.batch_size)
        base_eff = int(args.batch_size * args.grad_accumulation)
        return StatePPOConfig(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            critic_gae_lambda=self.critic_gae_lambda,
            clip_coef=float(args.clip_coef),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            card_entropy_coef=getattr(args, "card_entropy_coef", None),
            tile_entropy_coef=getattr(args, "tile_entropy_coef", None),
            reference_kl_coef=float(args.reference_kl_coef),
            anchor_kl_coef=float(getattr(args, "anchor_kl_coef", 0.0) or 0.0),
            target_kl=float(args.target_kl),
            target_card_kl=getattr(args, "target_card_kl", None),
            target_tile_kl=getattr(args, "target_tile_kl", None),
            continue_after_kl_reject=bool(getattr(args, "continue_after_kl_reject", True)),
            actor_choice_only=bool(
                True if getattr(args, "actor_choice_only", None) is None else args.actor_choice_only
            ),
            epochs=int(args.epochs),
            critic_epochs=int(getattr(args, "critic_epochs", args.epochs)),
            actor_batch_size=int(getattr(args, "actor_batch_size", None) or base_batch),
            critic_batch_size=int(getattr(args, "critic_batch_size", None) or base_batch),
            actor_effective_batch_size=int(getattr(args, "actor_effective_batch_size", None) or base_eff),
            critic_effective_batch_size=int(
                getattr(args, "critic_effective_batch_size", None) or base_eff
            ),
            full_episode_bptt=bool(getattr(args, "full_episode_bptt", False)),
            tbptt_chunk=int(getattr(args, "tbptt_chunk", 64)),
            max_grad_norm=float(args.max_grad_norm),
            critic_max_grad_norm=float(getattr(args, "critic_max_grad_norm", None) or args.max_grad_norm),
            max_initial_logp_error=(
                0.05
                if getattr(args, "max_initial_logp_error", None) is None
                else args.max_initial_logp_error
            ),
            precision=self.precision,
            cache_reference=bool(getattr(args, "cache_reference", False)),
        )

    def _mandatory_pool_ids(self) -> set[int]:
        """Return pool IDs pinned by active matches or anchor role."""
        pool_ids = {entry.policy_id for entry in self.pool.entries}
        live_ids = {
            entry.policy_id
            for entry in self.opponents
            if entry is not None and entry.policy_id in pool_ids
        }
        anchor_ids = {entry.policy_id for entry in self.pool.entries if entry.role == "anchor"}
        return live_ids | anchor_ids

    def _update_opponent_pool(self, actor_steps: int) -> None:
        """Admit a new snapshot only if pool capacity accommodates live matches and prune safely."""
        capacity = self.pool.max_entries
        if (self.rank == 0 and actor_steps > 0 and self.admission.ready(self.samples)
                and getattr(self, "out", None) is not None and self.out.exists()
                and shutil.disk_usage(self.out).free < 9 * 1024**3):
            # An 800-update checkpoint interval must not let retired history
            # files fill the volume before the next full save. Keep recent
            # restore points; release optional old periodic points first.
            retention = prune_backup(self.out, self.update,
                recent=int(getattr(self.args, "keep_checkpoints", 2)), max_periodic=0,
                rollout_recent=int(getattr(self.args, "keep_rollouts", 2)))
            retention.update(prune_local_opponents(self.out,
                active=[e.checkpoint for e in self.opponents if e is not None]))
            self.emit(phase="storage_headroom", update=self.update, **retention)
        if actor_steps > 0 and self.admission.ready(self.samples) and self.rank == 0:
            mandatory = self._mandatory_pool_ids()
            # Defer admission rather than hitting OpponentPool.prune's protected-capacity error
            # when all slots are simultaneously pinned across concurrent rollout envs.
            if capacity is None or (len(mandatory) + 1) <= capacity:
                snapshot_id = next_history_snapshot_id(self.pool.entries, self.update)
                opp_path = self.save_opponent(snapshot_id)
                if all(
                    torch.isfinite(p).all()
                    for p in self.actor.state_dict().values()
                    if isinstance(p, torch.Tensor)
                ):
                    role = "historical" if self.role_league else "main"
                    self.pool.add(
                        OpponentEntry(
                            snapshot_id,
                            str(opp_path.resolve()),
                            self.update,
                            deck_id="multi_deck",
                            role=role,
                        )
                    )
                    self.admission.admitted(self.samples)
                else:
                    self.admission.last_failed_candidate_samples = self.samples
                    opp_path.unlink(missing_ok=True)

        pool_ids = {entry.policy_id for entry in self.pool.entries}
        live_ids = [
            entry.policy_id
            for entry in self.opponents
            if entry is not None and entry.policy_id in pool_ids
        ]
        latest_ids = (
            {max(self.pool.entries, key=lambda entry: entry.generation).policy_id}
            if self.pool.entries
            else set()
        )
        if capacity is not None and len(self._mandatory_pool_ids() | latest_ids) > capacity:
            evicted = []
        else:
            evicted = self.pool.prune(live_ids, learner_version=self.update)
        if evicted and self.rank == 0:
            with (self.out / "archived_opponents.jsonl").open("a", encoding="utf-8") as stream:
                for entry in evicted:
                    stream.write(json.dumps(entry.__dict__) + "\n")
        if self.rank == 0:
            self.pool.save(self.out / "opponent_pool.json")
            if evicted:
                retention = prune_local_opponents(self.out,
                    active=[e.checkpoint for e in self.opponents if e is not None])
                self.emit(phase="opponent_storage_retention", update=self.update, **retention)

    def save_opponent(self, policy_id: int, actor_cpu: dict | None = None) -> Path:
        target = self.out / "opponents" / f"opponent_{policy_id:06d}.pt"
        tmp = target.with_suffix(".tmp.pt")
        meta = self.env.metadata()
        weights = _model_state_refs(self.actor) if actor_cpu is None else actor_cpu
        streaming_torch_save(
            {
                "format": "crrl-state-actor-v1",
                "model_architecture_version": 3,
                "decision_ticks": 8,
                "config": self.actor.cfg.__dict__,
                "temporal_config": self.actor.temporal_cfg.__dict__,
                "learned_placement_definition": getattr(
                    self.actor, "learned_placement_definition", None
                ),
                "model": weights,
                "policy_id": policy_id,
                "updates": self.update,
                "backend_commit": meta.get("commit"),
                "data_version": self.env.vector.data_version,
                "content_sha": self.env.vector.content_sha,
                "level_profile": self.level_profile,
            },
            tmp,
            fsync=False,
        )
        del weights
        tmp.replace(target)
        release_host_memory()
        return target

    def checkpoint(self) -> Path | None:
        if self.rank != 0:
            return None
        self._flush_async_checkpoint(wait=True)
        actor_sd = _model_state_refs(self.actor)
        actor_opt_sd = self.actor_optimizer.state_dict()
        actor_scaler = self.actor_optimizer.amp_scaler.state_dict()
        critic_sd = _model_state_refs(self.critic) if self.critic is not None else None
        critic_opt_sd = (
            self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None
        )
        meta = self.env.metadata()
        payload = {
            "format": "crrl-state-ppo-v1",
            "training_driver": "persistent-state-v3",
            "model_architecture_version": 3,
            "decision_ticks": 8,
            "critic_mode": self.critic_mode,
            "config": self.actor.cfg.__dict__,
            "temporal_config": self.actor.temporal_cfg.__dict__,
            "learned_placement_definition": getattr(
                self.actor, "learned_placement_definition", None
            ),
            "model": actor_sd,
            "actor": actor_sd,
            "optimizer": actor_opt_sd,
            "actor_optimizer": actor_opt_sd,
            "scaler": actor_scaler,
            "actor_amp_scaler": actor_scaler,
            "critic": critic_sd,
            "critic_config": self.critic.cfg.__dict__ if self.critic is not None else None,
            "critic_temporal_config": (
                self.critic.temporal_cfg.__dict__ if self.critic is not None else None
            ),
            "critic_optimizer": critic_opt_sd,
            "critic_amp_scaler": (
                self.critic_optimizer.amp_scaler.state_dict()
                if self.critic_optimizer is not None
                else None
            ),
            "adaptive_actor_lr": (
                self.actor_lr_controller.state_dict() if self.actor_lr_controller else None
            ),
            "progress": {
                "update": self.update,
                "learner_samples_total": self.samples,
                "completed_games": self.game_counter.count,
            },
            "updates": self.update,
            "learner_samples_total": self.samples,
            "optimizer_steps_total": self.optimizer_steps_total,
            "critic_steps_total": self.critic_steps_total,
            "training_game_counter": self.game_counter.state_dict(),
            "admission": self.admission.state_dict(),
            "role_mixture": self.role_mixture.state_dict(),
            "role_draws": dict(self.role_draws),
            "league_options": {
                "role_league": self.role_league,
                "pool_capacity": self.pool.max_entries,
                "history_snapshot_interval": int(getattr(self.args, 'history_snapshot_interval', 20_000)),
                "snapshot_interval": int(getattr(self.args, 'snapshot_interval', 250_000)),
            },
            "awbc_playbuffer": self.awbc_buffer.state if getattr(self, 'awbc_buffer', None) else None,
            "expert_bc": self.expert_bc.state_dict() if getattr(self, 'expert_bc', None) else None,
            "macro_search": self.macro_search.state_dict() if getattr(self, 'macro_search', None) else None,
            "lookahead_distill": self.lookahead_distiller.state_dict() if getattr(self, 'lookahead_distiller', None) else None,
            "next_episode": self.next_episode,
            "reward_config": self.rewards.metadata(),
            "league_config": self.league,
            "deck_pool_transitions": self.deck_pool_transitions,
            "deck_vs_bc": copy.deepcopy(getattr(self, "_deck_vs_bc", None)),
            "resume_learning_rates": self.resume_learning_rates,
            "bc_anchor": str(self.anchor_path),
            "backend_commit": meta.get("commit"),
            "data_version": self.env.vector.data_version,
            "content_sha": self.env.vector.content_sha,
            "level_profile": self.level_profile,
            "level_profile_migrated": self.level_profile_migrated,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "numpy_rng_state": self.rng.bit_generator.state,
            "training_args": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()
            },
        }
        from cr_rl.training.async_checkpoint import (
            AsyncCheckpointWriter, freeze_cpu_state, write_checkpoint_files,
        )
        pool = json.loads((self.out / "opponent_pool.json").read_text())
        if getattr(self.args, "async_checkpoint", False):
            began = time.perf_counter()
            frozen = freeze_cpu_state(payload)
            writer = getattr(self, "_checkpoint_writer", None)
            if writer is None:
                writer = self._checkpoint_writer = AsyncCheckpointWriter()
            writer.submit(self.out, frozen, pool, self.update)
            self._pending_checkpoint = dict(update=self.update, publish=False)
            self.emit(update=self.update, phase="checkpoint_staged",
                      cpu_snapshot_seconds=time.perf_counter() - began)
        else:
            write_checkpoint_files(self.out, payload, pool, self.update)
            self._last_checkpoint_update = self.update
            release_host_memory()
        return self.out / "ppo_latest.pt"

    def _flush_async_checkpoint(self, *, wait=False):
        pending = getattr(self, "_pending_checkpoint", None)
        if pending is None:
            return
        writer = self._checkpoint_writer
        if not wait and not writer.ready():
            return
        result = writer.wait()
        self._pending_checkpoint = None
        self._last_checkpoint_update = result['update']
        retention = prune_backup(self.out, result['update'],
            recent=int(getattr(self.args, "keep_checkpoints", 2)),
            rollout_recent=int(getattr(self.args, "keep_rollouts", 2)),
            max_periodic=getattr(self.args, "keep_periodic_checkpoints", 2))
        retention.update(prune_local_opponents(self.out,
            active=[entry.checkpoint for entry in self.opponents if entry is not None]))
        self.emit(phase="checkpoint_saved", **result, **retention)
        if 'training_win_rates' in pending:
            self._maybe_save_best_train_traj(pending['training_win_rates'], update=result['update'])
        if pending['publish']:
            self._maybe_publish_backup(True, update=result['update'])

    def emit(self, **event) -> None:
        if self.rank != 0:
            return
        if self.telemetry is not None:
            self.telemetry.log(event)
        progress_path = getattr(self.args, "progress_jsonl", None) or (self.out / "progress.jsonl")
        progress_path = Path(progress_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def _maybe_expert_bc_phase(self, segments) -> dict:
        bc = getattr(self, 'expert_bc', None)
        if bc is None or self.update % self.args.expert_bc_interval:
            return {'ran': False}
        health = None
        if getattr(bc, 'bc_plan', None) == 'sparky_b':
            from cr_rl.training.expert_bc_guard import plan_b_health

            h_kw = {} if self.actor.cfg.reserve_to_wait else {"min_entropy": 0.40, "max_anchor_kl": 1.80}
            health = plan_b_health(self.out, self.update, **h_kw)
            if not health['passed']:
                result = {'ran': False, 'reason': 'health_circuit_breaker', 'ppo_health': health}
                atomic_json(self.out/'expert_bc_latest_report.json', dict(update=self.update, **result))
                self.emit(phase='expert_bc', update=self.update, **result)
                return result
        result = bc.run_phase(self.actor, self.adapter, select_awbc_probes(segments),
            update=self.update, device=self.device, precision=self.precision,
            sequences=self.args.expert_bc_sequences,
            lr=self.args.expert_bc_learning_rate, max_kl=self.args.expert_bc_max_kl)
        if result.get('memory_invalidated'):
            for name, param in self.actor.named_parameters():
                if name in result.get('changed_parameters', []):
                    self.actor_optimizer.state.pop(param, None)
            self.actor_weights_dirty = True
            self._sync_behavior_model()
            if self.peer_actor is not None:
                self._sync_peer_actor()
        if health is not None:
            result['ppo_health'] = health
        atomic_json(self.out/'expert_bc_latest_report.json', dict(update=self.update, **result))
        self.emit(phase='expert_bc', update=self.update, **result)
        return result

    def _macro_search_due(self, update: int) -> bool:
        return (getattr(self, 'macro_search', None) is not None
                and update % self.args.macro_search_interval == getattr(self.args, 'macro_search_offset', 0))

    def _maybe_macro_search_phase(self, segments) -> dict:
        if not self._macro_search_due(self.update):
            return {'ran': False}
        ms = self.macro_search
        from cr_rl.training.expert_bc_guard import plan_b_health

        h_kw = {} if self.actor.cfg.reserve_to_wait else {"min_entropy": 0.40, "max_anchor_kl": 1.80}
        health = plan_b_health(self.out, self.update, **h_kw)
        if not health['passed']:
            result = {'ran': False, 'reason': 'health_circuit_breaker', 'ppo_health': health}
            atomic_json(self.out / 'macro_search_latest_report.json', dict(update=self.update, **result))
            self.emit(phase='macro_search', update=self.update, **result)
            return result
        result = ms.run_phase(
            self.actor,
            self.adapter,
            select_awbc_probes(segments),
            env_paths=self.env.paths,
            decks=self.decks,
            forms=self.forms,
            deck_names=self.deck_names,
            deck_probabilities=self.deck_probabilities,
            update=self.update,
            device=self.device,
            precision=self.precision,
            candidates=getattr(self.args, 'macro_search_candidates', 2),
            win_margin=getattr(self.args, 'macro_search_win_margin', 0.30),
            batch_episodes=getattr(self.args, 'macro_search_batch_episodes', 8),
            lr=getattr(self.args, 'macro_search_learning_rate', 1e-6),
            max_kl=getattr(self.args, 'macro_search_max_kl', 0.005),
            focal_deck_indices=(
                self.focal_deck_indices
                if getattr(self, 'focal_deck_indices', None) and getattr(self.args, 'focal_deck_prob', 0.0) > 0
                else None
            ),
        )
        if result.get('memory_invalidated'):
            for name, param in self.actor.named_parameters():
                if name in result.get('changed_parameters', []):
                    self.actor_optimizer.state.pop(param, None)
            self.actor_weights_dirty = True
            self._sync_behavior_model()
            if self.peer_actor is not None:
                self._sync_peer_actor()
        result['ppo_health'] = health
        atomic_json(self.out / 'macro_search_latest_report.json', dict(update=self.update, **result))
        self.emit(phase='macro_search', update=self.update, **result)
        return result

    def _lookahead_distill_due(self, update: int) -> bool:
        return (
            getattr(self, 'lookahead_distiller', None) is not None
            and update % getattr(self.args, 'lookahead_distill_interval', 3) == getattr(self.args, 'lookahead_distill_offset', 1)
        )

    def _maybe_lookahead_distill_phase(self, segments) -> dict:
        if not self._lookahead_distill_due(self.update):
            return {'ran': False}
        ld = self.lookahead_distiller
        from cr_rl.training.expert_bc_guard import plan_b_health

        h_kw = {} if self.actor.cfg.reserve_to_wait else {"min_entropy": 0.40, "max_anchor_kl": 1.80}
        health = plan_b_health(self.out, self.update, **h_kw)
        if not health['passed']:
            result = {'ran': False, 'reason': 'health_circuit_breaker', 'ppo_health': health}
            atomic_json(self.out / 'lookahead_distill_latest_report.json', dict(update=self.update, **result))
            self.emit(phase='lookahead_distill', update=self.update, **result)
            return result
        result = ld.run_phase(
            self.actor,
            self.critic,
            self.adapter,
            select_awbc_probes(segments),
            env_paths=self.env.paths,
            decks=self.decks,
            forms=self.forms,
            deck_names=self.deck_names,
            deck_probabilities=self.deck_probabilities,
            update=self.update,
            device=self.device,
            precision=self.precision,
            games=getattr(self.args, 'lookahead_distill_games', 3),
            parallel_games=getattr(self.args, 'lookahead_distill_parallel_games', 12),
            max_parallel_searches=getattr(self.args, 'lookahead_distill_parallel_searches', 4),
            candidates=getattr(self.args, 'lookahead_distill_candidates', 4),
            rollouts=getattr(self.args, 'lookahead_distill_rollouts', 3),
            horizon_steps=getattr(self.args, 'lookahead_distill_horizon', 10),
            trigger_elixir=getattr(self.args, 'lookahead_distill_trigger_elixir', 6.0),
            max_searches_per_game=getattr(self.args, 'lookahead_distill_max_searches', 16),
            batch_episodes=getattr(self.args, 'lookahead_distill_batch_episodes', 4),
            optimizer_steps=getattr(self.args, 'lookahead_distill_steps', 1),
            override_margin=getattr(self.args, 'lookahead_distill_margin', 0.008),
            lr=getattr(self.args, 'lookahead_distill_learning_rate', 1e-6),
            max_kl=getattr(self.args, 'lookahead_distill_max_kl', 0.005),
            threads=(
                int(getattr(self.args, 'lookahead_distill_threads', 0) or 0)
                or min(44, max(8, (os.cpu_count() or 12) - 4))
            ),
            focal_deck_indices=(
                self.focal_deck_indices
                if getattr(self, 'focal_deck_indices', None) and getattr(self.args, 'focal_deck_prob', 0.0) > 0
                else None
            ),
        )
        if result.get('memory_invalidated'):
            for name, param in self.actor.named_parameters():
                if name in result.get('changed_parameters', []):
                    self.actor_optimizer.state.pop(param, None)
            self.actor_weights_dirty = True
            self._sync_behavior_model()
            if self.peer_actor is not None:
                self._sync_peer_actor()
        result['ppo_health'] = health
        atomic_json(self.out / 'lookahead_distill_latest_report.json', dict(update=self.update, **result))
        self.emit(phase='lookahead_distill', update=self.update, **result)
        return result

    def _maybe_awbc_phase(self, segments=None) -> dict:
        buffer = getattr(self, 'awbc_buffer', None)
        if (getattr(self.args, 'require_awbc_smoke', False)
                and getattr(self, '_awbc_smoke_report', {}).get('actor_attempt_update') is not None):
            return {'ran': False, 'reason': 'smoke_followup_ppo_and_bc'}
        if (buffer is None or self.update % self.args.awbc_interval
                or buffer.state['last_update'] >= self.update):
            return {'ran': False}
        refresh = buffer.refresh()
        atomic_json(self.out/'awbc_buffer_audit.json', refresh)
        source_ids = sorted(key for key, row in buffer.state['entries'].items() if row.get('active'))
        skipped = buffer.state.get('last_skipped_batch', {})
        if skipped.get('source_ids') == source_ids:
            self.emit(phase='awbc_playbuffer', update=self.update, ran=False,
                reason='unchanged_skipped_source', previous_reason=skipped.get('reason'),
                known_games=refresh['known_games'], active_games=len(source_ids),
                added_games=refresh['added_games'])
            return {'ran': False, 'actor_attempted': False, 'reason': 'unchanged_skipped_source'}
        selected = buffer.select(self.args.awbc_min_new_games, self.args.awbc_batch_games)
        if not selected:
            self.emit(phase='awbc_playbuffer', update=self.update, ran=False,
                      reason='insufficient_new_or_validation_games',
                      added_games=refresh['added_games'], known_games=refresh['known_games'])
            return {'ran': False, 'actor_attempted': False, 'reason': 'insufficient_new_or_validation_games'}
        episodes=[]
        for identity in selected:
            row=buffer.state['entries'][identity];run=Path(row['run'])
            from cr_rl.client.rl_recording import sha256_file
            if sha256_file(run/'manifest.json')+sha256_file(run/'result.json') != row['signature']:
                raise ValueError('Selected AWBC game changed after admission')
            episodes.append(read_game(run))
        train=[i for i,key in enumerate(selected) if buffer.state['entries'][key]['partition']=='train']
        holdout=[i for i,key in enumerate(selected) if buffer.state['entries'][key]['partition']=='validation']
        fresh=[i for i in train if buffer.state['entries'][selected[i]]['uses']==0]
        split={'train':train,'holdout':holdout}
        probes=select_awbc_probes(segments)
        if segments is not None and not probes:
            self.emit(phase='awbc_playbuffer',update=self.update,ran=False,reason='no_ppo_probe_choices')
            return {'ran':False,'actor_attempted':False,'reason':'no_ppo_probe_choices'}
        # PPO rollout optimization is finished. No optimizer may still use old segments.
        result=run_phase(self.actor,self.critic,self.adapter,episodes,split,
            self.out/'awbc_value_features',actor_device=self.device,critic_device=self.critic_device,
            precision=self.precision,epochs=self.args.awbc_epochs,value_epochs=self.args.awbc_value_epochs,
            lr=self.args.awbc_learning_rate,max_kl=self.args.awbc_max_kl,seed=self.args.seed+self.update,
            probes=probes,gamma=self.gamma,fresh_indices=fresh,min_new_games=self.args.awbc_min_new_games)
        atomic_json(self.out/'awbc_latest_report.json',dict(result,selected_game_ids=selected,
            ppo_probes=[probe['probe_info'] for probe in probes]))
        if not result['actor_attempted']:
            reason=result.get('reason','actor_not_attempted')
            if result.get('memory_invalidated',False):
                # A zero-step numerical rejection can still reload readouts
                # and advance the Actor's memory generation during rollback.
                self.actor_weights_dirty=True
                self._sync_behavior_model()
                if self.peer_actor is not None:self._sync_peer_actor()
            # A data-gated skip is not a training use. Retry when admitted data
            # changes, without cycling repeatedly between old candidate batches.
            buffer.state['last_skipped_batch']=dict(game_ids=selected,reason=reason,
                checked_update=self.update,known_games=refresh['known_games'],source_ids=source_ids)
            self.emit(phase='awbc_playbuffer',update=self.update,ran=False,
                actor_attempted=False,reason=reason,batch_games=len(selected),
                train_games=len(result.get('eligible_actor_train_indices',[])),
                holdout_games=len(result.get('eligible_actor_holdout_indices',[])))
            return {'ran':False,'actor_attempted':False,'reason':reason,'buffer_updated':True}
        actor_train=list(result['actor_train_indices'])
        actor_holdout=list(result['actor_holdout_indices'])
        if (not actor_train or not actor_holdout or not set(actor_train).issubset(train)
                or not set(actor_holdout).issubset(holdout)):
            raise ValueError('AWBC returned invalid consumed episode indices')
        for name,param in self.actor.named_parameters():
            if name in result['changed_parameters']:
                self.actor_optimizer.state.pop(param,None)
        # Even a rejected candidate invalidates memory generations during rollback.
        self.actor_weights_dirty=True
        self._sync_behavior_model()
        if self.peer_actor is not None:self._sync_peer_actor()
        consumed=set(actor_train+actor_holdout)
        buffer.commit([key for i,key in enumerate(selected) if i in consumed],self.update)
        buffer.state.pop('last_skipped_batch',None)
        last=result['epochs'][-1] if result['epochs'] else result['baseline']
        self.emit(phase='awbc_playbuffer',update=self.update,ran=True,
            actor_attempted=True,accepted=result['accepted'],batch_games=len(selected),train_games=len(actor_train),
            holdout_games=len(actor_holdout),best_epoch=result['best_epoch'],
            reference_forward_kl=last['reference_forward_kl'],
            value_gate_passed=result['value_report']['value_gate_passed'],
            weighted_train_games=result['value_report']['weighted_train_games'],
            unit_weight_train_games=result['value_report']['unit_weight_train_games'],
            ppo_critic_updated=False)
        return {'ran':True,'actor_attempted':True,'accepted':result['accepted'],'batch_games':len(selected)}

    def run_update(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.device.type == "cuda":
            for dev in {
                d
                for d in (
                    self.device,
                    self.critic_device,
                    self.critic_peer_device,
                    self.opponent_device,
                    self.dual_rollout_device,
                )
                if d is not None and d.type == "cuda"
            }:
                torch.cuda.reset_peak_memory_stats(dev)

        rollout_interval = int(getattr(self.args, "rollout_save_interval", 0) or 0)
        want_rollout = rollout_interval > 0 and ((self.update + 1) % rollout_interval == 0)
        rollout_out = self.collect(save_rollout_buffer=want_rollout)
        # Complete the prior immutable save before changing pool files or summaries.
        self._flush_async_checkpoint(wait=True)
        segments = rollout_out["segments"]

        actor_enabled = (
            self.update >= self.warmup_updates and self.game_counter.count >= self.warmup_games
        )
        if not actor_enabled:
            self.emit(
                phase="critic_warmup",
                update=self.update + 1,
                actor_enabled=False,
                completed_games=self.game_counter.count,
                critic_samples=sum(int(s["valid"].sum()) for s in segments),
                warmup_rounds=self.warmup_updates,
                warmup_games=self.warmup_games,
            )

        if self.actor_lr_controller is not None:
            self.actor_lr_controller.begin_update(self.actor_optimizer)

        # Avoid colliding peer_critic onto actor GPU when actor and critic update concurrently.
        opt_peer_critic = (
            None if (actor_enabled and self.critic_peer_device == self.device) else self.peer_critic
        )
        opt_peer_device = None if opt_peer_critic is None else self.critic_peer_device

        opt_peer_actor = self.peer_actor if (actor_enabled and self.actor_peer_device is not None) else None
        opt_peer_actor_device = self.actor_peer_device if opt_peer_actor is not None else None
        opt_peer_behavior = (
            self.behavior_model if opt_peer_actor_device == self.device else None
        )

        ppo_cfg = self._build_ppo_config()
        frame_cache_context = (
            state_frame_cache(segments, self.adapter,
                [self.device, self.critic_device, opt_peer_device, opt_peer_actor_device, self.opponent_device])
            if getattr(self.args, "cache_state_frames", False) else nullcontext({}))
        optimization_started = time.perf_counter()
        with frame_cache_context as frame_cache_stats:
            opt_out = optimize_state_separate(
                self._actor_opt_model,
                self._critic_opt_model,
                self.behavior_model,
                self.actor_optimizer,
                self.critic_optimizer,
                self.adapter,
                segments,
                ppo_cfg,
                rng=self.rng,
                actor_device=self.device,
                critic_device=self.critic_device,
                peer_critic=opt_peer_critic,
                peer_device=opt_peer_device,
                peer_actor=opt_peer_actor,
                peer_actor_device=opt_peer_actor_device,
                peer_behavior=opt_peer_behavior,
                anchor=self.anchor_model,
                anchor_device=self.opponent_device,
                actor_enabled=actor_enabled,
                lr_controller=self.actor_lr_controller,
            )
        optimization_seconds = time.perf_counter() - optimization_started
        if self.peer_actor is not None:
            self.peer_actor.eval()

        self.update += 1
        batch_samples = sum(len(s["raw_obs"]) for s in segments) * self.world_size
        valid_samples = sum(int(s["valid"].sum()) for s in segments) * self.world_size
        self.samples += batch_samples
        self.optimizer_steps_total += opt_out["actor_steps"]
        self.critic_steps_total += opt_out["critic_steps"]
        if opt_out["actor_steps"] > 0:
            self.actor_weights_dirty = True
            self._sync_behavior_model()
            if self.dual_rollout_device is not None and self.peer_actor is not None:
                self._sync_peer_actor()

        awbc_result = self._maybe_awbc_phase(segments) if actor_enabled else {'ran':False}
        # Give each auxiliary phase a separate PPO boundary and KL budget.
        expert_bc_started = time.perf_counter()
        expert_bc_result = (self._maybe_expert_bc_phase(segments)
            if actor_enabled and not awbc_result.get('ran') else {'ran': False})
        expert_bc_seconds = time.perf_counter() - expert_bc_started
        macro_search_started = time.perf_counter()
        macro_search_result = (self._maybe_macro_search_phase(segments)
            if actor_enabled and not awbc_result.get('ran') else {'ran': False})
        macro_search_seconds = time.perf_counter() - macro_search_started
        lookahead_distill_started = time.perf_counter()
        lookahead_distill_result = (self._maybe_lookahead_distill_phase(segments)
            if actor_enabled and not awbc_result.get('ran') else {'ran': False})
        lookahead_distill_seconds = time.perf_counter() - lookahead_distill_started
        # Snapshot only after auxiliary updates; next rollout rebuilds recurrent states.
        self._update_opponent_pool(opt_out["actor_steps"] + int(awbc_result.get('accepted', False))
                                   + int(expert_bc_result.get('accepted', False))
                                   + int(macro_search_result.get('accepted', False))
                                   + int(lookahead_distill_result.get('accepted', False)))

        rollout_file = None
        if want_rollout and rollout_out["buffer_rows"] is not None and self.rank == 0:
            buf = TrajectoryBuffer(
                {
                    **self.env.metadata(),
                    "training_driver": "persistent-state-v3",
                    "model_architecture_version": 3,
                    "decision_ticks": 8,
                    "level_profile": self.level_profile,
                    "model_config": self.actor.cfg.__dict__,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "critic_gae_lambda": self.critic_gae_lambda,
                    "training_reward_config": self.rewards.metadata(),
                }
            )
            for row in rollout_out["buffer_rows"]:
                buf.append(**row)
            for ep_id in {s["episode_id"] for s in segments if not s["valid"].all()}:
                buf.invalidate_episode(int(ep_id))
            rollout_file = buf.save(self.out / "rollouts" / f"rollout_{self.update:06d}.npz")

        save_interval = max(1, int(getattr(self.args, "checkpoint_interval", 1)))
        should_ckpt = (expert_bc_result['ran'] or macro_search_result['ran'] or lookahead_distill_result['ran'] or awbc_result['ran'] or awbc_result.get('buffer_updated',False)
            or (getattr(self.args, 'checkpoint_first_update', False) and self.update == self.start_update + 1)
            or (self.update % save_interval == 0) or (self.update >= self.target_update))
        ckpt_path = self.out / "ppo_latest.pt"
        if should_ckpt and self.rank == 0:
            ckpt_path = self.checkpoint()
            if not getattr(self.args, "async_checkpoint", False):
                retention = prune_backup(
                    self.out,
                    self.update,
                    recent=int(getattr(self.args, "keep_checkpoints", 2)),
                    rollout_recent=int(getattr(self.args, "keep_rollouts", 2)),
                    max_periodic=getattr(self.args, "keep_periodic_checkpoints", 2),
                )
                retention.update(
                    prune_local_opponents(
                        self.out,
                        active=[entry.checkpoint for entry in self.opponents if entry is not None],
                    )
                )
                self.emit(update=self.update, phase="storage_retention", **retention)

        if self.world_size > 1:
            dist.barrier()

        val_interval = int(getattr(self.args, "validation_interval", 0) or 0)
        training_win_rates = self._compute_training_win_rates(rollout_out)
        if should_ckpt and val_interval > 0 and self.update % val_interval == 0 and self.rank == 0:
            self._flush_async_checkpoint(wait=True)
            val_ckpt = self.out / "actor_export.pt"
            self._run_validation(val_ckpt if val_ckpt.is_file() else ckpt_path)
        elif should_ckpt and val_interval <= 0 and self.rank == 0:
            pending = getattr(self, "_pending_checkpoint", None)
            if pending is not None:
                pending['training_win_rates'] = copy.deepcopy(training_win_rates)
            else:
                self._maybe_save_best_train_traj(training_win_rates)
        if self.world_size > 1:
            dist.barrier()

        coverage = _compute_decision_coverage(segments, self.schema)
        reward_sums = {
            name: float(sum(s["reward_components"][name][s["valid"]].sum() for s in segments))
            for name in ("hp", "crown", "army", "overflow")
        }
        learner_reward_sum = float(sum(s["rewards"][s["valid"]].sum() for s in segments))
        learner_raw_sum = float(sum(s["raw_rewards"][s["valid"]].sum() for s in segments))
        adv_all = np.concatenate([s["advantages"] for s in segments]) if segments else np.empty(0)
        ret_all = np.concatenate([s["returns"] for s in segments]) if segments else np.empty(0)

        cuda_devs = {
            d
            for d in (
                self.device,
                self.critic_device,
                self.critic_peer_device,
                self.opponent_device,
                self.dual_rollout_device,
            )
            if d is not None and d.type == "cuda"
        }
        peak_per_dev = (
            {str(d): float(torch.cuda.max_memory_allocated(d) / 2**30) for d in sorted(cuda_devs, key=str)}
            if cuda_devs
            else {}
        )
        total_seconds = time.perf_counter() - started
        metrics = opt_out["metrics"]
        summary = {
            "phase": "state_ppo",
            "update": self.update,
            "architecture": 3,
            "decision_ticks": 8,
            "critic_mode": self.critic_mode,
            "actor_enabled": actor_enabled,
            "learner_samples_total": self.samples,
            "valid_learner_samples": valid_samples,
            "actor_training_samples": opt_out["actor_samples"],
            "critic_training_samples": opt_out["critic_samples"],
            "actor_choice_samples": opt_out["actor_choice_samples"],
            "completed_training_games": rollout_out["completed_games_new"],
            "completed_training_games_total": self.game_counter.count,
            "optimizer_steps": opt_out["actor_steps"],
            "critic_optimizer_steps": opt_out["critic_steps"],
            "optimizer_steps_total": self.optimizer_steps_total,
            "critic_steps_total": self.critic_steps_total,
            "actor_overflow_steps": opt_out["actor_overflow_steps"],
            "critic_overflow_steps": opt_out["critic_overflow_steps"],
            "actor_kl_stopped": opt_out["stopped_early"],
            "kl_stops": opt_out["kl_stops"],
            "rejected_groups": opt_out["rejected_groups"],
            "metrics": metrics,
            "decision_coverage": coverage,
            **({"sparky_curriculum": rollout_out["sparky_curriculum"]} if "sparky_curriculum" in rollout_out else {}),
            "actor_lr_schedule": opt_out["actor_lr_event"] or {"enabled": False},
            "sample_utilization": {
                "actor_fraction": opt_out["actor_samples"] / max(1, valid_samples),
                "choice_fraction": opt_out["actor_choice_samples"] / max(1, valid_samples),
            },
            "seconds": total_seconds,
            "rollout_seconds": rollout_out["rollout_seconds"],
            "rollout_step_seconds": rollout_out["rollout_step_seconds"],
            "rollout_model_seconds": rollout_out.get("rollout_model_seconds", 0.),
            "rollout_env_seconds": rollout_out.get("rollout_env_seconds", 0.),
            "rollout_bookkeeping_seconds": rollout_out.get("rollout_bookkeeping_seconds", 0.),
            "max_frozen_opponent_groups": rollout_out.get("max_frozen_groups", 0),
            "secondary_learner_samples": rollout_out.get("secondary_samples", 0),
            "full_episode_bptt": rollout_out.get("full_episode_bptt", False),
            "temporal_checkpoint_mode": getattr(self.args, "temporal_checkpoint_mode", "nested"),
            "max_episode_frames": rollout_out.get("max_episode_frames", 0),
            "state_frame_cache": frame_cache_stats,
            "frame_microbatch": self.actor.frame_microbatch,
            "deck_sample_counts": rollout_out.get("deck_sample_counts", {}),
            "critic_eval_seconds": rollout_out["critic_eval_seconds"],
            "actor_update_seconds": opt_out["actor_update_seconds"],
            "critic_update_seconds": opt_out["critic_update_seconds"],
            "samples_per_second": valid_samples / max(total_seconds, 1e-9),
            "rollout_samples_per_second": valid_samples / max(rollout_out["rollout_seconds"], 1e-9),
            "terminal_games": rollout_out["terminal_games"],
            "terminal_seat_wins": rollout_out["terminal_wins"],
            "faulted_samples": rollout_out["faulted_samples"],
            "learner_reward_sum": learner_reward_sum,
            "learner_raw_reward_sum": learner_raw_sum,
            "reward_component_sums": reward_sums,
            "army_reward_distribution": self._army_reward_distribution(segments),
            "adv_std": float(adv_all.std()) if adv_all.size else 0.0,
            "return_mean": float(ret_all.mean()) if ret_all.size else 0.0,
            "episode_steps_at_update_start": rollout_out["starts"],
            "episode_steps_at_update_end": rollout_out["ends"],
            "initial_logp_max_error": max(opt_out["initial_logp_errors"], default=0.0),
            "opponent_pool_size": len(self.pool.entries),
            "opponent_retention": self.pool.diversity_metadata(self.update),
            "role_mixture": self.role_mixture.state_dict(),
            "role_draws": dict(self.role_draws),
            "training_win_rates": training_win_rates,
            "awbc_playbuffer": awbc_result,
            "expert_bc": expert_bc_result,
            "expert_bc_seconds": expert_bc_seconds,
            "macro_search": macro_search_result,
            "macro_search_seconds": macro_search_seconds,
            "lookahead_distill": lookahead_distill_result,
            "lookahead_distill_seconds": lookahead_distill_seconds,
            "ppo_optimization_seconds": optimization_seconds,
            "army_elixir_diagnostics": rollout_out.get('army_elixir_diagnostics',{}),
            "learning_rates": {
                "actor": [float(g["lr"]) for g in self.actor_optimizer.param_groups],
                **(
                    {"critic": [float(g["lr"]) for g in self.critic_optimizer.param_groups]}
                    if self.critic_optimizer is not None
                    else {}
                ),
            },
            "peak_cuda_per_device_gib": peak_per_dev,
            "peak_cuda_allocated_gib": (
                float(torch.cuda.max_memory_allocated(self.device) / 2**30)
                if self.device.type == "cuda"
                else None
            ),
            "rollout": str(rollout_file) if rollout_file else None,
            "checkpoint": str(ckpt_path),
            "resume_semantics": ("whole_episodes_single_behavior_version"
                if getattr(self.args, "full_episode_bptt", False)
                else "continuous_simulator_between_updates_reset_on_process_restart"),
        }

        if self.rank == 0:
            atomic_json(self.out / "summary.json", summary)
            with (self.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary) + "\n")
            progress_path = getattr(self.args, "progress_jsonl", None)
            if progress_path:
                Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(progress_path).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(summary) + "\n")
            if self.telemetry is not None:
                self.telemetry.log_summary(summary)
            self._write_tensorboard(summary, rollout_out["step_policy_stats"])
            atomic_json(
                self.out / "timings" / f"update_{self.update:06d}.json",
                {
                    "update": self.update,
                    "learner_samples_total": self.samples,
                    "valid_learner_samples": valid_samples,
                    "update_wall_seconds": total_seconds,
                    "rollout_seconds": rollout_out["rollout_seconds"],
                    "rollout_step_seconds": rollout_out["rollout_step_seconds"],
                    "critic_eval_seconds": rollout_out["critic_eval_seconds"],
                    "actor_update_seconds": opt_out["actor_update_seconds"],
                    "critic_update_seconds": opt_out["critic_update_seconds"],
                    "effective_samples_per_second": valid_samples / max(total_seconds, 1e-9),
                },
            )
            self._maybe_publish_backup(should_ckpt)
            print(json.dumps(summary), flush=True)

        return summary

    @staticmethod
    def _army_reward_distribution(segments):
        parts=[s['reward_components']['army'][s['valid'].astype(bool)] for s in segments
               if s.get('reward_components') and s['valid'].any()]
        values=np.concatenate(parts) if parts else np.array([],dtype=np.float32)
        if not values.size: return {}
        terminal=sum(float(np.abs(s['raw_rewards'][s['valid'].astype(bool)]).sum()) for s in segments)
        return dict(mean=float(values.mean()),abs_mean=float(np.abs(values).mean()),
            p95=float(np.quantile(np.abs(values),.95)),p99=float(np.quantile(np.abs(values),.99)),
            abs_sum=float(np.abs(values).sum()),terminal_abs_sum=terminal,
            shaping_to_terminal_ratio=float(np.abs(values).sum()/terminal) if terminal else None)

    def _write_tensorboard(self, summary: dict, step_stats: dict) -> None:
        if self.tensorboard is None:
            return
        u = self.update
        for k, v in summary["metrics"].items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.tensorboard.add_scalar(f"ppo/{k}", v, u)
        for k in (
            "samples_per_second",
            "rollout_samples_per_second",
            "rollout_seconds",
            "critic_eval_seconds",
            "actor_update_seconds",
            "critic_update_seconds",
            "completed_training_games_total",
            "optimizer_steps",
            "critic_optimizer_steps",
            "peak_cuda_allocated_gib",
        ):
            val = summary.get(k)
            if isinstance(val, (int, float)) and np.isfinite(val):
                self.tensorboard.add_scalar(f"performance/{k}", val, u)
        for k,v in summary.get("reward_component_sums", {}).items():
            self.tensorboard.add_scalar(f"reward/{k}_sum",v,u)
        for k,v in summary.get("peak_cuda_per_device_gib", {}).items():
            self.tensorboard.add_scalar(f"performance/peak_memory_{k.replace(':','_')}_gib",v,u)
        for dev, size in summary.get("state_frame_cache", {}).get("bytes_per_device", {}).items():
            self.tensorboard.add_scalar(f"performance/state_frame_cache_{dev.replace(':', '_')}_mib", size / 2**20, u)
        if summary.get("state_frame_cache"):
            self.tensorboard.add_scalar("performance/state_frame_cache_prepare_seconds",
                summary["state_frame_cache"]["prepare_seconds"], u)
        for k in ("initial_logp_max_error", "actor_overflow_steps", "critic_overflow_steps",
                  "secondary_learner_samples", "max_frozen_opponent_groups",
                  "full_episode_bptt", "max_episode_frames", "frame_microbatch",
                  "rollout_model_seconds", "rollout_env_seconds", "rollout_bookkeeping_seconds"):
            if k in summary:
                self.tensorboard.add_scalar(f"performance/{k}",summary[k],u)
        for k, v in summary["decision_coverage"].items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.tensorboard.add_scalar(f"decision/{k}", v, u)
        for k, v in summary.get("training_win_rates", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.tensorboard.add_scalar(f"train_traj/{k}", v, u)
        if step_stats["rows"] > 0:
            self.tensorboard.add_scalar(
                "policy/rollout_entropy", step_stats["entropy_sum"] / step_stats["rows"], u
            )
        self.tensorboard.flush()

    def _compute_training_win_rates(self, rollout_out: dict[str, Any]) -> dict[str, Any]:
        """Compute per-update and opponent-pool trajectory win-rate metrics."""
        terminal_games = int(rollout_out.get("terminal_games", 0) or 0)
        terminal_wins = int(rollout_out.get("terminal_wins", 0) or 0)
        role_stats = rollout_out.get("role_terminal_stats") or {}
        role_games = role_stats.get("games") or {}
        role_wins = role_stats.get("wins") or {}
        role_scores = role_stats.get("score_sums") or {}

        roles = ("anchor", "historical", "current")
        total_role_games = sum(int(role_games.get(r, 0) or 0) for r in roles)
        overall_win_rate = float(terminal_wins / terminal_games) if terminal_games > 0 else None
        overall_score_rate = (
            float(sum(float(role_scores.get(r, 0.0) or 0.0) for r in roles) / total_role_games)
            if total_role_games > 0
            else overall_win_rate
        )
        out: dict[str, Any] = {
            "overall_win_rate": overall_win_rate,
            "overall_score_rate": overall_score_rate,
        }
        for role in roles:
            g = int(role_games.get(role, 0) or 0)
            w = int(role_wins.get(role, 0) or 0)
            out[f"{role}_games"] = g
            out[f"{role}_wins"] = w
            out[f"{role}_win_rate"] = float(w / g) if g > 0 else None

        entries = getattr(getattr(self, "pool", None), "entries", None) or []
        anchor_entries = [e for e in entries if getattr(e, "role", None) == "anchor"]
        if anchor_entries:
            anc_games = int(sum(int(getattr(e, "games", 0) or 0) for e in anchor_entries))
            out["anchor_pool_games"] = anc_games
            out["anchor_pool_ema_win_rate"] = float(
                sum(float(getattr(e, "ema_score", 0.0) or 0.0) for e in anchor_entries)
                / len(anchor_entries)
            )
            out["anchor_pool_cumulative_win_rate"] = (
                float(sum(float(getattr(e, "score_sum", 0.0) or 0.0) for e in anchor_entries) / anc_games)
                if anc_games > 0
                else None
            )
        else:
            out["anchor_pool_games"] = 0
            out["anchor_pool_ema_win_rate"] = None
            out["anchor_pool_cumulative_win_rate"] = None

        hist_entries = [
            e
            for e in entries
            if getattr(e, "role", None) == "historical" and int(getattr(e, "games", 0) or 0) > 0
        ]
        if hist_entries:
            hist_games = int(sum(int(e.games) for e in hist_entries))
            out["historical_pool_games"] = hist_games
            out["historical_pool_mean_ema_win_rate"] = float(
                sum(float(e.ema_score) for e in hist_entries) / len(hist_entries)
            )
            out["historical_pool_cumulative_win_rate"] = float(
                sum(float(e.score_sum) for e in hist_entries) / hist_games
            )
        else:
            out["historical_pool_games"] = 0
            out["historical_pool_mean_ema_win_rate"] = None
            out["historical_pool_cumulative_win_rate"] = None
        adv, bal_wr, n_matched = self._deck_vs_bc_summary()
        out["skill_advantage_vs_bc"] = adv
        out["deck_balanced_vs_bc_win_rate"] = bal_wr
        out["skill_advantage_vs_bc_decks"] = n_matched
        return out

    @staticmethod
    def _update_deck_vs_bc_ema(
        counts: dict[str, int], ema: dict[str, float], deck: str | None, score: float
    ) -> None:
        if not deck:
            return
        cnt = int(counts.get(deck, 0)) + 1
        counts[deck] = cnt
        alpha = 1.0 / min(cnt, 30)
        prev = float(ema.get(deck, score))
        ema[deck] = (1.0 - alpha) * prev + alpha * float(score)

    def _init_deck_vs_bc_tracker(self, args, checkpoint: dict) -> dict[str, dict]:
        saved = checkpoint.get("deck_vs_bc") if getattr(args, "resume", False) else None
        if isinstance(saved, dict) and all(
            isinstance(saved.get(k), dict)
            for k in ("learner_counts", "learner_ema", "bc_counts", "bc_ema")
        ):
            return {
                "learner_counts": {str(k): int(v) for k, v in saved["learner_counts"].items()},
                "learner_ema": {str(k): float(v) for k, v in saved["learner_ema"].items()},
                "bc_counts": {str(k): int(v) for k, v in saved["bc_counts"].items()},
                "bc_ema": {str(k): float(v) for k, v in saved["bc_ema"].items()},
            }
        tracker: dict[str, dict] = {
            "learner_counts": {},
            "learner_ema": {},
            "bc_counts": {},
            "bc_ema": {},
        }
        prog_raw = getattr(args, "progress_jsonl", None)
        if getattr(args, "resume", False) and prog_raw:
            prog_path = Path(prog_raw)
            if prog_path.is_file():
                try:
                    with prog_path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            if '"episode"' not in line or '"anchor"' not in line:
                                continue
                            row = json.loads(line)
                            if (
                                row.get("phase") == "episode"
                                and row.get("opponent_role") == "anchor"
                                and int(row.get("update", 0) or 0) <= self.update
                            ):
                                rew = float(row.get("reward", 0.0) or 0.0)
                                sc = 1.0 if rew > 0 else (0.5 if rew == 0 else 0.0)
                                self._update_deck_vs_bc_ema(
                                    tracker["learner_counts"],
                                    tracker["learner_ema"],
                                    row.get("learner_deck"),
                                    sc,
                                )
                                self._update_deck_vs_bc_ema(
                                    tracker["bc_counts"],
                                    tracker["bc_ema"],
                                    row.get("opponent_deck"),
                                    1.0 - sc,
                                )
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
        return tracker

    def _record_deck_vs_bc(
        self, learner_deck: str | None, opp_deck: str | None, raw_reward: float
    ) -> None:
        tracker = getattr(self, "_deck_vs_bc", None)
        if tracker is None:
            tracker = {
                "learner_counts": {},
                "learner_ema": {},
                "bc_counts": {},
                "bc_ema": {},
            }
            self._deck_vs_bc = tracker
        score = 1.0 if raw_reward > 0 else (0.5 if raw_reward == 0 else 0.0)
        self._update_deck_vs_bc_ema(tracker["learner_counts"], tracker["learner_ema"], learner_deck, score)
        self._update_deck_vs_bc_ema(tracker["bc_counts"], tracker["bc_ema"], opp_deck, 1.0 - score)

    def _deck_vs_bc_summary(self) -> tuple[float | None, float | None, int]:
        tracker = getattr(self, "_deck_vs_bc", None)
        if not tracker:
            return None, None, 0
        l_cnt = tracker["learner_counts"]
        l_ema = tracker["learner_ema"]
        b_cnt = tracker["bc_counts"]
        b_ema = tracker["bc_ema"]
        matched = [d for d in l_ema if d in b_ema]
        if not matched:
            return None, None, 0
        well_sampled = [d for d in matched if l_cnt.get(d, 0) >= 3 and b_cnt.get(d, 0) >= 3]
        chosen = well_sampled if len(well_sampled) >= 10 else matched
        adv = float(sum(float(l_ema[d]) - float(b_ema[d]) for d in chosen) / len(chosen))
        return adv, 0.5 + 0.5 * adv, len(chosen)

    def _maybe_save_best_train_traj(self, training_win_rates: dict[str, Any], *, update=None) -> None:
        """Persist best_val checkpoint links using pool trajectory win rates when validation is off."""
        update = self.update if update is None else update
        anchor_games = int(training_win_rates.get("anchor_pool_games") or 0)
        anc_ema = training_win_rates.get("anchor_pool_ema_win_rate")
        if anchor_games < 100 or not isinstance(anc_ema, (int, float)) or not np.isfinite(anc_ema):
            return
        hist_ema = training_win_rates.get("historical_pool_mean_ema_win_rate")
        hist_score = (
            float(hist_ema)
            if isinstance(hist_ema, (int, float)) and np.isfinite(hist_ema)
            else -1.0
        )
        score = (float(anc_ema), hist_score)

        best_meta_path = self.out / "best_train_traj.json"
        prev_score: tuple[float, float] | None = None
        if best_meta_path.is_file():
            try:
                prev = json.loads(best_meta_path.read_text(encoding="utf-8"))
                prev_score = (
                    float(prev.get("anchor_pool_ema_win_rate", -1.0)),
                    float(prev.get("historical_pool_mean_ema_win_rate", -1.0)),
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                prev_score = None
        if prev_score is not None and score < prev_score:
            return

        ppo_src = self.out / "ppo_latest.pt"
        actor_src = self.out / "actor_export.pt"
        if ppo_src.is_file():
            self._atomic_snapshot_file(ppo_src, self.out / "ppo_best_val.pt")
        if actor_src.is_file():
            self._atomic_snapshot_file(actor_src, self.out / "actor_best_val.pt")
        atomic_json(
            best_meta_path,
            {
                "update": int(update),
                "anchor_pool_ema_win_rate": float(anc_ema),
                "historical_pool_mean_ema_win_rate": hist_score,
                "training_win_rates": training_win_rates,
                "ppo_checkpoint": str(self.out / "ppo_best_val.pt") if ppo_src.is_file() else None,
                "actor_checkpoint": str(self.out / "actor_best_val.pt") if actor_src.is_file() else None,
                "time_unix": time.time(),
            },
        )

    def _maybe_publish_backup(self, should_ckpt: bool, *, update=None) -> None:
        update = self.update if update is None else update
        backup_dir = getattr(self.args, "backup_dir", None)
        if not backup_dir or not should_ckpt:
            return
        pub_int = max(1, int(getattr(self.args, "publication_interval", 1)))
        if update % pub_int != 0 and update < self.target_update:
            return
        pending = getattr(self, "_pending_checkpoint", None)
        if pending is not None:
            pending["publish"] = True
            self._flush_async_checkpoint()
            return
        dest = Path(backup_dir)
        if self._publisher is not None:
            self._publisher.submit(update)
        elif any((self.out / "rollouts").glob("*.npz")) and getattr(
            self.args, "incremental_backup", False
        ):
            # Full manifest backup when rollout buffers exist; else atomic direct copy.
            publish_update(self.out, dest, update)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            copy_verified(self.out / "ppo_latest.pt", dest / "ppo_latest.pt")
            if (self.out / "actor_export.pt").exists():
                copy_verified(self.out / "actor_export.pt", dest / "actor_export.pt")

    def _validation_devices(self) -> list[torch.device]:
        """Return candidate validation devices ordered with the freest CUDA GPUs first."""
        candidates: list[torch.device] = []
        for dev in (
            getattr(self, "opponent_device", None),
            getattr(self, "device", None),
            getattr(self, "critic_device", None),
        ):
            if dev is not None and dev not in candidates:
                candidates.append(dev)
        cuda_candidates = [d for d in candidates if d.type == "cuda"]
        if cuda_candidates:
            for dev in cuda_candidates:
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
            return sorted(
                cuda_candidates,
                key=lambda d: torch.cuda.mem_get_info(d)[0],
                reverse=True,
            )
        return [candidates[0]] if candidates else [torch.device("cpu")]

    def _validation_device(self) -> torch.device:
        return self._validation_devices()[0]

    @staticmethod
    def _parse_validation_result(eval_path: Path, proc_error: str | None) -> dict[str, Any]:
        """Parse an evaluation.json report while preserving partial results on warnings."""
        parsed = None
        if eval_path.is_file():
            try:
                payload = json.loads(eval_path.read_text(encoding="utf-8"))
                games = int(payload.get("games", 0))
                wins = int(payload.get("wins", 0))
                if games > 0:
                    parsed = {
                        "wins": wins,
                        "games": games,
                        "win_rate": wins / games,
                        "invalid_games": int(payload.get("invalid_games", 0)),
                    }
                    budget_passed = payload.get("candidate_inference_budget_passed")
                    if budget_passed is not None:
                        parsed["budget_passed"] = bool(budget_passed)
                    latency = payload.get("latency", {}).get("candidate_end_to_end")
                    if isinstance(latency, dict):
                        for k, src_k in (
                            ("p50_ms", "p50_ms"),
                            ("p99_ms", "p99_ms"),
                            ("max_ms", "max_ms"),
                            ("latency_violations", "violations"),
                        ):
                            if latency.get(src_k) is not None:
                                parsed[k] = latency[src_k]
            except Exception as parse_exc:
                proc_error = proc_error or repr(parse_exc)
        if parsed is not None:
            if proc_error is not None:
                parsed["warning"] = proc_error
            return parsed
        return {
            "wins": 0,
            "games": 0,
            "win_rate": 0.0,
            "error": proc_error or "missing evaluation.json",
        }

    def _run_validation(self, checkpoint: Path) -> None:
        root = Path(__file__).resolve().parents[3]
        evaluator = root / "scripts" / "evaluate_state_policy.py"
        decks = Path(getattr(self.args, "validation_decks", None) or "configs/decks_eval.yaml")
        out_root = self.out / "validation" / f"update_{self.update:08d}"
        pairs = int(getattr(self.args, "validation_pairs", 5))
        argmax_lanes = int(getattr(self.args, "validation_argmax_lanes", 0))
        out_root.mkdir(parents=True, exist_ok=True)
        gc.collect()
        release_host_memory()
        val_devices = self._validation_devices()
        opponents = [("bc", self.anchor_path)]
        hist_entries = [e for e in self.pool.entries if e.role == "historical"]
        if hist_entries:
            opponents.append(("history", Path(hist_entries[-1].checkpoint)))
        big_vram = (val_devices[0].type == "cuda" and torch.cuda.get_device_properties(val_devices[0]).total_memory >= 45 * 1024**3)
        parallel = (len(val_devices) >= 2 or big_vram) and len(opponents) >= 2

        specs: list[tuple[str, Path, list[str]]] = []
        for idx, (label, opponent) in enumerate(opponents):
            target = out_root / f"vs_{label}"
            target.parent.mkdir(parents=True, exist_ok=True)
            val_device = val_devices[idx % len(val_devices)] if parallel else val_devices[0]
            cmd = [
                sys.executable,
                str(evaluator),
                "--policy",
                str(checkpoint),
                "--opponent",
                str(opponent),
                "--decks",
                str(decks),
                "--pairs",
                str(pairs),
                "--envs",
                str(argmax_lanes or 2),
                "--device",
                str(val_device),
                "--precision",
                self.precision,
                "--out",
                str(target),
                "--no-native-audit",
            ]
            if argmax_lanes:
                cmd.extend(("--action-mode", "argmax", "--batch-argmax"))
            if getattr(self.args, "no_backup_hash", False):
                cmd.append("--no-file-hashes")
            specs.append((label, target, cmd))

        proc_errors: dict[str, str | None] = {label: None for label, _, _ in specs}
        if parallel:
            self.emit(
                phase="validation_started",
                update=self.update,
                opponents=[label for label, _, _ in specs],
                time_unix=time.time(),
            )
            launched: list[tuple[str, list[str], Any, subprocess.Popen]] = []
            try:
                for label, target, cmd in specs:
                    log_fh = target.with_suffix(".log").open("w", encoding="utf-8")
                    try:
                        proc = subprocess.Popen(cmd, cwd=root, stdout=log_fh, stderr=subprocess.STDOUT)
                    except Exception as exc:
                        log_fh.close()
                        proc_errors[label] = repr(exc)
                        continue
                    launched.append((label, cmd, log_fh, proc))
                for label, cmd, _log_fh, proc in launched:
                    try:
                        rc = proc.wait()
                        if rc != 0:
                            proc_errors[label] = repr(subprocess.CalledProcessError(rc, cmd))
                    except Exception as exc:
                        proc_errors[label] = repr(exc)
            finally:
                for _label, _cmd, log_fh, proc in launched:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    log_fh.close()
        else:
            for label, target, cmd in specs:
                self.emit(phase="validation_started", update=self.update, opponent=label, time_unix=time.time())
                try:
                    with target.with_suffix(".log").open("w", encoding="utf-8") as log:
                        subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                except Exception as exc:  # validation never kills long run
                    proc_errors[label] = repr(exc)
                self.emit(
                    phase="validation_opponent_finished",
                    update=self.update,
                    opponent=label,
                    error=proc_errors[label],
                    time_unix=time.time(),
                )

        results = {
            label: self._parse_validation_result(target / "evaluation.json", proc_errors[label])
            for label, target, _cmd in specs
        }
        atomic_json(out_root / "summary.json", results)
        self._maybe_save_best_validation(results, checkpoint)
        if self.tensorboard is not None:
            for label, vals in results.items():
                if int(vals.get("games", 1) or 0) == 0 and vals.get("error"):
                    continue
                self.tensorboard.add_scalar(f"validation/{label}_win_rate", vals["win_rate"], self.update)
                if isinstance(vals.get("p99_ms"), (int, float)):
                    self.tensorboard.add_scalar(f"validation/{label}_p99_ms", vals["p99_ms"], self.update)
            self.tensorboard.flush()
        event = {"phase": "validation", "update": self.update, "results": results, "time_unix": time.time()}
        self.emit(**event)
        print(json.dumps(event), flush=True)

    @staticmethod
    def _atomic_snapshot_file(src: Path, dst: Path) -> None:
        """Atomically persist a snapshot of `src` at `dst` via hardlink or copy."""
        tmp = dst.with_name(f"{dst.name}.tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        tmp.replace(dst)

    def _maybe_save_best_validation(self, results: dict[str, dict[str, Any]], ckpt_path: Path) -> None:
        """Persist `ppo_best_val.pt` and `actor_best_val.pt` when validation sets a new best."""
        bc_res = results.get("bc", {})
        bc_games = int(bc_res.get("games", 0) or 0)
        if bc_games <= 0 or bc_res.get("error"):
            return
        bc_wr = float(bc_res.get("win_rate", 0.0) or 0.0)
        hist_res = results.get("history", {})
        hist_games = int(hist_res.get("games", 0) or 0)
        hist_wr = (
            float(hist_res.get("win_rate", 0.0) or 0.0)
            if hist_games > 0 and not hist_res.get("error")
            else -1.0
        )

        best_meta_path = self.out / "best_validation.json"
        prev_score: tuple[float, float] | None = None
        if best_meta_path.is_file():
            try:
                prev = json.loads(best_meta_path.read_text(encoding="utf-8"))
                prev_score = (
                    float(prev.get("bc_win_rate", -1.0)),
                    float(prev.get("history_win_rate", -1.0)),
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                prev_score = None
        if prev_score is not None and (bc_wr, hist_wr) < prev_score:
            return

        ppo_src = self.out / "ppo_latest.pt"
        actor_src = self.out / "actor_export.pt"
        if not actor_src.is_file() and ckpt_path.is_file() and ckpt_path != ppo_src:
            actor_src = ckpt_path
        if ppo_src.is_file():
            self._atomic_snapshot_file(ppo_src, self.out / "ppo_best_val.pt")
        if actor_src.is_file():
            self._atomic_snapshot_file(actor_src, self.out / "actor_best_val.pt")
        atomic_json(
            best_meta_path,
            {
                "update": int(self.update),
                "bc_win_rate": bc_wr,
                "history_win_rate": hist_wr,
                "results": results,
                "ppo_checkpoint": str(self.out / "ppo_best_val.pt") if ppo_src.is_file() else None,
                "actor_checkpoint": str(self.out / "actor_best_val.pt") if actor_src.is_file() else None,
                "time_unix": time.time(),
            },
        )

    def _record_awbc_smoke(self, summary: dict) -> None:
        """Require a real Actor attempt and a subsequent successful PPO update."""
        report = getattr(self, '_awbc_smoke_report', None)
        if report is None:
            report = dict(passed=False, actor_attempt_update=None, post_aux_ppo_update=None)
            self._awbc_smoke_report = report
        event = summary.get('awbc_playbuffer', {})
        error = None
        if report['actor_attempt_update'] is None:
            if not event.get('actor_attempted', False):
                error = 'AWBC smoke did not exercise Actor optimization: ' + event.get(
                    'reason', 'hook skipped or PPO Actor disabled')
            else:
                report['actor_attempt_update'] = int(summary['update'])
                report['accepted'] = bool(event.get('accepted', False))
        elif int(summary['update']) > report['actor_attempt_update']:
            if int(summary.get('optimizer_steps', 0)) < 1:
                error = 'AWBC smoke requires a non-skipped PPO Actor update after the auxiliary attempt'
            elif not np.isfinite(summary.get('initial_logp_max_error', float('nan'))):
                error = 'AWBC smoke requires a finite post-auxiliary initial log-probability check'
            else:
                report['post_aux_ppo_update'] = int(summary['update'])
                report['initial_logp_max_error'] = float(summary['initial_logp_max_error'])
                report['passed'] = True
        if error:
            report['failure'] = error
        atomic_json(self.out/'awbc_smoke_report.json', report)
        if error:
            raise RuntimeError(error)

    def run(self) -> None:
        stop_requested = False

        def _request_stop(_signum, _frame):
            nonlocal stop_requested
            stop_requested = True

        prev_term = signal.signal(signal.SIGTERM, _request_stop)
        prev_int = signal.signal(signal.SIGINT, _request_stop)
        target_games = getattr(self.args, "min_training_games", None)
        try:
            for _ in range(int(self.args.updates)):
                if stop_requested:
                    break
                if target_games is not None and self.game_counter.count >= target_games:
                    break
                summary = self.run_update()
                if getattr(self.args, 'require_awbc_smoke', False):
                    self._record_awbc_smoke(summary)
                if getattr(self.args, "compact_host_memory", False):
                    gc.collect()
                    release_host_memory()
                if stop_requested:
                    break
            if (getattr(self.args, 'require_awbc_smoke', False)
                    and not getattr(self, '_awbc_smoke_report', {}).get('passed', False)):
                raise RuntimeError('AWBC smoke incomplete: require an auxiliary Actor attempt followed by PPO')
        finally:
            if (stop_requested and self.rank == 0 and self.update > 0
                    and getattr(self, "_last_checkpoint_update", None) != self.update
                    and (getattr(self, "_pending_checkpoint", None) or {}).get('update') != self.update):
                self.checkpoint()
                self._flush_async_checkpoint(wait=True)
                self._maybe_publish_backup(True)
                # Graceful profiling/restarts publish restore points too; apply
                # the same bounded retention as a scheduled checkpoint.
                retention = prune_backup(self.out, self.update,
                    recent=int(getattr(self.args, "keep_checkpoints", 2)),
                    rollout_recent=int(getattr(self.args, "keep_rollouts", 2)),
                    max_periodic=getattr(self.args, "keep_periodic_checkpoints", 2))
                retention.update(prune_local_opponents(self.out,
                    active=[e.checkpoint for e in self.opponents if e is not None]))
                self.emit(update=self.update, phase="storage_retention", **retention)
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)

    def close(self) -> None:
        try:
            self._flush_async_checkpoint(wait=True)
            if getattr(self, "_checkpoint_writer", None) is not None:
                self._checkpoint_writer.close()
            if self._publisher is not None:
                self._publisher.wait()
                self._publisher.close()
        finally:
            if self._opp_executor is not None:
                self._opp_executor.shutdown(wait=True)
            if self._peer_executor is not None:
                self._peer_executor.shutdown(wait=True)
            self.env.close()
            if self.world_size > 1 and dist.is_initialized():
                dist.barrier()
            if self.tracking is not None:
                self.tracking.finish()
            if self.tensorboard is not None:
                self.tensorboard.close()


StateSelfplay = PersistentStateSelfplay


def run(args) -> None:
    trainer = PersistentStateSelfplay(args)
    try:
        trainer.run()
    finally:
        trainer.close()
        if trainer.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
