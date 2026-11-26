"""Online AlphaZero-style Lookahead Search Distiller for architecture-v3 PPO.

Runs short-horizon Critic-bootstrapped CrForge lookahead rollouts (`LookaheadPolicyStream`)
on focal learner decks against the full opponent deck pool, admits winning search-improved
trajectories into a persistent SSD ring buffer, and distills search-overridden macro decisions
directly into Actor readout heads under strict KL and WAIT-drift trust-region gates.

This transfers the +16.7% (+120 Elo) test-time search advantage into the standalone
`actor_export.pt` network so real-time private-server inference (`run_nulls_policy.py`,
<=400ms budget, no simulator seed) benefits with zero live search latency.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models.state_policy import reserve_step_memory, slice_memory
from cr_rl.training.expert_bc_plan_b import MAX_KL, MAX_WAIT_DRIFT, _score, _train_accumulated
from cr_rl.training.live_awbc import freeze_actor_readouts
from cr_rl.training.macro_card_search import probe_anchor_weights
from cr_rl.training.test_time_search import (
    LookaheadConfig,
    LookaheadPolicyStream,
    _emergency_tower_threat,
    run_batched_lookahead,
)

MAX_DISTILL_KL = 0.010
MAX_DISTILL_WAIT_DRIFT = 0.008
MIN_DISTILL_MEAN_P_WAIT = 0.75
MAX_DISTILL_MEAN_P_WAIT = 0.90
MAX_DISTILL_LR = 3e-5


def unfreeze_distill_heads(
    actor,
    *,
    include_placement: bool = True,
    include_last_temporal: bool = True,
    temporal_blocks: int | None = None,
) -> list[str]:
    """Unfreeze Actor readouts, spatial placement heads, and top temporal blocks (never `.anchor`)."""
    readout_prefixes = ("wait_actor.", "slot_actor.", "ability_actor.", "wait_hand_gate.")
    has_lp = hasattr(actor, "learned_placement")
    tile_prefixes = ("learned_placement.",) if has_lp else ("tile_actor.", "tile_film.", "tile_spatial_gate")
    temporal_prefixes: tuple[str, ...] = ()
    if hasattr(actor, "temporal") and hasattr(actor.temporal, "blocks") and len(actor.temporal.blocks) > 0:
        total_b = len(actor.temporal.blocks)
        n_unfreeze = int(temporal_blocks) if temporal_blocks is not None else (1 if include_last_temporal else 0)
        if n_unfreeze > 0:
            start_idx = max(0, total_b - n_unfreeze)
            temporal_prefixes = tuple(f"temporal.blocks.{idx}." for idx in range(start_idx, total_b)) + (
                "temporal.final_norm.",
            )
    names: list[str] = []
    for name, p in actor.named_parameters():
        keep = name.startswith(readout_prefixes)
        if include_placement and not keep and name.startswith(tile_prefixes) and not name.endswith(".anchor"):
            keep = True
        if not keep and temporal_prefixes and name.startswith(temporal_prefixes):
            keep = True
        p.requires_grad_(keep)
        if keep:
            names.append(name)
    if not names:
        raise ValueError("No compatible Actor readout/placement parameters found for distillation")
    return names


def _train_distill_accumulated(actor, adapter, episodes, weights, optimizer, scaler, device, precision) -> None:
    """Weight-normalized batched sequence NLL update so sparse overrides receive crisp gradients fast."""
    from cr_rl.training.expert_bc_plan_b import _aux_episode_batch_size, _batched_episodes

    optimizer.zero_grad(set_to_none=True)
    n_eps = max(1, len(episodes))
    for b_start, group, lengths, logp_b in _batched_episodes(
        actor, adapter, episodes, device, precision, batch_size=_aux_episode_batch_size(device, 6)
    ):
        batch_loss = None
        for pos, (episode, n_len) in enumerate(zip(group, lengths)):
            weight = weights[b_start + pos][:n_len]
            pos_mask = (weight > 0) & np.asarray(episode["actor_keep"][:n_len], dtype=bool)
            if not pos_mask.any():
                continue
            denom = max(float(weight[pos_mask].sum()), 1.0) * n_eps
            active = torch.as_tensor(pos_mask, device=device)
            actions = torch.as_tensor(episode["actions"][:n_len][pos_mask], device=device)
            row_weights = torch.as_tensor(weight[pos_mask], device=device)
            ep_loss = -(logp_b[pos, :n_len][active].gather(1, actions[:, None]).squeeze(1) * row_weights).sum() / denom
            batch_loss = ep_loss if batch_loss is None else (batch_loss + ep_loss)
        if batch_loss is not None:
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("Nonfinite LookaheadDistiller training loss")
            scaler.scale(batch_loss).backward()
        del logp_b, batch_loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad], 1.0
    )
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite LookaheadDistiller gradients")
    scaler.step(optimizer)
    scaler.update()
    actor.mark_weights_updated()


def _is_adverse_frame(raw_obs: np.ndarray, schema) -> bool:
    """True when learner is behind on crowns, >=600 tower HP down, or facing an emergency push."""
    try:
        if int(schema.field(raw_obs, "crowns")) < int(schema.field(raw_obs, "crowns_enemy")):
            return True
        own_hp = sum(int(raw_obs[schema.header[f"own_{t}_hp"][0]]) for t in ("left", "right", "king"))
        opp_hp = sum(int(raw_obs[schema.header[f"enemy_{t}_hp"][0]]) for t in ("left", "right", "king"))
        if own_hp - opp_hp <= -600:
            return True
    except Exception:
        pass
    return _emergency_tower_threat(raw_obs, schema)


def lookahead_frame_weights(
    actions: np.ndarray,
    actor_keep: np.ndarray,
    searched_mask: np.ndarray,
    override_mask: np.ndarray,
    *,
    override_boost: float = 1.0,
    bank_wait_mask: np.ndarray | None = None,
    adverse_mask: np.ndarray | None = None,
    override_only: bool = False,
    pure_override: bool = False,
    score_gains: np.ndarray | None = None,
) -> np.ndarray:
    """Emphasize Critic-verified lookahead overrides, elixir-banking WAITs, and adverse defense."""
    keep = np.asarray(actor_keep, dtype=bool)
    acts = np.asarray(actions, dtype=np.int64)
    searched = np.asarray(searched_mask, dtype=bool)
    overridden = np.asarray(override_mask, dtype=bool)
    boost = max(0.5, float(override_boost))

    if override_only:
        w = np.zeros(len(acts), dtype=np.float64)
    elif pure_override:
        # Zero weight on unsearched garbage PLAYs; keep 0.24 WAIT patience baseline so wait_delta stays ~0!
        w = np.where(acts == 0, 0.24, 0.0)
        w = np.where(searched & ~overridden & (acts > 0), 0.35, w)
        w = np.where(searched & ~overridden & (acts == 0), 0.24, w)
    else:
        # Balanced non-override baseline (prevents 10x anti-WAIT erosion across updates)
        w = np.where(acts > 0, 0.35, 0.15)
        w = np.where(searched & ~overridden & (acts > 0), 0.50, w)
        w = np.where(searched & ~overridden & (acts == 0), 0.20, w)

    w = np.where(overridden & (acts == 0), 1.80 * boost, w)
    if bank_wait_mask is not None:
        bank_w = np.asarray(bank_wait_mask, dtype=bool)
        if override_only:
            w = np.where(bank_w & overridden & (acts == 0), 2.50 * boost, w)
        else:
            w = np.where(bank_w & (acts == 0), 2.50 * boost, w)
    w = np.where(overridden & (acts > 0), 2.50 * boost, w)
    if score_gains is not None:
        gains = np.clip(np.asarray(score_gains, dtype=np.float64), 0.0, 0.25)
        w = np.where(overridden, w * (1.0 + 8.0 * gains), w)
    if adverse_mask is not None:
        adv = np.asarray(adverse_mask, dtype=bool)
        w = np.where(adv, w * 1.60, w)
    return np.where(keep, w, 0.0).astype(np.float32)


class _GreedyMirrorStream:
    """Mirror opponent stream sharing current Actor weights (argmax or low-temp sampled)."""

    def __init__(
        self,
        actor,
        adapter,
        device: torch.device,
        precision: str = "fp16",
        *,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.actor = actor
        self.adapter = adapter
        self.device = torch.device(device)
        self.precision = precision
        self.temperature = float(temperature)
        self.rng = rng
        self.hand_indices = list(adapter.schema.header["hand_card"])
        self.state = None

    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def act(self, raw: np.ndarray, legal: np.ndarray) -> int:
        h, e, m = self.adapter(np.asarray(raw)[None])
        tensors = [torch.as_tensor(v, device=self.device) for v in (h, e, m)]
        hand = torch.as_tensor(np.asarray(raw)[self.hand_indices][None], device=self.device)
        legal_b = np.asarray(legal, dtype=bool)
        mask_t = torch.as_tensor(legal_b[None], device=self.device)
        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if (
                self.precision == "fp16"
                and self.device.type == "cuda"
                and not getattr(self.actor, "ppo_actor_fp32", False)
            )
            else nullcontext()
        )
        with ctx:
            logp, _, self.state = self.actor.step(
                *tensors, mask_t, state=self.state, hand_card_ids=hand, compute_value=False
            )
        masked = logp[0].masked_fill(~mask_t[0], -torch.inf)
        if self.temperature > 1e-3 and self.rng is not None and int(legal_b.sum()) > 1:
            scaled = (masked.float() / self.temperature)
            scaled = scaled - scaled.max()
            probs = torch.where(mask_t[0], scaled.exp(), torch.zeros_like(scaled)).cpu().numpy()
            s = float(probs.sum())
            if s > 0 and np.isfinite(s):
                probs = probs / s
                choice = int(self.rng.choice(len(probs), p=probs))
                if legal_b[choice]:
                    return choice
        return int(masked.argmax(dim=-1).item())


class LookaheadDistiller:
    """Online lookahead trajectory generator + KL/WAIT-gated readout distiller."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        max_buffer_episodes: int = 192,
        seed: int = 20261122,
        state: dict | None = None,
    ):
        self.buffer_dir = Path(out_dir).resolve() / "lookahead_distill_buffer"
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_episodes = int(max_buffer_episodes)
        self.rng = np.random.default_rng(seed)
        self.last_update = -1
        self.phase_count = 0
        self.games_played = 0
        self.admitted_total = 0
        self.total_overrides = 0
        self.next_shard_id = 0
        self.entries: list[dict] = []
        if state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
            self.last_update = int(state.get("last_update", -1))
            self.phase_count = int(state.get("phase_count", 0))
            self.games_played = int(state.get("games_played", 0))
            self.admitted_total = int(state.get("admitted_total", 0))
            self.total_overrides = int(state.get("total_overrides", 0))
            self.next_shard_id = int(state.get("next_shard_id", 0))
            for item in state.get("entries", []):
                if (self.buffer_dir / item["filename"]).is_file():
                    self.entries.append(dict(item))

    def state_dict(self) -> dict:
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state),
            last_update=self.last_update,
            phase_count=self.phase_count,
            games_played=self.games_played,
            admitted_total=self.admitted_total,
            total_overrides=self.total_overrides,
            next_shard_id=self.next_shard_id,
            entries=copy.deepcopy(self.entries),
        )

    def _save_episode(self, episode: dict, metadata: dict) -> dict:
        shard_id = self.next_shard_id
        self.next_shard_id += 1
        filename = f"lookahead_{shard_id:06d}.npz"
        target = self.buffer_dir / filename
        tmp = self.buffer_dir / f"lookahead_{shard_id:06d}.tmp.npz"
        np.savez_compressed(
            tmp,
            raw_obs=episode["raw_obs"].astype(np.int32),
            legal_masks=episode["legal_masks"].astype(bool),
            actions=episode["actions"].astype(np.int64),
            actor_keep=episode["actor_keep"].astype(bool),
            weights=episode["weights"].astype(np.float32),
        )
        tmp.replace(target)
        entry = dict(id=shard_id, filename=filename, **metadata)
        self.entries.append(entry)
        while len(self.entries) > self.max_buffer_episodes:
            evicted = self.entries.pop(0)
            (self.buffer_dir / evicted["filename"]).unlink(missing_ok=True)
        return entry

    def _load_episodes(self, entries: list[dict]) -> tuple[list[dict], list[np.ndarray]]:
        episodes, weights = [], []
        for item in entries:
            with np.load(self.buffer_dir / item["filename"], allow_pickle=False) as data:
                ep = dict(
                    raw_obs=data["raw_obs"],
                    legal_masks=data["legal_masks"].astype(bool),
                    actions=data["actions"].astype(np.int64),
                    actor_keep=data["actor_keep"].astype(bool),
                )
                weights.append(data["weights"].astype(np.float32))
                episodes.append(ep)
        return episodes, weights

    def _play_distill_game(
        self,
        play_env: CrForgeVector,
        search_env: CrForgeVector,
        actor,
        critic,
        adapter,
        *,
        seed: int,
        seat: int,
        own_deck_idx: int,
        opp_deck_idx: int,
        decks,
        forms,
        deck_names,
        device: torch.device,
        precision: str,
        cfg: LookaheadConfig,
        max_game_steps: int,
    ) -> tuple[dict | None, dict]:
        opp_seat = 1 - seat
        d_pair = [opp_deck_idx, opp_deck_idx]
        d_pair[seat] = own_deck_idx
        d0, d1 = decks[d_pair[0]], decks[d_pair[1]]
        s0, s1 = forms[d_pair[0]], forms[d_pair[1]]

        play_env.vector.reset_env(0, seed, d0, d1, s0, s1)
        schema = play_env.vector.schema
        tiles = int(schema.cols) * int(schema.rows)

        rage_ids = tuple(
            i for i, name in enumerate(play_env.vector.vocabulary) if name == "Rage"
        )
        stream = LookaheadPolicyStream(
            actor,
            adapter,
            device,
            critic=critic,
            search_env=search_env,
            config=cfg,
            precision=precision,
            seed=int(self.rng.integers(1, 2**31 - 1)),
            rage_card_ids=rage_ids,
        )
        stream.bind_match(seed=seed, deck0=d0, deck1=d1, slots0=s0, slots1=s1, seat=seat)
        opp_temp = 0.50 if (self.games_played % 2 == 1) else 0.0
        opp_stream = _GreedyMirrorStream(
            actor, adapter, device, precision=precision, temperature=opp_temp, rng=self.rng
        )

        traj_obs: list[np.ndarray] = []
        traj_masks: list[np.ndarray] = []
        traj_acts: list[int] = []
        traj_searched: list[bool] = []
        traj_overridden: list[bool] = []
        traj_bank_wait: list[bool] = []
        traj_adverse: list[bool] = []
        traj_gains: list[float] = []
        override_cards: list[str] = []

        steps = 0
        while not play_env.dones[0] and steps < max_game_steps:
            raw_own = play_env.observations[0, seat].copy()
            leg_own = play_env.masks[0, seat].copy()
            raw_opp = play_env.observations[0, opp_seat].copy()
            leg_opp = play_env.masks[0, opp_seat].copy()

            a_own, _ = stream.act(raw_own, leg_own, opp_raw=raw_opp, opp_legal=leg_opp)
            a_opp = opp_stream.act(raw_opp, leg_opp)
            meta = stream.last_step_info

            traj_obs.append(raw_own)
            traj_masks.append(leg_own)
            traj_acts.append(int(a_own))
            traj_searched.append(bool(meta.get("searched", False)))
            is_ovr = bool(meta.get("overridden", False))
            is_bank_w = bool(meta.get("bank_wait", False))
            traj_overridden.append(is_ovr)
            traj_bank_wait.append(is_bank_w)
            traj_adverse.append(_is_adverse_frame(raw_own, schema))
            traj_gains.append(float(meta.get("score_gain", 0.0)))
            if is_ovr:
                if a_own == 0:
                    bs = meta.get("bank_slot")
                    if bs is not None:
                        bcid = int(raw_own[schema.header["hand_card"][int(bs)]])
                        bname = play_env.vector.vocabulary[bcid] if bcid >= 0 else f"slot{bs}"
                        prefix = "HOLD" if meta.get("hold_full") else "BANK"
                        tag = f"{prefix}->{bname}"
                        if not override_cards or override_cards[-1] != tag:
                            override_cards.append(tag)
                    else:
                        override_cards.append("WAIT")
                elif 0 < a_own < 1 + 4 * tiles:
                    slot = (a_own - 1) // tiles
                    cid = int(raw_own[schema.header["hand_card"][slot]])
                    override_cards.append(play_env.vector.vocabulary[cid])
                else:
                    override_cards.append(f"ABILITY_{a_own}")

            pair_acts = [0, 0]
            pair_acts[seat] = int(a_own)
            pair_acts[opp_seat] = int(a_opp)
            stream.record_step_actions(pair_acts[0], pair_acts[1])
            play_env.vector.actions[0, 0] = pair_acts[0]
            play_env.vector.actions[0, 1] = pair_acts[1]
            play_env.vector.step_lanes([0])
            steps += 1

        info = play_env.vector.info(0)
        obs_end = play_env.observations[0, seat]
        cd = int(schema.field(obs_end, "crowns") - schema.field(obs_end, "crowns_enemy"))
        faulted = bool(info.get("fault") or not play_env.dones[0])
        won = (not faulted) and (info.get("winner") == seat)
        tied = (not faulted) and (info.get("winner") not in (0, 1))
        n_ovr = int(sum(traj_overridden))
        n_srch = int(sum(traj_searched))

        report = dict(
            deck=deck_names[own_deck_idx],
            opp_deck=deck_names[opp_deck_idx],
            seat=int(seat),
            steps=steps,
            won=bool(won),
            tied=bool(tied),
            faulted=faulted,
            crown_diff=cd,
            searches=n_srch,
            overrides=n_ovr,
            override_cards=override_cards[:16],
            search_seconds=round(float(stream.stats["total_search_seconds"]), 2),
        )

        # Admit winning/tied episodes, plus contested -1 crown games (strictly override-only weights!)
        adverse_ovr = int(sum(1 for o, adv in zip(traj_overridden, traj_adverse) if o and adv))
        contested_adverse = (not faulted) and (not won) and (not tied) and (cd >= -1) and (n_ovr >= 3) and (adverse_ovr >= 2)
        admit = (not faulted) and (n_ovr >= 1) and (won or (tied and n_ovr >= 2) or contested_adverse)
        if not admit or not traj_obs:
            report["status"] = "rejected_outcome_or_no_override"
            return None, report

        raw_arr = np.stack(traj_obs).astype(np.int32)
        mask_arr = np.stack(traj_masks).astype(bool)
        act_arr = np.asarray(traj_acts, dtype=np.int64)
        keep_arr = mask_arr.sum(-1) > 1
        if not keep_arr.any():
            report["status"] = "no_choice_frames"
            return None, report

        weights_arr = lookahead_frame_weights(
            act_arr,
            keep_arr,
            np.asarray(traj_searched, dtype=bool),
            np.asarray(traj_overridden, dtype=bool),
            override_boost=1.8,
            bank_wait_mask=np.asarray(traj_bank_wait, dtype=bool),
            adverse_mask=np.asarray(traj_adverse, dtype=bool),
            override_only=contested_adverse,
            pure_override=not contested_adverse,
            score_gains=np.asarray(traj_gains, dtype=np.float32),
        )
        report["status"] = "admitted_adverse_overrides" if contested_adverse else "admitted"
        episode = dict(
            raw_obs=raw_arr,
            legal_masks=mask_arr,
            actions=act_arr,
            actor_keep=keep_arr,
            weights=weights_arr,
        )
        return episode, report

    @staticmethod
    def _sample_opp_action_from_logp(
        opp_lp_1d: torch.Tensor,
        opp_legal_1d: np.ndarray,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> int:
        legal_b = np.asarray(opp_legal_1d, dtype=bool)
        mask_t = torch.as_tensor(legal_b, device=opp_lp_1d.device, dtype=torch.bool)
        masked = opp_lp_1d.masked_fill(~mask_t, -torch.inf)
        if temperature > 1e-3 and rng is not None and int(legal_b.sum()) > 1:
            scaled = masked.float() / float(temperature)
            scaled = scaled - scaled.max()
            probs = torch.where(mask_t, scaled.exp(), torch.zeros_like(scaled)).cpu().numpy()
            s = float(probs.sum())
            if s > 0 and np.isfinite(s):
                probs = probs / s
                choice = int(rng.choice(len(probs), p=probs))
                if legal_b[choice]:
                    return choice
        return int(masked.argmax(dim=-1).item())

    @torch.inference_mode()
    def _play_distill_batch(
        self,
        play_env: CrForgeVector,
        search_env: CrForgeVector,
        actor,
        critic,
        adapter,
        *,
        specs: list[dict],
        decks,
        forms,
        deck_names,
        device: torch.device,
        precision: str,
        cfg: LookaheadConfig,
        max_game_steps: int,
    ) -> list[tuple[dict | None, dict]]:
        """Play `len(specs)` distillation matches in parallel with fused Actor/Critic steps & batched lookahead."""
        w_count = len(specs)
        if w_count < 1:
            return []
        dev = torch.device(device)
        if critic is not None:
            try:
                critic_device = next(critic.parameters()).device
            except StopIteration:
                critic_device = dev
        else:
            critic_device = dev

        schema = play_env.vector.schema
        tiles = int(schema.cols) * int(schema.rows)
        hand_indices = list(adapter.schema.header["hand_card"])
        rage_ids = tuple(i for i, name in enumerate(play_env.vector.vocabulary) if name == "Rage")

        streams: list[LookaheadPolicyStream] = []
        for g_idx, sp in enumerate(specs):
            seat = int(sp["seat"])
            d_pair = [int(sp["opp_deck_idx"]), int(sp["opp_deck_idx"])]
            d_pair[seat] = int(sp["own_deck_idx"])
            d0, d1 = decks[d_pair[0]], decks[d_pair[1]]
            s0, s1 = forms[d_pair[0]], forms[d_pair[1]]
            play_env.vector.reset_env(g_idx, int(sp["seed"]), d0, d1, s0, s1)
            st = LookaheadPolicyStream(
                actor,
                adapter,
                dev,
                critic=critic,
                search_env=search_env,
                config=cfg,
                precision=precision,
                seed=int(sp["search_seed"]),
                rage_card_ids=rage_ids,
            )
            st.bind_match(seed=int(sp["seed"]), deck0=d0, deck1=d1, slots0=s0, slots1=s1, seat=seat)
            streams.append(st)

        actor_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if (precision == "fp16" and dev.type == "cuda" and not getattr(actor, "ppo_actor_fp32", False))
            else nullcontext()
        )
        critic_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if (precision == "fp16" and critic_device.type == "cuda")
            else nullcontext()
        )
        with actor_ctx:
            actor_mem = actor.initial_state(2 * w_count, device=dev)
        if critic is not None:
            with critic_ctx:
                crit_mem = critic.initial_state(w_count, device=critic_device)
        else:
            crit_mem = None

        traj_obs: list[list[np.ndarray]] = [[] for _ in range(w_count)]
        traj_masks: list[list[np.ndarray]] = [[] for _ in range(w_count)]
        traj_acts: list[list[int]] = [[] for _ in range(w_count)]
        traj_searched: list[list[bool]] = [[] for _ in range(w_count)]
        traj_overridden: list[list[bool]] = [[] for _ in range(w_count)]
        traj_bank_wait: list[list[bool]] = [[] for _ in range(w_count)]
        traj_adverse: list[list[bool]] = [[] for _ in range(w_count)]
        override_cards: list[list[str]] = [[] for _ in range(w_count)]
        step_counts = [0] * w_count

        active_slots = list(range(w_count))
        n_search_cap = int(search_env.count)

        for _step_num in range(int(max_game_steps)):
            keep_pos = [pos for pos, g in enumerate(active_slots) if not play_env.dones[g]]
            if not keep_pos:
                break
            if len(keep_pos) < len(active_slots):
                old_a = len(active_slots)
                actor_keep = keep_pos + [old_a + pos for pos in keep_pos]
                actor_mem = slice_memory(actor_mem, actor_keep)
                if crit_mem is not None:
                    crit_mem = slice_memory(crit_mem, keep_pos)
                active_slots = [active_slots[pos] for pos in keep_pos]

            a_cnt = len(active_slots)
            lanes_arr = np.asarray(active_slots, dtype=np.int32)
            seats_arr = np.asarray([int(specs[g]["seat"]) for g in active_slots], dtype=np.int32)
            opp_seats_arr = 1 - seats_arr

            own_raw_b = play_env.observations[lanes_arr, seats_arr].copy()
            own_mask_b = play_env.masks[lanes_arr, seats_arr].copy()
            opp_raw_b = play_env.observations[lanes_arr, opp_seats_arr].copy()
            opp_mask_b = play_env.masks[lanes_arr, opp_seats_arr].copy()

            # Single fused Actor forward for both learner & mirror-opponent streams [2 * A]
            both_raw = np.concatenate([own_raw_b, opp_raw_b], axis=0)
            both_mask = np.concatenate([own_mask_b, opp_mask_b], axis=0)
            bh, be, bm = adapter(both_raw)
            bh_t = torch.as_tensor(bh, device=dev)
            be_t = torch.as_tensor(be, device=dev)
            bm_t = torch.as_tensor(bm, device=dev)
            bmask_t = torch.as_tensor(both_mask, dtype=torch.bool, device=dev)
            bhand_t = torch.as_tensor(both_raw[:, hand_indices], device=dev)

            actor_mem = reserve_step_memory(actor_mem, 128)
            with actor_ctx:
                both_logp, _, actor_mem = actor.step(
                    bh_t, be_t, bm_t, bmask_t, state=actor_mem, hand_card_ids=bhand_t, compute_value=False
                )
            if critic is not None and crit_mem is not None:
                ch_t = bh_t[:a_cnt] if critic_device == dev else bh_t[:a_cnt].to(critic_device)
                ce_t = be_t[:a_cnt] if critic_device == dev else be_t[:a_cnt].to(critic_device)
                cm_t = bm_t[:a_cnt] if critic_device == dev else bm_t[:a_cnt].to(critic_device)
                crit_mem = reserve_step_memory(crit_mem, 128)
                with critic_ctx:
                    _, crit_mem = critic.step(ch_t, ce_t, cm_t, state=crit_mem)

            own_logp = both_logp[:a_cnt]
            opp_logp = both_logp[a_cnt:]

            a_own_list = [0] * a_cnt
            a_opp_list = [0] * a_cnt
            search_tasks: list[dict] = []

            for pos, g in enumerate(active_slots):
                sp = specs[g]
                a_opp_list[pos] = self._sample_opp_action_from_logp(
                    opp_logp[pos], opp_mask_b[pos], float(sp["opp_temp"]), sp["opp_rng"]
                )
                base_act, cands, pending = streams[g].prepare_step_from_logp(
                    own_raw_b[pos], own_mask_b[pos], own_logp[pos]
                )
                if cands is not None:
                    search_tasks.append(
                        dict(
                            pos=pos,
                            mem_slot=pos,
                            source_vector=play_env.vector,
                            source_lane=g,
                            stream=streams[g],
                            candidates=cands,
                            root_logp=own_logp[pos],
                            root_raw=own_raw_b[pos],
                            root_legal=own_mask_b[pos],
                            baseline_action=base_act,
                            pending=pending,
                        )
                    )
                else:
                    a_own_list[pos] = int(base_act)

            if search_tasks:
                # Pack concurrent search triggers into batches up to `search_env.count` lanes
                idx_t = 0
                r_roll = int(cfg.rollouts_per_candidate)
                while idx_t < len(search_tasks):
                    chunk: list[dict] = []
                    lanes_acc = 0
                    while idx_t < len(search_tasks):
                        need_l = len(search_tasks[idx_t]["candidates"]) * r_roll
                        if chunk and lanes_acc + need_l > n_search_cap:
                            break
                        chunk.append(search_tasks[idx_t])
                        lanes_acc += need_l
                        idx_t += 1
                    t_s = time.perf_counter()
                    batch_out = run_batched_lookahead(
                        chunk,
                        search_env=search_env,
                        actor=actor,
                        critic=critic,
                        adapter=adapter,
                        actor_mem=actor_mem,
                        opp_mem_offset=a_cnt,
                        critic_mem=crit_mem,
                        cfg=cfg,
                        device=dev,
                        critic_device=critic_device,
                        precision=precision,
                    )
                    per_task_sec = (time.perf_counter() - t_s) / max(1, len(chunk))
                    for task, (raw_choice, gain) in zip(chunk, batch_out, strict=True):
                        pos = int(task["pos"])
                        a_own_list[pos] = task["stream"].apply_search_result(
                            task["root_raw"], raw_choice, gain, task["pending"], per_task_sec
                        )

            for pos, g in enumerate(active_slots):
                seat = int(specs[g]["seat"])
                opp_seat = 1 - seat
                a_own = int(a_own_list[pos])
                a_opp = int(a_opp_list[pos])
                raw_own = own_raw_b[pos]
                meta = streams[g].last_step_info

                traj_obs[g].append(raw_own)
                traj_masks[g].append(own_mask_b[pos])
                traj_acts[g].append(a_own)
                traj_searched[g].append(bool(meta.get("searched", False)))
                is_ovr = bool(meta.get("overridden", False) or (meta.get("unlocked_bank_slot") is not None and a_own > 0))
                is_bank_w = bool(meta.get("bank_wait", False))
                traj_overridden[g].append(is_ovr)
                traj_bank_wait[g].append(is_bank_w)
                traj_adverse[g].append(_is_adverse_frame(raw_own, schema))
                if is_ovr:
                    ovr_cards = override_cards[g]
                    if a_own == 0:
                        bs = meta.get("bank_slot")
                        if bs is not None:
                            bcid = int(raw_own[schema.header["hand_card"][int(bs)]])
                            bname = play_env.vector.vocabulary[bcid] if bcid >= 0 else f"slot{bs}"
                            prefix = "HOLD" if meta.get("hold_full") else "BANK"
                            tag = f"{prefix}->{bname}"
                            if not ovr_cards or ovr_cards[-1] != tag:
                                ovr_cards.append(tag)
                        else:
                            ovr_cards.append("WAIT")
                    elif 0 < a_own < 1 + 4 * tiles:
                        slot = (a_own - 1) // tiles
                        cid = int(raw_own[schema.header["hand_card"][slot]])
                        ovr_cards.append(play_env.vector.vocabulary[cid])
                    else:
                        ovr_cards.append(f"ABILITY_{a_own}")

                pair_acts = [0, 0]
                pair_acts[seat] = a_own
                pair_acts[opp_seat] = a_opp
                streams[g].record_step_actions(pair_acts[0], pair_acts[1])
                play_env.vector.actions[g, 0] = pair_acts[0]
                play_env.vector.actions[g, 1] = pair_acts[1]
                step_counts[g] += 1

            play_env.vector.step_lanes(active_slots)

        results: list[tuple[dict | None, dict]] = []
        for g_idx, sp in enumerate(specs):
            seat = int(sp["seat"])
            own_deck_idx = int(sp["own_deck_idx"])
            opp_deck_idx = int(sp["opp_deck_idx"])
            info = play_env.vector.info(g_idx)
            obs_end = play_env.observations[g_idx, seat]
            cd = int(schema.field(obs_end, "crowns") - schema.field(obs_end, "crowns_enemy"))
            faulted = bool(info.get("fault") or not play_env.dones[g_idx])
            won = (not faulted) and (info.get("winner") == seat)
            tied = (not faulted) and (info.get("winner") not in (0, 1))
            n_ovr = int(sum(traj_overridden[g_idx]))
            n_srch = int(sum(traj_searched[g_idx]))

            report = dict(
                deck=deck_names[own_deck_idx],
                opp_deck=deck_names[opp_deck_idx],
                seat=seat,
                steps=step_counts[g_idx],
                won=bool(won),
                tied=bool(tied),
                faulted=faulted,
                crown_diff=cd,
                searches=n_srch,
                overrides=n_ovr,
                override_cards=override_cards[g_idx][:16],
                search_seconds=round(float(streams[g_idx].stats["total_search_seconds"]), 2),
            )
            adverse_ovr = int(sum(1 for o, adv in zip(traj_overridden[g_idx], traj_adverse[g_idx]) if o and adv))
            contested_adverse = (
                (not faulted)
                and (not won)
                and (not tied)
                and (cd >= -1)
                and (n_ovr >= 3)
                and (adverse_ovr >= 2)
            )
            admit = (not faulted) and (n_ovr >= 1) and (won or (tied and n_ovr >= 2) or contested_adverse)
            if not admit or not traj_obs[g_idx]:
                report["status"] = "rejected_outcome_or_no_override"
                results.append((None, report))
                continue

            raw_arr = np.stack(traj_obs[g_idx]).astype(np.int32)
            mask_arr = np.stack(traj_masks[g_idx]).astype(bool)
            act_arr = np.asarray(traj_acts[g_idx], dtype=np.int64)
            keep_arr = mask_arr.sum(-1) > 1
            if not keep_arr.any():
                report["status"] = "no_choice_frames"
                results.append((None, report))
                continue

            weights_arr = lookahead_frame_weights(
                act_arr,
                keep_arr,
                np.asarray(traj_searched[g_idx], dtype=bool),
                np.asarray(traj_overridden[g_idx], dtype=bool),
                override_boost=1.8,
                bank_wait_mask=np.asarray(traj_bank_wait[g_idx], dtype=bool),
                adverse_mask=np.asarray(traj_adverse[g_idx], dtype=bool),
                override_only=contested_adverse,
                pure_override=not contested_adverse,
            )
            report["status"] = "admitted_adverse_overrides" if contested_adverse else "admitted"
            episode = dict(
                raw_obs=raw_arr,
                legal_masks=mask_arr,
                actions=act_arr,
                actor_keep=keep_arr,
                weights=weights_arr,
            )
            results.append((episode, report))
        return results

    def run_phase(
        self,
        actor,
        critic,
        adapter,
        probes,
        *,
        env_paths,
        decks,
        forms,
        deck_names,
        update: int,
        device,
        precision: str = "fp16",
        games: int = 3,
        parallel_games: int = 12,
        max_parallel_searches: int = 4,
        candidates: int = 4,
        rollouts: int = 3,
        horizon_steps: int = 10,
        trigger_elixir: float = 6.0,
        defense_elixir: float = 2.5,
        override_margin: float = 0.008,
        max_searches_per_game: int = 16,
        max_game_steps: int = 800,
        batch_episodes: int = 4,
        optimizer_steps: int = 1,
        lr: float = 1e-6,
        max_kl: float = MAX_KL,
        threads: int = 32,
        deck_probabilities=None,
        focal_deck_indices: list[int] | None = None,
        include_placement: bool = True,
        include_last_temporal: bool = True,
        temporal_blocks: int | None = None,
    ) -> dict:
        if not probes or not any(np.any(ep["actor_keep"]) for ep in probes):
            return dict(ran=False, reason="no_ppo_choice_probes")
        if update <= self.last_update:
            return dict(ran=False, reason="already_processed")
        if not 0 < lr <= MAX_DISTILL_LR or not 0 < max_kl <= MAX_DISTILL_KL or int(optimizer_steps) < 1:
            raise ValueError(
                f"Lookahead distillation requires 0 < lr <= {MAX_DISTILL_LR}, 0 < max_kl <= {MAX_DISTILL_KL}, and steps >= 1"
            )

        began = time.perf_counter()
        old_mode = actor.training
        actor.eval()
        if critic is not None:
            critic.eval()

        eff_defense = min(float(defense_elixir), float(trigger_elixir) - 0.5)
        cfg = LookaheadConfig(
            max_candidates=max(2, int(candidates)),
            rollouts_per_candidate=max(1, int(rollouts)),
            horizon_steps=max(2, int(horizon_steps)),
            trigger_elixir=float(trigger_elixir),
            defense_elixir=max(2.0, eff_defense),
            entropy_threshold=0.35,
            min_search_interval_steps=4,
            max_searches_per_game=max(1, int(max_searches_per_game)),
            prior_weight=0.005,
            override_margin=float(override_margin),
            threads=max(2, int(threads)),
        )
        total_games = max(1, int(games))
        n_play = min(total_games, max(1, int(parallel_games)))
        lanes_per_search = cfg.max_candidates * cfg.rollouts_per_candidate
        n_search_lanes = max(1, min(n_play, max(1, int(max_parallel_searches)))) * lanes_per_search
        n_decks = len(decks)
        probs = deck_probabilities if deck_probabilities is not None else np.ones(n_decks) / n_decks
        counter_cards = {
            "GoblinBarrel",
            "MegaKnight",
            "HogRider",
            "Balloon",
            "Princess",
            "SkeletonArmy",
            "WallBreakers",
        }
        opp_weights = np.asarray(probs, dtype=np.float64).copy()
        for d_i, d_cards in enumerate(decks):
            if any(str(c) in counter_cards for c in d_cards):
                opp_weights[d_i] *= 2.5
        opp_probs = opp_weights / max(float(opp_weights.sum()), 1e-12)

        all_specs: list[dict] = []
        for g_idx in range(total_games):
            seed = int(self.rng.integers(1, 2**31 - 1))
            search_seed = int(self.rng.integers(1, 2**31 - 1))
            opp_seed = int(self.rng.integers(1, 2**31 - 1))
            seat = int((self.games_played + g_idx) % 2)
            own_idx = (
                int(self.rng.choice(focal_deck_indices))
                if focal_deck_indices
                else int(self.rng.choice(n_decks, p=probs))
            )
            opp_idx = int(self.rng.choice(n_decks, p=opp_probs))
            opp_temp = 0.50 if ((self.games_played + g_idx) % 2 == 1) else 0.0
            all_specs.append(
                dict(
                    seed=seed,
                    search_seed=search_seed,
                    opp_rng=np.random.default_rng(opp_seed),
                    opp_temp=opp_temp,
                    seat=seat,
                    own_deck_idx=own_idx,
                    opp_deck_idx=opp_idx,
                )
            )

        game_reports = []
        admitted_this_phase = 0
        wins_this_phase = 0
        with (
            CrForgeVector(n_play, min(cfg.threads, n_play), paths=env_paths, independent=True, arena="Arena_L", decision_ticks=8) as play_env,
            CrForgeVector(n_search_lanes, min(cfg.threads, n_search_lanes), paths=env_paths, independent=True, arena="Arena_L", decision_ticks=8) as search_env,
        ):
            play_env.set_autoreset(False)
            search_env.set_autoreset(False)
            for start_g in range(0, total_games, n_play):
                wave_specs = all_specs[start_g : start_g + n_play]
                wave_out = self._play_distill_batch(
                    play_env,
                    search_env,
                    actor,
                    critic,
                    adapter,
                    specs=wave_specs,
                    decks=decks,
                    forms=forms,
                    deck_names=deck_names,
                    device=torch.device(device),
                    precision=precision,
                    cfg=cfg,
                    max_game_steps=int(max_game_steps),
                )
                for ep, rep in wave_out:
                    self.games_played += 1
                    self.total_overrides += int(rep.get("overrides", 0))
                    if rep.get("won"):
                        wins_this_phase += 1
                    game_reports.append(rep)
                    if ep is not None:
                        self._save_episode(
                            ep,
                            dict(
                                update=int(update),
                                deck=rep["deck"],
                                opp_deck=rep["opp_deck"],
                                crown_diff=rep["crown_diff"],
                                overrides=rep["overrides"],
                                override_cards=rep["override_cards"],
                            ),
                        )
                        admitted_this_phase += 1
                        self.admitted_total += 1

        search_seconds = time.perf_counter() - began
        self.last_update = int(update)
        self.phase_count += 1

        if not self.entries or (admitted_this_phase == 0 and len(self.entries) < 2):
            actor.train(old_mode)
            return dict(
                ran=True,
                accepted=False,
                actor_attempted=False,
                memory_invalidated=False,
                changed_parameters=[],
                reason="insufficient_admitted_episodes",
                games_run=len(game_reports),
                parallel_games=n_play,
                search_lanes=n_search_lanes,
                wins_this_phase=wins_this_phase,
                admitted_this_phase=admitted_this_phase,
                admitted_total=self.admitted_total,
                total_overrides=self.total_overrides,
                buffer_size=len(self.entries),
                game_reports=game_reports,
                search_seconds=search_seconds,
                phase_count=self.phase_count,
            )

        k_distill = min(int(batch_episodes), len(self.entries))
        chosen_idx = self.rng.choice(len(self.entries), size=k_distill, replace=False)
        chosen_entries = [self.entries[int(i)] for i in chosen_idx]
        distill_eps, distill_w = self._load_episodes(chosen_entries)

        anchor_probes = list(probes[: min(len(probes), max(2, k_distill // 2))])
        anchor_w = [0.05 * probe_anchor_weights(ep) for ep in anchor_probes]
        train_eps = distill_eps + anchor_probes
        train_w = distill_w + anchor_w

        parameters = dict(actor.named_parameters())
        original_requires = {name: p.requires_grad for name, p in parameters.items()}
        originals = {}
        last_valid_weights = None
        reasons = []
        steps_attempted = 0
        accepted_steps = 0
        probe_before = probe_after = last_valid_probe = None

        try:
            names = unfreeze_distill_heads(
                actor,
                include_placement=include_placement,
                include_last_temporal=include_last_temporal,
                temporal_blocks=temporal_blocks,
            )
            originals = {name: parameters[name].detach().clone() for name in names}
            readout_params = [
                parameters[n]
                for n in names
                if not n.startswith(("learned_placement.", "tile_", "temporal."))
            ]
            tile_params = [parameters[n] for n in names if n.startswith(("learned_placement.", "tile_"))]
            temporal_params = [parameters[n] for n in names if n.startswith("temporal.")]
            param_groups = [{"params": readout_params, "lr": lr}]
            if tile_params:
                # Unfreeze spatial cross-attention + CNN residual at 0.22x LR so tactical placements stay sharp
                param_groups.append({"params": tile_params, "lr": lr * 0.22})
            if temporal_params:
                # Top temporal blocks learn cross-step multi-card synergy & elixir counting at 0.15x LR
                param_groups.append({"params": temporal_params, "lr": lr * 0.15})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda", init_scale=128.0)
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            actor.eval()
            probe_before, reference = _score(actor, adapter, probes, device, precision, capture=True)
            for _step_i in range(max(1, int(optimizer_steps))):
                _train_distill_accumulated(actor, adapter, train_eps, train_w, optimizer, scaler, device, precision)
                steps_attempted += 1
                cand_probe, _ = _score(actor, adapter, probes, device, precision, reference=reference)
                step_reasons = []
                if cand_probe["reference_forward_kl"] > max_kl:
                    step_reasons.append("ppo_forward_kl_limit")
                wait_delta = cand_probe["mean_p_wait"] - probe_before["mean_p_wait"]
                if (
                    abs(wait_delta) > MAX_DISTILL_WAIT_DRIFT
                    or (wait_delta > 0.0 and cand_probe["mean_p_wait"] > MAX_DISTILL_MEAN_P_WAIT)
                    or (wait_delta < 0.0 and cand_probe["mean_p_wait"] < MIN_DISTILL_MEAN_P_WAIT)
                ):
                    step_reasons.append("mean_p_wait_drift")
                if step_reasons:
                    reasons = step_reasons
                    probe_after = cand_probe
                    break
                accepted_steps = steps_attempted
                last_valid_probe = cand_probe
                probe_after = cand_probe
                last_valid_weights = {name: parameters[name].detach().clone() for name in names}
        except FloatingPointError:
            reasons = ["nonfinite_candidate"]
        except BaseException:
            with torch.no_grad():
                for name, orig in originals.items():
                    parameters[name].copy_(orig)
            if originals:
                actor.mark_weights_updated()
            raise
        finally:
            for name, p in parameters.items():
                p.requires_grad_(original_requires[name])
                p.grad = None
            actor.train(old_mode)

        if accepted_steps > 0 and last_valid_weights is not None:
            # Keep the furthest trust-region-valid optimizer step even if a later step hit the KL/WAIT gate!
            if accepted_steps < steps_attempted or reasons:
                with torch.no_grad():
                    for name, w_ok in last_valid_weights.items():
                        parameters[name].copy_(w_ok)
                actor.mark_weights_updated()
                probe_after = last_valid_probe
                reasons = []
            accepted = True
        else:
            accepted = False
            if originals:
                with torch.no_grad():
                    for name, orig in originals.items():
                        parameters[name].copy_(orig)
                actor.mark_weights_updated()

        before_w = probe_before["mean_p_wait"] if probe_before is not None else None
        after_w = probe_after["mean_p_wait"] if probe_after is not None else None
        return dict(
            ran=True,
            accepted=accepted,
            rejected_reasons=reasons,
            actor_attempted=bool(steps_attempted),
            memory_invalidated=bool(steps_attempted),
            changed_parameters=list(originals) if accepted else [],
            games_run=len(game_reports),
            parallel_games=n_play,
            search_lanes=n_search_lanes,
            wins_this_phase=wins_this_phase,
            admitted_this_phase=admitted_this_phase,
            admitted_total=self.admitted_total,
            total_overrides=self.total_overrides,
            buffer_size=len(self.entries),
            trained_distill_episodes=k_distill,
            trained_anchor_probes=len(anchor_probes),
            optimizer_steps=accepted_steps if accepted else steps_attempted,
            ppo_forward_kl=probe_after["reference_forward_kl"] if probe_after is not None else None,
            mean_p_wait_before=before_w,
            mean_p_wait_after=after_w,
            wait_prob_delta=None if after_w is None else after_w - before_w,
            game_reports=game_reports,
            search_seconds=search_seconds,
            phase_count=self.phase_count,
        )
