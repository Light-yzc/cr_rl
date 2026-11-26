"""Universal 4x4 two-card sequence rollout search (Expert Iteration) for v3 PPO.

Discovers multi-card delayed synergies (e.g. back-row Sparky -> front Giant,
Golem -> Night Witch, Lava Hound -> Balloon) across all 51 decks with zero
card-specific hardcoding, stores high-advantage trajectories on local SSD, and
distills them into Actor readouts under strict KL and WAIT-drift gates.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models.state_policy import reserve_step_memory, slice_memory
from cr_rl.training.expert_bc_plan_b import MAX_KL, MAX_WAIT_DRIFT, _score, _train_accumulated
from cr_rl.training.inference import StateInference
from cr_rl.training.live_awbc import freeze_actor_readouts
from cr_rl.training.persistence import atomic_json

LANES = 18  # Wave 1: 16 (4x4 slot pairs) + 2 original; Wave 2: 3 arms x 6 rollouts
CONFIRM_ROLLOUTS = 6
GLOBAL_SPELL_TILE_THRESHOLD = 500


def low_pressure(schema, obs) -> bool:
    if int(schema.field(obs, "entity_overflow")) != 0:
        return False
    e = schema.entity
    rows = schema.entities(obs)
    enemy = rows[rows[:, e["side"]] == 1]
    kinds = enemy[:, e["kind"]]
    incoming = (kinds == schema.kinds["projectile"]) | (kinds == schema.kinds["area"])
    near = (
        (kinds == schema.kinds["character"])
        & ((enemy[:, e["flags"]] & schema.flags["alive"]) != 0)
        & (enemy[:, e["hp"]] > 0)
        & (enemy[:, e["y"]] <= schema.height_units // 2 + 2 * schema.tile_units)
    )
    return not bool((incoming | near).any())


def emergency_defense_needed(schema, obs) -> bool:
    e = schema.entity
    rows = schema.entities(obs)
    deep = (
        (rows[:, e["side"]] == 1)
        & (rows[:, e["kind"]] == schema.kinds["character"])
        & ((rows[:, e["flags"]] & schema.flags["alive"]) != 0)
        & (rows[:, e["hp"]] > 0)
        & (rows[:, e["y"]] <= 9500)
    )
    return bool(deep.any())


def tower_hp_margin(schema, obs) -> int:
    own = sum(int(obs[schema.header[f"own_{t}_hp"][0]]) for t in ("left", "right", "king"))
    enemy = sum(int(obs[schema.header[f"enemy_{t}_hp"][0]]) for t in ("left", "right", "king"))
    return own - enemy


def sample_with_uniforms(logits: torch.Tensor, uniforms: torch.Tensor) -> np.ndarray:
    probs = logits.softmax(-1)
    cdf = probs.cumsum(-1)
    chosen = torch.searchsorted(cdf, uniforms * cdf[:, -1:], right=True).squeeze(1)
    return chosen.cpu().numpy().astype(np.int32)


def combo_frame_weights(actions: np.ndarray, actor_keep: np.ndarray, macro_play: np.ndarray, macro_wait: np.ndarray) -> np.ndarray:
    """Plan-B-compatible weights: boost discovered combo plays/waits, damp ordinary WAIT to 0.06."""
    keep = np.asarray(actor_keep, dtype=bool)
    w = np.where(actions > 0, 1.0, 0.06)
    w = np.where(macro_wait & (actions == 0), 0.5, w)
    w = np.where(macro_play & (actions > 0), 1.5, w)
    return np.where(keep, w, 0.0).astype(np.float32)


def probe_anchor_weights(episode: dict) -> np.ndarray:
    keep = np.asarray(episode["actor_keep"], dtype=bool)
    actions = np.asarray(episode["actions"])
    w = np.where(actions > 0, 1.0, 0.06)
    return np.where(keep, w, 0.0).astype(np.float32)


def _apply_pair_mask(
    schema,
    row,
    native_row: np.ndarray,
    behavior_row: np.ndarray,
    state: dict,
    step: int,
    t_star: int,
    tiles: int,
    scale: int,
) -> None:
    """Advance universal 2-card macro option (slot_1 -> slot_2) on a single learner seat."""
    ph = state["phase"]
    if ph == "done":
        return

    pending = int(schema.field(row, "pending_count"))
    avail = int(schema.field(row, "elixir")) - int(schema.field(row, "elixir_reserved"))

    if ph == "play_first":
        if step - t_star > 22 or not low_pressure(schema, row):
            state["phase"] = "done"
            return
        if pending != 0:
            return
        slot1 = state["slot1"]
        first = 1 + slot1 * tiles
        s1_native = native_row[first : first + tiles]
        if s1_native.any():
            if int(s1_native.sum()) >= GLOBAL_SPELL_TILE_THRESHOLD:
                # Never force naked global damage spells onto an empty low-pressure arena.
                state["phase"] = "done"
                return
            behavior_row[:] = False
            behavior_row[first : first + tiles] = s1_native
            state["armed_play"] = 1
        elif native_row[schema.wait]:
            behavior_row[:] = False
            behavior_row[schema.wait] = True
            state["armed_wait"] = True
        else:
            state["phase"] = "done"

    elif ph == "play_second":
        t1 = state["t1"]
        if step - t1 > 40 or emergency_defense_needed(schema, row):
            state["phase"] = "done"
            return
        if step <= t1 + 2 or pending != 0:
            return
        slot2 = state["slot2"]
        cost2 = int(schema.field(row, "hand_cost")[slot2])
        if cost2 <= 0:
            return
        first2 = 1 + slot2 * tiles
        s2_native = native_row[first2 : first2 + tiles]
        if s2_native.any():
            if int(s2_native.sum()) >= GLOBAL_SPELL_TILE_THRESHOLD:
                state["phase"] = "done"
                return
            behavior_row[:] = False
            behavior_row[first2 : first2 + tiles] = s2_native
            state["armed_play"] = 2
        elif avail >= max(25 * scale // 10, (cost2 - 2) * scale) and native_row[schema.wait]:
            behavior_row[:] = False
            behavior_row[schema.wait] = True
            state["armed_wait"] = True


class MacroCardSearch:
    """Two-stage 4x4 card-sequence rollout search + gated readout distillation."""

    def __init__(self, out_dir: str | Path, *, max_buffer_episodes: int = 128, seed: int = 20261121, state: dict | None = None):
        self.buffer_dir = Path(out_dir).resolve() / "macro_search_buffer"
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_episodes = int(max_buffer_episodes)
        self.rng = np.random.default_rng(seed)
        self.last_update = -1
        self.phase_count = 0
        self.searched_branches = 0
        self.admitted_total = 0
        self.next_shard_id = 0
        self.entries: list[dict] = []
        if state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
            self.last_update = int(state.get("last_update", -1))
            self.phase_count = int(state.get("phase_count", 0))
            self.searched_branches = int(state.get("searched_branches", 0))
            self.admitted_total = int(state.get("admitted_total", 0))
            self.next_shard_id = int(state.get("next_shard_id", 0))
            for item in state.get("entries", []):
                if (self.buffer_dir / item["filename"]).is_file():
                    self.entries.append(dict(item))

    def state_dict(self) -> dict:
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state),
            last_update=self.last_update,
            phase_count=self.phase_count,
            searched_branches=self.searched_branches,
            admitted_total=self.admitted_total,
            next_shard_id=self.next_shard_id,
            entries=copy.deepcopy(self.entries),
        )

    def _save_episode(self, episode: dict, metadata: dict) -> dict:
        shard_id = self.next_shard_id
        self.next_shard_id += 1
        filename = f"combo_{shard_id:06d}.npz"
        target = self.buffer_dir / filename
        tmp = self.buffer_dir / f"combo_{shard_id:06d}.tmp.npz"
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

    def _sample_lane_min_steps(self, max_scan_steps: int, *, fast_cpu: bool = False) -> tuple[np.ndarray, int]:
        """Stratify per-lane trigger thresholds across [10, 350] (early/mid single + 2x elixir)."""
        if fast_cpu or max_scan_steps <= 120:
            upper = max(12, min(48, max_scan_steps - 8))
            return self.rng.integers(8, upper, size=LANES), 0
        # Lanes 0..8: early-to-mid single elixir [10, 180) (4s..72s)
        # Lanes 9..17: late-single & double elixir [180, 351) (72s..140s, crossing 120s 2x mark)
        half = LANES // 2
        early_mid = self.rng.integers(10, 180, size=half)
        late_2x = self.rng.integers(180, min(351, max_scan_steps - 15), size=LANES - half)
        return np.concatenate([early_mid, late_2x]), half

    def _scan_branch_candidates(
        self,
        env: CrForgeVector,
        inference: StateInference,
        decks,
        forms,
        deck_probs,
        count: int,
        max_scan_steps: int = 380,
        focal_deck_indices: list[int] | None = None,
        snap_env: CrForgeVector | None = None,
    ) -> list[dict]:
        vector = env.vector
        s = vector.schema
        tiles = s.cols * s.rows
        scale = s.scales["elixir_scale"]
        n_decks = len(decks)
        probs = deck_probs if deck_probs is not None else np.ones(n_decks) / n_decks

        seeds = [int(self.rng.integers(1, 2**31 - 1)) for _ in range(LANES)]
        if focal_deck_indices:
            own_ids = [int(self.rng.choice(focal_deck_indices)) for _ in range(LANES)]
        else:
            own_ids = [int(self.rng.choice(n_decks, p=probs)) for _ in range(LANES)]
        opp_ids = [int(self.rng.choice(n_decks, p=probs)) for _ in range(LANES)]
        seats = np.array([i % 2 for i in range(LANES)], dtype=int)

        for lane in range(LANES):
            pair = [opp_ids[lane], opp_ids[lane]]
            pair[seats[lane]] = own_ids[lane]
            vector.reset_env(lane, seeds[lane], *(decks[i] for i in pair), *(forms[i] for i in pair))
        inference.reset()

        fast_cpu = torch.device(inference.device).type == "cpu"
        min_steps, late_split = self._sample_lane_min_steps(max_scan_steps, fast_cpu=fast_cpu)
        if late_split == 0:
            want_early, want_late = count, 0
        elif count == 1:
            want_late = int(self.phase_count % 2 == 0)
            want_early = 1 - want_late
        else:
            want_late = (count + 1) // 2
            want_early = count - want_late

        rngs = [np.random.default_rng([seeds[l], 0x5343414E]) for l in range(LANES)]
        history = [[] for _ in range(LANES)]
        obs_hist = [[] for _ in range(LANES)]
        mask_hist = [[] for _ in range(LANES)]
        early_pool: list[dict] = []
        late_pool: list[dict] = []
        used_lanes: set[int] = set()
        live = np.ones(LANES, dtype=bool)
        can_snap = (
            snap_env is not None
            and hasattr(snap_env.vector, "fork_from_lane")
            and getattr(vector, "lanes", None)
            and hasattr(vector.lanes[0], "fork_from")
        )

        for step in range(max_scan_steps):
            if not live.any():
                break
            if len(early_pool) >= max(want_early, 1 if want_early else 0) and len(late_pool) >= max(want_late, 1 if want_late else 0) and (len(early_pool) + len(late_pool)) >= count:
                break
            raw = env.observations
            native = env.masks
            logits, _ = inference.evaluate(raw.reshape(2 * LANES, s.seat_ints), native.reshape(2 * LANES, s.actions))
            u = torch.as_tensor(
                np.array([r.random(2) for r in rngs], dtype=np.float32), device=inference.device
            ).reshape(-1, 1)
            actions = sample_with_uniforms(logits, u).reshape(LANES, 2)

            active = np.flatnonzero(live)
            relax_elixir = step >= max_scan_steps - 40
            for lane in active:
                if lane not in used_lanes and step >= int(min_steps[lane]):
                    seat = int(seats[lane])
                    row = raw[lane, seat]
                    act = int(actions[lane, seat])
                    avail = int(s.field(row, "elixir")) - int(s.field(row, "elixir_reserved"))
                    if (
                        0 < act < s.first_ability
                        and int(s.field(row, "pending_count")) == 0
                        and native[lane, seat, s.wait]
                        and (relax_elixir or avail >= 45 * scale // 10)
                        and low_pressure(s, row)
                    ):
                        chosen_slot = (act - 1) // tiles
                        costs = s.field(row, "hand_cost")
                        # Trigger when hand holds an alternative heavy card (cost >= 5) that wasn't played
                        has_heavy_alt = any(
                            sl != chosen_slot and int(costs[sl]) >= 5 for sl in range(s.hand_slots)
                        )
                        if has_heavy_alt:
                            if can_snap:
                                snap_env.vector.fork_from_lane(vector, int(lane), [int(lane)])
                            cand = dict(
                                seed=seeds[lane],
                                own_deck=own_ids[lane],
                                opp_deck=opp_ids[lane],
                                seat=seat,
                                t_star=step,
                                scan_lane=int(lane),
                                chosen_slot=chosen_slot,
                                chosen_card_id=int(s.field(row, "hand_card")[chosen_slot]),
                                prefix_actions=list(history[lane]),
                                prefix_obs_both=list(obs_hist[lane]) if can_snap else None,
                                prefix_masks_seat=list(mask_hist[lane]) if can_snap else None,
                                orig_pair=(int(actions[lane, 0]), int(actions[lane, 1])),
                            )
                            if late_split > 0 and lane >= late_split:
                                late_pool.append(cand)
                            else:
                                early_pool.append(cand)
                            used_lanes.add(int(lane))
                if can_snap and lane not in used_lanes:
                    obs_hist[lane].append(raw[lane].copy())
                    mask_hist[lane].append(native[lane, int(seats[lane])].copy())
                history[lane].append((int(actions[lane, 0]), int(actions[lane, 1])))

            vector.actions[:] = actions
            vector.step_lanes(active)
            live &= env.dones == 0

        if late_split == 0:
            return early_pool[:count]
        selected: list[dict] = []
        if want_late > 0 and late_pool:
            idx = self.rng.choice(len(late_pool), size=min(want_late, len(late_pool)), replace=False)
            selected.extend(late_pool[int(i)] for i in idx)
        if want_early > 0 and early_pool:
            idx = self.rng.choice(len(early_pool), size=min(want_early, len(early_pool)), replace=False)
            selected.extend(early_pool[int(i)] for i in idx)
        if len(selected) < count:
            chosen_keys = {(c["seed"], c["t_star"]) for c in selected}
            remainder = [c for c in (late_pool + early_pool) if (c["seed"], c["t_star"]) not in chosen_keys]
            selected.extend(remainder[: count - len(selected)])
        return selected[:count]

    def _evaluate_branch(
        self,
        env: CrForgeVector,
        inference: StateInference,
        br: dict,
        decks,
        forms,
        deck_names,
        win_margin: float,
    ) -> tuple[dict | None, dict | None]:
        """Wave 1 screens all 16 (slot1, slot2) pairs; Wave 2 confirms Top-2 across 6 rollouts."""
        vector = env.vector
        s = vector.schema
        tiles = s.cols * s.rows
        scale = s.scales["elixir_scale"]
        seat = br["seat"]
        t_star = br["t_star"]
        deck_pair = [br["opp_deck"], br["opp_deck"]]
        deck_pair[seat] = br["own_deck"]
        all_lanes = np.arange(LANES)

        # --- Wave 1: 16 (slot1, slot2) pairs on lanes 0..15 + 2 original on lanes 16..17 ---
        for lane in range(LANES):
            vector.reset_env(lane, br["seed"], *(decks[i] for i in deck_pair), *(forms[i] for i in deck_pair))
        inference.reset()
        for a0, a1 in br["prefix_actions"]:
            inference.evaluate(env.observations.reshape(2 * LANES, s.seat_ints), env.masks.reshape(2 * LANES, s.actions))
            vector.actions[:, 0] = a0
            vector.actions[:, 1] = a1
            vector.step_lanes(all_lanes)

        origin_hp = tower_hp_margin(s, env.observations[0, seat])
        hand_at_branch = [vector.vocabulary[int(c)] for c in s.field(env.observations[0, seat], "hand_card")]
        next_at_branch = vector.vocabulary[int(s.field(env.observations[0, seat], "next_card"))]

        w1_states = []
        for lane in range(16):
            s1, s2 = divmod(lane, 4)
            w1_states.append(dict(phase="play_first", slot1=s1, slot2=s2, t1=None, armed_play=0, armed_wait=False, plays=0))
        for _ in range(2):
            w1_states.append(dict(phase="done", slot1=-1, slot2=-1, t1=None, armed_play=0, armed_wait=False, plays=0))

        rng_common = np.random.default_rng([br["seed"], 0x57415645, 0])
        rng_orig2 = np.random.default_rng([br["seed"], 0x57415645, 1])
        hp_32s = np.full(LANES, np.nan, dtype=np.float64)
        live = np.ones(LANES, dtype=bool)

        for step in range(t_star, 1000):
            if not live.any():
                break
            raw = env.observations
            native = env.masks
            behavior = native.copy()
            active = np.flatnonzero(live)

            for lane in active:
                st = w1_states[lane]
                st["armed_play"] = 0
                st["armed_wait"] = False
                if lane >= 16:
                    if step == t_star:
                        behavior[lane, 0, :] = False
                        behavior[lane, 0, br["orig_pair"][0]] = True
                        behavior[lane, 1, :] = False
                        behavior[lane, 1, br["orig_pair"][1]] = True
                else:
                    _apply_pair_mask(s, raw[lane, seat], native[lane, seat], behavior[lane, seat], st, step, t_star, tiles, scale)

            logits, _ = inference.evaluate(raw.reshape(2 * LANES, s.seat_ints), behavior.reshape(2 * LANES, s.actions))
            d0 = rng_common.random(2).astype(np.float32)
            d1 = rng_orig2.random(2).astype(np.float32)
            draws = np.vstack([np.tile(d0, (17, 1)), d1[None, :]])
            actions = sample_with_uniforms(logits, torch.as_tensor(draws, device=inference.device).reshape(-1, 1)).reshape(LANES, 2)

            for lane in active:
                st = w1_states[lane]
                if st["armed_play"] == 1:
                    st["t1"] = step
                    st["plays"] += 1
                    st["phase"] = "play_second"
                elif st["armed_play"] == 2:
                    st["plays"] += 1
                    st["phase"] = "done"

            vector.actions[:] = actions
            vector.step_lanes(active)
            dt = step - t_star + 1
            for lane in active:
                if np.isnan(hp_32s[lane]) and (dt >= 80 or env.dones[lane]):
                    hp_32s[lane] = tower_hp_margin(s, env.observations[lane, seat]) - origin_hp
            live &= env.dones == 0

        def lane_score(lane: int) -> tuple[float, int, float]:
            info = vector.info(lane)
            if info.get("fault") or info.get("truncated"):
                return (-1.0, -3, -9999.0)
            ws = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
            obs_e = env.observations[lane, seat]
            cd = int(s.field(obs_e, "crowns") - s.field(obs_e, "crowns_enemy"))
            h32 = float(0.0 if np.isnan(hp_32s[lane]) else hp_32s[lane])
            return (ws, cd, h32)

        orig_s1 = [lane_score(16), lane_score(17)]
        orig_w1 = [orig_s1[0][0], orig_s1[1][0]]
        orig_cd1 = max(orig_s1[0][1], orig_s1[1][1])
        ranked_pairs = sorted(
            [lane for lane in range(16) if w1_states[lane]["plays"] >= 1],
            key=lambda l: (lane_score(l)[0], lane_score(l)[1], lane_score(l)[2]),
            reverse=True,
        )
        if not ranked_pairs:
            return None, dict(status="no_valid_macro_pair")
        best_s1 = lane_score(ranked_pairs[0])
        best_w1, best_cd1 = best_s1[0], best_s1[1]
        mean_orig_w1 = float(np.mean(orig_w1))
        if win_margin > 0.0 and (
            best_w1 < mean_orig_w1
            or (best_w1 == mean_orig_w1 and best_cd1 <= orig_cd1)
        ):
            return None, dict(
                status="early_pruned_orig_winning_or_tied",
                deck=deck_names[br["own_deck"]],
                opp_deck=deck_names[br["opp_deck"]],
                t_star=t_star,
                orig_w1=orig_w1,
                best_w1=best_w1,
            )

        top_pairs = [divmod(ranked_pairs[0], 4)]
        for cand in ranked_pairs[1:]:
            p = divmod(cand, 4)
            if p not in top_pairs:
                top_pairs.append(p)
                break
        while len(top_pairs) < 2:
            top_pairs.append(top_pairs[0])

        # --- Wave 2: Confirmation (original, top_pairs[0], top_pairs[1]) x 6 rollouts = 18 lanes ---
        for lane in range(LANES):
            vector.reset_env(lane, br["seed"], *(decks[i] for i in deck_pair), *(forms[i] for i in deck_pair))
        inference.reset()

        prefix_obs, prefix_masks, prefix_acts = [], [], []
        for a0, a1 in br["prefix_actions"]:
            prefix_obs.append(env.observations[0, seat].copy())
            prefix_masks.append(env.masks[0, seat].copy())
            prefix_acts.append(a0 if seat == 0 else a1)
            inference.evaluate(env.observations.reshape(2 * LANES, s.seat_ints), env.masks.reshape(2 * LANES, s.actions))
            vector.actions[:, 0] = a0
            vector.actions[:, 1] = a1
            vector.step_lanes(all_lanes)

        w2_states = []
        for lane in range(LANES):
            arm_idx = lane % 3
            if arm_idx == 0:
                w2_states.append(dict(phase="done", slot1=-1, slot2=-1, t1=None, armed_play=0, armed_wait=False, plays=0))
            else:
                s1, s2 = top_pairs[arm_idx - 1]
                w2_states.append(dict(phase="play_first", slot1=s1, slot2=s2, t1=None, armed_play=0, armed_wait=False, plays=0))

        traj_obs = [list(prefix_obs) for _ in range(LANES)]
        traj_masks = [list(prefix_masks) for _ in range(LANES)]
        traj_acts = [list(prefix_acts) for _ in range(LANES)]
        traj_mplay = [[False] * len(prefix_acts) for _ in range(LANES)]
        traj_mwait = [[False] * len(prefix_acts) for _ in range(LANES)]

        rngs = [np.random.default_rng([br["seed"], 0x434F4E46, k]) for k in range(CONFIRM_ROLLOUTS)]
        live = np.ones(LANES, dtype=bool)

        for step in range(t_star, 1000):
            if not live.any():
                break
            raw = env.observations
            native = env.masks
            behavior = native.copy()
            active = np.flatnonzero(live)

            for lane in active:
                st = w2_states[lane]
                st["armed_play"] = 0
                st["armed_wait"] = False
                if lane % 3 == 0:
                    if step == t_star:
                        behavior[lane, 0, :] = False
                        behavior[lane, 0, br["orig_pair"][0]] = True
                        behavior[lane, 1, :] = False
                        behavior[lane, 1, br["orig_pair"][1]] = True
                else:
                    _apply_pair_mask(s, raw[lane, seat], native[lane, seat], behavior[lane, seat], st, step, t_star, tiles, scale)

            logits, _ = inference.evaluate(raw.reshape(2 * LANES, s.seat_ints), behavior.reshape(2 * LANES, s.actions))
            draws = np.repeat([r.random(2) for r in rngs], 3, axis=0).astype(np.float32)
            actions = sample_with_uniforms(logits, torch.as_tensor(draws, device=inference.device).reshape(-1, 1)).reshape(LANES, 2)

            for lane in active:
                st = w2_states[lane]
                act_seat = int(actions[lane, seat])
                is_mp = st["armed_play"] in (1, 2) and act_seat > 0
                is_mw = bool(st["armed_wait"] and act_seat == 0)
                if st["armed_play"] == 1:
                    st["t1"] = step
                    st["plays"] += 1
                    st["phase"] = "play_second"
                elif st["armed_play"] == 2:
                    st["plays"] += 1
                    st["phase"] = "done"

                traj_obs[lane].append(raw[lane, seat].copy())
                traj_masks[lane].append(native[lane, seat].copy())
                traj_acts[lane].append(act_seat)
                traj_mplay[lane].append(is_mp)
                traj_mwait[lane].append(is_mw)

            vector.actions[:] = actions
            vector.step_lanes(active)
            live &= env.dones == 0

        arm_scores = {0: [], 1: [], 2: []}
        for k in range(CONFIRM_ROLLOUTS):
            for arm_idx in range(3):
                lane = k * 3 + arm_idx
                info = vector.info(lane)
                valid_end = not (live[lane] or info.get("fault") or info.get("truncated"))
                ws = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
                obs_e = env.observations[lane, seat]
                cd = int(s.field(obs_e, "crowns") - s.field(obs_e, "crowns_enemy"))
                arm_scores[arm_idx].append(dict(lane=lane, valid=valid_end, ws=ws, won=int(info["winner"] == seat), cd=cd, plays=w2_states[lane]["plays"]))

        orig_ws = float(np.mean([x["ws"] for x in arm_scores[0]]))
        orig_cd = float(np.mean([x["cd"] for x in arm_scores[0]]))

        best_arm = None
        best_adv = -999.0
        for arm_idx in (1, 2):
            cand_ws = float(np.mean([x["ws"] for x in arm_scores[arm_idx]]))
            cand_cd = float(np.mean([x["cd"] for x in arm_scores[arm_idx]]))
            crown_bonus = 0.06 * max(0.0, cand_cd - orig_cd) if (cand_ws >= orig_ws and win_margin <= 0.25) else 0.0
            adv = (cand_ws - orig_ws) + crown_bonus
            if adv > best_adv and cand_cd - orig_cd >= 0.15:
                best_adv = adv
                best_arm = arm_idx

        summary = dict(
            deck=deck_names[br["own_deck"]],
            opp_deck=deck_names[br["opp_deck"]],
            t_star=t_star,
            hand=hand_at_branch,
            next_card=next_at_branch,
            orig_card=vector.vocabulary[br["chosen_card_id"]],
            orig_win_score=orig_ws,
            top_pairs=[list(p) for p in top_pairs],
            pair1_win_score=float(np.mean([x["ws"] for x in arm_scores[1]])),
            pair2_win_score=float(np.mean([x["ws"] for x in arm_scores[2]])),
            best_win_advantage=float(best_adv if best_arm is not None else 0.0),
        )

        if best_arm is None or best_adv < win_margin:
            return None, dict(status="below_win_margin", **summary)

        # Pick best valid winning rollout from best_arm
        winners = [x for x in arm_scores[best_arm] if x["valid"] and x["won"] == 1 and x["plays"] >= 1]
        if not winners:
            return None, dict(status="no_clean_winning_rollout", **summary)
        chosen = max(winners, key=lambda x: (x["plays"], x["cd"]))
        lane = chosen["lane"]

        raw_arr = np.stack(traj_obs[lane]).astype(np.int32)
        mask_arr = np.stack(traj_masks[lane]).astype(bool)
        act_arr = np.asarray(traj_acts[lane], dtype=np.int64)
        mplay_arr = np.asarray(traj_mplay[lane], dtype=bool)
        mwait_arr = np.asarray(traj_mwait[lane], dtype=bool)
        keep_arr = mask_arr.sum(-1) > 1
        weights_arr = combo_frame_weights(act_arr, keep_arr, mplay_arr, mwait_arr)

        s1, s2 = top_pairs[best_arm - 1]
        card1_name = hand_at_branch[s1]
        card2_name = next_at_branch if s2 == s1 else hand_at_branch[s2]
        summary.update(
            status="admitted",
            chosen_pair=[s1, s2],
            combo_cards=[card1_name, card2_name],
            chosen_crown_diff=chosen["cd"],
            macro_plays=int(mplay_arr.sum()),
            macro_waits=int(mwait_arr.sum()),
        )
        episode = dict(
            raw_obs=raw_arr,
            legal_masks=mask_arr,
            actions=act_arr,
            actor_keep=keep_arr,
            weights=weights_arr,
        )
        return episode, summary

    @torch.inference_mode()
    def _evaluate_branches_batched(
        self,
        snap_env: CrForgeVector,
        eval_env: CrForgeVector,
        actor,
        adapter,
        found: list[dict],
        deck_names,
        win_margin: float,
        device,
        precision: str,
    ) -> list[tuple[dict | None, dict]]:
        """Evaluate all `B` branches simultaneously in 2 batched GPU+JVM waves with zero prefix replay."""
        b_cnt = len(found)
        if b_cnt == 0:
            return []
        dev = torch.device(device)
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if (precision == "fp16" and dev.type == "cuda")
            else nullcontext()
        )
        vector = eval_env.vector
        s = vector.schema
        tiles = s.cols * s.rows
        scale = s.scales["elixir_scale"]
        hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)

        # 1. Single batched prefill across [2 * B, max_t_star] to obtain exact recurrent memories at t_star
        max_t = max((len(br["prefix_obs_both"]) for br in found), default=0)
        init_mem = actor.initial_state(2 * b_cnt, device=dev)
        if max_t > 0:
            pref_raw = np.zeros((2 * b_cnt, max_t, s.seat_ints), dtype=np.int32)
            pref_tmask = np.zeros((2 * b_cnt, max_t), dtype=bool)
            for b_idx, br in enumerate(found):
                t_len = len(br["prefix_obs_both"])
                if t_len > 0:
                    arr = np.stack(br["prefix_obs_both"], axis=0)
                    pref_raw[2 * b_idx, :t_len] = arr[:, 0]
                    pref_raw[2 * b_idx + 1, :t_len] = arr[:, 1]
                    pref_tmask[2 * b_idx, :t_len] = True
                    pref_tmask[2 * b_idx + 1, :t_len] = True
            ph, pe, pm = adapter(pref_raw)
            with amp_ctx:
                _, _, _, _, mem_t_star, _, _, _, _ = actor._encode_state_sequence(
                    torch.as_tensor(ph, device=dev),
                    torch.as_tensor(pe, device=dev),
                    torch.as_tensor(pm, device=dev),
                    state=init_mem,
                    time_mask=torch.as_tensor(pref_tmask, device=dev),
                )
        else:
            mem_t_star = init_mem

        # 2. Wave 1: Fork all B branches (B * 18 lanes) simultaneously with zero prefix replay
        total_w1 = b_cnt * LANES
        vector.fork_tasks_parallel(
            [
                (snap_env.vector, int(br["scan_lane"]), range(b_idx * LANES, (b_idx + 1) * LANES))
                for b_idx, br in enumerate(found)
            ]
        )
        w1_bcast = []
        for b_idx in range(b_cnt):
            for _ in range(LANES):
                w1_bcast.extend([2 * b_idx, 2 * b_idx + 1])
        w1_mem = slice_memory(mem_t_star, w1_bcast, device=dev, detach=True, trim_gqa=True, position_bound=max(1, max_t))

        origin_hps = []
        hands_at_branch = []
        nexts_at_branch = []
        w1_states: list[dict] = []
        rng_commons = []
        rng_orig2s = []
        for b_idx, br in enumerate(found):
            seat = int(br["seat"])
            base_l = b_idx * LANES
            origin_hps.append(tower_hp_margin(s, eval_env.observations[base_l, seat]))
            hands_at_branch.append([vector.vocabulary[int(c)] for c in s.field(eval_env.observations[base_l, seat], "hand_card")])
            nexts_at_branch.append(vector.vocabulary[int(s.field(eval_env.observations[base_l, seat], "next_card"))])
            for loc_l in range(16):
                s1, s2 = divmod(loc_l, 4)
                w1_states.append(dict(phase="play_first", slot1=s1, slot2=s2, t1=None, armed_play=0, armed_wait=False, plays=0))
            for _ in range(2):
                w1_states.append(dict(phase="done", slot1=-1, slot2=-1, t1=None, armed_play=0, armed_wait=False, plays=0))
            rng_commons.append(np.random.default_rng([br["seed"], 0x57415645, 0]))
            rng_orig2s.append(np.random.default_rng([br["seed"], 0x57415645, 1]))

        hp_32s = np.full(total_w1, np.nan, dtype=np.float64)
        live_w1 = np.ones(total_w1, dtype=bool)
        active_slots = list(range(total_w1))
        max_rem_w1 = max(1000 - int(br["t_star"]) for br in found)

        for offset in range(max_rem_w1):
            for b_idx, br in enumerate(found):
                if int(br["t_star"]) + offset >= 1000:
                    live_w1[b_idx * LANES : (b_idx + 1) * LANES] = False
            keep_pos = [pos for pos, gl in enumerate(active_slots) if live_w1[gl]]
            if not keep_pos:
                break
            if len(keep_pos) < len(active_slots):
                mem_keep = []
                for pos in keep_pos:
                    mem_keep.extend([2 * pos, 2 * pos + 1])
                w1_mem = slice_memory(w1_mem, mem_keep, device=dev, detach=True, trim_gqa=False)
                active_slots = [active_slots[pos] for pos in keep_pos]

            draws_all = np.zeros((total_w1, 2), dtype=np.float32)
            for b_idx in range(b_cnt):
                lo_b, hi_b = b_idx * LANES, (b_idx + 1) * LANES
                if live_w1[lo_b:hi_b].any():
                    d0 = rng_commons[b_idx].random(2).astype(np.float32)
                    d1 = rng_orig2s[b_idx].random(2).astype(np.float32)
                    draws_all[lo_b : lo_b + 17] = d0
                    draws_all[lo_b + 17] = d1

            a_cnt = len(active_slots)
            raw_sub = eval_env.observations[active_slots]
            beh_sub = eval_env.masks[active_slots].copy()
            for pos, gl in enumerate(active_slots):
                b_idx = gl // LANES
                loc_l = gl % LANES
                br = found[b_idx]
                seat = int(br["seat"])
                t_star = int(br["t_star"])
                step_b = t_star + offset
                st = w1_states[gl]
                st["armed_play"] = 0
                st["armed_wait"] = False
                if loc_l >= 16:
                    if offset == 0:
                        beh_sub[pos, 0, :] = False
                        beh_sub[pos, 0, br["orig_pair"][0]] = True
                        beh_sub[pos, 1, :] = False
                        beh_sub[pos, 1, br["orig_pair"][1]] = True
                else:
                    _apply_pair_mask(s, raw_sub[pos, seat], eval_env.masks[gl, seat], beh_sub[pos, seat], st, step_b, t_star, tiles, scale)

            flat_raw = raw_sub.reshape(2 * a_cnt, s.seat_ints)
            flat_beh = beh_sub.reshape(2 * a_cnt, s.actions)
            fh, fe, fm = adapter(flat_raw)
            w1_mem = reserve_step_memory(w1_mem, 128)
            u_flat = draws_all[active_slots].reshape(-1)
            with amp_ctx:
                if hasattr(actor, "step_sample_crn"):
                    acts_flat, w1_mem = actor.step_sample_crn(
                        torch.as_tensor(fh, device=dev),
                        torch.as_tensor(fe, device=dev),
                        torch.as_tensor(fm, device=dev),
                        flat_beh,
                        u_flat,
                        state=w1_mem,
                        hand_card_ids=torch.as_tensor(flat_raw[:, hand_indices], device=dev),
                    )
                    acts_sub = acts_flat.reshape(a_cnt, 2)
                else:
                    logits, _, w1_mem = actor.step(
                        torch.as_tensor(fh, device=dev),
                        torch.as_tensor(fe, device=dev),
                        torch.as_tensor(fm, device=dev),
                        legal_action_mask=torch.as_tensor(flat_beh, dtype=torch.bool, device=dev),
                        state=w1_mem,
                        hand_card_ids=torch.as_tensor(flat_raw[:, hand_indices], device=dev),
                        compute_value=False,
                    )
                    u_sub = torch.as_tensor(u_flat.reshape(-1, 1), device=dev)
                    acts_sub = sample_with_uniforms(logits.float(), u_sub).reshape(a_cnt, 2)

            for pos, gl in enumerate(active_slots):
                st = w1_states[gl]
                step_b = int(found[gl // LANES]["t_star"]) + offset
                if st["armed_play"] == 1:
                    st["t1"] = step_b
                    st["plays"] += 1
                    st["phase"] = "play_second"
                elif st["armed_play"] == 2:
                    st["plays"] += 1
                    st["phase"] = "done"
                vector.actions[gl] = acts_sub[pos]

            vector.step_lanes(active_slots)
            dt = offset + 1
            for gl in active_slots:
                b_idx = gl // LANES
                seat = int(found[b_idx]["seat"])
                if np.isnan(hp_32s[gl]) and (dt >= 80 or eval_env.dones[gl]):
                    hp_32s[gl] = tower_hp_margin(s, eval_env.observations[gl, seat]) - origin_hps[b_idx]
                if eval_env.dones[gl]:
                    live_w1[gl] = False

        results: list[tuple[dict | None, dict] | None] = [None] * b_cnt
        surviving: list[tuple[int, dict, list[tuple[int, int]]]] = []
        for b_idx, br in enumerate(found):
            seat = int(br["seat"])
            base_l = b_idx * LANES

            def _w1_score(loc_l: int) -> tuple[float, int, float]:
                gl = base_l + loc_l
                info = vector.info(gl)
                if info.get("fault") or info.get("truncated"):
                    return (-1.0, -3, -9999.0)
                ws = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
                obs_e = eval_env.observations[gl, seat]
                cd = int(s.field(obs_e, "crowns") - s.field(obs_e, "crowns_enemy"))
                h32 = float(0.0 if np.isnan(hp_32s[gl]) else hp_32s[gl])
                return (ws, cd, h32)

            orig_s1 = [_w1_score(16), _w1_score(17)]
            orig_w1 = [orig_s1[0][0], orig_s1[1][0]]
            orig_cd1 = max(orig_s1[0][1], orig_s1[1][1])
            ranked_pairs = sorted(
                [loc_l for loc_l in range(16) if w1_states[base_l + loc_l]["plays"] >= 1],
                key=lambda l: _w1_score(l),
                reverse=True,
            )
            if not ranked_pairs:
                results[b_idx] = (None, dict(status="no_valid_macro_pair"))
                continue
            best_s1 = _w1_score(ranked_pairs[0])
            best_w1, best_cd1 = best_s1[0], best_s1[1]
            mean_orig_w1 = float(np.mean(orig_w1))
            if win_margin > 0.0 and (
                best_w1 < mean_orig_w1
                or (best_w1 == mean_orig_w1 and best_cd1 <= orig_cd1)
            ):
                results[b_idx] = (
                    None,
                    dict(
                        status="early_pruned_orig_winning_or_tied",
                        deck=deck_names[br["own_deck"]],
                        opp_deck=deck_names[br["opp_deck"]],
                        t_star=br["t_star"],
                        orig_w1=orig_w1,
                        best_w1=best_w1,
                    ),
                )
                continue
            top_pairs = [divmod(ranked_pairs[0], 4)]
            for cand in ranked_pairs[1:]:
                p = divmod(cand, 4)
                if p not in top_pairs:
                    top_pairs.append(p)
                    break
            while len(top_pairs) < 2:
                top_pairs.append(top_pairs[0])
            surviving.append((b_idx, br, top_pairs))

        if not surviving:
            return [r for r in results if r is not None]

        # 3. Wave 2: Confirm all surviving branches simultaneously with zero prefix replay
        s_cnt = len(surviving)
        total_w2 = s_cnt * LANES
        vector.fork_tasks_parallel(
            [
                (snap_env.vector, int(br["scan_lane"]), range(s_idx * LANES, (s_idx + 1) * LANES))
                for s_idx, (_b_idx, br, _tp) in enumerate(surviving)
            ]
        )
        w2_bcast = []
        for b_idx, _br, _tp in surviving:
            for _ in range(LANES):
                w2_bcast.extend([2 * b_idx, 2 * b_idx + 1])
        w2_mem = slice_memory(mem_t_star, w2_bcast, device=dev, detach=True, trim_gqa=True, position_bound=max(1, max_t))

        w2_states: list[dict] = []
        traj_obs: list[list[np.ndarray]] = []
        traj_masks: list[list[np.ndarray]] = []
        traj_acts: list[list[int]] = []
        traj_mplay: list[list[bool]] = []
        traj_mwait: list[list[bool]] = []
        w2_rngs: list[list[np.random.Generator]] = []

        for s_idx, (_b_idx, br, top_pairs) in enumerate(surviving):
            seat = int(br["seat"])
            p_obs = [o[seat].copy() for o in br["prefix_obs_both"]]
            p_masks = [m.copy() for m in br["prefix_masks_seat"]]
            p_acts = [a0 if seat == 0 else a1 for a0, a1 in br["prefix_actions"]]
            w2_rngs.append([np.random.default_rng([br["seed"], 0x434F4E46, k]) for k in range(CONFIRM_ROLLOUTS)])
            for loc_l in range(LANES):
                arm_idx = loc_l % 3
                if arm_idx == 0:
                    w2_states.append(dict(phase="done", slot1=-1, slot2=-1, t1=None, armed_play=0, armed_wait=False, plays=0))
                else:
                    s1, s2 = top_pairs[arm_idx - 1]
                    w2_states.append(dict(phase="play_first", slot1=s1, slot2=s2, t1=None, armed_play=0, armed_wait=False, plays=0))
                traj_obs.append(list(p_obs))
                traj_masks.append(list(p_masks))
                traj_acts.append(list(p_acts))
                traj_mplay.append([False] * len(p_acts))
                traj_mwait.append([False] * len(p_acts))

        live_w2 = np.ones(total_w2, dtype=bool)
        active_w2 = list(range(total_w2))
        max_rem_w2 = max(1000 - int(br["t_star"]) for _, br, _ in surviving)

        for offset in range(max_rem_w2):
            for s_idx, (_b_idx, br, _tp) in enumerate(surviving):
                if int(br["t_star"]) + offset >= 1000:
                    live_w2[s_idx * LANES : (s_idx + 1) * LANES] = False
            keep_pos = [pos for pos, gl in enumerate(active_w2) if live_w2[gl]]
            if not keep_pos:
                break
            if len(keep_pos) < len(active_w2):
                mem_keep = []
                for pos in keep_pos:
                    mem_keep.extend([2 * pos, 2 * pos + 1])
                w2_mem = slice_memory(w2_mem, mem_keep, device=dev, detach=True, trim_gqa=False)
                active_w2 = [active_w2[pos] for pos in keep_pos]

            draws_w2 = np.zeros((total_w2, 2), dtype=np.float32)
            for s_idx in range(s_cnt):
                lo_s, hi_s = s_idx * LANES, (s_idx + 1) * LANES
                if live_w2[lo_s:hi_s].any():
                    draws_w2[lo_s:hi_s] = np.repeat([r.random(2) for r in w2_rngs[s_idx]], 3, axis=0).astype(np.float32)

            a_cnt = len(active_w2)
            raw_sub = eval_env.observations[active_w2]
            beh_sub = eval_env.masks[active_w2].copy()
            for pos, gl in enumerate(active_w2):
                s_idx = gl // LANES
                loc_l = gl % LANES
                _b_idx, br, _tp = surviving[s_idx]
                seat = int(br["seat"])
                t_star = int(br["t_star"])
                step_b = t_star + offset
                st = w2_states[gl]
                st["armed_play"] = 0
                st["armed_wait"] = False
                if loc_l % 3 == 0:
                    if offset == 0:
                        beh_sub[pos, 0, :] = False
                        beh_sub[pos, 0, br["orig_pair"][0]] = True
                        beh_sub[pos, 1, :] = False
                        beh_sub[pos, 1, br["orig_pair"][1]] = True
                else:
                    _apply_pair_mask(s, raw_sub[pos, seat], eval_env.masks[gl, seat], beh_sub[pos, seat], st, step_b, t_star, tiles, scale)

            flat_raw = raw_sub.reshape(2 * a_cnt, s.seat_ints)
            flat_beh = beh_sub.reshape(2 * a_cnt, s.actions)
            fh, fe, fm = adapter(flat_raw)
            w2_mem = reserve_step_memory(w2_mem, 128)
            u_flat = draws_w2[active_w2].reshape(-1)
            with amp_ctx:
                if hasattr(actor, "step_sample_crn"):
                    acts_flat, w2_mem = actor.step_sample_crn(
                        torch.as_tensor(fh, device=dev),
                        torch.as_tensor(fe, device=dev),
                        torch.as_tensor(fm, device=dev),
                        flat_beh,
                        u_flat,
                        state=w2_mem,
                        hand_card_ids=torch.as_tensor(flat_raw[:, hand_indices], device=dev),
                    )
                    acts_sub = acts_flat.reshape(a_cnt, 2)
                else:
                    logits, _, w2_mem = actor.step(
                        torch.as_tensor(fh, device=dev),
                        torch.as_tensor(fe, device=dev),
                        torch.as_tensor(fm, device=dev),
                        legal_action_mask=torch.as_tensor(flat_beh, dtype=torch.bool, device=dev),
                        state=w2_mem,
                        hand_card_ids=torch.as_tensor(flat_raw[:, hand_indices], device=dev),
                        compute_value=False,
                    )
                    u_sub = torch.as_tensor(u_flat.reshape(-1, 1), device=dev)
                    acts_sub = sample_with_uniforms(logits.float(), u_sub).reshape(a_cnt, 2)

            for pos, gl in enumerate(active_w2):
                s_idx = gl // LANES
                _b_idx, br, _tp = surviving[s_idx]
                seat = int(br["seat"])
                step_b = int(br["t_star"]) + offset
                st = w2_states[gl]
                act_seat = int(acts_sub[pos, seat])
                is_mp = st["armed_play"] in (1, 2) and act_seat > 0
                is_mw = bool(st["armed_wait"] and act_seat == 0)
                if st["armed_play"] == 1:
                    st["t1"] = step_b
                    st["plays"] += 1
                    st["phase"] = "play_second"
                elif st["armed_play"] == 2:
                    st["plays"] += 1
                    st["phase"] = "done"
                traj_obs[gl].append(raw_sub[pos, seat].copy())
                traj_masks[gl].append(eval_env.masks[gl, seat].copy())
                traj_acts[gl].append(act_seat)
                traj_mplay[gl].append(is_mp)
                traj_mwait[gl].append(is_mw)
                vector.actions[gl] = acts_sub[pos]

            vector.step_lanes(active_w2)
            for gl in active_w2:
                if eval_env.dones[gl]:
                    live_w2[gl] = False

        for s_idx, (b_idx, br, top_pairs) in enumerate(surviving):
            seat = int(br["seat"])
            base_s = s_idx * LANES
            hand_at_branch = hands_at_branch[b_idx]
            next_at_branch = nexts_at_branch[b_idx]
            arm_scores = {0: [], 1: [], 2: []}
            for k in range(CONFIRM_ROLLOUTS):
                for arm_idx in range(3):
                    gl = base_s + k * 3 + arm_idx
                    info = vector.info(gl)
                    valid_end = not (live_w2[gl] or info.get("fault") or info.get("truncated"))
                    ws = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
                    obs_e = eval_env.observations[gl, seat]
                    cd = int(s.field(obs_e, "crowns") - s.field(obs_e, "crowns_enemy"))
                    arm_scores[arm_idx].append(dict(lane=gl, valid=valid_end, ws=ws, won=int(info["winner"] == seat), cd=cd, plays=w2_states[gl]["plays"]))

            orig_ws = float(np.mean([x["ws"] for x in arm_scores[0]]))
            orig_cd = float(np.mean([x["cd"] for x in arm_scores[0]]))
            best_arm = None
            best_adv = -999.0
            for arm_idx in (1, 2):
                cand_ws = float(np.mean([x["ws"] for x in arm_scores[arm_idx]]))
                cand_cd = float(np.mean([x["cd"] for x in arm_scores[arm_idx]]))
                crown_bonus = 0.06 * max(0.0, cand_cd - orig_cd) if (cand_ws >= orig_ws and win_margin <= 0.25) else 0.0
                adv = (cand_ws - orig_ws) + crown_bonus
                if adv > best_adv and cand_cd - orig_cd >= 0.15:
                    best_adv = adv
                    best_arm = arm_idx

            summary = dict(
                deck=deck_names[br["own_deck"]],
                opp_deck=deck_names[br["opp_deck"]],
                t_star=br["t_star"],
                hand=hand_at_branch,
                next_card=next_at_branch,
                orig_card=vector.vocabulary[br["chosen_card_id"]],
                orig_win_score=orig_ws,
                top_pairs=[list(p) for p in top_pairs],
                pair1_win_score=float(np.mean([x["ws"] for x in arm_scores[1]])),
                pair2_win_score=float(np.mean([x["ws"] for x in arm_scores[2]])),
                best_win_advantage=float(best_adv if best_arm is not None else 0.0),
            )
            if best_arm is None or best_adv < win_margin:
                results[b_idx] = (None, dict(status="below_win_margin", **summary))
                continue
            winners = [x for x in arm_scores[best_arm] if x["valid"] and x["won"] == 1 and x["plays"] >= 1]
            if not winners:
                results[b_idx] = (None, dict(status="no_clean_winning_rollout", **summary))
                continue
            chosen = max(winners, key=lambda x: (x["plays"], x["cd"]))
            gl = chosen["lane"]
            raw_arr = np.stack(traj_obs[gl]).astype(np.int32)
            mask_arr = np.stack(traj_masks[gl]).astype(bool)
            act_arr = np.asarray(traj_acts[gl], dtype=np.int64)
            mplay_arr = np.asarray(traj_mplay[gl], dtype=bool)
            mwait_arr = np.asarray(traj_mwait[gl], dtype=bool)
            keep_arr = mask_arr.sum(-1) > 1
            weights_arr = combo_frame_weights(act_arr, keep_arr, mplay_arr, mwait_arr)
            s1, s2 = top_pairs[best_arm - 1]
            card1_name = hand_at_branch[s1]
            card2_name = next_at_branch if s2 == s1 else hand_at_branch[s2]
            summary.update(
                status="admitted",
                chosen_pair=[s1, s2],
                combo_cards=[card1_name, card2_name],
                chosen_crown_diff=chosen["cd"],
                macro_plays=int(mplay_arr.sum()),
                macro_waits=int(mwait_arr.sum()),
            )
            episode = dict(
                raw_obs=raw_arr,
                legal_masks=mask_arr,
                actions=act_arr,
                actor_keep=keep_arr,
                weights=weights_arr,
            )
            results[b_idx] = (episode, summary)

        return [r for r in results if r is not None]

    def run_phase(
        self,
        actor,
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
        candidates: int = 2,
        win_margin: float = 0.30,
        batch_episodes: int = 4,
        lr: float = 1e-6,
        max_kl: float = MAX_KL,
        deck_probabilities=None,
        focal_deck_indices: list[int] | None = None,
    ) -> dict:
        if not probes or not any(np.any(ep["actor_keep"]) for ep in probes):
            return dict(ran=False, reason="no_ppo_choice_probes")
        if update <= self.last_update:
            return dict(ran=False, reason="already_processed")
        if not 0 < lr <= 1e-5 or not 0 < max_kl <= MAX_KL:
            raise ValueError("Macro search distillation requires 0 < lr <= 1e-5 and 0 < max_kl <= 0.005")

        began = time.perf_counter()
        old_mode = actor.training
        actor.eval()

        branch_reports = []
        admitted_this_phase = 0
        max_threads = max(6, (os.cpu_count() or 12) - 4)
        orig_fp32 = getattr(actor, "ppo_actor_fp32", False)
        if precision == "fp16" and torch.device(device).type == "cuda":
            actor.ppo_actor_fp32 = False
        try:
            with (
                CrForgeVector(LANES, min(max_threads, LANES), paths=env_paths, independent=True, arena="Arena_L", decision_ticks=8) as search_env,
                CrForgeVector(LANES, 1, paths=env_paths, independent=True, arena="Arena_L", decision_ticks=8) as snap_env,
            ):
                search_env.set_autoreset(False)
                snap_env.set_autoreset(False)
                inference = StateInference(actor, adapter, device, 2 * LANES, compute_value=False, precision=precision)
                found = self._scan_branch_candidates(
                    search_env,
                    inference,
                    decks,
                    forms,
                    deck_probabilities,
                    count=max(1, int(candidates)),
                    focal_deck_indices=focal_deck_indices,
                    snap_env=snap_env,
                )
                if found and all(br.get("prefix_obs_both") is not None for br in found):
                    eval_lanes = len(found) * LANES
                    with CrForgeVector(eval_lanes, min(max_threads, eval_lanes), paths=env_paths, independent=True, arena="Arena_L", decision_ticks=8) as eval_env:
                        eval_env.set_autoreset(False)
                        batch_outcomes = self._evaluate_branches_batched(
                            snap_env, eval_env, actor, adapter, found, deck_names, float(win_margin), device, precision
                        )
                    for ep, rep in batch_outcomes:
                        self.searched_branches += 1
                        branch_reports.append(rep)
                        if ep is not None:
                            self._save_episode(
                                ep,
                                dict(
                                    update=int(update),
                                    deck=rep["deck"],
                                    combo_cards=rep["combo_cards"],
                                    win_advantage=rep["best_win_advantage"],
                                ),
                            )
                            admitted_this_phase += 1
                            self.admitted_total += 1
                else:
                    for br in found:
                        self.searched_branches += 1
                        ep, rep = self._evaluate_branch(search_env, inference, br, decks, forms, deck_names, float(win_margin))
                        branch_reports.append(rep)
                        if ep is not None:
                            self._save_episode(
                                ep,
                                dict(
                                    update=int(update),
                                    deck=rep["deck"],
                                    combo_cards=rep["combo_cards"],
                                    win_advantage=rep["best_win_advantage"],
                                ),
                            )
                            admitted_this_phase += 1
                            self.admitted_total += 1
        finally:
            actor.ppo_actor_fp32 = orig_fp32

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
                reason="insufficient_admitted_combos",
                candidates_found=len(found),
                admitted_this_phase=admitted_this_phase,
                buffer_size=len(self.entries),
                branch_reports=branch_reports,
                search_seconds=search_seconds,
            )

        # Sample up to batch_episodes from buffer + equal count of on-policy PPO probes for 50/50 deck balance
        k_combo = min(int(batch_episodes), len(self.entries))
        chosen_idx = self.rng.choice(len(self.entries), size=k_combo, replace=False)
        chosen_entries = [self.entries[int(i)] for i in chosen_idx]
        combo_eps, combo_w = self._load_episodes(chosen_entries)

        anchor_probes = list(probes[:k_combo])
        anchor_w = [probe_anchor_weights(ep) for ep in anchor_probes]
        train_eps = combo_eps + anchor_probes
        train_w = combo_w + anchor_w

        parameters = dict(actor.named_parameters())
        original_requires = {name: p.requires_grad for name, p in parameters.items()}
        originals = {}
        reasons = []
        steps = 0
        probe_before = probe_after = None

        try:
            names = freeze_actor_readouts(actor)
            originals = {name: parameters[name].detach().clone() for name in names}
            optimizer = torch.optim.AdamW([parameters[name] for name in names], lr=lr, weight_decay=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda", init_scale=128.0)
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            actor.eval()
            probe_before, reference = _score(actor, adapter, probes, device, precision, capture=True)
            _train_accumulated(actor, adapter, train_eps, train_w, optimizer, scaler, device, precision)
            steps = 1
            probe_after, _ = _score(actor, adapter, probes, device, precision, reference=reference)
            if probe_after["reference_forward_kl"] > max_kl:
                reasons.append("ppo_forward_kl_limit")
            if probe_after["mean_p_wait"] - probe_before["mean_p_wait"] > MAX_WAIT_DRIFT:
                reasons.append("mean_p_wait_drift")
        except FloatingPointError:
            reasons.append("nonfinite_candidate")
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

        accepted = not reasons and steps == 1
        if not accepted and originals:
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
            actor_attempted=bool(steps),
            memory_invalidated=bool(steps),
            changed_parameters=list(originals) if accepted else [],
            candidates_found=len(found),
            admitted_this_phase=admitted_this_phase,
            admitted_total=self.admitted_total,
            buffer_size=len(self.entries),
            trained_combo_episodes=k_combo,
            trained_anchor_probes=len(anchor_probes),
            optimizer_steps=steps,
            ppo_forward_kl=probe_after["reference_forward_kl"] if probe_after is not None else None,
            mean_p_wait_before=before_w,
            mean_p_wait_after=after_w,
            wait_prob_delta=None if after_w is None else after_w - before_w,
            branch_reports=branch_reports,
            search_seconds=search_seconds,
            phase_count=self.phase_count,
        )
