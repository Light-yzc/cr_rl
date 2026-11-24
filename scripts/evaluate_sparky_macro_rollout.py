"""Two-phase Sparky diagnostic + MCTS-style Macro-Rollout counterfactual lookahead."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

if sys.platform == "win32" and not os.environ.get("JAVA_HOME") and Path(r"F:\JDK\bin\java.exe").exists():
    os.environ["JAVA_HOME"] = r"F:\JDK"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / ".cache/vendor/crforge/python"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from crforge_env import _jvm
from cr_rl.config import SimulatorPaths
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.sparky_curriculum import SparkyCurriculum
from cr_rl.training.state_selfplay import _extract_decks
from run_nulls_dryrun import load_policy_inference

DECKS = ("user_giant_sparky_pump_plain", "user_giant_sparky_pump_horde_plain")
MACRO_ARMS = ("original", "sink_sparky", "sparky_giant_combo")


def sparky_status(schema, obs, mask, card_id: int):
    available = int(schema.field(obs, "elixir")) - int(schema.field(obs, "elixir_reserved"))
    slots = np.flatnonzero(schema.field(obs, "hand_card") == card_id)
    slot = int(slots[0]) if len(slots) == 1 else None
    legal = False
    if slot is not None:
        tiles = schema.cols * schema.rows
        first = 1 + slot * tiles
        legal = bool(mask[first : first + tiles].any())
    return slot, available, legal


def sample_actions(logits: torch.Tensor, uniforms: torch.Tensor):
    probs = logits.softmax(-1)
    cdf = probs.cumsum(-1)
    chosen = torch.searchsorted(cdf, uniforms * cdf[:, -1:], right=True).squeeze(1)
    return chosen.cpu().numpy().astype(np.int32), probs


def find_own_sparky(schema, obs, left_lane: bool):
    """Locate own alive Sparky (max_hp in [1000, 2400]) on the chosen lane."""
    e = schema.entity
    rows = schema.entities(obs)
    own_alive = (
        (rows[:, e["side"]] == 0)
        & (rows[:, e["kind"]] == schema.kinds["character"])
        & ((rows[:, e["flags"]] & schema.flags["alive"]) != 0)
        & (rows[:, e["hp"]] > 0)
        & (rows[:, e["max_hp"]] >= 1000)
        & (rows[:, e["max_hp"]] <= 2400)
    )
    lane_match = (rows[:, e["x"]] < 9000) if left_lane else (rows[:, e["x"]] >= 9000)
    candidates = rows[own_alive & lane_match]
    if not len(candidates):
        return None
    best = candidates[np.argmax(candidates[:, e["y"]])]
    return int(best[e["x"]]), int(best[e["y"]]), int(best[e["hp"]])


def emergency_defense_needed(schema, obs) -> bool:
    """Abort Giant hold if an enemy character is deep on our side (y <= 9500)."""
    e = schema.entity
    rows = schema.entities(obs)
    deep_enemy = (
        (rows[:, e["side"]] == 1)
        & (rows[:, e["kind"]] == schema.kinds["character"])
        & ((rows[:, e["flags"]] & schema.flags["alive"]) != 0)
        & (rows[:, e["hp"]] > 0)
        & (rows[:, e["y"]] <= 9500)
    )
    return bool(deep_enemy.any())


def run_phase1_scan(env, inference, seed_rows, decks, forms):
    """Run natural policy games to measure Sparky usage and collect branch points."""
    branches = []
    natural_records = []
    lanes = env.count
    batch_size = lanes
    vector = env.vector
    s = vector.schema
    sparky_id = vector.card_id("ZapMachine")
    tiles = s.cols * s.rows
    scale = s.scales["elixir_scale"]
    diag = SparkyCurriculum(s, sparky_id, vector.max_elixir, 0, 0)

    for start in range(0, len(seed_rows), batch_size):
        chunk = seed_rows[start : start + batch_size]
        expanded = chunk + [chunk[-1]] * (batch_size - len(chunk))
        seats = np.array([r[3] for r in expanded], dtype=int)
        for lane, (seed, own, opp, seat) in enumerate(expanded):
            deck_ids = [opp, opp]
            deck_ids[seat] = own
            vector.reset_env(lane, seed, *(decks[i] for i in deck_ids), *(forms[i] for i in deck_ids))
        inference.reset()
        rngs = [np.random.default_rng([seed, 0x53504152]) for seed, *_ in expanded]
        live = np.ones(lanes, dtype=bool)
        prev_legal = np.zeros(lanes, dtype=bool)
        pending_counts = np.zeros(lanes, dtype=int)
        stats = [Counter() for _ in range(lanes)]
        history = [[] for _ in range(lanes)]
        found_branch = [None] * lanes
        lane_ids = np.arange(lanes)

        for step in range(1000):
            if not live.any():
                break
            raw = env.observations
            native = env.masks
            obs_flat = raw.reshape(2 * lanes, s.seat_ints)
            mask_flat = native.reshape(2 * lanes, s.actions)
            logits, _ = inference.evaluate(obs_flat, mask_flat)
            u = torch.as_tensor(
                np.array([rng.random(2) for rng in rngs], dtype=np.float32),
                device=inference.device,
            ).reshape(-1, 1)
            actions_flat, probs = sample_actions(logits, u)
            actions = actions_flat.reshape(lanes, 2)
            learner_idx = torch.as_tensor(2 * lane_ids + seats, device=inference.device)
            card_mass = (
                probs[learner_idx, 1 : s.first_ability]
                .reshape(lanes, s.hand_slots, tiles)
                .sum(-1)
                .cpu()
                .numpy()
            )

            active = np.flatnonzero(live)
            for lane in active:
                seat = seats[lane]
                row = raw[lane, seat]
                slot, avail, legal = sparky_status(s, row, native[lane, seat], sparky_id)
                act = int(actions[lane, seat])
                if slot is not None and 5 * scale <= avail < 56 * scale // 10 and diag.low_pressure(row):
                    stats[lane]["low_pressure_5_to_5_6_frames"] += 1
                    if 0 < act < s.first_ability:
                        stats[lane]["low_pressure_5_to_5_6_plays"] += 1
                if legal:
                    stats[lane]["legal_windows"] += int(not prev_legal[lane])
                    stats[lane]["legal_frames"] += 1
                    stats[lane]["sparky_prob_sum"] += float(card_mass[lane, slot])
                    first = 1 + slot * tiles
                    if first <= act < first + tiles:
                        stats[lane]["sparky_requests"] += 1
                prev_legal[lane] = legal

                if (
                    lane < len(chunk)
                    and found_branch[lane] is None
                    and 8 <= step <= 520
                    and slot is not None
                    and int(s.field(row, "pending_count")) == 0
                    and native[lane, seat, s.wait]
                    and diag.low_pressure(row)
                    and 0 < act < s.first_ability
                ):
                    chosen_slot = (act - 1) // tiles
                    chosen_card_id = int(s.field(row, "hand_card")[chosen_slot])
                    if chosen_card_id != sparky_id:
                        if 48 * scale // 10 <= avail < 6 * scale:
                            found_branch[lane] = dict(
                                branch_type="pre6_dump",
                                t_star=step,
                                avail_elixir=avail / scale,
                                chosen_card=vector.vocabulary[chosen_card_id],
                                chosen_cost=int(s.field(row, "hand_cost")[chosen_slot]),
                                prefix_actions=list(history[lane]),
                                orig_pair=(int(actions[lane, 0]), int(actions[lane, 1])),
                            )
                        elif legal and avail >= 6 * scale:
                            found_branch[lane] = dict(
                                branch_type="at6_skip",
                                t_star=step,
                                avail_elixir=avail / scale,
                                chosen_card=vector.vocabulary[chosen_card_id],
                                chosen_cost=int(s.field(row, "hand_cost")[chosen_slot]),
                                prefix_actions=list(history[lane]),
                                orig_pair=(int(actions[lane, 0]), int(actions[lane, 1])),
                            )
                history[lane].append((int(actions[lane, 0]), int(actions[lane, 1])))

            vector.actions[:] = actions
            vector.step_lanes(active)
            for lane in active:
                seat = seats[lane]
                enemy_view = env.observations[lane, 1 - seat]
                cnt = int(s.field(enemy_view, "enemy_play_count"))
                if cnt > pending_counts[lane]:
                    newest = s.field(enemy_view, "enemy_play_card")[: cnt - pending_counts[lane]]
                    stats[lane]["confirmed_sparky"] += int((newest == sparky_id).sum())
                    pending_counts[lane] = cnt
            live &= env.dones == 0

        for lane in range(len(chunk)):
            seed, own, opp, seat = chunk[lane]
            info = vector.info(lane)
            obs_end = env.observations[lane, seat]
            win = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
            rec = dict(
                seed=seed,
                own_deck=own,
                opp_deck=opp,
                seat=seat,
                win_score=win,
                won=int(info["winner"] == seat),
                crown_diff=int(s.field(obs_end, "crowns") - s.field(obs_end, "crowns_enemy")),
                legal_windows=stats[lane]["legal_windows"],
                legal_frames=stats[lane]["legal_frames"],
                sparky_prob_sum=stats[lane]["sparky_prob_sum"],
                sparky_requests=stats[lane]["sparky_requests"],
                confirmed_sparky=stats[lane]["confirmed_sparky"],
                low_pressure_5_to_5_6_frames=stats[lane]["low_pressure_5_to_5_6_frames"],
                low_pressure_5_to_5_6_plays=stats[lane]["low_pressure_5_to_5_6_plays"],
            )
            natural_records.append(rec)
            if found_branch[lane] is not None:
                b = found_branch[lane]
                b.update(seed=seed, own_deck=own, opp_deck=opp, seat=seat)
                branches.append(b)
    return natural_records, branches


def run_phase2_macro_rollouts(env, inference, branches, decks, forms, rollouts_per_arm: int):
    """For each branch state s_{t*}, replay prefix identically across 3*K lanes and run MC rollouts."""
    lanes = env.count
    assert lanes == len(MACRO_ARMS) * rollouts_per_arm
    branch_results = []
    vector = env.vector
    s = vector.schema
    sparky_id = vector.card_id("ZapMachine")
    giant_id = vector.card_id("Giant")
    tiles = s.cols * s.rows
    scale = s.scales["elixir_scale"]
    diag = SparkyCurriculum(s, sparky_id, vector.max_elixir, 0, 0)

    back_tile_mask = np.zeros(tiles, dtype=bool)
    back_tile_mask[: 6 * s.cols] = True
    col_grid = np.arange(tiles) % s.cols
    row_grid = np.arange(tiles) // s.cols
    tile_y_grid = row_grid * s.tile_units + s.tile_units // 2

    began = time.perf_counter()
    for b_idx, br in enumerate(branches):
            seat = br["seat"]
            t_star = br["t_star"]
            deck_ids = [br["opp_deck"], br["opp_deck"]]
            deck_ids[seat] = br["own_deck"]
            for lane in range(lanes):
                vector.reset_env(lane, br["seed"], *(decks[i] for i in deck_ids), *(forms[i] for i in deck_ids))
            inference.reset()

            # 1) Exact prefix replay [0 .. t_star - 1]
            all_lanes = np.arange(lanes)
            for a0, a1 in br["prefix_actions"]:
                obs_flat = env.observations.reshape(2 * lanes, s.seat_ints)
                mask_flat = env.masks.reshape(2 * lanes, s.actions)
                inference.evaluate(obs_flat, mask_flat)
                vector.actions[:, 0] = a0
                vector.actions[:, 1] = a1
                vector.step_lanes(all_lanes)

            if not np.array_equal(env.observations, np.broadcast_to(env.observations[0:1], env.observations.shape)):
                raise RuntimeError(f"Prefix replay state mismatch at branch {b_idx} (t_star={t_star})")

            origin_hp = diag.tower_hp_margin(env.observations[0, seat])
            pending_counts = np.array(
                [int(s.field(env.observations[l, 1 - seat], "enemy_play_count")) for l in range(lanes)], dtype=int
            )

            # Lane layout: lane = k * 3 + arm_idx
            phases = ["done" if lane % 3 == 0 else "wait_sparky" for lane in range(lanes)]
            sparky_left = [None] * lanes
            t_sparky = [None] * lanes
            sparky_triggered = np.zeros(lanes, dtype=bool)
            giant_triggered = np.zeros(lanes, dtype=bool)
            confirmed_sparky = np.zeros(lanes, dtype=int)
            confirmed_giant = np.zeros(lanes, dtype=int)
            hp_16s = np.full(lanes, np.nan, dtype=np.float64)
            hp_32s = np.full(lanes, np.nan, dtype=np.float64)

            rngs = [np.random.default_rng([br["seed"], 0x4D435453, k]) for k in range(rollouts_per_arm)]
            live = np.ones(lanes, dtype=bool)

            for step in range(t_star, 1000):
                if not live.any():
                    break
                raw = env.observations
                native = env.masks
                behavior = native.copy()
                active = np.flatnonzero(live)

                for lane in active:
                    arm = MACRO_ARMS[lane % 3]
                    row = raw[lane, seat]
                    if step == t_star and arm == "original":
                        # Force the exact non-Sparky action the model took in Phase 1
                        behavior[lane, 0, :] = False
                        behavior[lane, 0, br["orig_pair"][0]] = True
                        behavior[lane, 1, :] = False
                        behavior[lane, 1, br["orig_pair"][1]] = True
                        continue

                    ph = phases[lane]
                    if ph == "wait_sparky":
                        slot, avail, legal = sparky_status(s, row, native[lane, seat], sparky_id)
                        if (
                            step - t_star > 10
                            or slot is None
                            or int(s.field(row, "pending_count")) != 0
                            or not diag.low_pressure(row)
                        ):
                            phases[lane] = "done"
                        elif not legal:
                            if native[lane, seat, s.wait]:
                                behavior[lane, seat, :] = False
                                behavior[lane, seat, s.wait] = True
                            else:
                                phases[lane] = "done"
                        else:
                            first = 1 + slot * tiles
                            sp_native = native[lane, seat, first : first + tiles]
                            sp_back = sp_native & back_tile_mask
                            behavior[lane, seat, :] = False
                            behavior[lane, seat, first : first + tiles] = sp_back if sp_back.any() else sp_native
                            sparky_triggered[lane] = True

                    elif ph == "wait_giant":
                        if step - t_sparky[lane] > 45 or emergency_defense_needed(s, row):
                            phases[lane] = "done"
                        elif step - t_sparky[lane] >= 6:
                            sp_pos = find_own_sparky(s, row, sparky_left[lane])
                            if sp_pos is None or sp_pos[1] > 16000:
                                phases[lane] = "done"
                            else:
                                _, sp_y, _ = sp_pos
                                g_slots = np.flatnonzero(s.field(row, "hand_card") == giant_id)
                                avail = int(s.field(row, "elixir")) - int(s.field(row, "elixir_reserved"))
                                if len(g_slots) == 1 and sp_y >= 6000 and int(s.field(row, "pending_count")) == 0:
                                    g_slot = int(g_slots[0])
                                    g_first = 1 + g_slot * tiles
                                    g_native = native[lane, seat, g_first : g_first + tiles]
                                    if g_native.any():
                                        same_lane = (col_grid < 9) if sparky_left[lane] else (col_grid >= 9)
                                        in_front = (tile_y_grid > sp_y) & (row_grid <= 15)
                                        front_legal = g_native & same_lane & in_front
                                        behavior[lane, seat, :] = False
                                        behavior[lane, seat, g_first : g_first + tiles] = (
                                            front_legal if front_legal.any() else g_native
                                        )
                                        giant_triggered[lane] = True
                                        phases[lane] = "done"
                                    elif avail >= 32 * scale // 10 and native[lane, seat, s.wait]:
                                        behavior[lane, seat, :] = False
                                        behavior[lane, seat, s.wait] = True

                obs_flat = raw.reshape(2 * lanes, s.seat_ints)
                mask_flat = behavior.reshape(2 * lanes, s.actions)
                logits, _ = inference.evaluate(obs_flat, mask_flat)
                # Common random numbers across the 3 arms for each rollout k
                draws = np.repeat([rng.random(2) for rng in rngs], len(MACRO_ARMS), axis=0).astype(np.float32)
                u = torch.as_tensor(draws, device=inference.device).reshape(-1, 1)
                actions_flat, _ = sample_actions(logits, u)
                actions = actions_flat.reshape(lanes, 2)

                for lane in active:
                    if phases[lane] == "wait_sparky" and sparky_triggered[lane]:
                        act = int(actions[lane, seat])
                        tile = (act - 1) % tiles
                        sparky_left[lane] = bool((tile % s.cols) < 9)
                        t_sparky[lane] = step
                        phases[lane] = "wait_giant" if MACRO_ARMS[lane % 3] == "sparky_giant_combo" else "done"

                if not native[active[:, None], np.arange(2), actions[active]].all():
                    raise RuntimeError("Selected macro action outside native legal mask")

                vector.actions[:] = actions
                vector.step_lanes(active)

                for lane in active:
                    following = env.observations[lane, seat]
                    ended = bool(env.dones[lane])
                    enemy_view = env.observations[lane, 1 - seat]
                    cnt = int(s.field(enemy_view, "enemy_play_count"))
                    if cnt > pending_counts[lane]:
                        newest = s.field(enemy_view, "enemy_play_card")[: cnt - pending_counts[lane]]
                        confirmed_sparky[lane] += int((newest == sparky_id).sum())
                        confirmed_giant[lane] += int((newest == giant_id).sum())
                        pending_counts[lane] = cnt

                    dt = step - t_star + 1
                    cur_margin = diag.tower_hp_margin(following) - origin_hp
                    if np.isnan(hp_16s[lane]) and (dt >= 40 or ended):
                        hp_16s[lane] = cur_margin
                    if np.isnan(hp_32s[lane]) and (dt >= 80 or ended):
                        hp_32s[lane] = cur_margin
                live &= env.dones == 0

            arm_outcomes = {arm: [] for arm in MACRO_ARMS}
            for k in range(rollouts_per_arm):
                for ai, arm in enumerate(MACRO_ARMS):
                    lane = k * 3 + ai
                    info = vector.info(lane)
                    obs_end = env.observations[lane, seat]
                    win_score = 1.0 if info["winner"] == seat else (0.0 if info["winner"] == 1 - seat else 0.5)
                    margin_end = diag.tower_hp_margin(obs_end) - origin_hp
                    arm_outcomes[arm].append(
                        dict(
                            k=k,
                            win_score=win_score,
                            won=int(info["winner"] == seat),
                            crown_diff=int(s.field(obs_end, "crowns") - s.field(obs_end, "crowns_enemy")),
                            hp_16s=float(margin_end if np.isnan(hp_16s[lane]) else hp_16s[lane]),
                            hp_32s=float(margin_end if np.isnan(hp_32s[lane]) else hp_32s[lane]),
                            sparky_triggered=bool(sparky_triggered[lane]),
                            giant_triggered=bool(giant_triggered[lane]),
                            confirmed_sparky=int(confirmed_sparky[lane]),
                            confirmed_giant=int(confirmed_giant[lane]),
                        )
                    )

            branch_results.append(
                dict(
                    branch_index=b_idx,
                    seed=br["seed"],
                    branch_type=br["branch_type"],
                    t_star=t_star,
                    avail_elixir=br["avail_elixir"],
                    chosen_card=br["chosen_card"],
                    chosen_cost=br["chosen_cost"],
                    outcomes=arm_outcomes,
                )
            )
            if (b_idx + 1) % 5 == 0 or b_idx + 1 == len(branches):
                print(
                    f"[Phase 2] {b_idx + 1}/{len(branches)} branches ({(b_idx + 1) * lanes} rollouts) "
                    f"in {time.perf_counter() - began:.1f}s",
                    flush=True,
                )
    return branch_results


def summarize_and_print(checkpoint: Path, natural_records, branch_results, rollouts_per_arm: int):
    n_games = len(natural_records)
    tot_win_rate = np.mean([r["won"] for r in natural_records])
    tot_score = np.mean([r["win_score"] for r in natural_records])
    tot_cd = np.mean([r["crown_diff"] for r in natural_records])
    tot_windows = sum(r["legal_windows"] for r in natural_records)
    tot_frames = sum(r["legal_frames"] for r in natural_records)
    tot_prob = sum(r["sparky_prob_sum"] for r in natural_records) / max(1, tot_frames)
    tot_conf = sum(r["confirmed_sparky"] for r in natural_records)
    games_with_sparky = sum(r["confirmed_sparky"] > 0 for r in natural_records)
    early_frames = sum(r["low_pressure_5_to_5_6_frames"] for r in natural_records)
    early_plays = sum(r["low_pressure_5_to_5_6_plays"] for r in natural_records)

    print("\n" + "=" * 72)
    print(f"PHASE 1: NATURAL POLICY SPARKY DIAGNOSTIC ({checkpoint.name}, {n_games} games)")
    print("=" * 72)
    print(f"  Win rate (strict wins): {tot_win_rate:.1%} (win_score={tot_score:.3f}, crown_diff={tot_cd:+.2f})")
    print(f"  Legal Sparky windows  : {tot_windows} ({tot_windows * 100 / n_games:.1f} / 100 games)")
    print(f"  Mean P(Sparky | legal): {tot_prob:.1%} across {tot_frames} legal frames")
    print(f"  Confirmed Sparky plays: {tot_conf} ({tot_conf * 100 / n_games:.1f} / 100 games, {games_with_sparky}/{n_games} games played >=1)")
    print(f"  Low-pressure [5.0,5.6): played other card on {early_plays}/{early_frames} frames ({early_plays / max(1, early_frames):.1%})")

    if not branch_results:
        print("\nNo missed-Sparky branch points found!")
        return {}

    print("\n" + "=" * 72)
    print(f"PHASE 2: MCTS MACRO-ROLLOUT LOOKAHEAD ({len(branch_results)} branches x 3 arms x {rollouts_per_arm} rollouts)")
    print("=" * 72)

    def agg_subset(subset, label):
        if not subset:
            return
        print(f"\n--- Subset: {label} (N={len(subset)} branches) ---")
        for arm in MACRO_ARMS:
            ws = [ np.mean([x["win_score"] for x in b["outcomes"][arm]]) for b in subset ]
            cd = [ np.mean([x["crown_diff"] for x in b["outcomes"][arm]]) for b in subset ]
            h16 = [ np.mean([x["hp_16s"] for x in b["outcomes"][arm]]) for b in subset ]
            h32 = [ np.mean([x["hp_32s"] for x in b["outcomes"][arm]]) for b in subset ]
            sp_c = [ np.mean([x["confirmed_sparky"] for x in b["outcomes"][arm]]) for b in subset ]
            gi_c = [ np.mean([x["confirmed_giant"] for x in b["outcomes"][arm]]) for b in subset ]
            print(
                f"  {arm:20s}: win_score={np.mean(ws):.3f} | crown_diff={np.mean(cd):+.2f} | "
                f"HP@16s={np.mean(h16):+6.0f} | HP@32s={np.mean(h32):+6.0f} | "
                f"Sparky={np.mean(sp_c):.2f} Giant={np.mean(gi_c):.2f}"
            )

        # Leave-One-Out Cross-Validated MCTS selector (unbiased out-of-sample evaluation!)
        loo_ws, loo_cd, loo_h32 = [], [], []
        chosen_counts = Counter()
        combo_beats_orig = 0
        any_sparky_beats_orig = 0
        for b in subset:
            m_orig = np.mean([x["win_score"] for x in b["outcomes"]["original"]])
            m_sink = np.mean([x["win_score"] for x in b["outcomes"]["sink_sparky"]])
            m_comb = np.mean([x["win_score"] for x in b["outcomes"]["sparky_giant_combo"]])
            if m_comb > m_orig:
                combo_beats_orig += 1
            if max(m_sink, m_comb) > m_orig:
                any_sparky_beats_orig += 1

            for k in range(rollouts_per_arm):
                # Score each arm using the OTHER K-1 rollouts (win_score + 0.05 * crown_diff tie-break)
                best_arm = max(
                    MACRO_ARMS,
                    key=lambda a: np.mean(
                        [
                            x["win_score"] + 0.05 * x["crown_diff"]
                            for j, x in enumerate(b["outcomes"][a])
                            if j != k
                        ]
                    ),
                )
                chosen_counts[best_arm] += 1
                held = b["outcomes"][best_arm][k]
                loo_ws.append(held["win_score"])
                loo_cd.append(held["crown_diff"])
                loo_h32.append(held["hp_32s"])

        print(
            f"  {'mcts_loo_selector':20s}: win_score={np.mean(loo_ws):.3f} | crown_diff={np.mean(loo_cd):+.2f} | "
            f"HP@32s={np.mean(loo_h32):+6.0f} | picked={dict(chosen_counts)}"
        )
        print(
            f"  Branches where Sparky option > Original: {any_sparky_beats_orig}/{len(subset)} "
            f"({any_sparky_beats_orig / len(subset):.1%}), Combo > Original: {combo_beats_orig}/{len(subset)} ({combo_beats_orig / len(subset):.1%})"
        )

    agg_subset(branch_results, "ALL MISSED-SPARKY BRANCHES")
    agg_subset([b for b in branch_results if b["branch_type"] == "pre6_dump"], "PRE-6 ELIXIR DUMP ([4.8, 6.0))")
    agg_subset([b for b in branch_results if b["branch_type"] == "at6_skip"], "AT-6+ ELIXIR SKIP (>= 6.0)")
    agg_subset(
        [
            b
            for b in branch_results
            if any(x["giant_triggered"] for x in b["outcomes"]["sparky_giant_combo"])
        ],
        "BRANCHES WHERE GIANT-IN-FRONT ACTUALLY TRIGGERED",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scan-games", type=int, default=84)
    parser.add_argument("--max-branches", type=int, default=25)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    torch.set_num_threads(2)
    t0 = time.perf_counter()
    # Start compact JVM BEFORE initializing CUDA so Windows commit charge stays tiny.
    sim_paths = SimulatorPaths.discover()
    _jvm.start(
        sim_paths.install,
        jvm_args=("-Xmx256m", "-Xms64m", "-XX:MaxMetaspaceSize=96m", "-XX:CompressedClassSpaceSize=32m", "-XX:+UseSerialGC"),
    )

    lanes = len(MACRO_ARMS) * args.rollouts
    model, loaded, _ = load_policy_inference(args.checkpoint, args.device, compute_value=False)
    gc.collect()
    _, names, decks, forms = _extract_decks(ROOT / "configs/decks_all51.yaml")
    own_indices = [names.index(name) for name in DECKS]
    rivals = [i for i in range(len(names)) if i not in own_indices]
    rng = np.random.default_rng(args.seed)
    seed_rows = [
        (int(rng.integers(2**31 - 1)), own_indices[i % 2], int(rng.choice(rivals)), i % 4 // 2)
        for i in range(args.scan_games)
    ]

    with CrForgeVector(lanes, min(4, lanes), paths=sim_paths, independent=True, decision_ticks=8) as env:
        env.set_autoreset(False)
        inference = type(loaded)(
            model, loaded.adapter, loaded.device, 2 * lanes, precision=loaded.precision, compute_value=False
        )
        print(f"[Phase 1] Scanning {args.scan_games} natural games ({lanes} lanes) with {args.checkpoint.name}...", flush=True)
        natural_records, branches = run_phase1_scan(env, inference, seed_rows, decks, forms)
        print(f"[Phase 1] Done in {time.perf_counter() - t0:.1f}s: found {len(branches)} missed-Sparky branches.", flush=True)

        selected_branches = branches[: args.max_branches]
        branch_results = []
        if selected_branches:
            print(
                f"[Phase 2] Running MCTS Macro-Rollout Lookahead on {len(selected_branches)} branches "
                f"(3 arms x {args.rollouts} rollouts = {lanes} lanes/branch)...",
                flush=True,
            )
            branch_results = run_phase2_macro_rollouts(
                env, inference, selected_branches, decks, forms, args.rollouts
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        checkpoint=str(args.checkpoint),
        scan_games=args.scan_games,
        rollouts_per_arm=args.rollouts,
        natural_records=natural_records,
        branch_results=branch_results,
    )
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summarize_and_print(args.checkpoint, natural_records, branch_results, args.rollouts)


if __name__ == "__main__":
    main()
