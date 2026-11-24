"""Paired native games with a frozen v3 actor and common action draws."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.sparky_curriculum import SparkyCurriculum
from cr_rl.training.state_selfplay import _extract_decks
from run_nulls_dryrun import load_policy_inference


DECKS = ('user_giant_sparky_pump_plain', 'user_giant_sparky_pump_horde_plain')
ARMS = ('original', 'one_step', 'three_steps')


def sparky_status(schema, obs, mask, card_id):
    available = schema.field(obs, 'elixir') - schema.field(obs, 'elixir_reserved')
    slots = np.flatnonzero(schema.field(obs, 'hand_card') == card_id)
    slot = int(slots[0]) if len(slots) == 1 else None
    legal = False
    if slot is not None:
        tiles = schema.cols * schema.rows
        first = 1 + slot * tiles
        legal = bool(mask[first:first + tiles].any())
    return slot, available, legal


def policy_choice(vector, obs, action):
    s = vector.schema
    if action == s.wait:
        return {'action': 'wait'}
    if action >= s.first_ability:
        return {'action': 'ability', 'ability_slot': int(action - s.first_ability)}
    slot = int((action - 1) // (s.cols * s.rows))
    return {'action': 'play', 'card': vector.vocabulary[int(s.field(obs, 'hand_card')[slot])],
            'cost': int(s.field(obs, 'hand_cost')[slot])}


def next_sparky_status(schema, obs, mask, card_id, ended):
    slot, available, legal = sparky_status(schema, obs, mask, card_id)
    pending = bool(slot is not None and schema.field(obs, 'hand_pending')[slot])
    if ended:
        status = 'episode_end'
    elif legal:
        status = 'still_legal'
    elif slot is None:
        status = 'card_left_hand'
    elif pending:
        status = 'card_pending'
    elif available < schema.field(obs, 'hand_cost')[slot] * schema.scales['elixir_scale']:
        status = 'below_card_cost'
    else:
        status = 'native_mask_closed'
    return {'status': status, 'available_elixir': available / schema.scales['elixir_scale'],
            'hand_pending': pending, 'legal': legal}


def play_wave(env, inference, seed_rows, decks, forms, arms=ARMS, human_bc=None):
    vector = env.vector
    s = vector.schema
    card_id = vector.card_id('ZapMachine')
    tiles = s.cols * s.rows
    lanes = len(seed_rows) * len(arms)
    seats = np.zeros(lanes, int)
    records = []
    for arm in arms:
        records.extend(dict(seed=seed, arm=arm, own_deck=own, opponent_deck=opp, seat=seat)
                       for seed, own, opp, seat in seed_rows)
    curricula = {arm: SparkyCurriculum(s, card_id, vector.max_elixir, 1, 0, steps)
                 for arm, steps in zip(ARMS[1:], (1, 3)) if arm in arms}
    diagnostics = SparkyCurriculum(s, card_id, vector.max_elixir, 0, 0)
    scale = s.scales['elixir_scale']
    hp_horizon = 16000 // (8 * s.scales['ms_per_tick'])
    live = np.ones(lanes, bool)
    for lane, row in enumerate(records):
        seat = seats[lane] = row['seat']
        deck_ids = [row['opponent_deck']] * 2
        deck_ids[seat] = row['own_deck']
        vector.reset_env(lane, row['seed'], *(decks[i] for i in deck_ids), *(forms[i] for i in deck_ids))
        if row['arm'] in curricula:
            curriculum = curricula[row['arm']]
            curriculum.reset_lane(lane, row['seed'], lane, env.observations[lane, seat])
    inference.reset()
    if human_bc is not None:
        human_bc.reset()
        bc_rows = np.flatnonzero([r['arm'] == 'human_bc' for r in records])
        bc_rows = 2 * bc_rows + seats[bc_rows]
        ppo_rows = np.setdiff1d(np.arange(2 * lanes), bc_rows, assume_unique=True)
        bc_device_rows = torch.as_tensor(bc_rows, device=inference.device)
        ppo_device_rows = torch.as_tensor(ppo_rows, device=inference.device)
    rngs = [np.random.default_rng([seed, 0x53504152]) for seed, *_ in seed_rows]
    metrics = [Counter() for _ in records]
    early_cards = [Counter() for _ in records]
    early_choices = [[] for _ in records]
    legal_choices = [[] for _ in records]
    hp_followups = [[] for _ in records]
    previously_legal = np.zeros(lanes, bool)
    previously_forced = np.zeros(lanes, bool)
    pending_play_counts = np.zeros(lanes, int)
    lane_ids = np.arange(lanes)
    for _ in range(1000):
        if not live.any():
            break
        raw = env.observations
        native = env.masks
        behavior = native
        forced = np.zeros(lanes, bool)
        for curriculum in curricula.values():
            behavior, candidates = curriculum.prepare(raw, behavior, live, seats, curriculum.new_stats())
            for lane, _, _, _, _, _, _, intervention in candidates:
                forced[lane] = intervention == 'forced_wait'
        # Full-width batches keep recurrent stream indexing stable for the whole wave.
        obs = raw.reshape(2 * lanes, s.seat_ints)
        masks = behavior.reshape(2 * lanes, s.actions)
        if human_bc is None:
            logits, _ = inference.evaluate(obs, masks)
        else:
            ppo_logits, _ = inference.evaluate(obs[ppo_rows], masks[ppo_rows])
            bc_logits, _ = human_bc.evaluate(obs[bc_rows], masks[bc_rows])
            logits = ppo_logits.new_empty((2 * lanes, s.actions))
            logits[ppo_device_rows] = ppo_logits
            logits[bc_device_rows] = bc_logits
        probabilities = logits.softmax(-1)
        uniforms = torch.as_tensor(np.tile([rng.random(2) for rng in rngs], (len(arms), 1)),
                                   device=inference.device, dtype=torch.float32).reshape(-1, 1)
        cdf = probabilities.cumsum(-1)
        actions = torch.searchsorted(cdf, uniforms * cdf[:, -1:], right=True).squeeze(1)
        learner = torch.as_tensor(2 * lane_ids + seats, device=inference.device)
        card_mass = probabilities[learner, 1:s.first_ability].reshape(lanes, s.hand_slots, tiles).sum(-1)
        actions = actions.cpu().numpy().reshape(lanes, 2).astype(np.int32)
        card_mass = card_mass.cpu().numpy()

        active = np.flatnonzero(live)
        annotated = {}
        early_annotated = {}
        for lane in active:
            m = metrics[lane]
            seat = seats[lane]
            row = raw[lane, seat]
            slot, available, legal = sparky_status(s, row, native[lane, seat], card_id)
            action = actions[lane, seat]
            if slot is not None and 5 * scale <= available < 56 * scale // 10 and diagnostics.low_pressure(row):
                m['low_pressure_5_to_5_6_frames'] += 1
                if 0 < action < s.first_ability:
                    choice = policy_choice(vector, row, action)
                    early_cards[lane][choice['card']] += 1
                    early_annotated[lane] = dict(choice, tick=s.field(row, 'tick'),
                                                 available_elixir=available / scale)
            if forced[lane]:
                m['forced_steps'] += 1
                m['started_windows'] += int(not previously_forced[lane])
                if not previously_forced[lane]:
                    m[f'started_bar_ms_{s.field(row, "full_bar_ms")}'] += 1
            if legal:
                first = 1 + slot * tiles
                m['legal_windows'] += int(not previously_legal[lane])
                m['voluntary_frames'] += 1
                p_sparky = float(card_mass[lane, slot])
                m['sparky_probability_sum'] += p_sparky
                chosen = first <= action < first + tiles
                m['voluntary_requests'] += int(chosen)
                annotated[lane] = dict(policy_choice(vector, row, action),
                                       tick=s.field(row, 'tick'), low_pressure=diagnostics.low_pressure(row),
                                       available_elixir=available / scale, sparky_probability=p_sparky)
                if previously_forced[lane]:
                    m['post_reserve_frames'] += 1
                    m['post_reserve_requests'] += int(chosen)
                    m['post_reserve_probability_sum'] += p_sparky
            if previously_forced[lane] and not forced[lane] and not legal:
                m['aborted_windows'] += 1
            previously_legal[lane] = legal
            previously_forced[lane] = forced[lane]

        if not native[active[:, None], np.arange(2), actions[active]].all():
            raise RuntimeError('Sampler selected an action outside the native mask')
        vector.actions[:] = actions
        vector.step_lanes(active)
        for lane in active:
            seat = seats[lane]
            following = env.observations[lane, seat]
            ended = bool(env.dones[lane])
            if lane in annotated:
                annotated[lane]['next'] = next_sparky_status(s, following, env.masks[lane, seat], card_id, ended)
                legal_choices[lane].append(annotated[lane])
            if lane in early_annotated:
                choice = early_annotated[lane]
                choice['next'] = next_sparky_status(s, following, env.masks[lane, seat], card_id, ended)
                choice['last_refusal'] = s.field(following, 'last_refusal')
                early_choices[lane].append(choice)
            enemy_view = env.observations[lane, 1 - seat]
            count = s.field(enemy_view, 'enemy_play_count')
            if count > pending_play_counts[lane]:
                newest = s.field(enemy_view, 'enemy_play_card')[:count - pending_play_counts[lane]]
                confirmed = int((newest == card_id).sum())
                metrics[lane]['confirmed_plays'] += confirmed
                pending_play_counts[lane] = count
                if confirmed:
                    origin = diagnostics.tower_hp_margin(following)
                    deadline = s.field(following, 'episode_step') + hp_horizon
                    hp_followups[lane].extend([(deadline, origin)] * confirmed)
            if hp_followups[lane]:
                now = s.field(following, 'episode_step')
                info = vector.info(lane) if ended else {}
                valid_end = ended and not (info.get('fault') or info.get('truncated'))
                remaining = []
                for deadline, origin in hp_followups[lane]:
                    if valid_end or (not ended and now >= deadline):
                        metrics[lane]['voluntary_sparky_hp_windows'] += 1
                        metrics[lane]['voluntary_sparky_hp_swing_sum'] += diagnostics.tower_hp_margin(following) - origin
                        metrics[lane]['voluntary_sparky_hp_early_terminal_windows'] += int(now < deadline)
                    elif not ended:
                        remaining.append((deadline, origin))
                hp_followups[lane] = remaining
        live &= env.dones == 0

    for lane, row in enumerate(records):
        info = vector.info(lane)
        metrics[lane]['aborted_windows'] += int(previously_forced[lane])
        for key in ('voluntary_sparky_hp_windows', 'voluntary_sparky_hp_swing_sum',
                    'voluntary_sparky_hp_early_terminal_windows', 'low_pressure_5_to_5_6_frames'):
            metrics[lane].setdefault(key, 0)
        observation = env.observations[lane, seats[lane]]
        row.update(metrics[lane], steps=info['episode_step'], winner=info['winner'], fault=info.get('fault', 0),
                   truncated=bool(live[lane] or info['truncated']),
                   crown_difference=s.field(observation, 'crowns') - s.field(observation, 'crowns_enemy'),
                   legal_choices=legal_choices[lane], low_pressure_5_to_5_6_card_requests=dict(early_cards[lane]),
                   low_pressure_5_to_5_6_play_choices=early_choices[lane])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=120)
    parser.add_argument('--parallel-pairs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--skip-pairs', type=int, default=0,
                        help='Skip completed seed groups when continuing into a new output file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--baseline-only', action='store_true', help='Replay only the original-policy arm')
    mode.add_argument('--compare-human-bc', type=Path,
                      help='Compare PPO and human BC learner seats against the same frozen PPO opponent')
    args = parser.parse_args()
    if args.parallel_pairs < 1 or not 0 <= args.skip_pairs < args.pairs:
        parser.error('Need a positive batch and at least one unplayed pair')
    torch.set_num_threads(2)
    model, loaded, _ = load_policy_inference(args.checkpoint, 'cuda', compute_value=False)
    if args.compare_human_bc:
        bc_model, bc_loaded, _ = load_policy_inference(args.compare_human_bc, 'cuda', compute_value=False)
        print(f'Reserve-to-WAIT config: PPO={model.cfg.reserve_to_wait}, BC={bc_model.cfg.reserve_to_wait}',
              flush=True)
    _, names, decks, forms = _extract_decks(Path('configs/decks_all51.yaml'))
    own_indices = [names.index(name) for name in DECKS]
    rivals = [i for i in range(len(names)) if i not in own_indices]
    rng = np.random.default_rng(args.seed)
    rows = [(int(rng.integers(2**31 - 1)), own_indices[i % 2], int(rng.choice(rivals)), i % 4 // 2)
            for i in range(args.pairs)][args.skip_pairs:]
    batch = min(args.parallel_pairs, len(rows))
    arms = ('original', 'human_bc') if args.compare_human_bc else ARMS[:1] if args.baseline_only else ARMS
    lanes = len(arms) * batch
    with args.out.open('x', encoding='utf-8') as output, CrForgeVector(
        lanes, min(8, lanes), independent=True, decision_ticks=8,
    ) as env:
        env.set_autoreset(False)
        bc_streams = batch if args.compare_human_bc else 0
        human_bc = type(bc_loaded)(bc_model, bc_loaded.adapter, bc_loaded.device, bc_streams,
                                   precision=bc_loaded.precision, compute_value=False) if bc_streams else None
        active_streams = 2 * lanes - bc_streams
        inference = type(loaded)(model, loaded.adapter, loaded.device, active_streams,
                                 precision=loaded.precision, compute_value=False)
        started = time.perf_counter()
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            # Repeat padding seeds only to keep tensor and simulator batches fixed.
            expanded = chunk + [chunk[-1]] * (batch - len(chunk))
            results = play_wave(env, inference, expanded, decks, forms, arms, human_bc)
            for arm in range(len(arms)):
                for row in results[arm * batch:arm * batch + len(chunk)]:
                    row['own_deck'] = names[row['own_deck']]
                    row['opponent_deck'] = names[row['opponent_deck']]
                    output.write(json.dumps(row) + '\n')
            output.flush()
            print(f'{args.skip_pairs + start + len(chunk)}/{args.pairs} pairs '
                  f'in {time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
