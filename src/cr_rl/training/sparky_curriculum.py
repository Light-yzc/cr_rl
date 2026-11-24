"""Opt-in, bounded interventions in native Sparky games."""

from collections import Counter

import numpy as np


class SparkyCurriculum:
    def __init__(self, schema, card_id, max_elixir, wait_fraction, play_fraction, wait_max_steps=3):
        self.schema = schema
        self.card_id = card_id
        self.max_elixir = max_elixir
        self.wait_fraction = wait_fraction
        self.play_fraction = play_fraction
        self.wait_max_steps = wait_max_steps
        self.tiles = schema.cols * schema.rows
        self.elixir_scale = schema.scales['elixir_scale']
        self.lanes = {}
        self.followups = {}
        self.own_hp = [schema.header[f'own_{tower}_hp'][0] for tower in ('left', 'right', 'king')]
        self.enemy_hp = [schema.header[f'enemy_{tower}_hp'][0] for tower in ('left', 'right', 'king')]

    def reset_lane(self, lane, seed, episode_id, observation):
        self.followups.pop(lane, None)
        if self.card_id not in self.schema.field(observation, 'deck'):
            self.lanes.pop(lane, None)
            return
        draw = np.random.default_rng([seed, episode_id, 0x53504152]).random()
        arm = 'control'
        if draw < self.wait_fraction:
            arm = 'reserve_wait'
        elif draw < self.wait_fraction + self.play_fraction:
            arm = 'sparky_play'
        self.lanes[lane] = (arm, 0)  # 0 = unused, positive = wait steps taken, -1 = closed.

    def low_pressure(self, observation):
        s = self.schema
        if s.field(observation, 'entity_overflow') != 0:
            return False
        e = s.entity
        rows = s.entities(observation)
        enemy = rows[rows[:, e['side']] == 1]
        kinds = enemy[:, e['kind']]
        incoming = (kinds == s.kinds['projectile']) | (kinds == s.kinds['area'])
        near = (kinds == s.kinds['character']) & (
            (enemy[:, e['flags']] & s.flags['alive']) != 0
        ) & (enemy[:, e['y']] <= s.height_units // 2 + 2 * s.tile_units)
        return not (incoming | near).any()

    def prepare(self, raw, masks, active, learner_seats, stats):
        """Restrict only selected learner rows. Return native Sparky opportunities."""
        s = self.schema
        candidates = []
        behavior_masks = masks
        for lane, (arm, wait_steps) in self.lanes.items():
            if not active[lane]:
                continue
            seat = int(learner_seats[lane])
            observation = raw[lane, seat]
            slots = np.flatnonzero(s.field(observation, 'hand_card') == self.card_id)
            counts = stats[arm]
            if len(slots) != 1 or s.field(observation, 'hand_cost')[slots[0]] != 6:
                if wait_steps > 0:
                    self.lanes[lane] = (arm, -1)
                    counts['reserve_windows_aborted'] += 1
                continue
            slot = int(slots[0])
            cost = 6
            first = 1 + slot * self.tiles
            end = first + self.tiles
            native = masks[lane, seat]
            legal = bool(native[first:end].any())
            available = s.field(observation, 'elixir') - s.field(observation, 'elixir_reserved')
            counts['in_hand_frames'] += 1
            counts['enough_elixir_frames'] += int(available >= cost * self.elixir_scale)
            counts['legal_frames'] += int(legal)

            pre_six = (cost - 1) * self.elixir_scale <= available < cost * self.elixir_scale
            intervention = None
            if arm == 'reserve_wait' and wait_steps >= 0:
                remaining = self.wait_max_steps - wait_steps
                if self._can_reserve(observation, native, available, legal, remaining):
                    intervention = 'forced_wait'
                    self.lanes[lane] = (arm, wait_steps + 1)
                    if wait_steps == 0:
                        counts['reserve_windows_started'] += 1
                elif wait_steps > 0:
                    self.lanes[lane] = (arm, -1)
                    counts['reserve_windows_reached_legal' if legal else 'reserve_windows_aborted'] += 1
            elif wait_steps == 0 and arm == 'sparky_play' and legal and native[s.wait] and (
                self.low_pressure(observation)
            ):
                intervention = 'forced_play'
                self.lanes[lane] = (arm, -1)
            if intervention is not None:
                if behavior_masks is masks:
                    behavior_masks = masks.copy()
                row = behavior_masks[lane, seat]
                row[:] = False
                if intervention == 'forced_wait':
                    row[s.wait] = True
                else:
                    row[first:end] = native[first:end]
                counts[intervention] += 1
            if legal or pre_six:
                candidates.append((lane, seat, arm, first, end, legal, pre_six, intervention))
        return behavior_masks, candidates

    def _can_reserve(self, observation, native, available, legal, remaining):
        s = self.schema
        if (remaining <= 0 or legal or not native[s.wait]
                or not 56 * self.elixir_scale // 10 <= available < 6 * self.elixir_scale
                or s.field(observation, 'pending_count') != 0):
            return False
        bar_ms = s.field(observation, 'full_bar_ms')
        if bar_ms <= 0:
            return False
        # MatchSide.regenerate truncates the regenerated elixir each native tick.
        gain_per_tick = self.max_elixir * self.elixir_scale * s.scales['ms_per_tick'] // bar_ms
        gain = gain_per_tick * s.field(observation, 'decision_ticks') * remaining
        return (available + gain >= 6 * self.elixir_scale
                and native[1:s.first_ability].any() and self.low_pressure(observation))

    def record_choices(self, candidates, actions, raw, stats):
        excluded = set()
        for lane, seat, arm, first, end, legal, pre_six, intervention in candidates:
            if intervention is not None:
                excluded.add((lane, seat))
            action = actions[lane, seat]
            sparky = first <= action < end
            if legal and intervention is None:
                stats[arm]['voluntary_legal_frames'] += 1
                stats[arm]['voluntary_sparky_requests'] += int(sparky)
            if pre_six and 0 < action < self.schema.first_ability and not sparky:
                stats[arm]['other_card_requests_at_5_to_6'] += 1
            if sparky:
                observation = raw[lane, seat]
                kind = 'forced' if intervention is not None else 'voluntary'
                deadline = self.schema.field(observation, 'episode_step') + 40
                self.followups.setdefault(lane, []).append((seat, arm, kind, deadline, self.tower_hp_margin(observation)))
        return excluded

    def tower_hp_margin(self, observation):
        return int(observation[self.own_hp].sum() - observation[self.enemy_hp].sum())

    def record_followups(self, following, terminated, truncated, stats):
        """Track net tower-HP change over 40 decisions, or to a true terminal."""
        for lane in np.flatnonzero(terminated | truncated):
            arm, wait_steps = self.lanes.get(lane, ('control', 0))
            if wait_steps > 0:
                self.lanes[lane] = (arm, -1)
                stats[arm]['reserve_windows_aborted'] += 1
        for lane, pending in list(self.followups.items()):
            remaining = []
            if not truncated[lane]:
                for seat, arm, kind, deadline, starting_margin in pending:
                    observation = following[lane, seat]
                    step = self.schema.field(observation, 'episode_step')
                    if terminated[lane] or step >= deadline:
                        stats[arm][f'{kind}_sparky_hp_windows'] += 1
                        stats[arm][f'{kind}_sparky_hp_swing_sum'] += self.tower_hp_margin(observation) - starting_margin
                    else:
                        remaining.append((seat, arm, kind, deadline, starting_margin))
            if remaining:
                self.followups[lane] = remaining
            else:
                del self.followups[lane]

    def record_terminal(self, lane, reward, stats):
        if lane in self.lanes:
            arm, _ = self.lanes[lane]
            stats[arm]['terminal_games'] += 1
            stats[arm]['wins'] += int(reward > 0)

    @staticmethod
    def new_stats():
        return {arm: Counter() for arm in ('control', 'reserve_wait', 'sparky_play')}
