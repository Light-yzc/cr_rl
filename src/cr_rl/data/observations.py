from __future__ import annotations

import numpy as np


class CrForgeObservationAdapter:
    """Split the pinned crforge v4 flat seat buffer into model-ready frame tensors."""

    def __init__(self, schema, vocabulary_size: int = 1466):
        self.schema = schema
        self.vocabulary_size = vocabulary_size

    @staticmethod
    def _scale(values: np.ndarray, divisor: float) -> np.ndarray:
        return values / np.float32(divisor)

    def __call__(self, observations: np.ndarray):
        expected = self.schema.seat_ints
        if observations.shape[-1] != expected:
            raise ValueError(f"Expected {expected} ints per seat, got {observations.shape[-1]}")
        header = observations[..., : self.schema.header_ints].astype(np.float32, copy=True)
        entities = (
            observations[..., self.schema.header_ints :]
            .reshape(*observations.shape[:-1], self.schema.max_entities, self.schema.entity_ints)
            .astype(np.float32, copy=True)
        )
        count_index = self.schema.header["entity_count"][0]
        counts = np.clip(observations[..., count_index], 0, self.schema.max_entities)
        entity_mask = np.arange(self.schema.max_entities) < counts[..., None]
        entity_mask[..., 0] |= counts == 0

        # Unknown (-1) remains distinct. Field names come from the pinned simulator schema.
        card_fields = {
            "hand_card",
            "next_card",
            "deck",
            "enemy_seen",
            "pending_card",
            "ability_champion",
            "enemy_play_card",
        }
        for index, name in enumerate(self.schema.header_fields):
            if name in card_fields:
                valid = header[..., index] >= 0
                header[..., index] = np.where(
                    valid, header[..., index] / self.vocabulary_size, -1.0
                )
            elif name in {"elixir", "elixir_reserved", "enemy_elixir"}:
                header[..., index] = self._scale(header[..., index], 100000.0)
            elif name in {"tick", "enemy_play_ticks", "enemy_play_point_ticks"}:
                header[..., index] = self._scale(header[..., index], 7200.0)
            elif name == "episode_step":
                header[..., index] = self._scale(header[..., index], 1440.0)
            elif name in {
                "time_ms",
                "section_left_ms",
                "full_bar_ms",
                "hand_refill_ms",
                "ability_cooldown_ms",
            }:
                header[..., index] = self._scale(header[..., index], 360000.0)
            elif name in {"pending_x", "enemy_play_x"}:
                header[..., index] = self._scale(header[..., index], 18000.0)
            elif name in {"pending_y", "enemy_play_y"}:
                header[..., index] = self._scale(header[..., index], 32000.0)
            elif name in {
                "own_king_hp",
                "own_king_max_hp",
                "own_left_hp",
                "own_left_max_hp",
                "own_right_hp",
                "own_right_max_hp",
                "enemy_king_hp",
                "enemy_king_max_hp",
                "enemy_left_hp",
                "enemy_left_max_hp",
                "enemy_right_hp",
                "enemy_right_max_hp",
            }:
                header[..., index] = self._scale(header[..., index], 10000.0)
            elif name == "hand_cost":
                header[..., index] = self._scale(header[..., index], 10.0)
            elif name == "deck_level":
                header[..., index] = self._scale(header[..., index], 15.0)
            elif name in {"crowns", "crowns_enemy"}:
                header[..., index] = self._scale(header[..., index], 3.0)
            elif name in {
                "pending_ticks",
                "play_delay_ticks",
                "play_open_in_ticks",
                "ability_command_ticks",
            }:
                header[..., index] = self._scale(header[..., index], 100.0)

        entity_index = {name: i for i, name in enumerate(self.schema.entity_fields)}
        divisors = {
            "row": self.vocabulary_size,
            "kind": 8.0,
            "side": 2.0,
            "state": 16.0,
            "flags": 65535.0,
            "x": 18000.0,
            "target_x": 18000.0,
            "aim_x": 18000.0,
            "y": 32000.0,
            "target_y": 32000.0,
            "aim_y": 32000.0,
            "hp": 10000.0,
            "max_hp": 10000.0,
            "shield": 10000.0,
            "level": 15.0,
            "dx": 18000.0,
            "dy": 32000.0,
            "deploy_ms": 10000.0,
            "attack_ms": 10000.0,
            "load_ms": 10000.0,
            "speed_percent": 200.0,
            "hit_speed_percent": 200.0,
            "buff_count": 3.0,
            "buff_0": 100.0,
            "buff_1": 100.0,
            "buff_2": 100.0,
            "buff_ms_0": 10000.0,
            "buff_ms_1": 10000.0,
            "buff_ms_2": 10000.0,
            "time_left_ms": 10000.0,
            "radius": 18000.0,
            "ability_ticks": 100.0,
        }
        for name, divisor in divisors.items():
            if name in entity_index:
                entities[..., entity_index[name]] = self._scale(
                    entities[..., entity_index[name]], divisor
                )
        for name in ("id", "target"):
            if name in entity_index:
                entities[..., entity_index[name]] = 0.0
        return header, entities, entity_mask
