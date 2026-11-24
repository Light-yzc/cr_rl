from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np

_ADAPTER_WORKERS = min(12, max(2, (os.cpu_count() or 8) // 3))
_ADAPTER_POOL: ThreadPoolExecutor | None = None


def _get_adapter_pool() -> ThreadPoolExecutor:
    global _ADAPTER_POOL
    if _ADAPTER_POOL is None:
        _ADAPTER_POOL = ThreadPoolExecutor(max_workers=_ADAPTER_WORKERS, thread_name_prefix="obs-adapt")
    return _ADAPTER_POOL


class CrForgeObservationAdapter:
    """Split the pinned crforge v4 flat seat buffer into model-ready frame tensors."""

    def __init__(self, schema, vocabulary_size: int = 1466, entity_bucket: int = 0):
        self.schema = schema
        self.vocabulary_size = vocabulary_size
        self.entity_bucket = entity_bucket
        self._count_index = schema.header["entity_count"][0]
        self._arange_max = np.arange(schema.max_entities)

        card_fields = {
            "hand_card",
            "next_card",
            "deck",
            "enemy_seen",
            "pending_card",
            "ability_champion",
            "enemy_play_card",
        }
        header_field_divisors = {
            "elixir": 100000.0,
            "elixir_reserved": 100000.0,
            "enemy_elixir": 100000.0,
            "tick": 7200.0,
            "enemy_play_ticks": 7200.0,
            "enemy_play_point_ticks": 7200.0,
            "episode_step": 1440.0,
            "time_ms": 360000.0,
            "section_left_ms": 360000.0,
            "full_bar_ms": 360000.0,
            "hand_refill_ms": 360000.0,
            "ability_cooldown_ms": 360000.0,
            "pending_x": 18000.0,
            "enemy_play_x": 18000.0,
            "pending_y": 32000.0,
            "enemy_play_y": 32000.0,
            "own_king_hp": 10000.0,
            "own_king_max_hp": 10000.0,
            "own_left_hp": 10000.0,
            "own_left_max_hp": 10000.0,
            "own_right_hp": 10000.0,
            "own_right_max_hp": 10000.0,
            "enemy_king_hp": 10000.0,
            "enemy_king_max_hp": 10000.0,
            "enemy_left_hp": 10000.0,
            "enemy_left_max_hp": 10000.0,
            "enemy_right_hp": 10000.0,
            "enemy_right_max_hp": 10000.0,
            "hand_cost": 10.0,
            "deck_level": 15.0,
            "crowns": 3.0,
            "crowns_enemy": 3.0,
            "pending_ticks": 100.0,
            "play_delay_ticks": 100.0,
            "play_open_in_ticks": 100.0,
            "ability_command_ticks": 100.0,
        }
        header_divisors = np.ones(schema.header_ints, dtype=np.float32)
        card_indices = []
        for index, name in enumerate(schema.header_fields):
            if name in card_fields:
                card_indices.append(index)
            elif name in header_field_divisors:
                header_divisors[index] = np.float32(header_field_divisors[name])
        self._header_divisors = header_divisors
        self._card_indices = np.asarray(card_indices, dtype=np.int64)
        self._vocab_divisor = np.float32(vocabulary_size)

        entity_field_divisors = {
            "row": float(vocabulary_size),
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
        entity_divisors = np.ones(schema.entity_ints, dtype=np.float32)
        entity_zero_indices = []
        for index, name in enumerate(schema.entity_fields):
            if name in ("id", "target"):
                entity_zero_indices.append(index)
            elif name in entity_field_divisors:
                entity_divisors[index] = np.float32(entity_field_divisors[name])
        self._entity_divisors = entity_divisors
        self._entity_zero_indices = np.asarray(entity_zero_indices, dtype=np.int64)

    @staticmethod
    def _scale(values: np.ndarray, divisor: float) -> np.ndarray:
        return values / np.float32(divisor)

    def _fill_chunk(
        self,
        obs_2d: np.ndarray,
        cnt_1d: np.ndarray,
        hdr_2d: np.ndarray,
        ent_3d: np.ndarray,
        msk_2d: np.ndarray,
        limit: int,
    ) -> None:
        hdr_raw = obs_2d[:, : self.schema.header_ints]
        hdr_2d[:] = hdr_raw
        hdr_2d /= self._header_divisors
        if self._card_indices.size:
            cards = hdr_raw[:, self._card_indices]
            hdr_2d[:, self._card_indices] = np.where(
                cards >= 0, cards / self._vocab_divisor, np.float32(-1.0)
            )
        ent_view = obs_2d[:, self.schema.header_ints :].reshape(
            obs_2d.shape[0], self.schema.max_entities, self.schema.entity_ints
        )[:, :limit, :]
        ent_3d[:] = ent_view
        ent_3d /= self._entity_divisors
        if self._entity_zero_indices.size:
            ent_3d[:, :, self._entity_zero_indices] = 0.0
        np.less(self._arange_max[:limit], cnt_1d[:, None], out=msk_2d)
        msk_2d[:, 0] |= cnt_1d == 0

    def __call__(self, observations: np.ndarray):
        expected = self.schema.seat_ints
        if observations.shape[-1] != expected:
            raise ValueError(f"Expected {expected} ints per seat, got {observations.shape[-1]}")
        counts = np.clip(observations[..., self._count_index], 0, self.schema.max_entities)
        limit = self.schema.max_entities
        if self.entity_bucket and counts.size > 0:
            occupied = max(1, int(counts.max()))
            limit = min(
                limit,
                ((occupied + self.entity_bucket - 1) // self.entity_bucket) * self.entity_bucket,
            )

        lead_shape = observations.shape[:-1]
        flat_rows = int(np.prod(lead_shape)) if lead_shape else 1
        if flat_rows >= 16384 and _ADAPTER_WORKERS >= 2:
            obs_2d = np.ascontiguousarray(observations).reshape(flat_rows, expected)
            cnt_1d = counts.reshape(flat_rows)
            hdr_2d = np.empty((flat_rows, self.schema.header_ints), dtype=np.float32)
            ent_3d = np.empty((flat_rows, limit, self.schema.entity_ints), dtype=np.float32)
            msk_2d = np.empty((flat_rows, limit), dtype=bool)
            step = max(64, (flat_rows + _ADAPTER_WORKERS - 1) // _ADAPTER_WORKERS)
            pool = _get_adapter_pool()
            futs = [
                pool.submit(
                    self._fill_chunk,
                    obs_2d[s : s + step],
                    cnt_1d[s : s + step],
                    hdr_2d[s : s + step],
                    ent_3d[s : s + step],
                    msk_2d[s : s + step],
                    limit,
                )
                for s in range(0, flat_rows, step)
            ]
            for f in futs:
                f.result()
            return (
                hdr_2d.reshape(*lead_shape, self.schema.header_ints),
                ent_3d.reshape(*lead_shape, limit, self.schema.entity_ints),
                msk_2d.reshape(*lead_shape, limit),
            )

        header = observations[..., : self.schema.header_ints].astype(np.float32, copy=True)
        entities = (
            observations[..., self.schema.header_ints :]
            .reshape(
                *observations.shape[:-1],
                self.schema.max_entities,
                self.schema.entity_ints,
            )[..., :limit, :]
            .astype(np.float32, copy=True)
        )
        entity_mask = self._arange_max[:limit] < counts[..., None]
        entity_mask[..., 0] |= counts == 0

        # Unknown (-1) remains distinct. Field names come from the pinned simulator schema.
        header /= self._header_divisors
        if self._card_indices.size:
            cards = header[..., self._card_indices]
            header[..., self._card_indices] = np.where(
                cards >= 0, cards / self._vocab_divisor, np.float32(-1.0)
            )
        entities /= self._entity_divisors
        if self._entity_zero_indices.size:
            entities[..., self._entity_zero_indices] = 0.0
        return header, entities, entity_mask
