"""Human-replay prefix warmup + live mid-game RL takeover pool.

Loads dual-seat human match trajectories from a frozen ``crrl-bc-shards-v1``
snapshot (such as ``sparky_plan_b`` or future live-play recordings), extracts
exact opening deck/slot/hand-cycle configurations from ``tick=0``, and samples
mid-game takeover plans (mixing counter-attack trigger points right after an
opponent card deployment with uniform mid-game cutpoints).

During rollout, a lane assigned a ``ReplayStartPlan`` replays the exact human
actions for both seats from ``step=0 .. cut_step-1`` while advancing both the
learner and opponent recurrent KDA/GQA memories. Prefix frames are excluded
from PPO actor and critic losses (``policy_loss_mask=False``,
``critic_loss_mask=False``), and at ``cut_step`` both seats hand over cleanly
to live policies (learner vs league opponent pool) through match termination.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


PRINCESS_TOWER_GROUP = "King_PrincessTowers"


@dataclass(frozen=True)
class ReplayEpisode:
    """Compact in-memory representation of one dual-seat human replay."""

    shard_index: int
    identity: str
    cohort: str
    deck0: tuple[str, ...]
    deck1: tuple[str, ...]
    slots0: tuple[int, ...]
    slots1: tuple[int, ...]
    hand_order0: tuple[int, ...]
    hand_order1: tuple[int, ...]
    towers: tuple[str, str]
    actions0: np.ndarray  # int32 [T]
    actions1: np.ndarray  # int32 [T]
    preferred_seats: tuple[int, ...]
    counter_cutpoints_seat0: np.ndarray  # int32 sorted valid cut steps for seat 0
    counter_cutpoints_seat1: np.ndarray  # int32 sorted valid cut steps for seat 1


@dataclass(frozen=True)
class ReplayStartPlan:
    """Per-lane concrete mid-game takeover schedule for one episode."""

    shard_index: int
    identity: str
    cohort: str
    learner_seat: int
    cut_step: int
    is_counter_cut: bool
    deck0: list[str]
    deck1: list[str]
    slots0: list[int]
    slots1: list[int]
    hand_order0: list[int]
    hand_order1: list[int]
    towers: tuple[str, str]
    actions0: np.ndarray  # int32 [cut_step]
    actions1: np.ndarray  # int32 [cut_step]
    is_late_cut: bool = False


def _extract_opening_order(
    obs0: np.ndarray,
    header: dict[str, Sequence[int]],
    vocab: Sequence[str],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Decode deck names, evo/hero slot flags, and exact [hand0..3, queue0..3] deck indices."""
    deck_idx = list(header["deck"])
    slot_idx = list(header["deck_slot"])
    hand_idx = list(header["hand_card"])
    queue_idx = list(header["queue"])
    d_ids = [int(x) for x in obs0[deck_idx]]
    if len(d_ids) != 8 or any(cid < 0 or cid >= len(vocab) for cid in d_ids):
        return None
    if len(set(d_ids)) != 8:
        return None
    hq_ids = [int(x) for x in obs0[hand_idx]] + [int(x) for x in obs0[queue_idx]]
    id_to_pos = {cid: pos for pos, cid in enumerate(d_ids)}
    if len(hq_ids) != 8 or any(cid not in id_to_pos for cid in hq_ids):
        return None
    order = tuple(id_to_pos[cid] for cid in hq_ids)
    if len(set(order)) != 8:
        return None
    deck_names = tuple(str(vocab[cid]) for cid in d_ids)
    slot_flags = tuple(int(x) for x in obs0[slot_idx])
    return deck_names, slot_flags, order


def _find_counter_cutpoints(
    opp_actions: np.ndarray,
    *,
    min_step: int,
    max_step: int,
    reaction_lags: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """Steps right after the opponent deploys a card, ideal for counter-play training."""
    if max_step < min_step or len(opp_actions) == 0:
        return np.zeros(0, dtype=np.int32)
    plays = np.flatnonzero(opp_actions > 0)
    if len(plays) == 0:
        return np.zeros(0, dtype=np.int32)
    cuts: set[int] = set()
    for p in plays:
        for lag in reaction_lags:
            t = int(p) + int(lag)
            if min_step <= t <= max_step:
                cuts.add(t)
    if not cuts:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(cuts), dtype=np.int32)


class ReplayStartPool:
    """In-memory index of dual-seat human replays for mid-game RL takeover."""

    def __init__(
        self,
        dataset: str | Path,
        *,
        schema=None,
        min_steps: int = 35,
        max_steps: int = 560,
        min_tail_steps: int = 30,
        counter_prob: float = 0.60,
        sparky_prob: float = 0.60,
        late_game_prob: float = 0.0,
        late_min_steps: int = 260,
        train_only: bool = True,
    ) -> None:
        self.root = Path(dataset).resolve()
        if self.root.is_file():
            self.root = self.root.parent
        self.min_steps = max(8, int(min_steps))
        self.max_steps = max(self.min_steps, int(max_steps))
        self.min_tail_steps = max(8, int(min_tail_steps))
        self.counter_prob = float(np.clip(counter_prob, 0.0, 1.0))
        self.sparky_prob = float(np.clip(sparky_prob, 0.0, 1.0))
        self.late_game_prob = float(np.clip(late_game_prob, 0.0, 1.0))
        self.late_min_steps = max(self.min_steps, int(late_min_steps))

        manifest = json.loads((self.root / "dataset.manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((self.root / manifest["metadata"]).read_text(encoding="utf-8"))
        splits_path = self.root / "split_registry.json"
        assignments = (
            json.loads(splits_path.read_text(encoding="utf-8")).get("assignments", {})
            if splits_path.is_file()
            else {}
        )
        if metadata.get("decision_ticks") != 8:
            raise ValueError("ReplayStartPool requires native 8-tick replay history")
        header = getattr(schema, "header", None) if schema is not None else None
        if header is None:
            schema_dict = metadata["schema"]
            header = schema_dict.get("header")
        if header is None:
            raise ValueError("ReplayStartPool requires schema.header field mapping (pass schema=env.schema)")
        vocab_raw = metadata["vocabulary"]
        vocab = vocab_raw["names"] if isinstance(vocab_raw, dict) else vocab_raw

        self.snapshot_id = str(manifest.get("snapshot_id", "unknown"))
        self.episodes: list[ReplayEpisode] = []
        self.by_cohort: dict[str, list[ReplayEpisode]] = {}
        self.late_episodes: list[ReplayEpisode] = []
        self.late_by_cohort: dict[str, list[ReplayEpisode]] = {}
        skipped = 0

        for shard_idx, shard in enumerate(manifest["shards"]):
            record = metadata["replays"][shard["replay_offset"]]
            identity = str(record.get("canonical_game_signature", f"shard_{shard_idx}"))
            if train_only and assignments:
                if assignments.get(identity) != "train":
                    continue
            path = (self.root / shard["path"]).resolve()
            if not path.is_file():
                skipped += 1
                continue
            with np.load(path, allow_pickle=False) as arrays:
                obs_all = arrays["observations"]
                acts_all = arrays["actions"]
                ticks_all = arrays["ticks"]
                seats_all = arrays["seats"]
            seat_rows = {}
            valid_game = True
            for s in (0, 1):
                rows = np.flatnonzero(seats_all == s)
                if len(rows) == 0:
                    valid_game = False
                    break
                rows = rows[np.argsort(ticks_all[rows], kind="stable")]
                t_arr = ticks_all[rows]
                if t_arr[0] != 0 or (len(t_arr) > 1 and not np.all(np.diff(t_arr) == 8)):
                    valid_game = False
                    break
                seat_rows[s] = rows
            if not valid_game:
                skipped += 1
                continue
            t_len = min(len(seat_rows[0]), len(seat_rows[1]))
            max_cut = min(self.max_steps, t_len - self.min_tail_steps)
            if max_cut < self.min_steps:
                skipped += 1
                continue

            open0 = _extract_opening_order(obs_all[seat_rows[0][0]], header, vocab)
            open1 = _extract_opening_order(obs_all[seat_rows[1][0]], header, vocab)
            if open0 is None or open1 is None:
                skipped += 1
                continue

            acts0 = acts_all[seat_rows[0][:t_len]].astype(np.int32, copy=True)
            acts1 = acts_all[seat_rows[1][:t_len]].astype(np.int32, copy=True)
            # Require at least some real card plays before max_cut so prefixes are non-trivial.
            if int((acts0[:max_cut] > 0).sum() + (acts1[:max_cut] > 0).sum()) < 2:
                skipped += 1
                continue

            cohort = str(shard.get("cohort") or "general")
            pref_raw = shard.get("selected_seats", [0, 1])
            pref = tuple(int(s) for s in pref_raw if int(s) in (0, 1)) or (0, 1)
            c0 = _find_counter_cutpoints(acts1, min_step=self.min_steps, max_step=max_cut)
            c1 = _find_counter_cutpoints(acts0, min_step=self.min_steps, max_step=max_cut)
            ep = ReplayEpisode(
                shard_index=shard_idx,
                identity=identity,
                cohort=cohort,
                deck0=open0[0],
                deck1=open1[0],
                slots0=open0[1],
                slots1=open1[1],
                hand_order0=open0[2],
                hand_order1=open1[2],
                towers=(PRINCESS_TOWER_GROUP, PRINCESS_TOWER_GROUP),
                actions0=acts0,
                actions1=acts1,
                preferred_seats=pref,
                counter_cutpoints_seat0=c0,
                counter_cutpoints_seat1=c1,
            )
            self.episodes.append(ep)
            self.by_cohort.setdefault(cohort, []).append(ep)
            if max_cut >= self.late_min_steps:
                self.late_episodes.append(ep)
                self.late_by_cohort.setdefault(cohort, []).append(ep)

        self.skipped_shards = skipped
        if not self.episodes:
            raise ValueError(f"ReplayStartPool found no usable dual-seat replays in {self.root}")

    @property
    def size(self) -> int:
        return len(self.episodes)

    def summary(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "episodes": len(self.episodes),
            "late_episodes": len(self.late_episodes),
            "cohorts": {k: len(v) for k, v in sorted(self.by_cohort.items())},
            "skipped_shards": self.skipped_shards,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "counter_prob": self.counter_prob,
            "sparky_prob": self.sparky_prob,
            "late_game_prob": self.late_game_prob,
            "late_min_steps": self.late_min_steps,
        }

    def sample(self, rng: np.random.Generator) -> ReplayStartPlan:
        """Sample a human replay episode, seat, and mid/late-game takeover cutpoint."""
        want_late = bool(self.late_episodes and float(rng.random()) < self.late_game_prob)
        cohort_map = self.late_by_cohort if want_late else self.by_cohort
        fallback_pool = self.late_episodes if want_late else self.episodes

        sparky_pool = cohort_map.get("sparky")
        general_pool = cohort_map.get("general")
        if sparky_pool and general_pool:
            pool = sparky_pool if float(rng.random()) < self.sparky_prob else general_pool
        else:
            pool = fallback_pool
        ep = pool[int(rng.integers(len(pool)))]

        # On specialist/Sparky replays, put learner on the specialist seat 75% of the time
        # and on the opposing seat 25% of the time so it learns both piloting & countering.
        if len(ep.preferred_seats) == 1:
            pref_seat = ep.preferred_seats[0]
            learner_seat = pref_seat if float(rng.random()) < 0.75 else (1 - pref_seat)
        else:
            learner_seat = int(rng.integers(0, 2))

        max_cut = min(self.max_steps, len(ep.actions0) - self.min_tail_steps)
        eff_min = self.late_min_steps if (want_late and max_cut >= self.late_min_steps) else self.min_steps
        counter_cuts = (
            ep.counter_cutpoints_seat0 if learner_seat == 0 else ep.counter_cutpoints_seat1
        )
        if eff_min > self.min_steps and len(counter_cuts) > 0:
            counter_cuts = counter_cuts[counter_cuts >= eff_min]
        is_counter = False
        if len(counter_cuts) > 0 and float(rng.random()) < self.counter_prob:
            cut_step = int(counter_cuts[int(rng.integers(len(counter_cuts)))])
            is_counter = True
        else:
            cut_step = int(rng.integers(eff_min, max_cut + 1))

        return ReplayStartPlan(
            shard_index=ep.shard_index,
            identity=ep.identity,
            cohort=ep.cohort,
            learner_seat=learner_seat,
            cut_step=cut_step,
            is_counter_cut=is_counter,
            is_late_cut=bool(cut_step >= self.late_min_steps),
            deck0=list(ep.deck0),
            deck1=list(ep.deck1),
            slots0=list(ep.slots0),
            slots1=list(ep.slots1),
            hand_order0=list(ep.hand_order0),
            hand_order1=list(ep.hand_order1),
            towers=ep.towers,
            actions0=ep.actions0[:cut_step],
            actions1=ep.actions1[:cut_step],
        )
