from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TrajectoryBuffer:
    """Versioned two-seat rollout storage suitable for symmetric self-play.

    Arrays keep the [time, environment, seat] axes explicit. ``learner_mask`` identifies
    which seat(s) produced on-policy samples for the learner; opponent actions remain in
    the same match record for replay, audits and future off-policy critic work.
    """

    def __init__(self, metadata: dict):
        self.metadata = {"trajectory_schema_version": 1, **metadata}
        self.rows: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "obs",
                "masks",
                "actions",
                "rewards",
                "training_rewards",
                "next_obs",
                "next_masks",
                "dones",
                "terminated",
                "truncated",
                "valid",
                "action_consumed",
                "action_refusals",
                "run_refusals",
                "old_log_probs",
                "old_values",
                "bootstrap_values",
                "bootstrap_valid",
                "learner_mask",
                "policy_ids",
                "opponent_ids",
                "policy_versions",
                "episode_ids",
                "episode_start",
                "episode_steps",
                "ticks",
                "winner",
                "end_reason",
                "fault_codes",
                "seeds",
            )
        }

    def append(
        self,
        obs,
        masks,
        actions,
        rewards,
        next_obs,
        next_masks,
        dones,
        *,
        terminated=None,
        truncated=None,
        valid=None,
        training_rewards=None,
        action_consumed=None,
        action_refusals=None,
        run_refusals=None,
        old_log_probs=None,
        old_values=None,
        bootstrap_values=None,
        bootstrap_valid=None,
        learner_mask=None,
        policy_ids=None,
        opponent_ids=None,
        policy_versions=None,
        episode_ids=None,
        episode_start=None,
        episode_steps=None,
        ticks=None,
        winner=None,
        end_reason=None,
        fault_codes=None,
        seeds=None,
    ) -> None:
        obs = np.asarray(obs)
        shape = np.asarray(actions).shape
        defaults = {
            "training_rewards": np.asarray(rewards).copy(),
            "terminated": np.zeros(np.asarray(dones).shape, dtype=np.uint8),
            "truncated": np.zeros(np.asarray(dones).shape, dtype=np.uint8),
            "valid": np.zeros(shape, dtype=np.uint8),
            "action_consumed": np.zeros(shape, dtype=np.uint8),
            "action_refusals": np.full(shape, -1, dtype=np.int32),
            "run_refusals": np.full(shape, -1, dtype=np.int32),
            "old_log_probs": np.full(shape, np.nan, dtype=np.float32),
            "old_values": np.full(shape, np.nan, dtype=np.float32),
            "bootstrap_values": np.full(shape, np.nan, dtype=np.float32),
            "bootstrap_valid": np.zeros(shape, dtype=np.uint8),
            "learner_mask": np.zeros(shape, dtype=np.uint8),
            "policy_ids": np.full(shape, -1, dtype=np.int32),
            "opponent_ids": np.full(shape, -1, dtype=np.int32),
            "policy_versions": np.full(shape, -1, dtype=np.int32),
            "episode_ids": np.full(shape[:-1] if len(shape) > 1 else shape, -1, dtype=np.int64),
            "episode_start": np.zeros(shape[:-1] if len(shape) > 1 else shape, dtype=np.uint8),
            "episode_steps": np.zeros(shape[:-1] if len(shape) > 1 else shape, dtype=np.int32),
            "ticks": np.zeros(shape[:-1] if len(shape) > 1 else shape, dtype=np.int32),
            "winner": np.full(shape[:-1] if len(shape) > 1 else shape, -1, dtype=np.int8),
            "end_reason": np.zeros(shape[:-1] if len(shape) > 1 else shape, dtype=np.uint8),
            "fault_codes": np.zeros(shape[:-1] if len(shape) > 1 else shape, dtype=np.int16),
            "seeds": np.full(shape[:-1] if len(shape) > 1 else shape, -1, dtype=np.int64),
        }
        supplied = {
            "obs": obs,
            "masks": masks,
            "actions": actions,
            "rewards": rewards,
            "training_rewards": training_rewards,
            "next_obs": next_obs,
            "next_masks": next_masks,
            "dones": dones,
            "terminated": terminated,
            "truncated": truncated,
            "valid": valid,
            "action_consumed": action_consumed,
            "action_refusals": action_refusals,
            "run_refusals": run_refusals,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "bootstrap_values": bootstrap_values,
            "bootstrap_valid": bootstrap_valid,
            "learner_mask": learner_mask,
            "policy_ids": policy_ids,
            "opponent_ids": opponent_ids,
            "policy_versions": policy_versions,
            "episode_ids": episode_ids,
            "episode_start": episode_start,
            "episode_steps": episode_steps,
            "ticks": ticks,
            "winner": winner,
            "end_reason": end_reason,
            "fault_codes": fault_codes,
            "seeds": seeds,
        }
        values = {
            name: defaults[name] if value is None and name in defaults else value
            for name, value in supplied.items()
        }
        required = (
            terminated,
            truncated,
            valid,
            action_consumed,
            action_refusals,
            run_refusals,
            old_log_probs,
            old_values,
            bootstrap_values,
            bootstrap_valid,
            learner_mask,
            policy_ids,
            opponent_ids,
            policy_versions,
            episode_ids,
            episode_start,
            episode_steps,
            ticks,
            winner,
            end_reason,
            fault_codes,
            seeds,
        )
        complete = all(value is not None for value in required)
        if complete:
            self._validate_v3_transition(values)
        previous_complete = self.metadata.get("provenance_complete")
        self.metadata["provenance_complete"] = bool(
            complete if previous_complete is None else previous_complete and complete
        )
        if self.metadata["provenance_complete"]:
            self.metadata["trajectory_schema_version"] = max(
                self.metadata["trajectory_schema_version"],
                4 if np.any(np.asarray(learner_mask).sum(-1) == 2) else 3,
            )
        else:
            self.metadata["trajectory_schema_version"] = 1
        for name, value in zip(self.rows, values.values(), strict=True):
            self.rows[name].append(np.asarray(value).copy())

    @staticmethod
    def _validate_v3_transition(values: dict[str, np.ndarray]) -> None:
        actions = np.asarray(values["actions"])
        seat_shape = actions.shape
        if len(seat_shape) != 2 or seat_shape[-1] != 2:
            raise ValueError("trajectory v3 actions must have [environment, 2 seats] shape")
        for name in (
            "rewards",
            "training_rewards",
            "old_log_probs",
            "old_values",
            "valid",
            "action_consumed",
            "action_refusals",
            "run_refusals",
            "bootstrap_values",
            "bootstrap_valid",
            "learner_mask",
            "policy_ids",
            "opponent_ids",
            "policy_versions",
        ):
            if np.asarray(values[name]).shape != seat_shape:
                raise ValueError(f"trajectory v3 {name} must match actions shape {seat_shape}")
        env_shape = (seat_shape[0],)
        for name in (
            "dones",
            "terminated",
            "truncated",
            "episode_ids",
            "episode_start",
            "episode_steps",
            "ticks",
            "winner",
            "end_reason",
            "fault_codes",
            "seeds",
        ):
            if np.asarray(values[name]).shape != env_shape:
                raise ValueError(f"trajectory v3 {name} must have shape {env_shape}")
        if np.any((values["terminated"] != 0) & (values["truncated"] != 0)):
            raise ValueError("a transition cannot be both terminated and truncated")
        if not np.array_equal(
            np.asarray(values["dones"], dtype=bool),
            np.asarray(values["terminated"], dtype=bool)
            | np.asarray(values["truncated"], dtype=bool),
        ):
            raise ValueError("dones must equal terminated OR truncated")
        dones = np.asarray(values["dones"], dtype=bool)
        reasons = np.asarray(values["end_reason"])
        expected_reasons = np.where(
            ~dones,
            0,
            np.where(
                np.asarray(values["fault_codes"]) > 0,
                3,
                np.where(np.asarray(values["truncated"], dtype=bool), 2, 1),
            ),
        )
        if not np.array_equal(reasons, expected_reasons):
            raise ValueError("end_reason disagrees with done/truncation/fault fields")
        if np.any(
            (np.asarray(values["winner"]) >= 0) & ~np.asarray(values["terminated"], dtype=bool)
        ):
            raise ValueError("winner is known only for a completed terminal match")
        learner_count = np.asarray(values["learner_mask"]).sum(axis=-1)
        if not np.all((learner_count == 1) | (learner_count == 2)):
            raise ValueError("self-play requires one or two on-policy learner seats per match")
        both = learner_count == 2
        for name in ("policy_ids", "policy_versions"):
            versions = np.asarray(values[name])
            if np.any(both & (versions[:, 0] != versions[:, 1])):
                raise ValueError("Two learner seats require identical behavior policy versions")
        if np.any(np.asarray(values["valid"]) > 1) or np.any(
            np.asarray(values["action_consumed"]) > 1
        ):
            raise ValueError("valid and action_consumed must be binary masks")
        for name in (
            "policy_ids",
            "opponent_ids",
            "policy_versions",
            "episode_ids",
            "episode_steps",
            "ticks",
            "fault_codes",
            "seeds",
        ):
            if np.any(np.asarray(values[name]) < 0):
                raise ValueError(f"trajectory v3 {name} must be non-negative")
        if np.any(np.asarray(values["episode_start"]) > 1):
            raise ValueError("episode_start must be a binary mask")
        if np.any((np.asarray(values["winner"]) < -1) | (np.asarray(values["winner"]) > 2)):
            raise ValueError("winner must be -1 (unknown), 0, 1 or 2 (draw)")
        if np.any(np.asarray(values["end_reason"]) > 3):
            raise ValueError("end_reason must be 0=ongoing, 1=terminal, 2=time-limit, 3=fault")
        masks = np.asarray(values["masks"])
        if masks.shape[:-1] != seat_shape:
            raise ValueError("trajectory v3 masks must be [environment, 2 seats, actions]")
        if np.any(actions < 0) or np.any(actions >= masks.shape[-1]):
            raise ValueError("trajectory v3 action index is outside the action codec")
        selected_legal = np.take_along_axis(masks, actions[..., None], axis=-1)[..., 0] > 0
        consumed = np.asarray(values["action_consumed"], dtype=bool)
        if np.any(consumed & ~selected_legal):
            raise ValueError("a consumed action was not legal under its stored sampling mask")
        valid = np.asarray(values["valid"], dtype=bool)
        if not np.isfinite(np.asarray(values["old_log_probs"])[valid]).all():
            raise ValueError("valid trajectory rows require finite behaviour log probabilities")
        if not np.isfinite(np.asarray(values["old_values"])[valid]).all():
            raise ValueError("valid trajectory rows require finite behaviour values")
        bootstrap_valid = np.asarray(values["bootstrap_valid"], dtype=bool)
        if np.any(bootstrap_valid & ~valid):
            raise ValueError("invalid rows cannot carry valid bootstrap targets")
        if not np.isfinite(np.asarray(values["bootstrap_values"])[bootstrap_valid]).all():
            raise ValueError("valid bootstrap entries must be finite")

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.stack(values) for name, values in self.rows.items()}
        np.savez_compressed(destination, **arrays)
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    def invalidate_episode(self, env_index: int, start_step: int, end_step: int) -> None:
        """Invalidate already-buffered rows for exactly one environment episode."""
        if start_step < 0 or end_step < start_step or end_step >= len(self.rows["valid"]):
            raise IndexError("episode step range is outside the buffered rollout")
        for step in range(start_step, end_step + 1):
            if env_index < 0 or env_index >= self.rows["valid"][step].shape[0]:
                raise IndexError("environment index is outside the buffered rollout")
            self.rows["valid"][step][env_index, :] = 0
            self.rows["bootstrap_valid"][step][env_index, :] = 0
