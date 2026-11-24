"""Bounded Actor learning-rate feedback from complete PPO optimizer groups.

This changes only the next rollout's Actor learning rate. The factorized KL
limits and their rejection/early-stop decisions remain owned by PPO itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdaptiveActorLRConfig:
    min_lr: float = 2.5e-6
    max_lr: float = 1e-5
    decrease_factor: float = 0.5
    increase_factor: float = 1.25
    low_ratio: float = 0.25
    high_ratio: float = 0.9
    low_updates_before_increase: int = 2
    min_groups: int = 2
    min_samples: int = 128

    def __post_init__(self):
        values = (
            self.min_lr, self.max_lr, self.decrease_factor, self.increase_factor,
            self.low_ratio, self.high_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Adaptive Actor LR settings must be finite")
        if not 0 < self.min_lr <= self.max_lr:
            raise ValueError("Adaptive Actor LR requires 0 < min_lr <= max_lr")
        if not 0 < self.decrease_factor < 1 < self.increase_factor:
            raise ValueError("Adaptive Actor LR factors must decrease/increase respectively")
        if not 0 <= self.low_ratio < self.high_ratio <= 1:
            raise ValueError("Adaptive Actor LR KL bands must satisfy 0 <= low < high <= 1")
        counts = (self.low_updates_before_increase, self.min_groups, self.min_samples)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in counts
        ):
            raise ValueError("Adaptive Actor LR evidence/streak counts must be positive integers")


class AdaptiveActorLearningRate:
    """Use the largest Card/Tile KL ratio, including rejected complete groups.

    A small evidence floor prevents a single optimizer group from controlling the
    rate. Increasing also requires consecutive low-KL updates with real committed
    steps; numerical-overflow updates never cause an increase. No optimizer
    parameter or state is read other than the shared Actor learning rate.
    """

    version = 1

    def __init__(self, config: AdaptiveActorLRConfig | None = None):
        self.config = config or AdaptiveActorLRConfig()
        self.current_lr = None
        self.completed_updates = 0
        self.low_kl_updates = 0
        self._active = False
        self._reset_observations()

    def _reset_observations(self):
        self.groups = self.samples = self.rejected_groups = self.rejected_samples = 0
        self.max_card_kl = self.max_tile_kl = self.max_ratio = 0.0

    @staticmethod
    def _optimizer_lr(optimizer):
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        if not rates or not all(math.isfinite(lr) and lr > 0 for lr in rates):
            raise ValueError("Adaptive Actor LR requires positive finite optimizer rates")
        if any(lr != rates[0] for lr in rates):
            raise ValueError("Adaptive Actor LR requires a common rate for all Actor groups")
        return rates[0]

    def begin_update(self, optimizer):
        if self._active:
            raise RuntimeError("Cannot restart an unfinished adaptive Actor LR update")
        lr = self._optimizer_lr(optimizer)
        if not self.config.min_lr <= lr <= self.config.max_lr:
            raise ValueError("Actor optimizer rate is outside the adaptive LR bounds")
        if self.current_lr is not None and not math.isclose(
            lr, self.current_lr, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("Actor optimizer and restored adaptive LR state disagree")
        self.current_lr = lr
        self._lr_used = lr
        self._reset_observations()
        self._active = True

    def observe(self, *, card_kl, tile_kl, target_card_kl, target_tile_kl, samples, rejected):
        if not self._active:
            raise RuntimeError("Begin the Actor LR update before recording KL groups")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
            raise ValueError("A KL observation must represent a nonempty complete group")
        values = tuple(float(x) for x in (card_kl, tile_kl, target_card_kl, target_tile_kl))
        if not all(math.isfinite(x) for x in values):
            raise FloatingPointError("Non-finite factorized KL observation")
        card, tile, card_limit, tile_limit = values
        if min(card_limit, tile_limit) <= 0:
            raise ValueError("Adaptive Actor LR requires positive factorized KL limits")
        # Small negative analytic KL values can occur from floating-point roundoff.
        card, tile = max(0.0, card), max(0.0, tile)
        self.groups += 1
        self.samples += samples
        self.rejected_groups += int(rejected)
        self.rejected_samples += samples * int(rejected)
        self.max_card_kl = max(self.max_card_kl, card)
        self.max_tile_kl = max(self.max_tile_kl, tile)
        self.max_ratio = max(self.max_ratio, card / card_limit, tile / tile_limit)

    def finish_update(self, optimizer, *, actor_enabled, committed_steps, overflow_steps=0):
        if not self._active:
            raise RuntimeError("No active Actor LR update")
        if self._optimizer_lr(optimizer) != self._lr_used:
            raise ValueError("Actor learning rate changed within an adaptive update")
        settings = self.config
        next_lr = self._lr_used
        enough = self.groups >= settings.min_groups and self.samples >= settings.min_samples
        if not actor_enabled or not self.groups:
            reason = "actor_disabled" if not actor_enabled else "no_groups"
            self.low_kl_updates = 0
        elif overflow_steps and overflow_steps > committed_steps // 4:
            reason = "amp_overflow"
            self.low_kl_updates = 0
        elif not enough:
            reason = "insufficient_evidence"
            self.low_kl_updates = 0
        elif self.rejected_groups or self.max_ratio >= settings.high_ratio:
            next_lr = max(settings.min_lr, next_lr * settings.decrease_factor)
            reason = "kl_rejected" if self.rejected_groups else "kl_near_limit"
            self.low_kl_updates = 0
        elif committed_steps >= settings.min_groups and self.max_ratio <= settings.low_ratio:
            self.low_kl_updates += 1
            reason = "low_kl_wait"
            if self.low_kl_updates >= settings.low_updates_before_increase:
                next_lr = min(settings.max_lr, next_lr * settings.increase_factor)
                self.low_kl_updates = 0
                reason = "low_kl_increase" if next_lr > self._lr_used else "at_ceiling"
        else:
            reason = "within_band"
            self.low_kl_updates = 0
        for group in optimizer.param_groups:
            group["lr"] = next_lr
        self.current_lr = next_lr
        self.completed_updates += 1
        self._active = False
        return {
            "enabled": True,
            "scope": "next_update",
            "lr_used": self._lr_used,
            "next_lr": next_lr,
            "reason": reason,
            "groups_observed": self.groups,
            "samples_observed": self.samples,
            "rejected_groups": self.rejected_groups,
            "rejected_samples": self.rejected_samples,
            "max_card_kl": self.max_card_kl,
            "max_tile_kl": self.max_tile_kl,
            "max_normalized_kl": self.max_ratio,
            "committed_steps": committed_steps,
            "overflow_steps": overflow_steps,
            "low_kl_updates": self.low_kl_updates,
            "completed_updates": self.completed_updates,
        }

    def state_dict(self):
        if self._active:
            raise RuntimeError("Cannot checkpoint an unfinished adaptive Actor LR update")
        return {
            "version": self.version,
            "config": asdict(self.config),
            "current_lr": self.current_lr,
            "completed_updates": self.completed_updates,
            "low_kl_updates": self.low_kl_updates,
        }

    def load_state_dict(self, state):
        if self._active:
            raise RuntimeError("Cannot restore over an unfinished adaptive Actor LR update")
        if state.get("version") != self.version:
            raise ValueError("Unsupported adaptive Actor LR state version")
        config = AdaptiveActorLRConfig(**state["config"])
        if config != self.config:
            raise ValueError("Restored adaptive Actor LR settings differ from the requested config")
        lr = state["current_lr"]
        if lr is not None and not (
            math.isfinite(lr) and config.min_lr <= lr <= config.max_lr
        ):
            raise ValueError("Invalid restored adaptive Actor learning rate")
        completed, streak = state["completed_updates"], state["low_kl_updates"]
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in (completed, streak)):
            raise ValueError("Invalid restored adaptive Actor LR counters")
        if streak >= config.low_updates_before_increase:
            raise ValueError("Invalid restored low-KL streak")
        self.current_lr = lr
        self.completed_updates = completed
        self.low_kl_updates = streak
        self._reset_observations()


def restore_actor_lr_controller(
    optimizer,
    saved_state=None,
    *,
    enabled=False,
    min_lr=2.5e-6,
    max_lr=None,
    allow_bound_override=False,
):
    """Resume saved feedback automatically; otherwise require explicit opt-in.

    A lowered current rate is never interpreted as a new ceiling on resume.
    Explicit incompatible settings or optimizer LR overrides fail before use
    unless `allow_bound_override` explicitly re-anchors bounds on resume.
    """
    if saved_state is None and not enabled:
        return None
    lr = AdaptiveActorLearningRate._optimizer_lr(optimizer)
    effective_state = saved_state
    adjusted = False
    if saved_state is not None:
        saved_cfg = dict(saved_state["config"])
        config = AdaptiveActorLRConfig(**saved_cfg)
        bounds_differ = min_lr != config.min_lr or (max_lr is not None and max_lr != config.max_lr)
        if enabled and bounds_differ:
            if not allow_bound_override:
                raise ValueError("Requested adaptive Actor LR bounds differ from the checkpoint")
            saved_cfg["min_lr"] = float(min_lr)
            if max_lr is not None:
                saved_cfg["max_lr"] = float(max_lr)
            config = AdaptiveActorLRConfig(**saved_cfg)
            adjusted = True
        if allow_bound_override:
            clamped_lr = min(config.max_lr, max(config.min_lr, lr))
            if not math.isclose(clamped_lr, lr, rel_tol=1e-12, abs_tol=1e-15):
                for group in optimizer.param_groups:
                    group["lr"] = clamped_lr
                lr = clamped_lr
                adjusted = True
            saved_lr = saved_state.get("current_lr")
            if saved_lr is None or not math.isclose(lr, saved_lr, rel_tol=1e-12, abs_tol=1e-15):
                adjusted = True
            if adjusted:
                effective_state = {
                    **saved_state,
                    "config": asdict(config),
                    "current_lr": lr,
                    "low_kl_updates": 0,
                }
    else:
        config = AdaptiveActorLRConfig(min_lr=min_lr, max_lr=lr if max_lr is None else max_lr)
    controller = AdaptiveActorLearningRate(config)
    if effective_state is not None:
        controller.load_state_dict(effective_state)
    if not config.min_lr <= lr <= config.max_lr:
        raise ValueError("Actor optimizer rate is outside the adaptive LR bounds")
    if controller.current_lr is not None and not math.isclose(
        lr, controller.current_lr, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("Actor optimizer and restored adaptive LR state disagree")
    controller.current_lr = lr
    return controller
