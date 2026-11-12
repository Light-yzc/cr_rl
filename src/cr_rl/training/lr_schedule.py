"""Warmup/cosine learning rates indexed only by successful optimizer updates."""
from __future__ import annotations

import math
from copy import deepcopy

FORMAT = "crrl-state-bc-warmup-cosine-v1"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def remaining_episode_updates(total_epochs, progress, train_seats):
    """The successful-update horizon planned from the remaining episode seats.

    Overflow consumes a seat without advancing this schedule. Consequently a
    run with skipped updates can finish the epoch budget before the LR floor.
    """
    _integer(total_epochs, "total_epochs", 1)
    _integer(train_seats, "train_seats", 1)
    epoch = _integer(progress["epoch"], "epoch")
    cursor = _integer(progress["next_episode"], "next_episode")
    if cursor > train_seats:
        raise ValueError("next_episode exceeds the fixed training seat count")
    remaining = total_epochs * train_seats - epoch * train_seats - cursor
    if remaining <= 0:
        raise ValueError("No remaining episode updates for the requested schedule")
    return remaining


def cosine_config(*, peak_lr, min_lr, warmup_steps, total_steps, anchor_step,
                  total_epochs, start_progress, train_seats, transition=None):
    config = {
        "format": FORMAT, "kind": "cosine", "peak_lr": float(peak_lr),
        "min_lr": float(min_lr), "warmup_start_factor": 0.1,
        "warmup_steps": warmup_steps, "total_steps": total_steps,
        "anchor_step": anchor_step, "total_epochs": total_epochs,
        "start_progress": {"epoch": start_progress["epoch"], "next_episode": start_progress["next_episode"]},
        "train_seats": train_seats,
    }
    if transition is not None:
        config["transition"] = deepcopy(transition)
    validate_config(config)
    return config


def validate_config(config):
    if config.get("format") != FORMAT or config.get("kind") != "cosine":
        raise ValueError("Unsupported state BC learning-rate schedule")
    for key in ("anchor_step", "warmup_steps"):
        _integer(config[key], key)
    for key in ("total_steps", "total_epochs", "train_seats"):
        _integer(config[key], key, 1)
    if config["warmup_steps"] >= config["total_steps"]:
        raise ValueError("Cosine total_steps must exceed warmup_steps")
    if not (math.isfinite(config["peak_lr"]) and config["peak_lr"] > 0
            and math.isfinite(config["min_lr"]) and 0 <= config["min_lr"] <= config["peak_lr"]
            and math.isfinite(config["warmup_start_factor"]) and 0 < config["warmup_start_factor"] <= 1):
        raise ValueError("Invalid cosine learning-rate bounds")
    remaining_episode_updates(config["total_epochs"], config["start_progress"], config["train_seats"])


def learning_rate_at(config, successful_updates):
    """LR for the next optimizer attempt after this many successful updates."""
    _integer(successful_updates, "successful_updates")
    warmup = config["warmup_steps"]
    if warmup and successful_updates < warmup:
        fraction = successful_updates / warmup
        return config["peak_lr"] * (config["warmup_start_factor"]
                                     + (1 - config["warmup_start_factor"]) * fraction)
    fraction = min(1.0, max(0.0, (successful_updates - warmup) / (config["total_steps"] - warmup)))
    return config["min_lr"] + 0.5 * (config["peak_lr"] - config["min_lr"]) * (1 + math.cos(math.pi * fraction))


class WarmupCosineLR:
    """Explicit state independent of Torch scheduler/GradScaler call ordering."""

    def __init__(self, config, successful_updates=0):
        validate_config(config)
        self.config = deepcopy(config)
        self.successful_updates = _integer(successful_updates, "successful_updates")

    @property
    def learning_rate(self):
        return learning_rate_at(self.config, self.successful_updates)

    def apply(self, optimizer):
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rate

    def after_attempt(self, optimizer, *, successful):
        if not isinstance(successful, bool):
            raise ValueError("successful must be a boolean")
        if successful:
            self.successful_updates += 1
        self.apply(optimizer)

    def state_dict(self):
        return {"format": FORMAT, "config": deepcopy(self.config),
                "successful_updates": self.successful_updates,
                "optimizer_step": self.config["anchor_step"] + self.successful_updates,
                "next_learning_rate": self.learning_rate}

    @classmethod
    def from_state_dict(cls, state, *, expected_config, optimizer_step, optimizer=None):
        if state.get("format") != FORMAT or state.get("config") != expected_config:
            raise ValueError("Checkpoint learning-rate schedule definition differs")
        schedule = cls(expected_config, state["successful_updates"])
        if state.get("optimizer_step") != optimizer_step or schedule.config["anchor_step"] + schedule.successful_updates != optimizer_step:
            raise ValueError("Schedule position differs from successful optimizer steps")
        if not math.isclose(state.get("next_learning_rate", math.nan), schedule.learning_rate, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Checkpoint learning-rate schedule position is inconsistent")
        if optimizer is not None and any(not math.isclose(float(group["lr"]), schedule.learning_rate,
                                                         rel_tol=1e-12, abs_tol=0)
                                         for group in optimizer.param_groups):
            raise ValueError("Optimizer LR differs from the saved schedule position")
        return schedule
