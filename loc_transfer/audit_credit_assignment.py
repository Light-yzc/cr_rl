"""Read-only GAE trace/coverage audit; never loads model weights or trains a policy.

Actor targets are reconstructed on the original physical-time grid, then
summarized on choice rows. Complete MC comparisons exclude incomplete, faulty,
truncated and mixed-behavior episodes. They diagnose logged targets, not strength.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cr_rl.opponents.role_sampling import FROZEN_VERSION_BASE
from cr_rl.training.advantages import compute_actor_critic_targets
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.sample_selection import normalize_actor_advantages
from cr_rl.training.telemetry import rollout_value_diagnostics

SMALL_FIELDS = (
    "training_rewards",
    "old_values",
    "bootstrap_values",
    "bootstrap_valid",
    "learner_mask",
    "valid",
    "terminated",
    "dones",
    "truncated",
    "fault_codes",
    "episode_ids",
    "episode_start",
    "episode_steps",
    "policy_versions",
)


def trace_decay(gamma: float, lam: float, step_seconds: float) -> dict:
    if not (0 < gamma <= 1 and 0 <= lam <= 1 and step_seconds > 0):
        raise ValueError("Discount, lambda and step duration are outside their ranges")
    decay = gamma * lam
    return dict(
        gamma_lambda=decay,
        e_fold_seconds=(-step_seconds / math.log(decay) if 0 < decay < 1 else None),
        half_life_seconds=(
            -step_seconds * math.log(2) / math.log(decay) if 0 < decay < 1 else None
        ),
        residual_trace_weights={str(t): decay ** (t / step_seconds) for t in (10, 30, 60, 120)},
        note="Exponential trace weight, not a hard credit horizon; value bootstrapping remains",
    )


def complete_episode_mask(data: dict) -> tuple[np.ndarray, dict]:
    """Select complete, contiguous, constant-behavior learner episode-seats."""
    mask = np.zeros_like(data["learner_mask"], dtype=bool)
    excluded: dict[str, int] = {}
    episodes = []
    games = set()

    def reject(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for lane in range(data["episode_ids"].shape[1]):
        ids = data["episode_ids"][:, lane]
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not data["episode_start"][left, lane]:
                reject("missing_prefix")
                continue
            if not data["terminated"][right - 1, lane]:
                reject("unfinished_or_nonterminal")
                continue
            if (
                data["fault_codes"][left:right, lane].any()
                or data["truncated"][left:right, lane].any()
            ):
                reject("fault_or_truncation")
                continue
            if not np.all(np.diff(data["episode_steps"][left:right, lane]) == 1):
                reject("noncontiguous_steps")
                continue
            for seat in range(data["learner_mask"].shape[2]):
                if (
                    not data["learner_mask"][left:right, lane, seat].all()
                    or not data["valid"][left:right, lane, seat].all()
                ):
                    reject("nonlearner_or_invalid_seat")
                    continue
                versions = np.unique(data["policy_versions"][left:right, lane, seat])
                if len(versions) != 1 or not 0 <= versions[0] < FROZEN_VERSION_BASE:
                    reject("mixed_or_frozen_behavior")
                    continue
                mask[left:right, lane, seat] = True
                games.add((lane, int(ids[left])))
                episodes.append(
                    dict(
                        lane=lane,
                        episode_id=int(ids[left]),
                        seat=seat,
                        rows=int(right - left),
                        behavior_version=int(versions[0]),
                    )
                )
    return mask, dict(
        eligible_games=len(games),
        eligible_episode_seats=len(episodes),
        target_count=int(mask.sum()),
        excluded=excluded,
        episodes=episodes,
    )


def load_chunk(path: Path) -> tuple[dict, dict]:
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata.get("training_driver") != "persistent-v2":
        raise ValueError("Requires persistent-v2 recorded target provenance")
    fields = metadata["schema"]["header_fields"]

    def columns(name):
        return [i for i, field in enumerate(fields) if field == name]

    with np.load(path) as archive:
        data = {key: archive[key] for key in SMALL_FIELDS}
        actions = archive["actions"]
        masks = archive["masks"]
        choice = masks.sum(-1) > 1
        legal_slots = masks[..., 1:2305].reshape(*actions.shape, 4, 576).any(-1)
        del masks
        obs = archive["obs"]
        costs = obs[..., columns("hand_cost")]
        cards = obs[..., columns("hand_card")]
        pending = obs[..., columns("hand_pending")]
        available = (
            obs[..., columns("elixir")[0]].astype(np.float32)
            - obs[..., columns("elixir_reserved")[0]].astype(np.float32)
        ) / 10000
        del obs
    high = (cards >= 0) & (pending == 0) & (costs >= 5)
    eligible = (data["learner_mask"] & data["valid"] & data["bootstrap_valid"]).astype(bool)
    chosen_slots = ((actions - 1) // 576).clip(0, 3)
    selected_costs = np.take_along_axis(costs, chosen_slots[..., None], -1)[..., 0]
    play = (actions > 0) & (actions < 2305)
    high_legal = (high & legal_slots).any(-1)
    data["groups"] = {
        "valid_learner": eligible,
        "choice": eligible & choice,
        "high_cost_legal_choice": eligible & choice & high_legal,
        "high_in_hand_unaffordable_choice": eligible
        & choice
        & (high & (costs > available[..., None])).any(-1),
        "available_elixir_below_two_choice": eligible & choice & (available < 2),
        "available_elixir_at_least_five_choice": eligible & choice & (available >= 5),
        "wait_choice": eligible & choice & (actions == 0),
        "high_cost_request": eligible & play & (selected_costs >= 5),
    }
    data["available_elixir"] = available
    if not all(
        np.isfinite(data[name]).all()
        for name in ("training_rewards", "old_values", "bootstrap_values")
    ):
        raise ValueError("Non-finite logged target inputs")
    return data, metadata


def reconstruct_chunk(data: dict, gamma: float, actor_lambda: float, critic_lambda: float):
    """Keep the full time grid, terminal bootstrap and episode-boundary masks."""
    rewards = torch.from_numpy(data["training_rewards"].reshape(len(data["training_rewards"]), -1))
    values = torch.from_numpy(data["old_values"].reshape(rewards.shape))
    following = torch.from_numpy(data["bootstrap_values"].reshape(rewards.shape))
    terminals = torch.from_numpy(
        np.repeat(
            data["terminated"][..., None], values.shape[1] // data["terminated"].shape[1], axis=-1
        ).reshape(values.shape)
    )
    ends = torch.from_numpy(
        np.repeat(
            data["dones"][..., None], values.shape[1] // data["dones"].shape[1], axis=-1
        ).reshape(values.shape)
    )
    advantages, critic_targets = compute_actor_critic_targets(
        rewards,
        values,
        terminals,
        following[-1],
        gamma=gamma,
        gae_lambda=actor_lambda,
        critic_gae_lambda=critic_lambda,
        next_values=following,
        episode_ends=ends,
    )
    choice_rows = np.flatnonzero(data["groups"]["choice"].reshape(-1))
    normalized = normalize_actor_advantages(advantages.flatten(), choice_rows)
    shape = data["training_rewards"].shape
    return dict(
        advantage=advantages.numpy().reshape(shape),
        normalized_advantage=normalized.numpy().reshape(shape),
        actor_return=(advantages + values).numpy().reshape(shape),
        critic_target=critic_targets.numpy().reshape(shape),
    )


def stats(values: np.ndarray) -> dict:
    if not len(values):
        return dict(count=0)
    return dict(
        count=int(len(values)),
        mean=float(values.mean()),
        std=float(values.std()),
        abs_p95=float(np.percentile(np.abs(values), 95)),
        abs_max=float(np.abs(values).max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.993, 0.997, 0.999])
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    loaded = [load_chunk(path) for path in args.rollouts]
    chunks, metadata = zip(*loaded, strict=True)
    gamma = metadata[0]["gamma"]
    critic_lambda = metadata[0]["critic_gae_lambda"]
    baseline_lambda = metadata[0]["gae_lambda"]
    if any(
        m["gamma"] != gamma
        or m["critic_gae_lambda"] != critic_lambda
        or m["gae_lambda"] != baseline_lambda
        for m in metadata
    ):
        raise ValueError("Baseline target provenance changed between input rollouts")
    lambdas = list(dict.fromkeys([baseline_lambda, *args.lambdas]))
    reconstructed = {}
    for lam in lambdas:
        parts = [reconstruct_chunk(d, gamma, lam, critic_lambda) for d in chunks]
        reconstructed[lam] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    baseline = reconstructed[baseline_lambda]
    groups = {
        name: np.concatenate([d["groups"][name] for d in chunks]) for name in chunks[0]["groups"]
    }
    data = {key: np.concatenate([d[key] for d in chunks]) for key in SMALL_FIELDS}
    mc_mask, mc_coverage = complete_episode_mask(data)
    rewards = torch.from_numpy(data["training_rewards"].reshape(len(data["training_rewards"]), -1))
    zeros = torch.zeros_like(rewards)
    terminal = torch.from_numpy(
        np.repeat(data["terminated"][..., None], 2, -1).reshape(rewards.shape)
    )
    ends = torch.from_numpy(np.repeat(data["dones"][..., None], 2, -1).reshape(rewards.shape))
    _, mc = compute_actor_critic_targets(
        rewards,
        zeros,
        terminal,
        zeros[-1],
        gamma=gamma,
        gae_lambda=1,
        critic_gae_lambda=1,
        next_values=zeros,
        episode_ends=ends,
    )
    mc = mc.numpy().reshape(data["training_rewards"].shape)
    report = dict(
        gamma=gamma,
        critic_lambda_fixed=critic_lambda,
        baseline_actor_lambda=baseline_lambda,
        sources=[dict(path=str(p), sha256=file_sha256(p)) for p in args.rollouts],
        full_episode_mc_coverage=mc_coverage,
        group_counts={name: int(group.sum()) for name, group in groups.items()},
        runs={},
        limitations=[
            "Offline recorded targets only; no new games, gradient updates, or strength evaluation",
            (
                "Larger lambda approaches logged MC by construction; "
                "lower offline MC error is not proof of faster convergence"
            ),
            (
                "Frames within games are correlated, and rare high-cost subsets "
                "have very small effective sample size"
            ),
            (
                "Critic target lambda, physical-time grid, true-terminal bootstrap "
                "and episode-end masks remain unchanged"
            ),
            (
                "Extending Actor lambda changes the bias/variance tradeoff; "
                "it is a separate algorithm experiment, not an equivalent speed change"
            ),
        ],
    )
    for lam, run in reconstructed.items():
        same_critic = np.array_equal(run["critic_target"], baseline["critic_target"])
        if not same_critic:
            raise AssertionError("Changing Actor lambda unexpectedly changed Critic targets")
        per_group = {}
        for name, group in groups.items():
            complete = group & mc_mask
            selected = run["actor_return"][complete]
            actual_mc = mc[complete]
            r = dict(
                advantage=stats(run["advantage"][group]),
                normalized_advantage=stats(run["normalized_advantage"][group]),
                actor_return=stats(run["actor_return"][group]),
                mc_comparison_count=int(complete.sum()),
            )
            if complete.any():
                r["actor_return_mc_mse"] = float(np.mean((selected - actual_mc) ** 2))
                r["actor_return_mc_bias"] = float(np.mean(selected - actual_mc))
                r["advantage_mc_correlation"] = (
                    float(
                        np.corrcoef(
                            run["advantage"][complete], actual_mc - data["old_values"][complete]
                        )[0, 1]
                    )
                    if len(selected) > 1
                    and run["advantage"][complete].std() > 0
                    and (actual_mc - data["old_values"][complete]).std() > 0
                    else None
                )
            if group.any():
                r["normalized_advantage_sign_flip_fraction_vs_baseline"] = float(
                    np.mean(
                        np.sign(run["normalized_advantage"][group])
                        != np.sign(baseline["normalized_advantage"][group])
                    )
                )
            per_group[name] = r
        report["runs"][str(lam)] = dict(
            trace=trace_decay(gamma, lam, args.step_seconds),
            critic_targets_identical=same_critic,
            groups=per_group,
        )
    report["existing_rollout_value_diagnostics"] = {
        p.name: rollout_value_diagnostics(p) for p in args.rollouts
    }
    report["code_sha256"] = file_sha256(Path(__file__))
    atomic_json(args.out, report)
    print(
        json.dumps(
            dict(
                out=str(args.out),
                group_counts=report["group_counts"],
                complete_mc_games=mc_coverage["eligible_games"],
                lambda_summary={
                    lam: dict(
                        e_fold_seconds=r["trace"]["e_fold_seconds"],
                        choice_adv_std=r["groups"]["choice"]["advantage"]["std"],
                        choice_mc_mse=r["groups"]["choice"].get("actor_return_mc_mse"),
                        normalized_choice_sign_flips=r["groups"]["choice"].get(
                            "normalized_advantage_sign_flip_fraction_vs_baseline"
                        ),
                    )
                    for lam, r in report["runs"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
