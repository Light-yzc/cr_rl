"""Learning metrics and explicit chart axes; progress stays in run metadata."""

from __future__ import annotations

import math


class TrainingTelemetry:
    def __init__(self, run, samples_per_update: int):
        self.run = run
        self.samples_per_update = samples_per_update
        self.episodes = 0
        self.score_sum = 0.0
        self.wins = 0
        self.per_deck = {}
        run.define_metric("charts/learner_samples", hidden=True)
        run.define_metric("charts/optimizer_steps", hidden=True)
        run.define_metric("charts/critic_optimizer_steps", hidden=True)
        run.define_metric("critic/*", step_metric="charts/critic_optimizer_steps")
        for group in ("policy", "selfplay", "value", "performance", "reward"):
            run.define_metric(group + "/*", step_metric="charts/learner_samples")
        run.define_metric("ppo/*", step_metric="charts/optimizer_steps")
        # These were accidentally emitted as charts by the first supervisor version.
        for group in ("rollout", "optimization", "train", "loss", "backup"):
            run.define_metric(group + "/*", hidden=True)

    def log(self, event):
        phase = event["phase"]
        update = event["update"]
        samples = event.get("learner_samples_total", update * self.samples_per_update)
        self.run.summary["progress"] = {
            "phase": phase,
            "update": update,
            "learner_samples": samples,
        }
        values = {"charts/learner_samples": samples}
        if phase in ("policy", "targets"):
            group = "policy" if phase == "policy" else "value"
            for key, value in event.items():
                if key not in ("phase", "update", "learner_samples_total", "time_unix"):
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        values[f"{group}/{key}"] = value
        elif phase == "episode":
            if not event.get("fault") and not event.get("truncated"):
                self.episodes += 1
                score = float(event["reward"])
                self.score_sum += score
                self.wins += int(score > 0)
                if event.get("learner_deck"):
                    row = self.per_deck.setdefault(
                        event["learner_deck"], {"games": 0, "wins": 0, "losses": 0, "draws": 0}
                    )
                    row["games"] += 1
                    row["wins"] += int(score > 0)
                    row["losses"] += int(score < 0)
                    row["draws"] += int(score == 0)
                    self.run.summary["per_deck_training_outcomes"] = self.per_deck
                values.update(
                    {
                        "selfplay/episode_return": score,
                        "selfplay/mean_return": self.score_sum / self.episodes,
                        "selfplay/win_rate": self.wins / self.episodes,
                        "selfplay/episode_length": event["episode_length"],
                        "selfplay/completed_games": self.episodes,
                    }
                )
        elif phase == "critic_optimization":
            values["charts/critic_optimizer_steps"] = event["critic_optimizer_steps"]
            for key in ("value_loss", "grad_norm", "learning_rate"):
                values["critic/" + key] = event[key]
        elif phase == "critic_warmup":
            for key in (
                "actor_enabled",
                "completed_games",
                "critic_samples",
                "warmup_rounds",
                "warmup_games",
            ):
                values["value/" + key] = event[key]
        elif phase == "kl_stop":
            values["policy/kl_stopped"] = 1
            values["policy/kl_at_stop"] = event.get("reference_kl", event["approx_kl"])
        elif phase == "optimization":
            steps = event.get("optimizer_steps_total", event["optimizer_steps"])
            values["charts/optimizer_steps"] = steps
            for key in (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "reference_kl",
                "grad_norm",
                "learning_rate",
                "ratio_mean",
            ):
                if key in event:
                    values["ppo/" + ("total_loss" if key == "loss" else key)] = event[key]
        elif (
            phase == "rollout" and event.get("seconds", 0) > 0 and event.get("rollout_step", 0) > 0
        ):
            # True rollout throughput, not replay optimizer throughput.
            local_samples = event.get(
                "rollout_samples", samples - (update - 1) * self.samples_per_update
            )
            values["performance/rollout_samples_per_second"] = local_samples / event["seconds"]
        if len(values) > 1:
            self.run.log(values)

    def log_summary(self, summary):
        samples = summary.get("learner_samples_total", summary["update"] * self.samples_per_update)
        values = {
            "charts/learner_samples": samples,
            "performance/update_seconds": summary["seconds"],
            "performance/peak_cuda_gib": summary["peak_cuda_allocated_gib"],
            "selfplay/terminal_games_per_update": summary["terminal_games"],
            "selfplay/faulted_samples": summary["faulted_samples"],
        }
        for key, value in summary["metrics"].items():
            values["ppo_update/" + key] = value
        for key, value in summary.get("reward_component_sums", {}).items():
            values[f"reward/{key}_sum"] = value
        if "learner_reward_sum" in summary:
            values["reward/training_sum"] = summary["learner_reward_sum"]
        if "learner_raw_reward_sum" in summary:
            values["reward/raw_terminal_sum"] = summary["learner_raw_reward_sum"]
        self.run.define_metric("ppo_update/*", step_metric="charts/learner_samples")
        self.run.log(values)


def replay_rollout_metrics(path, telemetry):
    """Recover observable metrics from legacy data; full entropy is not stored there."""
    import numpy as np

    with np.load(path) as data:
        update = int(path.stem.rsplit("_", 1)[1])
        envs = data["actions"].shape[1]
        offset = (update - 1) * telemetry.samples_per_update
        for left in range(0, len(data["actions"]), 64):
            right = min(left + 64, len(data["actions"]))
            for step in range(left, right):
                for env in np.flatnonzero(data["dones"][step]):
                    seat = int(data["learner_mask"][step, env].argmax())
                    telemetry.log(
                        {
                            "phase": "episode",
                            "update": update,
                            "learner_samples_total": offset + (step + 1) * envs,
                            "reward": float(data["rewards"][step, env, seat]),
                            "episode_length": int(data["episode_steps"][step, env]),
                            "fault": int(data["fault_codes"][step, env]),
                            "truncated": int(data["truncated"][step, env]),
                        }
                    )
            selected = (data["learner_mask"][left:right] & data["valid"][left:right]).astype(bool)
            if not selected.any():
                continue
            actions = data["actions"][left:right][selected]
            legal = data["masks"][left:right][selected]
            available = legal[:, 1:2305].any(-1)
            values = data["old_values"][left:right][selected]
            telemetry.log(
                {
                    "phase": "policy",
                    "update": update,
                    "learner_samples_total": offset + right * envs,
                    "play_fraction": float((actions != 0).mean()),
                    "wait_when_play_legal": float((actions[available] == 0).mean())
                    if available.any()
                    else None,
                    "legal_action_count": float(legal.sum(-1).mean()),
                    "sampled_surprisal": float(-data["old_log_probs"][left:right][selected].mean()),
                    "value_mean": float(values.mean()),
                    "value_std": float(values.std()),
                }
            )


def rollout_value_diagnostics(path):
    """Reconstruct the original pre-fit GAE targets from immutable rollout data.

    This is explained variance of the frozen rollout critic, not a post-fit value
    measurement. Nothing here changes training arrays, rewards or old values.
    """
    import json

    import numpy as np
    import torch

    from cr_rl.training.advantages import compute_gae

    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if metadata.get("training_driver") != "persistent-v2":
        raise ValueError("Value diagnostics require persistent-v2 target provenance")
    with np.load(path) as archive:
        rewards, values, next_values = (
            archive[name].astype(np.float32)
            for name in ("training_rewards", "old_values", "bootstrap_values")
        )
        valid = (archive["learner_mask"] & archive["valid"] & archive["bootstrap_valid"]).astype(
            bool
        )
        terminal, ends = archive["terminated"], archive["dones"]
    if not valid.any() or not all(np.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError("No finite valid value targets for diagnostics")
    shape = rewards.shape[0], -1
    advantages, targets = compute_gae(
        torch.from_numpy(rewards.reshape(shape)),
        torch.from_numpy(values.reshape(shape)),
        torch.from_numpy(np.repeat(terminal[..., None], 2, axis=-1).reshape(shape)),
        torch.from_numpy(next_values[-1].reshape(-1)),
        gamma=metadata["gamma"],
        gae_lambda=metadata["gae_lambda"],
        next_values=torch.from_numpy(next_values.reshape(shape)),
        episode_ends=torch.from_numpy(np.repeat(ends[..., None], 2, axis=-1).reshape(shape)),
    )
    target = targets.numpy().reshape(rewards.shape)[valid]
    advantage = advantages.numpy().reshape(rewards.shape)[valid]
    variance = float(target.var())
    result = dict(
        return_mean=float(target.mean()),
        return_std=float(target.std()),
        advantage_mean=float(advantage.mean()),
        advantage_std=float(advantage.std()),
        value_mean=float(values[valid].mean()),
        value_std=float(values[valid].std()),
        target_count=int(valid.sum()),
        explained_variance_defined=variance > 1e-12,
    )
    if variance > 1e-12:
        result["explained_variance"] = 1.0 - float((target - values[valid]).var()) / variance
    return result


def create_training_workspace(entity, project, run_name):
    """A manual workspace keeps metadata counters out of the learning dashboard."""
    import wandb_workspaces.reports.v2 as panels
    import wandb_workspaces.workspaces as workspaces

    sample_axis = "charts/learner_samples"
    optimizer_axis = "charts/optimizer_steps"
    groups = [
        (
            "Policy during sampling",
            sample_axis,
            [
                ("Policy entropy", ["policy/entropy", "policy/entropy_when_play_legal"]),
                ("Card play fraction", ["policy/play_fraction"]),
                ("WAIT when a card can be played", ["policy/wait_when_play_legal"]),
                ("Largest action probability", ["policy/max_action_probability"]),
            ],
        ),
        (
            "Self-play outcomes (training opponents)",
            sample_axis,
            [
                ("Episode reward", ["selfplay/episode_return", "selfplay/mean_return"]),
                ("Self-play win rate", ["selfplay/win_rate"]),
                ("Game length in decisions", ["selfplay/episode_length"]),
            ],
        ),
        (
            "PPO optimization",
            optimizer_axis,
            [
                ("Actor and total loss", ["ppo/policy_loss", "ppo/total_loss"]),
                ("Value loss", ["ppo/value_loss"]),
                ("KL divergence", ["ppo/approx_kl", "ppo/reference_kl"]),
                ("Clipped probability ratios", ["ppo/clip_fraction"]),
                ("Gradient norm before clipping", ["ppo/grad_norm"]),
            ],
        ),
        (
            "Independent critic optimization",
            "charts/critic_optimizer_steps",
            [
                ("Critic regression loss", ["critic/value_loss"]),
                ("Critic gradient norm", ["critic/grad_norm"]),
                ("Critic learning rate", ["critic/learning_rate"]),
            ],
        ),
        (
            "Value targets and throughput",
            sample_axis,
            [
                ("Value explained variance", ["value/explained_variance"]),
                ("Return mean and spread", ["value/return_mean", "value/return_std"]),
                ("Rollout learner samples per second", ["performance/rollout_samples_per_second"]),
                ("Peak allocated GPU GiB", ["performance/peak_cuda_gib"]),
            ],
        ),
    ]
    workspace = workspaces.Workspace(
        entity=entity,
        project=project,
        name="CR RL 119M learning metrics",
        auto_generate_panels=False,
        runset_settings=workspaces.RunsetSettings(query=run_name),
        sections=[
            workspaces.Section(
                name=name,
                is_open=True,
                panels=[
                    panels.LinePlot(title=title, x=axis, y=metrics) for title, metrics in specs
                ],
            )
            for name, axis, specs in groups
        ],
    )
    workspace.save()
    return workspace.url
