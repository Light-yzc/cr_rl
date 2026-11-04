"""Pre-update value diagnostics against complete, constant-policy episode returns."""

import argparse
import json
from pathlib import Path

import numpy as np

from cr_rl.opponents.role_sampling import FROZEN_VERSION_BASE
from cr_rl.training.persistence import atomic_json, file_sha256


def analyze_chunks(chunks, gamma):
    if not 0 < gamma <= 1:
        raise ValueError("Discount must be in (0, 1]")
    pending, excluded, episodes = {}, {}, []
    targets, predictions = [], []

    def reject(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for data in chunks:
        for step in range(len(data["episode_ids"])):
            for env in range(data["episode_ids"].shape[1]):
                key = env, int(data["episode_ids"][step, env])
                trace = pending.setdefault(
                    key,
                    dict(
                        started=bool(data["episode_start"][step, env]),
                        rewards=[],
                        values=[],
                        versions=[],
                        learners=[],
                        valid=[],
                        steps=[],
                    ),
                )
                for name, source in (
                    ("rewards", "training_rewards"),
                    ("values", "old_values"),
                    ("versions", "policy_versions"),
                    ("learners", "learner_mask"),
                    ("valid", "valid"),
                    ("steps", "episode_steps"),
                ):
                    trace[name].append(np.array(data[source][step, env], copy=True))
                terminal = bool(data["terminated"][step, env])
                fault = bool(data["fault_codes"][step, env])
                truncated = bool(data["truncated"][step, env])
                if not (terminal or fault or truncated):
                    continue
                del pending[key]
                if not terminal or fault or truncated:
                    reject("fault_or_truncation")
                    continue
                if not trace["started"]:
                    reject("missing_prefix")
                    continue
                if not np.all(np.diff(trace["steps"]) == 1):
                    reject("noncontiguous_steps")
                    continue
                reward = np.asarray(trace["rewards"], dtype=np.float64)
                values = np.asarray(trace["values"], dtype=np.float64)
                valid = np.asarray(trace["valid"], dtype=bool)
                versions = np.asarray(trace["versions"])
                learners = np.asarray(trace["learners"], dtype=bool)
                mc = np.zeros_like(reward)
                total = np.zeros(2, dtype=np.float64)
                for index in reversed(range(len(reward))):
                    total = reward[index] + gamma * total
                    mc[index] = total
                for seat in (0, 1):
                    if not learners[:, seat].all() or not valid[:, seat].all():
                        reject("nonlearner_or_invalid_seat")
                        continue
                    unique = np.unique(versions[:, seat])
                    if len(unique) != 1 or not 0 <= unique[0] < FROZEN_VERSION_BASE:
                        reject("mixed_behavior_versions")
                        continue
                    if not np.isfinite(mc[:, seat]).all() or not np.isfinite(values[:, seat]).all():
                        raise ValueError("Nonfinite complete-episode values or returns")
                    target, predicted = mc[:, seat], values[:, seat]
                    targets.extend(target.tolist())
                    predictions.extend(predicted.tolist())
                    episodes.append(
                        dict(
                            env=env,
                            episode_id=key[1],
                            seat=seat,
                            behavior_version=int(unique[0]),
                            decisions=len(target),
                            initial_value=float(predicted[0]),
                            initial_mc_return=float(target[0]),
                            mse=float(((target - predicted) ** 2).mean()),
                        )
                    )
    result = dict(
        eligible_episode_seats=len(episodes),
        eligible_games=len({(e["env"], e["episode_id"]) for e in episodes}),
        target_count=len(targets),
        excluded=excluded,
        unfinished_episodes=len(pending),
        episodes=episodes,
        gamma=gamma,
        note=(
            "Stored pre-fit values versus complete Monte Carlo shaped returns; "
            "constant behavior versions only; samples within games are correlated; "
            "not win-rate evidence"
        ),
    )
    if targets:
        target, predicted = np.asarray(targets), np.asarray(predictions)
        error = target - predicted
        variance = float(target.var())
        result.update(
            target_mean=float(target.mean()),
            target_std=float(target.std()),
            value_mean=float(predicted.mean()),
            value_std=float(predicted.std()),
            mse=float((error**2).mean()),
            mean_error=float(error.mean()),
            episode_equal_weight_mse=float(np.mean([e["mse"] for e in episodes])),
            explained_variance_defined=variance > 1e-12,
            explained_variance=1 - float(error.var()) / variance if variance > 1e-12 else None,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.rollouts)
    updates = [int(p.stem.rsplit("_", 1)[1]) for p in paths]
    if any(b != a + 1 for a, b in zip(updates, updates[1:], strict=False)):
        raise ValueError("Use unique contiguous rollout generations")
    gamma = None
    records = []
    fields = (
        "episode_ids",
        "episode_start",
        "episode_steps",
        "training_rewards",
        "old_values",
        "policy_versions",
        "learner_mask",
        "valid",
        "terminated",
        "truncated",
        "fault_codes",
    )
    chunks = []
    for path in paths:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if gamma is None:
            gamma = metadata["gamma"]
        elif gamma != metadata["gamma"]:
            raise ValueError("Discount changed within diagnostic series")
        with np.load(path) as archive:
            chunks.append({field: archive[field] for field in fields})
        records.append(dict(path=str(path), sha256=file_sha256(path)))
    result = analyze_chunks(chunks, gamma)
    result.update(sources=records, code_sha256=file_sha256(Path(__file__)))
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "sources")}))


if __name__ == "__main__":
    main()
