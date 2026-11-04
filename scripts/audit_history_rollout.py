"""Independently audit current-policy membership and behavior of a saved history rollout."""

import argparse
from pathlib import Path

import numpy as np

from cr_rl.opponents.role_sampling import FROZEN_VERSION_BASE
from cr_rl.training.persistence import atomic_json, file_sha256


def audit_arrays(data, behavior_version):
    if not 0 <= behavior_version < FROZEN_VERSION_BASE:
        raise ValueError("Expected current behavior version is outside its namespace")
    versions = data["policy_versions"]
    learners = data["learner_mask"].astype(bool)
    valid = data["valid"].astype(bool)
    if not np.array_equal(learners, versions == behavior_version):
        raise ValueError("Learner mask contains a frozen or mismatched behavior policy")
    if (versions[~learners] < FROZEN_VERSION_BASE).any():
        raise ValueError("Frozen policy is outside the reserved version namespace")
    current = learners & valid
    if not current.any():
        raise ValueError("Rollout contains no valid current-policy samples")
    if not np.isfinite(data["old_log_probs"][current]).all() or not np.isfinite(
        data["old_values"][current]
    ).all():
        raise ValueError("Current-policy snapshots are not finite")
    actions = data["actions"][current]
    legal = data["masks"][current]
    if ((actions < 0) | (actions >= legal.shape[-1])).any():
        raise ValueError("Recorded action is outside the action codec")
    if not np.take_along_axis(legal, actions[:, None], axis=-1).all():
        raise ValueError("Current-policy action was not legal under its recorded mask")
    available = legal[:, 1:2305].any(-1)
    frozen_ids, counts = np.unique(versions[~learners], return_counts=True)
    return dict(
        passed=True,
        behavior_version=int(behavior_version),
        valid_learner_samples=int(current.sum()),
        frozen_opponent_decisions=int((~learners).sum()),
        frozen_decisions_in_learner_mask=0,
        frozen_version_counts={str(k): int(v) for k, v in zip(frozen_ids, counts, strict=True)},
        card_action_decisions=int(((actions >= 1) & (actions <= 2304)).sum()),
        nonwait_fraction=float((actions != 0).mean()),
        card_action_fraction=float(((actions >= 1) & (actions <= 2304)).mean()),
        card_available_decisions=int(available.sum()),
        wait_when_card_legal=float((actions[available] == 0).mean()) if available.any() else None,
        sampled_surprisal=float(-data["old_log_probs"][current].mean()),
        note="All valid learner decisions; card commands are not confirmed successful plays",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--behavior-version", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.rollout) as archive:
        report = audit_arrays(archive, args.behavior_version)
    report.update(rollout_sha256=file_sha256(args.rollout), code_sha256=file_sha256(Path(__file__)))
    atomic_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
