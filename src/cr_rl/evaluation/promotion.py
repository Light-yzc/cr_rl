"""Verify frozen paired evidence before promoting calibrated BC weights to PPO."""

import json
from pathlib import Path

import torch

from cr_rl.evaluation.paired import paired_outcomes
from cr_rl.training.persistence import file_sha256


def verify_promotion(path):
    path = Path(path)
    report = json.loads(path.read_text())
    checkpoint = Path(report["checkpoint"])
    digest = file_sha256(checkpoint)
    if digest != report["checkpoint_sha256"]:
        raise ValueError("Promoted checkpoint hash changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state.get("stage") != "bc" or state.get("model_architecture_version") != 2:
        raise ValueError("PPO promotion requires architecture-v2 BC weights")
    if not state.get("calibration", {}).get("baked_into_weights"):
        raise ValueError("PPO initialization must contain the evaluated WAIT correction")
    del state
    comparisons = report["comparisons"]
    if set(comparisons) != {"early_step237", "conditional_step896"}:
        raise ValueError("Both frozen baseline comparisons are required")
    summaries, cohorts = {}, []
    for name, record in comparisons.items():
        evaluation_path = Path(record["evaluation"])
        if file_sha256(evaluation_path) != record["evaluation_sha256"]:
            raise ValueError("Evaluation evidence changed")
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation["policy_sha256"] != digest or evaluation.get("wait_logit_bias") != 0:
            raise ValueError("Evaluation must use the exact baked PPO initialization weights")
        if evaluation["opponent_sha256"] != record["baseline_sha256"]:
            raise ValueError("Baseline evidence hash differs")
        if evaluation.get("opponent_wait_logit_bias", 0) != 0:
            raise ValueError("Frozen historical baselines must remain uncalibrated")
        if evaluation["invalid_games"]:
            raise ValueError("Incomplete or faulted evaluation cannot pass")
        paired = paired_outcomes(evaluation, minimum_pairs=8)
        if paired["mean_paired_reward"] <= 0 or paired["swept_pair_sign_test_one_sided_p"] > 0.05:
            raise ValueError(f"Significant paired improvement remains unproven against {name}")
        cohort = []
        unique = set()
        for row in paired["rows"]:
            decks = tuple(sorted((row["deck0"], row["deck1"])))
            if decks in unique or row["pair"] < 5:
                raise ValueError("Final cohort must contain fresh unique deck pairs")
            unique.add(decks)
            cohort.append((row["pair"], row["seed"], *decks))
        cohorts.append(sorted(cohort))
        for match in evaluation["matchups"]:
            audit = match.get("decision_audit", {}).get("learner")
            if audit is None:
                raise ValueError("Direct active-game WAIT audit is missing")
            if (
                audit["full_elixir_wait_legal_seconds"] > 2
                or audit["longest_full_elixir_wait_legal_seconds"] > 1
            ):
                raise ValueError("Full-elixir stalling exceeds the declared cadence gate")
        native_path = Path(record["execution_audit"])
        if file_sha256(native_path) != record["execution_audit_sha256"]:
            raise ValueError("Execution audit evidence changed")
        native = json.loads(native_path.read_text())
        if native.get("evaluation_sha256") != file_sha256(evaluation_path):
            raise ValueError("Native audit belongs to another evaluation")
        if len(native["games"]) != paired["games"]:
            raise ValueError("Native execution coverage is incomplete")
        for index, (match, game) in enumerate(
            zip(evaluation["matchups"], native["games"], strict=True)
        ):
            replay = evaluation_path.parent / f"match_{index:03d}.json"
            if game.get("replay_sha256") != file_sha256(replay):
                raise ValueError("Native execution replay hash differs")
            if (
                not game["native_outcome_parity"]
                or game["pair"] != match["pair"]
                or game["learner_seat"] != match["learner_seat"]
                or game["reward"] != match["reward"]
            ):
                raise ValueError("Native audit and evaluated games differ")
            if game["learner"]["successful_plays"] <= 0:
                raise ValueError("A policy that never places a card cannot pass cadence")
        summaries[name] = paired
    if cohorts[0] != cohorts[1]:
        raise ValueError("Both baselines must use the same frozen final cohort")
    return checkpoint, {"report_sha256": file_sha256(path), "comparisons": summaries}
