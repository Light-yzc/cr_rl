import json

import pytest
import torch

from cr_rl.evaluation.promotion import verify_promotion
from cr_rl.training.persistence import file_sha256


def make_evidence(tmp_path):
    checkpoint = tmp_path / "calibrated.pt"
    torch.save(
        dict(stage="bc", model_architecture_version=2, calibration={"baked_into_weights": True}),
        checkpoint,
    )
    report = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), comparisons={}
    )
    for name in ["early_step237", "conditional_step896"]:
        matches = [
            dict(
                pair=i + 5,
                seed=i,
                deck0=f"a{i}",
                deck1=f"b{i}",
                learner_seat=seat,
                reward=1,
                valid=True,
                decision_audit={
                    "learner": dict(
                        full_elixir_wait_legal_seconds=0, longest_full_elixir_wait_legal_seconds=0
                    )
                },
            )
            for i in range(8)
            for seat in range(2)
        ]
        evaluation = dict(
            policy_sha256=report["checkpoint_sha256"],
            wait_logit_bias=0,
            opponent_sha256=name,
            invalid_games=0,
            matchups=matches,
        )
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(evaluation))
        for index in range(len(matches)):
            (tmp_path / f"match_{index:03d}.json").write_text("{}")
        native_path = tmp_path / f"{name}.native.json"
        native_path.write_text(
            json.dumps(
                dict(
                    evaluation=str(path.resolve()),
                    evaluation_sha256=file_sha256(path),
                    games=[
                        dict(
                            pair=m["pair"],
                            learner_seat=m["learner_seat"],
                            reward=m["reward"],
                            native_outcome_parity=True,
                            replay_sha256=file_sha256(tmp_path / f"match_{index:03d}.json"),
                            learner={"successful_plays": 20},
                        )
                        for index, m in enumerate(matches)
                    ],
                )
            )
        )
        report["comparisons"][name] = dict(
            evaluation=str(path),
            evaluation_sha256=file_sha256(path),
            baseline_sha256=name,
            execution_audit=str(native_path),
            execution_audit_sha256=file_sha256(native_path),
        )
    out = tmp_path / "promotion.json"
    out.write_text(json.dumps(report))
    return out, report


def test_promotion_requires_significant_fresh_paired_games_for_exact_weights(tmp_path):
    path, report = make_evidence(tmp_path)
    checkpoint, result = verify_promotion(path)
    assert checkpoint == tmp_path / "calibrated.pt"
    assert result["comparisons"]["early_step237"]["pairs"] == 8
    record = report["comparisons"]["early_step237"]
    eval_path = tmp_path / "early_step237.json"
    evaluation = json.loads(eval_path.read_text())
    for match in evaluation["matchups"]:
        match["reward"] = 1 if match["learner_seat"] == 0 else -1
    eval_path.write_text(json.dumps(evaluation))
    record["evaluation_sha256"] = file_sha256(eval_path)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unproven"):
        verify_promotion(path)


def test_promotion_rejects_changed_evidence_and_missing_baseline(tmp_path):
    path, report = make_evidence(tmp_path)
    (tmp_path / "early_step237.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_promotion(path)
    del report["comparisons"]["early_step237"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Both frozen"):
        verify_promotion(path)
