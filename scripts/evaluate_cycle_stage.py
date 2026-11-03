"""Run a frozen staged-BC calibration protocol and audit promotion; never auto-start PPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__:
    from .followup_cycle_stage import paused_next_trainer
else:
    from followup_cycle_stage import paused_next_trainer

from cr_rl.evaluation.paired import paired_outcomes
from cr_rl.evaluation.promotion import verify_promotion
from cr_rl.training.calibrated_export import export_calibrated_bc
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def choose_timing_bias(calibration, evaluations):
    """Minimize held-out timing NLL subject to predeclared tuning WAIT limits."""
    eligible = []
    for row in calibration["grid"]:
        evaluated = evaluations[row["bias"]]
        if evaluated["games"] != 4 or evaluated["invalid_games"]:
            continue
        paired_outcomes(evaluated, minimum_pairs=2)
        audits = [m["decision_audit"]["learner"] for m in evaluated["matchups"]]
        if len(audits) != 4:
            raise ValueError("Expected four tuning games, not a partial decision audit")
        if all(
            a["full_elixir_wait_legal_seconds"] <= 2
            and a["longest_full_elixir_wait_legal_seconds"] <= 1
            for a in audits
        ):
            eligible.append(row)
    if not eligible:
        raise ValueError("No predeclared bias passed tuning WAIT limits")
    return min(eligible, key=lambda row: row["binary_timing_nll"])["bias"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--drive-run", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--pair-start", type=int, default=13)
    args = parser.parse_args()
    if args.stage < 2 or args.pair_start < 13:
        parser.error("Use a new stage and fresh final deck pairs after the first assessment")
    run = args.run.resolve()
    followup = run / f"stage_{args.stage:04d}_followup"
    out = run / f"stage_{args.stage:04d}_evaluation"
    if (out / "status.json").exists():
        raise ValueError(
            "Evaluation already initialized; inspect its recorded worker before retrying"
        )
    out.mkdir(exist_ok=True)
    drive = args.drive_run / out.name
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def publish(phase, **extra):
        atomic_json(
            out / "status.json",
            dict(
                phase=phase, observed_unix=time.time(), stage=args.stage, ppo_started=False, **extra
            ),
        )
        copy_verified(out / "status.json", drive / "status.json")

    def sync(directory):
        for path in directory.rglob("*"):
            if path.is_file() and (
                path.suffix in (".json", ".jsonl") or path.name.endswith(".calibration.json")
            ):
                copy_verified(path, drive / path.relative_to(out))

    def execute(command):
        subprocess.run([sys.executable, *command], env=env, check=True)

    def evaluate(policy, baseline, directory, pairs, start, bias=0):
        execute(
            [
                "scripts/evaluate_policy.py",
                "--policy",
                str(policy),
                "--opponent",
                str(baseline),
                "--decks",
                "configs/decks_eval.yaml",
                "--pairs",
                str(pairs),
                "--pair-start",
                str(start),
                "--wait-logit-bias",
                str(bias),
                "--out",
                str(directory),
            ]
        )
        evaluation = json.loads((directory / "evaluation.json").read_text())
        atomic_json(directory / "paired.json", paired_outcomes(evaluation, minimum_pairs=pairs))
        execute(
            [
                "scripts/audit_policy_replays.py",
                "--evaluation",
                str(directory / "evaluation.json"),
                "--out",
                str(directory / "execution_audit.json"),
            ]
        )
        execute(
            [
                "scripts/prepare_replay_archive.py",
                str(directory),
                "--policy-label",
                f"BC CYCLE STAGE {args.stage} WAIT+{bias:g}",
                "--opponent-label",
                f"BC {baseline.parent.name}",
            ]
        )
        sync(directory)
        return evaluation

    try:
        protocol = dict(
            bias_grid=[0, 1, 1.5, 2, 2.5],
            tuning_pairs=[0, 1],
            final_pair_start=args.pair_start,
            final_pairs_per_baseline=8,
            selection="Lowest natural timing NLL passing all tuning WAIT limits",
            full_wait_total_limit_seconds=2,
            full_wait_continuous_limit_seconds=1,
            final_outcomes_not_used_for_bias_selection=True,
        )
        atomic_json(out / "protocol.json", protocol)
        sync(out)
        publish("waiting_stage_calibration")
        while True:
            path = followup / "status.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status["phase"] == "failed":
                    raise RuntimeError("Stage calibration failed; retain all evidence")
                if status["phase"] == "completed":
                    worker = json.loads((followup / "worker.json").read_text())["pid"]
                    process = Path(f"/proc/{worker}/stat")
                    if not process.exists() or process.read_text().split()[2] == "Z":
                        break
            if json.loads((run / "status.json").read_text())["phase"] == "failed":
                raise RuntimeError("BC failed before the evaluation candidate was ready")
            time.sleep(15)
        candidate = followup / "candidate.pt"
        calibration = json.loads((followup / "report.json").read_text())
        if file_sha256(candidate) != calibration["provenance"]["checkpoint_sha256"]:
            raise ValueError("Calibration candidate hash changed")
        baselines = {
            "early_step237": Path("artifacts/v2_gui_bc_20261103_01/early.actor.pt").resolve(),
            "conditional_step896": Path(
                "artifacts/v2_conditional_expanded_eval_20261103/current.actor.pt"
            ).resolve(),
        }
        protocol.update(
            candidate_sha256=file_sha256(candidate),
            baseline_sha256={k: file_sha256(v) for k, v in baselines.items()},
        )
        atomic_json(out / "protocol.json", protocol)
        with paused_next_trainer(run, args.stage) as pid:
            evaluations = {}
            for bias in protocol["bias_grid"]:
                publish("tuning", bias=bias, paused_trainer_pid=pid)
                directory = out / f"tuning_{bias:g}"
                evaluations[bias] = evaluate(
                    candidate, baselines["conditional_step896"], directory, 2, 0, bias
                )
            try:
                selected = choose_timing_bias(calibration, evaluations)
            except ValueError as error:
                atomic_json(
                    out / "promotion_gate.json",
                    dict(
                        passed=False,
                        ppo_started=False,
                        reason=str(error),
                        next="Continue BC; no timing bias passed the tuning protocol",
                    ),
                )
                sync(out)
                publish("completed", promotion_passed=False, reason=str(error))
                return
            baked = out / "calibrated_bc.pt"
            export = export_calibrated_bc(candidate, selected, followup / "report.json", baked)
            protocol.update(selected_bias=selected, checkpoint_sha256=export["checkpoint_sha256"])
            atomic_json(out / "protocol.json", protocol)
            sync(out)
            copy_verified(baked, drive / baked.name)
            report = dict(
                checkpoint=str(baked), checkpoint_sha256=file_sha256(baked), comparisons={}
            )
            records = []
            for name, baseline in baselines.items():
                publish(
                    "final_evaluation",
                    baseline=name,
                    selected_bias=selected,
                    paused_trainer_pid=pid,
                )
                directory = out / name
                evaluate(baked, baseline, directory, 8, args.pair_start)
                # The bias is baked; the viewer label must describe the effective correction.
                execute(
                    [
                        "scripts/prepare_replay_archive.py",
                        str(directory),
                        "--policy-label",
                        f"BC CYCLE STAGE {args.stage} WAIT+{selected:g}",
                        "--opponent-label",
                        f"BC {name} WAIT+0",
                    ]
                )
                sync(directory)
                records.extend((directory / "all_matches.jsonl").read_text().splitlines())
                report["comparisons"][name] = dict(
                    evaluation=str(directory / "evaluation.json"),
                    evaluation_sha256=file_sha256(directory / "evaluation.json"),
                    baseline_sha256=file_sha256(baseline),
                    execution_audit=str(directory / "execution_audit.json"),
                    execution_audit_sha256=file_sha256(directory / "execution_audit.json"),
                )
            (out / "all_matches.jsonl").write_text("\n".join(records) + "\n")
            atomic_json(out / "promotion_report.json", report)
        try:
            _, evidence = verify_promotion(out / "promotion_report.json")
            gate = dict(
                passed=True,
                evidence=evidence,
                next="Verify final full-corpus BC readiness before PPO",
            )
        except ValueError as error:
            gate = dict(
                passed=False, reason=str(error), next="Continue BC; do not retune on final games"
            )
        atomic_json(out / "promotion_gate.json", dict(ppo_started=False, **gate))
        sync(out)
        publish("completed", selected_bias=selected, promotion_passed=gate["passed"])
    except Exception:
        publish("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
