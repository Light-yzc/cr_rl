"""Real-simulator training acceptance suite. Run after bootstrap_linux.sh."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log: Path) -> None:
    print("+", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(log.read_text(encoding="utf-8")[-12000:], flush=True)
    result.check_returncode()


def audit_rollout(path: Path) -> dict:
    with np.load(path) as data:
        legal = np.take_along_axis(data["masks"], data["actions"][..., None], -1)[..., 0]
        assert legal.all(), f"Illegal actions in {path}"
        assert (data["learner_mask"].sum(-1) == 1).all()
        valid = data["valid"].astype(bool)
        assert valid.all(), f"Simulator faults in {path}"
        assert np.isfinite(data["old_log_probs"][valid]).all()
        assert np.isfinite(data["old_values"][valid]).all()
        assert not (data["terminated"] & data["truncated"]).any()
        np.testing.assert_array_equal(data["dones"], data["terminated"] | data["truncated"])
        learner = data["learner_mask"].astype(bool) & valid
        assert data["bootstrap_valid"][learner].all()
        assert np.isfinite(data["bootstrap_values"][learner]).all()
        terminals = data["terminated"].astype(bool)
        assert (data["bootstrap_values"][learner & terminals[..., None]] == 0).all()
        assert np.allclose(data["rewards"].sum(-1), 0)
        for step in range(1, len(data["actions"])):
            boundary = data["dones"][step - 1].astype(bool)
            if boundary.any():
                assert data["episode_start"][step, boundary].all()
                assert (
                    data["episode_ids"][step, boundary]
                    == data["episode_ids"][step - 1, boundary] + 1
                ).all()
        return {
            "path": str(path),
            "learner_samples": int(learner.sum()),
            "terminal_games": int(terminals.sum()),
            "nonzero_learner_rewards": int(np.count_nonzero(data["training_rewards"][learner])),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/linux_validation")
    parser.add_argument("--skip-general", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "validation.json"
    if report_path.exists():
        parser.error("Use a fresh output directory to preserve previous evidence")
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": args.device,
        "cuda": torch.version.cuda,
        "checks": [],
    }
    if args.device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name()
    run([sys.executable, "-m", "pytest", "-q"], output / "pytest.log")
    run([sys.executable, "-m", "cr_rl.cli", "doctor"], output / "doctor.log")
    if sys.platform.startswith("linux"):
        run(
            ["env", "-u", "JAVA_HOME", sys.executable, "-m", "cr_rl.cli", "doctor"],
            output / "doctor_java_discovery.log",
        )
    tiny = output / "tiny"
    base = [
        sys.executable,
        "scripts/train_selfplay.py",
        "--critic-mode",
        "shared",
        "--device",
        args.device,
        "--epochs",
        "1",
        "--model",
        "tiny",
        "--envs",
        "2",
        "--steps",
        "1300",
        "--batch-size",
        "8",
        "--grad-accumulation",
        "4",
        "--out",
        str(tiny),
    ]
    run([*base, "--updates", "2"], output / "tiny_train.log")
    checkpoint = torch.load(tiny / "ppo_latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 2
    assert checkpoint["optimizer"]["state"]
    before = checkpoint["model"]["wait_actor.weight"].clone()
    optimizer_step = max(
        float(state["step"]) for state in checkpoint["optimizer"]["state"].values()
    )
    del checkpoint
    # Move the entire run to prove that historical snapshots survive a different path.
    relocated = output / "tiny_resumed"
    shutil.move(str(tiny), relocated)
    resumed = [
        sys.executable,
        "scripts/train_selfplay.py",
        "--critic-mode",
        "shared",
        "--device",
        args.device,
        "--epochs",
        "1",
        "--model",
        "tiny",
        "--envs",
        "2",
        "--steps",
        "32",
        "--batch-size",
        "8",
        "--grad-accumulation",
        "4",
        "--out",
        str(relocated),
        "--resume",
        str(relocated / "ppo_latest.pt"),
        "--current-opponent-fraction",
        "0",
        "--historical-pfsp-fraction",
        "1",
        "--forgotten-opponent-fraction",
        "0",
    ]
    run(resumed, output / "tiny_resume.log")
    checkpoint = torch.load(relocated / "ppo_latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["updates"] == 3
    assert not torch.equal(before, checkpoint["model"]["wait_actor.weight"])
    assert max(float(s["step"]) for s in checkpoint["optimizer"]["state"].values()) > optimizer_step
    del checkpoint
    tiny_rows = [
        json.loads(line) for line in (relocated / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["update"] for row in tiny_rows] == [1, 2, 3]
    assert sum(row["terminal_games"] for row in tiny_rows) > 0
    assert sum(row["nonzero_learner_rewards"] for row in tiny_rows) > 0
    report["checks"].append({"name": "tiny_two_updates_and_relocated_resume", "metrics": tiny_rows})
    report["rollout_audits"] = [
        audit_rollout(path) for path in sorted((relocated / "rollouts").glob("*.npz"))
    ]
    if not args.skip_general:
        general = output / "general"
        run(
            [
                sys.executable,
                "scripts/train_selfplay.py",
                "--critic-mode",
                "shared",
                "--model",
                "general",
                "--envs",
                "1",
                "--steps",
                "132",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--grad-accumulation",
                "4",
                "--target-kl",
                "100",
                "--updates",
                "2",
                "--device",
                args.device,
                "--out",
                str(general),
            ],
            output / "general_train.log",
        )
        rows = [json.loads(line) for line in (general / "metrics.jsonl").read_text().splitlines()]
        assert [row["update"] for row in rows] == [1, 2]
        assert all(row["max_training_history_len"] == 128 for row in rows)
        report["checks"].append({"name": "general_128_frame_context_two_updates", "metrics": rows})
        report["rollout_audits"].extend(
            audit_rollout(p) for p in sorted((general / "rollouts").glob("*.npz"))
        )
    for check in report["checks"]:
        for row in check["metrics"]:
            assert row["optimizer_steps"] > 0
            assert all(np.isfinite(value) for value in row["metrics"].values())
            assert row["initial_logp_max_error"] < 0.1, row
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
