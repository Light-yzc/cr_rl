"""Run the preregistered frozen-policy seed replication on an existing mounted VM."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cr_rl.evaluation.repeated import summarize_repeated_evaluations
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    original = Path("/content/drive/MyDrive/cr_rl/runs/121m_ppo_pilot_20261104")
    destination = original.parent / "121m_ppo_seed_replication_20261104"
    out = root / "artifacts/121m_ppo_seed_replication_20261104"
    if not original.is_dir() or not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Use the existing Colab kernel after mounting Drive")
    if out.exists():
        raise RuntimeError("Replication already exists; inspect its worker before restarting")
    out.mkdir(parents=True)
    protocol = json.loads((original / "followup_protocol.json").read_text())
    protocol["evaluation_code_sha256"] = file_sha256(root / "scripts/evaluate_policy.py")
    protocol["aggregator_sha256"] = file_sha256(root / "src/cr_rl/evaluation/repeated.py")
    if file_sha256(root / protocol["deck_pool"]) != protocol["deck_pool_sha256"]:
        raise ValueError("Deck pool changed since registration")
    atomic_json(out / "protocol.json", protocol)
    copy_verified(out / "protocol.json", destination / "protocol.json")
    copy_verified(Path(__file__), destination / "source/colab_seed_replication.py")
    began = time.monotonic()
    child = None

    def status(phase, **details):
        record = dict(phase=phase, pid=os.getpid(), observed_unix=time.time(), **details)
        atomic_json(out / "status.json", record)
        copy_verified(out / "status.json", destination / "status.json")
        print(json.dumps(record), flush=True)

    def run(command, target, timeout):
        nonlocal child
        with target.open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
            )
            started = time.monotonic()
            while child.poll() is None:
                status(
                    "running",
                    child_pid=child.pid,
                    task=target.name,
                    elapsed_seconds=time.monotonic() - began,
                )
                if time.monotonic() - started > timeout or time.monotonic() - began > 7200:
                    import signal

                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    raise TimeoutError("Replication budget elapsed")
                time.sleep(15)
        copy_verified(target, destination / target.relative_to(out))
        if child.returncode:
            raise RuntimeError(f"Subprocess exited {child.returncode}; see {target.name}")

    try:
        status("restoring_frozen_actors")
        policy = out / "policy.pt"
        opponent = out / "opponent.pt"
        for source, target, expected in [
            (original / "exports/update_000006.actor.pt", policy, protocol["policy_sha256"]),
            (
                original.parent / "121m_bc_repair_20261104/repaired/bc/bc_latest.pt",
                opponent,
                protocol["opponent_sha256"],
            ),
        ]:
            receipt = copy_verified(source, target)
            if receipt["sha256"] != expected:
                raise ValueError("Frozen policy/opponent checksum mismatch")
        names = list(yaml.safe_load((root / protocol["deck_pool"]).read_text())["decks"])
        evaluations = []
        for block in protocol["blocks"]:
            for start in range(
                protocol["pair_start"],
                protocol["pair_start"] + protocol["pair_count"],
                protocol["chunk_pairs"],
            ):
                directory = out / f"seed_{block['simulator_seed_base']}" / f"pairs_{start:03d}"
                directory.mkdir(parents=True)
                command = [
                    sys.executable,
                    "scripts/evaluate_policy.py",
                    "--policy",
                    str(policy),
                    "--opponent",
                    str(opponent),
                    "--decks",
                    protocol["deck_pool"],
                    "--pairs",
                    str(protocol["chunk_pairs"]),
                    "--pair-start",
                    str(start),
                    "--simulator-seed-base",
                    str(block["simulator_seed_base"]),
                    "--pairing-seed",
                    str(protocol["pairing_seed"]),
                    "--sampling-seed",
                    str(block["sampling_seed"]),
                    "--device",
                    "cuda",
                    "--frame-cache",
                    "--out",
                    str(directory),
                ]
                run(command, directory / "evaluation.log", 900)
                result = json.loads((directory / "evaluation.json").read_text())
                if any(
                    result.get(k) != protocol["evaluation_code_sha256"]
                    for k in ("evaluation_code_sha256", "evaluation_code_file_sha256_at_end")
                ):
                    raise ValueError("Evaluation code changed during replication")
                run(
                    [
                        sys.executable,
                        "scripts/audit_policy_replays.py",
                        "--evaluation",
                        str(directory / "evaluation.json"),
                        "--out",
                        str(directory / "execution_audit.json"),
                        "--java-home",
                        "/usr/lib/jvm/java-21-openjdk-amd64",
                    ],
                    directory / "audit.log",
                    300,
                )
                audit = json.loads((directory / "execution_audit.json").read_text())
                if audit["evaluation_sha256"] != file_sha256(
                    directory / "evaluation.json"
                ) or not all(g["native_outcome_parity"] for g in audit["games"]):
                    raise ValueError("Native replay mismatch")
                for path in directory.iterdir():
                    if path.is_file():
                        copy_verified(path, destination / path.relative_to(out))
                evaluations.append(result)
                status("chunk_backed_up", finished_chunks=len(evaluations), total_chunks=16)
        summary = summarize_repeated_evaluations(protocol, names, evaluations)
        atomic_json(out / "summary.json", summary)
        copy_verified(out / "summary.json", destination / "summary.json")
        atomic_json(
            out / "completed.json",
            dict(
                summary_sha256=file_sha256(out / "summary.json"),
                observed_unix=time.time(),
                games=summary["games"],
            ),
        )
        copy_verified(out / "completed.json", destination / "completed.json")
        status(
            "completed",
            games=summary["games"],
            score=summary["score"],
            evidence_of_improvement=summary["evidence_of_improvement"],
        )
        from google.colab import runtime

        runtime.unassign()
    except BaseException as error:
        if child is not None and child.poll() is None:
            import signal

            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        status("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
