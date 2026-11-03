"""Replay native outcomes using the pinned simulator; no training data is downloaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cr_rl.evaluation.paired import paired_outcomes  # noqa: E402
from cr_rl.evaluation.replay_audit import summarize_executions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--simulator-cache", type=Path, default=ROOT / ".cache/replay-viewer")
    parser.add_argument("--java-home", type=Path)
    args = parser.parse_args()
    cache = args.simulator_cache.resolve()
    marker = cache / "UPSTREAM_COMMIT"
    if marker.exists():
        commit = marker.read_text().strip()
        library = cache / "crforge/desktop/build/install/desktop/lib"
        tables = cache / "tables/16.635.3"
    else:
        # Colab already has the headless simulator; no desktop bundle is needed.
        vendor = ROOT / ".cache/vendor/crforge"
        commit = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
        ).strip()
        library = vendor / "env/build/install/env/lib"
        tables = Path.home() / ".crforge/tables/16.635.3"
    if commit != "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6":
        raise ValueError("Audit requires the pinned simulator revision")
    if not library.is_dir():
        raise ValueError("Build the pinned simulator distribution before auditing")
    java = str(args.java_home / "bin/java") if args.java_home else shutil.which("java")
    javac = str(args.java_home / "bin/javac") if args.java_home else shutil.which("javac")
    if not java or not javac:
        raise ValueError("Java 21 JDK is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    classes = args.out.parent / ".audit_classes"
    classes.mkdir(exist_ok=True)
    subprocess.run(
        [
            javac,
            "-cp",
            str(library / "*"),
            "-d",
            str(classes),
            str(ROOT / "scripts/java/ReplayExecutionAudit.java"),
        ],
        check=True,
    )
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8-sig"))
    pair_summary = paired_outcomes(evaluation)
    paths = [
        args.evaluation.parent / f"match_{i:03d}.json" for i in range(len(evaluation["matchups"]))
    ]
    native_path = args.out.with_suffix(".native.json")
    subprocess.run(
        [
            java,
            "-cp",
            os.pathsep.join((str(classes), str(library / "*"))),
            "ReplayExecutionAudit",
            str(tables),
            str(native_path),
            *map(str, paths),
        ],
        check=True,
    )
    native = json.loads(native_path.read_text())
    games = []
    for index, (match, execution) in enumerate(zip(evaluation["matchups"], native, strict=True)):
        seat = int(match["learner_seat"])
        games.append(
            {
                "pair": match["pair"],
                "learner_seat": seat,
                "reward": match["reward"],
                "duration_seconds": execution["duration_seconds"],
                "active_simulation_seconds": execution["active_simulation_seconds"],
                "active_overtime_seconds": execution["active_overtime_seconds"],
                "active_ticks_by_elixir_full_bar_ms": execution[
                    "active_ticks_by_elixir_full_bar_ms"
                ],
                "learner": summarize_executions(execution, seat),
                "opponent": summarize_executions(execution, 1 - seat),
                "decision_audit": match.get("decision_audit"),
                "native_outcome_parity": execution["native_outcome_parity"],
                "replay_sha256": hashlib.sha256(paths[index].read_bytes()).hexdigest(),
            }
        )
    result = {
        "evaluation": str(args.evaluation.resolve()),
        "evaluation_sha256": hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
        "paired": pair_summary,
        "simulator_commit": commit,
        "games": games,
        "limitation": (
            "Replay requests do not reveal unrecorded WAIT decisions; "
            "use decision_audit when available."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "paired": pair_summary}, indent=2))


if __name__ == "__main__":
    main()
