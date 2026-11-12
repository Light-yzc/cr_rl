"""Isolated collection-only probe; never publishes weights or changes production counters."""

import argparse
import json
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, required=True)
parser.add_argument("--workers", type=int, required=True)
parser.add_argument("--torch-threads", type=int, required=True)
parser.add_argument("--dual", action="store_true")
parser.add_argument("--steps", type=int, default=768)
options = parser.parse_args()
run = root / "artifacts/local_dual_v100"
name = f"e{options.envs}_w{options.workers}_t{options.torch_threads}_dual{int(options.dual)}"
out = root / "artifacts/local_dual_v100_checks/rollout_benchmark" / name
out.mkdir(parents=True, exist_ok=True)
pool = json.loads((run / "opponent_pool.json").read_text())
for entry in pool["entries"]:
    path = Path(entry["checkpoint"])
    entry["checkpoint"] = str((run / path).resolve())
    assert Path(entry["checkpoint"]).exists()
(out / "opponent_pool.json").write_text(json.dumps(pool, indent=2))


def probe(args):
    from cr_rl.training.selfplay_v2 import PersistentSelfplay

    runner = PersistentSelfplay(args)
    original_step = runner.env.step
    simulator_seconds = [0.0]

    def timed_step(actions):
        began = time.perf_counter()
        original_step(actions)
        simulator_seconds[0] += time.perf_counter() - began

    runner.env.step = timed_step
    try:
        # Both candidates warm the same 128 decisions before measured collection.
        args.steps = 128
        warm = runner.collect()
        del warm
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        for i in (0, 1):
            torch.cuda.reset_peak_memory_stats(i)
        simulator_seconds[0] = 0
        args.steps = options.steps
        cpu_began = time.process_time()
        began = time.perf_counter()
        snapshots, _, completed, _, _ = runner.collect()
        elapsed = time.perf_counter() - began
        cpu_seconds = time.process_time() - cpu_began
        eligible = np.stack(
            [row["learner_mask"].astype(bool) & row["valid"].astype(bool) for row in snapshots]
        )
        masks = np.stack([row["masks"] for row in snapshots])
        actions = np.stack([row["actions"] for row in snapshots])
        assert np.take_along_axis(masks, actions[..., None], -1).all()
        assert all(np.isfinite(row["old_log_probs"]).all() for row in snapshots)
        assert all(not row["fault_codes"].any() for row in snapshots)
        choices = int((eligible & (masks.sum(-1) > 1)).sum())
        result = {
            "name": name,
            "source_update": runner.update,
            "envs": options.envs,
            "workers": options.workers,
            "torch_threads": options.torch_threads,
            "dual_rollout": options.dual,
            "steps": options.steps,
            "seconds": elapsed,
            "simulator_seconds": simulator_seconds[0],
            "process_cpu_seconds": cpu_seconds,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "host_memory_kib": {
                line.split(":")[0]: int(line.split()[1])
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith(("VmRSS:", "VmSwap:"))
            },
            "new_learner_rows": int(eligible.sum()),
            "choice_rows": choices,
            "learner_rows_per_second": float(eligible.sum() / elapsed),
            "choice_rows_per_second": choices / elapsed,
            "completed_probe_games": len(completed),
            "peak_allocated_gib": {
                str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in (0, 1)
            },
            "rollout_parallel": runner.rollout_parallel,
            "scope": (
                "Collection only; no optimizer/weight publication; "
                "probe games excluded from training counter"
            ),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        runner.close()


import cr_rl.training.selfplay_v2 as training  # noqa: E402
from scripts.train_selfplay import main  # noqa: E402

training.run = probe
launcher = (root / "loc_transfer/run_v100.sh").read_text()
launcher = launcher.replace(
    'ROOT="$(cd "$(dirname "$0")/.." && pwd)"', "ROOT=" + shlex.quote(str(root))
)
launcher = launcher.replace('"$PYTHON" loc_transfer/prepare_resume.py', "")
print_command = (
    '"$PYTHON" -c '
    + shlex.quote("import json,sys; print(json.dumps(sys.argv[1:]))")
    + ' "${ARGS[@]}"'
)
launcher = launcher.replace(
    'exec "$PYTHON" -u scripts/train_selfplay.py "${ARGS[@]}" "$@"', print_command
)
source_args = json.loads(subprocess.check_output(["bash"], input=launcher, text=True, cwd=root))
source_args = [argument for argument in source_args if argument != "--dual-rollout"]
sys.argv = [
    "train_selfplay",
    *source_args,
    "--out",
    str(out),
    "--envs",
    str(options.envs),
    "--env-threads",
    str(options.workers),
    "--torch-threads",
    str(options.torch_threads),
    "--torch-interop-threads",
    "1",
    "--opponent-cache-size",
    str(max(2, options.envs)),
]
if options.dual:
    sys.argv.append("--dual-rollout")
main()
