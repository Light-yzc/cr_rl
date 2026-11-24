"""Start Dual-V100 v3 training in a memory-bounded user service, excluding RTX 4090."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(os.environ.get("RUN_DIR", root / "artifacts/ppo_v3_v100")).resolve()
out.mkdir(parents=True, exist_ok=True)
pid_path = out / "trainer.pid"
unit = "crrl-v100-v3.service"

active_state = subprocess.run(
    ["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
if active_state in {"active", "activating", "reloading"}:
    raise SystemExit(f"Trainer service already running: {unit} ({active_state})")
if active_state == "failed":
    subprocess.run(["systemctl", "--user", "reset-failed", unit], check=False)
if pid_path.exists():
    raw_pid = pid_path.read_text(encoding="utf-8").strip()
    if raw_pid.isdigit():
        pid = int(raw_pid)
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and b"train_selfplay" in command.read_bytes():
            raise SystemExit(f"Trainer already running: PID {pid}")

gpu_ids = os.environ.get("GPU_IDS", "2,1")
if "0" in [item.strip() for item in gpu_ids.split(",") if item.strip()]:
    raise SystemExit(
        f"Refusing GPU_IDS={gpu_ids}: RTX 4090 (GPU 0) is excluded; use Dual-V100 (2,1)."
    )

environment = dict(os.environ, RUN_DIR=str(out), GPU_IDS=gpu_ids)
allowed = (
    "RUN_DIR",
    "PYTHON",
    "JAVA_HOME",
    "GPU_IDS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ENVS",
    "ENV_THREADS",
    "STEPS",
    "UPDATES",
    "TARGET_UPDATES",
    "EPOCHS",
    "CRITIC_EPOCHS",
    "BATCH",
    "ACTOR_BATCH",
    "CRITIC_BATCH",
    "ACCUM",
    "ACTOR_EFF",
    "CRITIC_EFF",
    "TBPTT_CHUNK",
    "FULL_EPISODE_BPTT",
    "TEMPORAL_CHECKPOINT_MODE",
    "FRAME_MICROBATCH",
    "CACHE_STATE_FRAMES",
    "FOCAL_DECKS", "FOCAL_DECK_PROB", "ARMY_REWARD",
    "HISTORICAL_COHORT_SIZE", "HISTORICAL_COHORT_UPDATES",
    "LEARNING_RATE",
    "CRITIC_LEARNING_RATE",
    "GAMMA",
    "GAE_LAMBDA",
    "CRITIC_GAE_LAMBDA",
    "CRITIC_WARMUP_UPDATES",
    "CRITIC_WARMUP_GAMES",
    "AMP_DTYPE",
    "PRECISION",
    "CHECKPOINT",
    "RESUME",
    "BC_ANCHOR",
    "DECKS",
    "POOL_CAPACITY",
    "HISTORY_SNAPSHOT_INTERVAL",
    "OPPONENT_CACHE_SIZE",
    "CRRL_OPPONENT_DISK_CACHE",
    "TRAINING_OPPONENT_CACHE_SIZE",
    "ROLLOUT_SAVE_INTERVAL",
    "KEEP_ROLLOUTS",
    "CHECKPOINT_INTERVAL",
    "KEEP_CHECKPOINTS",
    "KEEP_PERIODIC_CHECKPOINTS",
    "VALIDATION_INTERVAL",
    "VALIDATION_PAIRS",
    "VALIDATION_DECKS",
    "MIN_TRAINING_GAMES",
    "TENSORBOARD_DIR",
    "PROGRESS_JSONL",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CRRL_RESUME_REASON",
    "REFERENCE_KL_COEF",
    "ANCHOR_KL_COEF",
    "CARD_ENTROPY_COEF",
    "TILE_ENTROPY_COEF",
    "ADAPTIVE_ACTOR_LR_MIN",
    "ADAPTIVE_ACTOR_LR_MAX",
    "MAX_INITIAL_LOGP_ERROR",
    "PPO_ACTOR_FP32",
)
command = [
    "systemd-run",
    "--user",
    "--unit=" + unit,
    "--collect",
    "--service-type=exec",
    "--working-directory=" + str(root),
    "--property=MemoryHigh=" + os.environ.get("MEMORY_HIGH", "10.5G"),
    "--property=MemoryMax=" + os.environ.get("MEMORY_MAX", "12G"),
    "--property=MemorySwapMax=" + os.environ.get("MEMORY_SWAP_MAX", "2G"),
    "--property=OOMPolicy=stop",
    "--property=KillMode=control-group",
    "--property=TimeoutStopSec=" + os.environ.get("STOP_TIMEOUT", "600"),
    "--property=Restart=" + os.environ.get("RESTART_POLICY", "on-failure"),
    "--property=RestartSec=" + os.environ.get("RESTART_SEC", "15s"),
    "--property=StartLimitIntervalSec=" + os.environ.get("START_LIMIT_INTERVAL_SEC", "900"),
    "--property=StartLimitBurst=" + os.environ.get("START_LIMIT_BURST", "6"),
    "--property=StandardOutput=append:" + str(out / "train.log"),
    "--property=StandardError=append:" + str(out / "train.log"),
]
for key in allowed:
    if key in environment:
        command.append("--setenv=" + key + "=" + environment[key])
command += ["bash", str(root / "loc_transfer/run_v100_v3.sh"), *sys.argv[1:]]
subprocess.run(command, check=True)
for _ in range(20):
    pid = int(
        subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True
        ).strip()
        or "0"
    )
    if pid:
        break
    time.sleep(0.1)
else:
    raise SystemExit("Trainer service failed to start; inspect journalctl --user -u " + unit)
pid_path.write_text(str(pid) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "pid": pid,
            "unit": unit,
            "gpu_ids": gpu_ids,
            "run_dir": str(out),
            "memory_high": os.environ.get("MEMORY_HIGH", "10.5G"),
            "memory_max": os.environ.get("MEMORY_MAX", "12G"),
            "swap_max": os.environ.get("MEMORY_SWAP_MAX", "2G"),
            "log": str(out / "train.log"),
        }
    )
)
