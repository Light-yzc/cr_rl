"""Verify the migrated full checkpoint and stage a separate local restore directory."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "artifacts/v100_handoff"
out = Path(os.environ["RUN_DIR"]).resolve()
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit("Exactly two CUDA devices must be visible (GPU_IDS=1,2 on this host)")
for i in range(2):
    if torch.cuda.get_device_capability(i) != (7, 0):
        raise SystemExit(f"Device {i} is not a V100/Volta: {torch.cuda.get_device_name(i)}")
if not torch.cuda.can_device_access_peer(0, 1):
    raise SystemExit("CUDA peer access between the V100s is unavailable")
print("V100 pair ready:", torch.__version__, torch.cuda.get_arch_list(), flush=True)
if (out / "ppo_latest.pt").exists():
    if not (out / "opponent_pool.json").exists():
        raise SystemExit("Local restore is missing opponent_pool.json")
else:
    manifest = json.loads((source / "checkpoints/latest.json").read_text())
    records = {r["path"]: r for r in manifest["files"]}
    checkpoint = source / "weights" / Path(manifest["checkpoint"]).name
    if not checkpoint.exists():
        raise SystemExit(f"Full checkpoint missing: {checkpoint}. See loc_transfer/DOWNLOADS.md")
    record = records[manifest["checkpoint"]]
    if checkpoint.stat().st_size != record["bytes"]:
        raise SystemExit("Checkpoint is incomplete (size mismatch)")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise SystemExit("Checkpoint SHA256 mismatch")
    pool_file = source / manifest["pool"]
    if hashlib.sha256(pool_file.read_bytes()).hexdigest() != manifest["pool_sha256"]:
        raise SystemExit("Opponent pool SHA256 mismatch")
    pool = json.loads(pool_file.read_text())
    for entry in pool["entries"]:
        path = (source / "opponents" / Path(entry["checkpoint"]).name).resolve()
        expected = records["opponents/" + path.name]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise SystemExit(f"Opponent SHA256 mismatch: {path}")
        entry["checkpoint"] = str(path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "opponent_pool.json").write_text(json.dumps(pool, indent=2))
    # Same filesystem: preserve the immutable source and avoid a 2.72 GiB copy.
    staged = out / ".resume-import.tmp.pt"
    staged.unlink(missing_ok=True)
    try:
        os.link(checkpoint.resolve(), staged)
    except OSError:
        shutil.copy2(checkpoint, staged)
    staged.replace(out / "ppo_latest.pt")
    (out / "migration_receipt.json").write_text(
        json.dumps(
            {
                "source_update": manifest["update"],
                "checkpoint_sha256": digest,
                "semantics": (
                    "model_optimizer_restored_simulator_reset_hardware_and_precision_changed"
                ),
                "source": str(checkpoint),
            },
            indent=2,
        )
    )
saved = torch.load(out / "ppo_latest.pt", map_location="cpu", mmap=True, weights_only=False)
boundary_path = out / "resume_boundaries.json"
boundaries = json.loads(boundary_path.read_text()) if boundary_path.exists() else []
boundaries.append(
    {
        "time_unix": time.time(),
        "source_update": saved["updates"],
        "completed_games": saved["training_game_counter"]["completed_games"],
        "discarded_unfinished_episode_steps": saved.get("unfinished_episode_steps", []),
        "simulator_reset": True,
        "unfinished_episodes_counted": False,
        "visible_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": os.environ.get("AMP_DTYPE", "fp16"),
        "reason": os.environ.get("CRRL_RESUME_REASON", "local_resume"),
    }
)
temporary = boundary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(boundaries, indent=2))
temporary.replace(boundary_path)
print("Resume directory:", out, flush=True)
