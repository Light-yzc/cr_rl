"""Download and verify all training-essential data from Google Drive for local migration.

According to loc_transfer/README.md and V100_HANDOFF_20261107.md:
1. Complete Checkpoint: update_000142.pt (authoritative baseline) and update_000143.pt
2. Checkpoint metadata & manifests: latest.json, update_000142.manifest.json, update_000142.pool.json, etc.
3. Opponent models: all 8 opponents in candidate/opponents/ (including BC anchor 0)
4. Evaluation probes: probe_00 through probe_04
5. Metadata & configs: protocol.json, command.json, resume_boundaries.json, run_config.json, evaluation_progress.json, all_matches.jsonl, training.log, trainer_exit.json
6. Recent rollouts: candidate/rollouts/
7. Local run preparation: artifacts/local_v100 with ppo_latest.pt and opponent_pool.json
8. Verification: SHA256 integrity check and PyTorch checkpoint load audit
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RCLONE = ROOT / "artifacts/long1000_20261106/rclone.exe"
RCLONE_CONF = ROOT / "artifacts/long1000_20261106/rclone.conf"
REMOTE_RUN = "gdrive:cr_rl/runs/122m_long1000_sample2x_20261106"
TARGET_DIR = ROOT / "artifacts/v100_handoff"
LOCAL_RUN_DIR = ROOT / "artifacts/local_v100"

def rclone_cmd(args):
    cmd = [
        str(RCLONE),
        *args,
        "--config", str(RCLONE_CONF),
        "--stats-one-line",
        "--stats", "10s",
        "--checkers", "8",
        "--transfers", "4",
    ]
    print(f"[{time.strftime('%H:%M:%S')}] RUN: {' '.join(cmd[:6])} ...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr[-1000:]}", flush=True)
        raise RuntimeError(f"rclone failed with code {res.returncode}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {args[0]}", flush=True)
    return res.stdout

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()

def main():
    start_time = time.time()
    print("=== Starting Training-Essential Data Fetch from Google Drive ===", flush=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "weights").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "opponents").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "exports").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "rollouts").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "evaluation").mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "meta").mkdir(parents=True, exist_ok=True)

    # 1. Pull top-level metadata & logs
    print("\n--- Phase 1: Pulling Run Metadata, Configs, Logs ---", flush=True)
    meta_files = [
        "protocol.json", "command.json", "resume_boundaries.json",
        "run_config.json", "evaluation_budget.json", "evaluation_progress.json",
        "all_matches.jsonl", "training.log", "trainer_exit.json",
        "processes.json", "status.json", "probe_ledger.json",
        "reward_stage_v2.json", "reward_stage_runtime.json"
    ]
    for mf in meta_files:
        try:
            rclone_cmd(["copyto", f"{REMOTE_RUN}/{mf}", str(TARGET_DIR / "meta" / mf)])
        except Exception as e:
            print(f"Notice: optional file {mf} not copied: {e}", flush=True)

    # 2. Pull candidate manifests, pools and latest.json
    print("\n--- Phase 2: Pulling Checkpoint Manifests and Pool Metadata ---", flush=True)
    rclone_cmd(["copyto", f"{REMOTE_RUN}/candidate/latest.json", str(TARGET_DIR / "checkpoints/latest.json")])
    for mf in ["update_000142.manifest.json", "update_000142.pool.json",
               "update_000144.manifest.json", "update_000144.pool.json"]:
        rclone_cmd(["copyto", f"{REMOTE_RUN}/candidate/checkpoints/{mf}", str(TARGET_DIR / "checkpoints" / mf)])

    # 3. Pull Opponent Models (All 8 opponents)
    print("\n--- Phase 3: Pulling Opponent Pool Models ---", flush=True)
    rclone_cmd(["copy", f"{REMOTE_RUN}/candidate/opponents", str(TARGET_DIR / "opponents")])

    # 4. Pull Actor Exports
    print("\n--- Phase 4: Pulling Actor Exports ---", flush=True)
    rclone_cmd(["copy", f"{REMOTE_RUN}/candidate/exports", str(TARGET_DIR / "exports")])

    # 5. Pull Rollouts (Recent candidate rollouts)
    print("\n--- Phase 5: Pulling Rollouts ---", flush=True)
    rclone_cmd(["copy", f"{REMOTE_RUN}/candidate/rollouts", str(TARGET_DIR / "rollouts")])

    # 6. Pull Full Checkpoints (update 142 authoritative + update 143)
    print("\n--- Phase 6: Pulling Checkpoints update_000142.pt and update_000143.pt ---", flush=True)
    # Remove any stale partial download if exists
    for partial in (TARGET_DIR / "weights").glob("*.partial"):
        print(f"Removing stale partial file: {partial.name}", flush=True)
        partial.unlink()

    rclone_cmd(["copyto", f"{REMOTE_RUN}/candidate/checkpoints/update_000142.pt",
                str(TARGET_DIR / "weights/update_000142.pt")])
    rclone_cmd(["copyto", f"{REMOTE_RUN}/candidate/checkpoints/update_000143.pt",
                str(TARGET_DIR / "weights/update_000143.pt")])

    # 7. Pull Evaluation Probes
    print("\n--- Phase 7: Pulling Evaluation Probes ---", flush=True)
    rclone_cmd(["copy", f"{REMOTE_RUN}/evaluation", str(TARGET_DIR / "evaluation")])

    print("\n--- Phase 8: Verifying Checksums and Integrity ---", flush=True)
    manifest = json.loads((TARGET_DIR / "checkpoints/latest.json").read_text(encoding="utf-8"))
    files_map = {f["path"]: f for f in manifest["files"]}

    # Verify update 142
    rec142 = files_map.get("checkpoints/update_000142.pt")
    if rec142:
        path142 = TARGET_DIR / "weights/update_000142.pt"
        size142 = path142.stat().st_size
        print(f"Checking update_000142.pt size: {size142} (expected {rec142['bytes']})", flush=True)
        assert size142 == rec142["bytes"], f"Size mismatch: {size142} vs {rec142['bytes']}"
        sha142 = sha256_file(path142)
        print(f"Checking update_000142.pt SHA256: {sha142}", flush=True)
        assert sha142 == rec142["sha256"], f"SHA mismatch: {sha142} vs {rec142['sha256']}"
        print("update_000142.pt: SHA256 PASS", flush=True)

    # Verify opponents
    for opp_rec in [f for f in manifest["files"] if f["path"].startswith("opponents/")]:
        opp_path = TARGET_DIR / opp_rec["path"]
        assert opp_path.exists(), f"Missing opponent: {opp_path}"
        assert opp_path.stat().st_size == opp_rec["bytes"], f"Opponent size mismatch: {opp_path}"
        opp_sha = sha256_file(opp_path)
        assert opp_sha == opp_rec["sha256"], f"Opponent SHA mismatch: {opp_path}"
        print(f"Opponent {opp_path.name}: SHA256 PASS", flush=True)

    # 9. Verify Checkpoint Loadability with PyTorch
    print("\n--- Phase 9: PyTorch Checkpoint State Audit ---", flush=True)
    import torch
    state142 = torch.load(TARGET_DIR / "weights/update_000142.pt", map_location="cpu", weights_only=False, mmap=True)
    required_keys = ["model", "critic", "optimizer", "critic_optimizer", "torch_rng_state",
                     "numpy_rng_state", "training_game_counter", "reward_config", "config"]
    missing = [k for k in required_keys if k not in state142]
    assert not missing, f"Missing required keys in update_000142.pt: {missing}"
    print(f"update_000142.pt verified:")
    print(f"  Updates: {state142.get('updates')}")
    print(f"  Completed games: {state142['training_game_counter'].get('completed_games')}")
    print(f"  Model keys: {len(state142['model'])}, Critic keys: {len(state142['critic'])}")
    print(f"  Reward config: {state142['reward_config']}")
    del state142

    state143 = torch.load(TARGET_DIR / "weights/update_000143.pt", map_location="cpu", weights_only=False, mmap=True)
    missing143 = [k for k in required_keys if k not in state143]
    assert not missing143, f"Missing required keys in update_000143.pt: {missing143}"
    print(f"update_000143.pt verified:")
    print(f"  Updates: {state143.get('updates')}")
    print(f"  Completed games: {state143['training_game_counter'].get('completed_games')}")
    del state143

    # 10. Prepare artifacts/local_v100 ready-to-run directory
    print("\n--- Phase 10: Initializing artifacts/local_v100 ---", flush=True)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_RUN_DIR / "opponents").mkdir(parents=True, exist_ok=True)
    
    # Copy or hardlink opponents into local_v100
    for opp_file in (TARGET_DIR / "opponents").glob("*.pt"):
        dst = LOCAL_RUN_DIR / "opponents" / opp_file.name
        if not dst.exists():
            try:
                os.link(opp_file, dst)
            except Exception:
                shutil.copy2(opp_file, dst)
        print(f"Linked opponent to {dst}", flush=True)

    # Copy update_000142.pool.json and adjust paths for local_v100
    pool_data = json.loads((TARGET_DIR / "checkpoints/update_000142.pool.json").read_text(encoding="utf-8"))
    for entry in pool_data["entries"]:
        name = Path(entry["checkpoint"]).name
        entry["checkpoint"] = str((LOCAL_RUN_DIR / "opponents" / name).resolve())
    (LOCAL_RUN_DIR / "opponent_pool.json").write_text(json.dumps(pool_data, indent=2), encoding="utf-8")
    print(f"Generated {LOCAL_RUN_DIR / 'opponent_pool.json'}", flush=True)

    # Link update_000142.pt as ppo_latest.pt
    target_checkpoint = LOCAL_RUN_DIR / "ppo_latest.pt"
    src_checkpoint = TARGET_DIR / "weights/update_000142.pt"
    if target_checkpoint.exists():
        target_checkpoint.unlink()
    try:
        os.link(src_checkpoint, target_checkpoint)
        print("Hardlinked ppo_latest.pt from update_000142.pt", flush=True)
    except Exception:
        shutil.copy2(src_checkpoint, target_checkpoint)
        print("Copied ppo_latest.pt from update_000142.pt", flush=True)

    elapsed = time.time() - start_time
    print(f"\n=== All Training Essentials Fetched and Verified Successfully in {elapsed:.1f}s ===", flush=True)

if __name__ == "__main__":
    main()
