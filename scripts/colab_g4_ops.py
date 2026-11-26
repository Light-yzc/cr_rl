"""Cross-machine Colab G4 / A100 / L4 operations CLI for Specialist v3 PPO + Heavy-MCTS.

Provides token-refreshed attachment to an existing Google Colab GPU instance (never
creating a new instance unprompted unless run inside Colab via `restore-ssd`),
15-minute keepalive & convergence monitoring, graceful start/stop/restart, SSD
restoration from Google Drive, and 16-stream parallel SHA256-verified Actor downloads.

Usage (run with `google-colab-cli`'s Python environment or any env with `colab-cli`):
    python scripts/colab_g4_ops.py status
    python scripts/colab_g4_ops.py watch --interval 900
    python scripts/colab_g4_ops.py download-actors --out artifacts/g4_heavy_mcts
    python scripts/colab_g4_ops.py start
    python scripts/colab_g4_ops.py stop
    python scripts/colab_g4_ops.py restart
    python scripts/colab_g4_ops.py exec path/to/remote_snippet.py --timeout 90
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib


DEFAULT_RUN_NAME = "longrun_sparky_value_bc_v2"


def _connect_existing_colab():
    """Attach to the highest-tier existing Colab runtime (G4 > A100 > L4) without allocating a new VM."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime
    from colab_cli.state import SessionState

    retry_adapter = HTTPAdapter(
        max_retries=Retry(total=5, backoff_factor=1.2, status_forcelist=[500, 502, 503, 504]),
        pool_connections=32,
        pool_maxsize=32,
    )
    http = requests.Session()
    http.mount("https://", retry_adapter)
    state.client.session.mount("https://", retry_adapter)

    def _ssl_retry(fn, attempts: int = 6):
        for i in range(attempts):
            try:
                return fn()
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                if i == attempts - 1:
                    raise
                time.sleep(1.5 * (i + 1))

    assignments = _ssl_retry(lambda: state.client.list_assignments())
    if not assignments:
        return None

    rank = {"G4": 0, "A100": 1, "L4": 2, "T4": 3}
    chosen = sorted(
        assignments,
        key=lambda x: rank.get(getattr(x.accelerator, "value", str(x.accelerator)), 9),
    )[0]
    acc_val = getattr(chosen.accelerator, "value", str(chosen.accelerator))
    rpi = chosen.runtime_proxy_info
    sname = f"cr-{acc_val.lower()}"
    base_url = rpi.url.rstrip("/")
    headers = {"X-Colab-Runtime-Proxy-Token": rpi.token, "X-Colab-Client-Agent": "vscode"}
    params = {"authuser": "0", "colab-runtime-proxy-token": rpi.token}

    prev = state.store.get(sname)
    sessions = _ssl_retry(
        lambda: http.get(base_url + "/api/sessions", headers=headers, params=params, timeout=20).json()
    )
    live_ids = [(x.get("kernel") or {}).get("id") for x in sessions if (x.get("kernel") or {}).get("id")]
    if prev and prev.endpoint == chosen.endpoint and prev.kernel_id in live_ids:
        kid = prev.kernel_id
    elif live_ids:
        kid = live_ids[-1]
    else:
        created = _ssl_retry(
            lambda: http.post(
                base_url + "/api/sessions",
                headers=headers,
                params=params,
                json={
                    "path": "g4_work.ipynb",
                    "type": "notebook",
                    "name": "g4_work.ipynb",
                    "kernel": {"name": "python3"},
                },
                timeout=25,
            ).json()
        )
        kid = created["kernel"]["id"]

    sess = SessionState(
        name=sname,
        token=rpi.token,
        url=rpi.url,
        endpoint=chosen.endpoint,
        token_expires_at=rpi.expires_at(),
        variant="GPU",
        accelerator=acc_val,
        machine_shape=chosen.machine_shape.name,
        kernel_id=kid,
    )
    state.store.add(sess)

    def _exec(code: str, timeout_sec: int = 60) -> str:
        for attempt in range(1, 5):
            try:
                rt = ColabRuntime(
                    sess.url,
                    sess.token,
                    session_name=sess.name,
                    history=state.history,
                    kernel_id=sess.kernel_id,
                )
                res = rt.execute_code(code, allow_stdin=False, timeout=timeout_sec)
                if hasattr(res, "stdout"):
                    out = res.stdout or ""
                    err = getattr(res, "stderr", "") or ""
                    return out + (f"\nSTDERR: {err}" if err else "")
                if isinstance(res, list):
                    parts = []
                    for item in res:
                        if isinstance(item, dict):
                            txt = (
                                item.get("text")
                                or item.get("data", {}).get("text/plain")
                                or item.get("traceback")
                                or ""
                            )
                            if isinstance(txt, list):
                                txt = "\n".join(txt)
                            if txt:
                                parts.append(txt)
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                return str(res)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * attempt)
        return ""

    return {
        "session": sess,
        "http": http,
        "base_url": base_url,
        "headers": headers,
        "params": params,
        "exec": _exec,
    }


def _build_status_probe(run_name: str) -> str:
    return f'''
import json, shutil, subprocess
from pathlib import Path

run_name = {run_name!r}
base = Path("/content/crrl_training/runs") / run_name
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader"],
        text=True, timeout=5
    ).strip()
except Exception as e:
    gpu = f"err: {{e}}"
print("=== GPU & HEARTBEAT ===")
print(gpu)

pids = []
for p in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        cmd = p.read_bytes().decode("utf-8", "replace").replace("\\x00", " ")
        if "train_selfplay.py" in cmd or "cr_train_pipeline.py" in cmd:
            pids.append((int(p.parent.name), cmd[:160]))
    except Exception:
        pass
print("ACTIVE_PROCS:", pids)

by_u = {{}}
for fn in ("metrics.jsonl", "progress.jsonl"):
    fp = base / fn
    if fp.exists():
        with fp.open("rb") as f:
            sz = fp.stat().st_size
            f.seek(max(0, sz - 2_500_000))
            chunk = f.read().decode("utf-8", "replace")
        for line in chunk.splitlines():
            if '"state_ppo"' in line and '"update"' in line:
                try:
                    d = json.loads(line)
                    if d.get("phase") == "state_ppo" and "update" in d:
                        by_u[int(d["update"])] = d
                except Exception:
                    pass
        if by_u:
            break

updates = sorted(by_u)
print(f"\\n=== RECENT UPDATES ({{updates[0] if updates else '-'}}..{{updates[-1] if updates else '-'}}) ===")
for u in updates[-8:]:
    d = by_u[u]
    m = d.get("metrics") or {{}}
    tw = d.get("training_win_rates") or {{}}
    ebc = d.get("expert_bc") or {{}}
    ms = d.get("macro_search") or {{}}
    ld = d.get("lookahead_distill") or {{}}
    dc = d.get("decision_coverage") or {{}}
    ms_str = f"{{ms.get('accepted')}}(adm={{ms.get('admitted_this_phase')}}/{{ms.get('buffer_size')}},{{d.get('macro_search_seconds',0):.1f}}s)" if ms.get("ran") else "-"
    ld_str = f"{{ld.get('accepted')}}(adm={{ld.get('admitted_this_phase')}}/{{ld.get('buffer_size')}},ovr={{ld.get('total_overrides')}},{{d.get('lookahead_distill_seconds',0):.1f}}s)" if ld.get("ran") else "-"
    cost_mean = dc.get("requested_card_cost_mean", 0.0)
    ge5 = (dc.get("at_least_five_cost_requests", 0) / max(1, dc.get("card_requests", 1))) * 100.0
    print(
        f"u={{u}}: sps={{d.get('samples_per_second',0):.1f}} (roll={{d.get('rollout_samples_per_second',0):.1f}}, ppo={{d.get('ppo_optimization_seconds',0):.1f}}s, vram={{d.get('peak_cuda_allocated_gib',0):.1f}}G) "
        f"| win={{tw.get('overall_win_rate',0)*100:.1f}}% | EV={{m.get('explained_variance',0):.3f}} "
        f"| ent={{m.get('entropy',0):.3f}}({{m.get('card_entropy',0):.2f}}/{{m.get('tile_entropy',0):.2f}}) | akl={{m.get('anchor_kl',0):.3f}} "
        f"| cost={{cost_mean:.2f}}(>=5:{{ge5:.1f}}%) | ebc={{ebc.get('accepted') if ebc.get('ran') else '-'}} | ms={{ms_str}} | ld={{ld_str}}"
    )

val_dir = base / "validation"
if val_dir.exists():
    vdirs = sorted(val_dir.glob("update_*"))[-4:]
    if vdirs:
        print("\\n=== RECENT VALIDATIONS ===")
        for vd in vdirs:
            sp = vd / "summary.json"
            if sp.exists():
                v = json.loads(sp.read_text())
                parts = []
                for side in ("bc", "history"):
                    s = dict(v.get(side) or {{}})
                    parts.append(f"{{side}}={{s.get('wins')}}/{{s.get('games')}} ({{s.get('win_rate',0)*100:.1f}}%)")
                print(vd.name + ": " + " | ".join(parts))

drv_latest = Path("/content/drive/MyDrive/cr_rl_ppo/training") / run_name / "latest.json"
if drv_latest.is_file():
    print("\\n=== DRIVE_LATEST ===")
    print(drv_latest.read_text().strip())

bv_local = base / "best_validation.json"
if bv_local.is_file():
    bv_drive_dir = Path("/content/drive/MyDrive/cr_rl_ppo/training") / run_name / "best_validation"
    bv_drive_meta = bv_drive_dir / "best_validation.json"
    loc_meta = json.loads(bv_local.read_text())
    drv_u = json.loads(bv_drive_meta.read_text()).get("update") if bv_drive_meta.is_file() else None
    if drv_u != loc_meta.get("update") or not (bv_drive_dir / "actor_best_val.pt").is_file():
        bv_drive_dir.mkdir(parents=True, exist_ok=True)
        for fn in ("actor_best_val.pt", "ppo_best_val.pt", "best_validation.json"):
            src = base / fn
            if src.is_file():
                tmp = bv_drive_dir / (fn + ".partial")
                shutil.copyfile(src, tmp)
                tmp.replace(bv_drive_dir / fn)
    print(f"=== BEST_VALIDATION: u={{loc_meta.get('update')}} bc={{loc_meta.get('bc_win_rate',0)*100:.1f}}% history={{loc_meta.get('history_win_rate',0)*100:.1f}}% ===")
'''


def cmd_status(args) -> int:
    conn = _connect_existing_colab()
    if conn is None:
        print("COLAB_INSTANCE_CLOSED_STOP_MONITORING", flush=True)
        return 2
    sess = conn["session"]
    print(f"[keepalive] kernel heartbeat ok on {sess.name} ({sess.endpoint})", flush=True)
    out = conn["exec"](_build_status_probe(args.run), timeout_sec=65)
    print(out, flush=True)
    return 0


def cmd_watch(args) -> int:
    interval = max(60, int(args.interval))
    print(f"Starting Colab keepalive & monitor loop every {interval}s (Ctrl+C to stop)...", flush=True)
    while True:
        rc = cmd_status(args)
        if rc == 2:
            print("Colab instance disconnected/closed; exiting monitor without allocating a new VM.", flush=True)
            return 0
        time.sleep(interval)


def cmd_exec(args) -> int:
    conn = _connect_existing_colab()
    if conn is None:
        print("NO_EXISTING_COLAB_INSTANCE", flush=True)
        return 2
    code = Path(args.script).read_text(encoding="utf-8")
    print(conn["exec"](code, timeout_sec=int(args.timeout)), flush=True)
    return 0


def cmd_pipeline_ctl(args, action: str) -> int:
    conn = _connect_existing_colab()
    if conn is None:
        print("NO_EXISTING_COLAB_INSTANCE", flush=True)
        return 2
    if action == "restart":
        snippet = """
import subprocess, sys, time
from pathlib import Path
root = Path('/content/crrl_training')
subprocess.run([sys.executable, str(root / 'scripts/cr_train_pipeline.py'), 'stop'], check=False)
for _ in range(90):
    alive = [p.parent.name for p in Path('/proc').glob('[0-9]*/cmdline')
             if any(x in p.read_bytes() for x in (b'train_selfplay.py', b'cr_train_pipeline.py'))]
    if not alive:
        break
    time.sleep(2)
res = subprocess.run([sys.executable, str(root / 'scripts/cr_train_pipeline.py'), 'start'], capture_output=True, text=True, check=True)
print(res.stdout, res.stderr)
"""
    else:
        snippet = f"""
import subprocess, sys
from pathlib import Path
root = Path('/content/crrl_training')
res = subprocess.run([sys.executable, str(root / 'scripts/cr_train_pipeline.py'), {action!r}], capture_output=True, text=True, check=True)
print(res.stdout, res.stderr)
"""
    print(conn["exec"](snippet, timeout_sec=240), flush=True)
    return 0


def cmd_restore_and_start(args) -> int:
    conn = _connect_existing_colab()
    if conn is None:
        print("NO_EXISTING_COLAB_INSTANCE", flush=True)
        return 2
    remote_code = f'''
from concurrent.futures import ThreadPoolExecutor
import importlib.util, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

t0 = time.perf_counter()
DRIVE = Path("/content/drive/MyDrive/cr_rl_ppo")
if not DRIVE.is_dir():
    raise RuntimeError("Mount Google Drive at /content/drive first")

RUN_NAME = {args.run!r}
ROOT = Path("/content/crrl_training")
PREP = ROOT / "prepared"
RUN_DIR = ROOT / "runs" / RUN_NAME
BC_PLAN_B = Path("/content/crrl_sparky_plan_b")
ROOT.mkdir(parents=True, exist_ok=True)
PREP.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

pkgs = {{"yaml": "PyYAML", "jpype": "jpype1", "gymnasium": "gymnasium", "pettingzoo": "pettingzoo", "tqdm": "tqdm", "tensorboard": "tensorboard", "zstandard": "zstandard"}}
missing = [v for k, v in pkgs.items() if importlib.util.find_spec(k) is None]
pip_proc = subprocess.Popen([sys.executable, "-m", "pip", "install", "--quiet", *missing]) if missing else None

def _copy_atomic(src: Path, dst: Path):
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    sz = src.stat().st_size
    if dst.is_file() and dst.stat().st_size == sz:
        return
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != sz:
        raise IOError(f"Incomplete copy {{src}} -> {{dst}}")
    tmp.replace(dst)

base_code_src = DRIVE / "preparation/sparky_bc_20261120/g4_training_code.zip"
if not base_code_src.is_file():
    base_code_src = DRIVE / "preparation/crrl_training_code.zip"
rt_overlay_src = DRIVE / "preparation/sparky_value_bc_v2/runtime_overlay.zip"
mcts_overlay_src = DRIVE / "preparation/sparky_value_bc_v2/g4_heavy_mcts_overlay.zip"
plan_b_zips = sorted((DRIVE / "preparation/sparky_value_bc_v2").glob("sparky_plan_b_*.zip"))
val_hist_drv = DRIVE / "training/ppo_phase0_baselines/u1380_and_u1340/latest_u1380/opponents/opponent_1073743204.pt"

copy_jobs = [
    (base_code_src, Path("/content/crrl_base_code.zip")),
    (DRIVE / "preparation/anchor_actor.pt", PREP / "anchor_actor.pt"),
]
if rt_overlay_src.is_file():
    copy_jobs.append((rt_overlay_src, Path("/content/crrl_rt_overlay.zip")))
if mcts_overlay_src.is_file():
    copy_jobs.append((mcts_overlay_src, Path("/content/crrl_mcts_overlay.zip")))
if plan_b_zips and not (BC_PLAN_B / "dataset.manifest.json").is_file():
    copy_jobs.append((plan_b_zips[-1], Path("/content/crrl_sparky_plan_b.zip")))
if val_hist_drv.is_file():
    copy_jobs.append((val_hist_drv, PREP / "validation_history.pt"))

bv_drv = DRIVE / "training" / RUN_NAME / "best_validation"
if bv_drv.is_dir():
    for fn in ("actor_best_val.pt", "ppo_best_val.pt", "best_validation.json"):
        if (bv_drv / fn).is_file():
            copy_jobs.append((bv_drv / fn, RUN_DIR / fn))

with ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(lambda pair: _copy_atomic(*pair), copy_jobs))

for z_local in (Path("/content/crrl_base_code.zip"), Path("/content/crrl_rt_overlay.zip"), Path("/content/crrl_mcts_overlay.zip")):
    if z_local.is_file():
        with zipfile.ZipFile(z_local) as zf:
            zf.extractall(ROOT)

env_bin = ROOT / ".cache/vendor/crforge/env/build/install/env/bin/env"
if env_bin.is_file():
    env_bin.chmod(0o755)
tables = Path.home() / ".crforge/tables/16.635.3"
if not tables.is_dir() and (ROOT / "assets/tables/16.635.3").is_dir():
    tables.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "assets/tables/16.635.3", tables)

if not (BC_PLAN_B / "dataset.manifest.json").is_file() and Path("/content/crrl_sparky_plan_b.zip").is_file():
    tmp_bc = Path("/content/crrl_sparky_plan_b.extracting")
    if tmp_bc.exists():
        shutil.rmtree(tmp_bc)
    tmp_bc.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("/content/crrl_sparky_plan_b.zip") as zf:
        zf.extractall(tmp_bc)
    tmp_bc.rename(BC_PLAN_B)

if pip_proc is not None and pip_proc.wait() != 0:
    raise RuntimeError("pip install failed")

sys.path.insert(0, str(ROOT / "src"))
from cr_rl.training.fast_publication import restore_latest
ckpt_path = restore_latest(DRIVE / "training" / RUN_NAME, RUN_DIR)
if ckpt_path is None or not Path(ckpt_path).is_file():
    raise RuntimeError("Failed to restore latest checkpoint")

import torch, yaml
snap = torch.load(RUN_DIR / "ppo_latest.pt", map_location="cpu", mmap=True, weights_only=False)
restored_update = int(snap["updates"])
(PREP / "source_metadata.json").write_text(json.dumps({{k: snap.get(k) for k in ("updates", "model_architecture_version", "decision_ticks", "training_args", "reward_config", "level_profile")}}, indent=2) + "\\n")
(PREP / "source_league.yaml").write_text(yaml.safe_dump(snap["league_config"]))
del snap
ckpt_dir = RUN_DIR / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
seed_ckpt = ckpt_dir / f"update_{{restored_update:06d}}.pt"
seed_pool = ckpt_dir / f"update_{{restored_update:06d}}.pool.json"
if not seed_ckpt.is_file():
    os.link(RUN_DIR / "ppo_latest.pt", seed_ckpt)
if (RUN_DIR / "opponent_pool.json").is_file() and not seed_pool.is_file():
    shutil.copyfile(RUN_DIR / "opponent_pool.json", seed_pool)

drv_cfg_path = DRIVE / "preparation/sparky_value_bc_v2/cr_train_config_g4.json"
cfg = json.loads(drv_cfg_path.read_text())
if cfg.get("focal_decks") and cfg.get("validation_decks") != "/content/crrl_training/configs/decks_eval_focal.yaml":
    cfg["validation_decks"] = "/content/crrl_training/configs/decks_eval_focal.yaml"
    drv_cfg_path.write_text(json.dumps(cfg, indent=2) + "\\n")
cfg["trainer_entry"] = str(ROOT / "scripts/train_selfplay.py")
cfg["expert_bc_dataset"] = str(BC_PLAN_B)
if cfg.get("focal_decks") and (ROOT / "configs/decks_eval_focal.yaml").is_file():
    cfg["validation_decks"] = str(ROOT / "configs/decks_eval_focal.yaml")
if (PREP / "validation_history.pt").is_file():
    cfg["validation_history"] = str(PREP / "validation_history.pt")

props = torch.cuda.get_device_properties(0)
vram_gib = props.total_memory / (1024 ** 3)
ncpu = len(os.sched_getaffinity(0))
if vram_gib >= 70.0:
    profile_tag = "wide_70g+"
    cfg.update(gradient_checkpointing=False, sparse_placement=True, frame_microbatch=1024 if vram_gib >= 88 else 512, actor_batch_size=4, critic_batch_size=8, cache_state_frames=True, cache_reference=True, opponent_cache_size=8, training_opponent_cache_size=8, env_threads=min(16, ncpu), torch_threads=min(16, ncpu), lookahead_distill_games=24, lookahead_distill_parallel_games=24, lookahead_distill_parallel_searches=min(14, max(4, ncpu // 3)), lookahead_distill_threads=min(42, ncpu), lookahead_distill_horizon=20, lookahead_distill_rollouts=3, lookahead_distill_batch_episodes=24, validation_argmax_lanes=min(24, max(8, ncpu // 2)))
elif vram_gib >= 38.0:
    profile_tag = "mid_40g"
    cfg.update(gradient_checkpointing=True, sparse_placement=False, temporal_checkpoint_mode="block", frame_microbatch=256, actor_batch_size=2, critic_batch_size=4, cache_state_frames=True, cache_reference=True, opponent_cache_size=6, training_opponent_cache_size=4, env_threads=min(12, ncpu), torch_threads=min(8, ncpu), lookahead_distill_games=12, lookahead_distill_parallel_games=12, lookahead_distill_parallel_searches=4, lookahead_distill_threads=min(12, ncpu), validation_argmax_lanes=8)
elif vram_gib >= 20.0:
    profile_tag = "l4_22g"
    cfg.update(gradient_checkpointing=True, sparse_placement=True, temporal_checkpoint_mode="block", frame_microbatch=128, actor_batch_size=2, critic_batch_size=2, cache_state_frames=False, cache_reference=False, opponent_cache_size=4, training_opponent_cache_size=2, env_threads=min(12, ncpu), torch_threads=min(8, ncpu), lookahead_distill_games=8, lookahead_distill_parallel_games=8, lookahead_distill_parallel_searches=4, lookahead_distill_threads=min(12, ncpu), lookahead_distill_candidates=8, lookahead_distill_rollouts=2, lookahead_distill_horizon=14, lookahead_distill_batch_episodes=8, lookahead_distill_steps=3, macro_search_candidates=6, macro_search_batch_episodes=6, validation_argmax_lanes=8)
else:
    profile_tag = "compact_t4_15g"
    cfg.update(gradient_checkpointing=True, sparse_placement=True, temporal_checkpoint_mode="block", frame_microbatch=32, actor_batch_size=1, critic_batch_size=1, cache_state_frames=False, cache_reference=False, opponent_cache_size=4, training_opponent_cache_size=1, env_threads=min(8, ncpu), torch_threads=min(4, ncpu), lookahead_distill_games=8, lookahead_distill_parallel_games=8, lookahead_distill_parallel_searches=2, lookahead_distill_threads=min(8, ncpu), lookahead_distill_candidates=8, lookahead_distill_rollouts=2, lookahead_distill_horizon=14, lookahead_distill_batch_episodes=8, lookahead_distill_steps=3, macro_search_candidates=4, macro_search_batch_episodes=4, validation_argmax_lanes=8)

(PREP / "cr_train_config.json").write_text(json.dumps(cfg, indent=2) + "\\n")
env = dict(os.environ, PYTHONPATH=f"{{ROOT / 'src'}}:{{ROOT / '.cache/vendor/crforge/python'}}", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
if cfg.get("sparse_placement"):
    env["CRRL_SPARSE_PLACEMENT"] = "1"
res = subprocess.run([sys.executable, str(ROOT / "scripts/cr_train_pipeline.py"), "start"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
print(json.dumps({{"status": "RESTORED_AND_STARTED_OK", "gpu": props.name, "vram_gib": round(vram_gib, 1), "cpus": ncpu, "profile": profile_tag, "restored_from_update": restored_update, "elapsed_sec": round(time.perf_counter() - t0, 1), "pipeline": json.loads(res.stdout.strip())}}, indent=2))
'''
    print(conn["exec"](remote_code, timeout_sec=420), flush=True)
    return 0


def cmd_download_actors(args) -> int:
    conn = _connect_existing_colab()
    if conn is None:
        print("NO_EXISTING_COLAB_INSTANCE", flush=True)
        return 2
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_code = f"""
import hashlib, json, shutil, zlib
from pathlib import Path
import torch

RUN = Path("/content/crrl_training/runs") / {args.run!r}
DRV_RUN = Path("/content/drive/MyDrive/cr_rl_ppo/training") / {args.run!r}
latest_src = RUN / "actor_export.pt"
if not latest_src.is_file() and (DRV_RUN / "latest.json").is_file():
    rel_act = json.loads((DRV_RUN / "latest.json").read_text()).get("actor")
    if rel_act and (DRV_RUN / rel_act).is_file():
        latest_src = DRV_RUN / rel_act
best_src = RUN / "actor_best_val.pt"
if not best_src.is_file() and (DRV_RUN / "best_validation/actor_best_val.pt").is_file():
    best_src = DRV_RUN / "best_validation/actor_best_val.pt"
STAGE = Path("/crrl_dl_stage")
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True, exist_ok=True)

CHUNK = 8 * 1024 * 1024
manifest = {{}}
for tag, src in [("latest", latest_src), ("best_val", best_src)]:
    if not src.is_file():
        continue
    snap = STAGE / f"{{tag}}_full.tmp"
    shutil.copyfile(src, snap)
    hdr = torch.load(snap, map_location="cpu", weights_only=False, mmap=True)
    upd = int(hdr.get("updates", hdr.get("update", -1)))
    del hdr
    raw_bytes = snap.read_bytes()
    snap.unlink()
    full_sha = hashlib.sha256(raw_bytes).hexdigest()
    comp = zlib.compress(raw_bytes, 1)
    del raw_bytes
    chunks = []
    for idx, off in enumerate(range(0, len(comp), CHUNK)):
        part = comp[off : off + CHUNK]
        cname = f"{{tag}}_{{idx:03d}}.z"
        (STAGE / cname).write_bytes(part)
        chunks.append((cname, len(part), hashlib.sha256(part).hexdigest()))
    manifest[tag] = {{
        "update": upd,
        "raw_size": int(src.stat().st_size),
        "comp_size": len(comp),
        "sha256": full_sha,
        "chunks": chunks,
    }}
print("MANIFEST_JSON:" + json.dumps(manifest))
"""
    t0 = time.perf_counter()
    out_txt = conn["exec"](prep_code, timeout_sec=150)
    lines_m = [ln for ln in out_txt.splitlines() if "MANIFEST_JSON:" in ln]
    if not lines_m:
        print("PREP_OUTPUT_RAW:", out_txt, flush=True)
        return 1
    manifest = json.loads(lines_m[-1].split("MANIFEST_JSON:", 1)[1])
    print(
        f"Prepared compressed chunks in {time.perf_counter()-t0:.1f}s:",
        {k: f"u={v['update']} ({v['comp_size']/1048576:.1f} MiB compressed)" for k, v in manifest.items()},
        flush=True,
    )

    http = conn["http"]
    base_url = conn["base_url"]
    headers = conn["headers"]
    params = conn["params"]

    def _fetch_chunk(item):
        cname, exp_len, exp_sha = item
        for attempt in range(5):
            try:
                r = http.get(
                    f"{base_url}/api/contents/crrl_dl_stage/{cname}",
                    headers=headers,
                    params={**params, "content": "1", "format": "base64"},
                    timeout=45,
                )
                r.raise_for_status()
                raw = base64.b64decode(r.json()["content"])
                if len(raw) != exp_len or hashlib.sha256(raw).hexdigest() != exp_sha:
                    raise ValueError(f"Chunk integrity mismatch: {cname}")
                return raw
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.0 * (attempt + 1))

    saved_paths = {}
    for tag, meta in manifest.items():
        t_dl = time.perf_counter()
        with ThreadPoolExecutor(max_workers=int(args.streams)) as pool:
            parts = list(pool.map(_fetch_chunk, meta["chunks"]))
        comp_bytes = b"".join(parts)
        del parts
        raw_bytes = zlib.decompress(comp_bytes)
        del comp_bytes
        if hashlib.sha256(raw_bytes).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"Full SHA256 mismatch for {tag}")
        dst = out_dir / f"actor_{tag}_u{meta['update']:04d}.pt"
        alias = out_dir / ("actor_best_val.pt" if tag == "best_val" else "actor_latest.pt")
        tmp = dst.with_suffix(".pt.tmp")
        tmp.write_bytes(raw_bytes)
        del raw_bytes
        tmp.replace(dst)
        shutil.copyfile(dst, alias)
        dt = time.perf_counter() - t_dl
        mb = meta["raw_size"] / 1048576.0
        print(
            f"DOWNLOADED [{tag}] u={meta['update']}: {mb:.1f} MiB in {dt:.1f}s "
            f"({mb/max(0.1,dt):.1f} MiB/s) -> {alias} (SHA256={meta['sha256'][:16]})",
            flush=True,
        )
        saved_paths[tag] = {
            "path": str(alias),
            "versioned_path": str(dst),
            "update": meta["update"],
            "sha256": meta["sha256"],
        }

    conn["exec"]("import shutil; shutil.rmtree('/crrl_dl_stage', ignore_errors=True)", timeout_sec=20)
    (out_dir / "manifest.json").write_text(json.dumps(saved_paths, indent=2), encoding="utf-8")
    print("ALL_DOWNLOADS_VERIFIED_OK", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=DEFAULT_RUN_NAME, help="Training run directory name on Colab/Drive")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Send kernel keepalive and print GPU, recent updates, validations, and Drive status")

    p_watch = sub.add_parser("watch", help="Loop keepalive + status poll at a fixed interval")
    p_watch.add_argument("--interval", type=int, default=900, help="Seconds between keepalive polls (default: 900)")

    p_exec = sub.add_parser("exec", help="Execute a local Python script file on the existing Colab kernel")
    p_exec.add_argument("script", type=Path)
    p_exec.add_argument("--timeout", type=int, default=90)

    sub.add_parser("start", help="Start cr_train_pipeline.py worker on Colab")
    sub.add_parser("stop", help="Request graceful checkpoint-safe stop on Colab")
    sub.add_parser("restart", help="Gracefully stop and restart cr_train_pipeline.py worker on Colab")
    sub.add_parser(
        "restore-and-start",
        help="Restore latest Drive checkpoint/opponents/code to /content SSD (auto-adapting G4/A100/L4/T4 VRAM) and start training",
    )

    p_dl = sub.add_parser("download-actors", help="Download & SHA256-verify actor_latest.pt and actor_best_val.pt")
    p_dl.add_argument("--out", type=Path, default=Path("artifacts/g4_heavy_mcts"))
    p_dl.add_argument("--streams", type=int, default=16)
    return p


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.cmd == "status":
        rc = cmd_status(args)
    elif args.cmd == "watch":
        rc = cmd_watch(args)
    elif args.cmd == "exec":
        rc = cmd_exec(args)
    elif args.cmd in ("start", "stop", "restart"):
        rc = cmd_pipeline_ctl(args, args.cmd)
    elif args.cmd == "restore-and-start":
        rc = cmd_restore_and_start(args)
    elif args.cmd == "download-actors":
        rc = cmd_download_actors(args)
    else:
        raise ValueError(f"Unhandled subcommand {args.cmd}")
    os._exit(rc)


if __name__ == "__main__":
    main()
