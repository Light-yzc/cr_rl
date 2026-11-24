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
STAGE = Path("/crrl_dl_stage")
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True, exist_ok=True)

CHUNK = 8 * 1024 * 1024
manifest = {{}}
for tag, src in [("latest", RUN / "actor_export.pt"), ("best_val", RUN / "actor_best_val.pt")]:
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
    elif args.cmd == "download-actors":
        rc = cmd_download_actors(args)
    else:
        raise ValueError(f"Unhandled subcommand {args.cmd}")
    os._exit(rc)


if __name__ == "__main__":
    main()
