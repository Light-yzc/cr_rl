"""Run a bounded CPU-only transport acceptance before resuming training.

Use the installed Colab CLI's Python. Requires one already-created CPU session
with an initialized kernel; no Drive mount, VM allocation or training is done.
The owned test session is released after evidence is saved. This host must stay
online; the test does not establish survival after this host powers off.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import requests
from watch_colab_session import ExistingColab

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def remote_health(adapter):
    if adapter.refresh() is None:
        raise ConnectionError("Test assignment not listed")
    session = adapter.session
    response = requests.get(
        session.url.rstrip("/") + "/api/contents/content/crrl_liveness_probe.json",
        params={"authuser": "0", "colab-runtime-proxy-token": session.token, "content": "1"},
        timeout=20,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    body = response.json()
    data = body["content"]
    if body.get("format") == "base64":
        data = base64.b64decode(data).decode()
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--minutes", type=float, default=75)
    parser.add_argument("--rotate-seconds", type=float, default=1200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.minutes < 3 or args.rotate_seconds <= 0 or args.out.exists():
        parser.error("Use >=3 minutes and a fresh evidence directory")
    args.out.mkdir(parents=True)
    logging.disable(logging.CRITICAL)
    adapter = ExistingColab(args.session)
    if adapter.refresh() is None or adapter.session.accelerator != "NONE":
        raise ValueError("An existing CPU-only session is required")
    endpoint, kernel = adapter.endpoint, adapter.kernel_id
    duration = args.minutes * 60
    source = args.out / "kernel_probe.py"
    source.write_text(
        """import json, os, resource, time, uuid
from pathlib import Path
p=Path('/content/crrl_liveness_probe.json')
began=time.monotonic(); nonce=uuid.uuid4().hex; count=0
while True:
 elapsed=time.monotonic()-began; count+=1
 row={'phase':'completed' if elapsed>=DURATION else 'observing',
      'nonce':nonce,'kernel_pid':os.getpid(),'elapsed_seconds':elapsed,
      'samples':count,'observed_unix':time.time(),
      'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(row)); tmp.replace(p)
 print(json.dumps(row),flush=True)
 if row['phase']=='completed': break
 time.sleep(15)
""".replace("DURATION", repr(duration))
    )
    foreground = watcher = None
    began = time.monotonic()
    result = {
        "phase": "starting",
        "endpoint": endpoint,
        "kernel_id": kernel,
        "requested_minutes": args.minutes,
        "cli_version": version("google-colab-cli"),
        "kernel_client_version": version("jupyter-kernel-client"),
        "driver_pid": os.getpid(),
    }

    def publish(**extra):
        result.update(extra, observed_unix=time.time())
        atomic_json(args.out / "status.json", result)
        print(json.dumps(result), flush=True)

    try:
        with (args.out / "foreground_client.log").open("w") as log:
            foreground = subprocess.Popen(
                [
                    "colab",
                    "exec",
                    "--session",
                    args.session,
                    "--file",
                    str(source),
                    "--timeout",
                    str(duration + 180),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        health = None
        startup = time.monotonic() + 120
        while time.monotonic() < startup:
            health = remote_health(adapter)
            if health:
                break
            if foreground.poll() is not None:
                raise RuntimeError("Foreground probe client exited during startup")
            time.sleep(5)
        if not health:
            raise TimeoutError("Kernel did not publish diagnostic health")
        nonce, kernel_pid = health["nonce"], health["kernel_pid"]
        publish(phase="observing", initial_health=health)
        with (args.out / "connection_stdout.log").open("w") as log:
            watcher = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/watch_colab_session.py"),
                    "--session",
                    args.session,
                    "--minutes",
                    str(args.minutes + 5),
                    "--poll-seconds",
                    "20",
                    "--refresh-seconds",
                    "60",
                    "--rotate-seconds",
                    str(args.rotate_seconds),
                    "--out",
                    str(args.out / "connection.jsonl"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        killed_client = False
        errors = 0
        deadline = began + duration + 300
        while time.monotonic() < deadline:
            if watcher.poll() is not None:
                raise RuntimeError("Connection watcher exited; inspect sanitized event log")
            try:
                health = remote_health(adapter)
                if health is None or (health["nonce"], health["kernel_pid"]) != (nonce, kernel_pid):
                    raise ValueError("Kernel health identity changed or disappeared")
                errors = 0
            except (ConnectionError, requests.RequestException) as error:
                errors += 1
                publish(phase="transport_retry", error_type=type(error).__name__)
                if errors >= 3:
                    raise ConnectionError("Three health reads failed") from None
                time.sleep(20)
                continue
            if not killed_client and health["elapsed_seconds"] >= 90:
                foreground.terminate()  # Close only this local client, never interrupt the kernel.
                foreground.wait(timeout=20)
                killed_client = True
            atomic_json(args.out / "remote_health.json", health)
            publish(
                phase="observing",
                remote_elapsed_seconds=health["elapsed_seconds"],
                foreground_client_disconnected=killed_client,
            )
            if health["phase"] == "completed":
                events = [
                    json.loads(line)
                    for line in (args.out / "connection.jsonl").read_text().splitlines()
                    if line
                ]
                reconnects = sum(event["phase"] == "connection_renewed" for event in events)
                if not killed_client or not reconnects:
                    raise RuntimeError("Disconnect/rotation coverage is incomplete")
                publish(
                    phase="completed",
                    proxy_socket_rotations=reconnects,
                    crossed_one_hour=health["elapsed_seconds"] >= 3600,
                    same_kernel_after_disconnect=True,
                    limitation="CPU probe with host online; not a GPU or host-poweroff guarantee",
                )
                atomic_json(args.out / "acceptance.json", result)
                return
            time.sleep(30)
        raise TimeoutError("Liveness acceptance budget exceeded")
    except Exception as error:
        publish(phase="failed", error_type=type(error).__name__)
        atomic_json(args.out / "acceptance.json", result)
        raise SystemExit(1) from None
    finally:
        for child in (watcher, foreground):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
        # The test owns only this exact CPU endpoint; never stop another assignment.
        try:
            if adapter.refresh() is not None and adapter.endpoint == endpoint:
                adapter.state.client.unassign(endpoint)
                adapter.state.store.remove(args.session)
                publish(test_session_released=True)
        except Exception as error:
            publish(cleanup_error_type=type(error).__name__)


if __name__ == "__main__":
    main()
