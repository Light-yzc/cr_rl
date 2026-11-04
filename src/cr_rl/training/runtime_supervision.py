"""Bounded kernel-side supervision of real Colab work, without synthetic activity."""

from __future__ import annotations

import json
import time
from pathlib import Path


def process_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def supervise_trial(
    status_path,
    *,
    budget_seconds=4 * 3600,
    interval_seconds=30,
    stale_seconds=1800,
    alive=process_alive,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    emit=print,
):
    """Keep a kernel cell executing actual worker checks until bounded work ends.

    A stale/missing worker raises; it never triggers a duplicate training process.
    Timeout ends supervision only and does not silently kill the trainer. This is
    not a guarantee against Colab resource limits or runtime reclamation.
    """
    if min(budget_seconds, interval_seconds, stale_seconds) <= 0:
        raise ValueError("Supervision limits must be positive")
    path = Path(status_path)
    began = monotonic()
    last_progress = began
    previous = None
    while monotonic() - began < budget_seconds:
        status = json.loads(path.read_text())
        phase = status["phase"]
        controller_live = alive(status.get("pid"))
        child_live = alive(status.get("child_pid"))
        now = monotonic()
        token = (
            phase,
            status.get("arm"),
            status.get("candidate"),
            status.get("completed_shards"),
            status.get("child_pid"),
            (status.get("latest_update") or {}).get("optimizer_steps"),
            (status.get("latest_update") or {}).get("samples_in_epoch"),
        )
        if token != previous:
            last_progress, previous = now, token
        report = {
            "phase": "kernel_supervision",
            "trial_phase": phase,
            "controller_live": controller_live,
            "child_live": child_live,
            "arm": status.get("arm"),
            "candidate": status.get("candidate"),
            "latest_update": status.get("latest_update"),
            "heartbeat_age_seconds": wall_time() - status["observed_unix"],
        }
        emit(json.dumps(report))
        if phase in ("completed", "failed"):
            if not controller_live and not child_live:
                if phase == "failed":
                    raise RuntimeError(f"Trial failed: {status.get('error', 'see saved logs')}")
                return status
        elif not controller_live:
            raise RuntimeError(
                "Trial controller is missing; inspect child and backups before restart"
            )
        if wall_time() - status["observed_unix"] > stale_seconds:
            raise RuntimeError("Trial heartbeat is stale; inspect actual worker before restarting")
        if now - last_progress > stale_seconds:
            raise RuntimeError("Trial progress is stale; inspect actual worker before restarting")
        sleep(interval_seconds)
    raise TimeoutError("Kernel supervision budget elapsed; inspect existing worker before restart")
