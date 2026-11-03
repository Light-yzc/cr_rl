import json
import os
import signal
from pathlib import Path

import pytest

from scripts.followup_cycle_stage import paused_next_trainer


def install_process(monkeypatch, tmp_path, command):
    (tmp_path / "status.json").write_text(
        json.dumps(dict(phase="bc_training", stage=3, training_pid=4242))
    )
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: "4242 (python3) T"
        if str(path) == "/proc/4242/stat"
        else original_read(path, *a, **k),
    )
    monkeypatch.setattr(Path, "read_bytes", lambda path: command)
    monkeypatch.setattr(os, "pidfd_open", lambda pid: 77, raising=False)
    calls, closed = [], []
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: calls.append((fd, sig)), raising=False
    )
    monkeypatch.setattr(os, "close", closed.append)
    return calls, closed


def test_failed_calibration_resumes_only_the_verified_trainer(monkeypatch, tmp_path):
    calls, closed = install_process(
        monkeypatch, tmp_path, b"python3\0train_bc.py\0" + os.fsencode(str(tmp_path))
    )
    with pytest.raises(RuntimeError, match="calibration failed"):
        with paused_next_trainer(tmp_path, 2) as pid:
            assert pid == 4242
            raise RuntimeError("calibration failed")
    assert calls == [(77, signal.SIGSTOP), (77, signal.SIGCONT)]
    assert closed == [77]


def test_reused_pid_is_neither_stopped_nor_resumed(monkeypatch, tmp_path):
    calls, closed = install_process(monkeypatch, tmp_path, b"unrelated process")
    with pytest.raises(RuntimeError, match="identity differs"):
        with paused_next_trainer(tmp_path, 2):
            pytest.fail("Must not enter GPU work with an unverified process")
    assert calls == []
    assert closed == [77]
