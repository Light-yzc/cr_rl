import json

import pytest

from cr_rl.training.runtime_supervision import supervise_trial


def test_supervision_checks_live_work_and_waits_for_controller_exit(tmp_path):
    path = tmp_path / "status.json"
    phases = [("training", True), ("completed", True), ("completed", False)]
    elapsed = [0]
    calls = []

    def write():
        phase, _ = phases[min(elapsed[0], 2)]
        path.write_text(json.dumps({"phase": phase, "pid": 10, "observed_unix": elapsed[0]}))

    def sleep(seconds):
        elapsed[0] += seconds
        write()

    write()
    result = supervise_trial(
        path,
        budget_seconds=5,
        interval_seconds=1,
        monotonic=lambda: elapsed[0],
        wall_time=lambda: elapsed[0],
        sleep=sleep,
        alive=lambda pid: pid == 10 and phases[min(elapsed[0], 2)][1],
        emit=calls.append,
    )
    assert elapsed[0] == 2 and result["phase"] == "completed"
    assert len(calls) == 3


@pytest.mark.parametrize(
    "phase,live,error",
    [
        ("training", False, "controller is missing"),
        ("failed", False, "Trial failed"),
    ],
)
def test_supervision_reports_real_worker_failure_without_restarting(tmp_path, phase, live, error):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": phase, "pid": 10, "observed_unix": 0}))
    with pytest.raises(RuntimeError, match=error):
        supervise_trial(path, alive=lambda pid: live, wall_time=lambda: 0, emit=lambda _: None)


def test_live_but_stale_controller_is_not_mistaken_for_progress(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "training", "pid": 10, "observed_unix": 0}))
    with pytest.raises(RuntimeError, match="heartbeat is stale"):
        supervise_trial(path, alive=lambda pid: True, wall_time=lambda: 1900, emit=lambda _: None)


def test_heartbeats_without_training_progress_do_not_hide_a_stall(tmp_path):
    path = tmp_path / "status.json"
    elapsed = [0]

    def write():
        path.write_text(
            json.dumps(
                {
                    "phase": "training",
                    "pid": 10,
                    "observed_unix": elapsed[0],
                    "latest_update": {"optimizer_steps": 4480},
                }
            )
        )

    def sleep(seconds):
        elapsed[0] += seconds
        write()

    write()
    with pytest.raises(RuntimeError, match="progress is stale"):
        supervise_trial(
            path,
            budget_seconds=10,
            interval_seconds=1,
            stale_seconds=2,
            alive=lambda pid: True,
            wall_time=lambda: elapsed[0],
            monotonic=lambda: elapsed[0],
            sleep=sleep,
            emit=lambda _: None,
        )
