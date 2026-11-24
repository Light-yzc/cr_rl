import json
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.envs.crforge import IndependentBattles, _apply_level_jitter


class Lane:
    def __init__(self, step):
        self.callback = step
        for name in ("observations", "masks", "actions", "rewards", "dones", "infos"):
            setattr(self, name, np.zeros((1, 2), dtype=np.int32))

    def step(self):
        self.callback(self)

    def close(self):
        pass


def test_lanes_run_concurrently_and_keep_buffers_separate():
    barrier = Barrier(3, timeout=5)

    def step(lane):
        barrier.wait()
        lane.observations[:] = lane.actions + 7

    env = IndependentBattles(3, lambda n, t: Lane(step), threads=3)
    try:
        env.actions[:] = np.arange(6).reshape(3, 2)
        env.step()
        np.testing.assert_array_equal(env.observations, env.actions + 7)
    finally:
        env.close()


def test_failed_lane_joins_other_workers_before_return():
    finished = Event()
    failed = Event()
    callbacks = iter(
        (
            lambda lane: (failed.set(), (_ for _ in ()).throw(ValueError("lane"))),
            lambda lane: (failed.wait(5), finished.set()),
        )
    )
    env = IndependentBattles(2, lambda n, t: Lane(next(callbacks)), threads=2)
    try:
        with pytest.raises(ValueError, match="lane"):
            env.step()
        assert finished.is_set()
    finally:
        env.close()


def test_slow_lane_diagnostic_records_pending_call_and_still_joins(monkeypatch, capsys):
    monkeypatch.setenv("CRRL_NATIVE_STEP_DIAGNOSTIC_SECONDS", "0.05")
    release = Event()
    finished = Event()

    class DiagnosticLane(Lane):
        schema = SimpleNamespace(info={"tick": 0})

        def reset_env(self, index, seed):
            self.infos[index, 0] = 8

    def slow_step(lane):
        assert release.wait(5)
        finished.set()

    callbacks = iter((lambda lane: None, slow_step))
    env = IndependentBattles(2, lambda n, t: DiagnosticLane(next(callbacks)), threads=2)
    report = env._report_slow_step

    def record_and_release(pending_lanes):
        try:
            report(pending_lanes)
        finally:
            release.set()

    monkeypatch.setattr(env, "_report_slow_step", record_and_release)
    try:
        env.reset_env(1, 873)
        env.actions[1] = [215, 0]
        env.step()
        assert finished.is_set()
        row = json.loads(capsys.readouterr().err)
        assert row["event"] == "native_step_slow"
        assert dict(lane=1, reset_seed=873, last_synced_tick=8, actions=[215, 0]) in row["pending_lanes"]
        worker = next(stack for stack in row["python_stacks"] if stack["lane"] == 1)
        assert worker["native_id"] > 0 and 0 < len(worker["stack"]) <= 12
    finally:
        env.close()


def test_level_mixture_factory_receives_stable_lane_index():
    indices = []

    def factory(n, t, lane_index):
        indices.append(lane_index)
        return Lane(lambda lane: None)

    env = IndependentBattles(
        4,
        factory,
        threads=2,
        level_mixture=[10, 11, 12],
        tower_level_mixture=[10, 11, 12],
    )
    try:
        assert indices == [0, 1, 2, 3]
    finally:
        env.close()


def test_level_jitter_is_seeded_and_preserves_base_for_most_lanes():
    base = [10, 11, 12, 10, 11, 12, 10, 11]
    first = _apply_level_jitter(base, 0.15, 2, 123)
    second = _apply_level_jitter(base, 0.15, 2, 123)
    assert first == second
    assert all(abs(value - original) in (0, 2) for value, original in zip(first, base))
    assert any(value != original for value, original in zip(first, base))


def test_step_lanes_steps_only_requested_indices():
    calls = []

    def make_lane(idx):
        return Lane(lambda lane: calls.append(idx))

    counter = iter(range(3))
    env = IndependentBattles(3, lambda n, t: make_lane(next(counter)), threads=2)
    try:
        env.actions[:] = 1
        env.step_lanes([0, 2])
        assert sorted(calls) == [0, 2]
    finally:
        env.close()
