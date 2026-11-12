from threading import Barrier, Event

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
