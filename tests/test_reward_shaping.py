from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.training.rewards import RewardConfig, shaped_rewards, tower_potentials


def arena():
    names = [
        f"{side}_{tower}_{suffix}"
        for side in ("own", "enemy")
        for tower in ("left", "right", "king")
        for suffix in ("hp", "max_hp")
    ]
    names += ["crowns", "crowns_enemy", "elixir"]
    schema = SimpleNamespace(header={name: [i] for i, name in enumerate(names)})
    obs = np.zeros((1, 2, len(names)), dtype=np.int32)
    for name in names:
        if "hp" in name:
            obs[..., schema.header[name][0]] = 1000
    return schema, obs


def test_discounted_telescoping_terminal_and_zero_sum():
    schema, current = arena()
    initial = current.copy()
    cfg = RewardConfig(enabled=True)
    gamma = 0.9995
    total = np.zeros((1, 2))
    for step in range(5):
        following = current.copy()
        following[0, 0, schema.header["enemy_left_hp"][0]] -= 200
        following[0, 1, schema.header["own_left_hp"][0]] -= 200
        if step == 4:
            following[0, 0, schema.header["crowns"][0]] = 1
            following[0, 1, schema.header["crowns_enemy"][0]] = 1
        reward, components = shaped_rewards(
            current,
            following,
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([step == 4]),
            schema,
            gamma,
            cfg,
        )
        assert reward.sum() == pytest.approx(0, abs=1e-7)
        total += gamma**step * reward
        if step == 0:
            assert reward[0, 0] > 0
        if step == 4:
            assert components["hp"][0, 0] < 0  # terminal potential is cancelled
        current = following
    expected = -sum(tower_potentials(initial, schema, cfg))
    np.testing.assert_allclose(total, expected, atol=1e-7)


def test_time_limit_keeps_final_potential_and_faults_do_not_shape():
    schema, current = arena()
    final = current.copy()
    final[..., schema.header["enemy_left_hp"][0]] = 500
    cfg = RewardConfig(enabled=True)
    reward, _ = shaped_rewards(
        current, final, np.ones((1, 2)), np.zeros((1, 2)), np.array([False]), schema, 0.99, cfg
    )
    np.testing.assert_allclose(reward, 0.99 * 0.075)
    fault_reward, _ = shaped_rewards(
        current,
        final,
        np.ones((1, 2)),
        np.zeros((1, 2)),
        np.array([False]),
        schema,
        0.99,
        cfg,
        faulted=np.array([True]),
    )
    np.testing.assert_array_equal(fault_reward, 0)


def test_overflow_is_opt_in_and_raw_rewards_are_preserved():
    schema, obs = arena()
    obs[..., schema.header["elixir"][0]] = 100000
    raw = np.array([[1, -1]], dtype=np.float32)
    actions = np.array([[0, 1]])
    reward, _ = shaped_rewards(
        obs,
        obs,
        actions,
        raw,
        np.array([False]),
        schema,
        0.999,
        RewardConfig(enabled=True, overflow=0.001),
    )
    np.testing.assert_allclose(reward, [[0.999, -1]])
    np.testing.assert_array_equal(raw, [[1, -1]])
    disabled, _ = shaped_rewards(
        obs, obs, actions, raw, np.array([True]), schema, 0.999, RewardConfig()
    )
    np.testing.assert_array_equal(disabled, raw)


def test_missing_or_destroyed_tower_maximum_is_finite():
    schema, obs = arena()
    obs[..., schema.header["own_left_hp"][0]] = 0
    obs[..., schema.header["own_left_max_hp"][0]] = 0
    hp, _ = tower_potentials(obs, schema, RewardConfig())
    assert np.isfinite(hp).all()
    np.testing.assert_allclose(hp, -0.15)
