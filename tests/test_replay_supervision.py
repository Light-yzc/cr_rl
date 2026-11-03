import numpy as np

from cr_rl.data.replay_supervision import (
    SOURCE_COMMAND_OMITTED,
    isolated_wait_rows,
    mask_omitted_commands,
    prefix_cutoff,
)


def test_omitted_human_command_never_becomes_wait_or_partial_collision_target():
    ticks = np.repeat([0, 5, 10], 2)
    seats = np.tile([0, 1], 3)
    flags = np.array([0, 0, 0, 2, 0, 0], dtype=np.int32)
    result = mask_omitted_commands(flags, ticks, seats, [(4, 0), (7, 1), (7, 1)])
    np.testing.assert_array_equal(
        result, [SOURCE_COMMAND_OMITTED, 0, 0, 2 | SOURCE_COMMAND_OMITTED, 0, 0]
    )
    np.testing.assert_array_equal(flags, [0, 0, 0, 2, 0, 0])
    np.testing.assert_array_equal(mask_omitted_commands(flags, ticks, seats, []), flags)


def test_omission_budget_cuts_at_the_third_event_without_changing_strict_default():
    omissions = [(29, 1), (12, 0), (18, 1)]
    assert prefix_cutoff(omissions) == 10
    assert prefix_cutoff(omissions, max_omissions=2) == 25
    assert prefix_cutoff(omissions[:2], max_omissions=2) is None
    flags = mask_omitted_commands(
        np.zeros(6, dtype=int),
        np.repeat([5, 10, 15], 2),
        np.tile([0, 1], 3),
        [(10, 0)],
        margin_ticks=5,
        both_seats=True,
    )
    assert (flags == SOURCE_COMMAND_OMITTED).all()


def test_wait_margin_uses_simulator_ticks_and_does_not_cross_seats_or_matches():
    ticks = np.tile(np.arange(0, 65, 5), 3)
    seats = np.repeat([0, 1, 0], 13)
    replays = np.repeat([0, 0, 1], 13)
    actions = np.zeros(39, dtype=int)
    actions[8] = 1  # Match 0, seat 0, tick 40.
    waits = np.flatnonzero(actions == 0)
    retained = isolated_wait_rows(waits, actions, replays, seats, ticks)
    np.testing.assert_array_equal(retained[retained < 13], [0, 1, 2, 3, 11, 12])
    np.testing.assert_array_equal(retained[retained >= 13], np.arange(13, 39))
