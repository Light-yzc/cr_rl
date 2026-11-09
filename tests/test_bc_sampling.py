import numpy as np

from cr_rl.data.bc_sampling import epoch_rows


def test_full_pass_covers_every_action_once_and_default_mixes_shards():
    plays = np.array([1, 8, 12, 19, 21, 25, 39])
    waits = np.array([0, 3, 10, 14, 20, 24, 30, 33, 35])
    offsets = np.array([0, 10, 20, 30, 40])
    result = epoch_rows(plays, waits, np.random.default_rng(5), 0.5, offsets=offsets)
    np.testing.assert_array_equal(np.sort(result[np.isin(result, plays)]), plays)
    assert len(result) == 2 * len(plays)
    shards = np.searchsorted(offsets[1:], result, side="right")
    # The default is a bounded 64-shard working set.  This fixture has fewer
    # than 64 shards, so one window contains every shard and is not shard-local.
    assert len(np.unique(shards)) > 1
    assert np.count_nonzero(np.diff(shards)) > len(np.unique(shards)) - 1
    np.testing.assert_array_equal(
        result,
        epoch_rows(plays, waits, np.random.default_rng(5), 0.5, offsets=offsets),
    )


def test_window_one_remains_available_for_old_reproducible_runs():
    plays = np.array([1, 8, 12, 19, 21, 25, 39])
    waits = np.array([0, 3, 10, 14, 20, 24, 30, 33, 35])
    offsets = np.array([0, 10, 20, 30, 40])
    result = epoch_rows(
        plays, waits, np.random.default_rng(5), 0.5, offsets=offsets, shard_mix_window=1
    )
    shards = np.searchsorted(offsets[1:], result, side="right")
    assert len(shards[1:][shards[1:] != shards[:-1]]) == len(np.unique(shards)) - 1
