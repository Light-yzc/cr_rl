import numpy as np

from cr_rl.data.bc_sampling import epoch_rows


def test_full_pass_covers_every_action_once_and_keeps_shard_access_contiguous():
    plays = np.array([1, 8, 12, 19, 21, 25, 39])
    waits = np.array([0, 3, 10, 14, 20, 24, 30, 33, 35])
    offsets = np.array([0, 10, 20, 30, 40])
    result = epoch_rows(plays, waits, np.random.default_rng(5), 0.5, offsets=offsets)
    np.testing.assert_array_equal(np.sort(result[np.isin(result, plays)]), plays)
    assert len(result) == 2 * len(plays)
    shards = np.searchsorted(offsets[1:], result, side="right")
    assert len(shards[1:][shards[1:] != shards[:-1]]) == len(np.unique(shards)) - 1
    np.testing.assert_array_equal(
        result,
        epoch_rows(plays, waits, np.random.default_rng(5), 0.5, offsets=offsets),
    )
