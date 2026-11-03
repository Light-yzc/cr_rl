from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.data.sharded_bc import CachedShardArray, MappedShardArray, ShardArrayCache
from cr_rl.data.validation_cache import SelectedRows, prepare_validation_arrays
from cr_rl.training.bc_batch import build_bc_batch
from cr_rl.training.persistence import file_sha256


class Adapter:
    schema = SimpleNamespace(
        seat_ints=12,
        header=dict(hand_card=[0, 1, 2, 3], deck=[0, 1, 2, 3, 4, 5, 6, 7], enemy_seen=[4, 5, 6, 7]),
    )

    def __call__(self, windows):
        return (
            windows[..., :2].astype(np.float32),
            windows[..., 2:6].reshape(*windows.shape[:-1], 2, 2).astype(np.float32),
            np.ones((*windows.shape[:-1], 2), dtype=bool),
        )


def test_compact_cache_preserves_history_labels_masks_and_paired_indexing(tmp_path, monkeypatch):
    obs = np.arange(24 * 12, dtype=np.int32).reshape(24, 12)
    masks = (np.arange(24 * 7).reshape(24, 7) % 3 != 0).astype(np.uint8)
    cache = ShardArrayCache(tmp_path / "full_cache", 512)
    pieces = {name: [] for name in ["observations", "masks"]}
    for index, (left, right) in enumerate([(0, 8), (8, 24)]):
        source = tmp_path / f"source_{index}.npz"
        np.savez_compressed(source, observations=obs[left:right], masks=masks[left:right])
        digest = file_sha256(source)
        for field in pieces:
            pieces[field].append(CachedShardArray(source, digest, field, cache))
    arrays = {name: MappedShardArray(value) for name, value in pieces.items()}
    # Any whole-array materialization would fail this spy.
    monkeypatch.setattr(cache, "read", lambda *args: pytest.fail("Whole shard was materialized"))
    ids = np.repeat([0, 1, 2], 8)
    seats = np.arange(24) % 2
    actions = np.zeros(24, np.int64)
    groups = {
        (game, seat): np.flatnonzero((ids == game) & (seats == seat)).tolist()
        for game in range(3)
        for seat in range(2)
    }
    positions = {k: {row: i for i, row in enumerate(group)} for k, group in groups.items()}
    rows = np.array([5, 12, 23])
    compact = prepare_validation_arrays(
        arrays["observations"],
        arrays["masks"],
        rows,
        ids,
        seats,
        groups,
        positions,
        3,
        tmp_path / "compact",
        "dataset_sha",
    )
    expected = build_bc_batch(
        rows, obs, masks, actions, ids, seats, groups, positions, Adapter(), 3, "cpu"
    )
    actual = build_bc_batch(
        rows, *compact, actions, ids, seats, groups, positions, Adapter(), 3, "cpu"
    )
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    np.testing.assert_array_equal(compact[1][rows, [1, 4, 6]], masks[rows, [1, 4, 6]])
    assert cache.extractions == 0
    assert not list((tmp_path / "full_cache").glob("*/*.npy"))
    with pytest.raises(IndexError, match="outside"):
        compact[0][0]
    # A completed compact cache must reuse exactly the same arrays without touching archives.
    for source in tmp_path.glob("source_*.npz"):
        source.unlink()
    reused = prepare_validation_arrays(
        arrays["observations"],
        arrays["masks"],
        rows,
        ids,
        seats,
        groups,
        positions,
        3,
        tmp_path / "compact",
        "dataset_sha",
    )
    np.testing.assert_array_equal(reused[0][rows], obs[rows])


def test_compact_checksum_and_membership_are_enforced(tmp_path):
    values = np.arange(12).reshape(4, 3)
    selected = SelectedRows(np.array([1, 3]), values[[1, 3]], values.shape)
    np.testing.assert_array_equal(selected[-1], values[-1])
    np.testing.assert_array_equal(selected[[3, 1, 3]], values[[3, 1, 3]])
    with pytest.raises(IndexError, match="outside"):
        selected[2]
    with pytest.raises(IndexError, match="integer"):
        selected[[1.5]]
    obs = MappedShardArray([values])
    masks = MappedShardArray([np.ones((4, 2), np.uint8)])
    ids = np.zeros(4, int)
    seats = np.zeros(4, int)
    groups = {(0, 0): list(range(4))}
    positions = {(0, 0): {i: i for i in range(4)}}
    prepare_validation_arrays(obs, masks, [3], ids, seats, groups, positions, 2, tmp_path, "sha")
    file = next(tmp_path.glob("*/observations.npy"))
    with file.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        prepare_validation_arrays(
            obs, masks, [3], ids, seats, groups, positions, 2, tmp_path, "sha"
        )


def test_readonly_calibration_cache_does_not_mutate_trainer_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.npz"
    observations = np.arange(24, dtype=np.int32).reshape(8, 3)
    masks = np.ones((8, 2), np.uint8)
    np.savez_compressed(source, observations=observations, masks=masks)
    # This wrapper intentionally has no read() or eviction method: calibration
    # may inspect immutable files but must never alter the trainer's cache LRU.
    wrapper = SimpleNamespace(readonly=tmp_path / "trainer_readonly")
    arrays = []
    for field in ["observations", "masks"]:
        arrays.append(MappedShardArray([CachedShardArray(source, "digest", field, wrapper)]))
    monkeypatch.setattr(
        "cr_rl.data.validation_cache.shutil.disk_usage", lambda path: SimpleNamespace(free=1024**4)
    )
    ids = np.zeros(8, int)
    seats = np.zeros(8, int)
    groups = {(0, 0): list(range(8))}
    positions = {(0, 0): {i: i for i in range(8)}}
    compact = prepare_validation_arrays(
        *arrays, [7], ids, seats, groups, positions, 3, tmp_path / "selected", "sha"
    )
    np.testing.assert_array_equal(compact[0][[5, 6, 7]], observations[[5, 6, 7]])
    assert not wrapper.readonly.exists()
