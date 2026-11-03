import json

import numpy as np
import pytest

from cr_rl.data.sharded_bc import MappedShardArray, load_sharded_bc
from cr_rl.training.persistence import file_sha256


def test_sharded_indexing_preserves_history_order_and_paired_legal_actions():
    whole = np.arange(60, dtype=np.int32).reshape(10, 6)
    sharded = MappedShardArray([whole[:3], whole[3:7], whole[7:]])
    for rows in ([8, 1, 8, 5], [0], [], slice(None), slice(2, 8), slice(None, None, 2)):
        np.testing.assert_array_equal(sharded[rows], whole[rows])
    np.testing.assert_array_equal(sharded[:, [2, 4]], whole[:, [2, 4]])
    rows = np.array([9, 0, 6, 2])
    actions = np.array([0, 1, 3, 5])
    np.testing.assert_array_equal(sharded[rows, actions], whole[rows, actions])
    np.testing.assert_array_equal(sharded[-1], whole[-1])
    with pytest.raises(IndexError):
        sharded[10]


def test_compressed_shards_expand_to_memmaps_and_global_replay_ids(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"schema": {"version": 4}}))
    records = []
    for i in range(2):
        path = tmp_path / f"shard_{i}.npz"
        np.savez_compressed(
            path,
            observations=np.full((4, 6), i, dtype=np.int32),
            masks=np.ones((4, 3), dtype=np.uint8),
            actions=np.zeros(4, np.int32),
            replay_ids=np.zeros(4, np.int32),
            seats=np.arange(4) % 2,
            flags=np.zeros(4, np.int32),
            ticks=np.arange(4),
        )
        records.append({"path": path.name, "sha256": file_sha256(path), "replay_offset": i})
    manifest = tmp_path / "dataset.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "crrl-bc-shards-v1",
                "shards": records,
                "metadata": metadata.name,
                "metadata_sha256": file_sha256(metadata),
            }
        )
    )
    arrays, meta = load_sharded_bc(manifest, tmp_path / "cache")
    assert isinstance(arrays["observations"].arrays[0], np.memmap)
    assert not arrays["observations"].arrays[0].flags.writeable
    np.testing.assert_array_equal(arrays["replay_ids"], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(arrays["observations"][[5, 0], 0], [1, 0])
    assert meta["schema"]["version"] == 4
    lazy, _ = load_sharded_bc(manifest, tmp_path / "lazy", cache_gib=256 / 1024**3)
    cache = lazy["observations"].arrays[0].cache
    assert cache.extractions == 0
    first = lazy["observations"][0]
    np.testing.assert_array_equal(lazy["observations"][[5, 0], 0], [1, 0])
    np.testing.assert_array_equal(first, np.zeros(6, np.int32))
    assert cache.evictions > 0
    assert cache.bytes <= 256
    assert sum(p.stat().st_size for p in (tmp_path / "lazy").glob("*/observations.npy")) <= 256
    metadata.write_text("{}")
    with pytest.raises(ValueError, match="metadata checksum"):
        load_sharded_bc(manifest, tmp_path / "cache")


def test_cache_reserves_scratch_space_without_removing_source_dataset(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cr_rl.data.sharded_bc import ShardArrayCache

    source = tmp_path / "source.npz"
    np.savez_compressed(
        source, observations=np.ones((4, 6), np.int32), masks=np.ones((4, 3), np.uint8)
    )
    cache = ShardArrayCache(tmp_path / "cache", 4096, minimum_free_bytes=1024)
    digest = file_sha256(source)
    np.testing.assert_array_equal(cache.read(source, digest, "observations", 0), np.ones(6))
    monkeypatch.setattr(
        "cr_rl.data.sharded_bc.shutil.disk_usage", lambda root: SimpleNamespace(free=0)
    )
    with pytest.raises(OSError, match="scratch space"):
        cache.read(source, digest, "masks", 0)
    assert not list((tmp_path / "cache").glob("*/*.npy"))
    assert source.exists() and file_sha256(source) == digest
