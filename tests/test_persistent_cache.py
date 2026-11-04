import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import zstandard as zstd

from cr_rl.data.persistent_cache import publish_validation_cache, restore_validation_cache
from cr_rl.data.sharded_bc import MappedShardArray
from cr_rl.data.validation_cache import prepare_validation_arrays
from cr_rl.training.persistence import file_sha256


def cache(tmp_path):
    signature = hashlib.sha256(b"test-cache").hexdigest()
    source = tmp_path / "source" / signature
    source.mkdir(parents=True)
    observations = np.zeros((128, 7040), np.int32)
    observations[:, :20] = np.arange(128)[:, None]
    np.save(source / "observations.npy", observations)
    np.save(source / "masks.npy", np.ones((32, 2307), np.uint8))
    record = {
        "signature": signature,
        "dataset_sha256": "dataset",
        "context": 128,
        "sha256": {name: file_sha256(source / f"{name}.npy") for name in ("observations", "masks")},
    }
    (source / "complete.json").write_text(json.dumps(record))
    remote = tmp_path / "drive"
    result = publish_validation_cache(source, remote, reserve_bytes=0)
    return source, remote, record, result


def restore(remote, destination, record):
    return restore_validation_cache(
        remote,
        destination,
        signature=record["signature"],
        dataset_sha256="dataset",
        context=128,
        maximum_bytes=16 * 1024**2,
        reserve_bytes=0,
    )


def test_streamed_archive_restores_exact_arrays_and_idempotently_reuses_publication(tmp_path):
    source, remote, record, published = cache(tmp_path)
    assert published["archive_bytes"] < published["uncompressed_bytes"] / 20
    restored = tmp_path / "restored"
    result = restore(remote, restored, record)
    assert result["event"] == "restored"
    for name in ("observations.npy", "masks.npy", "complete.json"):
        assert file_sha256(restored / name) == file_sha256(source / name)
    assert publish_validation_cache(source, remote, reserve_bytes=0)["event"] == "already_published"
    with pytest.raises(FileExistsError):
        restore(remote, restored, record)
    with pytest.raises(ValueError, match="binding"):
        restore_validation_cache(
            remote,
            tmp_path / "bad",
            signature=record["signature"],
            dataset_sha256="different",
            context=128,
            maximum_bytes=16 * 1024**2,
            reserve_bytes=0,
        )


def test_corrupt_or_partial_remote_archives_never_publish_a_local_completion_marker(tmp_path):
    _, remote, record, published = cache(tmp_path)
    package = remote / record["signature"] / f"{published['archive_sha256']}.tar.zst"
    with package.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="checksum"):
        restore(remote, destination, record)
    assert not (destination / "complete.json").exists()
    assert not list(tmp_path.glob(".cache-restore-*"))


@pytest.mark.parametrize("name", ["../escaped", "observations.npy"])
def test_checksums_do_not_allow_path_traversal_or_missing_members(tmp_path, name):
    source, remote, record, published = cache(tmp_path)
    raw = io.BytesIO()
    data = b"escape" if name.startswith("..") else (source / name).read_bytes()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    encoded = zstd.ZstdCompressor().compress(raw.getvalue())
    digest = hashlib.sha256(encoded).hexdigest()
    directory = remote / record["signature"]
    (directory / f"{digest}.tar.zst").write_bytes(encoded)
    published.update(archive_sha256=digest, archive_bytes=len(encoded))
    (directory / "archive.json").write_text(json.dumps(published))
    with pytest.raises(ValueError, match="archive"):
        restore(remote, tmp_path / "invalid", record)
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "invalid/complete.json").exists()


def test_validation_preparation_restores_without_rebuilding_and_falls_back_after_corruption(
    tmp_path,
    monkeypatch,
):
    obs = MappedShardArray([np.arange(48, dtype=np.int32).reshape(8, 6)])
    masks = MappedShardArray([np.ones((8, 3), np.uint8)])
    ids = seats = np.zeros(8, int)
    groups, positions = {(0, 0): list(range(8))}, {(0, 0): {i: i for i in range(8)}}
    remote, events = tmp_path / "drive", []

    def prepare(root):
        return prepare_validation_arrays(
            obs,
            masks,
            [5, 7],
            ids,
            seats,
            groups,
            positions,
            3,
            root,
            "dataset",
            persistent_root=remote,
            persistence_event=events.append,
        )

    reference = prepare(tmp_path / "first")
    import cr_rl.data.validation_cache as module

    original = module.compact_field
    monkeypatch.setattr(module, "compact_field", lambda *a, **k: pytest.fail("Rebuilt cache"))
    restored = prepare(tmp_path / "second")
    assert events[-1]["event"] == "restored"
    np.testing.assert_array_equal(reference[0][[3, 5, 7]], restored[0][[3, 5, 7]])
    metadata = next(remote.glob("*/archive.json"))
    record = json.loads(metadata.read_text())
    (metadata.parent / f"{record['archive_sha256']}.tar.zst").write_bytes(b"corrupt")
    monkeypatch.setattr(module, "compact_field", original)
    rebuilt = prepare(tmp_path / "third")
    assert any(e["event"] == "restore_failed_rebuilding" for e in events)
    np.testing.assert_array_equal(reference[0][[3, 5, 7]], rebuilt[0][[3, 5, 7]])
