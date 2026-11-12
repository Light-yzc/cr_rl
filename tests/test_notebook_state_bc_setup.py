"""Bounded notebook bootstrap checks; no cloud, Java or production tensors."""
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def setup():
    path = Path(__file__).resolve().parents[1] / "scripts/notebook_state_bc_setup.py"
    spec = importlib.util.spec_from_file_location("notebook_setup_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_fixture(root, setup):
    root.mkdir(parents=True)
    metadata = {
        "decision_ticks": 8, "training_ready": True, "smoke_only": False,
        "schema": {"header_fields": ["entity_count", "elixir"], "entity_fields": ["kind", "row"],
                   "header_ints": 2, "entity_ints": 2, "max_entities": 1, "seat_ints": 4},
        "vocabulary": {"names": ["knight", "giant"]},
        "replays": [{"replay_tag": str(index), "repair_records": [{"string": 'brackets [ ] { } and "quote"'}]}
                    for index in range(3)],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    records = []
    for index in range(3):
        shard = root / "shards" / f"replay_{index}.npz"
        shard.parent.mkdir(exist_ok=True)
        shard.write_bytes(f"immutable-shard-{index}".encode())
        records.append({"path": f"shards/{shard.name}", "sha256": setup.sha256_file(shard), "replay_offset": index})
    manifest = root / "dataset.manifest.json"
    manifest.write_text(json.dumps({"format": "crrl-bc-shards-v1", "replays": 3, "rows": 36,
                                    "metadata": "metadata.json", "metadata_sha256": setup.sha256_file(root / "metadata.json"),
                                    "shards": records}), encoding="utf-8")
    return manifest, metadata


def checkpoint_fixture(path, manifest, metadata, setup, *, step=606, epoch=1, next_episode=0, **changes):
    import torch
    names = metadata["vocabulary"]["names"]
    settings = {"chunk_size": 32, "freeze_frame": False, "precision": "fp16", "learning_rate": 3e-5,
                "weight_decay": .01, "position_coef": 1., "tile_label_sigma": 1.,
                "timing_objective": "natural", "timing_target_action_prior": .2,
                "sampled_action_prior": .065, "seed": 20261110, "decision_ticks": 8,
                "max_grad_norm": 1., "recurrent_matrix_dtype": "float32", "kv_cache_dtype": "fp16",
                "vocabulary_sha256": hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
    state = {"format": "crrl-state-bc-v1", "model_architecture_version": 3,
             "decision_ticks": 8, "dataset_fingerprint": setup.sha256_file(manifest),
             "config": {"header_dim": 2, "entity_feature_dim": 2, "header_fields": ["entity_count", "elixir"],
                        "entity_fields": ["kind", "row"], "vocab_size": 2, "temporal_dim": 8,
                        "temporal_heads": 2, "frame_encoder_fp32": True},
             "temporal_config": {"width": 8, "heads": 2, "kv_heads": 1, "head_dim": 4,
                                 "pattern": ("kda", "gqa")},
             "progress": {"epoch": epoch, "next_episode": next_episode, "optimizer_steps": step},
             "training_config": settings, "split": {"0": "train", "1": "validation", "2": "heldout"},
             "model": {"tiny_weight": torch.zeros(2, 2)}, "optimizer": {"state": {}, "param_groups": []},
             "rng": {}, "scaler": None, **changes}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def tar_fixture(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in entries:
            entry = tarfile.TarInfo(name)
            if kind == "symlink":
                entry.type = tarfile.SYMTYPE
                entry.linkname = "../outside"
                archive.addfile(entry)
            elif kind == "hardlink":
                entry.type = tarfile.LNKTYPE
                entry.linkname = "../outside"
                archive.addfile(entry)
            else:
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))


def test_source_extract_validates_all_files_and_reuses_same_bundle(tmp_path, setup):
    payload = b"# trusted source\n"
    manifest = json.dumps({"scripts/train_state_bc.py": hashlib.sha256(payload).hexdigest()}).encode()
    archive = tmp_path / "source.tar.gz"
    tar_fixture(archive, [("scripts/train_state_bc.py", payload, "file"), ("source_manifest.json", manifest, "file")])
    root = setup.extract_source(archive, tmp_path / "work")
    assert root == tmp_path / "work/src_bundle"
    assert setup.extract_source(archive, tmp_path / "work") == root
    (root / "scripts/train_state_bc.py").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        setup.extract_source(archive, tmp_path / "work")


@pytest.mark.parametrize("name,kind", [("../outside", "file"), ("/outside", "file"),
                                      ("scripts/link", "symlink"), ("scripts/link", "hardlink")])
def test_source_extract_refuses_escape_and_links(tmp_path, setup, name, kind):
    archive = tmp_path / "source.tar.gz"
    tar_fixture(archive, [(name, b"no", kind)])
    with pytest.raises(ValueError, match="escapes|link"):
        setup.extract_source(archive, tmp_path / "work")
    assert not (tmp_path / "work/src_bundle").exists()
    assert not (tmp_path / "outside").exists()


def test_stream_copy_does_not_prehash_cloud_file_or_use_read_bytes_and_cached_copy_skips_source(
    tmp_path, setup, monkeypatch
):
    source = tmp_path / "drive.pt"
    with source.open("wb") as stream:
        for _ in range(10):
            stream.write(b"a" * 1024**2)
    original_hash = setup.sha256_file
    hashed = []

    def record_hash(path):
        hashed.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(setup, "sha256_file", record_hash)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("Never load large files with read_bytes"))
    destination = tmp_path / "cache/latest.pt"
    receipt = setup.copy_verified(source, destination)
    assert receipt["copied"] and source not in hashed
    assert destination.stat().st_size == 10 * 1024**2
    hashed.clear()
    assert not setup.copy_verified(source, destination)["copied"]
    assert source not in hashed


def test_wrong_checksum_preserves_previous_file_and_cleans_uploading(tmp_path, setup):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("new"); destination.write_text("previous")
    with pytest.raises(OSError, match="checksum"):
        setup.copy_verified(source, destination, "0" * 64)
    assert destination.read_text() == "previous"
    assert not destination.with_name("destination.uploading").exists()


def test_dataset_stages_only_referenced_files_and_commits_manifest_last(tmp_path, setup):
    run = tmp_path / "drive_run"
    source, _ = dataset_fixture(run / "dataset", setup)
    (run / "audit").mkdir(); (run / "audit/huge.log").write_text("not needed")
    work = tmp_path / "work"
    observed = []

    def progress(index, total, relative):
        assert not (work / "dataset/dataset.manifest.json").exists()
        observed.append((index, total, relative))

    local = setup.stage_dataset(run, work, progress)
    assert setup.sha256_file(local) == setup.sha256_file(source)
    assert [row[0] for row in observed] == [1, 2, 3]
    assert not (work / "audit").exists()
    manifest, metadata = setup.dataset_contract(local)
    assert len(manifest["shards"]) == 3 and "replays" not in metadata
    assert metadata["schema"]["seat_ints"] == 4


def test_incomplete_dataset_copy_never_publishes_manifest(tmp_path, setup):
    run = tmp_path / "drive_run"
    source, _ = dataset_fixture(run / "dataset", setup)
    (source.parent / "shards/replay_1.npz").write_text("corrupt")
    with pytest.raises(OSError, match="checksum"):
        setup.stage_dataset(run, tmp_path / "work")
    assert not (tmp_path / "work/dataset/dataset.manifest.json").exists()


@pytest.mark.parametrize("field,value", [("decision_ticks", 5), ("model_architecture_version", 2),
                                        ("dataset_fingerprint", "different")])
def test_checkpoint_inspection_binds_architecture_interval_and_dataset(tmp_path, setup, field, value):
    manifest, metadata = dataset_fixture(tmp_path / "dataset", setup)
    checkpoint = tmp_path / "latest.pt"
    checkpoint_fixture(checkpoint, manifest, metadata, setup, **{field: value})
    with pytest.raises(ValueError, match="architecture|fingerprint|grid"):
        setup.inspect_checkpoint(checkpoint, manifest, verify_file=False)


def test_mmap_preview_avoids_full_checkpoint_sha_and_can_check_split(tmp_path, setup, monkeypatch):
    manifest, metadata = dataset_fixture(tmp_path / "dataset", setup)
    checkpoint = tmp_path / "latest.pt"
    checkpoint_fixture(checkpoint, manifest, metadata, setup)
    original = setup.sha256_file

    def no_cloud_sha(path):
        assert Path(path) != checkpoint, "Metadata-only preview must not stream 2GB from Drive"
        return original(path)

    monkeypatch.setattr(setup, "sha256_file", no_cloud_sha)
    info = setup.inspect_checkpoint(checkpoint, manifest, verify_file=False)
    assert info["checkpoint_sha256"] is None and info["progress"]["optimizer_steps"] == 606
    assert info["train_seats_per_epoch"] == 2
    with pytest.raises(ValueError, match="split"):
        setup.inspect_checkpoint(checkpoint, manifest, {"0": "heldout"}, verify_file=False)


@pytest.mark.parametrize("steps,chosen", [((600, 620, 610), "notebook"), ((630, 620, 610), "source"),
                                         ((600, 620, 640), "local"), ((620, 620, 620), "local")])
def test_resume_selects_highest_progress_with_local_tie_and_copies_only_selected(
    tmp_path, setup, monkeypatch, steps, chosen
):
    manifest, metadata = dataset_fixture(tmp_path / "work/dataset", setup)
    source, notebook, work = tmp_path / "source", tmp_path / "notebook", tmp_path / "work"
    for run, step in zip((source, notebook, work), steps):
        checkpoint_fixture(run / "training/latest.pt", manifest, metadata, setup, step=step)
    copy_calls = []
    original = setup.copy_verified

    def copied(source_path, destination, *args):
        copy_calls.append(Path(source_path))
        return original(source_path, destination, *args)

    monkeypatch.setattr(setup, "copy_verified", copied)
    staged = setup.stage_resume(source, notebook, work)
    assert staged["info"]["progress"]["optimizer_steps"] == max(steps)
    assert staged["source_run"] == {"source": source, "notebook": notebook, "local": work}[chosen]
    assert copy_calls == [staged["source_run"] / "training/latest.pt"]
    assert staged["checkpoint"] == work / "training/latest.pt"


def test_fork_keeps_original_metrics_separate_and_notebook_resume_restores_own_history(tmp_path, setup):
    manifest, metadata = dataset_fixture(tmp_path / "work/dataset", setup)
    source, notebook, work = tmp_path / "source", tmp_path / "notebook", tmp_path / "work"
    checkpoint_fixture(source / "training/latest.pt", manifest, metadata, setup)
    (source / "training/metrics.jsonl").write_text("old history\n")
    staged = setup.stage_resume(source, notebook, work)
    assert not staged["notebook_continuation"] and not (work / "training/metrics.jsonl").exists()
    checkpoint_fixture(notebook / "training/latest.pt", manifest, metadata, setup, step=607)
    (notebook / "training/metrics.jsonl").write_text("notebook history\n")
    staged = setup.stage_resume(source, notebook, work)
    assert staged["notebook_continuation"]
    assert (work / "training/metrics.jsonl").read_text() == "notebook history\n"


def test_equal_successful_steps_prefer_more_consumed_seats_before_local_priority(tmp_path, setup):
    manifest, metadata = dataset_fixture(tmp_path / "work/dataset", setup)
    source, notebook, work = tmp_path / "source", tmp_path / "notebook", tmp_path / "work"
    checkpoint_fixture(source / "training/latest.pt", manifest, metadata, setup, step=606, next_episode=1)
    checkpoint_fixture(work / "training/latest.pt", manifest, metadata, setup, step=606, next_episode=0)
    staged = setup.stage_resume(source, notebook, work)
    assert staged["source_run"] == source
    assert staged["info"]["progress"]["next_episode"] == 1


def test_corrupt_newer_candidate_is_rejected_without_preventing_valid_continuation(tmp_path, setup):
    manifest, metadata = dataset_fixture(tmp_path / "work/dataset", setup)
    source, notebook, work = tmp_path / "source", tmp_path / "notebook", tmp_path / "work"
    checkpoint_fixture(source / "training/latest.pt", manifest, metadata, setup)
    (notebook / "training").mkdir(parents=True)
    (notebook / "training/latest.pt").write_bytes(b'')
    staged = setup.stage_resume(source, notebook, work)
    assert staged["source_run"] == source and len(staged["rejected_candidates"]) == 1


def test_checkpoint_invalid_episode_cursor_is_never_a_latest_candidate(tmp_path, setup):
    manifest, metadata = dataset_fixture(tmp_path / "dataset", setup)
    checkpoint = tmp_path / "latest.pt"
    checkpoint_fixture(checkpoint, manifest, metadata, setup, next_episode=3)
    with pytest.raises(ValueError, match="next_episode"):
        setup.inspect_checkpoint(checkpoint, manifest, verify_file=False)


def test_same_path_copy_never_rewrites_the_checkpoint(tmp_path, setup):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"local checkpoint")
    before = checkpoint.stat().st_mtime_ns
    assert not setup.copy_verified(checkpoint, checkpoint)["copied"]
    assert checkpoint.stat().st_mtime_ns == before
    assert not (tmp_path / "latest.pt.uploading").exists()


def test_built_cli_preserves_saved_settings_and_real_subprocess_import_parse_preflight(tmp_path, setup):
    manifest, metadata = dataset_fixture(tmp_path / "dataset", setup)
    checkpoint = tmp_path / "work/training/latest.pt"
    saved = checkpoint_fixture(checkpoint, manifest, metadata, setup)
    info = setup.inspect_checkpoint(checkpoint, manifest)
    root = Path(__file__).resolve().parents[1]
    registry = tmp_path / "split_registry.json"
    registry.write_text(json.dumps({"assignments": {
        json.dumps([None, None, str(index)]): partition
        for index, partition in enumerate(("train", "validation", "heldout"))
    }}))
    argv = setup.build_trainer_argv(info, source_root=root, dataset_manifest=manifest,
                                    checkpoint=checkpoint, out=checkpoint.parent,
                                    backup_dir=tmp_path / "drive/training", epochs=3,
                                    ram_limit_gib=12, split_registry=registry)
    assert "--resume" in argv and "--init-from-bc" not in argv and "--init-from-state" not in argv
    assert argv[argv.index("--position-coef") + 1] == str(saved["training_config"]["position_coef"])
    assert "--no-freeze-frame" in argv and "--frame-encoder-fp32" in argv
    assert argv[argv.index("--epochs") + 1] == "3"
    report = setup.preflight(argv, root, require_cuda=False, env={"CUDA_VISIBLE_DEVICES": "-1"})
    assert report["decision_ticks"] == 8 and not report["cuda_checked"]
    assert Path(report["python"]) == Path(sys.executable)
    with pytest.raises(RuntimeError, match="unrecognized arguments"):
        setup.preflight([*argv, "--invalid-notebook-flag"], root, require_cuda=False)
    registry.write_text(json.dumps({"assignments": {}}))
    with pytest.raises(RuntimeError, match="split differs"):
        setup.preflight(argv, root, require_cuda=False)
