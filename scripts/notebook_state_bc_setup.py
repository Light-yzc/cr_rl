"""Notebook bootstrap for an existing continuous BC run, without simulator setup.

Torch is imported only by checkpoint inspection or a separate preflight process.
Large archives/checkpoints are always copied and hashed in bounded-size blocks.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

BLOCK_BYTES = 4 * 1024**2
CONTRACT_FIELDS = {"schema", "vocabulary", "decision_ticks", "training_ready", "smoke_only", "label_source", "excluded_replay_ids"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(root, relative):
    relative = str(relative)
    name = PurePosixPath(relative)
    if "\\" in relative or name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Artifact path escapes its owned directory: {relative}")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"Artifact path escapes its owned directory: {relative}")
    return target


def _owned_subdir(workspace, name):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / name
    if target.is_symlink() or target.resolve().parent != workspace:
        raise ValueError("Notebook cache directory escapes its workspace")
    return target


def copy_verified(source, destination, expected_sha256=None):
    """Atomic verified copy; an already valid local cache does not read Drive."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        digest = sha256_file(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise OSError(f"Artifact checksum mismatch: {source.name}")
        return {"path": str(destination), "sha256": digest,
                "bytes": destination.stat().st_size, "copied": False}
    receipt_path = destination.with_name(destination.name + ".copy.json")
    digest = expected_sha256
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected a lowercase SHA256 digest")
    before = None
    if digest is None:
        stat = source.stat()
        before = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
            if receipt.get("source") == str(source.resolve()) and receipt.get("source_stat") == before:
                digest = receipt.get("sha256")
    if digest is not None and destination.is_file() and sha256_file(destination) == digest:
        return {"path": str(destination), "sha256": digest,
                "bytes": destination.stat().st_size, "copied": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".uploading")
    try:
        if before is None:
            stat = source.stat()
            before = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        source_digest = hashlib.sha256()
        with source.open("rb") as src, temporary.open("wb") as dst:
            for block in iter(lambda: src.read(BLOCK_BYTES), b""):
                source_digest.update(block)
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        copied_digest = source_digest.hexdigest()
        if digest is not None and copied_digest != digest:
            raise OSError(f"Artifact checksum mismatch: {source.name}")
        digest = copied_digest
        stat = source.stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != before:
            raise OSError(f"Source changed while copying: {source.name}")
        if sha256_file(temporary) != digest:
            raise OSError(f"Artifact checksum mismatch: {source.name}")
        temporary.replace(destination)
        receipt = {"source": str(source.resolve()), "source_stat": before, "sha256": digest,
                   "bytes": destination.stat().st_size}
        receipt_tmp = receipt_path.with_name(receipt_path.name + ".tmp")
        receipt_tmp.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_tmp.replace(receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest,
            "bytes": destination.stat().st_size, "copied": True}


def _source_manifest(root, manifest_name):
    manifest = json.loads(_checked(root, manifest_name).read_text(encoding="utf-8"))
    files = manifest.get("files", manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError("Source manifest must contain file SHA256 records")
    for relative, record in files.items():
        digest = record["sha256"] if isinstance(record, dict) else record
        target = _checked(root, relative)
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Source manifest checksum mismatch: {relative}")
    return files


def extract_source(archive, workspace, manifest_name="source_manifest.json"):
    """Extract verified regular files to WORK/src_bundle; never follow links."""
    archive, workspace = Path(archive), Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "src_bundle"
    archive_digest = sha256_file(archive)
    marker = ".notebook_source.json"
    if target.exists():
        saved = json.loads((target / marker).read_text(encoding="utf-8")) if (target / marker).is_file() else {}
        if target.is_symlink() or saved.get("archive_sha256") != archive_digest:
            raise ValueError("Existing source bundle differs; choose a fresh workspace")
        _source_manifest(target, manifest_name)
        return target
    with tempfile.TemporaryDirectory(prefix=".notebook-source-", dir=workspace) as temporary:
        pending = Path(temporary) / "bundle"
        pending.mkdir()
        names = set()
        with tarfile.open(archive, "r:gz") as packed:
            for entry in packed:
                if entry.isdir() and entry.name in (".", "./"):
                    continue
                output = _checked(pending, entry.name)
                if entry.issym() or entry.islnk() or not (entry.isfile() or entry.isdir()):
                    raise ValueError(f"Source archive contains a link/special file: {entry.name}")
                if entry.name in names:
                    raise ValueError("Source archive has duplicate members")
                names.add(entry.name)
                if entry.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with packed.extractfile(entry) as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=BLOCK_BYTES)
        files = _source_manifest(pending, manifest_name)
        regular = {str(path.relative_to(pending)).replace("\\", "/") for path in pending.rglob("*") if path.is_file()}
        if regular != set(files) | {manifest_name}:
            raise ValueError("Source archive contains files absent from its checksum manifest")
        (pending / marker).write_text(json.dumps({"archive_sha256": archive_digest}), encoding="utf-8")
        pending.replace(target)
    return target


def _metadata_contract(path, with_replay_identities=False):
    """Keep schema/vocabulary only; do not materialize the large repair ledger."""
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    tokens = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]]')
    position = 0
    result = {}

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    position = whitespace(position)
    if text[position:position + 1] != "{":
        raise ValueError("Dataset metadata must be a JSON object")
    position += 1
    while True:
        position = whitespace(position)
        if text[position:position + 1] == "}":
            break
        key, position = decoder.raw_decode(text, position)
        position = whitespace(position)
        if text[position:position + 1] != ":":
            raise ValueError("Malformed metadata field")
        position = whitespace(position + 1)
        if key == "replays" and with_replay_identities:
            if text[position:position + 1] != "[":
                raise ValueError("Metadata replays must be an array")
            position += 1
            replays = []
            while True:
                position = whitespace(position)
                if text[position:position + 1] == "]":
                    position += 1
                    break
                replay, position = decoder.raw_decode(text, position)
                replays.append({name: replay[name] for name in (
                    "canonical_game_signature", "game_signature", "source", "source_revision", "replay_tag"
                ) if name in replay})
                del replay
                position = whitespace(position)
                if text[position:position + 1] == ",":
                    position += 1
                elif text[position:position + 1] != "]":
                    raise ValueError("Malformed metadata replay separator")
            result[key] = replays
        elif key in CONTRACT_FIELDS or text[position:position + 1] not in ("[", "{"):
            value, position = decoder.raw_decode(text, position)
            if key in CONTRACT_FIELDS:
                result[key] = value
        else:
            depth = 0
            for token in tokens.finditer(text, position):
                token_value = token.group(0)
                if token_value[0] == '"':
                    continue
                depth += 1 if token_value in ("[", "{") else -1
                if depth == 0:
                    position = token.end()
                    break
            else:
                raise ValueError("Unclosed metadata field")
        position = whitespace(position)
        if text[position:position + 1] == ",":
            position += 1
        elif text[position:position + 1] != "}":
            raise ValueError("Malformed metadata separator")
    return result


def dataset_contract(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "crrl-bc-shards-v1":
        raise ValueError("Expected the existing complete BC shard dataset")
    metadata_path = _checked(manifest_path.parent, manifest["metadata"])
    if sha256_file(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("Dataset metadata checksum mismatch")
    metadata = _metadata_contract(metadata_path)
    if metadata.get("decision_ticks") != 8 or not metadata.get("training_ready") or metadata.get("smoke_only"):
        raise ValueError("Only the admitted 8-tick training-ready BC dataset can resume")
    return manifest, metadata


def stage_dataset(drive_run, workspace, progress=None):
    """Copy existing verified compressed shards; commit the original manifest last."""
    source = Path(drive_run) / "dataset"
    manifest_path = source / "dataset.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "crrl-bc-shards-v1":
        raise ValueError("Expected the existing complete BC shard dataset")
    fingerprint = sha256_file(manifest_path)
    target = _owned_subdir(workspace, "dataset")
    target.mkdir(parents=True, exist_ok=True)
    committed = target / "dataset.manifest.json"
    if committed.exists() and sha256_file(committed) != fingerprint:
        raise ValueError("Workspace already contains a different committed dataset")
    copy_verified(_checked(source, manifest["metadata"]), _checked(target, manifest["metadata"]),
                  manifest["metadata_sha256"])
    metadata = _metadata_contract(_checked(target, manifest["metadata"]))
    if metadata.get("decision_ticks") != 8 or not metadata.get("training_ready") or metadata.get("smoke_only"):
        raise ValueError("Only the admitted 8-tick training-ready BC dataset can resume")
    for index, record in enumerate(manifest["shards"], start=1):
        copy_verified(_checked(source, record["path"]), _checked(target, record["path"]), record["sha256"])
        if progress:
            progress(index, len(manifest["shards"]), record["path"])
    copy_verified(manifest_path, committed, fingerprint)
    return committed


def stage_resume(original_run, notebook_run, workspace, manifest=None, expected_split=None, extra_runs=()):
    """Preview small mmap metadata for both candidates, copy only the newest valid one."""
    original_run, notebook_run = Path(original_run), Path(notebook_run)
    manifest = Path(manifest) if manifest is not None else Path(workspace) / "dataset/dataset.manifest.json"
    candidates, rejected = [], []
    workspace = Path(workspace).resolve()
    runs = [(original_run, False)]
    runs.extend((Path(run), True) for run in extra_runs)
    runs.extend([(notebook_run, True), (workspace, True)])
    for priority, (run, continued) in enumerate(runs):
        path = run / "training/latest.pt"
        if not path.is_file():
            continue
        try:
            preview = inspect_checkpoint(path, manifest, expected_split, verify_file=False)
            candidates.append((preview["progress"]["optimizer_steps"], priority, continued, run, preview))
        except (ValueError, OSError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            rejected.append({"source_run": str(run), "error": str(error)})
    if not candidates:
        raise FileNotFoundError(f"No valid existing state checkpoint; initialization fallback is disabled: {rejected}")
    _, _, continued, selected, preview = max(candidates, key=lambda item: (
        item[0], item[4]["progress"]["epoch"], item[4]["progress"]["next_episode"], item[1]
    ))
    source = selected / "training/latest.pt"
    output = _owned_subdir(workspace, "training")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    if not continued and (output / "metrics.jsonl").exists():
        raise ValueError("Original-run fork cannot reuse notebook metric history; choose a fresh workspace")
    receipt = copy_verified(source, checkpoint)
    local = inspect_checkpoint(checkpoint, manifest, expected_split, verify_file=False)
    if local["progress"] != preview["progress"] or local["dataset_fingerprint"] != preview["dataset_fingerprint"]:
        raise ValueError("Selected checkpoint changed between preview and local staging")
    # Original metrics belong to its historical run; keep the notebook fork's
    # new history separate. On later notebook resumes restore its own history.
    if continued:
        for name in ("metrics.jsonl", "run_config.json", "split.json", "summary.json"):
            if (selected / "training" / name).is_file():
                copy_verified(selected / "training" / name, output / name)
    return {"checkpoint": checkpoint, "out": output, "source_run": selected,
            "notebook_continuation": continued, "info": {**local, "checkpoint_sha256": receipt["sha256"]},
            "rejected_candidates": rejected}


def inspect_checkpoint(checkpoint, manifest, expected_split=None, verify_file=True):
    """Read metadata via CPU mmap, without constructing a model or copying tensors."""
    import torch

    checkpoint, manifest = Path(checkpoint), Path(manifest)
    _, metadata = dataset_contract(manifest)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    try:
        if state.get("format") != "crrl-state-bc-v1" or state.get("model_architecture_version") != 3:
            raise ValueError("Checkpoint is not a resumable architecture-v3 BC stage")
        if state.get("decision_ticks") != 8 or state.get("dataset_fingerprint") != sha256_file(manifest):
            raise ValueError("Checkpoint decision grid or dataset fingerprint differs")
        settings, model, temporal = state["training_config"], state["config"], state["temporal_config"]
        if settings.get("precision") != "fp16" or settings.get("decision_ticks") != 8:
            raise ValueError("Expected the existing FP16, 8-tick training settings")
        if not model.get("frame_encoder_fp32"):
            raise ValueError("Checkpoint lacks the required FP32 frame-encoder safety boundary")
        schema = metadata["schema"]
        if (model["header_dim"] != schema["header_ints"]
                or model["entity_feature_dim"] != schema["entity_ints"]
                or tuple(model["header_fields"]) != tuple(schema["header_fields"])
                or tuple(model["entity_fields"]) != tuple(schema["entity_fields"])):
            raise ValueError("Checkpoint model schema differs from the admitted dataset")
        names = metadata["vocabulary"]["names"]
        vocabulary = hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if model["vocab_size"] != len(names) or settings.get("vocabulary_sha256") != vocabulary:
            raise ValueError("Checkpoint vocabulary differs from the admitted dataset")
        if model["temporal_dim"] != temporal["width"] or model["temporal_heads"] != temporal["heads"]:
            raise ValueError("Checkpoint temporal configuration is inconsistent")
        progress = state["progress"]
        if any(not isinstance(progress.get(key), int) or isinstance(progress[key], bool) or progress[key] < 0
               for key in ("epoch", "next_episode", "optimizer_steps")):
            raise ValueError("Checkpoint episode progress is invalid")
        split = state["split"]
        train_seats = 2 * sum(value == "train" for value in split.values())
        if not train_seats or progress["next_episode"] > train_seats:
            raise ValueError("Checkpoint next_episode exceeds its fixed training seats")
        if expected_split is not None and split != {str(key): value for key, value in expected_split.items()}:
            raise ValueError("Checkpoint split differs from the fixed game assignments")
        if "optimizer" not in state or "rng" not in state or "scaler" not in state:
            raise ValueError("Checkpoint lacks optimizer/RNG/scaler required for continuation")
        # Only small plain metadata leaves this function; mapped storage dies here.
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint) if verify_file else None,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "dataset_fingerprint": state["dataset_fingerprint"], "progress": dict(progress),
                "training_config": dict(settings), "model_config": dict(model),
                "temporal_config": dict(temporal), "split": dict(split),
                "train_seats_per_epoch": train_seats}
    finally:
        del state
        gc.collect()


def build_trainer_argv(info, *, source_root, dataset_manifest, checkpoint, out, backup_dir,
                       epochs=3, ram_limit_gib=40, python=sys.executable, split_registry=None,
                       device="cuda", frame_microbatch=8, lr_schedule=None, warmup_steps=None,
                       min_learning_rate=None, defer_metrics=False, fused_optimizer=False,
                       float32_matmul_precision="highest", checkpoint_every_seats=32,
                       validation_every_seats=64):
    """Recreate immutable optimizer/loss settings from the saved stage."""
    if not isinstance(epochs, int) or epochs < 1 or epochs < info["progress"]["epoch"]:
        raise ValueError("Target total epochs must not precede the checkpoint")
    out, checkpoint = Path(out).resolve(), Path(checkpoint).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings, model, temporal = info["training_config"], info["model_config"], info["temporal_config"]
    argv = [str(python), str(Path(source_root).resolve() / "scripts/train_state_bc.py"), str(Path(dataset_manifest).resolve()),
            "--resume", str(checkpoint), "--out", str(out), "--backup-dir", str(Path(backup_dir).resolve()),
            "--device", device, "--epochs", str(epochs), "--ram-limit-gib", str(ram_limit_gib),
            "--frame-microbatch", str(frame_microbatch), "--temporal-width", str(temporal["width"]),
            "--temporal-heads", str(temporal["heads"]), "--kv-heads", str(temporal["kv_heads"]),
            "--pattern", ",".join(temporal["pattern"]),
            "--checkpoint-every-episodes", str(checkpoint_every_seats),
            "--validation-every-episodes", str(validation_every_seats),
            "--max-validation-episodes", "16"]
    for field in ("chunk_size", "precision", "learning_rate", "weight_decay", "position_coef", "tile_label_sigma",
                  "timing_objective", "timing_target_action_prior", "seed", "decision_ticks", "max_grad_norm"):
        argv.extend(("--" + field.replace("_", "-"), str(settings[field])))
    argv.append("--freeze-frame" if settings["freeze_frame"] else "--no-freeze-frame")
    if model.get("frame_encoder_fp32"):
        argv.append("--frame-encoder-fp32")
    if split_registry:
        argv.extend(("--split-registry", str(Path(split_registry).resolve())))
    existing_schedule = settings.get("lr_schedule")
    if lr_schedule is not None:
        argv.extend(("--lr-schedule", lr_schedule))
        if lr_schedule == "cosine" and not existing_schedule:
            argv.append("--start-lr-schedule")
    if warmup_steps is not None:
        argv.extend(("--warmup-steps", str(warmup_steps)))
    if min_learning_rate is not None:
        argv.extend(("--min-learning-rate", str(min_learning_rate)))
    if defer_metrics:
        argv.append("--defer-metrics")
    if fused_optimizer:
        argv.append("--fused-optimizer")
    if float32_matmul_precision != "highest":
        argv.extend(("--float32-matmul-precision", float32_matmul_precision))
    return argv


def preflight(argv, source_root, require_cuda=True, env=None, timeout=120):
    """Validate this Python, source imports and the exact argv in a fresh process."""
    source_root = Path(source_root).resolve()
    process_env = {**os.environ, **(env or {})}
    process_env["PYTHONPATH"] = str(source_root / "src")
    process_env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = r'''
import importlib.util,json,pathlib,sys,torch,cr_rl
root=pathlib.Path(sys.argv[1]).resolve()
actual=pathlib.Path(cr_rl.__file__).resolve()
if not actual.is_relative_to(root/'src'):raise RuntimeError('Stale cr_rl import: '+str(actual))
argv=json.loads(sys.argv[2]);need_cuda=sys.argv[3]=='1'
spec=importlib.util.spec_from_file_location('notebook_train_preflight',argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
args=module.parse_args(argv[2:])
if not args.resume or args.init_from_bc or args.init_from_state:raise RuntimeError('Notebook must resume the saved BC stage')
saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
module.ModelConfig(**saved['config']);module.StateTemporalConfig(**saved['temporal_config'])
for field in ('chunk_size','freeze_frame','precision','learning_rate','weight_decay','position_coef','tile_label_sigma','timing_objective','timing_target_action_prior','seed','decision_ticks','max_grad_norm'):
 if getattr(args,field)!=saved['training_config'][field]:raise RuntimeError('Requested resume setting differs: '+field)
helper_spec=importlib.util.spec_from_file_location('notebook_setup_preflight',sys.argv[4])
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
manifest_path=pathlib.Path(args.dataset)
manifest=json.loads(manifest_path.read_text())
metadata=helper._metadata_contract(manifest_path.parent/manifest['metadata'],with_replay_identities=True)
registry=json.loads(args.split_registry.read_text())['assignments'] if args.split_registry else None
from cr_rl.training.state_bc import fixed_replay_split
split=fixed_replay_split(metadata,map(int,saved['split']),args.validation_fraction,registry)
if {str(key):value for key,value in split.items()}!=saved['split']:raise RuntimeError('Fixed replay split differs from checkpoint')
del saved
if need_cuda:
 if not torch.version.cuda or not torch.cuda.is_available():raise RuntimeError('CUDA-enabled Torch and a GPU runtime are required')
 with torch.autocast('cuda',dtype=torch.float16):
  x=torch.ones((4,4),device=args.device);value=x@x
 torch.cuda.synchronize()
 if value.dtype!=torch.float16 or not torch.isfinite(value).all():raise RuntimeError('CUDA FP16 preflight failed')
print('NOTEBOOK_PREFLIGHT '+json.dumps({'python':sys.executable,'cr_rl':str(actual),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_checked':need_cuda,'resume':str(args.resume),'decision_ticks':args.decision_ticks}))
'''
    process = subprocess.run([str(argv[0]), "-c", script, str(source_root), json.dumps(argv), "1" if require_cuda else "0", str(Path(__file__).resolve())],
                             cwd=source_root, env=process_env, capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Training preflight exited {process.returncode}\n{process.stdout[-2000:]}\n{process.stderr[-6000:]}")
    for line in reversed(process.stdout.splitlines()):
        if line.startswith("NOTEBOOK_PREFLIGHT "):
            return json.loads(line[len("NOTEBOOK_PREFLIGHT "):])
    raise RuntimeError("Training preflight returned no verified report")
