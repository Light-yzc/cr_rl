"""Bounded-memory continuous-episode BC for policies with explicit temporal state.

Only one compressed replay shard is expanded at a time. Observation arrays are
memory mapped, and each adapter call owns at most one chunk. An optimizer step
occurs after an entire replay/seat, so every cached state in that sequence was
computed with the same weights. Detaching between chunks is truncated BPTT; it
does not truncate the forward memory or turn training into full-game BPTT.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.training.bc import BCBatchNormalization, hierarchical_bc_loss
from cr_rl.training.persistence import file_sha256

CHECKPOINT_FORMAT = "crrl-state-bc-v1"
FIELDS = ("observations", "masks", "actions", "ticks", "seats", "flags", "replay_ids")


def process_rss_bytes():
    """Current process RSS without a required psutil dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("Cannot inspect process RAM usage")
        return int(counters.WorkingSetSize)
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    import resource
    import sys

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def check_ram_budget(limit_gib):
    current = process_rss_bytes()
    if limit_gib and current > limit_gib * 1024**3:
        raise MemoryError(
            f"State BC RAM guard: RSS {current / 1024**3:.2f} GiB > {limit_gib:.2f} GiB"
        )
    return current


def make_state_schema(metadata):
    raw = metadata["schema"]
    header = {}
    for index, name in enumerate(raw["header_fields"]):
        if name != "reserved":
            header.setdefault(name, []).append(index)
    return SimpleNamespace(
        seat_ints=int(raw["seat_ints"]),
        header_ints=int(raw["header_ints"]),
        entity_ints=int(raw["entity_ints"]),
        max_entities=int(raw["max_entities"]),
        header={key: tuple(value) for key, value in header.items()},
        header_fields=tuple(raw["header_fields"]),
        entity_fields=tuple(raw["entity_fields"]),
    )


def replay_identity(replay):
    identity = replay.get("canonical_game_signature") or replay.get("game_signature")
    if identity:
        return str(identity)
    if not replay.get("replay_tag"):
        raise ValueError("A stable replay_tag or canonical_game_signature is required")
    return json.dumps(
        [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]],
    )


def fixed_replay_split(metadata, replay_ids, validation_fraction=0.1, registry=None):
    """Partition game identities, so seats and chunks never cross partitions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    registry = registry or {}
    assignments = {}
    identity_partitions = {}
    excluded = set(metadata.get("excluded_replay_ids", []))
    for replay_id in sorted(set(map(int, replay_ids))):
        replay = metadata["replays"][replay_id]
        identity = replay_identity(replay)
        score = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") / 2**64
        keys = [identity]
        if replay.get("replay_tag"):
            values = [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
            keys.extend((json.dumps(values), json.dumps(values, separators=(",", ":"))))
        recorded = {registry[key] for key in keys if key in registry}
        if len(recorded) > 1:
            raise ValueError("Canonical and legacy registry keys disagree for one game")
        partition = (
            next(iter(recorded))
            if recorded
            else ("validation" if score < validation_fraction else "train")
        )
        if replay_id in excluded:
            partition = "heldout"
        if partition not in ("train", "validation", "heldout"):
            raise ValueError(f"Unknown split partition: {partition}")
        previous = identity_partitions.setdefault(identity, partition)
        if previous != partition:
            raise ValueError("Duplicate game identity has inconsistent split assignments")
        assignments[replay_id] = partition
    return assignments


@dataclass
class StateEpisode:
    replay_id: int
    seat: int
    rows: np.ndarray
    # The dataset owns mmap lifetime. Exception/logging repr must never traverse
    # these resource views after their shard context has closed.
    arrays: dict = field(repr=False)
    decision_ticks: int

    def __post_init__(self):
        ticks = np.asarray(self.arrays["ticks"][self.rows], dtype=np.int64)
        if not len(ticks) or ticks[0] != 0:
            raise ValueError("State episodes must include the real opening frame at tick zero")
        if len(ticks) > 1 and not np.all(np.diff(ticks) == self.decision_ticks):
            raise ValueError("State episode contains missing, repeated, or non-grid time steps")
        labels = np.asarray(self.arrays["actions"][self.rows], dtype=np.int64)
        if ((labels < 0) | (labels >= 2307)).any():
            raise ValueError("State BC action is outside the 2307-action codec")
        legal = self.arrays["masks"]
        valid = np.asarray(self.arrays["flags"][self.rows]) == 0
        valid &= np.asarray(legal[self.rows, labels], dtype=bool)
        self.labels = labels
        self.supervised = valid
        self.normalization = (
            BCBatchNormalization.from_actions(torch.from_numpy(labels[valid].copy()))
            if valid.any()
            else None
        )

    def chunks(self, chunk_size):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for start in range(0, len(self.rows), chunk_size):
            yield start, self.rows[start : start + chunk_size]


class ReplaySequenceDataset:
    """Stream existing per-replay NPZ shards through a single owned scratch dir."""

    def __init__(
        self,
        manifest_path,
        cache_root,
        *,
        expected_decision_ticks=8,
        verify=True,
        ram_limit_gib=3,
        trust_data_files=False,
    ):
        self.path = Path(manifest_path)
        if self.path.is_dir():
            self.path = self.path / "dataset.manifest.json"
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "crrl-bc-shards-v1":
            raise ValueError("State BC requires a crrl-bc-shards-v1 manifest")
        self.fingerprint = file_sha256(self.path)
        metadata_path = self._source(self.manifest["metadata"])
        if file_sha256(metadata_path) != self.manifest["metadata_sha256"]:
            raise ValueError("State BC metadata checksum mismatch")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.decision_ticks = int(self.metadata["decision_ticks"])
        if self.decision_ticks != expected_decision_ticks:
            raise ValueError("Dataset decision_ticks differs from the requested training grid")
        self.schema = make_state_schema(self.metadata)
        self.cache_root = Path(cache_root).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.replay_ids = set()
        self.supervision_counts = {}
        self.ram_limit_gib = ram_limit_gib
        # Only small label/index arrays are read here, never observations or masks.
        self.shard_replay_ids = []
        for record in self.manifest["shards"]:
            source = self._source(record["path"])
            if verify and not trust_data_files and file_sha256(source) != record["sha256"]:
                raise ValueError(f"State BC shard checksum mismatch: {source.name}")
            with np.load(source, allow_pickle=False) as arrays:
                local_ids = arrays["replay_ids"] + int(record.get("replay_offset", 0))
                labels, flags = arrays["actions"], arrays["flags"]
                if ((labels < 0) | (labels >= 2307)).any():
                    raise ValueError("State BC action is outside the 2307-action codec")
                masks = arrays["masks"]  # One replay's small mask array, never observations.
                valid = (flags == 0) & masks[np.arange(len(labels)), labels].astype(bool)
                ids = np.unique(local_ids)
                for replay_id in ids:
                    chosen = labels[valid & (local_ids == replay_id)]
                    self.supervision_counts[int(replay_id)] = {
                        "rows": int(len(chosen)),
                        "act": int((chosen != 0).sum()),
                        "play": int(((chosen > 0) & (chosen < 2305)).sum()),
                    }
                del masks
            ids = set(map(int, ids))
            if self.replay_ids & ids:
                raise ValueError(
                    "A replay appears in multiple shards; contiguous episodes required"
                )
            if any(index < 0 or index >= len(self.metadata["replays"]) for index in ids):
                raise ValueError("Shard replay id is outside metadata")
            self.replay_ids.update(ids)
            self.shard_replay_ids.append(ids)
            check_ram_budget(self.ram_limit_gib)

    def partition_counts(self, assignments, partition):
        counts = {
            key: sum(
                value[key]
                for replay_id, value in self.supervision_counts.items()
                if assignments.get(replay_id) == partition
            )
            for key in ("rows", "act", "play")
        }
        counts["wait"] = counts["rows"] - counts["act"]
        counts["action_prior"] = counts["act"] / max(counts["rows"], 1)
        return counts

    def _source(self, relative):
        source = (self.path.parent / relative).resolve()
        if not source.is_relative_to(self.path.parent.resolve()):
            raise ValueError("Shard and metadata paths must stay inside the dataset directory")
        return source

    @contextmanager
    def _open_shard(self, record):
        # The scratch parent is user-selected; remove only our own new subdir.
        check_ram_budget(self.ram_limit_gib)
        with tempfile.TemporaryDirectory(prefix="state-bc-", dir=self.cache_root) as directory:
            directory = Path(directory)
            arrays = {}
            try:
                with zipfile.ZipFile(self._source(record["path"])) as archive:
                    required = sum(archive.getinfo(f"{field}.npy").file_size for field in FIELDS)
                    if shutil.disk_usage(directory).free < required + 64 * 1024**2:
                        raise OSError("Insufficient scratch disk space for one replay shard")
                    for field in FIELDS:
                        target = directory / f"{field}.npy"
                        with archive.open(target.name) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024**2)
                        arrays[field] = np.load(target, mmap_mode="r", allow_pickle=False)
                n = len(arrays["actions"])
                if any(len(value) != n for value in arrays.values()):
                    raise ValueError("State BC shard arrays have different row counts")
                if arrays["observations"].shape != (n, self.schema.seat_ints):
                    raise ValueError("State BC observation schema mismatch")
                if arrays["masks"].shape != (n, 2307):
                    raise ValueError("State BC action mask schema mismatch")
                yield arrays
            finally:
                for value in arrays.values():
                    value._mmap.close()

    def episodes(self, assignments, partition, *, epoch=0, seed=0):
        wanted = [
            index
            for index, ids in enumerate(self.shard_replay_ids)
            if any(assignments.get(i) == partition for i in ids)
        ]
        if partition == "train":
            np.random.default_rng(seed + epoch).shuffle(wanted)
        for index in wanted:
            record = self.manifest["shards"][index]
            with self._open_shard(record) as arrays:
                ids = np.asarray(arrays["replay_ids"]) + int(record.get("replay_offset", 0))
                for replay_id in sorted(self.shard_replay_ids[index]):
                    if assignments.get(replay_id) != partition:
                        continue
                    for seat in (0, 1):
                        rows = np.flatnonzero((ids == replay_id) & (arrays["seats"] == seat))
                        if not len(rows):
                            continue
                        order = np.argsort(arrays["ticks"][rows], kind="stable")
                        yield StateEpisode(
                            replay_id, seat, rows[order], arrays, self.decision_ticks
                        )

    def episode_groups(
        self,
        assignments,
        partition,
        *,
        batch_size,
        epoch=0,
        seed=0,
        start_episode=0,
        max_episodes=0,
    ):
        """Own all grouped mmap lifetimes; never keep views from a closed generator."""
        if batch_size < 1:
            raise ValueError("Episode batch size must be positive")
        wanted = [
            i
            for i, ids in enumerate(self.shard_replay_ids)
            if any(assignments.get(j) == partition for j in ids)
        ]
        if partition == "train":
            np.random.default_rng(seed + epoch).shuffle(wanted)
        descriptors = [
            (index, replay_id, seat)
            for index in wanted
            for replay_id in sorted(self.shard_replay_ids[index])
            if assignments.get(replay_id) == partition
            for seat in (0, 1)
        ]
        stop = (
            min(len(descriptors), start_episode + max_episodes)
            if max_episodes
            else len(descriptors)
        )
        for offset in range(start_episode, stop, batch_size):
            group = descriptors[offset : min(offset + batch_size, stop)]
            with ExitStack() as stack:
                opened = {}
                episodes = []
                for index, replay_id, seat in group:
                    record = self.manifest["shards"][index]
                    if index not in opened:
                        opened[index] = stack.enter_context(self._open_shard(record))
                    arrays = opened[index]
                    ids = arrays["replay_ids"] + int(record.get("replay_offset", 0))
                    rows = np.flatnonzero((ids == replay_id) & (arrays["seats"] == seat))
                    if not len(rows):
                        raise ValueError("Grouped dataset requires both seats in every game")
                    order = np.argsort(arrays["ticks"][rows], kind="stable")
                    episodes.append(
                        StateEpisode(replay_id, seat, rows[order], arrays, self.decision_ticks)
                    )
                yield offset, episodes


def prepare_state_chunk(episode, rows, adapter, device):
    observations = np.array(episode.arrays["observations"][rows], copy=True)
    header, entities, entity_mask = adapter(observations)
    legal = np.array(episode.arrays["masks"][rows], dtype=bool, copy=True)
    if not legal.any(-1).all():
        raise ValueError("A history frame has no legal action")
    values = {
        "header": header,
        "entities": entities,
        "entity_mask": entity_mask,
        "legal_action_mask": legal,
    }
    for argument, field in (
        ("hand_card_ids", "hand_card"),
        ("deck_card_ids", "deck"),
        ("enemy_seen_card_ids", "enemy_seen"),
    ):
        if field in adapter.schema.header:
            values[argument] = observations[:, list(adapter.schema.header[field])].astype(np.int64)
    return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


def _autocast(device, precision):
    if torch.device(device).type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


class EpisodeProfiler:
    """Opt-in CPU/CUDA stage timing; production calls allocate no events.

    CUDA events are resolved only after the episode. Nested frame/temporal
    spans are inclusive subsets of forward_loss, not additive top-level work.
    CPU times include host waits; CUDA event intervals can include device idle
    gaps caused by CPU preparation, so neither is a pure kernel profiler.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.events = []
        self.wall = {}
        self.calls = {}

    @contextmanager
    def measure(self, name):
        if self.device.type == "cuda":
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0.0) + time.perf_counter() - started
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.device.type == "cuda":
                last.record(torch.cuda.current_stream(self.device))
                self.events.append((name, first, last))

    def resolve(self):
        cuda = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            for name, first, last in self.events:
                cuda[name] = cuda.get(name, 0.0) + first.elapsed_time(last) / 1000
        return {
            name: {
                "cpu_wall_seconds": value,
                "cuda_interval_seconds": cuda.get(name),
                "calls": self.calls[name],
            }
            for name, value in self.wall.items()
        }


def _state_bc_optimizer_step(model, optimizer, scaler, max_grad_norm):
    """Reject a nonfinite aggregate norm before Adam can change any state.

    GradScaler detects nonfinite gradient *elements*. Their reduction can still
    overflow when every element is finite, so scaler.step is not a safe rejection
    path for a nonfinite aggregate norm.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    raw_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                   error_if_nonfinite=False))
    if not math.isfinite(raw_norm):
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
            return False, True, 0.0
        raise FloatingPointError("Nonfinite state BC gradients")
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True, False, raw_norm


def run_state_episode(
    model,
    episode,
    adapter,
    *,
    chunk_size=32,
    device="cpu",
    precision="fp32",
    optimizer=None,
    scaler=None,
    position_coef=0.5,
    tile_label_sigma=1.0,
    timing_objective="natural",
    timing_target_action_prior=0.20,
    sampled_action_prior=None,
    max_grad_norm=1.0,
    ram_limit_gib=3,
    defer_metrics=False,
    profile=None,
):
    """Train/evaluate one seat; no optimizer update occurs with a live memory."""
    if timing_objective not in ("natural", "balanced"):
        raise ValueError("Unknown timing objective")
    if timing_objective == "natural" and not (
        0 < timing_target_action_prior < 1
        and sampled_action_prior is not None
        and 0 < sampled_action_prior < 1
    ):
        raise ValueError("Natural timing needs target and actual epoch sampled priors in (0, 1)")
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    state_dtype = (
        torch.float16
        if precision == "fp16"
        else torch.bfloat16
        if precision == "bf16"
        else torch.float32
    )
    state = model.initial_state(batch_size=1, device=device, dtype=state_dtype)
    totals = {
        "loss": 0.0,
        "timing_loss": 0.0,
        "conditional_slot_loss": 0.0,
        "position_loss": 0.0,
        "card_loss": 0.0,
    }
    counts = episode.normalization
    predictions = {
        "correct": 0,
        "wait_correct": 0,
        "act_correct": 0,
        "slot_correct": 0,
        "ability_correct": 0,
        "abilities": 0,
        "tile_distance_sum": 0.0,
    }
    total_keys, prediction_keys = tuple(totals), tuple(predictions)
    metric_packets = []
    did_backward = False
    measure = profile.measure if profile is not None else lambda name: nullcontext()
    started = time.perf_counter()
    grad_context = nullcontext() if training else torch.no_grad()
    with grad_context:
        for start, rows in episode.chunks(chunk_size):
            check_ram_budget(ram_limit_gib)
            with measure("adapter_h2d"):
                inputs = prepare_state_chunk(episode, rows, adapter, device)
                selected_np = episode.supervised[start : start + len(rows)]
                selected = torch.as_tensor(selected_np, device=device)
                labels_np = episode.labels[start : start + len(rows)]
                if training:
                    placement = selected_np & (labels_np > 0) & (labels_np < 2305)
                    inputs["placement_mask"] = torch.as_tensor(placement, device=device)
                    if hasattr(model, 'learned_placement'):
                        inputs['placement_slots'] = torch.as_tensor(
                            ((labels_np - 1) // 576).clip(0, 3), device=device)
            with measure("forward_loss"), _autocast(device, precision):
                logp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
                if not torch.isfinite(logp).all():
                    raise FloatingPointError("Nonfinite state policy output")
                if selected_np.any():
                    labels = torch.as_tensor(
                        episode.labels[start : start + len(rows)], device=device
                    )
                    loss, components = hierarchical_bc_loss(
                        logp[selected],
                        labels[selected],
                        inputs["legal_action_mask"][selected],
                        position_coef=position_coef,
                        tile_label_sigma=tile_label_sigma,
                        conditional_card_loss=True,
                        normalization=counts,
                        timing_prior=timing_target_action_prior
                        if timing_objective == "natural"
                        else None,
                        sampled_action_prior=sampled_action_prior,
                    )
                else:
                    loss, components = None, None
            if loss is not None:
                with measure("finite_loss_guard"):
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite state BC loss")
                if training and loss.requires_grad:
                    with measure("backward"):
                        (scaler.scale(loss) if scaler is not None else loss).backward()
                    did_backward = True
                with measure("scalar_metrics"), torch.no_grad():
                    chunk_totals = {
                        "loss": loss.detach(),
                        **{key: components[key].detach() for key in totals.keys() - {"loss"}},
                    }
                    target, lp = labels[selected], logp[selected].float()
                    chosen = lp.argmax(-1)
                    wait, play = target == 0, (target > 0) & (target < 2305)
                    ability = target >= 2305
                    cards = torch.cat(
                        (
                            lp[:, :1],
                            torch.logsumexp(lp[:, 1:2305].reshape(-1, 4, 576), -1),
                            lp[:, 2305:],
                        ),
                        -1,
                    )
                    predict_act = torch.logsumexp(cards[:, 1:], -1) > cards[:, 0]
                    chunk_predictions = {
                        "correct": (chosen == target).sum(),
                        "wait_correct": (~predict_act & wait).sum(),
                        "act_correct": (predict_act & ~wait).sum(),
                        "ability_correct": ((chosen == target) & ability).sum(),
                        "abilities": ability.sum(),
                        "slot_correct": lp.new_zeros(()),
                        "tile_distance_sum": lp.new_zeros(()),
                    }
                    # Labels and eligibility are already on CPU. The deferred
                    # path does not synchronize just to count play rows.
                    play_count = (
                        int(((labels_np[selected_np] > 0) & (labels_np[selected_np] < 2305)).sum())
                        if defer_metrics
                        else None
                    )
                    if play_count if defer_metrics else play.any():
                        target_slot = (target[play] - 1) // 576
                        chunk_predictions["slot_correct"] = (
                            cards[play, 1:5].argmax(-1) == target_slot
                        ).sum()
                        tile = (
                            lp[play, 1:2305]
                            .reshape(-1, 4, 576)[
                                torch.arange(
                                    play_count if defer_metrics else int(play.sum()),
                                    device=target.device,
                                ),
                                target_slot,
                            ]
                            .argmax(-1)
                        )
                        target_tile = (target[play] - 1) % 576
                        distance = (
                            (tile % 18 - target_tile % 18).float().square()
                            + (tile // 18 - target_tile // 18).float().square()
                        ).sqrt()
                        chunk_predictions["tile_distance_sum"] = distance.sum()
                    if defer_metrics:
                        metric_packets.append(
                            torch.stack(
                                [
                                    *(chunk_totals[key].float() for key in total_keys),
                                    *(chunk_predictions[key].float() for key in prediction_keys),
                                ]
                            )
                        )
                    else:
                        for key in totals:
                            totals[key] += float(chunk_totals[key])
                        for key in predictions:
                            predictions[key] += float(chunk_predictions[key])
            # TBPTT: retain all forward memory, discard every previous chunk's graph.
            state = model.detach_state(state)
    del state
    if defer_metrics and metric_packets:
        with measure("final_metrics"):
            # One metrics transfer per episode. Sum the original FP32 chunk
            # scalars in Python's original order, preserving reported values.
            packets = torch.stack(metric_packets).cpu().tolist()
            for packet in packets:
                for index, key in enumerate(total_keys):
                    totals[key] += packet[index]
                for index, key in enumerate(prediction_keys, start=len(total_keys)):
                    predictions[key] += packet[index]
        del packets, metric_packets
    stepped, overflow, norm = False, False, 0.0
    with measure("optimizer"):
        if training and counts is not None and did_backward:
            stepped, overflow, norm = _state_bc_optimizer_step(model, optimizer, scaler, max_grad_norm)
            if stepped:
                model.mark_weights_updated()
    rss = check_ram_budget(ram_limit_gib)
    n, a, p, w = (counts.rows, counts.act, counts.play, counts.wait) if counts else (0, 0, 0, 0)
    return {
        **totals,
        "rows": n,
        "act_rows": a,
        "play_rows": p,
        "wait_rows": w,
        "history_frames": len(episode.rows),
        "replay_id": episode.replay_id,
        "seat": episode.seat,
        "exact_accuracy": predictions["correct"] / max(n, 1),
        "wait_accuracy": predictions["wait_correct"] / max(w, 1),
        "act_recall": predictions["act_correct"] / max(a, 1),
        "slot_accuracy": predictions["slot_correct"] / max(p, 1),
        "ability_accuracy": predictions["ability_correct"] / max(predictions["abilities"], 1),
        "tile_mean_distance": predictions["tile_distance_sum"] / max(p, 1),
        "optimizer_stepped": stepped,
        "amp_overflow_skipped": overflow,
        "grad_norm": norm,
        "rss_mib": rss / 1024**2,
        "seconds": time.perf_counter() - started,
    }


def save_state_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    *,
    dataset_fingerprint,
    split,
    epoch,
    next_episode,
    optimizer_steps,
    training_config=None,
    lr_schedule=None,
    training_control=None,
):
    """Save only at an episode boundary; temporal memory is rebuilt from tick zero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "architecture": "state_actor_critic",
        **model.checkpoint_config(),
        "model_definition": model.checkpoint_config(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "dataset_fingerprint": dataset_fingerprint,
        "split": {str(key): value for key, value in split.items()},
        "progress": {
            "epoch": epoch,
            "next_episode": next_episode,
            "optimizer_steps": optimizer_steps,
        },
        "training_config": training_config or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "resume_semantics": "episode_boundary_memory_reset_tbptt_no_stale_state",
    }
    if lr_schedule is not None:
        payload["lr_schedule"] = lr_schedule
    if training_control is not None:
        payload["training_control"] = training_control
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state_checkpoint(path, *, dataset_fingerprint, split):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (
        state.get("format") != CHECKPOINT_FORMAT
        or state.get("architecture") != "state_actor_critic"
    ):
        raise ValueError("Checkpoint is not an independent state BC architecture")
    if state["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("State BC resume dataset fingerprint differs")
    if state["split"] != {str(key): value for key, value in split.items()}:
        raise ValueError("State BC resume split differs")
    progress = state["progress"]
    if any(
        not isinstance(progress.get(key), int) or progress[key] < 0
        for key in ("epoch", "next_episode", "optimizer_steps")
    ):
        raise ValueError("Invalid state BC episode-boundary progress")
    return state


def restore_state_rng(state):
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
