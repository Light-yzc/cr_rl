"""Notebook progress and W&B mirroring for the existing state-BC JSONL log.

This module never imports Torch, starts a trainer, or creates a W&B run. The
notebook owns those operations. Log delivery is at least once: the durable
cursor advances only after ``run.log`` accepts the event. A kernel crash in the
small interval between delivery and cursor save can replay that final event.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resume_completed(progress, seats_per_epoch):
    """Checkpoint epochs are zero based; next_episode counts completed seats."""
    seats = _integer(seats_per_epoch, "seats_per_epoch", 1)
    epoch = _integer(progress.get("epoch"), "epoch")
    next_episode = _integer(progress.get("next_episode"), "next_episode")
    if next_episode > seats:
        raise ValueError("Checkpoint next_episode exceeds this dataset's epoch")
    return epoch * seats + next_episode


def completed_seats(record, seats_per_epoch):
    """Train rows use one-based epoch and a zero-based completed episode index."""
    if record.get("phase") != "state_bc_train":
        return None  # Initial/mid-epoch validation is not an epoch completion.
    seats = _integer(seats_per_epoch, "seats_per_epoch", 1)
    epoch = _integer(record.get("epoch"), "epoch", 1)
    index = _integer(record.get("episode_index"), "episode_index")
    if index >= seats:
        raise ValueError("Logged episode index exceeds this dataset's epoch")
    return (epoch - 1) * seats + index + 1


def metric_payload(record):
    """Map finite scalar metrics; do not compare W&B's own step to model steps."""
    phase = record.get("phase")
    prefix = {"state_bc_train": "train", "state_bc_validation": "validation"}.get(phase)
    if prefix is None:
        if phase != "state_bc_start":
            return {}
        result = {}
        for key in ("parameter_count", "trainable_parameter_count", "ram_limit_gib"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                result[f"run/{key}"] = value
        return result
    step = _integer(record.get("optimizer_steps"), "optimizer_steps")
    result = {"trainer/optimizer_steps": step}
    for key, value in record.items():
        if key in {"optimizer_steps", "elapsed_seconds"}:
            continue
        if isinstance(value, bool):
            result[f"{prefix}/{key}"] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[f"{prefix}/{key}"] = value
    if phase=='state_bc_train' and record.get('aggregation')=='optimizer_update_window_v1':
        for key,value in list(result.items()):
            if key.startswith('train/'):
                result['train_window/'+key[len('train/'):]]=value
    return result


def configure_wandb(run):
    """A custom x axis preserves train and validation events at the same step."""
    run.define_metric("trainer/optimizer_steps")
    run.define_metric("train/*", step_metric="trainer/optimizer_steps", step_sync=True)
    run.define_metric("validation/*", step_metric="trainer/optimizer_steps", step_sync=True)
    run.define_metric('train_window/*',step_metric='trainer/optimizer_steps',step_sync=True)


class JSONLTailer:
    """Read complete appended lines and persist explicitly acknowledged offsets.

    Cursor identity should be the W&B run ID. Prefix and near-offset anchors
    allow identical Drive-restored copies while detecting truncation or log
    replacement. If the log rolls back, replay is explicit rather than silently
    skipping old bytes. Cursor files are small JSON, never model/data tensors.
    """

    def __init__(self, path, cursor_path=None, *, identity="default", skip_non_json=False):
        self.path = Path(path)
        self.skip_non_json = bool(skip_non_json)
        self.cursor_path = Path(cursor_path) if cursor_path is not None else None
        self.identity = str(identity)
        self.offset = 0
        self.prefix_length = 0
        self.prefix_sha256 = None
        self.anchor_start = 0
        self.anchor_sha256 = None
        self.resets = 0
        if self.cursor_path is not None and self.cursor_path.exists():
            saved = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            if saved.get("identity") == self.identity:
                self.offset = _integer(saved.get("offset"), "cursor offset")
                self.prefix_length = _integer(saved.get("prefix_length", 0), "prefix_length")
                self.prefix_sha256 = saved.get("prefix_sha256")
                self.anchor_start = _integer(saved.get("anchor_start", 0), "anchor_start")
                self.anchor_sha256 = saved.get("anchor_sha256")

    @staticmethod
    def _digest(stream, start, length):
        stream.seek(start)
        value = stream.read(length)
        return hashlib.sha256(value).hexdigest()

    def _reset(self):
        self.offset = 0
        self.prefix_length = 0
        self.prefix_sha256 = None
        self.anchor_start = 0
        self.anchor_sha256 = None
        self.resets += 1
        warnings.warn("Training log was truncated/replaced; replaying its available events", RuntimeWarning)

    def pending(self):
        """Yield (record, byte_offset); acknowledge each only after delivery."""
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            changed = size < self.offset
            if self.prefix_length:
                changed |= size < self.prefix_length or self._digest(stream, 0, self.prefix_length) != self.prefix_sha256
            if self.offset and self.anchor_sha256:
                changed |= self._digest(stream, self.anchor_start, self.offset - self.anchor_start) != self.anchor_sha256
            if changed:
                self._reset()
            if not self.prefix_length and size:
                self.prefix_length = min(size, 1024)
                self.prefix_sha256 = self._digest(stream, 0, self.prefix_length)
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline(8 * 1024**2 + 1)
                if len(line) > 8 * 1024**2:
                    raise ValueError(f"Training JSONL line exceeds 8MiB at byte {start}")
                if not line or not line.endswith(b"\n"):
                    return  # A concurrent writer can finish this row later.
                end = stream.tell()
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as error:
                    if self.skip_non_json:
                        self.acknowledge(end)
                        continue
                    raise ValueError(f"Invalid complete training JSONL line at byte {start}") from error
                if not isinstance(record, dict):
                    if self.skip_non_json:
                        self.acknowledge(end)
                        continue
                    raise ValueError(f"Training JSONL record at byte {start} is not an object")
                yield record, end

    def acknowledge(self, offset):
        offset = _integer(offset, "acknowledged offset")
        if offset < self.offset:
            raise ValueError("Cannot acknowledge a previous log position")
        with self.path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size < offset:
                raise OSError("Training log truncated before acknowledgement")
            self.anchor_start = max(0, offset - 128)
            self.anchor_sha256 = self._digest(stream, self.anchor_start, offset - self.anchor_start)
        self.offset = offset
        if self.cursor_path is not None:
            self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cursor_path.with_name(self.cursor_path.name + ".tmp")
            temporary.write_text(json.dumps({
                "identity": self.identity, "offset": self.offset,
                "prefix_length": self.prefix_length, "prefix_sha256": self.prefix_sha256,
                "anchor_start": self.anchor_start, "anchor_sha256": self.anchor_sha256,
            }), encoding="utf-8")
            temporary.replace(self.cursor_path)


def log_tail(path, lines=60, max_bytes=65536):
    path = Path(path)
    if not path.exists():
        return "(trainer log does not exist)"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - max_bytes))
        return "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-lines:])


def monitor_process(process, metrics_path, train_log_path, *, run=None, target_epochs=3,
                    seats_per_epoch=350, initial_completed=0, cursor_path=None,
                    poll_seconds=2, bar=None, on_poll=None):
    """Mirror logs and monitor an existing subprocess without altering training.

    KeyboardInterrupt closes the display and propagates. The notebook must
    explicitly decide whether its owned trainer should stop or continue.
    Temporary W&B delivery failures leave their cursor pending and are retried
    while training continues. On successful trainer exit, unresolved delivery
    returns ``logging_status='pending'`` and prints an explicit warning.
    ``run.finish`` is also the notebook's responsibility.
    """
    total = _integer(target_epochs, "target_epochs", 1) * _integer(seats_per_epoch, "seats_per_epoch", 1)
    initial = _integer(initial_completed, "initial_completed")
    if initial > total:
        raise ValueError("Checkpoint progress exceeds requested target epochs")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if bar is None:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, initial=initial, desc="BC seats", unit="seat")
    identity = getattr(run, "id", "no-wandb")
    tailer = JSONLTailer(metrics_path, cursor_path, identity=identity)
    display_tailer = JSONLTailer(metrics_path, identity="notebook-display")
    latest = {}
    validations = []
    completed = initial
    logging_error = None
    monitor_sync_errors = 0

    def sync_monitor_metadata():
        nonlocal monitor_sync_errors
        if on_poll is not None:
            try:
                on_poll()
            except Exception as error:
                monitor_sync_errors += 1
                print(f"监控元数据同步失败（{type(error).__name__}）；训练继续，本次同步未成功。")

    class DeliveryFailure(Exception):
        def __init__(self, original):
            self.original = original

    def consume_display():
        nonlocal completed
        postfix = None
        for record, offset in display_tailer.pending():
            phase = record.get("phase")
            if phase in ("state_bc_train", "state_bc_validation"):
                latest[phase] = record
                if phase == "state_bc_validation":
                    validations.append(record)
                count = completed_seats(record, seats_per_epoch)
                if count is not None:
                    if count > total:
                        raise ValueError("Training log contains epochs beyond the notebook target")
                    completed = max(completed, count)
                    if completed > bar.n:
                        bar.update(completed - bar.n)
                postfix = {"phase": "validation" if phase.endswith("validation") else "training",
                           "epoch": record.get("epoch"), "step": record.get("optimizer_steps")}
                if isinstance(record.get("loss"), (float, int)):
                    postfix["loss"] = f"{record['loss']:.4f}"
                if "amp_scale" in record:
                    postfix["scale"] = record["amp_scale"]
            display_tailer.acknowledge(offset)
        if postfix is not None:
            bar.set_postfix(postfix)

    def consume():
        for record, offset in tailer.pending():
            payload = metric_payload(record)
            if run is not None and payload:
                try:
                    run.log(payload)  # Do not supply step=optimizer_steps.
                except Exception as error:
                    raise DeliveryFailure(error) from error
            tailer.acknowledge(offset)

    try:
        while True:
            consume_display()  # Independent of W&B connectivity/cursor state.
            try:
                consume()
                logging_error = None
            except DeliveryFailure as error:
                if logging_error is None:
                    print("W&B日志暂未送达，训练继续，待发送记录将自动重试；请勿重启trainer。")
                logging_error = type(error.original).__name__
            sync_monitor_metadata()
            code = process.poll()
            if code is not None:
                consume_display()
                try:
                    consume()  # Drain final records after the child has flushed/exited.
                    logging_error = None
                except DeliveryFailure as error:
                    logging_error = type(error.original).__name__
                sync_monitor_metadata()
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("监控已暂停；trainer未被本监控停止。重新运行监控cell可继续查看同一进程。")
        raise
    finally:
        bar.close()
    if code:
        failure_tail = log_tail(train_log_path)
        print(f"trainer returncode: {code}\n--- training.log tail ---\n{failure_tail}")
        raise RuntimeError(f"BC trainer exited with code {code}.\n\n{failure_tail}")
    if logging_error:
        print("trainer已完成，但W&B仍有待发送日志；JSONL及cursor已保留，可重新运行监控补传。")
    return {"returncode": code, "completed_seats": completed, "target_seats": total,
            "latest": latest, "validation_events": len(validations), "log_resets": tailer.resets,
            "logging_status": "pending" if logging_error else "accepted",
            "logging_error_type": logging_error,
            "monitor_sync_errors": monitor_sync_errors}
