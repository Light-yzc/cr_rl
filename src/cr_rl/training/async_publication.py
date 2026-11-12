"""Bounded publication of independent, immutable local file snapshots.

Only bytes already serialized by the training thread are published. No model,
optimizer, simulator or live tensor is accessed by the worker.
"""
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cr_rl.training.incremental_backup import publish_incremental
from cr_rl.training.persistence import atomic_json, copy_verified


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def freeze_publication_files(local, update):
    """Checkpoint/export use atomic replacement in PersistentSelfplay.checkpoint."""
    local = Path(local)
    snapshot = local / '.publication_snapshots' / f'update_{update:06d}'
    snapshot.mkdir(parents=True)
    try:
        for name in ('ppo_latest.pt', 'actor_export.pt'):
            if (local / name).exists():
                _link_or_copy(local / name, snapshot / name)
        receipt = local / '.publication_receipts.json'
        if receipt.exists():
            shutil.copyfile(receipt, snapshot / receipt.name)
            rollouts = [local / 'rollouts' / f'rollout_{update:06d}.{suffix}'
                        for suffix in ('npz', 'npz.json')]
        else:
            rollouts = [p for p in (local / 'rollouts').iterdir() if p.is_file()]
        for source in rollouts:
            _link_or_copy(source, snapshot / 'rollouts' / source.name)
        pool = json.loads((local / 'opponent_pool.json').read_text())
        opponents = {}
        for entry in pool['entries']:
            source = Path(entry['checkpoint'])
            source = source if source.is_absolute() else local / source
            source = source.resolve()
            if source.name in opponents and opponents[source.name] != source:
                raise ValueError('Opponent snapshot filenames must be unique')
            target = snapshot / 'opponents' / source.name
            if source.name not in opponents:
                _link_or_copy(source, target)
                opponents[source.name] = source
            entry['checkpoint'] = str(target.resolve())
        atomic_json(snapshot / 'opponent_pool.json', pool)
        for name in ('summary.json', 'metrics.jsonl', 'progress.jsonl', 'wandb_run.json',
                     'run_config.json', 'league.yaml', 'archived_opponents.jsonl'):
            if (local / name).exists():
                # These files may be overwritten/appended next round: copy, never link.
                shutil.copyfile(local / name, snapshot / name)
        return snapshot
    except BaseException:
        # The owned directory contains only files created above; no symlinks.
        shutil.rmtree(snapshot)
        raise


class BoundedAsyncPublisher:
    def __init__(self, local, destination):
        self.local, self.destination = Path(local), Path(destination)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-publisher')
        self.pending = None
        self.closed = False

    def wait(self):
        if self.pending is None:
            return None
        result = self.pending.result()  # Do not hide failed uploads or advance over them.
        self.pending = None
        return result

    def submit(self, update, after_publish=None):
        if self.closed:
            raise RuntimeError('Publisher is closed')
        wait_started = time.perf_counter()
        self.wait()  # At most one snapshot/worker; errors propagate before making another.
        backpressure_seconds = time.perf_counter() - wait_started
        started = time.perf_counter()
        snapshot = freeze_publication_files(self.local, update)
        snapshot_seconds = time.perf_counter() - started

        def publish():
            began = time.perf_counter()
            manifest = publish_incremental(snapshot, self.destination, update)
            if (snapshot / 'archived_opponents.jsonl').exists():
                copy_verified(snapshot / 'archived_opponents.jsonl',
                              self.destination / 'archived_opponents.jsonl')
            receipt = json.loads((snapshot / '.publication_receipts.json').read_text())
            atomic_json(self.local / '.publication_receipts.json', receipt)
            seconds = time.perf_counter() - began
            if after_publish is not None:
                after_publish(manifest, seconds, snapshot_seconds, backpressure_seconds)
            shutil.rmtree(snapshot)
            return manifest

        self.pending = self.executor.submit(publish)
        return self.pending

    def check(self):
        if self.pending is not None and self.pending.done():
            self.pending.result()

    def close(self):
        try:
            self.wait()
        finally:
            self.closed = True
            self.executor.shutdown(wait=True)
