"""One immutable CPU checkpoint in flight; never serialize live optimizer tensors."""
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from cr_rl.training.host_memory import streaming_torch_save
from cr_rl.training.persistence import atomic_json


def freeze_cpu_state(value, memo=None):
    """Finish device copies before returning; CPU state also needs independent storage."""
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device='cpu', copy=True)
        memo[identity] = result
        return result
    if isinstance(value, dict):
        result = type(value)()
        memo[identity] = result
        result.update((key, freeze_cpu_state(item, memo)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, (tuple, list)):
        result = [freeze_cpu_state(item, memo) for item in value]
        result = tuple(result) if isinstance(value, tuple) else result
        memo[identity] = result
        return result
    return copy.deepcopy(value, memo)


def write_checkpoint_files(out, payload, pool, update):
    """Called with isolated state and pool metadata; all latest files replace atomically."""
    out = Path(out)
    started = time.perf_counter()
    temporary = out / 'ppo_latest.tmp.pt'
    streaming_torch_save(payload, temporary, fsync=True)
    restore = out / 'checkpoints' / f'update_{update:06d}.pt'
    restore.parent.mkdir(parents=True, exist_ok=True)
    restore_tmp = restore.with_suffix('.pt.tmp')
    restore_tmp.unlink(missing_ok=True)
    os.link(temporary, restore_tmp)
    restore_tmp.replace(restore)
    for entry in pool['entries']:
        source = Path(entry['checkpoint'])
        if not source.is_absolute():
            entry['checkpoint'] = str((out / source).resolve())
    atomic_json(restore.with_suffix('.pool.json'), pool)
    export = out / 'actor_export.tmp.pt'
    streaming_torch_save({key: payload[key] for key in (
        'model', 'config', 'temporal_config', 'learned_placement_definition',
        'model_architecture_version', 'decision_ticks', 'updates', 'backend_commit',
        'data_version', 'content_sha', 'level_profile')}, export, fsync=False)
    # Only announce a complete local pair. Publication waits for this job's result.
    export.replace(out / 'actor_export.pt')
    temporary.replace(out / 'ppo_latest.pt')
    return dict(update=update, local_write_seconds=time.perf_counter() - started)


class AsyncCheckpointWriter:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-checkpoint')
        self.pending = None

    def submit(self, out, payload, pool, update):
        if self.pending is not None:
            raise RuntimeError('Drain the previous checkpoint before staging another')
        self.pending = self.executor.submit(write_checkpoint_files, out, payload, pool, update)

    def ready(self):
        return self.pending is not None and self.pending.done()

    def wait(self):
        if self.pending is None:
            return None
        result = self.pending.result()  # Propagate disk errors to the training thread.
        self.pending = None
        return result

    def close(self):
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)
