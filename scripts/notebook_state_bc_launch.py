"""One durable BC trainer per notebook work directory; no model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def process_identity(pid):
    """Linux PID plus start ticks prevents attaching to a reused PID."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, TypeError, ValueError):
        return None
    if fields[0] in ('Z', 'X'):
        return None
    return fields[19]


def alive(record, role):
    pid, start = record.get(role + '_pid'), record.get(role + '_start')
    return bool(pid and start and process_identity(pid) == start)


def read_state(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


class TrainingJob:
    """Popen-compatible poll for both original and reattached notebook kernels."""
    def __init__(self, state_path):
        self.state_path = Path(state_path)

    @property
    def pid(self):
        record = read_state(self.state_path)
        return record.get('trainer_pid') or record.get('worker_pid')

    @property
    def returncode(self):
        return self.poll()

    def poll(self):
        record = read_state(self.state_path)
        if alive(record, 'worker') or alive(record, 'trainer'):
            return None
        # Never mistake a disappeared supervisor / reclaimed runtime for success.
        if record.get('state') == 'exited' and isinstance(record.get('returncode'), int):
            return record['returncode']
        return 125


def launch_or_attach(argv, *, source_root, workspace, log_path, env_overrides=None,
                     restart_completed=False):
    """Attach to live jobs; optionally rerun a normally exited identical command.

    A controller that exits successfully while waiting for more data can be
    restarted explicitly. The default retains completed-command idempotency,
    and even an explicit restart never duplicates a still-live process.
    """
    if sys.platform != 'linux':
        raise RuntimeError('This Colab process launcher requires Linux')
    import fcntl

    source_root, workspace = Path(source_root).resolve(), Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    control = workspace / '.notebook'
    control.mkdir(exist_ok=True)
    state_path = control / 'trainer_process.json'
    argv = [str(argument) for argument in argv]
    digest = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
    with (control / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = read_state(state_path)
        active = alive(record, 'worker') or alive(record, 'trainer')
        if active:
            if record.get('argv_sha256') != digest:
                raise RuntimeError('A trainer is already running with different arguments; attach first')
            print(f'Attached to existing trainer PID {record.get("trainer_pid")}', flush=True)
            return TrainingJob(state_path)
        if (not restart_completed and record.get('state') == 'exited'
                and record.get('returncode') == 0 and record.get('argv_sha256') == digest):
            print('This training command already finished successfully', flush=True)
            return TrainingJob(state_path)
        spec = {
            'argv': argv, 'source_root': str(source_root), 'workspace': str(workspace),
            'log_path': str(Path(log_path).resolve()), 'state_path': str(state_path),
            'argv_sha256': digest,
            'env_overrides': env_overrides or {},
        }
        spec_path = control / 'launch_spec.json'
        atomic_json(spec_path, spec)
        with (control / 'supervisor.log').open('a', buffering=1) as output:
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--worker', str(spec_path)],
                cwd='/content' if Path('/content').is_dir() else str(workspace),
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        # The worker publishes its own identity before importing / launching Torch.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            record = read_state(state_path)
            if record.get('worker_pid') == worker.pid:
                print(f'Trainer supervisor PID {worker.pid}', flush=True)
                return TrainingJob(state_path)
            if worker.poll() is not None:
                raise RuntimeError('Trainer supervisor failed; inspect ' + str(control / 'supervisor.log'))
            time.sleep(.1)
        raise RuntimeError('Trainer is still starting; rerun this cell to attach, do not start another command')


def worker_main(spec_path):
    import fcntl

    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    workspace = Path(spec['workspace'])
    state_path = Path(spec['state_path'])
    record = {
        'argv_sha256': spec['argv_sha256'], 'worker_pid': os.getpid(),
        'worker_start': process_identity(os.getpid()), 'state': 'starting',
        'started_unix': time.time(), 'returncode': None,
    }
    lock_path = workspace / '.notebook/trainer.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(state_path, record)
        child = None
        try:
            log_path = Path(spec['log_path'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = {**os.environ, **spec['env_overrides'], 'PYTHONUNBUFFERED': '1'}
            with log_path.open('a', buffering=1) as output:
                child = subprocess.Popen(spec['argv'], cwd=spec['source_root'], env=environment,
                                         stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record.update(state='running', trainer_pid=child.pid,
                              trainer_start=process_identity(child.pid))
                atomic_json(state_path, record)
                returncode = child.wait()
        except BaseException:
            # A live child remains discoverable; do not falsely record success.
            traceback.print_exc()
            returncode = 125
        record.update(state='exited', returncode=returncode, finished_unix=time.time())
        atomic_json(state_path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    worker_main(parser.parse_args().worker)
