"""Notebook launch identity and repeat-cell behavior without real GPU processes."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/notebook_state_bc_launch.py'
    spec = importlib.util.spec_from_file_location('notebook_launch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    return module


def marker(module, root, **fields):
    path = root / '.notebook/trainer_process.json'
    module.atomic_json(path, fields)
    return path


def test_poll_preserves_failure_and_never_calls_missing_worker_success(launch, tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'process_identity', lambda pid: None)
    path = marker(launch, tmp_path, state='running', worker_pid=18, worker_start='1')
    assert launch.TrainingJob(path).poll() == 125
    launch.atomic_json(path, {'state': 'exited', 'returncode': 2})
    assert launch.TrainingJob(path).poll() == 2


def test_orphaned_trainer_is_still_live_and_pid_reuse_is_rejected(launch, tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'process_identity', lambda pid: 'live' if pid == 22 else None)
    path = marker(launch, tmp_path, worker_pid=18, worker_start='old', trainer_pid=22, trainer_start='live')
    assert launch.TrainingJob(path).poll() is None
    launch.atomic_json(path, {'state': 'running', 'trainer_pid': 22, 'trainer_start': 'reused'})
    assert launch.TrainingJob(path).poll() == 125


def test_live_repeat_cell_attaches_without_starting_second_trainer(launch, tmp_path, monkeypatch):
    import hashlib
    argv = ['python', 'train.py', '--epochs', '3']
    digest = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
    marker(launch, tmp_path, argv_sha256=digest, state='running', worker_pid=18, worker_start='live', trainer_pid=22)
    monkeypatch.setattr(launch, 'process_identity', lambda pid: 'live' if pid == 18 else None)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('duplicate trainer'))
    job = launch.launch_or_attach(argv, source_root=tmp_path, workspace=tmp_path, log_path=tmp_path/'train.log')
    assert job.pid == 22 and job.poll() is None
    with pytest.raises(RuntimeError, match='different arguments'):
        launch.launch_or_attach([*argv, '--bad-option'], source_root=tmp_path, workspace=tmp_path, log_path=tmp_path/'train.log')


def test_launch_spec_has_exact_argv_but_no_environment_credentials(launch, tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY', 'test-key-not-a-real-secret')
    monkeypatch.setattr(launch, 'process_identity', lambda pid: 'live' if pid == 27 else None)
    calls = []
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        spec = json.loads(Path(command[-1]).read_text())
        launch.atomic_json(Path(spec['state_path']), {'worker_pid':27, 'worker_start':'live', 'state':'starting'})
        return SimpleNamespace(pid=27, poll=lambda: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', fake_popen)
    argv = ['python', 'train.py', '--resume', str(tmp_path/'latest.pt'), '--device', 'cuda']
    launch.launch_or_attach(argv, source_root=tmp_path, workspace=tmp_path,
                            log_path=tmp_path/'train.log', env_overrides={'PYTHONPATH':'bundle/src'})
    content = (tmp_path/'.notebook/launch_spec.json').read_text()
    assert json.loads(content)['argv'] == argv
    assert 'test-key-not-a-real-secret' not in content
    assert len(calls) == 1
