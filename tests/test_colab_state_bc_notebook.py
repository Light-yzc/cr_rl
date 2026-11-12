"""Check the shipped notebook itself, including recoverable W&B login failures."""
import ast
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/make_colab_state_bc_notebook.py'
    spec = importlib.util.spec_from_file_location('build_notebook_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_compiles_and_embedded_tools_match_exact_shipped_sources():
    module = builder()
    notebook = module.build()
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 6
    for cell in cells:
        compile(cell, '<notebook-cell>', 'exec')
    tree = ast.parse(cells[1])
    embedded = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'EMBEDDED_MODULES' for target in node.targets))
    for name, record in embedded.items():
        raw = base64.b64decode(record['base64'])
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        assert raw == (module.ROOT / 'scripts' / name).read_bytes()


def test_online_initialization_failure_does_not_cache_bad_credentials(tmp_path, monkeypatch):
    module = builder()
    cells = [''.join(c['source']) for c in module.build()['cells'] if c['cell_type'] == 'code']
    fake = ModuleType('wandb')
    fake.run = None
    fake.Settings = lambda **kwargs: kwargs
    attempts = []
    def login(**kwargs):
        attempts.append(kwargs['key'])
        return True
    fake.login = login
    calls = []
    def initialize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError('online authentication failed')
        fake.run = SimpleNamespace(id=kwargs['id'], url='https://example.invalid/run')
        return fake.run
    fake.init = initialize
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    keys = iter(['test-wrong-key', 'test-correct-key'])
    import getpass
    monkeypatch.setattr(getpass, 'getpass', lambda message: next(keys))
    userdata = ModuleType('google.colab')
    userdata.userdata = SimpleNamespace(get=lambda name: None)
    monkeypatch.setitem(sys.modules, 'google.colab', userdata)
    def atomic_json(path, value):
        path.write_text(json.dumps(value))
    namespace = {'NOTEBOOK_RUN': tmp_path, 'WANDB_PROJECT':'test', 'WANDB_ENTITY':None,
                 'WANDB_NAME':'test', 'WORK':tmp_path, 'SOURCE_RUN':tmp_path,
                 'TARGET_EPOCHS':3, 'os':__import__('os'), 'json':json,
                 'atomic_json':atomic_json, 'configure_wandb':lambda run: None}
    with pytest.raises(RuntimeError, match='authentication failed'):
        exec(cells[2], namespace)
    assert not namespace.get('_bc_wandb_logged_in', False)
    namespace['ensure_wandb']()
    assert namespace['_bc_wandb_logged_in']
    assert attempts == ['test-wrong-key', 'test-correct-key']
