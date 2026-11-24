import json
import threading

import pytest
import torch

from cr_rl.training import async_checkpoint as module


def payload(weight):
    state = {'weight': weight}
    return dict(model=state, actor=state, actor_optimizer={'state': {0: {'step': torch.tensor(2.)}}},
        config={}, temporal_config={}, learned_placement_definition=None,
        model_architecture_version=3, decision_ticks=8, updates=7,
        backend_commit='test', data_version='test', content_sha='test', level_profile='test')


def test_async_save_cannot_observe_subsequent_optimizer_mutation(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = module.write_checkpoint_files

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(module, 'write_checkpoint_files', delayed)
    live = payload(torch.tensor([1., 2.]))
    frozen = module.freeze_cpu_state(live)
    assert frozen['model'] is frozen['actor']
    writer = module.AsyncCheckpointWriter()
    try:
        writer.submit(tmp_path, frozen, {'entries': []}, 7)
        assert entered.wait(5)
        assert not writer.ready()
        live['model']['weight'].add_(10)
        live['actor_optimizer']['state'][0]['step'].add_(1)
        with pytest.raises(RuntimeError, match='Drain'):
            writer.submit(tmp_path, frozen, {'entries': []}, 8)
        release.set()
        result = writer.wait()
        assert result['update'] == 7
        saved = torch.load(tmp_path/'ppo_latest.pt', weights_only=False)
        assert saved['model']['weight'].tolist() == [1., 2.]
        assert saved['actor_optimizer']['state'][0]['step'].item() == 2
        actor = torch.load(tmp_path/'actor_export.pt', weights_only=False)
        assert actor['model']['weight'].tolist() == [1., 2.]
        assert json.loads((tmp_path/'checkpoints/update_000007.pool.json').read_text()) == {'entries': []}
    finally:
        release.set()
        writer.close()


def test_async_disk_error_preserves_old_latest_and_propagates(tmp_path, monkeypatch):
    (tmp_path/'ppo_latest.pt').write_bytes(b'previous-checkpoint')

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(module, 'streaming_torch_save', fail)
    writer = module.AsyncCheckpointWriter()
    writer.submit(tmp_path, payload(torch.tensor([1.])), {'entries': []}, 7)
    try:
        with pytest.raises(OSError, match='disk full'):
            writer.wait()
        assert (tmp_path/'ppo_latest.pt').read_bytes() == b'previous-checkpoint'
    finally:
        with pytest.raises(OSError, match='disk full'):
            writer.close()
