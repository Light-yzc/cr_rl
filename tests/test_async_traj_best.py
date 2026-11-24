import json
from types import SimpleNamespace

import torch

from cr_rl.training.async_checkpoint import AsyncCheckpointWriter, freeze_cpu_state
from cr_rl.training.state_selfplay import PersistentStateSelfplay


def test_deferred_traj_best_uses_saved_update_and_keeps_immutable_weights(tmp_path):
    trainer = PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.out = tmp_path
    trainer.update = 8
    trainer.args = SimpleNamespace(keep_checkpoints=1, keep_rollouts=0,
                                   keep_periodic_checkpoints=0, backup_dir=None)
    trainer.opponents = []
    events = []
    trainer.emit = lambda **event: events.append(event)
    pool = {'entries': []}
    (tmp_path/'opponent_pool.json').write_text(json.dumps(pool))
    state = dict(model={'weight': torch.tensor([1.])}, config={}, temporal_config={},
        learned_placement_definition=None, model_architecture_version=3, decision_ticks=8,
        updates=7, backend_commit='test', data_version='test', content_sha='test', level_profile='test')
    trainer._checkpoint_writer = AsyncCheckpointWriter()
    trainer._checkpoint_writer.submit(tmp_path, freeze_cpu_state(state), pool, 7)
    trainer._pending_checkpoint = dict(update=7, publish=False,
        training_win_rates=dict(anchor_pool_games=150, anchor_pool_ema_win_rate=.6,
                               historical_pool_mean_ema_win_rate=.5))
    try:
        trainer._flush_async_checkpoint(wait=True)
        assert json.loads((tmp_path/'best_train_traj.json').read_text())['update'] == 7
        assert torch.load(tmp_path/'ppo_best_val.pt', weights_only=False)['updates'] == 7
        assert trainer._last_checkpoint_update == 7
        assert events[-1]['phase'] == 'checkpoint_saved'
        assert trainer._pending_checkpoint is None
    finally:
        trainer._checkpoint_writer.close()
