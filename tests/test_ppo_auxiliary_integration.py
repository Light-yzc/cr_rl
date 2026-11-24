"""CPU regression checks for auxiliary update boundaries and explicit smoke coverage."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.training import state_selfplay as module
from cr_rl.training.awbc_playbuffer import PlayBuffer


def script(name):
    spec = importlib.util.spec_from_file_location(
        'aux_test_' + name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def league_args(**overrides):
    values = dict(resume='checkpoint.pt', v3=True, role_league=False, pool_capacity=8,
                  history_snapshot_interval=20_000, snapshot_interval=250_000,
                  current_opponent_fraction=None, historical_opponent_fraction=None,
                  anchor_opponent_fraction=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_league_resume_preserves_saved_configuration_and_explicit_cli_wins():
    restore = script('train_selfplay').restore_league_options
    saved = dict(role_mixture=dict(current=.5, historical=.25, anchor=.25),
                 league_options=dict(role_league=True, pool_capacity=4, history_snapshot_interval=777,
                                     snapshot_interval=888),
                 training_args=dict(pool_capacity=8, history_snapshot_interval=111))
    args = league_args()
    restore(args, saved, ['--resume', 'checkpoint.pt'])
    assert args.role_league is True
    assert (args.current_opponent_fraction, args.historical_opponent_fraction,
            args.anchor_opponent_fraction) == (.5, .25, .25)
    assert (args.pool_capacity, args.history_snapshot_interval, args.snapshot_interval) == (4, 777, 888)
    args = league_args(pool_capacity=6, current_opponent_fraction=.6,
                       history_snapshot_interval=222)
    restore(args, saved, ['--pool-capacity=6', '--current-opponent-fraction', '.6',
                         '--history-snapshot-interval', '222'])
    assert (args.pool_capacity, args.current_opponent_fraction,
            args.history_snapshot_interval) == (6, .6, 222)
    assert args.anchor_opponent_fraction == .25
    legacy = league_args()
    restore(legacy, dict(training_args=dict(pool_capacity=4,
        current_opponent_fraction=.5, historical_opponent_fraction=.25,
        anchor_opponent_fraction=.25, history_snapshot_interval=123)), [])
    assert legacy.pool_capacity == 4 and legacy.history_snapshot_interval == 123
    fresh = league_args(resume=None)
    before = vars(fresh).copy()
    restore(fresh, saved, [])
    assert vars(fresh) == before


def segment(game, *, choice=True, role='current', deck='a', prefix=0):
    legal = np.zeros((3, 3), dtype=bool)
    legal[:, 0] = True
    legal[:, 1] = choice
    return dict(episode_id=game, opponent_role=role, own_deck=deck, opp_deck='other',
        raw_obs=np.arange(6, dtype=np.int32).reshape(3, 2), legal_masks=legal,
        actions=np.array([0, 1 if choice else 0, 0]), valid=np.ones(3, dtype=bool),
        prefix_raw=np.full((prefix, 2), 9, dtype=np.int32))


def test_probe_selection_scans_past_forced_wait_prefix_segments():
    segments = [segment(0, choice=False), segment(1, choice=False), segment(2, prefix=2)]
    probes = module.select_awbc_probes(segments)
    assert len(probes) == 1
    probe = probes[0]
    assert probe['probe_info']['episode_id'] == 2
    np.testing.assert_array_equal(probe['actor_keep'], [False, False, True, True, True])
    np.testing.assert_array_equal(probe['raw_obs'][:2], segments[2]['prefix_raw'])
    assert probe['legal_masks'].shape == (5, 3)


def test_probe_selection_requires_recurrent_history_for_continuing_game():
    tail = segment(0)
    tail['has_complete_prefix'] = False
    assert module.select_awbc_probes([tail]) == []
    tail['prefix_raw'] = np.zeros((4, 2), dtype=np.int32)
    assert module.select_awbc_probes([tail])[0]['probe_info']['prefix_steps'] == 4


def test_awbc_smoke_leaves_followup_boundary_for_bc():
    trainer = module.PersistentStateSelfplay.__new__(module.PersistentStateSelfplay)
    trainer.args = SimpleNamespace(require_awbc_smoke=True)
    trainer.awbc_buffer = object()
    trainer._awbc_smoke_report = {'actor_attempt_update': 930}
    assert trainer._maybe_awbc_phase() == {
        'ran': False, 'reason': 'smoke_followup_ppo_and_bc'}


def test_probe_selection_covers_distinct_games_roles_and_decks_with_a_bound():
    segments = [segment(1), segment(1), segment(2, role='anchor', deck='b'),
                segment(3, role='historical', deck='c'), segment(4, deck='d'),
                segment(5, deck='e')]
    probes = module.select_awbc_probes(segments)
    assert len(probes) == 4
    assert len({p['probe_info']['episode_id'] for p in probes}) == 4
    assert {p['probe_info']['opponent_role'] for p in probes} == {'current', 'anchor', 'historical'}
    assert len({p['probe_info']['own_deck'] for p in probes}) == 4
    assert module.select_awbc_probes([segment(0, choice=False)]) == []


def test_awbc_smoke_command_forces_hook_and_post_auxiliary_ppo_without_mutating_plan():
    configure = script('launch_prepared_pipeline').configure_awbc_smoke
    original = ['python', 'train.py', '--updates', '1', '--awbc-interval=50',
                '--awbc-sessions', 'old']
    before = original.copy()
    command = configure(original, Path('new_sessions'))
    assert original == before
    assert command[command.index('--updates') + 1] == '2'
    assert command[command.index('--awbc-interval') + 1] == '1'
    assert command[command.index('--awbc-sessions') + 1] == 'new_sessions'
    assert '--awbc-interval=50' not in command
    assert '--require-awbc-smoke' in command
    assert configure(command, Path('new_sessions')) == command


def test_launcher_plan_only_emits_required_awbc_smoke(tmp_path, monkeypatch, capsys):
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    (prepared/'single_checkpoint_preparation.json').write_text(json.dumps(dict(
        smoke_argv=['python', 'scripts/train_selfplay.py', '--updates', '1'])))
    monkeypatch.setattr(sys, 'argv', ['launch_prepared_pipeline.py', '--root', str(tmp_path),
        '--stage', 'ppo-smoke', '--sessions', 'real_sessions', '--awbc-playbuffer', '--plan-only'])
    script('launch_prepared_pipeline').main()
    plan = json.loads(capsys.readouterr().out)
    command = plan['argv']
    assert plan['plan_only'] and plan['awbc_smoke_required']
    assert command[command.index('--updates') + 1] == '2'
    assert command[command.index('--awbc-interval') + 1] == '1'
    assert '--require-awbc-smoke' in command


@pytest.mark.parametrize('attempted,post_steps,expected', [
    (False, 1, 'did not exercise'), (True, 0, 'non-skipped PPO'), (True, 1, None)])
def test_required_smoke_never_reports_a_skipped_hook_as_passing(tmp_path, attempted, post_steps, expected):
    trainer = module.PersistentStateSelfplay.__new__(module.PersistentStateSelfplay)
    trainer.args = SimpleNamespace(updates=2, require_awbc_smoke=True)
    trainer.out = tmp_path
    trainer.update = 929
    calls = []

    def update():
        trainer.update += 1
        calls.append(trainer.update)
        return dict(update=trainer.update, optimizer_steps=post_steps,
            initial_logp_max_error=0., awbc_playbuffer=dict(actor_attempted=attempted,
                accepted=False, reason='untrusted_value_cohort'))

    trainer.run_update = update
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            trainer.run()
    else:
        trainer.run()
    report = json.loads((tmp_path/'awbc_smoke_report.json').read_text())
    assert report['passed'] == (expected is None)
    assert calls == ([930] if not attempted else [930, 931])
    if expected is None:
        assert report['accepted'] is False  # A genuine KL-rejected attempt also exercises rollback.
        assert report['post_aux_ppo_update'] == 931


def hook_trainer(tmp_path, monkeypatch):
    from cr_rl.client.rl_recording import sha256_file
    trainer = module.PersistentStateSelfplay.__new__(module.PersistentStateSelfplay)
    trainer.actor = torch.nn.Linear(1, 1)
    trainer.critic = None
    trainer.adapter = None
    trainer.actor_optimizer = torch.optim.AdamW(trainer.actor.parameters())
    trainer.out = tmp_path
    trainer.update = 50
    trainer.gamma = .9995
    trainer.device = trainer.critic_device = torch.device('cpu')
    trainer.precision = 'fp32'
    trainer.peer_actor = None
    trainer.actor_weights_dirty = False
    trainer.args = SimpleNamespace(awbc_interval=1, awbc_min_new_games=12, awbc_batch_games=24,
        awbc_epochs=2, awbc_value_epochs=1, awbc_learning_rate=1e-6, awbc_max_kl=.01, seed=0)
    trainer.emit = lambda **event: None
    trainer._sync_behavior_model = lambda: None
    buffer = PlayBuffer(tmp_path)
    trainer.awbc_buffer = buffer
    for i in range(6):
        run = tmp_path / str(i)
        run.mkdir()
        (run/'manifest.json').write_text('{}')
        (run/'result.json').write_text('{}')
        buffer.state['entries'][str(i)] = dict(run=str(run), active=True, uses=0,
            signature=sha256_file(run/'manifest.json') + sha256_file(run/'result.json'),
            partition='train' if i < 4 else 'validation')
    monkeypatch.setattr(buffer, 'refresh', lambda: dict(added_games=0, known_games=len(buffer.state['entries'])))
    monkeypatch.setattr(buffer, 'select', lambda *args: list(buffer.state['entries']))
    monkeypatch.setattr(module, 'read_game', lambda path: {'id': int(path.name)})
    return trainer


@pytest.mark.parametrize('memory_invalidated', [False, True])
def test_gate_skip_does_not_consume_or_cycle_old_batches_and_new_source_retries(tmp_path, monkeypatch, memory_invalidated):
    trainer = hook_trainer(tmp_path, monkeypatch)
    calls = []

    def skip(*args, **kwargs):
        calls.append(kwargs)
        return dict(actor_attempted=False, accepted=False, reason='untrusted_value_cohort',
                    actor_train_indices=[], actor_holdout_indices=[], baseline=None,
                    memory_invalidated=memory_invalidated)

    monkeypatch.setattr(module, 'run_phase', skip)
    result = trainer._maybe_awbc_phase([segment(1)])
    assert not result['ran'] and result['buffer_updated']
    assert all(row['uses'] == 0 for row in trainer.awbc_buffer.state['entries'].values())
    assert trainer.awbc_buffer.state['last_update'] == 0
    assert trainer.actor_weights_dirty is memory_invalidated
    assert calls[0]['fresh_indices'] == [0, 1, 2, 3]
    assert calls[0]['min_new_games'] == 12
    monkeypatch.setattr(trainer.awbc_buffer, 'select',
                        lambda *args: list(trainer.awbc_buffer.state['entries'])[::-1])
    trainer.update += 1
    assert trainer._maybe_awbc_phase([segment(2)])['reason'] == 'unchanged_skipped_source'
    assert len(calls) == 1
    from cr_rl.client.rl_recording import sha256_file
    run = tmp_path/'6'
    run.mkdir()
    (run/'manifest.json').write_text('{"new":true}')
    (run/'result.json').write_text('{}')
    trainer.awbc_buffer.state['entries']['6'] = dict(run=str(run), active=True, uses=0,
        signature=sha256_file(run/'manifest.json')+sha256_file(run/'result.json'), partition='validation')
    trainer.update += 1
    assert trainer._maybe_awbc_phase([segment(3)])['buffer_updated']
    assert len(calls) == 2


def test_rejected_real_attempt_consumes_only_optimized_games_and_used_holdout(tmp_path, monkeypatch):
    trainer = hook_trainer(tmp_path, monkeypatch)
    result = dict(actor_attempted=True, accepted=False, actor_train_indices=[1, 3],
        actor_holdout_indices=[4, 5], changed_parameters=[], best_epoch=0,
        epochs=[dict(reference_forward_kl=.02)], baseline=dict(reference_forward_kl=0.),
        value_report=dict(value_gate_passed=True, weighted_train_games=2, unit_weight_train_games=0))
    monkeypatch.setattr(module, 'run_phase', lambda *args, **kwargs: result)
    actual = trainer._maybe_awbc_phase([segment(1)])
    assert actual['ran'] and actual['actor_attempted'] and not actual['accepted']
    entries = trainer.awbc_buffer.state['entries']
    assert [entries[str(i)]['uses'] for i in range(6)] == [0, 1, 0, 1, 1, 1]
    assert trainer.awbc_buffer.state['last_update'] == 50
    assert trainer.actor_weights_dirty
