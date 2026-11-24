import json
from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.training.expert_bc import ExpertBC


def dataset(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    manifest=dict(snapshot_id='fixed',metadata='dataset_metadata.json',shards=[])
    metadata=dict(decision_ticks=8,schema=dict(header_fields=['a'],entity_fields=['b']),
                  vocabulary=['card'],replays=[],training_ready=False)
    assignments={}
    for i in range(6):
        manifest['shards'].append(dict(path=f'{i}.npz',replay_offset=i))
        metadata['replays'].append(dict(canonical_game_signature=str(i)))
        assignments[str(i)]='train' if i<4 else 'validation'
    for name,value in [('dataset.manifest.json',manifest),('dataset_metadata.json',metadata),
                       ('split_registry.json',dict(assignments=assignments))]:
        (root/name).write_text(json.dumps(value))
    return root,SimpleNamespace(cfg=SimpleNamespace(header_fields=['a'],entity_fields=['b'],vocab_size=1))


def test_requires_explicit_candidate_opt_in_and_resumes_sampling_exactly(tmp_path):
    root,actor=dataset(tmp_path)
    with pytest.raises(ValueError,match='explicit'): ExpertBC(root,actor)
    bc=ExpertBC(root,actor,experimental=True)
    assert len(bc.draw(3))==3
    state=bc.state_dict()
    restored=ExpertBC(root,actor,experimental=True,state=state)
    for _ in range(5): assert restored.draw(5)==bc.draw(5)
    state['snapshot_id']='other'
    with pytest.raises(ValueError,match='changed'): ExpertBC(root,actor,experimental=True,state=state)


def test_sparky_snapshot_admits_only_matching_seats_and_keeps_game_splits(tmp_path, monkeypatch):
    import torch
    import cr_rl.training.expert_bc as module
    from scripts.stage_sparky_bc import build_index, stage_index

    root = tmp_path / 'quality'
    root.mkdir()
    manifest = dict(metadata='dataset_metadata.json', decision_ticks=8, shards=[])
    metadata = dict(schema=dict(header_fields=['a'], entity_fields=['b']),
                    vocabulary=['ZapMachine', 'Knight'], decision_ticks=8, replays=[])
    assignments = {}
    for i in range(14):
        identity = f'game-{i}'
        partition = 'train' if i < 8 else 'validation' if i < 12 else 'heldout'
        cards = [['Knight'], ['Knight']]
        if i < 13:
            cards[i % 2] = ['ZapMachine']
        path = f'{i}.npz'
        np.savez(root / path, seats=np.asarray([0, 1]))
        manifest['shards'].append(dict(path=path, replay_offset=i))
        metadata['replays'].append(dict(canonical_game_signature=identity, decks=cards))
        assignments[identity] = partition
    for name, value in [('dataset.manifest.json', manifest), ('dataset_metadata.json', metadata),
                        ('split_registry.json', dict(assignments=assignments))]:
        (root / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    index = build_index(root)
    assert index['source_games'] == 14
    assert index['games_by_partition'] == {'train': 8, 'validation': 4, 'heldout': 1}
    target = tmp_path / 'sparky'
    stage_index(root, target, index, workers=2)
    actor = SimpleNamespace(cfg=SimpleNamespace(header_fields=['a'], entity_fields=['b'], vocab_size=2))
    dataset = ExpertBC(target, actor, experimental=True)
    assert dataset.train == [(i, i % 2) for i in range(8)]
    assert dataset.validation == [(i, i % 2) for i in range(8, 12)]
    full = tmp_path / 'all_train'
    stage_index(root, full, index, workers=2, all_train=True, array_cache=target)
    dataset = ExpertBC(full, actor, experimental=True)
    assert not dataset.validation_enabled and not dataset.validation
    assert set(dataset.draw(13)) == {(i, i % 2) for i in range(13)}
    assert dataset.snapshot_id == index['snapshot_id'] + '_all_train'
    chunks = [dataset.draw(5) for _ in range(3)]
    assert [len(chunk) for chunk in chunks] == [5, 5, 3]
    assert len({seat for chunk in chunks for seat in chunk}) == 13

    model = torch.nn.Module()
    model.wait_actor = torch.nn.Linear(1, 1)
    model.mark_weights_updated = lambda: None
    before = model.wait_actor.weight.detach().clone()
    loads = []
    probe_calls = []

    def load(descriptors):
        loads.append(len(descriptors))
        return [{'actions': [0]} for _ in descriptors]

    def ref(*args, capture=False, **kwargs):
        probe_calls.append(capture)
        return {'reference_forward_kl': 0.0 if capture else 0.02}

    def train(actor, adapter, episodes, indices, weights, opt, **kwargs):
        with torch.no_grad():
            actor.wait_actor.weight.add_(1)
        return {'optimizer_steps': len(indices), 'optimized_indices': indices, 'episode_mean_weighted_nll': 1.0}

    monkeypatch.setattr(dataset, 'load', load)
    monkeypatch.setattr(module, 'reference_pass', ref)
    monkeypatch.setattr(module, 'train_actor_epoch', train)
    result = dataset.run_phase(model, None, [{}], update=1, device='cpu', sequences=5)
    assert loads == [4, 1] and probe_calls == [True, False]
    assert result['optimizer_steps'] == 5 and result['validation'] is None and not result['accepted']
    torch.testing.assert_close(model.wait_actor.weight, before)


def test_readout_hook_syncs_memory_and_only_clears_changed_adam_state(tmp_path):
    import torch
    from cr_rl.training.state_selfplay import PersistentStateSelfplay
    trainer=PersistentStateSelfplay.__new__(PersistentStateSelfplay)
    trainer.actor=torch.nn.Linear(1,1)
    parameters=dict(trainer.actor.named_parameters())
    trainer.actor_optimizer=torch.optim.AdamW(trainer.actor.parameters())
    trainer.actor_optimizer.state[parameters['weight']]={'keep':False}
    trainer.actor_optimizer.state[parameters['bias']]={'keep':True}
    trainer.args=SimpleNamespace(expert_bc_interval=2,expert_bc_sequences=1,
        expert_bc_learning_rate=1e-6,expert_bc_max_kl=.01)
    trainer.update=2;trainer.adapter=None;trainer.device='cpu';trainer.precision='fp32'
    trainer.peer_actor=None;trainer.out=tmp_path
    events=[];trainer.emit=lambda **event:events.append(event)
    sync=[];trainer._sync_behavior_model=lambda:sync.append(True)
    trainer.expert_bc=SimpleNamespace(run_phase=lambda *a,**k:dict(ran=True,
        memory_invalidated=True,accepted=True,changed_parameters=['weight']))
    assert trainer._maybe_expert_bc_phase([])['ran']
    assert parameters['weight'] not in trainer.actor_optimizer.state
    assert trainer.actor_optimizer.state[parameters['bias']]=={'keep':True}
    assert sync and trainer.actor_weights_dirty and events[0]['phase']=='expert_bc'
