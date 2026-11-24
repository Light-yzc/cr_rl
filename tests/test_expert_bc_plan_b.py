import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.training import expert_bc_plan_b as plan


SCHEMA = SimpleNamespace(
    header={'hand_card': (0, 1, 2, 3), 'elixir': (4,), 'elixir_reserved': (5,)},
    scales={'elixir_scale': 10000},
)


def test_frame_weights_use_actual_slot_and_reserved_elixir():
    ep = dict(
        raw_obs=np.array([[0, 1, 1, 1, 58000, 10000]] * 9),
        actions=np.array([0, 0, 0, 0, 1, 577, 2305, 0, 0]),
        actor_keep=np.array([1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool),
    )
    ep['raw_obs'][1, 4] = 57999
    ep['raw_obs'][2, 4] = 70000
    ep['raw_obs'][3, 0] = 1
    ep['raw_obs'][8, 5] = -1
    np.testing.assert_allclose(plan.frame_weights(ep, SCHEMA, 0),
                               [.5, .06, .06, .06, 1.5, 1., 1., 0., .06])


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(header_fields=['h'], entity_fields=['e'], vocab_size=2)
        self.wait_actor = torch.nn.Linear(1, 1, bias=False)
        self.wait_actor.weight.data.fill_(.5)
        self.backbone = torch.nn.Linear(1, 1, bias=False)
        self.backbone.weight.data.fill_(.2)
        self.generation = 0

    def mark_weights_updated(self):
        self.generation += 1

    def forward_state(self, h, e, mask, legal, **kwargs):
        scores = h.new_zeros(*h.shape[:2], 2307)
        scores[..., 0] = self.wait_actor(h).squeeze(-1)
        scores[..., 1] = self.backbone(h).squeeze(-1)
        return scores.masked_fill(~legal, -1e9).log_softmax(-1), None, None

    def detach_state(self, state):
        return state


class ToyAdapter:
    schema = SCHEMA

    def __call__(self, raw):
        header = np.ones((*raw.shape[:2], 1), dtype=np.float32)
        return header, header, header


@pytest.fixture
def snapshot(tmp_path):
    manifest = dict(snapshot_id='mixed-test', metadata='metadata.json', shards=[])
    metadata = dict(bc_plan=plan.PLAN, validation_enabled=True, decision_ticks=8, training_ready=True,
                    schema=dict(header_fields=['h'], entity_fields=['e'], seat_ints=6),
                    vocabulary=['ZapMachine', 'Knight'], replays=[])
    assignments = {}
    for partition in ('train', 'validation'):
        for cohort in plan.COHORTS:
            for _ in range(7):
                index = len(metadata['replays'])
                identity = f'game-{index}'
                seat = index % 2
                cards = ['ZapMachine' if cohort == 'sparky' else 'Knight']
                metadata['replays'].append(dict(canonical_game_signature=identity, decks=[cards, cards]))
                assignments[identity] = partition
                path = f'{index}.npz'
                manifest['shards'].append(dict(path=path, replay_offset=index, cohort=cohort, selected_seats=[seat]))
                masks = np.zeros((2, 2307), dtype=bool)
                masks[:, [0, 1, 577]] = True
                np.savez(tmp_path/path, observations=np.array([[int(cohort == 'general'), 1, 1, 1, 60000, 0]] * 2),
                         masks=masks, actions=[1, 1], ticks=[0, 8], flags=[0, 0], seats=[seat]*2, replay_ids=[0, 0])
    for path, value in [('dataset.manifest.json', manifest), ('metadata.json', metadata),
                        ('split_registry.json', dict(assignments=assignments))]:
        (tmp_path/path).write_text(json.dumps(value))
    return tmp_path, ToyActor()


def test_actual_balanced_steps_improve_nll_use_all_holdout_and_drop_optimizer_momentum(snapshot, monkeypatch):
    root, actor = snapshot
    bc = plan.PlanBExpertBC(root, actor)
    assert bc.bc_plan == 'sparky_b'
    adapter = ToyAdapter()
    probes = bc.load(bc.validation[:2])
    wrapped = plan._train_accumulated
    groups = []
    opts = []

    def record(actor, adapter, episodes, weights, optimizer, *args):
        groups.append([ep['descriptor'] for ep in episodes])
        opts.append((optimizer, bool(optimizer.state)))
        wrapped(actor, adapter, episodes, weights, optimizer, *args)

    monkeypatch.setattr(plan, '_train_accumulated', record)
    backbone = actor.backbone.weight.detach().clone()
    for update in (5, 10):
        report = bc.run_phase(actor, adapter, probes, update=update, device='cpu')
        assert report['accepted']
        assert report['optimizer_steps'] == 2 and report['trained_sequences'] == 12
        assert report['validation_sequences'] == len(bc.validation) == 14
        assert report['validation']['unweighted_nll'] < report['baseline']['unweighted_nll']
        assert report['validation']['deploy_card_top1_acc'] == 1.
        assert report['validation']['weighted_nll'] > 0
        assert report['wait_prob_delta'] <= .005
    torch.testing.assert_close(actor.backbone.weight, backbone)
    assert actor.training and actor.backbone.weight.requires_grad
    assert opts[0][0] is opts[1][0] and opts[2][0] is opts[3][0] and opts[2][0] is not opts[0][0]
    assert [state for _, state in opts] == [False, True, False, True]
    for group in groups:
        cohorts = [bc.manifest['shards'][index]['cohort'] for index, _ in group]
        assert cohorts.count('sparky') == cohorts.count('general') == 3
        assert not set(group) & set(bc.validation)
    state = bc.state_dict()
    assert 'optimizer_state' not in state and 'scaler_state' not in state
    restored = plan.PlanBExpertBC(root, actor, state=state)
    assert restored.draw(12) == bc.draw(12)


@pytest.mark.parametrize(('gate', 'reason'), [
    ('wait', 'mean_p_wait_drift'),
    ('kl', 'ppo_forward_kl_limit'),
    ('heldout', 'heldout_unweighted_nll_not_improved'),
])
def test_rejected_candidate_returns_exact_readout_and_does_not_retry(snapshot, monkeypatch, gate, reason):
    root, actor = snapshot
    bc = plan.PlanBExpertBC(root, actor)
    before = actor.wait_actor.weight.detach().clone()
    original = plan._score
    report_calls = []

    def score(*args, **kwargs):
        result, probability = original(*args, **kwargs)
        report_calls.append(result)
        if kwargs.get('reference') is not None:
            if gate == 'wait':
                result['mean_p_wait'] = report_calls[1]['mean_p_wait'] + .006
            if gate == 'kl':
                result['reference_forward_kl'] = .006
        elif len(report_calls) == 3 and gate == 'heldout':
            result['unweighted_nll'] = report_calls[0]['unweighted_nll']
            if 'weighted_nll' in report_calls[0]:
                result['weighted_nll'] = report_calls[0]['weighted_nll']
        return result, probability

    monkeypatch.setattr(plan, '_score', score)
    probes = bc.load(bc.validation[:1])
    result = bc.run_phase(actor, ToyAdapter(), probes, update=5, device='cpu')
    assert not result['accepted'] and reason in result['rejected_reasons']
    assert result['memory_invalidated'] and result['changed_parameters'] == []
    assert torch.equal(actor.wait_actor.weight, before)
    assert bc.run_phase(actor, ToyAdapter(), probes, update=5, device='cpu')['reason'] == 'already_processed'


def test_plan_b_accepts_weighted_nll_improvement_even_when_wait_heavy_unweighted_ticks_up(snapshot, monkeypatch):
    root, actor = snapshot
    bc = plan.PlanBExpertBC(root, actor)
    original = plan._score
    calls = []

    def score(*args, **kwargs):
        res, prob = original(*args, **kwargs)
        calls.append(res)
        if len(calls) == 3:
            # Simulate 94%-WAIT holdout where deploy weighted_nll improves while unweighted_nll rises +0.002
            res['unweighted_nll'] = calls[0]['unweighted_nll'] + 0.002
            res['weighted_nll'] = calls[0]['weighted_nll'] - 0.003
        return res, prob

    monkeypatch.setattr(plan, '_score', score)
    probes = bc.load(bc.validation[:1])
    result = bc.run_phase(actor, ToyAdapter(), probes, update=5, device='cpu')
    assert result['accepted'] is True
    assert result['rejected_reasons'] == []
