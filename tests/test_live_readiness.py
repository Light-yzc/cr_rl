"""Readiness reports must follow the current importer, not obsolete log layout."""
import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


def load_auditor():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_nulls_rl_readiness.py'
    spec = importlib.util.spec_from_file_location('live_readiness_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recording(run, *, veto=False):
    (run / 'diagnostics').mkdir()
    (run / 'manifest.json').write_text(json.dumps({
        'collection_complete': True, 'outcome_verified': True, 'outcome': 'win',
        'arguments': {'selection_mode': 'policy_sample', 'execute': True},
    }), encoding='utf-8')
    (run / 'terminal.json').write_text(json.dumps({'clock_ended': True}), encoding='utf-8')
    events = [{'event': 'decision', 'decision_id': i + 1, 'tick': 8 * i}
              for i in range(2)]
    (run / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)), encoding='utf-8')
    probabilities = [.4, .6] + [0.] * 2305
    with gzip.open(run / 'diagnostics/frames_0000.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for i in range(2):
            action = i
            row = {'tick': 8 * i, 'decision_id': i + 1, 'obs': [0] * 7024,
                   'probabilities': probabilities, 'legal_actions': [0, 1], 'action': action,
                   'selected_log_probability': math.log(probabilities[action]),
                   'snapshot': {'clock_ended': False},
                   'action_selection': {'proposed_action': action, 'spell_vetoed': veto}}
            stream.write(json.dumps(row) + '\n')
    masks = np.zeros((2, 2307), dtype=bool)
    masks[:, :2] = True
    return {'actions': np.array([0, 1]), 'legal_masks': masks,
            'policy_loss_mask': np.ones(2, dtype=bool)}


def test_separate_terminal_snapshot_is_valid_but_not_automatically_on_policy(tmp_path):
    episode = recording(tmp_path)
    report = load_auditor().audit(tmp_path, episode_loader=lambda _: episode)
    assert report['counts']['terminal_clock_frames'] == 0
    assert report['episode_contract_valid']
    assert report['awbc_recording_eligible']
    assert report['awbc_usable_choice_frames'] == 2
    assert report['blockers'] == []
    assert not report['direct_ppo_ready']
    assert 'checkpoint' in report['direct_ppo_blockers'][0]


def test_importer_rejection_is_reported_without_claiming_importer_is_missing(tmp_path):
    recording(tmp_path)
    def reject(_):
        raise ValueError('Observation gap exceeds 0.8s; incomplete sequence')
    report = load_auditor().audit(tmp_path, episode_loader=reject)
    assert not report['episode_contract_valid']
    assert not report['awbc_recording_eligible']
    assert report['episode_validation_status'] == 'rejected'
    assert '0.8s' in report['episode_validation_error']
    assert not any('importer yet' in reason for reason in report['blockers'])


def test_stale_policy_mask_is_reflected_in_usable_labels(tmp_path):
    episode = recording(tmp_path)
    episode['policy_loss_mask'][1] = False
    report = load_auditor().audit(tmp_path, episode_loader=lambda _: episode)
    assert report['episode_contract_valid']
    assert report['excluded_policy_frames'] == 1
    assert report['awbc_usable_choice_frames'] == 1


@pytest.mark.parametrize('veto,forced', [(True, False), (False, True)])
def test_context_only_episodes_are_not_awbc_training_data(tmp_path, veto, forced):
    episode = recording(tmp_path, veto=veto)
    if forced:
        episode['legal_masks'][:, 1:] = False
    report = load_auditor().audit(tmp_path, episode_loader=lambda _: episode)
    assert report['episode_contract_valid']
    assert not report['awbc_recording_eligible']
    assert report['awbc_usable_choice_frames'] == 0


def test_unavailable_validator_is_distinguished_from_rejected_data(tmp_path):
    recording(tmp_path)
    def unavailable(_):
        raise ModuleNotFoundError('torch')
    report = load_auditor().audit(tmp_path, episode_loader=unavailable)
    assert report['episode_validation_status'] == 'unavailable'
    assert not report['episode_contract_valid']
    assert not report['awbc_recording_eligible']


@pytest.mark.parametrize('damage', ['bad_manifest', 'nonobject_manifest', 'missing_events',
                                  'nonobject_frame', 'nonobject_event', 'invalid_decision_id'])
def test_damaged_session_is_reported_without_aborting_other_sessions(tmp_path, damage):
    bad = tmp_path / 'bad'
    good = tmp_path / 'good'
    bad.mkdir()
    good.mkdir()
    episode = recording(bad)
    good_episode = recording(good)
    if damage == 'bad_manifest':
        (bad / 'manifest.json').write_text('{', encoding='utf-8')
    elif damage == 'nonobject_manifest':
        (bad / 'manifest.json').write_text('null', encoding='utf-8')
    elif damage == 'missing_events':
        (bad / 'events.jsonl').unlink()
    elif damage == 'nonobject_frame':
        with gzip.open(bad / 'diagnostics/frames_0000.jsonl.gz', 'wt', encoding='utf-8') as stream:
            stream.write('null\n')
    elif damage == 'nonobject_event':
        (bad / 'events.jsonl').write_text('[]\n', encoding='utf-8')
    else:
        (bad / 'events.jsonl').write_text(json.dumps({'event': 'decision', 'decision_id': []}), encoding='utf-8')
    auditor = load_auditor()
    report = auditor.audit(bad, episode_loader=lambda _: episode)
    assert not report['awbc_recording_eligible']
    assert report['blockers']
    assert report['input_parse_errors'] or report['diagnostic_parse_errors'] or report['malformed_event_lines']
    assert auditor.audit(good, episode_loader=lambda _: good_episode)['awbc_recording_eligible']
