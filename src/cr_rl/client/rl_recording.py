"""Versioned real-client PPO records, verified outcomes and proposal-action semantics."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .schema_v4 import SchemaV4 as S

CONTRACT = 'crrl-live-proposal-v1'
OUTCOMES = {'win': 1., 'loss': -1., 'draw': 0.}


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def actor_fingerprint(checkpoint):
    """Same identity for a full training checkpoint and its matching Actor export.

    Call on CPU mmap state, before copying tensors to VRAM. Bounded temporary
    storage is one parameter; do not serialize the whole model into RAM.
    """
    import torch

    fields = ('config', 'temporal_config', 'learned_placement_definition',
              'model_architecture_version', 'decision_ticks')
    digest = hashlib.sha256(json.dumps({k: checkpoint.get(k) for k in fields},
                                      sort_keys=True, default=str).encode())
    state = checkpoint.get('model', checkpoint.get('actor'))
    if not isinstance(state, dict):
        raise ValueError('Checkpoint has no Actor state')
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            if name != '_extra_state' and not name.endswith('._extra_state'):
                raise ValueError(f'Unexpected non-tensor Actor state: {name}')
            digest.update(json.dumps([name, 'extra_state', tensor], sort_keys=True,
                                     allow_nan=False).encode())
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        # Byte view also supports BF16, which NumPy cannot directly represent.
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def record_terminal(run, *, match_id, snapshot, obs, legal_mask):
    """Store a real final observation; never infer a win from missing towers."""
    if not snapshot.clock_ended or snapshot.own_seat not in (0, 1):
        raise ValueError('Terminal observation needs ended clock and known seat')
    payload = {'format': 'crrl-live-terminal-v1', 'session_id': Path(run).name,
               'match_id': match_id, 'tick': snapshot.tick, 'time_ms': snapshot.time_ms,
               'observed_monotonic': snapshot.monotonic_ts, 'clock_ended': True,
               'own_seat': snapshot.own_seat, 'obs': np.asarray(obs).tolist(),
               'legal_actions': np.flatnonzero(legal_mask).tolist(),
               'snapshot': snapshot.to_dict()}
    atomic_json(Path(run)/'terminal.json', payload)
    return payload


def set_result(run, outcome, *, evidence, source='operator_confirmation'):
    """Persist an evidenced outcome, with operator/inference provenance kept explicit."""
    run = Path(run)
    if outcome not in OUTCOMES or not evidence.strip():
        raise ValueError('Provide win/loss/draw and nonempty result evidence')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    terminal_path = run/'terminal.json'
    if not terminal_path.exists():
        raise ValueError('No saved terminal observation; cannot certify a complete RL episode')
    terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
    if not terminal.get('clock_ended') or terminal.get('session_id') != run.name:
        raise ValueError('Invalid terminal evidence')
    if manifest.get('stop_reason') != 'battle_end' or not manifest.get('collection_complete'):
        raise ValueError('Session did not reach a recorded terminal boundary')
    result = {'format': 'crrl-live-result-v1', 'session_id': run.name,
              'match_id': terminal['match_id'], 'outcome': outcome,
              'reward': OUTCOMES[outcome], 'outcome_verified': True,
              'source': source, 'evidence': evidence.strip(), 'unix': time.time(),
              'terminal_sha256': sha256_file(terminal_path),
              'checkpoint_sha256': manifest.get('checkpoint_sha256')}
    existing = run/'result.json'
    if existing.exists():
        old = json.loads(existing.read_text(encoding='utf-8'))
        if old.get('outcome') != outcome:
            raise ValueError('Conflicting result already exists; review the original annotation')
    atomic_json(existing, result)
    manifest.update(outcome=outcome, outcome_verified=True, outcome_source=source)
    atomic_json(run/'manifest.json', manifest)
    return result


def _native_own(frame):
    snapshot=frame.get('snapshot',{})
    if not isinstance(snapshot,dict):return None
    seat=snapshot.get('own_seat');summoners=snapshot.get('summoners',[])
    if (type(seat) is int and seat in (0,1) and isinstance(summoners,(list,tuple))
            and len(summoners)==2 and isinstance(summoners[seat],dict)):
        return summoners[seat]
    return None


def _unexecuted_timeout(frames, terminal, events, event_index, idx, decisions):
    """Prove a click never became a play; never reinterpret it as WAIT."""
    before=decisions[idx]
    after=next((e for e in events[event_index+1:]
                if e.get('event') in ('decision','submitted','battle_end')), {})
    if after.get('event')=='decision' and after.get('decision_id',0)>idx:
        observed=frames[after['decision_id']-1]
        obs=observed['obs']
        old_native=_native_own(frames[idx-1]);new_native=_native_own(observed)
        native_consistent=(old_native is None and new_native is None) or (
            old_native is not None and new_native is not None
            and old_native.get('played_count')==before.get('played_count')
            and new_native.get('played_count')==before.get('played_count')
            and old_native.get('hand')==new_native.get('hand')
            and new_native.get('pending_commands')==[])
        if (type(before.get('played_count')) is int and before['played_count']>=0
                and after.get('played_count')==before['played_count']
                and events[event_index].get('played_count',before['played_count'])==before['played_count']
                and before.get('hand') and before['hand']==after.get('hand')
                and obs[S.H_PENDING_COUNT]==0 and obs[S.H_ELIXIR_RESERVED]==0 and native_consistent):
            return 'unchanged_count_hand_no_server_pending'
    # A terminal observation is also a next state. Late queued cards are
    # canceled by battle end; the score does not depend on their confirmation.
    if not any(e.get('event')=='submitted' for e in events[event_index+1:]):
        old=_native_own(frames[idx-1]);end=_native_own(terminal)
        if (old is not None and end is not None and terminal.get('clock_ended')
                and type(old.get('played_count')) is int and old['played_count']>=0
                and old['played_count']==before.get('played_count')
                and end.get('played_count')==old['played_count']
                and len(old.get('hand',[]))==4
                and all(type(card) is int and 0<=card<8 for card in old['hand'])
                and old['hand']==end.get('hand')):
            return 'ended_battle_unchanged_native_count_and_hand'
    return None


def load_episode(run, *, purpose='ppo'):
    """Strictly validate and import a complete proposal-policy rollout.

    No collection result alone can make legacy greedy data on-policy. Forced
    WAIT and all intervened proposals remain in chronological recurrent context.
    Explicitly skipped executions retain their sampled proposal but do not
    provide an Actor label; missing execution provenance rejects the episode.
    """
    if purpose not in ('ppo','awbc'):raise ValueError('Unknown training purpose')
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    contract = manifest.get('rl_contract', {})
    if contract.get('version') != CONTRACT or contract.get('action_semantics') != 'policy_proposal':
        raise ValueError('Recording lacks the proposal-action RL contract; legacy logs are not PPO rollouts')
    if manifest.get('arguments', {}).get('selection_mode') != 'policy_sample':
        raise ValueError('PPO collection requires policy_sample behavior')
    if not manifest.get('arguments', {}).get('execute'):
        raise ValueError('Dry-run observations are not an executed-policy rollout')
    if not contract.get('initial_memory_reset') or not contract.get('actor_fingerprint'):
        raise ValueError('Missing behavior identity or recurrent reset provenance')
    if not manifest.get('outcome_verified') or manifest.get('outcome') not in OUTCOMES:
        raise ValueError('No verified terminal outcome')
    if not manifest.get('collection_complete') or manifest.get('stop_reason') != 'battle_end':
        raise ValueError('Incomplete recording is not a complete PPO episode')
    terminal_path = run/'terminal.json'
    terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    if (result.get('terminal_sha256') != sha256_file(terminal_path)
            or result.get('checkpoint_sha256') != manifest.get('checkpoint_sha256')
            or result.get('outcome') != manifest['outcome']
            or not result.get('outcome_verified') or not result.get('evidence')):
        raise ValueError('Result provenance mismatch')
    events = []
    with (run/'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            events.append(json.loads(line))
    if any(e.get('event') in ('unattributed_play', 'error') for e in events):
        raise ValueError('Unattributed manual play or collection error in session')
    decision_events = [e for e in events if e.get('event') == 'decision']
    decision_ids = [e.get('decision_id') for e in decision_events]
    if (any(type(idx) is not int or idx < 1 for idx in decision_ids)
            or len(set(decision_ids)) != len(decision_ids)):
        raise ValueError('Invalid or duplicate decision events')
    decisions = dict(zip(decision_ids, decision_events))
    frames = []
    for path in sorted((run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            frames.extend(json.loads(line) for line in stream)
    if not frames or len(frames) != manifest.get('frames') or len(decisions) != len(frames):
        raise ValueError('Frame/event/manifest count mismatch')
    starts = [e for e in events if e.get('event') == 'battle_start']
    first_tick = frames[0].get('tick')
    # The client can signal battle start at tick -1 before its first playable
    # clock frame. No inference is done at that sentinel.
    if (len(starts) != 1 or type(first_tick) is not int or not 0 <= first_tick <= 8
            or starts[0].get('tick') not in (-1, first_tick)):
        raise ValueError('Capture must start at the beginning of a single battle')
    ids = [f.get('decision_id') for f in frames]
    if ids != list(range(1, len(frames)+1)):
        raise ValueError('Missing/reordered recurrent prefix or duplicated decisions')
    match_ids = {f.get('match_id') for f in frames}
    if (len(match_ids)!=1 or None in match_ids or terminal.get('match_id') not in match_ids
            or result.get('match_id') != terminal.get('match_id')
            or terminal.get('session_id') != run.name or result.get('session_id') != run.name
            or not terminal.get('clock_ended')):
        raise ValueError('Mixed match identities or missing terminal')
    raw, masks, actions, logps, post_actions = [], [], [], [], []
    for frame in frames:
        decision = decisions[frame['decision_id']]
        if decision.get('tick') != frame['tick']:
            raise ValueError('Event/observation tick mismatch')
        probabilities = np.asarray(frame.get('probabilities'), dtype=np.float64)
        if (probabilities.shape != (S.ACTIONS,) or not np.isfinite(probabilities).all()
                or (probabilities<0).any() or abs(probabilities.sum()-1)>1e-4):
            raise ValueError('Invalid behavior probability vector')
        selection = frame.get('action_selection', {})
        action = selection.get('proposed_action')
        if not isinstance(action,int) or not 0<=action<S.ACTIONS:
            raise ValueError('Missing sampled proposal')
        mask = np.zeros(S.ACTIONS, dtype=bool)
        legal = frame.get('legal_actions', [])
        if not legal or any(not isinstance(a,int) or not 0<=a<S.ACTIONS for a in legal):
            raise ValueError('Invalid legal mask')
        mask[legal] = True
        if not mask[action] or probabilities[action]<=0 or probabilities[~mask].sum()>1e-6:
            raise ValueError('Proposal is illegal or has invalid behavior mass')
        post = frame.get('action')
        if post != decision.get('action') or post != (0 if selection.get('spell_vetoed') else action):
            raise ValueError('Proposal/controller action mismatch')
        old_lp = frame.get('proposal_log_probability')
        if not isinstance(old_lp,(float,int)) or not math.isfinite(old_lp) or abs(old_lp-math.log(probabilities[action]))>1e-3:
            raise ValueError('Proposal log-probability mismatch')
        obs = np.asarray(frame['obs'])
        if obs.shape != (S.SEAT_INTS,) or not np.issubdtype(obs.dtype, np.integer):
            raise ValueError('Invalid observation shape/dtype')
        if int(obs[S.H_TICK]) != frame['tick']:
            raise ValueError('Observation tick differs from recorded tick')
        raw.append(obs.astype(np.int32)); masks.append(mask); actions.append(action)
        logps.append(old_lp); post_actions.append(frame['action'])
    ticks = np.array([f['tick'] for f in frames]+[terminal['tick']], dtype=np.int64)
    durations = np.diff(ticks)
    if (durations[:-1]<=0).any() or durations[-1]<0:
        raise ValueError('Nonpositive action-to-next-state duration')
    if purpose=='ppo' and durations.max() > 16:
        raise ValueError('Observation gap exceeds 0.8s; incomplete sequence')
    if purpose=='awbc' and durations.max()>200:
        raise ValueError('Real-game pause exceeds 10s; isolate this recording')
    terminal_obs = np.asarray(terminal['obs'], dtype=np.int32)
    if terminal_obs.shape != (S.SEAT_INTS,) or terminal_obs[S.H_TICK] != terminal['tick']:
        raise ValueError('Invalid terminal observation')
    rewards = np.zeros(len(frames), dtype=np.float32)
    rewards[-1] = OUTCOMES[manifest['outcome']]
    terminated = np.zeros(len(frames), dtype=bool); terminated[-1] = True
    executed = np.zeros(len(frames), dtype=np.int64)
    policy_loss_mask = np.ones(len(frames), dtype=bool)
    irregular=np.flatnonzero(durations>16)
    if purpose=='awbc':
        # Offline imitation permits irregular observation times. Never invent
        # missed frames: retain actual tick durations and mask both boundaries.
        policy_loss_mask[irregular]=False
        policy_loss_mask[np.minimum(irregular+1,len(frames)-1)]=False
    execution_exclusions=[]
    submits = {}
    skipped = set()
    seen_decisions = set()
    for event_index, event in enumerate(events):
        kind = event.get('event')
        if kind == 'decision':
            seen_decisions.add(event['decision_id'])
            continue
        if kind not in ('submitted','receipt','stale_action_skipped'):
            continue
        idx = event.get('decision_id')
        if type(idx) is not int or not 1 <= idx <= len(frames) or idx not in seen_decisions:
            raise ValueError('Execution event has no matching decision')
        if kind == 'stale_action_skipped':
            if (type(event.get('action')) is not int or not post_actions[idx-1]
                    or event['action'] != post_actions[idx-1]):
                raise ValueError('Stale skip does not match a non-WAIT controller action')
            if idx in submits or idx in skipped:
                raise ValueError('Conflicting or duplicate stale skip execution evidence')
            # The controller never sent this proposal. Keep its observation,
            # duration and action for recurrent context, never relabel it WAIT.
            skipped.add(idx)
            policy_loss_mask[idx-1] = False
            continue
        if idx in skipped:
            raise ValueError('Conflicting stale skip and submission/receipt evidence')
        receipt = event['receipt']
        if receipt.get('match_id') != terminal['match_id'] or receipt.get('action') != post_actions[idx-1] or not post_actions[idx-1]:
            raise ValueError('Execution receipt does not match controller action')
        if event['event']=='submitted':
            if idx in submits or receipt.get('status')!='sent_unconfirmed':
                raise ValueError('Invalid or duplicate submission')
            submits[idx] = False
        else:
            if idx not in submits or submits[idx]:
                raise ValueError('Unconfirmed or ambiguous execution; inspect this session')
            status = receipt.get('status')
            if status in ('timed_out','rejected') and 1 <= post_actions[idx-1] < S.FIRST_ABILITY:
                proof=_unexecuted_timeout(frames,terminal,events,event_index,idx,decisions)
                if proof is None:
                    raise ValueError('Unconfirmed or ambiguous execution; inspect this session')
                policy_loss_mask[idx-1] = False
                execution_exclusions.append(dict(decision_id=idx,proof=proof))
                submits[idx] = True
                continue
            if status != 'confirmed':
                raise ValueError('Unconfirmed or ambiguous execution; inspect this session')
            submits[idx] = True
            executed[idx-1] = post_actions[idx-1]
    if not all(submits.values()):
        raise ValueError('Some submissions have no confirmation')
    expected_submissions = {idx+1 for idx, action in enumerate(post_actions) if action}
    if expected_submissions - submits.keys() - skipped:
        raise ValueError('Non-WAIT controller action has no submission or verified skip')
    legal_masks=np.stack(masks)
    actor_keep=(policy_loss_mask & (legal_masks.sum(-1)>1)
                & ~np.array([bool(f.get('action_selection',{}).get('spell_vetoed')) for f in frames]))
    return {'format':CONTRACT, 'session_id':run.name, 'manifest':manifest,
            'raw_obs':np.stack(raw), 'legal_masks':legal_masks,'actor_keep':actor_keep,
            'actions':np.array(actions,dtype=np.int64), 'old_logp':np.array(logps,dtype=np.float32),
            'post_selection_actions':np.array(post_actions,dtype=np.int64),
            'confirmed_actions':executed,
            'policy_loss_mask':policy_loss_mask,
            'training_purpose':purpose,'max_observation_gap_ticks':int(durations.max()),
            'irregular_transition_indices':irregular.tolist(),
            'execution_exclusions':execution_exclusions,
            'rewards':rewards, 'terminated':terminated, 'truncated':np.zeros(len(frames),dtype=bool),
            'step_ticks':durations, 'terminal_obs':terminal_obs,
            'behavior_version':contract['actor_fingerprint']}


def as_ppo_segment(episode):
    """Existing state-PPO segment; irregular offline-only timing is rejected."""
    if episode.get('max_observation_gap_ticks',0)>16:
        raise ValueError('Irregular AWBC recording is not an admitted PPO segment')
    n = len(episode['actions'])
    return {k:episode[k] for k in ('raw_obs','legal_masks','actions','rewards',
                                   'terminated','truncated','step_ticks','behavior_version')} | {
        'old_log_probs':episode['old_logp'],
        'policy_loss_mask':episode.get('policy_loss_mask', np.ones(n, dtype=bool)),
        'raw_rewards':episode['rewards'].copy(), 'valid':np.ones(n,dtype=bool),
        'learner_mask':np.ones(n,dtype=bool), 'dones':episode['terminated'].copy(),
        'old_values':np.zeros(n,dtype=np.float32), 'bootstrap_value':0.,
        'prefix_raw':episode['raw_obs'][:0],
        'h0_actor':None, 'h0_behavior':None, 'h0_critic':None,
        'full_episode':True, 'has_complete_prefix':True,
        'decision_ticks':8, 'live_session_id':episode['session_id']}


def export_episode(run):
    episode = load_episode(run)
    out = Path(run)/'training'
    out.mkdir(exist_ok=True)
    path = out/'rl_episode.npz'
    temporary = path.with_suffix('.npz.partial')
    arrays = {k:v for k,v in episode.items() if isinstance(v,np.ndarray)}
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary,path)
    summary = {'format':CONTRACT, 'session_id':Path(run).name,
               'frames':len(episode['actions']), 'outcome':episode['manifest']['outcome'],
               'ppo_eligible':True, 'behavior_version':episode['behavior_version'],
               'excluded_policy_frames':int((~episode['policy_loss_mask']).sum()),
               'action_semantics':'policy_proposal', 'reward_scheme':'verified_terminal_only',
               'npz_sha256':sha256_file(path)}
    atomic_json(out/'rl_summary.json',summary)
    return summary
