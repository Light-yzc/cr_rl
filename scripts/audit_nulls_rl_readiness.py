"""Read-only audit of real-client recordings; never invent rewards or terminal outcomes."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path


def audit(run, *, episode_loader=None):
    events = []
    malformed = 0
    input_errors = []
    try:
        event_lines = (run/'events.jsonl').read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        event_lines = []
        input_errors.append({'file':'events.jsonl','error':str(error)})
    for line in event_lines:
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Event must be an object')
            if row.get('event') == 'decision' and (
                    not isinstance(row.get('decision_id'), int) or isinstance(row['decision_id'], bool)):
                raise ValueError('Decision ID must be an integer')
            if row.get('event') == 'receipt' and (
                    not isinstance(row.get('receipt'), dict)
                    or not isinstance(row['receipt'].get('status'), str)):
                raise ValueError('Receipt must have a string status')
            events.append(row)
        except (ValueError, TypeError):
            malformed += 1
    manifest_path = run/'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or any(
                not isinstance(manifest.get(key, {}), dict) for key in ('arguments', 'checkpoint_metadata')):
            raise ValueError('Manifest and its metadata must be objects')
    except (OSError, ValueError, UnicodeError) as error:
        manifest = {}
        input_errors.append({'file':'manifest.json','error':str(error)})
    decisions = {r['decision_id']: r for r in events if r.get('event')=='decision' and 'decision_id' in r}
    receipt_counts = Counter(r.get('receipt', {}).get('status') for r in events if r.get('event')=='receipt')
    counts = Counter()
    counts['stale_action_skipped_events'] = sum(
        row.get('event') == 'stale_action_skipped' for row in events)
    gaps = Counter()
    reasons = Counter()
    fields = Counter()
    ticks = []
    ids = []
    probability_error_max = 0.
    logp_error_max = 0.
    previous = None
    parse_errors = []
    vetoes = []
    for path in sorted((run/'diagnostics').glob('frames_*.jsonl.gz')):
        try:
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for line in stream:
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise ValueError('Diagnostic frame must be an object')
                    counts['frames'] += 1
                    for key in ['obs','legal_actions','probabilities','selected_log_probability',
                                'match_id','decision_id','observed_monotonic',
                                'read_finished_monotonic','inference_finished_monotonic',
                                'action_selection']:
                        fields[key] += int(key in frame)
                    reasons[frame.get('reason','unknown')] += 1
                    tick = frame.get('tick')
                    identity = frame.get('decision_id')
                    ticks.append(tick)
                    ids.append(identity)
                    if isinstance(tick,int) and previous is not None:
                        gaps[tick-previous] += 1
                    previous = tick if isinstance(tick,int) else None
                    decision = decisions.get(identity,{})
                    counts['event_tick_aligned_frames'] += int(bool(decision) and decision.get('tick')==tick)
                    counts['terminal_clock_frames'] += int(bool(frame.get('snapshot',{}).get('clock_ended')))
                    obs = frame.get('obs',[])
                    probs = frame.get('probabilities',[])
                    legal = frame.get('legal_actions',[])
                    action = frame.get('action')
                    good_probs = len(probs)==2307 and all(isinstance(p,(int,float)) and math.isfinite(p) and p>=0 for p in probs)
                    counts['correct_observation_shape'] += int(len(obs)==7024)
                    counts['valid_probability_vectors'] += int(good_probs)
                    if good_probs:
                        error = abs(sum(probs)-1)
                        probability_error_max = max(probability_error_max,error)
                        counts['normalized_probability_vectors'] += int(error<1e-4)
                        counts['action_legal_in_recorded_mask'] += int(action in legal)
                        logp = frame.get('selected_log_probability')
                        if isinstance(action,int) and 0<=action<len(probs) and probs[action]>0 and isinstance(logp,(int,float)):
                            logp_error_max = max(logp_error_max,abs(logp-math.log(probs[action])))
                            counts['stored_logp_matches_raw_model_action_probability'] += int(abs(logp-math.log(probs[action]))<1e-3)
                    selection = frame.get('action_selection',{})
                    counts['explicit_sampled_proposal_frames'] += int('proposed_action' in selection)
                    vetoed = bool(selection.get('spell_vetoed',False) or decision.get('spell_vetoed',False))
                    counts['spell_veto_frames'] += int(vetoed)
                    vetoes.append(vetoed)
        except (OSError, EOFError, ValueError, TypeError, AttributeError) as error:
            parse_errors.append({'file':path.name,'error':str(error)})
    stops = [r for r in events if r.get('event')=='stopped']
    # Use the current importer as the source of truth. A terminal snapshot may
    # live in terminal.json rather than in the last policy diagnostic frame.
    contract_valid = False
    validation_status = 'rejected'
    validation_error = None
    usable_choices = excluded_policy_frames = 0
    try:
        if episode_loader is None:
            from cr_rl.client.rl_recording import load_episode
            episode_loader = load_episode
        episode = episode_loader(run)
        if len(vetoes) != len(episode['actions']):
            raise ValueError('Incomplete veto ledger')
        policy_mask = episode['policy_loss_mask']
        choices = episode['legal_masks'].sum(-1) > 1
        excluded_policy_frames = int((~policy_mask).sum())
        usable_choices = sum(bool(keep) and bool(choice) and not veto
                             for keep, choice, veto in zip(policy_mask, choices, vetoes, strict=True))
        contract_valid = True
        validation_status = 'valid'
        del episode
    except ImportError as error:
        validation_status = 'unavailable'
        validation_error = f'{type(error).__name__}: {error}'
    except (ValueError, OSError, KeyError, EOFError, TypeError) as error:
        validation_error = f'{type(error).__name__}: {error}'
    blocks = []
    if not manifest.get('outcome_verified') or manifest.get('outcome') is None:
        blocks.append('No verified win/loss/draw; terminal reward unavailable.')
    if not manifest.get('collection_complete'):
        blocks.append('Session not marked complete; cannot assume a true terminal.')
    mode = manifest.get('arguments',{}).get('selection_mode')
    if mode != 'policy_sample':
        blocks.append('No manifest evidence of policy_sample collection; raw model probabilities cannot be assumed to be behavior probabilities.')
    if counts['explicit_sampled_proposal_frames'] != counts['frames'] or not counts['frames']:
        blocks.append('Sampled proposals and post-selection actions are not fully distinguished.')
    if not counts['terminal_clock_frames'] and not (run/'terminal.json').is_file():
        blocks.append('No terminal-clock snapshot retained; battle_end alone does not verify result or final observation.')
    if parse_errors:
        blocks.append('Diagnostic stream has parse/integrity errors.')
    if input_errors:
        blocks.append('Manifest or event file has parse/integrity errors.')
    if malformed:
        blocks.append('Event stream has malformed records.')
    if not contract_valid:
        blocks.append('Episode contract validation ' + validation_status + ': ' + str(validation_error))
    elif not usable_choices:
        blocks.append('No usable non-vetoed Actor choice labels; retain as context only.')
    awbc_eligible = contract_valid and usable_choices > 0 and not blocks
    return {
        'run':str(run),'manifest_present':manifest_path.exists(),
        'checkpoint_sha256':manifest.get('checkpoint_sha256'),
        'checkpoint_update':manifest.get('checkpoint_metadata',{}).get('updates'),
        'selection_mode_recorded':mode,'execute':manifest.get('arguments',{}).get('execute'),
        'outcome':manifest.get('outcome'),'outcome_verified':manifest.get('outcome_verified',False),
        'collection_complete':manifest.get('collection_complete',False),
        'stop_reason':stops[-1].get('reason') if stops else manifest.get('stop_reason'),
        'counts':dict(counts),'receipt_counts':dict(receipt_counts),'field_coverage':dict(fields),
        'first_tick':ticks[0] if ticks else None,'last_tick':ticks[-1] if ticks else None,
        'first_decision_id':ids[0] if ids else None,
        'tick_gap_histogram':dict(sorted(gaps.items())),
        'decision_reasons':dict(reasons),'probability_sum_error_max':probability_error_max,
        'raw_logp_error_max':logp_error_max,'malformed_event_lines':malformed,
        'input_parse_errors':input_errors,
        'diagnostic_parse_errors':parse_errors,
        'episode_contract_valid':contract_valid,
        'episode_validation_status':validation_status,
        'episode_validation_error':validation_error,
        'excluded_policy_frames':excluded_policy_frames,
        'awbc_usable_choice_frames':usable_choices,
        'awbc_recording_eligible':bool(awbc_eligible),
        'direct_ppo_ready':False,'blockers':blocks,
        'direct_ppo_blockers':blocks + [
            'Matching collector checkpoint and old-log-probability replay have not been verified by this audit.'],
        'warning':'A valid recording is not proof of on-policy PPO eligibility or of sufficient/reliable AWBC cohorts.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    results = [audit(run) for run in args.runs]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for result in results:
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
