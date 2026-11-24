"""Backfill real terminal rewards; preserve unknown/conflicting games for review."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from cr_rl.client.outcome import finalize_result
from cr_rl.client.rl_recording import atomic_json, export_episode, load_episode


def sweep(root, export=False, validate=False, legacy_only=False, purpose='ppo'):
    counts=Counter();details=[]
    for path in sorted(root.glob('*/manifest.json')):
        manifest=json.loads(path.read_text(encoding='utf8'))
        if not manifest.get('collection_complete'):continue
        run=path.parent;existed=(run/'result.json').exists()
        if legacy_only and manifest.get('outcome_policy') and not existed:
            counts['awaiting_controller_result']+=1
            continue
        try:
            result=finalize_result(run)
            counts['already_labeled' if existed else 'backfilled']+=1
            counts[result['outcome']]+=1
            row=dict(run=run.name,outcome=result['outcome'],reward=result['reward'])
            if export or validate:
                try:
                    if export:
                        report=export_episode(run);frames=report['frames']
                    else:
                        episode=load_episode(run,purpose=purpose);frames=len(episode['actions'])
                        if purpose=='awbc' and not episode['actor_keep'].any():
                            raise ValueError('No usable AWBC choice labels')
                        row.update(masked_policy_rows=int((~episode['policy_loss_mask']).sum()),
                            usable_actor_rows=int(episode['actor_keep'].sum()),
                            irregular_transitions=len(episode.get('irregular_transition_indices',[])),
                            max_observation_gap_ticks=episode.get('max_observation_gap_ticks'),
                            reconciled_unexecuted=len(episode.get('execution_exclusions',[])))
                        del episode
                    row['training_eligible']=True
                    counts['training_eligible']+=1;row['frames']=frames
                except (ValueError,OSError,EOFError,KeyError) as error:
                    row.update(training_eligible=False,training_rejection=str(error))
                    counts['training_rejected']+=1
        except (ValueError,OSError,KeyError) as error:
            counts['unresolved']+=1;row=dict(run=run.name,error=str(error))
        details.append(row)
    report=dict(unix=time.time(),training_purpose=purpose,counts=dict(counts),games=details)
    filename='result_training_readiness.json' if export or validate else 'result_recovery_report.json'
    if (export or validate) and purpose=='awbc':filename='result_awbc_readiness.json'
    atomic_json(root/filename,report)
    print(json.dumps(dict(counts=counts,training_checked=export or validate)),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions',type=Path,required=True)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--validate',action='store_true',help='Check training contract without duplicating arrays')
    parser.add_argument('--purpose',choices=('ppo','awbc'),default='ppo')
    parser.add_argument('--watch',action='store_true',help='Reconcile old in-flight children every 30s')
    args=parser.parse_args()
    if args.export and args.purpose!='ppo':parser.error('Use --validate for AWBC; PPO export retains strict timing')
    while True:
        sweep(args.sessions,args.export,args.validate,legacy_only=args.watch,purpose=args.purpose)
        if not args.watch:break
        time.sleep(30)
