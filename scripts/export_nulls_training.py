"""Export real-client sequences with honest execution labels and unknown returns.

Policy-generated actions are not expert demonstrations. These partial real-game
observations are off-policy data, not a drop-in PPO rollout or simulator replay.
"""
import argparse
import gzip
import json
from pathlib import Path


def training_rows(run):
    events=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    outcomes={};decisions={};previous_id=None;pending_id=None
    for event in events:
        if event['event']=='decision':
            previous_id=event.get('decision_id',len(decisions)+1)
            decisions[previous_id]=event
        elif event['event']=='submitted' and event['receipt']['status']=='sent_unconfirmed':
            pending_id=event.get('decision_id',previous_id)
        elif event['event']=='receipt':
            identity=event.get('decision_id',pending_id)
            if identity is not None:outcomes[identity]=event['receipt']['status']
            pending_id=None
    manifest_path=run/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    ordinal=0
    for file in sorted((run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);ordinal+=1
                identity=row.get('decision_id',ordinal)
                decision=decisions.get(identity,{})
                # Old rolling logs may have lost their beginning: do not align
                # retained tails by position and accidentally label other frames.
                aligned=decision.get('tick')==row.get('tick')
                status=outcomes.get(identity,'not_submitted') if aligned else 'unlinked'
                selection=row.get('action_selection',{})
                # A controller veto becomes WAIT at execution, but is not a
                # model WAIT demonstration. Keep the frame for recurrent context.
                vetoed=bool(selection.get('spell_vetoed') or decision.get('spell_vetoed')
                            or selection.get('selection_reason')=='spell_guard_veto'
                            or decision.get('selection_reason')=='spell_guard_veto')
                target=None
                if status=='confirmed':target=row['action']
                elif aligned and not vetoed and row['action']==0 and row['reason']=='policy_wait':target=0
                yield {**row,'format':'crrl-null-training-frame-v1',
                       'session_id':run.name,'decision_id':identity,
                       'execution_status':status,'behavior_cloning_target':target,
                       'behavior_cloning_eligible':target is not None,
                       'demonstrator':'model_policy_not_expert',
                       'reward':None,'return':None,'ppo_eligible':False,
                       'outcome':manifest.get('outcome'),
                       'outcome_verified':manifest.get('outcome_verified',False),
                       'checkpoint_sha256':manifest.get('checkpoint_sha256'),
                       'tick_gap':decision.get('tick_gap') if aligned else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--rl',action='store_true',help='Validate and export a versioned proposal-policy PPO episode')
    a=p.parse_args()
    if a.rl:
        from cr_rl.client.rl_recording import export_episode
        print(json.dumps(export_episode(a.run),ensure_ascii=False))
        return
    out=a.run/'training';out.mkdir(exist_ok=True)
    report={'frames':0,'confirmed_plays':0,'policy_wait_labels':0,'unlabeled':0}
    with gzip.open(out/'frames.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in training_rows(a.run):
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
            report['frames']+=1
            if row['behavior_cloning_target'] is None:report['unlabeled']+=1
            elif row['behavior_cloning_target']==0:report['policy_wait_labels']+=1
            else:report['confirmed_plays']+=1
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
