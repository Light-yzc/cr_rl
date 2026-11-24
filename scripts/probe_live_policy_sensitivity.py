"""Read-only policy precision and current-enemy feature sensitivity on fixed history."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from run_nulls_dryrun import load_policy_inference
from cr_rl.models.state_policy import clone_memory
from cr_rl.client.schema_v4 import SchemaV4 as S


def card_distribution(p):
    return np.concatenate((p[:1],p[1:2305].reshape(4,576).sum(1),p[2305:]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    a=parser.parse_args();torch.set_num_threads(2)
    manifest=json.loads((a.run/'manifest.json').read_text())
    model,infer,_=load_policy_inference(manifest['checkpoint'],'cuda',compute_value=False)
    model.ppo_actor_fp32=True
    events=[json.loads(line) for line in (a.run/'events.jsonl').read_text().splitlines()]
    free_ids={e['decision_id'] for e in events if e['event']=='decision'
              and e.get('reason') in ('policy_wait','action_ready','stale_state')}
    precisions=[];probes=[];last_probe=-1000;frames=0
    for file in sorted((a.run/'diagnostics').glob('*.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);obs=np.asarray(row['obs'],dtype=np.int32)
                legal=np.zeros(S.ACTIONS,dtype=bool);legal[row['legal_actions']]=True
                entities=obs[S.HEADER_INTS:].reshape(S.MAX_ENTITIES,S.ENTITY_INTS)
                count=int(obs[S.H_ENTITY_COUNT]);visible=entities[:count]
                threats=(visible[:,S.E_KIND]==S.KIND_CHARACTER)&(visible[:,S.E_SIDE]==1)&(visible[:,S.E_HP]>0)&(visible[:,S.E_Y]<16000)
                eligible=bool(row['decision_id'] in free_ids and legal[1:2305].any()
                              and threats.any() and obs[S.H_TICK]-last_probe>=100 and len(probes)<12)
                before=clone_memory(infer.state) if eligible else None
                logits,_=infer.evaluate(obs[None],legal[None],compute_value=False)
                p=logits.softmax(-1)[0].cpu().numpy();after=infer.state
                original=np.asarray(row['probabilities'])
                precisions.append({'tick':int(obs[S.H_TICK]),
                    'joint_max_error':float(np.max(np.abs(p-original))),
                    'card_l1_error':float(np.abs(card_distribution(p)-card_distribution(original)).sum())})
                if eligible:
                    result={'tick':int(obs[S.H_TICK]),'threats':int(threats.sum()),
                            'baseline_card_distribution':card_distribution(p).tolist()}
                    for name in ('enemy_removed','enemy_lane_mirrored'):
                        changed=obs.copy();rows=changed[S.HEADER_INTS:].reshape(S.MAX_ENTITIES,S.ENTITY_INTS)
                        enemy=(visible[:,S.E_KIND]==S.KIND_CHARACTER)&(visible[:,S.E_SIDE]==1)
                        if name=='enemy_removed':
                            kept=visible[~enemy].copy();rows[:]=0;rows[:len(kept)]=kept
                            changed[S.H_ENTITY_COUNT]=len(kept)
                        else:
                            rows[:count,S.E_X][enemy]=18000-rows[:count,S.E_X][enemy]
                        infer.state=clone_memory(before)
                        other,_=infer.evaluate(changed[None],legal[None],compute_value=False)
                        q=other.softmax(-1)[0].cpu().numpy()
                        result[name]={'card_l1_change':float(np.abs(card_distribution(p)-card_distribution(q)).sum()),
                            'card_distribution':card_distribution(q).tolist(),
                            'top_tile_changed_cards':int((p[1:2305].reshape(4,576).argmax(1)!=q[1:2305].reshape(4,576).argmax(1)).sum())}
                    infer.state=after;last_probe=int(obs[S.H_TICK]);probes.append(result)
                frames+=1
                if frames%100==0:print(f'checked {frames} frames',flush=True)
    report={'frames':frames,'precision_fp32_vs_recorded_fp16':{
                'max_joint_probability_error':max(x['joint_max_error'] for x in precisions),
                'mean_card_l1_error':float(np.mean([x['card_l1_error'] for x in precisions])),
                'max_card_l1_error':max(x['card_l1_error'] for x in precisions)},
            'probes':probes,
            'warning':'Feature ablations hold history and legal mask fixed; not physical alternate games or evidence of correct tactics.'}
    (a.run/'behavior_sensitivity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='probes'},indent=2),flush=True)
    print('probes',len(probes),'enemy-removal mean card L1',np.mean([r['enemy_removed']['card_l1_change'] for r in probes]),flush=True)


if __name__=='__main__':main()
