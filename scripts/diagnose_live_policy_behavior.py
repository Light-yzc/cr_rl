"""Fixed-trajectory head ablations; no training, game actions, or checkpoint writes."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from run_nulls_dryrun import load_policy_inference
from cr_rl.client.schema_v4 import SchemaV4 as S


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    checkpoint=Path(manifest['checkpoint'])
    model,infer,_=load_policy_inference(checkpoint,'cuda',compute_value=False)
    torch.set_num_threads(2)
    events=[json.loads(l) for l in (a.run/'events.jsonl').read_text().splitlines()]
    decisions={r['decision_id']:r for r in events if r['event']=='decision'}
    compose=model._joint_log_probs
    components={};summaries=defaultdict(list);rows=[]
    placement=model.learned_placement
    decoder=placement.decoder
    prior=decoder.original
    spatial=decoder.spatial
    original_prior=prior.prior_logits
    original_spatial=spatial.residual_logits
    def capture_prior(*args,**kwargs):
        result=original_prior(*args,**kwargs);components['prior']=result;return result
    def capture_spatial(*args,**kwargs):
        result=original_spatial(*args,**kwargs);components.setdefault('cnn',[]).append(result);return result
    prior.prior_logits=capture_prior
    spatial.residual_logits=capture_spatial
    gate_hook=model.wait_hand_gate.register_forward_hook(lambda m,i,o:components.update(gate=o))
    current={}
    def joint(cards,tiles,legal,*,hand_present_mask=None):
        baseline=compose(cards,tiles,legal,hand_present_mask=hand_present_mask)
        variants={'baseline':baseline,'no_reservation':compose(cards,tiles,legal,hand_present_mask=None)}
        without_gate=cards.clone();without_gate[...,0]-=components['gate'][...,0]
        variants['no_wait_gate']=compose(without_gate,tiles,legal,hand_present_mask=hand_present_mask)
        temperature=placement.definition['temperature']
        if 'prior' in components:
            variants['no_location_prior']=compose(cards,tiles-components['prior']/temperature,legal,
                                                  hand_present_mask=hand_present_mask)
            variants['no_cnn']=compose(cards,tiles-torch.stack(components['cnn'],1)/temperature,legal,
                                       hand_present_mask=hand_present_mask)
        current['gate']=float(components['gate'][0,0])
        current['variants']={name:value.exp()[0].cpu().numpy() for name,value in variants.items()}
        return baseline
    model._joint_log_probs=joint
    for f in sorted((a.run/'diagnostics').glob('*.gz')):
        with gzip.open(f,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);d=decisions[r['decision_id']]
                obs=np.array(r['obs'],dtype=np.int32);legal=np.zeros(S.ACTIONS,dtype=bool);legal[r['legal_actions']]=True
                components.clear();current.clear()
                infer.evaluate(obs[None],legal[None],compute_value=False)
                free=d['reason'] in ('policy_wait','action_ready','stale_state')
                result={'decision_id':r['decision_id'],'tick':d['tick'],'free_choice':free,'gate':current['gate'],
                        'hand':d['hand'],'variants':{}}
                base=current['variants']['baseline']
                result['reproduction_max_probability_error']=float(np.max(np.abs(base-np.array(r['probabilities']))))
                for name,probs in current['variants'].items():
                    slots=probs[1:2305].reshape(4,576).sum(1)
                    result['variants'][name]={'wait':float(probs[0]),'slots':slots.tolist(),
                        'top_card':int(slots.argmax()),
                        'top_tiles':probs[1:2305].reshape(4,576).argmax(1).tolist()}
                    if free:summaries[name].append(float(probs[0]))
                rows.append(result)
    gate_hook.remove();model._joint_log_probs=compose
    report={'run':str(a.run),'checkpoint_metadata':model.deployment_metadata,
            'frames':len(rows),'free_choice_frames':sum(r['free_choice'] for r in rows),
            'max_reproduction_probability_error':max(r['reproduction_max_probability_error'] for r in rows),
            'mean_wait_probability_free_choice':{k:float(np.mean(v)) for k,v in summaries.items()},
            'gate_free_choice':{k:float(fn([r['gate'] for r in rows if r['free_choice']]))
                                for k,fn in [('min',min),('mean',np.mean),('max',max)]},
            'top_tile_changes':{name:sum(
                x!=y for r in rows if r['free_choice'] and name in r['variants']
                for x,y in zip(r['variants']['baseline']['top_tiles'],r['variants'][name]['top_tiles']))
                for name in ('no_location_prior','no_cnn')},
            'warning':'Head ablations on a fixed real-game trajectory; not win-rate or counterfactual-game evidence.'}
    (a.run/'behavior_head_ablations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with gzip.open(a.run/'behavior_head_ablations_frames.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
