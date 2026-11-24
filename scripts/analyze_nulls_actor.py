"""Compare actors on the same recorded recurrent sequence, without game actions.

This measures policy disagreement on a fixed behavior trajectory, not win rates
or the counterfactual trajectory that a different actor would have generated.
"""
import argparse
import gc
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from run_nulls_dryrun import load_policy_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    torch.set_num_threads(2)
    checkpoints={'ppo697':Path('artifacts/ppo_v3_v100/actor_export.pt'),
                 'ppo184':Path('artifacts/ppo_v3_v100/actor_best_val.pt'),
                 'bc5112':Path('artifacts/ppo_prep_20261113/best.pt')}
    report={}
    for name,path in checkpoints.items():
        model,infer,_=load_policy_inference(path,a.device,compute_value=False)
        n=0;available=[];choice_wait=[];unaffordable=0
        for file in sorted((a.run/'diagnostics').glob('*.gz')):
            with gzip.open(file,'rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);n+=1
                    obs=np.asarray(row['obs'],dtype=np.int32)
                    mask=np.zeros(2307,dtype=bool);mask[row['legal_actions']]=True
                    logits,_=infer.evaluate(obs[None],mask[None],compute_value=False)
                    probs=logits.softmax(-1)[0].cpu().numpy()
                    if mask[1:].any():choice_wait.append(float(probs[0]))
                    if 'ZapMachine' in row['hand']:
                        slot=row['hand'].index('ZapMachine');indices=slice(1+slot*576,1+(slot+1)*576)
                        if mask[indices].any():available.append(float(probs[indices].sum()))
                        else:unaffordable+=1
        report[name]={'frames':n,'sparky_legal_frames':len(available),
                      'sparky_mean_probability_when_legal':float(np.mean(available)) if available else None,
                      'sparky_max_probability_when_legal':max(available,default=None),
                      'sparky_unavailable_frames':unaffordable,
                      'mean_wait_probability_when_choice':float(np.mean(choice_wait)) if choice_wait else None}
        print(name,json.dumps(report[name]),flush=True)
        del infer,model
        gc.collect()
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
    (a.run/'actor_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
