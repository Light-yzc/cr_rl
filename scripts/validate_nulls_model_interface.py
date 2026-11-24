"""Strict-load the current actor and validate recorded model inputs without ADB."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_nulls_dryrun import load_policy_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('artifacts/ppo_v3_v100/actor_export.pt'))
    p.add_argument('--run',type=Path,default=Path('artifacts/nulls_live/final_verify_20261116_v2'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',type=Path,default=Path('artifacts/nulls_live/model_interface_validation.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    model,infer,ticks=load_policy_inference(a.checkpoint,a.device,compute_value=False)
    frames=0;durations=[]
    for file in sorted((a.run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                obs=np.asarray(row['obs'],dtype=np.int32)
                mask=np.zeros(2307,dtype=bool);mask[row['legal_actions']]=True
                start=time.perf_counter()
                logits,_=infer.evaluate(obs[None],mask[None],compute_value=False)
                assert logits.shape==(1,2307)
                assert bool(torch.isfinite(logits).all()),'Nonfinite actor logits'
                probs=logits.softmax(-1)[0].cpu().numpy()
                assert np.isclose(probs.sum(),1.,atol=1e-5)
                assert not np.any(probs[~mask]>1e-6),'Probability assigned to masked actions'
                durations.append(1000*(time.perf_counter()-start));frames+=1
    if not frames:raise ValueError('No recorded observations found')
    report={'checkpoint':str(a.checkpoint),'checkpoint_metadata':model.deployment_metadata,
            'frames':frames,'precision':infer.precision,'decision_ticks':ticks,
            'all_logits_finite':True,'all_masks_respected':True,
            'inference_p50_ms':float(np.median(durations)),
            'inference_max_including_cold_ms':max(durations),
            'note':'Offline interface validation; no game actions or claim of complete real-game coverage.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
