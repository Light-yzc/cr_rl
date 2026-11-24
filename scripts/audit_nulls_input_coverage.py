"""Audit schema coverage from recorded real-client inputs; no ADB or game actions."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('artifacts/ppo_v3_v100/actor_export.pt'))
    p.add_argument('--run',type=Path,default=Path('artifacts/nulls_live/play_20261115_210255'))
    p.add_argument('--out',type=Path,default=Path('artifacts/nulls_live/input_coverage_20261116.json'))
    a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=checkpoint['config'];hf=cfg['header_fields'];ef=cfg['entity_fields']
    count_index=hf.index('entity_count')
    header=[Counter() for _ in hf];entity=[Counter() for _ in ef]
    frames=0;moving=0;tracked=0
    for file in sorted((a.run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as stream:
            for line in stream:
                obs=np.asarray(json.loads(line)['obs'],dtype=np.int32)
                frames+=1
                for i,value in enumerate(obs[:len(hf)]):header[i][int(value)]+=1
                rows=obs[len(hf):].reshape(-1,len(ef))[:max(0,int(obs[count_index]))]
                if len(rows):
                    moving+=int(((rows[:,ef.index('dx')]!=0)|(rows[:,ef.index('dy')]!=0)).sum())
                    tracked+=len(rows)
                for i in range(len(ef)):entity[i].update(int(x) for x in rows[:,i])
    def summarize(names,counts):
        return [{'index':i,'name':name,'samples':sum(c.values()),'minus_one_count':c[-1],
                 'unique_values':len(c),'min':min(c) if c else None,'max':max(c) if c else None}
                for i,(name,c) in enumerate(zip(names,counts))]
    report={'checkpoint':str(a.checkpoint),'update':checkpoint.get('updates'),
            'architecture':checkpoint.get('model_architecture_version'),
            'decision_ticks':checkpoint.get('decision_ticks'),'recording':str(a.run),
            'frames':frames,'entity_rows':tracked,'nonzero_motion_rows':moving,
            'warning':'Old recording coverage, not evidence of new live reader offsets. '
                      'Negative dx/dy are valid displacements, not UNKNOWN sentinels.',
            'model_preprocessing_zeroes':['entity.id','entity.target'],
            'header':summarize(hf,header),'entities':summarize(ef,entity)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('header','entities')},indent=2))
    print('Entity fields always -1:',[r['name'] for r in report['entities'] if r['samples'] and r['minus_one_count']==r['samples']])


if __name__=='__main__':main()
