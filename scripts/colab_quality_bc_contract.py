"""Audit committed native BC/model contracts and run bounded CPU inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/content/crrl_training')
WORK=ROOT/'public_bc_full'


def worker():
    import numpy as np
    import torch
    from catalog_quality_bc import catalog
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.models.state_policy import state_model_from_checkpoint
    from run_nulls_dryrun import schema_from_model_config
    torch.set_num_threads(2)
    summary=catalog(WORK)
    metadata=json.loads((WORK/'dataset_metadata.json').read_text())
    checkpoint=torch.load(ROOT/'prepared/anchor_actor.pt',map_location='cpu',mmap=True,weights_only=False)
    actor=state_model_from_checkpoint(checkpoint,device='cpu').eval()
    schema=schema_from_model_config(actor.cfg)
    source=metadata['schema']
    if tuple(source['header_fields'])!=tuple(schema.header_fields):
        raise ValueError('BC header order differs from model')
    if tuple(source['entity_fields'])!=tuple(schema.entity_fields):
        raise ValueError('BC entity field order differs from model')
    vocabulary=metadata['vocabulary']
    names=vocabulary['names'] if isinstance(vocabulary,dict) else vocabulary
    if len(names)!=actor.cfg.vocab_size:
        raise ValueError('BC vocabulary differs from model embedding size')
    if isinstance(vocabulary,dict):
        for key in ('data_version','content_sha'):
            if vocabulary.get(key)!=checkpoint.get(key):
                raise ValueError('BC/model data provenance mismatch: '+key)
    adapter=CrForgeObservationAdapter(schema,vocabulary_size=actor.cfg.vocab_size,entity_bucket=16)
    manifest=json.loads((WORK/'dataset.manifest.json').read_text())
    split=json.loads((WORK/'split_registry.json').read_text())['assignments']
    probes=[]
    for shard in manifest['shards']:
        replay=metadata['replays'][shard['replay_offset']]
        if split[replay['canonical_game_signature']]!='train':continue
        with np.load(WORK/shard['path'],allow_pickle=False) as arrays:
            seats=arrays['seats'];rows=np.flatnonzero(seats==0)[:32]
            raw=arrays['observations'][rows][None]
            masks=arrays['masks'][rows][None].astype(bool)
            labels=arrays['actions'][rows]
            if raw.shape[-1]!=schema.seat_ints or masks.shape[-1]!=2307:
                raise ValueError('BC shape mismatch')
            if not np.all(masks[0,np.arange(len(rows)),labels]):
                raise ValueError('Illegal target in accepted BC shard')
            h,e,m=adapter(raw)
            with torch.no_grad():
                logits,_,_=actor.forward_state(torch.as_tensor(h),torch.as_tensor(e),torch.as_tensor(m),
                    torch.as_tensor(masks),hand_card_ids=torch.as_tensor(raw[...,list(schema.header['hand_card'])]),
                    compute_value=False)
            if not torch.isfinite(logits[torch.as_tensor(masks)]).all():
                raise ValueError('Nonfinite legal model outputs')
            probes.append(dict(shard=shard['path'],frames=len(rows),finite_legal_logits=True,
                               output_shape=list(logits.shape)))
        if len(probes)>=2:break
    if not probes:raise ValueError('No committed training game available')
    report=dict(catalog=summary,model_input_contract_verified=True,cpu_forward_probes=probes,
                gpu_precision_verified=False,expert_skill_verified=False,training_ready=False)
    (WORK/'model_contract_audit.json').write_text(json.dumps(report,indent=2))
    backup=Path('/content/drive/MyDrive/cr_rl_ppo/preparation/full_quality_bc')
    shutil.copyfile(WORK/'model_contract_audit.json',backup/'model_contract_audit.json')
    print(json.dumps(report),flush=True)


if '--worker' in sys.argv:
    worker()
else:
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python'))
    with (WORK/'contract_audit.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/colab_quality_bc_contract.py'),'--worker'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'contract_audit_pid':process.pid,'log':str(WORK/'contract_audit.log')}))
