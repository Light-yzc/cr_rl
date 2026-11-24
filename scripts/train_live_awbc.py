"""Fit cross-validated real-game value heads, then conservative AWBC readouts."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from cr_rl.client.rl_recording import actor_fingerprint, atomic_json, load_episode, sha256_file
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint, state_value_model_from_checkpoint
from cr_rl.training.live_awbc import (amp, freeze_actor_readouts,
                                     mc_returns, split_episodes, train_actor_epoch,
                                     behavior_value_weights, reference_pass,
                                     weighted_actor_split, actor_label_summary)
from run_nulls_dryrun import schema_from_model_config


def save_checkpoint(path,data):
    temporary=path.with_suffix('.pt.partial')
    with temporary.open('wb') as stream:
        torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def evaluate(actor,adapter,episodes,indices,device,precision,weights=None):
    return reference_pass(actor,adapter,episodes,indices,device=device,precision=precision,weights=weights)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--value-epochs',type=int,default=50)
    p.add_argument('--actor-lr',type=float,default=1e-6)
    p.add_argument('--value-lr',type=float,default=1e-4)
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--strength',type=float,default=.5)
    p.add_argument('--max-kl',type=float,default=.02)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261117)
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--wandb-project',default='crrl-live-awbc')
    p.add_argument('--max-games',type=int,default=64,help='Bound resident offline games; PPO playbuffer uses smaller batches')
    a=p.parse_args()
    if (min(a.epochs,a.value_epochs,a.patience)<1 or a.max_games<12 or min(a.actor_lr,a.value_lr,a.beta,a.max_kl)<=0
            or not 0<=a.strength<=1):p.error('Invalid training parameters')
    a.out.mkdir(parents=True,exist_ok=True)
    controller_lock=None
    if sys.platform=='win32':
        import msvcrt
        lock_path=Path(__file__).resolve().parents[1]/'artifacts/nulls_live/policy-control.lock'
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        controller_lock=lock_path.open('a+b')
        if lock_path.stat().st_size==0:controller_lock.write(b'0');controller_lock.flush()
        controller_lock.seek(0)
        try:msvcrt.locking(controller_lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:p.error('Live collection is active; run local AWBC after stopping collection')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    precision='fp16' if a.device.startswith('cuda') else 'fp32'
    # Mmap avoids loading optimizer tensors into RAM; this entrypoint starts a
    # new offline branch and never reuses PPO optimizer moments.
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if 'critic' not in ckpt or 'critic_config' not in ckpt:
        p.error('Use the original full PPO checkpoint, not an Actor-only export')
    fingerprint=actor_fingerprint(ckpt)
    episodes=[];paths=[];excluded=[];seen_matches=set()
    for path in sorted(a.sessions.rglob('manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        if len(episodes)>=a.max_games:break
        run=path.parent
        try:
            ep=load_episode(run,purpose='awbc')
            terminal=json.loads((run/'terminal.json').read_text(encoding='utf8'))
            match_key=(run.name,terminal['match_id'])
            if match_key in seen_matches:raise ValueError('Duplicate recorded match')
            if not ep['actor_keep'].any():raise ValueError('No usable policy choice actions')
            episodes.append(ep);paths.append(run);seen_matches.add(match_key)
        except (ValueError,OSError,KeyError,EOFError) as error:
            excluded.append({'run':str(run),'reason':str(error)})
    atomic_json(a.out/'excluded_sessions.json',excluded)
    groups=[(ep['behavior_version'],tuple(sorted(map(int,ep['raw_obs'][0,20:28])))) for ep in episodes]
    split=split_episodes(groups,seed=a.seed)
    dataset={'runs':[{'run':str(run.resolve()),'manifest':sha256_file(run/'manifest.json'),
                     'result':sha256_file(run/'result.json')} for run in paths],
             'actor_fingerprint':fingerprint,'split':split,
             'initial_checkpoint_sha256':sha256_file(a.checkpoint),
             'settings':{k:v for k,v in vars(a).items() if k not in ('epochs','out','wandb','wandb_project')}}
    # JSON normalization handles Path and integer fold keys deterministically.
    dataset=json.loads(json.dumps(dataset,default=str))
    provenance=a.out/'dataset.json'
    if provenance.exists() and json.loads(provenance.read_text())!=dataset:
        raise ValueError('Dataset or settings changed: use a fresh output directory')
    atomic_json(provenance,dataset)
    wandb=None
    if a.wandb:
        import wandb as wb
        wandb=wb
        id_path=a.out/'wandb_id.json'
        identity=json.loads(id_path.read_text())['id'] if id_path.exists() else wb.util.generate_id()
        atomic_json(id_path,{'id':identity})
        wb.init(project=a.wandb_project,id=identity,resume='allow',config=dataset['settings'])
    def emit(phase,**data):
        row={'phase':phase,**data}
        with (a.out/'metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if wandb:wandb.log(row)
    emit('data_ready',qualified_games=len(episodes),excluded_games=len(excluded),
         train_games=len(split['train']),holdout_games=len(split['holdout']))
    critic=state_value_model_from_checkpoint(ckpt,device=a.device).eval()
    for param in critic.parameters():param.requires_grad_(False)
    schema=schema_from_model_config(critic.cfg)
    adapter=CrForgeObservationAdapter(schema,vocabulary_size=critic.cfg.vocab_size,entity_bucket=16)
    cache=a.out/'value_features';cache.mkdir(exist_ok=True)
    features=[];targets=[]
    for idx,ep in enumerate(tqdm(episodes,desc='Frozen Critic features',unit='game')):
        path=cache/f'{idx:05d}.npy';partial=path.with_suffix('.partial')
        if not path.exists():
            arr=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,
                                         shape=(len(ep['actions']),critic.cfg.temporal_dim))
            state=None
            with torch.no_grad():
                for start in range(0,len(ep['actions']),32):
                    end=min(start+32,len(ep['actions']));h,e,m=adapter(ep['raw_obs'][None,start:end])
                    with amp(a.device,precision):
                        result=critic._encode_state_sequence(torch.as_tensor(h,device=a.device),
                            torch.as_tensor(e,device=a.device),torch.as_tensor(m,device=a.device),state=state)
                    context=result[0];state=critic.detach_state(result[4])
                    arr[start:end]=context.float().cpu().numpy()
            arr.flush();del arr;os.replace(partial,path)
        features.append(np.load(path,mmap_mode='r'))
        targets.append(mc_returns(ep['rewards'],ep['step_ticks']))
    initial=critic.critic.cpu()
    del critic
    if a.device.startswith('cuda'):torch.cuda.empty_cache()
    weights,report=behavior_value_weights(initial,features,targets,split,
        [ep['behavior_version'] for ep in episodes],groups,device=a.device,
        epochs=a.value_epochs,lr=a.value_lr,beta=a.beta,strength=a.strength)
    atomic_json(a.out/'value_crossfit.json',report)
    active=weighted_actor_split(episodes,split,weights)
    actor_report={'actor_attempted':False,'reason':active['reason'],
                  'eligible_actor_train_indices':active['train'],'eligible_actor_holdout_indices':active['holdout'],
                  'actor_train_indices':[],'actor_holdout_indices':[],
                  'train_summary':actor_label_summary(episodes,active['train'],weights),
                  'holdout_summary':actor_label_summary(episodes,active['holdout'],weights),
                  'validation_objective':'fixed_holdout_advantage_weighted_nll'}
    atomic_json(a.out/'actor_update_report.json',actor_report)
    np.savez_compressed(a.out/'fixed_weights.npz',**{str(i):v for i,v in weights.items()})
    emit('value_crossfit',**{k:v for k,v in report.items() if k!='folds'},effective_strength=a.strength,
         mode='skip_actor' if active['reason'] else 'awbc')
    # There is no single deployable V_pi head for a mixed-behavior replay batch.
    save_checkpoint(a.out/'value_head.pt',{'head':None,'report':report,
        'source_actor_fingerprint':fingerprint,'diagnostic_only':True,
        'note':'Per-behavior cross-fit heads produce fixed weights; PPO Critic is unchanged'})
    del features,initial
    if a.device.startswith('cuda'):torch.cuda.empty_cache()
    if active['reason']:
        emit('finished',**actor_report,note='No reliable Actor update; gather more compatible games')
        if wandb:wandb.finish()
        return
    actor=state_model_from_checkpoint(ckpt,device=a.device)
    actor.ppo_actor_fp32=bool(ckpt.get('training_args',{}).get('ppo_actor_fp32',False))
    metadata={k:ckpt.get(k) for k in ('config','temporal_config','learned_placement_definition',
        'model_architecture_version','decision_ticks','updates','backend_commit','data_version','content_sha','level_profile')}
    del ckpt
    trainable=freeze_actor_readouts(actor)
    atomic_json(a.out/'trainable_actor_parameters.json',trainable)
    optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=a.actor_lr,weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',enabled=a.device.startswith('cuda'),init_scale=128.)
    latest=a.out/'latest.pt';best=a.out/'best.pt';epoch0=0;stale=0
    baseline=reference_pass(actor,adapter,episodes,active['holdout'],device=a.device,
                            precision=precision,capture=True,weights=weights)
    if not np.isfinite(list(baseline.values())).all():
        raise ValueError('Initial Actor/reference distribution mismatch; investigate collection or precision before training')
    best_nll=baseline['weighted_nll']
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
        if saved.get('validation_objective')!='fixed_holdout_advantage_weighted_nll':
            raise ValueError('Legacy unweighted-NLL run: use a fresh output directory')
        actor.load_state_dict(saved['model']);optimizer.load_state_dict(saved['awbc_optimizer'])
        scaler.load_state_dict(saved['awbc_scaler']);epoch0=saved['awbc_epoch']
        best_nll=saved['best_validation_weighted_nll'];stale=saved['stale_epochs'];del saved
        if stale>=a.patience:
            emit('finished',best=str(best),note='Restored an early-stopped run; use a new experiment for changed settings')
            if wandb:wandb.finish()
            return
        if evaluate(actor,adapter,episodes,active['holdout'],a.device,precision,weights)['reference_forward_kl']>a.max_kl:
            emit('finished',best=str(best),note='Latest crossed KL limit; no further reuse permitted in this run')
            if wandb:wandb.finish()
            return
    else:
        save_checkpoint(best,{**metadata,'model':{k:v.cpu() if torch.is_tensor(v) else v for k,v in actor.state_dict().items()},
                              'training_method':'live_awbc_readouts','awbc_epoch':0,'validation':baseline,
                              'validation_objective':'fixed_holdout_advantage_weighted_nll'})
    optimized=set()
    for epoch in range(epoch0+1,a.epochs+1):
        progress=tqdm(total=1,desc=f'AWBC epoch {epoch}/{a.epochs}')
        metrics=train_actor_epoch(actor,adapter,episodes,active['train'],weights,optimizer,
            device=a.device,precision=precision,seed=a.seed+epoch,scaler=scaler)
        optimized.update(metrics['optimized_indices'])
        validation=evaluate(actor,adapter,episodes,active['holdout'],a.device,precision,weights)
        progress.update();progress.close()
        if not np.isfinite(list(validation.values())).all():raise ValueError('Nonfinite validation')
        acceptable=validation['reference_forward_kl']<=a.max_kl
        improved=acceptable and metrics['optimizer_steps'] and validation['weighted_nll']<best_nll-1e-4
        if improved:best_nll=validation['weighted_nll'];stale=0
        else:stale+=1
        state={**metadata,'model':{k:v.cpu() if torch.is_tensor(v) else v for k,v in actor.state_dict().items()},
            'training_method':'live_awbc_readouts','awbc_epoch':epoch,'validation':validation,
            'awbc_optimizer':optimizer.state_dict(),'awbc_scaler':scaler.state_dict(),
            'best_validation_weighted_nll':best_nll,'stale_epochs':stale,
            'validation_objective':'fixed_holdout_advantage_weighted_nll'}
        save_checkpoint(latest,state)
        if improved:save_checkpoint(best,{k:v for k,v in state.items() if k not in ('awbc_optimizer','awbc_scaler')})
        emit('actor_epoch',epoch=epoch,**metrics,**validation,accepted=acceptable,best_improved=improved)
        if not acceptable or stale>=a.patience:break
    actor_report.update(actor_attempted=bool(optimized),actor_train_indices=sorted(optimized),
                        actor_holdout_indices=active['holdout'] if optimized else [],
                        reason='actor_epochs_completed' if optimized else 'no_actor_optimizer_steps')
    atomic_json(a.out/'actor_update_report.json',actor_report)
    emit('finished',best=str(best),**actor_report,note='Offline diagnostics only; new-game strength has not been evaluated')
    if wandb:wandb.finish()


if __name__=='__main__':
    main()
