"""Bounded disk-backed real-game replay, consumed at PPO update boundaries.

Checkpoint state is authoritative for consumption. The shadow MC value heads
estimate V_mu within collection-policy cohorts; live PPO Critic is read-only.
"""
import copy
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from cr_rl.client.rl_recording import load_episode, sha256_file
from cr_rl.training.live_awbc import (amp, behavior_value_weights, freeze_actor_readouts,
    mc_returns, reference_pass, train_actor_epoch, weighted_actor_split, actor_label_summary)


def restore_options(args, previous, argv):
    """Keep checkpointed replay settings unless explicitly overridden on CLI."""
    for name in ('awbc_sessions','awbc_interval','awbc_min_new_games','awbc_batch_games',
                 'awbc_max_reuse','awbc_epochs','awbc_value_epochs','awbc_learning_rate','awbc_max_kl',
                 'expert_bc_dataset','expert_bc_experimental','expert_bc_interval',
                 'expert_bc_sequences','expert_bc_learning_rate','expert_bc_max_kl'):
        option='--'+name.replace('_','-')
        explicit=any(item==option or item.startswith(option+'=') for item in argv)
        if not explicit and previous.get(name) is not None:
            value=previous[name]
            setattr(args,name,Path(value) if name in ('awbc_sessions','expert_bc_dataset') else value)


def read_game(run):
    episode=load_episode(run,purpose='awbc')
    if not episode['actor_keep'].any():raise ValueError('No usable choice labels')
    return episode


def game_id(episode):
    digest=hashlib.sha256(episode['behavior_version'].encode())
    digest.update(episode['raw_obs'].tobytes());digest.update(episode['actions'].tobytes())
    return digest.hexdigest()


class PlayBuffer:
    def __init__(self, source, state=None, capacity=256, max_reuse=2):
        self.source=Path(source);self.capacity=capacity;self.max_reuse=max_reuse
        self.state=copy.deepcopy(state or {'entries':{},'last_update':0,'phases':0})

    def refresh(self):
        excluded=[];added=0
        entries=self.state['entries']
        for row in entries.values():row['active']=False
        # Metadata only stays resident. Decode one complete game at a time.
        candidates=sorted(self.source.rglob('result.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for result in candidates[:self.capacity]:
            run=result.parent
            if not (run/'manifest.json').is_file():continue
            signature=sha256_file(run/'manifest.json')+sha256_file(result)
            existing=next((row for row in entries.values() if row['signature']==signature),None)
            if existing is not None:
                existing.update(run=str(run.resolve()),active=True)
                continue
            try:
                ep=read_game(run);identity=game_id(ep)
                if identity in entries:
                    if entries[identity]['signature']!=signature:
                        raise ValueError('Existing game result/manifest changed; do not relabel consumed games')
                    continue
                entries[identity]=dict(run=str(run.resolve()),signature=signature,uses=0,
                    behavior=ep['behavior_version'],active=True,
                    partition='validation' if int(identity[:8],16)%5==0 else 'train')
                added+=1
                del ep
            except (ValueError,OSError,KeyError,EOFError) as error:
                excluded.append(dict(run=str(run),reason=str(error)))
        # Never forget consumed IDs: their tiny tombstones prevent re-import on restart.
        return dict(added_games=added,known_games=len(entries),excluded=excluded)

    def select(self, min_new=12, batch_games=24):
        """Pack usable behavior cohorts; do not strand each version below its fit floor."""
        entries=self.state['entries']
        train=[key for key,row in entries.items() if row.get('active') and row['partition']=='train' and row['uses']<self.max_reuse]
        fresh=[key for key in train if entries[key]['uses']==0]
        validation=[key for key,row in entries.items() if row.get('active') and row['partition']=='validation']
        if len(fresh)<min_new or len(train)<9 or len(validation)<2:return []
        previous=set(self.state.get('last_skipped_batch',{}).get('game_ids',[]))
        def train_rank(key):return (entries[key]['uses']>0,key in previous,entries[key]['uses'],key)
        def val_rank(key):return (key in previous,entries[key]['uses'],key)
        by_train={};by_val={}
        for key in train:by_train.setdefault(entries[key].get('behavior'),[]).append(key)
        for key in validation:by_val.setdefault(entries[key].get('behavior'),[]).append(key)
        cohorts=[key for key,rows in by_train.items() if len(rows)>=9 and by_val.get(key)]
        for rows in by_train.values():rows.sort(key=train_rank)
        for rows in by_val.values():rows.sort(key=val_rank)
        def cohort_rank(key):
            rows=by_train[key]
            return (-sum(entries[k]['uses']==0 and k not in previous for k in rows),
                    -sum(entries[k]['uses']==0 for k in rows),
                    -sum(k not in previous for k in by_val[key]),-len(rows),str(key))
        cohorts.sort(key=cohort_rank)
        def pack(keys):
            chosen=[row for key in keys for row in by_train[key][:9]]
            held=[by_val[key][0] for key in keys]
            all_train=sorted((row for key in keys for row in by_train[key]),key=train_rank)
            all_val=sorted((row for key in keys for row in by_val[key]),key=val_rank)
            for row in all_val:
                if len(held)>=2:break
                if row not in held:held.append(row)
            if len(held)<2 or len(chosen)+len(held)>batch_games:return None
            needed=max(0,min_new-sum(entries[k]['uses']==0 for k in chosen))
            extras=[k for k in all_train if k not in chosen and entries[k]['uses']==0]
            if len(extras)<needed or len(chosen)+len(held)+needed>batch_games:return None
            chosen.extend(extras[:needed])
            for row in all_val:
                if len(held)>=max(2,batch_games//5) or len(chosen)+len(held)>=batch_games:break
                if row not in held:held.append(row)
            for row in all_train:
                if len(chosen)+len(held)>=batch_games:break
                if row not in chosen:chosen.append(row)
            score=(sum(entries[k]['uses']==0 and k not in previous for k in chosen),
                   sum(entries[k]['uses']==0 for k in chosen),
                   sum(k not in previous for k in held),-len(keys),len(chosen))
            return chosen+held,score
        best=[];best_score=None
        def candidate_cohorts():
            # Exhaustive for the default 24-game budget: every valid pack has
            # at most two cohorts. A high-priority but low-fresh cohort must
            # not hide a feasible pair among the remaining versions.
            for size in (1,2):yield from combinations(cohorts,size)
            # Larger custom budgets retain bounded greedy candidates rather
            # than enumerating every subset of a capacity-sized buffer.
            if batch_games>=30:
                for seed in cohorts:
                    keys=[]
                    for key in [seed,*[other for other in cohorts if other!=seed]]:
                        keys.append(key)
                        if 9*len(keys)+max(2,len(keys))>batch_games:break
                        if len(keys)>2:yield tuple(keys)
        for keys in candidate_cohorts():
            if 9*len(keys)+max(2,len(keys))>batch_games:continue
            candidate=pack(keys)
            if candidate is not None and (best_score is None or candidate[1]>best_score):
                best,best_score=candidate
        return best

    def commit(self, selected, update):
        for key in selected:self.state['entries'][key]['uses']+=1
        self.state['last_update']=update;self.state['phases']+=1


def _extract_features(critic, adapter, episodes, work, device, precision, gamma):
    work.mkdir(parents=True,exist_ok=True);features=[];targets=[]
    mode=critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for idx,ep in enumerate(episodes):
                path=work/f'{idx:04d}.npy'
                array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,
                    shape=(len(ep['actions']),critic.cfg.temporal_dim))
                state=None
                for start in range(0,len(ep['actions']),32):
                    end=min(start+32,len(ep['actions']));h,e,m=adapter(ep['raw_obs'][None,start:end])
                    with amp(device,precision):
                        result=critic._encode_state_sequence(torch.as_tensor(h,device=device),
                            torch.as_tensor(e,device=device),torch.as_tensor(m,device=device),state=state)
                    state=critic.detach_state(result[4]);array[start:end]=result[0].float().cpu().numpy()
                array.flush();del array
                features.append(np.load(path,mmap_mode='r'))
                targets.append(mc_returns(ep['rewards'],ep['step_ticks'],gamma=gamma))
    finally:critic.train(mode)
    return features,targets


def run_phase(actor, critic, adapter, episodes, split, work, *, actor_device='cpu',
              critic_device='cpu', precision='fp32', epochs=2, value_epochs=20,
              lr=1e-6, max_kl=.01, seed=1, probes=None, gamma=.9995,
              fresh_indices=None, min_new_games=0):
    """Transactional readout-only AWBC. Restore best accepted weights or baseline."""
    features,targets=_extract_features(critic,adapter,episodes,Path(work),critic_device,precision,gamma)
    groups=[(ep['behavior_version'],tuple(sorted(map(int,ep['raw_obs'][0,20:28])))) for ep in episodes]
    initial=copy.deepcopy(critic.critic).cpu()
    try:
        weights,report=behavior_value_weights(initial,features,targets,split,
            [ep['behavior_version'] for ep in episodes],groups,device='cpu',epochs=value_epochs)
    finally:
        for array in features:array._mmap.close()
        del features,initial
        for path in Path(work).glob('*.npy'):path.unlink()
    active=weighted_actor_split(episodes,split,weights)
    fresh=set(split['train'] if fresh_indices is None else fresh_indices)
    if not fresh<=set(split['train']) or min_new_games<0:
        raise ValueError('Fresh Actor indices must be training games; minimum must be nonnegative')
    eligible_fresh=[i for i in active['train'] if i in fresh]
    if active['reason'] is None and len(eligible_fresh)<min_new_games:
        active['reason']='fewer_than_required_reliable_new_train_games'
    common=dict(value_report=report,ppo_critic_updated=False,ppo_probe_games=len(probes or []),
                validation_objective='fixed_holdout_advantage_weighted_nll',
                eligible_actor_train_indices=active['train'],eligible_actor_holdout_indices=active['holdout'],
                eligible_new_train_indices=eligible_fresh,min_new_games=min_new_games,
                train_summary=actor_label_summary(episodes,active['train'],weights),
                holdout_summary=actor_label_summary(episodes,active['holdout'],weights))
    if active['reason']:
        return dict(accepted=False,actor_attempted=False,memory_invalidated=False,reason=active['reason'],best_epoch=0,
                    changed_parameters=[],actor_train_indices=[],actor_holdout_indices=[],
                    baseline=None,epochs=[],**common)
    old_requires={name:p.requires_grad for name,p in actor.named_parameters()}
    mode=actor.training;names=freeze_actor_readouts(actor)
    baseline_weights={name:p.detach().cpu().clone() for name,p in actor.named_parameters() if name in names}
    best_weights=copy.deepcopy(baseline_weights);best_epoch=0
    def restore(weights):
        with torch.no_grad():
            for name,p in actor.named_parameters():
                if name in weights:p.copy_(weights[name].to(device=p.device,dtype=p.dtype))
        if hasattr(actor,'mark_weights_updated'):actor.mark_weights_updated()
    try:
        baseline=reference_pass(actor,adapter,episodes,active['holdout'],device=actor_device,
                                precision=precision,capture=True,weights=weights)
        if not np.isfinite(list(baseline.values())).all():raise ValueError('Nonfinite AWBC baseline')
        if probes:
            reference_pass(actor,adapter,probes,list(range(len(probes))),device=actor_device,
                           precision=precision,capture=True)
        best_nll=baseline['weighted_nll']
        optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=lr,weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda',enabled=str(actor_device).startswith('cuda'),init_scale=128.)
        history=[];optimized=set()
        for epoch in range(1,epochs+1):
            metrics=train_actor_epoch(actor,adapter,episodes,active['train'],weights,optimizer,
                device=actor_device,precision=precision,seed=seed+epoch,scaler=scaler)
            optimized.update(metrics['optimized_indices'])
            validation=reference_pass(actor,adapter,episodes,active['holdout'],device=actor_device,
                                      precision=precision,weights=weights)
            if probes:
                validation['ppo_probe_forward_kl']=reference_pass(actor,adapter,probes,
                    list(range(len(probes))),device=actor_device,precision=precision)['reference_forward_kl']
            if not np.isfinite(list(validation.values())).all():raise ValueError('Nonfinite AWBC validation')
            acceptable=max(validation['reference_forward_kl'],validation.get('ppo_probe_forward_kl',0.))<=max_kl
            history.append(dict(epoch=epoch,**metrics,**validation,acceptable=acceptable))
            if acceptable and metrics['optimizer_steps'] and validation['weighted_nll']<best_nll-1e-4:
                best_nll=validation['weighted_nll'];best_epoch=epoch
                best_weights={name:p.detach().cpu().clone() for name,p in actor.named_parameters() if name in names}
            else:break
        restore(best_weights)
        reason=('accepted_best_weighted_epoch' if best_epoch else 'no_actor_optimizer_steps' if not optimized
                else 'kl_limit' if history and not history[-1]['acceptable'] else 'no_weighted_holdout_improvement')
        return dict(accepted=best_epoch>0,actor_attempted=bool(optimized),memory_invalidated=True,reason=reason,
                    best_epoch=best_epoch,changed_parameters=names if best_epoch else [],
                    actor_train_indices=sorted(optimized),actor_holdout_indices=active['holdout'] if optimized else [],
                    baseline=baseline,epochs=history,**common)
    except BaseException:
        restore(baseline_weights)
        raise
    finally:
        for name,p in actor.named_parameters():p.requires_grad_(old_requires[name]);p.grad=None
        actor.train(mode)
