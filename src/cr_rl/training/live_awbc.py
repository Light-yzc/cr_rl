"""Conservative advantage-weighted BC with episode-disjoint value cross-fitting.

The production Actor/independent Critic architecture is unchanged. Only the
Critic head and Actor card/WAIT/ability readouts are fitted in this first stage.
"""
from contextlib import nullcontext
import copy
import math

import numpy as np
import torch


def mc_returns(rewards, step_ticks, gamma=.9995, reference_ticks=8):
    rewards=np.asarray(rewards,dtype=np.float64)
    ticks=np.asarray(step_ticks,dtype=np.float64)
    if rewards.ndim!=1 or ticks.shape!=rewards.shape or not np.isfinite(rewards).all() or not np.isfinite(ticks).all() or (ticks<0).any():
        raise ValueError('Invalid rewards or durations')
    out=np.zeros_like(rewards); future=0.
    for t in range(len(rewards)-1,-1,-1):
        future=rewards[t]+gamma**(ticks[t]/reference_ticks)*future
        out[t]=future
    return out.astype(np.float32)


def split_episodes(groups, seed=20261117, validation_fraction=.2, folds=3):
    """Split entire episodes by known own-deck/behavior stratum, never frames."""
    if len(groups)<12 or folds<2 or not 0<validation_fraction<.5:
        raise ValueError('Need at least 12 qualified games, >=2 folds and a valid holdout fraction')
    rng=np.random.default_rng(seed); strata={}
    for i,key in enumerate(groups): strata.setdefault(str(key),[]).append(i)
    train=[]; holdout=[]
    for indices in strata.values():
        rng.shuffle(indices)
        n=max(1,round(len(indices)*validation_fraction)) if len(indices)>1 else 0
        holdout.extend(indices[:n]);train.extend(indices[n:])
    rng.shuffle(train)
    if len(train)<folds*2 or not holdout:
        raise ValueError('Not enough episodes for disjoint fitting and final holdout')
    fold_ids={idx:i%folds for i,idx in enumerate(train)}
    return {'train':train,'holdout':holdout,'fold_ids':fold_ids,'seed':seed}


def amp(device,precision):
    if not str(device).startswith('cuda'):
        return nullcontext()
    if precision=='fp16':
        return torch.autocast('cuda',dtype=torch.float16)
    if precision=='bf16':
        return torch.autocast('cuda',dtype=torch.bfloat16)
    return nullcontext()


def predict_head(head, features, device, chunk=256):
    result=[]
    with torch.no_grad():
        for start in range(0,len(features),chunk):
            x=torch.as_tensor(np.array(features[start:start+chunk]),device=device,dtype=torch.float32)
            result.append(head(x).reshape(-1).float().cpu().numpy())
    return np.concatenate(result)


def fit_head(initial, features, targets, fit_ids, val_ids, *, device='cpu', epochs=50,
             lr=1e-4, patience=5, seed=1):
    if not fit_ids or not val_ids or set(fit_ids)&set(val_ids):
        raise ValueError('Value fit and early-stop games must be nonempty and disjoint')
    head=copy.deepcopy(initial).to(device=device,dtype=torch.float32)
    for p in head.parameters(): p.requires_grad_(True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed)
    def score():
        return float(np.mean([np.mean((predict_head(head,features[i],device)-targets[i])**2) for i in val_ids]))
    best=score(); best_epoch=0; saved=copy.deepcopy(head.state_dict()); stale=0
    for epoch in range(1,epochs+1):
        for idx in rng.permutation(fit_ids):
            # Normalize each episode equally, not each long match's frame count.
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(targets[idx]),256):
                x=torch.as_tensor(np.array(features[idx][start:start+256]),device=device,dtype=torch.float32)
                y=torch.as_tensor(targets[idx][start:start+256],device=device)
                loss=(head(x).reshape(-1)-y).square().sum()/len(targets[idx])
                loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
        value=score()
        if not math.isfinite(value):raise ValueError('Nonfinite value loss')
        if value<best-1e-5:
            best=value;best_epoch=epoch;saved=copy.deepcopy(head.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    head.load_state_dict(saved)
    return head,{'fit_ids':list(map(int,fit_ids)),'validation_ids':list(map(int,val_ids)),
                 'best_epoch':best_epoch,'validation_mse':best}


def crossfit(initial, features, targets, split, *, device='cpu',epochs=50,lr=1e-4,patience=5,groups=None,
             prediction_ids=()):
    train=split['train']; ids=split['fold_ids']; predictions={}; baselines={}; reports=[]
    prediction_ids=list(prediction_ids)
    if set(prediction_ids)&set(train) or not set(prediction_ids)<=set(split['holdout']):
        raise ValueError('Ensemble predictions require disjoint outer holdout games')
    ensemble={i:[] for i in prediction_ids}
    for fold in sorted(set(ids.values())):
        test=[i for i in train if ids[i]==fold]
        pool=[i for i in train if ids[i]!=fold]
        cut=max(1,round(len(pool)*.2)); val=pool[:cut]; fit=pool[cut:]
        head,report=fit_head(initial,features,targets,fit,val,device=device,
                             epochs=epochs,lr=lr,patience=patience,seed=fold)
        # A constant trained only on this fold's fit episodes is an honest baseline.
        constant=float(np.mean([np.mean(targets[i]) for i in fit]))
        for i in test:
            predictions[i]=predict_head(head,features[i],device)
            matching=[j for j in fit if groups is not None and groups[j]==groups[i]]
            baselines[i]=float(np.mean([np.mean(targets[j]) for j in matching])) if matching else constant
        # Outer holdout never participates in fitting, early stopping or the OOF gate.
        for i in prediction_ids:
            ensemble[i].append(predict_head(head,features[i],device))
        reports.append(dict(report,excluded_test_ids=test))
        del head
    mse=float(np.mean([np.mean((predictions[i]-targets[i])**2) for i in train]))
    baseline=float(np.mean([np.mean((baselines[i]-targets[i])**2) for i in train]))
    for i,values in ensemble.items():
        predictions[i]=np.mean(values,axis=0).astype(np.float32)
    return predictions,{'folds':reports,'oof_mse':mse,'constant_oof_mse':baseline,
                        'value_gate_passed':mse < baseline*.98,
                        'ensemble_prediction_ids':prediction_ids}


def fixed_weights(advantages, *, beta=.5, strength=.5, minimum=.25,maximum=4.):
    if not 0<=strength<=1 or beta<=0 or not 0<minimum<=1<=maximum:
        raise ValueError('Invalid advantage weight parameters')
    advantage=np.asarray(advantages,dtype=np.float64)
    if not np.isfinite(advantage).all():raise ValueError('Nonfinite advantage')
    raw=np.exp(np.clip(advantage/beta,np.log(minimum),np.log(maximum)))
    return (1+strength*(raw-1)).astype(np.float32)


def behavior_value_weights(initial, features, targets, split, behaviors, groups, *,
                           device='cpu', epochs=20, lr=1e-4, beta=.5, strength=.5):
    """Fit shadow V_mu heads per behavior cohort, never the live PPO V_pi.

    Current Critic features are frozen. MC targets describe the recorded
    behavior, so different collection policies must not share a fitted head.
    Unreliable cohorts receive zero weights and no Actor update. Holdout weights
    use an ensemble of the training-only fold heads and stay fixed within a batch.
    """
    weights={i:np.zeros_like(targets[i],dtype=np.float32)
             for i in [*split['train'],*split['holdout']]}
    cohorts={}; all_folds=[]
    for behavior in sorted(set(behaviors[i] for i in split['train'])):
        ids=[i for i in split['train'] if behaviors[i]==behavior]
        holdout=[i for i in split['holdout'] if behaviors[i]==behavior]
        outcomes,counts=np.unique([float(targets[i][-1]) for i in ids],return_counts=True)
        support={'train_games':len(ids),'holdout_games':len(holdout),
                 'terminal_outcomes':{str(outcome):int(count) for outcome,count in zip(outcomes,counts)}}
        if len(ids)<9:
            cohorts[behavior]={'value_gate_passed':False,'reason':'fewer_than_9_train_games',
                               **support}
            continue
        if len(outcomes)<2:
            cohorts[behavior]={'value_gate_passed':False,'reason':'single_terminal_outcome',**support}
            continue
        local={'train':ids,'holdout':split['holdout'],
               'fold_ids':{idx:j%3 for j,idx in enumerate(ids)}}
        predictions,report=crossfit(initial,features,targets,local,device=device,
                                    epochs=epochs,lr=lr,groups=groups,prediction_ids=holdout)
        cohorts[behavior]=report
        report.update(support)
        all_folds.extend(report['folds'])
        if report['value_gate_passed'] and holdout:
            for i in [*ids,*holdout]:
                weights[i]=fixed_weights(targets[i]-predictions[i],beta=beta,strength=strength)
        else:
            report['reason']='no_same_behavior_holdout' if report['value_gate_passed'] else 'oof_value_gate_failed'
    train_ids=[i for i in split['train'] if np.any(weights[i]>0)]
    holdout_ids=[i for i in split['holdout'] if np.any(weights[i]>0)]
    return weights,{'cohorts':cohorts,'folds':all_folds,
                    'weighted_train_games':len(train_ids),'weighted_holdout_games':len(holdout_ids),
                    'unit_weight_train_games':0,'skipped_train_games':len(split['train'])-len(train_ids),
                    'eligible_actor_train_indices':train_ids,'eligible_actor_holdout_indices':holdout_ids,
                    'value_gate_passed':any(r['value_gate_passed'] for r in cohorts.values()),
                    'value_semantics':'MC V_mu per recorded behavior; not current-policy V_pi',
                    'ppo_critic_updated':False}


def weighted_actor_split(episodes, split, weights, *, min_holdout=2):
    """Choose only supported cohorts without discarding any recurrent prefix frames."""
    def usable(i):
        w=np.asarray(weights[i])
        if w.shape!=episodes[i]['actor_keep'].shape or not np.isfinite(w).all() or (w<0).any():
            raise ValueError('Invalid fixed Actor weights')
        return bool(np.any(episodes[i]['actor_keep'] & (w>0)))
    train=[i for i in split['train'] if usable(i)]
    holdout=[i for i in split['holdout'] if usable(i)]
    reason=('no_reliable_training_cohorts' if not train else
            'fewer_than_2_reliable_holdout_games' if len(holdout)<min_holdout else None)
    return {'train':train,'holdout':holdout,'reason':reason}


def actor_label_summary(episodes, indices, weights):
    """Diagnostic frame ESS is not a count of independent games."""
    counts={'wait':0,'deploy':0,'ability':0};values=[];outcomes={}
    for i in indices:
        ep=episodes[i];keep=ep['actor_keep'] & (weights[i]>0)
        actions=ep['actions'][keep]
        counts['wait']+=int((actions==0).sum())
        counts['deploy']+=int(((actions>0)&(actions<2305)).sum())
        counts['ability']+=int((actions>=2305).sum())
        values.append(weights[i][keep])
        outcome=str(float(ep['rewards'][-1]));outcomes[outcome]=outcomes.get(outcome,0)+1
    w=np.concatenate(values).astype(np.float64) if values else np.array([],dtype=np.float64)
    return {'games':len(indices),'terminal_outcomes':outcomes,'choice_labels':counts,
            'weight_min':float(w.min()) if len(w) else None,
            'weight_max':float(w.max()) if len(w) else None,
            'frame_weight_ess':float(w.sum()**2/np.square(w).sum()) if len(w) else 0.,
            'ess_unit':'correlated_actor_frames_not_independent_games'}


def reference_pass(actor, adapter, episodes, indices, *, device='cpu', precision='fp32', capture=False,
                   weights=None):
    """Snapshot/evaluate the current pre-AWBC Actor, independently of logged mu."""
    actor.eval(); nll=[];weighted_nll=[];kls=[]
    hands=np.array(adapter.schema.header['hand_card'])
    with torch.no_grad():
        for i in indices:
            ep=episodes[i];state=None;total=weighted_total=kl=0.;count=0;refs=[]
            for start in range(0,len(ep['actions']),32):
                end=min(start+32,len(ep['actions']));raw=ep['raw_obs'][None,start:end]
                h,e,m=adapter(raw)
                with amp(device,precision):
                    lp,_,state=actor.forward_state(torch.as_tensor(h,device=device),
                        torch.as_tensor(e,device=device),torch.as_tensor(m,device=device),
                        torch.as_tensor(ep['legal_masks'][None,start:end],device=device),
                        hand_card_ids=torch.as_tensor(raw[...,hands],device=device),state=state,compute_value=False)
                state=actor.detach_state(state);new=lp[0].float()
                if capture:
                    refs.append(new.exp().cpu().numpy())
                keep=torch.as_tensor(ep['actor_keep'][start:end],device=device)
                if not keep.any():continue
                actions=torch.as_tensor(ep['actions'][start:end],device=device)
                action_nll=-new.gather(-1,actions[:,None]).squeeze(-1)
                total+=float(action_nll[keep].sum())
                if weights is not None:
                    w=torch.as_tensor(weights[i][start:end],device=device)
                    weighted_total+=float((action_nll*w)[keep].sum())
                if not capture:
                    old=torch.as_tensor(ep['reference_probs'][start:end],device=device).float()
                    terms=torch.where(old>0,old*(old.clamp_min(1e-30).log()-new),0.)
                    kl+=float(terms.sum(-1)[keep].sum())
                count+=int(keep.sum())
            if capture:ep['reference_probs']=np.concatenate(refs)
            if count:
                nll.append(total/count);kls.append(kl/count)
                if weights is not None:weighted_nll.append(weighted_total/count)
    if not nll:raise ValueError('No validation choice actions')
    result={'unweighted_nll':float(np.mean(nll)), 'reference_forward_kl':float(np.mean(kls))}
    if weights is not None:result['weighted_nll']=float(np.mean(weighted_nll))
    return result


def freeze_actor_readouts(actor):
    prefixes=('wait_actor.','slot_actor.','ability_actor.','wait_hand_gate.')
    names=[]
    for name,p in actor.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
        if p.requires_grad:names.append(name)
    if not names:raise ValueError('No compatible Actor readout parameters')
    return names


def train_actor_epoch(actor, adapter, episodes, indices, weights, optimizer, *,
                      device='cpu',precision='fp32',chunk=32,seed=1,scaler=None):
    actor.eval()  # No dropout; gradients still flow through unfrozen readouts.
    rng=np.random.default_rng(seed); steps=0; loss_sum=0.;games=0;optimized=[]
    hands=np.array(adapter.schema.header['hand_card'],dtype=np.int64)
    for idx in rng.permutation(indices):
        episode=episodes[idx]; keep=episode['actor_keep'] & (weights[idx]>0); denom=int(keep.sum())
        if not denom:continue
        optimizer.zero_grad(set_to_none=True); state=None; total=0.
        for start in range(0,len(keep),chunk):
            end=min(len(keep),start+chunk);raw=episode['raw_obs'][None,start:end]
            h,e,m=adapter(raw)
            with amp(device,precision):
                lp,_,state=actor.forward_state(torch.as_tensor(h,device=device),
                    torch.as_tensor(e,device=device),torch.as_tensor(m,device=device),
                    torch.as_tensor(episode['legal_masks'][None,start:end],device=device),
                    state=state,hand_card_ids=torch.as_tensor(raw[...,hands],device=device),compute_value=False)
            selected=torch.as_tensor(keep[start:end],device=device)
            if selected.any():
                actions=torch.as_tensor(episode['actions'][start:end],device=device)
                w=torch.as_tensor(weights[idx][start:end],device=device)
                loss=-(lp[0].float().gather(-1,actions[:,None]).squeeze(-1)*w)[selected].sum()/denom
                if not torch.isfinite(loss):raise ValueError('Nonfinite AWBC loss')
                (scaler.scale(loss) if scaler is not None else loss).backward()
                total+=float(loss.detach())
            state=actor.detach_state(state)
        if scaler is not None:scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.)
        if not torch.isfinite(norm):
            if scaler is not None:scaler.update()
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is not None:scaler.step(optimizer);scaler.update()
        else:optimizer.step()
        if hasattr(actor,'mark_weights_updated'):actor.mark_weights_updated()
        steps+=1;games+=1;loss_sum+=total;optimized.append(int(idx))
    return {'optimizer_steps':steps,'episode_mean_weighted_nll':loss_sum/max(1,games),
            'optimized_indices':optimized}
