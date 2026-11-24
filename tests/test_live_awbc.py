import copy

import numpy as np
import pytest
import torch

from cr_rl.training.live_awbc import (crossfit, fixed_weights, freeze_actor_readouts,
                                      mc_returns, split_episodes, train_actor_epoch,
                                      behavior_value_weights, weighted_actor_split)


def test_mc_variable_time_and_weak_clipped_weights():
    result=mc_returns([0,0,-1],[8,16,0],gamma=.9)
    np.testing.assert_allclose(result,[-.9**3,-.9**2,-1],rtol=1e-6)
    assert np.all(fixed_weights([-100,0,100],strength=0)==1)
    np.testing.assert_allclose(fixed_weights([-100,0,100]),[.625,1,2.5])
    with pytest.raises(ValueError):mc_returns([1],[float('nan')])


def test_crossfit_never_fits_oof_or_outer_holdout_games():
    groups=[i%2 for i in range(18)]
    split=split_episodes(groups)
    again=split_episodes(groups)
    assert split==again
    assert not set(split['train'])&set(split['holdout'])
    features=[np.array([[i/18],[i/18+.1]],dtype=np.float32) for i in range(18)]
    targets=[x[:,0]*.2 for x in features]
    head=torch.nn.Linear(1,1)
    predictions,report=crossfit(head,features,targets,split,epochs=2,patience=1,groups=groups)
    assert set(predictions)==set(split['train'])
    for fold in report['folds']:
        fit=set(fold['fit_ids']);val=set(fold['validation_ids']);test=set(fold['excluded_test_ids'])
        assert fit and val and test
        assert not (fit&val or fit&test or val&test)
        assert not (fit|val|test)&set(split['holdout'])
    with pytest.raises(ValueError,match='12 qualified'):split_episodes([1]*5)


def test_actor_small_awbc_changes_readouts_but_not_backbone():
    from test_state_ppo_v3 import make_state_models, make_synthetic_segment
    schema,adapter,actor,critic,behavior,_=make_state_models(seed=133)
    ep=make_synthetic_segment(schema,adapter,actor,length=8,seed=134,prefix_len=0)
    ep['actor_keep']=ep['legal_masks'].sum(-1)>1
    ep['actor_keep'][0]=False
    names=freeze_actor_readouts(actor)
    before={k:p.detach().clone() for k,p in actor.named_parameters()}
    optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=1e-3)
    result=train_actor_epoch(actor,adapter,[ep],[0],{0:np.ones(8,dtype=np.float32)},optimizer,chunk=3)
    assert result['optimizer_steps']==1
    assert np.isfinite(result['episode_mean_weighted_nll'])
    changed=[]
    for name,param in actor.named_parameters():
        if name not in names:torch.testing.assert_close(param,before[name],rtol=0,atol=0)
        elif not torch.equal(param,before[name]):changed.append(name)
    assert changed


def test_holdout_predictions_and_value_gate_do_not_use_holdout_returns():
    torch.set_num_threads(2)
    features=[np.array([[i/12],[i/12+.1]],np.float32) for i in range(11)]
    targets=[x[:,0]*.2 for x in features]
    split={'train':list(range(9)),'holdout':[9,10],
           'fold_ids':{i:i%3 for i in range(9)}}
    head=torch.nn.Linear(1,1)
    predictions,report=crossfit(head,features,targets,split,epochs=1,prediction_ids=[9,10])
    altered=[x.copy() for x in targets]
    altered[9][:]=999;altered[10][:]=-999
    again,other=crossfit(head,features,altered,split,epochs=1,prediction_ids=[9,10])
    assert set(predictions)==set(range(11))
    for idx in range(11):np.testing.assert_allclose(predictions[idx],again[idx],rtol=0,atol=0)
    assert report['oof_mse']==other['oof_mse']
    assert report['value_gate_passed']==other['value_gate_passed']
    for fold in report['folds']:
        assert not {9,10}&set(fold['fit_ids']+fold['validation_ids']+fold['excluded_test_ids'])
    with pytest.raises(ValueError,match='disjoint outer holdout'):
        crossfit(head,features,targets,split,epochs=1,prediction_ids=[0])


@pytest.mark.parametrize('reason',['fewer_than_9_train_games','single_terminal_outcome','oof_value_gate_failed'])
def test_unreliable_cohorts_have_zero_actor_weights(monkeypatch,reason):
    from cr_rl.training import live_awbc as module
    count=8 if reason=='fewer_than_9_train_games' else 9
    features=[np.zeros((2,1),np.float32) for _ in range(count+2)]
    targets=[np.full(2,1. if reason=='single_terminal_outcome' else (-1.)**i,np.float32)
             for i in range(count+2)]
    split={'train':list(range(count)),'holdout':[count,count+1]}
    def fail_gate(initial,features,targets,split,**kwargs):
        return {i:np.zeros_like(targets[i]) for i in range(count+2)}, {
            'value_gate_passed':False,'folds':[],'oof_mse':2.,'constant_oof_mse':1.}
    monkeypatch.setattr(module,'crossfit',fail_gate)
    weights,report=behavior_value_weights(torch.nn.Linear(1,1),features,targets,split,
                                         ['policy']*(count+2),['deck']*(count+2))
    assert report['cohorts']['policy']['reason']==reason
    assert report['unit_weight_train_games']==0 and report['weighted_train_games']==0
    assert all(not value.any() for value in weights.values())
    episodes=[{'actor_keep':np.ones(2,bool)} for _ in targets]
    assert weighted_actor_split(episodes,split,weights)['reason']=='no_reliable_training_cohorts'


def test_zero_weight_games_do_not_step_actor_optimizer(monkeypatch):
    from types import SimpleNamespace
    actor=torch.nn.Linear(1,1)
    before=copy.deepcopy(actor.state_dict())
    optimizer=torch.optim.AdamW(actor.parameters(),lr=.1)
    monkeypatch.setattr(optimizer,'step',lambda:pytest.fail('Unreliable data stepped Actor'))
    adapter=SimpleNamespace(schema=SimpleNamespace(header={'hand_card':(0,1,2,3)}))
    result=train_actor_epoch(actor,adapter,[{'actor_keep':np.ones(3,bool)}],[0],
                             {0:np.zeros(3,np.float32)},optimizer)
    assert result['optimizer_steps']==0 and result['optimized_indices']==[]
    for key,value in actor.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_mixed_batch_only_exposes_supported_cohort_to_actor(monkeypatch):
    from cr_rl.training import live_awbc as module
    features=[np.zeros((2,1),np.float32) for _ in range(15)]
    targets=[np.full(2,(-1.)**i,np.float32) for i in range(15)]
    split={'train':list(range(12)),'holdout':[12,13,14]}
    behaviors=['supported']*9+['small']*3+['supported','supported','small']
    def trusted_fit(initial,features,targets,local,**kwargs):
        assert local['train']==list(range(9)) and kwargs['prediction_ids']==[12,13]
        ids=local['train']+kwargs['prediction_ids']
        return {i:np.zeros_like(targets[i]) for i in ids},{'value_gate_passed':True,'folds':[]}
    monkeypatch.setattr(module,'crossfit',trusted_fit)
    weights,report=behavior_value_weights(torch.nn.Linear(1,1),features,targets,split,behaviors,['deck']*15)
    episodes=[{'actor_keep':np.ones(2,bool)} for _ in targets]
    active=weighted_actor_split(episodes,split,weights)
    assert active=={'train':list(range(9)),'holdout':[12,13],'reason':None}
    assert report['weighted_train_games']==9 and report['skipped_train_games']==3
    assert all(not weights[i].any() for i in [9,10,11,14])
