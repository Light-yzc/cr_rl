import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.training.awbc_playbuffer import PlayBuffer, run_phase
from cr_rl.training.live_awbc import behavior_value_weights, reference_pass


def test_resume_keeps_buffer_settings_and_honors_explicit_cli():
    from pathlib import Path
    from cr_rl.training.awbc_playbuffer import restore_options
    args=SimpleNamespace(awbc_sessions=None,awbc_interval=50)
    restore_options(args,dict(awbc_sessions='/sessions',awbc_interval=20),['--awbc-interval=50'])
    assert args.awbc_sessions==Path('/sessions') and args.awbc_interval==50


def test_buffer_deduplicates_and_restores_consumption_without_arrays(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    def read(path):
        idx=int(path.name.split('_')[-1])
        return dict(behavior_version='old-policy',raw_obs=np.array([[idx,1]],np.int32),
                    actions=np.array([0]),rewards=np.array([1.]))
    monkeypatch.setattr(module,'read_game',read)
    for idx in range(60):
        run=tmp_path/f'game_{idx}';run.mkdir()
        (run/'manifest.json').write_text(json.dumps({'i':idx}))
        (run/'result.json').write_text(json.dumps({'i':idx}))
    buffer=PlayBuffer(tmp_path)
    assert buffer.refresh()['added_games']==60
    assert buffer.refresh()['added_games']==0
    selected=buffer.select();assert len(selected)==24
    partitions={key:row['partition'] for key,row in buffer.state['entries'].items()}
    buffer.commit(selected,50)
    saved=json.loads(json.dumps(buffer.state))
    restored=PlayBuffer(tmp_path,saved)
    restored.refresh()
    assert all(restored.state['entries'][key]['uses']==1 for key in selected)
    assert {key:row['partition'] for key,row in restored.state['entries'].items()}==partitions
    assert restored.state['last_update']==50
    assert all('raw_obs' not in row for row in saved['entries'].values())


def test_shadow_values_never_mix_collection_policies_or_fit_holdout():
    features=[np.array([[i/20]],np.float32) for i in range(24)]
    targets=[np.array([(-1.)**i],np.float32) for i in range(24)]
    split={'train':list(range(18)),'holdout':list(range(18,24))}
    behaviors=['a']*9+['b']*9+['a']*6
    weights,report=behavior_value_weights(torch.nn.Linear(1,1),features,targets,split,
        behaviors,['deck']*24,epochs=1)
    assert set(weights)==set(split['train']+split['holdout']) and report['ppo_critic_updated'] is False
    for behavior,cohort in report['cohorts'].items():
        for fold in cohort['folds']:
            seen=set(fold['fit_ids']+fold['validation_ids']+fold['excluded_test_ids'])
            assert all(behaviors[i]==behavior for i in seen)
            assert not seen & set(split['holdout'])


def episodes_for(actor,schema,adapter):
    from test_state_ppo_v3 import make_synthetic_segment
    episodes=[]
    for i in range(12):
        ep=make_synthetic_segment(schema,adapter,actor,length=3,seed=i,prefix_len=0)
        ep.update(behavior_version='old-policy',step_ticks=np.full(3,8),
                  actor_keep=ep['legal_masks'].sum(-1)>1)
        ep['actions'][:]=0
        episodes.append(ep)
    return episodes,{'train':list(range(9)),'holdout':list(range(9,12))}


def trusted_phase_weights(initial,features,targets,split,behaviors,groups,**kwargs):
    # Statistical gating is tested separately; isolate the transactional Actor path.
    weights={i:np.ones_like(targets[i],dtype=np.float32) for i in split['train']+split['holdout']}
    return weights,{'value_gate_passed':True,'weighted_train_games':len(split['train']),
                    'weighted_holdout_games':len(split['holdout']),'unit_weight_train_games':0,
                    'skipped_train_games':0,'folds':[],'cohorts':{}}


def test_phase_changes_only_readouts_keeps_ppo_critic_and_restores_grad_flags(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    monkeypatch.setattr(module,'behavior_value_weights',trusted_phase_weights)
    from test_state_ppo_v3 import make_state_models
    torch.set_num_threads(2)
    schema,adapter,actor,critic,_,_=make_state_models(seed=10)
    episodes,split=episodes_for(actor,schema,adapter)
    actor_before={key:p.detach().clone() for key,p in actor.named_parameters()}
    critic_before=copy.deepcopy(critic.state_dict())
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1,lr=1e-3,max_kl=1.)
    assert result['accepted'] and result['actor_attempted'] and result['ppo_critic_updated'] is False
    assert result['actor_train_indices']==split['train']
    for name,param in actor.named_parameters():
        if name not in result['changed_parameters']:
            torch.testing.assert_close(param,actor_before[name],rtol=0,atol=0)
        assert param.requires_grad and param.grad is None
    for key,value in critic.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,critic_before[key],rtol=0,atol=0)
    assert not list(tmp_path.glob('*.npy'))


def test_kl_rejection_and_exception_restore_original_readouts(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    monkeypatch.setattr(module,'behavior_value_weights',trusted_phase_weights)
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,_,_=make_state_models(seed=12)
    episodes,split=episodes_for(actor,schema,adapter)
    before={name:p.detach().clone() for name,p in actor.named_parameters()}
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1,lr=.01,max_kl=0.)
    assert not result['accepted'] and result['actor_attempted'] and result['reason']=='kl_limit'
    for name,p in actor.named_parameters():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    def fail(model,*args,**kwargs):
        with torch.no_grad():next(p for p in model.parameters() if p.requires_grad).add_(10)
        raise RuntimeError('injected failure')
    monkeypatch.setattr(module,'train_actor_epoch',fail)
    with pytest.raises(RuntimeError,match='injected'):
        run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1)
    for name,p in actor.named_parameters():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        assert p.requires_grad


def test_prefix_mask_only_changes_prefix_logits_not_future_memory():
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,_,_,_=make_state_models(seed=14)
    episodes,_=episodes_for(actor,schema,adapter)
    ep=episodes[0]
    reference_pass(actor,adapter,episodes,[0],capture=True)
    original=ep['reference_probs'].copy()
    ep['legal_masks'][0]=False;ep['legal_masks'][0,0]=True
    reference_pass(actor,adapter,episodes,[0],capture=True)
    np.testing.assert_allclose(ep['reference_probs'][1:],original[1:],rtol=1e-5,atol=1e-6)


def test_ppo_hook_syncs_behavior_resets_only_changed_moments_and_commits(tmp_path,monkeypatch):
    from cr_rl.training import state_selfplay as module
    import cr_rl.training.awbc_playbuffer as awbc_module
    monkeypatch.setattr(awbc_module,'behavior_value_weights',trusted_phase_weights)
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,behavior,_=make_state_models(seed=20)
    episodes,_=episodes_for(actor,schema,adapter)
    trainer=module.PersistentStateSelfplay.__new__(module.PersistentStateSelfplay)
    trainer.actor=actor;trainer.critic=critic;trainer.behavior_model=behavior
    trainer.adapter=adapter;trainer.out=tmp_path;trainer.update=50;trainer.gamma=.9995
    trainer.device=trainer.critic_device=torch.device('cpu');trainer.precision='fp32';trainer.peer_actor=None
    trainer.args=SimpleNamespace(awbc_interval=50,awbc_min_new_games=9,awbc_batch_games=12,
        awbc_epochs=1,awbc_value_epochs=1,awbc_learning_rate=1e-3,awbc_max_kl=1.,seed=1)
    trainer.actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-3)
    for param in actor.parameters():trainer.actor_optimizer.state[param]={'sentinel':1}
    buffer=PlayBuffer(tmp_path);trainer.awbc_buffer=buffer
    for i in range(12):
        run=tmp_path/str(i);run.mkdir()
        (run/'manifest.json').write_text('{}');(run/'result.json').write_text('{}')
        from cr_rl.client.rl_recording import sha256_file
        buffer.state['entries'][str(i)]=dict(run=str(run),uses=0,active=True,
            signature=sha256_file(run/'manifest.json')+sha256_file(run/'result.json'),
            partition='train' if i<9 else 'validation')
    monkeypatch.setattr(buffer,'refresh',lambda:dict(added_games=0,known_games=12))
    monkeypatch.setattr(module,'read_game',lambda path:episodes[int(path.name)])
    events=[];trainer.emit=lambda **event:events.append(event)
    result=trainer._maybe_awbc_phase()
    assert result['ran'] and result['accepted'] and trainer.actor_weights_dirty
    assert buffer.state['last_update']==50 and buffer.state['phases']==1
    assert trainer._maybe_awbc_phase()=={'ran':False}
    changed=json.loads((tmp_path/'awbc_latest_report.json').read_text())['changed_parameters']
    for name,param in actor.named_parameters():
        assert (param not in trainer.actor_optimizer.state)==(name in changed)
    for key,value in actor.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,behavior.state_dict()[key],rtol=0,atol=0)
    assert events[-1]['phase']=='awbc_playbuffer'


def test_phase_skips_unreliable_cohorts_without_touching_actor(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,_,_=make_state_models(seed=31)
    episodes,split=episodes_for(actor,schema,adapter)
    for i,ep in enumerate(episodes):ep['behavior_version']=f'underpopulated-{i}'
    before={name:p.detach().clone() for name,p in actor.named_parameters()}
    monkeypatch.setattr(module,'train_actor_epoch',lambda *a,**k:pytest.fail('Unreliable cohort trained Actor'))
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1)
    assert not result['accepted'] and not result['actor_attempted']
    assert result['reason']=='no_reliable_training_cohorts'
    assert result['actor_train_indices']==result['actor_holdout_indices']==[]
    for name,p in actor.named_parameters():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        assert p.requires_grad


def test_phase_requires_two_supported_holdout_games(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,_,_=make_state_models(seed=32)
    episodes,split=episodes_for(actor,schema,adapter)
    def one_holdout(*args,**kwargs):
        weights,report=trusted_phase_weights(*args,**kwargs)
        for i in split['holdout'][1:]:weights[i][:]=0
        return weights,report
    monkeypatch.setattr(module,'behavior_value_weights',one_holdout)
    monkeypatch.setattr(module,'train_actor_epoch',lambda *a,**k:pytest.fail('Insufficient holdout trained Actor'))
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1)
    assert not result['actor_attempted']
    assert result['reason']=='fewer_than_2_reliable_holdout_games'
    assert result['eligible_actor_holdout_indices']==split['holdout'][:1]


def test_weighted_improvement_is_accepted_when_old_behavior_nll_rises(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.wait_actor=torch.nn.Linear(1,1)
            torch.nn.init.zeros_(self.wait_actor.weight);torch.nn.init.zeros_(self.wait_actor.bias)
        def forward_state(self,header,entities,mask,legal,**kwargs):
            score=self.wait_actor(header).squeeze(-1)
            lp=torch.stack((torch.nn.functional.logsigmoid(score),torch.nn.functional.logsigmoid(-score)),-1)
            return lp,torch.zeros_like(score),None
        def detach_state(self,state):return state
    class Adapter:
        schema=SimpleNamespace(header={'hand_card':(0,1,2,3)})
        def __call__(self,raw):
            b,t=raw.shape[:2]
            return np.ones((b,t,1),np.float32),np.zeros((b,t,1,1),np.float32),np.ones((b,t,1),bool)
    actor=Actor();critic=SimpleNamespace(critic=torch.nn.Linear(1,1))
    episodes=[dict(raw_obs=np.zeros((4,28),np.int32),actions=np.array([0,0,0,1]),
                   legal_masks=np.ones((4,2),bool),actor_keep=np.ones(4,bool),
                   behavior_version='policy',rewards=np.array([0,0,0,(-1.)**i],np.float32)) for i in range(5)]
    split={'train':[0,1,2],'holdout':[3,4]}
    targets=[np.ones(4,np.float32) for _ in episodes]
    monkeypatch.setattr(module,'_extract_features',lambda *a,**k:([],targets))
    def trusted(*args,**kwargs):
        weights,report=trusted_phase_weights(*args,**kwargs)
        for i in weights:weights[i]=np.array([.625,.625,.625,2.5],np.float32)
        return weights,report
    monkeypatch.setattr(module,'behavior_value_weights',trusted)
    result=run_phase(actor,critic,Adapter(),episodes,split,tmp_path,epochs=1,lr=.02,max_kl=.1)
    assert result['accepted'] and result['actor_attempted']
    assert result['epochs'][0]['weighted_nll']<result['baseline']['weighted_nll']
    assert result['epochs'][0]['unweighted_nll']>result['baseline']['unweighted_nll']
    assert result['actor_train_indices']==split['train']


def add_buffer_cohort(buffer,behavior,train_games,validation_games,*,prefix='',fresh=None):
    fresh=train_games if fresh is None else fresh
    for i in range(train_games):
        key=f'{prefix}{i:03d}-{behavior}'
        buffer.state['entries'][key]=dict(active=True,behavior=behavior,partition='train',uses=0 if i<fresh else 1)
    for i in range(validation_games):
        key=f'{prefix}val-{i:03d}-{behavior}'
        buffer.state['entries'][key]=dict(active=True,behavior=behavior,partition='validation',uses=0)


def test_selection_packs_cohorts_instead_of_splitting_three_versions_below_nine(tmp_path):
    buffer=PlayBuffer(tmp_path)
    for behavior in ('a','b','c'):add_buffer_cohort(buffer,behavior,12,3)
    # A global hash-like interleaving would previously pick 7/7/6 train games.
    selected=buffer.select(min_new=12,batch_games=24)
    rows=[buffer.state['entries'][key] for key in selected]
    assert len(rows)<=24
    train=[row for row in rows if row['partition']=='train']
    held=[row for row in rows if row['partition']=='validation']
    assert len(held)>=2 and sum(row['uses']==0 for row in train)>=12
    for behavior in {row['behavior'] for row in train}:
        assert sum(row['behavior']==behavior for row in train)>=9
        assert any(row['behavior']==behavior for row in held)


def test_selection_admits_new_late_sorting_cohort_after_skipped_batch(tmp_path):
    buffer=PlayBuffer(tmp_path)
    add_buffer_cohort(buffer,'old',20,4,prefix='000-')
    before=buffer.select();buffer.state['last_skipped_batch']={'game_ids':before}
    add_buffer_cohort(buffer,'new',12,2,prefix='zzz-')
    selected=buffer.select()
    assert len(selected)<=24
    assert sum(buffer.state['entries'][key]['behavior']=='new' and
               buffer.state['entries'][key]['partition']=='train' for key in selected)==12
    assert any(buffer.state['entries'][key]['behavior']=='new' and
               buffer.state['entries'][key]['partition']=='validation' for key in selected)


def test_selection_admits_new_same_cohort_holdout_after_skipped_batch(tmp_path):
    buffer=PlayBuffer(tmp_path);add_buffer_cohort(buffer,'policy',20,4)
    buffer.state['last_skipped_batch']={'game_ids':buffer.select()}
    key='zzz-new-validation'
    buffer.state['entries'][key]=dict(active=True,behavior='policy',partition='validation',uses=0)
    assert key in buffer.select()


def test_selection_requires_fresh_training_quota_inside_usable_cohorts(tmp_path):
    buffer=PlayBuffer(tmp_path)
    add_buffer_cohort(buffer,'usable',13,2,fresh=9)
    add_buffer_cohort(buffer,'too-small',3,2)
    assert buffer.select(min_new=12,batch_games=24)==[]
    selected=buffer.select(min_new=9,batch_games=24)
    assert selected and all(buffer.state['entries'][key]['behavior']=='usable' for key in selected)
    for row in buffer.state['entries'].values():
        if row['partition']=='train':row['uses']=2
    assert buffer.select(min_new=1,batch_games=24)==[]


@pytest.mark.parametrize('drop_unreliable',[False,True])
def test_phase_rechecks_new_game_quota_after_value_filtering(tmp_path,monkeypatch,drop_unreliable):
    import cr_rl.training.awbc_playbuffer as module
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,_,_=make_state_models(seed=41)
    episodes,_=episodes_for(actor,schema,adapter)
    episodes.extend(copy.deepcopy(episodes[:3]))
    split={'train':list(range(12)),'holdout':[12,13,14]}
    def supported(*args,**kwargs):
        weights,report=trusted_phase_weights(*args,**kwargs)
        if drop_unreliable:
            for i in [9,10,11]:weights[i][:]=0
        return weights,report
    monkeypatch.setattr(module,'behavior_value_weights',supported)
    monkeypatch.setattr(module,'train_actor_epoch',lambda *a,**k:pytest.fail('Insufficient new games trained Actor'))
    fresh=list(range(12 if drop_unreliable else 9))
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1,
                     fresh_indices=fresh,min_new_games=12)
    assert not result['actor_attempted'] and not result['memory_invalidated']
    assert result['reason']=='fewer_than_required_reliable_new_train_games'
    assert len(result['eligible_new_train_indices'])==9


def test_zero_optimizer_steps_reports_rollback_memory_invalidation(tmp_path,monkeypatch):
    import cr_rl.training.awbc_playbuffer as module
    from test_state_ppo_v3 import make_state_models
    schema,adapter,actor,critic,_,_=make_state_models(seed=42)
    episodes,split=episodes_for(actor,schema,adapter)
    monkeypatch.setattr(module,'behavior_value_weights',trusted_phase_weights)
    monkeypatch.setattr(module,'train_actor_epoch',lambda *a,**k:{
        'optimizer_steps':0,'optimized_indices':[],'episode_mean_weighted_nll':0.})
    generation=actor.memory_generation
    result=run_phase(actor,critic,adapter,episodes,split,tmp_path,epochs=1,value_epochs=1)
    assert not result['actor_attempted'] and result['memory_invalidated']
    assert result['reason']=='no_actor_optimizer_steps'
    assert actor.memory_generation>generation
    assert result['actor_train_indices']==[]


def test_selection_considers_pair_without_highest_priority_cohort(tmp_path):
    buffer=PlayBuffer(tmp_path)
    add_buffer_cohort(buffer,'a',9,2,fresh=1)
    add_buffer_cohort(buffer,'b',10,2)
    add_buffer_cohort(buffer,'c',10,2)
    # A has the only unseen fresh game and sorts first, but A+B/A+C only have
    # 11 fresh games. B+C is a feasible 20-train + 4-holdout batch.
    buffer.state['last_skipped_batch']={'game_ids':[
        key for key,row in buffer.state['entries'].items() if row['behavior'] in ('b','c')]}
    selected=buffer.select(min_new=12,batch_games=24)
    rows=[buffer.state['entries'][key] for key in selected]
    assert len(rows)==24
    assert {row['behavior'] for row in rows}=={'b','c'}
    assert sum(row['partition']=='train' and row['uses']==0 for row in rows)==20
