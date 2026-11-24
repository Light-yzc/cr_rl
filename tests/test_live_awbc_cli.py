"""Small end-to-end value cross-fit -> Actor update -> resume, no live GPU use."""
import gzip
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cr_rl.client.rl_recording import CONTRACT, actor_fingerprint, set_result
from cr_rl.models.policy import ModelConfig
from cr_rl.models.state_policy import StateActorCritic, StateValueNetwork
from cr_rl.models.state_temporal import StateTemporalConfig


@pytest.mark.parametrize('trusted_value',[False,True])
def test_cli_trains_and_resumes_only_offline_readouts(tmp_path,monkeypatch,trusted_value):
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('awbc_cli',root/'scripts/train_live_awbc.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if sys.platform=='win32':
        import msvcrt
        # Tiny CPU fixture; avoid interfering with the user's live collection lock.
        monkeypatch.setattr(msvcrt,'locking',lambda *args:None)
    fields=['reserved']*224
    fields[0]='tick';fields[2]='elixir';fields[48]='entity_count'
    for i in range(7,11):fields[i]='hand_card'
    entity=['reserved']*34
    entity[0]='row';entity[1]='kind';entity[2]='side';entity[4]='x';entity[5]='y'
    cfg=ModelConfig(entity_dim=16,entity_heads=2,entity_layers=1,temporal_dim=16,
                    temporal_heads=2,header_fields=tuple(fields),entity_fields=tuple(entity),vocab_size=16)
    temporal=StateTemporalConfig(width=16,heads=2,kv_heads=1,head_dim=8,pattern=('kda','gqa'),ffn_multiplier=2)
    torch.manual_seed(5)
    actor=StateActorCritic(cfg,temporal,decision_ticks=8)
    critic=StateValueNetwork(cfg,temporal,decision_ticks=8)
    ckpt={'model':actor.state_dict(),'critic':critic.state_dict(),'config':asdict(actor.cfg),
          'critic_config':asdict(critic.cfg),'temporal_config':asdict(temporal),
          'model_architecture_version':3,'decision_ticks':8,'updates':3}
    checkpoint=tmp_path/'source.pt';torch.save(ckpt,checkpoint)
    fp=actor_fingerprint(ckpt)
    sessions=tmp_path/'sessions';sessions.mkdir()
    from run_nulls_dryrun import schema_from_model_config
    from cr_rl.data.observations import CrForgeObservationAdapter
    adapter=CrForgeObservationAdapter(schema_from_model_config(actor.cfg),vocabulary_size=16,entity_bucket=16)
    actor.eval()
    for idx in range(12):
        run=sessions/f'game_{idx:02d}';run.mkdir();(run/'diagnostics').mkdir()
        obs=np.zeros(7024,dtype=np.int32);obs[2]=100000;obs[7:11]=[0,1,2,3];obs[20:28]=np.arange(8)
        terminal=obs.copy();terminal[0]=17
        manifest={'arguments':{'selection_mode':'policy_sample','execute':True},
                  'rl_contract':{'version':CONTRACT,'action_semantics':'policy_proposal','initial_memory_reset':True,'actor_fingerprint':fp},
                  'checkpoint_sha256':'synthetic-source','frames':2,'collection_complete':True,'stop_reason':'battle_end'}
        (run/'manifest.json').write_text(json.dumps(manifest))
        (run/'terminal.json').write_text(json.dumps({'clock_ended':True,'session_id':run.name,'match_id':run.name,
                                                     'tick':17,'obs':terminal.tolist()}))
        events=[{'event':'battle_start','tick':1}]
        raw=np.stack([obs.copy(),obs.copy()]);raw[:,0]=[1,9]
        legal=np.zeros((1,2,2307),dtype=bool);legal[:,:,0:2]=True
        h,e,m=adapter(raw[None])
        with torch.no_grad():
            logits,_,_=actor.forward_state(torch.as_tensor(h),torch.as_tensor(e),torch.as_tensor(m),
                                           torch.as_tensor(legal),hand_card_ids=torch.as_tensor(raw[None,:,7:11]),compute_value=False)
            behavior=logits.exp()[0].numpy()
        with gzip.open(run/'diagnostics/frames_0001.jsonl.gz','wt') as stream:
            for decision,tick in [(1,1),(2,9)]:
                obs[0]=tick
                frame={'decision_id':decision,'tick':tick,'match_id':run.name,'action':0,
                       'action_selection':{'proposed_action':0},'obs':obs.tolist(),'probabilities':behavior[decision-1].tolist(),
                       'legal_actions':[0,1],'proposal_log_probability':float(np.log(behavior[decision-1,0]))}
                stream.write(json.dumps(frame)+'\n')
                events.append({'event':'decision','decision_id':decision,'tick':tick,'action':0})
        (run/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
        set_result(run,'win' if idx%2 else 'loss',evidence='synthetic test result')
    # The training Actor differs from the collection policy. Historical data
    # must be admitted, while KL references the current pre-AWBC Actor.
    key=next(k for k in ckpt['model'] if k.startswith('wait_actor.') and k.endswith('bias'))
    ckpt['model'][key]=ckpt['model'][key].clone()+.05
    torch.save(ckpt,checkpoint)
    assert actor_fingerprint(ckpt)!=fp
    out=tmp_path/'train'
    original_weights=module.behavior_value_weights
    def controlled_weights(initial,features,targets,split,behaviors,groups,**kwargs):
        # Exercise both CLI branches without relying on a random tiny Value fit.
        weights,report=original_weights(initial,features,targets,split,behaviors,groups,**kwargs)
        for i in weights:weights[i][:]=1. if trusted_value else 0.
        report.update(value_gate_passed=trusted_value,unit_weight_train_games=0,
                      weighted_train_games=len(split['train']) if trusted_value else 0,
                      weighted_holdout_games=len(split['holdout']) if trusted_value else 0)
        return weights,report
    monkeypatch.setattr(module,'behavior_value_weights',controlled_weights)
    def call(epochs):
        monkeypatch.setattr(sys,'argv',['train_live_awbc.py','--sessions',str(sessions),
            '--checkpoint',str(checkpoint),'--out',str(out),'--device','cpu','--epochs',str(epochs),
            '--value-epochs','1','--patience','3'])
        module.main()
    call(1)
    actor_report=json.loads((out/'actor_update_report.json').read_text())
    if not trusted_value:
        assert not actor_report['actor_attempted']
        assert actor_report['reason']=='no_reliable_training_cohorts'
        assert not (out/'latest.pt').exists() and not (out/'best.pt').exists()
        return
    assert actor_report['actor_attempted']
    first=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    assert first['awbc_epoch']==1
    assert json.loads((out/'value_crossfit.json').read_text())['folds']
    call(2)
    second=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    assert second['awbc_epoch']==2
    assert (out/'best.pt').is_file() and (out/'value_head.pt').is_file()
    trainable=set(json.loads((out/'trainable_actor_parameters.json').read_text()))
    changed=0
    for key,original in ckpt['model'].items():
        if not torch.is_tensor(original):continue
        if key not in trainable:torch.testing.assert_close(second['model'][key],original,rtol=0,atol=0)
        elif not torch.equal(second['model'][key],original):changed+=1
    assert changed
