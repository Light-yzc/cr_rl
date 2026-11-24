"""CPU/mocked evaluator tests; no JVM, network, CUDA, or full checkpoint."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.models import ActorCritic, ActorPolicy, ModelConfig
from cr_rl.models.state_policy import StateActorCritic
from cr_rl.models.state_temporal import StateTemporalConfig

spec=importlib.util.spec_from_file_location("evaluate_state_policy",Path(__file__).resolve().parents[1]/"scripts/evaluate_state_policy.py")
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def schema():
    header_fields=["entity_count","elixir",*(["hand_card"]*4),*(["hand_cost"]*4),
                   *(["deck"]*8),*(["enemy_seen"]*8)]
    entity_fields=["id","kind","side","row","x","y","state","flags","buff_0","buff_1","buff_2"]
    header={name:tuple(i for i,value in enumerate(header_fields) if value==name) for name in set(header_fields)}
    return SimpleNamespace(header=header,header_fields=header_fields,entity_fields=entity_fields,
        header_ints=len(header_fields),entity_ints=len(entity_fields),max_entities=1,
        seat_ints=len(header_fields)+len(entity_fields))


def configuration():
    return ModelConfig(entity_layers=1,entity_dim=8,entity_heads=2,temporal_layers=1,
        temporal_dim=8,temporal_heads=2,context_len=2,max_entities=1,vocab_size=16,
        header_dim=schema().header_ints,entity_feature_dim=schema().entity_ints).for_v2(schema(),pooling="mean")


def raw_values():
    s=schema();raw=np.zeros(s.seat_ints,dtype=np.int32)
    raw[s.header['entity_count'][0]]=1
    raw[s.header['elixir'][0]]=100000
    raw[list(s.header['hand_card'])]=[0,1,2,3]
    raw[list(s.header['hand_cost'])]=[1,3,5,8]
    raw[list(s.header['deck'])]=np.arange(8)
    legal=np.zeros(2307,dtype=bool);legal[0]=True
    return raw,legal


class MockStatePolicy:
    architecture_version=3
    def __init__(self):
        self.cfg=configuration();self.calls=[]
    def initial_state(self,batch_size,device,dtype=None):
        return 0
    def step(self,*tensors,state,**kwargs):
        self.calls.append(state)
        logp=torch.full((1,2307),-1e9);logp[:,0]=0
        return logp,torch.zeros(1),state+1


class MockLegacyPolicy:
    def __init__(self):
        self.cfg=replace(configuration(),context_len=3,temporal_dim=16,temporal_heads=2)
        self.time_masks=[]
    def __call__(self,header,entities,entity_mask,legal,time_mask,**kwargs):
        self.time_masks.append(time_mask.cpu().tolist())
        logp=torch.full((1,2307),-1e9);logp[:,0]=0
        return logp,torch.zeros(1)


class MockEnvironment:
    created=[]
    def __init__(self,count,threads,**kwargs):
        assert kwargs['decision_ticks']==8 and kwargs['independent']
        self.count=count;self.tick=0
        raw,legal=raw_values()
        self.observations=np.broadcast_to(raw,(count,2,len(raw))).copy()
        self.masks=np.broadcast_to(legal,(count,2,len(legal))).copy()
        self.rewards=np.zeros((count,2),dtype=np.float32)
        self.dones=np.zeros(count,dtype=bool)
        self.vector=SimpleNamespace(schema=schema(),vocabulary_size=16,
            info=self.info,lanes=[SimpleNamespace(replay_json=lambda _:json.dumps({'test':True})) for _ in range(count)])
        self.created.append(self)
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def set_autoreset(self,value):assert not value
    def reset(self,*args):self.reset_args=args
    def step(self,actions):
        assert not actions.any()
        self.tick+=1
        self.dones[:]=self.tick>=3
    def info(self,index):return {'fault':0,'truncated':0,'tick':self.tick*8,'illegal_0':0,'illegal_1':0}


class StateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp_root=Path(__file__).resolve().parents[1]/'artifacts/state_evaluation_tests'
        cls.temp_root.mkdir(parents=True,exist_ok=True)

    def test_argmax_uses_highest_legal_action_and_advances_memory(self):
        class RankedPolicy(MockStatePolicy):
            def step(self, *tensors, state, **kwargs):
                logp, value, new_state = super().step(*tensors, state=state, **kwargs)
                logp[:, 1] = 9.0
                logp[:, 2] = 2.0
                return logp, value, new_state

        raw, legal = raw_values()
        legal[2] = True
        stream = evaluation.PolicyStream(RankedPolicy(), CrForgeObservationAdapter(schema(), 16),
                                         'cpu', precision='fp32', action_mode='argmax')
        self.assertEqual([stream.sample(raw, legal)[0] for _ in range(2)], [2, 2])
        self.assertEqual(stream.state, 2)

    def test_batched_argmax_matches_single_streams_after_lanes_finish(self):
        torch.manual_seed(7)
        temporal = StateTemporalConfig(width=8, heads=2, kv_heads=1, head_dim=4,
            pattern=('kda', 'gqa'), ffn_multiplier=1, cache_initial_capacity=2)
        model = StateActorCritic(configuration(), temporal, decision_ticks=8).eval()
        adapter = CrForgeObservationAdapter(schema(), 16)
        batch = evaluation.BatchedArgmaxStream(model, adapter, 'cpu', precision='fp32')
        streams = [evaluation.PolicyStream(model, adapter, 'cpu', precision='fp32',
            action_mode='argmax') for _ in range(3)]
        raw, legal = raw_values()
        observations = np.stack([raw.copy() for _ in streams])
        observations[:, schema().header['elixir'][0]] = [100000, 45000, 70000]
        masks = np.stack([legal.copy() for _ in streams])
        masks[1, 4] = True
        masks[2, 8] = True
        for step, ids in enumerate(([0, 1, 2], [0, 1, 2], [0, 2], [2])):
            lanes = np.array(ids)
            actions, _ = batch.sample(observations[lanes], masks[lanes], lanes)
            expected = [streams[i].sample(observations[i], masks[i])[0] for i in lanes]
            self.assertEqual(actions.tolist(), expected)
            self.assertEqual(batch.state.temporal.positions.tolist(), [step + 1] * len(ids))
            for j, lane in enumerate(lanes):
                torch.testing.assert_close(batch.state.temporal.layers[0].matrix[j],
                    streams[lane].state.temporal.layers[0].matrix[0], atol=1e-5, rtol=1e-5)

    def test_contract_does_not_require_equal_temporal_architectures(self):
        a=configuration();b=replace(a,temporal_dim=16,temporal_layers=3,context_len=128)
        evaluation.check_policy_compatibility(a,b)
        with self.assertRaisesRegex(ValueError,'vocab_size'):
            evaluation.check_policy_compatibility(a,replace(b,vocab_size=17))

    def test_every_forced_wait_advances_independent_memory_and_reset_clears(self):
        model=MockStatePolicy();adapter=CrForgeObservationAdapter(schema(),16)
        a=evaluation.PolicyStream(model,adapter,'cpu',precision='fp32')
        b=evaluation.PolicyStream(model,adapter,'cpu',precision='fp32')
        raw,legal=raw_values()
        for _ in range(3):self.assertEqual(a.sample(raw,legal)[0],0)
        self.assertEqual(a.state,3)
        self.assertIsNone(b.state)
        b.sample(raw,legal)
        self.assertEqual(b.state,1)
        a.reset();a.sample(raw,legal)
        self.assertEqual(a.state,1)
        self.assertEqual(model.calls,[0,1,2,0,0])

    @unittest.skipUnless(torch.cuda.is_available(),"FP16 cache regression requires existing CUDA")
    def test_fp16_policy_stream_initializes_half_kv_and_reuses_it_on_later_steps(self):
        # Exercise the actual evaluator path. The benchmark and trainer already
        # supplied an explicit FP16 KV dtype; this online stream previously did
        # not, so its first autocast GQA scatter used FP32 storage + FP16 values.
        cfg=replace(configuration(),frame_encoder_fp32=True)
        temporal=StateTemporalConfig(width=8,heads=2,kv_heads=1,head_dim=4,
            pattern=('kda','gqa'),ffn_multiplier=1,cache_initial_capacity=2)
        model=StateActorCritic(cfg,temporal,decision_ticks=8).to('cuda').eval()
        stream=evaluation.PolicyStream(model,CrForgeObservationAdapter(schema(),16),
            'cuda',precision='fp16')
        raw,legal=raw_values()
        with torch.inference_mode():
            for expected_position in (1,2,3):
                action,latency=stream.sample(raw,legal)
                self.assertEqual(action,0)
                self.assertTrue(np.isfinite(latency))
                self.assertEqual(int(stream.state.temporal.positions[0]),expected_position)
                kda,gqa=stream.state.temporal.layers
                self.assertEqual(kda.matrix.dtype,torch.float32)
                self.assertTrue(torch.isfinite(kda.matrix).all())
                self.assertEqual(gqa.key.dtype,torch.float16)
                self.assertEqual(gqa.value.dtype,torch.float16)
                self.assertTrue(torch.isfinite(gqa.key).all())
                self.assertTrue(torch.isfinite(gqa.value).all())
            stream.reset()
            self.assertIsNone(stream.state)
            self.assertEqual(stream.sample(raw,legal)[0],0)
            self.assertEqual(int(stream.state.temporal.positions[0]),1)
            self.assertEqual(stream.state.temporal.layers[1].key.dtype,torch.float16)
        self.assertTrue(all(parameter.dtype==torch.float32 for parameter in model.parameters()))
        del stream,model
        torch.cuda.empty_cache()

    def test_legacy_uses_its_own_window_length_at_new_grid(self):
        model=MockLegacyPolicy();adapter=CrForgeObservationAdapter(schema(),16)
        stream=evaluation.PolicyStream(model,adapter,'cpu',precision='fp32')
        raw,legal=raw_values()
        for _ in range(5):stream.sample(raw,legal)
        self.assertEqual(len(stream.history),3)
        self.assertEqual(model.time_masks[0],[[False,False,True]])
        self.assertEqual(model.time_masks[-1],[[True,True,True]])

    def test_actual_state_policy_advances_all_wait_frames_and_resets_positions(self):
        torch.manual_seed(1)
        temporal=StateTemporalConfig(width=8,heads=2,kv_heads=1,head_dim=4,
            pattern=('kda','gqa'),ffn_multiplier=1,cache_initial_capacity=2)
        model=StateActorCritic(configuration(),temporal,decision_ticks=8).eval()
        stream=evaluation.PolicyStream(model,CrForgeObservationAdapter(schema(),16),'cpu',precision='fp32')
        raw,legal=raw_values()
        for _ in range(4):
            self.assertEqual(stream.sample(raw,legal)[0],0)
        self.assertEqual(stream.state.temporal.positions.tolist(),[4])
        stream.reset();stream.sample(raw,legal)
        self.assertEqual(stream.state.temporal.positions.tolist(),[1])

    def test_end_to_end_latency_quantiles_include_over_budget_samples(self):
        summary=evaluation.latency_summary([.001,.1,.4,.401])
        self.assertEqual(summary['count'],4)
        self.assertEqual(summary['max_ms'],401.)
        self.assertEqual(summary['violations'],1)
        with self.assertRaises(ValueError):evaluation.latency_summary([float('nan')])

    def test_wait_request_audit_uses_point_four_seconds_and_real_cost(self):
        raw,legal=raw_values();legal[1]=True
        audit=evaluation.new_decision_audit()
        evaluation.record_decision(audit,raw,legal,0,schema())
        evaluation.record_decision(audit,raw,legal,0,schema())
        action=1+2*576;legal[action]=True
        evaluation.record_decision(audit,raw,legal,action,schema())
        result=evaluation.finish_decision_audit(audit)
        self.assertEqual(result['full_elixir_wait_legal_seconds'],.8)
        self.assertEqual(result['longest_full_elixir_wait_legal_seconds'],.8)
        self.assertEqual(result['mean_requested_card_cost'],5.)
        self.assertEqual(result['requested_cost_at_least5_fraction'],1.)

    def test_paired_mock_games_preserve_decks_seeds_and_state_progress(self):
        model=MockStatePolicy();opponent=MockLegacyPolicy()
        matches=paired_match_specs(['a','b','c','d'],2)
        league={'decks':{key:['A']*8 for key in ['a','b','c','d']}}
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            args=SimpleNamespace(envs=2,device='cpu',precision='fp32',sampling_seed=42,
                max_decisions=1000,out=Path(temp))
            summary=evaluation.evaluate_games(model,opponent,[],league,matches,args,env_factory=MockEnvironment)
            self.assertEqual(len(list(Path(temp).glob('match_*.json'))),4)
        self.assertTrue(all(match['valid'] for match in matches))
        self.assertEqual(summary['candidate_end_to_end']['count'],12)
        self.assertEqual(model.calls,[0,0,1,1,2,2]*2)
        for a,b in zip(matches[::2],matches[1::2]):
            self.assertEqual((a['seed'],a['deck0'],a['deck1']),(b['seed'],b['deck0'],b['deck1']))

    def test_timeout_is_invalid_not_draw(self):
        model=MockStatePolicy();matches=paired_match_specs(['a','b'],1)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            args=SimpleNamespace(envs=2,device='cpu',precision='fp32',sampling_seed=42,
                max_decisions=1,out=Path(temp))
            evaluation.evaluate_games(model,model,[],{'decks':{'a':['A']*8,'b':['B']*8}},
                matches,args,env_factory=MockEnvironment)
        self.assertTrue(all(not match['valid'] for match in matches))
        self.assertTrue(all('reward' not in match for match in matches))
        self.assertTrue(all('not_a_draw' in match['reason'] for match in matches))

    def test_native_metrics_count_executions_not_requests_and_require_parity(self):
        side={'successful_card_plays':2,'paid_card_elixir_total':8.,
            'core_cost_thresholds':{'5':{'count':1}},'refused_card_commands':3,'successful_ability_commands':1}
        audit={'games':[{'native_outcome_parity':True,'learner':side,'opponent':side}]}
        summary=evaluation.native_strategy_summary(audit)
        self.assertEqual(summary['learner']['mean_paid_elixir_per_successful_card'],4.)
        self.assertEqual(summary['learner']['cost_at_least5_fraction'],.5)
        audit['games'][0]['native_outcome_parity']=False
        with self.assertRaises(ValueError):evaluation.native_strategy_summary(audit)

    def test_loaders_stream_v2_actor_policy_and_v3_and_do_not_apply_calibration(self):
        cfg=configuration()
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            for policy_only in (False,True):
                model=(ActorPolicy if policy_only else ActorCritic)(cfg)
                with torch.no_grad():model.wait_actor.bias.fill_(.6)
                path=Path(temp)/f'legacy_{policy_only}.pt'
                torch.save({'model_architecture_version':2,'policy_only':policy_only,
                    'config':asdict(cfg),'model':model.state_dict(),
                    'calibration':{'wait_logit_bias':100}},path)
                loaded,info=evaluation.load_frozen_policy(path,'cpu')
                self.assertEqual(isinstance(loaded,ActorPolicy),policy_only)
                self.assertAlmostEqual(loaded.wait_actor.bias.item(),.6,places=6)
                self.assertIsNone(loaded.holding_prior)
                self.assertFalse(info['calibration_applied'])
                self.assertIn('0.25s',info['grid_migration_limitation'])
            temporal=StateTemporalConfig(width=8,heads=2,kv_heads=1,head_dim=4,
                pattern=('gqa',),ffn_multiplier=1,cache_initial_capacity=4)
            state_model=StateActorCritic(cfg,temporal,decision_ticks=8)
            path=Path(temp)/'state.pt'
            torch.save({**state_model.checkpoint_config(),'model':state_model.state_dict()},path)
            loaded,info=evaluation.load_frozen_policy(path,'cpu')
            self.assertIsInstance(loaded,StateActorCritic)
            self.assertEqual(info['architecture'],3)
            for name,value in state_model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name],value)


if __name__=='__main__':unittest.main()
