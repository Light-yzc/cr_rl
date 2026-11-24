"""Observation caching preserves policy outputs, gradients and update lifetimes."""
import copy
import numpy as np
import pytest
import torch
from cr_rl.training.state_ppo import _adapt_segment_batch, state_frame_cache
from test_state_ppo_v3 import make_state_models, make_synthetic_segment


def test_cached_state_frames_preserve_policy_and_gradients_with_padding():
    schema, adapter, actor, _, _, _ = make_state_models(seed=491)
    segments = [make_synthetic_segment(schema, adapter, actor, length=n, seed=20+n, prefix_len=0)
                for n in (7, 4)]
    kwargs=dict(start=0,end=7,device=torch.device('cpu'))
    baseline=_adapt_segment_batch(segments,adapter,**kwargs)
    expected=copy.deepcopy(actor).train()
    actual=copy.deepcopy(actor).train()
    legal=np.zeros((2,7,2307),dtype=bool);legal[...,0]=True
    for i,s in enumerate(segments):legal[i,:len(s['raw_obs'])]=s['legal_masks']
    def forward(model, rows):
        h,e,m,t,hands,_=rows
        lp,v,_=model.forward_state(h,e,m,torch.as_tensor(legal),hand_card_ids=hands,time_mask=t)
        loss=(lp[...,0]*t).sum()+(v.square()*t).sum()
        loss.backward()
        return lp.detach(),v.detach()
    exp=forward(expected,baseline)
    with state_frame_cache(segments,adapter,['cpu']) as stats:
        assert stats['bytes_per_device']['cpu']>0
        cached=_adapt_segment_batch(segments,adapter,**kwargs)
        np.testing.assert_array_equal(cached[-1],baseline[-1])
        for tensor,reference in zip(cached[:5],baseline[:5]):
            torch.testing.assert_close(tensor[cached[3]],reference[baseline[3]])
        got=forward(actual,cached)
        for a,b in zip(exp,got):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        for a,b in zip(expected.parameters(),actual.parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-4)
    assert all('_state_frame_cache' not in s for s in segments)


def test_state_frame_cache_budget_fallback_and_exception_cleanup():
    schema,adapter,actor,_,_,_=make_state_models(seed=492)
    segments=[make_synthetic_segment(schema,adapter,actor,length=4,seed=12,prefix_len=0)]
    with state_frame_cache(segments,adapter,['cpu'],max_bytes_per_device=0) as stats:
        assert stats['bytes_per_device']['cpu']==0
        _adapt_segment_batch(segments,adapter,start=0,end=4,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='deliberate'):
        with state_frame_cache(segments,adapter,['cpu']):
            raise RuntimeError('deliberate')
    assert all('_state_frame_cache' not in s for s in segments)


def test_geometric_kv_reservation_matches_untrimmed_recurrent_steps():
    from cr_rl.models.state_policy import PolicyMemory, reserve_step_memory, slice_memory
    from cr_rl.models.state_temporal import StateTemporalConfig, StateTemporalCore
    torch.manual_seed(87)
    cfg=StateTemporalConfig(width=16,heads=2,kv_heads=1,head_dim=8,
        pattern=('kda','gqa'),chunk_size=4,cache_initial_capacity=4)
    core=StateTemporalCore(cfg).eval()
    x=torch.randn(2,17,16)
    expected=core.init_state(2)
    actual=PolicyMemory(core.init_state(2),id(core),0)
    with torch.inference_mode():
        for t in range(x.shape[1]):
            a,expected=core.step(x[:,t],expected)
            trimmed=slice_memory(actual,[0,1],detach=True,trim_gqa=True)
            before=trimmed.temporal.positions.clone()
            actual=reserve_step_memory(trimmed,cfg.cache_initial_capacity)
            torch.testing.assert_close(actual.temporal.positions,before)
            assert actual.owner==id(core) and actual.generation==0
            assert actual.temporal.layers[1].key.shape[2]>=actual.temporal.length_bound+1
            b,next_state=core.step(x[:,t],actual.temporal)
            actual=PolicyMemory(next_state,id(core),0)
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            if t==7:
                reset=torch.tensor([True,False])
                expected=expected.reset(reset)
                actual=PolicyMemory(actual.temporal.reset(reset),id(core),0)


def test_graceful_training_stop_prunes_restore_points(tmp_path, monkeypatch):
    import signal
    from types import SimpleNamespace
    from cr_rl.training import state_selfplay as training
    trainer=training.PersistentStateSelfplay.__new__(training.PersistentStateSelfplay)
    trainer.args=SimpleNamespace(updates=1,keep_checkpoints=2,keep_rollouts=2,keep_periodic_checkpoints=1)
    trainer.update=711;trainer.rank=0;trainer._last_checkpoint_update=710
    trainer.out=tmp_path;trainer.opponents=[];events=[]
    trainer.emit=lambda **row:events.append(row)
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    for update in (640,708,710):(checkpoints/f'update_{update:06d}.pt').write_bytes(b'restore')
    def update():
        trainer.update+=1
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    trainer.run_update=update
    trainer.checkpoint=lambda:(checkpoints/f'update_{trainer.update:06d}.pt').write_bytes(b'new')
    monkeypatch.setattr(training,'prune_local_opponents',lambda *a,**kw:{})
    trainer.run()
    assert sorted(p.name for p in checkpoints.glob('*.pt'))==[
        'update_000640.pt','update_000710.pt','update_000712.pt']
    assert events[-1]['phase']=='storage_retention' and events[-1]['removed_files']==1
