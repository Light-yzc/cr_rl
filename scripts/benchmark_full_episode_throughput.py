"""Compare input batching/cache on identical full games, with zero learning rates.

Uses the production train CLI via the same wrapper convention as the smoke tool.
No training checkpoint, league snapshot or rollout dataset is published.
"""
import gc
import os
import json
import resource
import runpy
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from cr_rl.training import state_selfplay as training
from cr_rl.training.state_ppo import optimize_state_separate, state_frame_cache


def benchmark(args):
    trainer=training.PersistentStateSelfplay(args)
    results=[]
    try:
        replay_path=Path(os.environ.get("CRRL_PROFILE_REPLAY", "/run/media/regen/本地磁盘/cr_rl_active/throughput_profile/segments.pt"))
        replay = torch.load(replay_path, map_location="cpu", weights_only=False) if replay_path.exists() else None
        if replay is not None and replay['version']==trainer.update:
            segments=replay['segments']
        else:
            rollout=trainer.collect()
            all_segments=sorted(rollout['segments'],key=lambda s:len(s['raw_obs']))
            selection=np.linspace(0,len(all_segments)-1,min(16,len(all_segments)),dtype=int)
            segments=[all_segments[int(i)] for i in selection]
            print(json.dumps({'rollout_timing':{k:v for k,v in rollout.items() if k.endswith('seconds')}}),flush=True)
            replay_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(version=trainer.update,segments=segments),replay_path)
            del rollout,all_segments
        del replay
        samples=sum(int(s['valid'].sum()) for s in segments)
        print(json.dumps({'benchmark_samples':samples,'segments':len(segments),
            'longest':max(len(s['raw_obs']) for s in segments)}),flush=True)
        for optimizer in (trainer.actor_optimizer,trainer.critic_optimizer):
            for group in optimizer.param_groups:
                group['lr']=0.0
                group['weight_decay']=0.0
        cfg=replace(trainer._build_ppo_config(),epochs=1,critic_epochs=1)
        devices=[trainer.device,trainer.critic_device]
        models=[trainer.actor,trainer.critic,trainer.peer_actor,trainer.behavior_model,trainer.anchor_model]
        for frame_batch,cached in ((32,False),(64,False),(128,False),(64,True),(128,True),(32,False)):
            for model in models:
                if model is not None:model.frame_microbatch=frame_batch
            gc.collect()
            for dev in devices:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            began=time.perf_counter()
            ctx=state_frame_cache(segments,trainer.adapter,devices) if cached else nullcontext({})
            with ctx as cache_stats:
                out=optimize_state_separate(trainer._actor_opt_model,trainer._critic_opt_model,
                    trainer.behavior_model,trainer.actor_optimizer,trainer.critic_optimizer,
                    trainer.adapter,segments,cfg,rng=np.random.default_rng(817),
                    actor_device=trainer.device,critic_device=trainer.critic_device,
                    peer_actor=trainer.peer_actor,peer_actor_device=trainer.actor_peer_device,
                    anchor=trainer.anchor_model,anchor_device=trainer.opponent_device)
            for dev in devices:torch.cuda.synchronize(dev)
            row=dict(frame_microbatch=frame_batch,cache_state_frames=cached,
                seconds=time.perf_counter()-began,samples=samples,
                actor_seconds=out['actor_update_seconds'],critic_seconds=out['critic_update_seconds'],
                actor_steps=out['actor_steps'],critic_steps=out['critic_steps'],
                initial_logp_error=max(out['initial_logp_errors'],default=0),
                peak_cuda_gib={str(d):torch.cuda.max_memory_allocated(d)/2**30 for d in devices},
                host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                cache=cache_stats)
            row['actor_overflow_steps']=out['actor_overflow_steps']
            row['critic_overflow_steps']=out['critic_overflow_steps']
            row['kl_stopped']=out['stopped_early']
            row['valid']=not out['stopped_early'] and out['actor_overflow_steps']==out['critic_overflow_steps']==0
            assert all(v<14.5 for v in row['peak_cuda_gib'].values())
            assert row['host_peak_rss_gib']<12.5
            if results and row['valid']:
                assert (row['actor_steps'],row['critic_steps'])==(results[0]['actor_steps'],results[0]['critic_steps'])
            results.append(row)
            print(json.dumps(row),flush=True)
            Path(args.out,'benchmark_results.json').write_text(json.dumps(results,indent=2))
    finally:
        trainer.close()


if __name__=='__main__':
    training.run=benchmark
    if sys.argv[1:3]==['-u','scripts/train_selfplay.py']:
        sys.argv=[sys.argv[0],*sys.argv[3:]]
    runpy.run_path('scripts/train_selfplay.py',run_name='__main__')
