"""Reuse the production CLI for a bounded, non-publishing two-update GPU check.

Run through run_v100_v3.sh with PYTHON pointing at a wrapper invoking this file.
No checkpoints or league snapshots are saved by this harness.
"""
import json
import os
import resource
import runpy
import sys
from pathlib import Path

from cr_rl.training import state_selfplay as training


def smoke(args):
    trainer = training.PersistentStateSelfplay(args)
    trainer.target_update = trainer.update+1000000
    trainer.args.checkpoint_interval = 1000000
    trainer._update_opponent_pool = lambda *a, **kw: None
    collect = trainer.collect
    def checked_collect(**kwargs):
        result = collect(**kwargs)
        for seg in result['segments']:
            ticks = seg['raw_obs'][:,trainer.schema.header['tick'][0]]
            if len(ticks)>1:
                assert ((ticks[1:]-ticks[:-1])==8).all(), 'Duplicated or skipped segment frames'
        count = sum(len(s['raw_obs']) for s in result['segments'])
        if getattr(args, "full_episode_bptt", False):
            assert all(s["full_episode"] and s["has_complete_prefix"] for s in result["segments"])
            assert all(s["terminated"][-1] or s["truncated"][-1] for s in result["segments"])
        else:
            assert args.steps*args.envs <= count <= args.steps*args.envs*2
        print(json.dumps({'smoke_samples':count,'base_samples':args.steps*args.envs,
            'segments':len(result['segments']), 'max_episode_frames':result.get('max_episode_frames')}),flush=True)
        return result
    trainer.collect = checked_collect
    rows = []
    try:
        for _ in range(int(os.environ.get("CRRL_SMOKE_UPDATES", "2"))):
            row = trainer.run_update()
            # run_update emits the summary; read the metrics file for a stable result.
            assert all(v < 14.5 for v in row.get("peak_cuda_per_device_gib", {}).values())
            assert not row.get("actor_kl_stopped", False)
            assert row.get("initial_logp_max_error", 0) < (args.max_initial_logp_error or 0.05)
            rows.append(row)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
            assert rss < 12.5
            print(json.dumps({'smoke_update':trainer.update,'host_peak_rss_gib':rss}),flush=True)
    finally:
        trainer.close()
    Path(args.out,'smoke_complete.json').write_text(json.dumps({'updates':len(rows),'host_peak_rss_gib':rss,'results':rows},default=str,indent=2))


if __name__ == '__main__':
    training.run = smoke
    # Strip the interpreter-style prefix supplied by the production shell launcher.
    if sys.argv[1:3] == ['-u','scripts/train_selfplay.py']:
        sys.argv = [sys.argv[0],*sys.argv[3:]]
    runpy.run_path('scripts/train_selfplay.py',run_name='__main__')
