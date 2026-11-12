"""Compare real resumed BC batches without advancing the saved checkpoint."""
import json
import os
import time
from pathlib import Path

import torch
from cr_rl.training.bc import hierarchical_bc_loss


def benchmark(model, optimizer, batch, rows, obs, args):
    cache = obs.arrays[0].cache
    model.train()
    for group in optimizer.param_groups:
        group['lr'] = 0.0
    if os.environ.get("CRRL_BC_TF32_BENCH"):
        sample = batch(rows[:4])
        h,e,em,legal,tm,hand,y,deck,seen=sample
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            torch.set_float32_matmul_precision('highest')
            a,_=model(h,e,em,legal,tm,hand_card_ids=hand,deck_card_ids=deck,enemy_seen_card_ids=seen)
            torch.set_float32_matmul_precision('high')
            b,_=model(h,e,em,legal,tm,hand_card_ids=hand,deck_card_ids=deck,enemy_seen_card_ids=seen)
        report={'probability_max_abs':float((a.exp()-b.exp()).abs().max()),
            'selected_logp_max_abs':float((a.gather(-1,y[:,None])-b.gather(-1,y[:,None])).abs().max()),
            'finite':bool(torch.isfinite(b).all())}
        report['passed']=report['finite'] and report['probability_max_abs']<0.001 and report['selected_logp_max_abs']<0.02
        print(json.dumps({'tf32_parity':report}),flush=True)
        Path(args.out.parent/'tf32_parity.json').write_text(json.dumps(report,indent=2))
        if not report['passed']:return
    results = []
    for microbatch, cache_gib in (((4,16),) if os.environ.get("CRRL_BC_PREFETCH_BENCH") else ((4, 1), (4, 16), (8, 16), (12, 16), (16, 16), (24, 16))):
        cache.max_bytes = cache_gib * 2**30
        trials = []
        try:
            for repeat in range(2):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                timers = dict(batch=0., forward_loss=0., backward=0., optimizer=0.)
                extractions = cache.extractions
                began = time.perf_counter()
                if os.environ.get("CRRL_BC_PREFETCH_BENCH"):
                    from cr_rl.training.bc_prefetch import prefetch_cpu_batches
                    batches = prefetch_cpu_batches(range(0,len(rows),microbatch), rows, microbatch, batch, pin=True)
                else:
                    batches = ((i,None) for i in range(0,len(rows),microbatch))
                for start, cpu in batches:
                    torch.cuda.synchronize(); t = time.perf_counter()
                    h,e,em,legal,tm,hand,y,deck,seen = (
                        tuple(v.to("cuda",non_blocking=True) for v in cpu) if cpu is not None
                        else batch(rows[start:start+microbatch]))
                    torch.cuda.synchronize(); timers['batch'] += time.perf_counter()-t
                    t=time.perf_counter()
                    with torch.autocast('cuda',dtype=torch.float16):
                        logp,_=model(h,e,em,legal,tm,hand_card_ids=hand,deck_card_ids=deck,enemy_seen_card_ids=seen)
                        loss,_=hierarchical_bc_loss(logp,y,legal,position_coef=args.position_coef,
                            card_class_weights=None, conditional_card_loss=args.bc_objective == "conditional",
                            tile_label_sigma=args.tile_label_sigma,
                            timing_prior=0.040741896342520366,
                            sampled_action_prior=0.5)
                    if not torch.isfinite(loss): raise FloatingPointError('benchmark loss')
                    torch.cuda.synchronize(); timers['forward_loss'] += time.perf_counter()-t
                    t=time.perf_counter()
                    (loss * (len(y)/len(rows)) * 1024).backward()
                    torch.cuda.synchronize();timers['backward']+=time.perf_counter()-t
                t=time.perf_counter()
                for p in model.parameters():
                    if p.grad is not None:p.grad.div_(1024)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();timers['optimizer']+=time.perf_counter()-t
                trials.append(dict(**timers, wall=time.perf_counter()-began,
                    cache_extractions=cache.extractions-extractions,
                    peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, grad_norm=float(norm)))
            result=dict(batch=microbatch,cache_gib=cache_gib,trials=trials)
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            result=dict(batch=microbatch,cache_gib=cache_gib,error=str(exc))
        results.append(result);print(json.dumps({'speed_benchmark':result}),flush=True)
        Path(args.out.parent/('tf32_benchmark.json' if os.environ.get('CRRL_BC_TF32_BENCH') else 'prefetch_benchmark.json' if os.environ.get('CRRL_BC_PREFETCH_BENCH') else 'speed_benchmark.json')).write_text(json.dumps(results,indent=2))
