"""Measure a live longrun or sweep Colab PPO forward/backward microbatch sizes."""

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path('/content/crrl_training')
DATASET = Path('/content/crrl_sparky_bc_all_train_20261120')


def observe(run, updates):
    metrics = run / 'metrics.jsonl'
    with metrics.open('rb') as stream:
        stream.seek(max(0, metrics.stat().st_size - 2_000_000))
        rows = []
        for raw in stream:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get('phase') == 'state_ppo':
                rows.append(row)
    recent = rows[-updates:]
    if not recent:
        raise ValueError('No completed PPO updates in the metrics log')
    times = defaultdict(float)
    names = ('rollout_seconds', 'rollout_model_seconds', 'rollout_env_seconds',
             'rollout_bookkeeping_seconds', 'critic_eval_seconds', 'ppo_optimization_seconds',
             'actor_update_seconds', 'critic_update_seconds', 'expert_bc_seconds', 'seconds')
    for row in recent:
        for name in names:
            times[name] += row.get(name) or 0
    hardware = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()
    bcs = [row.get('expert_bc', {}) for row in recent]
    return dict(updates=[row['update'] for row in recent], gpu_snapshot=hardware,
                weighted_samples_per_second=(sum(row['valid_learner_samples'] for row in recent)
                                             / max(times['seconds'], 1e-9)),
                seconds=dict(times), peak_cuda_gib=max(row.get('peak_cuda_allocated_gib') or 0 for row in recent),
                actor_overflows=sum(row.get('actor_overflow_steps') or 0 for row in recent),
                critic_overflows=sum(row.get('critic_overflow_steps') or 0 for row in recent),
                expert_bc_accepted=sum(bool(row.get('accepted')) for row in bcs),
                expert_bc_attempts=sum(bool(row.get('ran')) for row in bcs),
                max_frozen_opponent_groups=max(row.get('max_frozen_opponent_groups') or 0 for row in recent))


def assert_no_trainer():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(a.endswith((b'/train_selfplay.py', b'/cr_train_pipeline.py')) for a in args[:3]):
            raise RuntimeError('Use observe during training; the sweep needs an idle GPU')


def sweep(checkpoint, candidates, *, network='actor', batch_sizes=None, sequence_steps=128,
          activation_checkpointing=True, memory_fraction=1.0):
    assert_no_trainer()
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
    import numpy as np
    import torch
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.models.state_policy import state_model_from_checkpoint, state_value_model_from_checkpoint
    from run_nulls_dryrun import schema_from_model_config

    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    card = torch.cuda.get_device_properties(device)
    if '6000' not in card.name and 'A100' not in card.name:
        raise ValueError(f'Expected RTX PRO 6000 or A100, got {card.name}')
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    snapshot = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    update = snapshot['updates']
    load = state_model_from_checkpoint if network == 'actor' else state_value_model_from_checkpoint
    model = load(snapshot, device).train()
    model.ppo_actor_fp32 = True
    model.gradient_checkpointing = activation_checkpointing
    model.temporal_checkpoint_mode = 'block'
    del snapshot
    schema = schema_from_model_config(model.cfg)
    adapter = CrForgeObservationAdapter(schema, model.cfg.vocab_size, entity_bucket=16)
    manifest = json.loads((DATASET/'dataset.manifest.json').read_text())
    examples = []
    total_examples = 16 if batch_sizes is not None else 8
    stride = 17 if sequence_steps <= 128 else 1
    for index in range(0, len(manifest['shards']), stride):
        desc = manifest['shards'][index]
        with np.load(DATASET / desc['path'], allow_pickle=False) as a:
            seat = a['seats'] == desc['selected_seats'][0]
            rows = np.flatnonzero(seat)
            if len(rows) < sequence_steps:
                continue
            rows = rows[-sequence_steps:]
            examples.append((a['observations'][rows], a['masks'][rows], a['actions'][rows]))
        if len(examples) == total_examples:
            break
    if len(examples) != total_examples:
        raise ValueError(f'Could not sample {total_examples} representative BC frame sequences')
    obs, masks, actions = (np.stack(x) for x in zip(*examples))
    h, e, em = (torch.as_tensor(x, device=device) for x in adapter(obs))
    lm = torch.as_tensor(masks, device=device, dtype=torch.bool)
    action = torch.as_tensor(actions, device=device, dtype=torch.long)
    hand = torch.as_tensor(obs[..., list(schema.header['hand_card'])], device=device)
    choice = lm.sum(-1) > 1
    count = int(choice.sum()) if network == 'actor' else (h.shape[0] * h.shape[1])
    if not count:
        raise ValueError('Profile probe has no legal choices')
    reports = []
    profiles = zip(candidates, batch_sizes) if batch_sizes is not None else ((n, total_examples) for n in candidates)
    for frames, batch in profiles:
        model.frame_microbatch = frames
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        seconds, values = [], []
        try:
            for index in range(3):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                model.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=device)
                for start in range(0, total_examples, batch):
                    part = slice(start, start + batch)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if network == 'actor':
                            logits, _, _ = model.forward_state(h[part], e[part], em[part], lm[part],
                                compute_value=False, hand_card_ids=hand[part])
                            nll = -logits.float().gather(-1, action[part].unsqueeze(-1)).squeeze(-1)
                            loss = nll[choice[part]].sum() / count
                        else:
                            value, _ = model.forward_state(h[part], e[part], em[part])
                            loss = value.float().square().sum() / count
                    loss.backward()
                    total_loss += loss.detach()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                values.append((float(total_loss), float(norm)))
                torch.cuda.synchronize(device)
                if index:
                    seconds.append(time.perf_counter() - begin)
            peak = torch.cuda.max_memory_reserved(device) / 2**30
            reports.append(dict(frame_microbatch=frames, batch_size=batch, median_seconds=statistics.median(seconds),
                                peak_reserved_gib=peak, finite=all(math.isfinite(v) for pair in values for v in pair),
                                loss=values[-1][0]))
        except torch.cuda.OutOfMemoryError:
            reports.append(dict(frame_microbatch=frames, batch_size=batch, oom=True))
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    baseline = next((r for r in reports if r.get('finite')), None)
    usable = [r for r in reports if baseline and r.get('finite') and r['peak_reserved_gib'] < card.total_memory / 2**30 * .8
              and abs(r['loss'] - baseline['loss']) < .002]
    best = min(usable, key=lambda r:r['median_seconds'], default=None)
    return dict(source_update=update, sequence_steps=sequence_steps, entity_count=e.shape[-2],
                activation_checkpointing=activation_checkpointing,
                scope=f'fixed {network} forward/backward, same frames for each profile',
                gpu=card.name, results=reports, suggested=best,
                suggested_frame_microbatch=best['frame_microbatch'] if best else None)


def sweep_env(candidates, lanes):
    assert_no_trainer()
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.cache/vendor/crforge/python')]
    import numpy as np
    from cr_rl.envs.crforge import CrForgeVector
    from cr_rl.training.state_selfplay import _extract_decks

    _, _, decks, forms = _extract_decks(ROOT/'prepared/source_league.yaml')
    results = []
    affinity = len(os.sched_getaffinity(0))
    for threads in sorted({min(t, affinity, lanes) for t in candidates}):
        rng = np.random.default_rng(20261120)
        with CrForgeVector(lanes, threads, independent=True, decision_ticks=8) as env:
            env.set_autoreset(False)
            for lane in range(lanes):
                left, right = rng.integers(len(decks), size=2)
                env.vector.reset_env(lane, 10300+lane, decks[left], decks[right], forms[left], forms[right])
            samples = 0
            elapsed = 0.0
            for step in range(144):
                active = np.flatnonzero(env.dones == 0)
                env.actions[:] = 0
                if step % 8 == 0:
                    for lane in active:
                        for seat in range(2):
                            legal = np.flatnonzero(env.masks[lane, seat])
                            env.actions[lane, seat] = rng.choice(legal)
                begin = time.perf_counter()
                env.vector.step_lanes(active)
                if step >= 16:
                    elapsed += time.perf_counter() - begin
                    samples += len(active)
            faults = sum(bool(env.vector.info(lane).get('fault', 0)) for lane in range(lanes))
        results.append(dict(env_threads=threads, lanes=lanes, seconds=elapsed,
                            lane_steps_per_second=samples / max(elapsed, 1e-9), faults=faults))
    usable = [r for r in results if r['faults'] == 0]
    best = max(usable, key=lambda r:r['lane_steps_per_second'], default=None)
    return dict(scope='native simulator stepping with fixed seeds', results=results,
                suggested_env_threads=best['env_threads'] if best else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'sweep', 'sweep-env'))
    parser.add_argument('--run', type=Path, default=ROOT/'runs/longrun')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--frames', type=int, nargs='+', default=[128, 256, 384, 512])
    parser.add_argument('--network', choices=('actor', 'critic'), default='actor')
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--sequence-steps', type=int, default=128)
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory-fraction', type=float, default=1.0)
    parser.add_argument('--env-threads', type=int, nargs='+', default=[4, 8, 12, 16])
    parser.add_argument('--lanes', type=int, default=48)
    args = parser.parse_args()
    if min(args.updates, args.lanes, args.sequence_steps, *args.frames, *args.env_threads,
           *(args.batch_sizes or [])) < 1:
        parser.error('Use positive counts')
    if args.batch_sizes and len(args.batch_sizes) != len(args.frames):
        parser.error('Pass one batch size per frame microbatch candidate')
    if not 0 < args.memory_fraction <= 1:
        parser.error('Use a memory fraction in (0, 1]')
    if args.mode == 'observe':
        result = observe(args.run, args.updates)
    elif args.mode == 'sweep-env':
        result = sweep_env(args.env_threads, args.lanes)
    else:
        result = sweep(args.checkpoint or args.run/'ppo_latest.pt', args.frames, network=args.network,
                       batch_sizes=args.batch_sizes, sequence_steps=args.sequence_steps,
                       activation_checkpointing=args.activation_checkpointing, memory_fraction=args.memory_fraction)
    print(json.dumps(result, indent=2), flush=True)
