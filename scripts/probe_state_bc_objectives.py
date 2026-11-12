"""Matched local whole-model fitting probes; no fitted weights are saved.

Saved-Adam joint, fresh-Adam joint, and fresh-Adam tile-only arms separate the
optimizer history from the objective. All process identical complete games.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from diagnose_state_bc_generalization import atomic_json, digest
from state_bc_probe_common import differentiable_components, position_loss_from_logits
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_bc import ReplaySequenceDataset, check_ram_budget, prepare_state_chunk, run_state_episode


def load_model_optimizer(path, arm, learning_rate):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model = state_model_from_checkpoint(checkpoint, 'cuda')
    model.frame_microbatch = 32
    settings = dict(checkpoint['training_config'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
        weight_decay=settings['weight_decay'], fused=True)
    if arm == 'joint_saved':
        saved = {**checkpoint['optimizer'], 'param_groups': [
            {**group, 'lr': learning_rate, 'fused': True, 'foreach': False}
            for group in checkpoint['optimizer']['param_groups']]}
        optimizer.load_state_dict(saved)
        for parameter, values in optimizer.state.items():
            if isinstance(values.get('step'), torch.Tensor):
                values['step'] = values['step'].to(parameter.device, dtype=torch.float32)
        del saved
    del checkpoint
    gc.collect()
    return model, optimizer, settings


def fit_episode(model, optimizer, scaler, episode, adapter, settings, objective, ram_limit):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    state = model.initial_state(1, device='cuda', dtype=torch.float16)
    totals = dict.fromkeys(('timing', 'card', 'tile'), 0.)
    started = time.perf_counter()
    raw_tiles = []
    original_placement = model.placement_logits
    if objective == 'tile':
        def capture_tiles(*args, **kwargs):
            value = original_placement(*args, **kwargs)
            raw_tiles.append(value)
            return value
        model.placement_logits = capture_tiles
    for start, rows in episode.chunks(32):
        raw_tiles.clear()
        check_ram_budget(ram_limit)
        inputs = prepare_state_chunk(episode, rows, adapter, 'cuda')
        selected_np = episode.supervised[start:start + len(rows)]
        labels_np = episode.labels[start:start + len(rows)]
        inputs['placement_mask'] = torch.as_tensor(selected_np & (labels_np > 0) & (labels_np < 2305), device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            logp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
            if selected_np.any():
                selected = torch.as_tensor(selected_np, device='cuda')
                labels = torch.as_tensor(labels_np[selected_np], device='cuda')
                parts = differentiable_components(logp[selected].float(), labels,
                    inputs['legal_action_mask'][selected], episode.normalization,
                    sigma=settings['tile_label_sigma'], target_prior=settings['timing_target_action_prior'],
                    sampled_prior=settings['sampled_action_prior'])
                loss = settings['position_coef'] * parts['tile']
                if objective == 'joint':
                    loss = loss + parts['timing'] + parts['card']
                else:
                    plays = (labels > 0) & (labels < 2305)
                    if plays.any():
                        direct = position_loss_from_logits(torch.cat(raw_tiles), labels[plays],
                            inputs['legal_action_mask'][selected][plays], play_denominator=episode.normalization.play,
                            sigma=settings['tile_label_sigma'])
                        if not torch.allclose(direct.detach(), parts['tile'].detach(), atol=1e-5, rtol=1e-5):
                            raise ValueError('Direct placement-only graph differs from production loss')
                        parts['tile'] = direct
                        loss = settings['position_coef'] * direct
                    else:
                        loss = None
            else:
                parts, loss = None, None
        if loss is not None:
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite bounded-probe loss')
            scaler.scale(loss).backward()
            for key, value in parts.items():
                totals[key] += float(value.detach())
        state = model.detach_state(state)
    if objective == 'tile':
        model.placement_logits = original_placement
    del state
    scaler.unscale_(optimizer)
    raw_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'],
                                                   error_if_nonfinite=False))
    if not math.isfinite(raw_norm):
        # Reject before optimizer.step even if only the aggregate norm overflowed.
        optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=max(1., scaler.get_scale() / 2))
        return {'stepped': False, 'nonfinite_rejected': True, 'grad_norm': None,
                'parts': totals, 'seconds': time.perf_counter() - started}
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= previous_scale
    if stepped:
        model.mark_weights_updated()
    return {'stepped': stepped, 'nonfinite_rejected': not stepped, 'grad_norm': raw_norm,
            'parts': totals, 'seconds': time.perf_counter() - started}


def evaluate(model, dataset, assignments, adapter, settings, partition, ram_limit):
    # All natural timing objectives retain the original full-partition priors.
    priors = {'train': settings['sampled_action_prior'], 'validation': 4023 / 57493,
              'heldout': 5176 / 69453}
    records = []
    episodes = dataset.episodes(assignments, partition)
    try:
        for episode in episodes:
            metrics = run_state_episode(model, episode, adapter, device='cuda', precision='fp16',
                chunk_size=32, position_coef=settings['position_coef'], tile_label_sigma=settings['tile_label_sigma'],
                timing_objective=settings['timing_objective'], timing_target_action_prior=settings['timing_target_action_prior'],
                sampled_action_prior=priors[partition], ram_limit_gib=ram_limit, defer_metrics=True)
            records.append(metrics)
    finally:
        episodes.close()
    keys = ('loss', 'timing_loss', 'conditional_slot_loss', 'position_loss', 'slot_accuracy', 'tile_mean_distance')
    return {'seats': len(records), 'equal_seat': {key: float(np.mean([row[key] for row in records])) for key in keys},
            'per_seat': records}


def gradient_alignment(model, optimizer, episode, adapter, settings, ram_limit):
    """Whole-seat gradients by head, with TBPTT boundaries and saved Adam direction."""
    model.train()
    names, params = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
    gradients = {key: [None] * len(params) for key in ('timing', 'card', 'tile')}
    state = model.initial_state(1, device='cuda', dtype=torch.float16)
    for start, rows in episode.chunks(32):
        check_ram_budget(ram_limit)
        inputs = prepare_state_chunk(episode, rows, adapter, 'cuda')
        selected_np = episode.supervised[start:start + len(rows)]
        labels_np = episode.labels[start:start + len(rows)]
        inputs['placement_mask'] = torch.as_tensor(selected_np & (labels_np > 0) & (labels_np < 2305), device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            lp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
            if selected_np.any():
                selected = torch.as_tensor(selected_np, device='cuda')
                parts = differentiable_components(lp[selected].float(), torch.as_tensor(labels_np[selected_np], device='cuda'),
                    inputs['legal_action_mask'][selected], episode.normalization, sigma=settings['tile_label_sigma'],
                    target_prior=settings['timing_target_action_prior'], sampled_prior=settings['sampled_action_prior'])
        if selected_np.any():
            for index, (key, loss) in enumerate(parts.items()):
                result = torch.autograd.grad(loss * 32., params, allow_unused=True, retain_graph=index < 2)
                for i, value in enumerate(result):
                    if value is None:
                        continue
                    if gradients[key][i] is None:
                        gradients[key][i] = value.detach().float().clone() / 32.
                    else:
                        gradients[key][i].add_(value.detach().float() / 32.)
        state = model.detach_state(state)
    del state
    def group(name):
        if name.startswith('frame.'):
            return 'frame'
        if name.startswith(('temporal.', 'to_temporal.')):
            return 'temporal'
        return 'heads'
    def product(left, right, selected_group=None):
        terms = []
        for name, a, b in zip(names, left, right):
            if a is not None and b is not None and (selected_group is None or group(name) == selected_group):
                terms.append((a.double() * b.double()).sum())
        return float(torch.stack(terms).sum()) if terms else 0.
    result = {'replay_id': episode.replay_id, 'seat': episode.seat, 'head_gradients': {}}
    for g in (None, 'frame', 'temporal', 'heads'):
        values = {}
        for key in gradients:
            values[key + '_norm'] = math.sqrt(max(0, product(gradients[key], gradients[key], g)))
        for a, b in (('timing', 'card'), ('timing', 'tile'), ('card', 'tile')):
            denominator = values[a + '_norm'] * values[b + '_norm']
            values[a + '_vs_' + b + '_cosine'] = product(gradients[a], gradients[b], g) / denominator if denominator else None
        result['head_gradients'][g or 'all'] = values
    joint = []
    for index in range(len(params)):
        values = [gradients[key][index] * (settings['position_coef'] if key == 'tile' else 1.)
                  for key in gradients if gradients[key][index] is not None]
        joint.append(sum(values) if values else None)
    norm = math.sqrt(product(joint, joint))
    projections = {}
    param_to_group = {id(p): group for group in optimizer.param_groups for p in group['params']}
    for clip in (1., 5.):
        dots = dict.fromkeys(gradients, 0.)
        update_squared = 0.
        for index, (parameter, value) in enumerate(zip(params, joint)):
            if value is None:
                continue
            h = param_to_group[id(parameter)]
            b1, b2 = h['betas']
            state_values = optimizer.state.get(parameter, {})
            old_m = state_values.get('exp_avg', torch.zeros_like(parameter))
            old_v = state_values.get('exp_avg_sq', torch.zeros_like(parameter))
            step = int(state_values.get('step', 0)) + 1
            clipped = value * min(1., clip / (norm + 1e-6))
            m = old_m * b1 + clipped * (1 - b1)
            v = old_v * b2 + clipped.square() * (1 - b2)
            update = -h['lr'] * (m / (1 - b1**step) /
                ((v / (1 - b2**step)).sqrt() + h['eps']) + h['weight_decay'] * parameter)
            update_squared += float(update.double().square().sum())
            for key in gradients:
                if gradients[key][index] is not None:
                    dots[key] += float((gradients[key][index].double() * update.double()).sum())
        projections[str(clip)] = {'head_gradient_dot_hypothetical_adam_delta': dots,
                                 'parameter_delta_norm': math.sqrt(update_squared)}
    result['joint_raw_norm'] = norm
    result['hypothetical_adam_step'] = projections
    result['note'] = 'Negative dot predicts local loss decrease to first order; does not guarantee finite-step improvement'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', default='joint_saved,joint_fresh,tile_fresh')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--eval-steps', default='16,32,64,128,256')
    parser.add_argument('--gradient-games', default='', help='Optional original train IDs; seat0, zero optimization')
    parser.add_argument('--ram-limit-gib', type=float, default=6.)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the existing CUDA environment; install nothing')
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error('Positive update count and learning rate required')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    cohort = json.loads(args.cohort.read_text())
    assignments = {int(game): part for part, values in cohort['partitions'].items() for game in values['game_ids']}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint['dataset_fingerprint'] != digest(args.parent_manifest):
        raise ValueError('Parent dataset is not bound to checkpoint')
    if any(checkpoint['split'][str(game)] != part for game, part in assignments.items()):
        raise ValueError('Diagnostic cohort crossed the immutable split')
    del checkpoint
    dataset = ReplaySequenceDataset(args.dataset, args.out / '.replay-cache', ram_limit_gib=args.ram_limit_gib)
    checkpoint_sha = digest(args.checkpoint)
    arms = args.arms.split(',')
    if any(arm not in ('joint_saved', 'joint_fresh', 'tile_fresh') for arm in arms):
        raise ValueError('Unknown probe arm')
    evaluation_steps = {0, args.steps} | set(map(int, args.eval_steps.split(',')))
    protocol = {'checkpoint_sha256': checkpoint_sha, 'cohort_sha256': digest(args.cohort),
        'learning_rate': args.learning_rate, 'steps': args.steps, 'arms': arms,
        'scope': 'Matched local diagnostic; no fitted checkpoint saved, no cloud modifications',
        'fresh_arms': 'Identical fresh AdamW/scaler, all parameters eligible; only objective differs',
        'sample_clock': 'One optimizer update per complete seat; identical deterministic exposure in each arm'}
    atomic_json(args.out / 'protocol.json', protocol)
    results = {'protocol': protocol, 'arms': {}, 'gradient_alignment': []}
    result_path = args.out / 'result.json'
    for arm in arms:
        torch.manual_seed(20261111)
        model, optimizer, settings = load_model_optimizer(args.checkpoint, arm, args.learning_rate)
        adapter = CrForgeObservationAdapter(dataset.schema, model.cfg.vocab_size, entity_bucket=16)
        scaler = torch.amp.GradScaler('cuda', init_scale=32)
        record = {'learning_curve': [], 'updates': [], 'seconds': 0.}
        results['arms'][arm] = record
        arm_started = time.perf_counter()
        if arm == 'joint_saved' and args.gradient_games:
            wanted = set(map(int, args.gradient_games.split(',')))
            iterator = dataset.episodes(assignments, 'train', seed=20261111)
            try:
                for episode in iterator:
                    if episode.replay_id in wanted and episode.seat == 0:
                        result = gradient_alignment(model, optimizer, episode, adapter, settings, args.ram_limit_gib)
                        results['gradient_alignment'].append(result)
                        atomic_json(result_path, results)
                        print(json.dumps({'phase': 'gradient_alignment', **result}), flush=True)
            finally:
                iterator.close()
        def evaluate_point(step):
            point = {'update': step, 'train': evaluate(model, dataset, assignments, adapter, settings, 'train', args.ram_limit_gib),
                     'validation': evaluate(model, dataset, assignments, adapter, settings, 'validation', args.ram_limit_gib)}
            if step == args.steps:
                point['heldout'] = evaluate(model, dataset, assignments, adapter, settings, 'heldout', args.ram_limit_gib)
            record['learning_curve'].append(point)
            atomic_json(result_path, results)
            print(json.dumps({'phase': 'objective_evaluate', 'arm': arm, 'update': step,
                **{part: value['equal_seat'] for part, value in point.items() if part != 'update'}}), flush=True)
        evaluate_point(0)
        step, epoch, attempts = 0, 0, 0
        while step < args.steps:
            episodes = dataset.episodes(assignments, 'train', epoch=epoch, seed=20261111)
            try:
                for episode in episodes:
                    row = fit_episode(model, optimizer, scaler, episode, adapter, settings,
                        'tile' if arm == 'tile_fresh' else 'joint', args.ram_limit_gib)
                    attempts += 1
                    step += int(row['stepped'])
                    record['updates'].append({'update': step, 'attempt': attempts,
                        'replay_id': episode.replay_id, 'seat': episode.seat, **row})
                    if attempts > args.steps + 16:
                        raise FloatingPointError('Too many rejected optimizer steps in diagnostic')
                    if step and step in evaluation_steps and row['stepped']:
                        evaluate_point(step)
                    if step % 16 == 0 or not row['stepped']:
                        atomic_json(result_path, results)
                        print(json.dumps({'phase': 'objective_train', 'arm': arm, 'update': step,
                            'parts': row['parts'], 'grad_norm': row['grad_norm'], 'seconds': row['seconds']}), flush=True)
                    if step >= args.steps:
                        break
            finally:
                episodes.close()
            epoch += 1
        record['seconds'] = time.perf_counter() - arm_started
        record['peak_vram_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        del model, optimizer, scaler, adapter
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(result_path, results)
    results['completed'] = True
    results['checkpoint_sha256_after'] = digest(args.checkpoint)
    if results['checkpoint_sha256_after'] != checkpoint_sha:
        raise ValueError('Original checkpoint changed')
    atomic_json(result_path, results)
    print(json.dumps({'phase': 'objectives_complete', 'result': str(result_path)}), flush=True)


if __name__ == '__main__':
    main()
