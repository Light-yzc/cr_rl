"""Read-only timing ranking/calibration diagnostics on fixed real BC seats.

This never trains a model or publishes a calibrated policy. A single scalar
bias is selected using named training games, then held fixed on disjoint
validation games to separate ranking from a 0.5 operating threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np


def binary_auroc(labels, scores):
    """Mann-Whitney AUROC with average ranks, including exact score ties."""
    y, p = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return None
    order = np.argsort(p, kind='stable')
    sorted_values = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    start = 0
    while start < len(p):
        stop = start + 1
        while stop < len(p) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))


def binary_average_precision(labels, scores):
    """Threshold-group AP; equal-score rows enter together, independent of order."""
    y, p = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    if not positive:
        return None
    order = np.argsort(-p, kind='stable')
    y, p = y[order], p[order]
    cumulative = np.cumsum(y)
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    tp = cumulative[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) / positive * precision))


def logit_bias(probabilities, bias, choice=None):
    p = np.asarray(probabilities, dtype=np.float64)
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    shifted = 1 / (1 + np.exp(-(np.log(bounded) - np.log1p(-bounded) + float(bias))))
    return shifted if choice is None else np.where(choice, shifted, 0.)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _quantiles(values):
    return dict(zip(('q10', 'q50', 'q90'), map(float, np.quantile(values, [.1, .5, .9])))) if len(values) else None


def timing_report(labels, probabilities, choice, *, sampled_prior, target_prior=.2):
    y, p, choice = np.asarray(labels, dtype=bool), np.asarray(probabilities, dtype=np.float64), np.asarray(choice, dtype=bool)
    if not (y.shape == p.shape == choice.shape) or y.ndim != 1:
        raise ValueError('Timing labels/probability/choice must be equal one-dimensional arrays')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Timing probability must be finite and in [0,1]')
    if not 0 < sampled_prior < 1 or not 0 < target_prior < 1:
        raise ValueError('Target and global sampled priors must be in (0,1)')
    if np.any(y & ~choice):
        raise ValueError('An eligible action label cannot have only WAIT legal')
    if not len(y):
        raise ValueError('No eligible timing labels')
    if not y.any() or y.all():
        warnings.warn('Cohort has only one timing class; AUROC/balanced NLL may be unavailable', RuntimeWarning)
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    loss = np.where(y, -np.log(bounded), -np.log1p(-bounded))
    weights = np.where(y, target_prior / sampled_prior, (1 - target_prior) / (1 - sampled_prior))
    action, wait, choice_wait = y, ~y, (~y & choice)
    balanced = None if not action.any() or not wait.any() else float((loss[action].mean() + loss[wait].mean()) / 2)
    thresholds = {}
    for threshold in (.1, .2, .3, .5):
        predicted = p >= threshold
        recalled = int((predicted & action).sum())
        thresholds[str(threshold)] = {'action_recall': recalled / int(action.sum()) if action.any() else None,
            'choice_wait_false_action_rate': float(predicted[choice_wait].mean()) if choice_wait.any() else None,
            'precision': recalled / int(predicted.sum()) if predicted.any() else None,
            'predicted_action_rows': int(predicted.sum())}
    return {'rows': len(y), 'action_rows': int(action.sum()), 'wait_rows': int(wait.sum()),
        'choice_rows': int(choice.sum()), 'choice_wait_rows': int(choice_wait.sum()), 'forced_wait_rows': int((~choice).sum()),
        'actual_action_fraction': float(y.mean()), 'global_sampled_prior': sampled_prior, 'target_prior': target_prior,
        'balanced_timing_nll': balanced, 'natural_timing_nll': float(loss.mean()),
        'importance_weighted_timing_nll': float((loss * weights).mean()),
        'brier': float(((p - y) ** 2).mean()), 'choice_brier': _mean(((p - y) ** 2)[choice]),
        'all_auroc': binary_auroc(y, p), 'choice_auroc': binary_auroc(y[choice], p[choice]),
        'all_average_precision': binary_average_precision(y, p),
        'choice_average_precision': binary_average_precision(y[choice], p[choice]),
        'mean_probability_on_action': _mean(p[action]), 'mean_probability_on_wait': _mean(p[wait]),
        'mean_probability_on_choice_wait': _mean(p[choice_wait]),
        'probability_on_action': _quantiles(p[action]), 'probability_on_choice_wait': _quantiles(p[choice_wait]),
        'thresholds': thresholds}


def fit_scalar_bias(labels, probabilities, choice, sampled_prior, target_prior=.2):
    candidates = []
    for bias in np.linspace(-2, 2, 81):
        shifted = logit_bias(probabilities, bias, choice)
        score = timing_report(labels, shifted, choice, sampled_prior=sampled_prior, target_prior=target_prior)
        candidates.append((score['importance_weighted_timing_nll'], abs(float(bias)), float(bias)))
    loss, _, bias = min(candidates)
    return {'bias': bias, 'training_weighted_timing_nll': loss, 'grid': {'minimum': -2., 'maximum': 2., 'count': 81},
        'selection': 'Minimum global-prior-weighted timing NLL on the named training games only'}


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def check_subset(parent_path, subset_path, expected_fingerprint):
    if _digest(parent_path) != expected_fingerprint:
        raise ValueError('Original full manifest does not match checkpoint dataset fingerprint')
    parent, subset = json.loads(Path(parent_path).read_text()), json.loads(Path(subset_path).read_text())
    if parent['metadata_sha256'] != subset['metadata_sha256']:
        raise ValueError('Subset metadata differs from the checkpoint-bound original dataset')
    records = {item['path']: item for item in parent['shards']}
    for item in subset['shards']:
        original = records.get(item['path'])
        if original is None or any(item.get(field) != original.get(field) for field in ('sha256', 'rows', 'replays', 'replay_offset')):
            raise ValueError('Subset shard differs from the original immutable record')


def _aggregate(records, key, bias, sampled_prior, target_prior):
    def arrays(record):
        return record['labels'], logit_bias(record['probabilities'], bias, record['choice']) if bias else record['probabilities'], record['choice']
    selected = [record for record in records if record['partition'] == key]
    if not selected:
        raise ValueError('Missing diagnostic cohort: ' + key)
    all_arrays = [arrays(record) for record in selected]
    global_report = timing_report(*(np.concatenate([value[index] for value in all_arrays]) for index in range(3)),
        sampled_prior=sampled_prior, target_prior=target_prior)
    constant_p = np.where(np.concatenate([value[2] for value in all_arrays]), target_prior, 0.)
    constant = timing_report(np.concatenate([value[0] for value in all_arrays]), constant_p,
        np.concatenate([value[2] for value in all_arrays]), sampled_prior=sampled_prior, target_prior=target_prior)
    seats = []
    for record, value in zip(selected, all_arrays):
        seats.append({'replay_id': record['replay_id'], 'seat': record['seat'],
            **timing_report(*value, sampled_prior=sampled_prior, target_prior=target_prior)})
    average_fields = ['balanced_timing_nll', 'natural_timing_nll', 'importance_weighted_timing_nll',
        'brier', 'choice_brier', 'choice_auroc', 'choice_average_precision',
        'mean_probability_on_action', 'mean_probability_on_choice_wait']
    averages = {field: _mean([row[field] for row in seats if row[field] is not None]) for field in average_fields}
    return {'global_row_weighted': global_report, 'equal_weight_episode_seat': averages, 'constant_legal_probability_baseline': constant,
        'seats': seats, 'probability_bias': bias}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Immutable subset manifest referencing downloaded shards')
    parser.add_argument('--parent-manifest', type=Path, required=True, help='Original complete manifest bound to checkpoint')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-games', default='252,112')
    parser.add_argument('--validation-games', default='5,13,15,23,33,37,54,58')
    parser.add_argument('--validation-action-prior', type=float, default=4023 / 57493)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--frame-microbatch', type=int, default=32)
    parser.add_argument('--ram-limit-gib', type=float, default=6)
    args = parser.parse_args(argv)
    args.train_games = set(map(int, args.train_games.split(',')))
    args.validation_games = set(map(int, args.validation_games.split(',')))
    if args.train_games & args.validation_games or not args.train_games or not args.validation_games:
        parser.error('Training calibration and validation games must be nonempty and disjoint')
    if min(args.chunk_size, args.frame_microbatch) < 1 or not 0 < args.validation_action_prior < 1:
        parser.error('Positive chunk/frame batch and validation global prior in (0,1) required')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.models.state_policy import state_model_from_checkpoint
    from cr_rl.training.state_bc import ReplaySequenceDataset, prepare_state_chunk
    from cr_rl.training.bc import hierarchical_bc_loss
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Current trained-checkpoint diagnostic requires existing CUDA; no Torch installation attempted')
    torch.set_num_threads(2)
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    check_subset(args.parent_manifest, args.dataset, checkpoint['dataset_fingerprint'])
    if checkpoint.get('model_architecture_version') != 3 or checkpoint.get('decision_ticks') != 8:
        raise ValueError('Expected current architecture-v3 0.4-second checkpoint')
    saved_split = {int(key): value for key, value in checkpoint['split'].items()}
    if any(saved_split.get(game) != 'train' for game in args.train_games) or any(saved_split.get(game) != 'validation' for game in args.validation_games):
        raise ValueError('Named cohort games are not in their original checkpoint partitions')
    progress = dict(checkpoint['progress'])
    settings = dict(checkpoint['training_config'])
    model = state_model_from_checkpoint(checkpoint, args.device)
    model.frame_microbatch = args.frame_microbatch
    del checkpoint
    gc.collect()
    model.eval()
    model.requires_grad_(False)
    dataset = ReplaySequenceDataset(args.dataset, args.out.parent / '.calibration-replay-cache',
        expected_decision_ticks=8, ram_limit_gib=args.ram_limit_gib)
    required = args.train_games | args.validation_games
    if not required <= dataset.replay_ids:
        raise ValueError('Subset is missing requested diagnostic games')
    adapter = CrForgeObservationAdapter(dataset.schema, model.cfg.vocab_size, entity_bucket=16)
    subset_split = {game: 'probe' for game in required}
    records = []
    started = time.perf_counter()
    episodes = dataset.episodes(subset_split, 'probe')
    try:
        with torch.inference_mode():
            for episode in episodes:
                state = model.initial_state(1, device=args.device, dtype=torch.float16)
                probabilities, labels, choices = [], [], []
                head_metrics = {'loss': 0., 'timing_loss': 0., 'conditional_slot_loss': 0., 'position_loss': 0.}
                slot_correct = 0
                for start, rows in episode.chunks(args.chunk_size):
                    inputs = prepare_state_chunk(episode, rows, adapter, args.device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
                    if not torch.isfinite(logp).all():
                        raise FloatingPointError('Nonfinite actual policy output')
                    selected = episode.supervised[start:start + len(rows)]
                    if selected.any():
                        pick = torch.as_tensor(selected, device=args.device)
                        target_labels = torch.as_tensor(episode.labels[start:start + len(rows)], device=args.device)[pick]
                        lp = logp[pick].float()
                        prior = settings['sampled_action_prior'] if episode.replay_id in args.train_games else args.validation_action_prior
                        loss, components = hierarchical_bc_loss(lp, target_labels,
                            inputs['legal_action_mask'][pick], position_coef=settings['position_coef'],
                            tile_label_sigma=settings['tile_label_sigma'], conditional_card_loss=True,
                            normalization=episode.normalization,
                            timing_prior=settings['timing_target_action_prior'], sampled_action_prior=prior)
                        head_metrics['loss'] += float(loss.cpu())
                        for name in ('timing_loss', 'conditional_slot_loss', 'position_loss'):
                            head_metrics[name] += float(components[name].cpu())
                        play = (target_labels > 0) & (target_labels < 2305)
                        card = torch.logsumexp(lp[:, 1:2305].reshape(-1, 4, 576), -1)
                        slot_correct += int((card[play].argmax(-1) == (target_labels[play] - 1) // 576).sum())
                    # Stable 1-exp(wait) when WAIT is near certain.
                    p = -torch.expm1(logp[:, 0].float()).clamp(max=0)
                    probability = p.cpu().numpy()[selected]
                    choice = inputs['legal_action_mask'][:, 1:].any(-1).cpu().numpy()[selected]
                    probabilities.append(probability)
                    labels.append(episode.labels[start:start + len(rows)][selected] != 0)
                    choices.append(choice)
                    state = model.detach_state(state)
                records.append({'replay_id': episode.replay_id, 'seat': episode.seat,
                    'partition': 'train' if episode.replay_id in args.train_games else 'validation',
                    'probabilities': np.concatenate(probabilities), 'labels': np.concatenate(labels),
                    'choice': np.concatenate(choices)})
                head_metrics['slot_accuracy'] = slot_correct / max(1, episode.normalization.play)
                records[-1]['head_metrics'] = head_metrics
                print(json.dumps({'phase': 'calibration_collect', 'replay_id': episode.replay_id,
                    'seat': episode.seat, 'rows': len(records[-1]['labels']), 'seats_done': len(records)}), flush=True)
                del state
    finally:
        episodes.close()
    train = [row for row in records if row['partition'] == 'train']
    fitted = fit_scalar_bias(np.concatenate([row['labels'] for row in train]),
        np.concatenate([row['probabilities'] for row in train]), np.concatenate([row['choice'] for row in train]),
        sampled_prior=settings['sampled_action_prior'], target_prior=settings['timing_target_action_prior'])
    report = {'checkpoint_progress': progress, 'model_unchanged': True, 'optimizer_created': False,
        'protocol': 'Read-only full chronological inference; scalar calibration fit on train games then frozen on original validation games',
        'target_action_prior': settings['timing_target_action_prior'],
        'train_games': sorted(args.train_games), 'validation_games': sorted(args.validation_games),
        'calibration': fitted, 'seconds': time.perf_counter() - started,
        'limitation': 'Small fixed validation cohort, not whole validation set or native evaluation. '
            'Post-hoc bias is a diagnostic of threshold/calibration, not a production policy change. '
            'Threshold reports include skills; ranking metrics exclude forced WAIT in their choice_* variants.'}
    # Importance-weighted timing is a different class prior. This inverse is
    # algebraic, fitted on no validation labels, and is diagnostic only.
    q = settings['sampled_action_prior']
    target = settings['timing_target_action_prior']
    inverse_prior_bias = float(np.log(q / (1 - q)) - np.log(target / (1 - target)))
    report['inverse_training_prior_bias'] = inverse_prior_bias
    report['validation_head_metrics_equal_seat'] = {
        name: float(np.mean([record['head_metrics'][name] for record in records if record['partition'] == 'validation']))
        for name in ('loss', 'timing_loss', 'conditional_slot_loss', 'position_loss', 'slot_accuracy')}
    for tag, games, prior in [('train', args.train_games, settings['sampled_action_prior']),
                              ('validation', args.validation_games, args.validation_action_prior)]:
        report[tag + '_raw'] = _aggregate(records, tag, 0., prior, settings['timing_target_action_prior'])
        report[tag + '_frozen_bias'] = _aggregate(records, tag, fitted['bias'], prior, settings['timing_target_action_prior'])
        report[tag + '_inverse_training_prior'] = _aggregate(records, tag, inverse_prior_bias, prior, settings['timing_target_action_prior'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['checkpoint_progress', 'calibration', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
