"""CPU-only summary of frozen state-BC cross-game inference artifacts.

No checkpoint is loaded, no model is trained, and no cloud file is touched.
Bootstrap units are complete games, preserving both seats when both qualify.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from diagnose_state_bc_calibration import timing_report


PARTITIONS = ('train', 'validation', 'heldout')
FREQUENCY_BINS = {
    'unseen_0': (0, 0), 'rare_1_9': (1, 9), 'low_10_49': (10, 49),
    'medium_50_199': (50, 199), 'common_ge200': (200, float('inf')),
}
PLAY_KEYS = ('slot_correct', 'uniform_slot_accuracy', 'tile_ce', 'target_entropy',
    'tile_hard_nll', 'tile_top1', 'tile_within2', 'tile_distance',
    'uniform_tile_ce', 'prior_ce_0.01', 'prior_ce_1.0')
TIMING_KEYS = ('natural_timing_nll', 'balanced_timing_nll',
    'importance_weighted_timing_nll', 'choice_auroc', 'choice_average_precision',
    'mean_probability_on_action', 'mean_probability_on_choice_wait', 'choice_brier')


def average(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def cluster_mean(values, game_ids, *, rng, repetitions):
    """Equal-seat estimate with a whole-game cluster percentile interval."""
    if not len(values):
        return {'estimate': None, 'ci95': None, 'equal_game_ci95': None, 'games': 0, 'seats': 0}
    grouped = defaultdict(list)
    for game, value in zip(game_ids, values):
        grouped[int(game)].append(float(value))
    clusters = list(grouped.values())
    totals = np.asarray([sum(cluster) for cluster in clusters])
    counts = np.asarray([len(cluster) for cluster in clusters])
    result = {'estimate': float(totals.sum() / counts.sum()),
        'equal_game_estimate': float(np.mean(totals / counts)),
        'games': len(clusters), 'seats': len(values), 'ci95': None, 'equal_game_ci95': None,
        'method': 'Percentile whole-game cluster bootstrap of equal-seat mean; selected seats retained together'}
    if len(clusters) >= 2:
        draw = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
        samples = totals[draw].sum(-1) / counts[draw].sum(-1)
        result['ci95'] = list(map(float, np.quantile(samples, [.025, .975])))
        game_samples = (totals / counts)[draw].mean(-1)
        result['equal_game_ci95'] = list(map(float, np.quantile(game_samples, [.025, .975])))
    return result


def independent_gap(left, right, key, *, rng, repetitions):
    """The right-minus-left gap; independently resample each disjoint partition."""
    def samples(records):
        grouped = defaultdict(list)
        for record in records:
            grouped[record['row']['replay_id']].append(record['row']['head_metrics'][key])
        totals = np.asarray([sum(values) for values in grouped.values()])
        counts = np.asarray([len(values) for values in grouped.values()])
        if not len(totals):
            return None, None
        estimate = float(totals.sum() / counts.sum())
        if len(totals) < 2:
            return estimate, None
        draws = rng.integers(0, len(totals), size=(repetitions, len(totals)))
        return estimate, totals[draws].sum(-1) / counts[draws].sum(-1)
    a, aa = samples(left)
    b, bb = samples(right)
    return {'right_minus_left': None if a is None or b is None else b - a,
        'ci95': None if aa is None or bb is None else list(map(float, np.quantile(bb - aa, [.025, .975]))),
        'method': 'Independent whole-game cluster bootstrap; no pairing across disjoint partitions'}


def prior_differences(records, selections, *, rng, repetitions):
    result = {}
    for alpha in ('0.01', '1.0'):
        values, game_ids = [], []
        for record, pick in zip(records, selections):
            if not pick.any():
                continue
            plays = record['plays']
            values.append(float(np.mean(plays['tile_ce'][pick] - plays['prior_ce_' + alpha][pick])))
            game_ids.append(record['row']['replay_id'])
        result['model_minus_prior_alpha_' + alpha] = cluster_mean(values, game_ids,
            rng=rng, repetitions=repetitions)
    return result


def play_summary(records, selections=None, *, rng, repetitions):
    if selections is None:
        selections = [np.ones(len(record['plays']['tile_ce']), dtype=bool) for record in records]
    usable = [(record, pick) for record, pick in zip(records, selections) if pick.any()]
    total = sum(int(pick.sum()) for _, pick in usable)
    equal_seat = {key: average([float(np.mean(record['plays'][key][pick])) for record, pick in usable])
        for key in PLAY_KEYS}
    pooled = {key: (sum(float(record['plays'][key][pick].sum()) for record, pick in usable) / total
        if total else None) for key in PLAY_KEYS}
    for metrics in (equal_seat, pooled):
        metrics['tile_ce_minus_target_entropy'] = (metrics['tile_ce'] - metrics['target_entropy']
            if metrics['tile_ce'] is not None else None)
    return {'games': len({record['row']['replay_id'] for record, _ in usable}),
        'seats_with_labels': len(usable), 'play_rows': total,
        'equal_nonempty_seat_means': equal_seat, 'pooled_play_row_weighted': pooled,
        'paired_prior_difference_cluster_ci': prior_differences(records, selections, rng=rng, repetitions=repetitions),
        'small_sample_note': ('Fewer than 100 labels; descriptive only, wide uncertainty likely' if total < 100 else None)}


def summarize_group(records, prior, target, *, rng, repetitions):
    if not records:
        return {'games': 0, 'seats': 0, 'available': False}
    timing_arrays = {key: np.concatenate([record['timing'][key] for record in records])
        for key in ('labels', 'probabilities', 'choice')}
    reports = [timing_report(record['timing']['labels'], record['timing']['probabilities'],
        record['timing']['choice'], sampled_prior=prior, target_prior=target) for record in records]
    micro = timing_report(timing_arrays['labels'], timing_arrays['probabilities'], timing_arrays['choice'],
        sampled_prior=prior, target_prior=target)
    q = timing_arrays['choice']
    labels, probabilities = timing_arrays['labels'][q], timing_arrays['probabilities'][q]
    bounded = np.clip(probabilities, 1e-12, 1 - 1e-12)
    micro['choice_natural_timing_nll'] = (float(np.where(labels, -np.log(bounded),
        -np.log1p(-bounded)).mean()) if len(labels) else None)
    micro['choice_action_prevalence_ap_baseline'] = float(labels.mean()) if len(labels) else None
    head_keys = ('loss', 'timing_loss', 'conditional_slot_loss', 'position_loss')
    heads = {key: average([record['row']['head_metrics'][key] for record in records]) for key in head_keys}
    def weighted(key, count):
        denominator = sum(record['row'][count] for record in records)
        return (sum(record['row']['head_metrics'][key] * record['row'][count] for record in records) /
            denominator) if denominator else None
    sample_heads = {'timing_loss_rows_weighted': weighted('timing_loss', 'valid_rows'),
        'conditional_card_nll_action_rows_weighted': weighted('conditional_slot_loss', 'action_rows'),
        'position_loss_play_rows_weighted': weighted('position_loss', 'play_rows')}
    return {'available': True, 'games': len({record['row']['replay_id'] for record in records}),
        'seats': len(records), 'rows': sum(record['row']['valid_rows'] for record in records),
        'action_rows': sum(record['row']['action_rows'] for record in records),
        'play_rows': sum(record['row']['play_rows'] for record in records),
        'trainer_equal_seat_head_metrics': heads, 'sample_weighted_head_metrics': sample_heads,
        'ordinary_slot_accuracy_equal_seat': average([record['row']['slot_accuracy'] for record in records]),
        'timing_pooled_row_weighted': micro,
        'timing_equal_seat_means': {key: average([report[key] for report in reports]) for key in TIMING_KEYS},
        'ordinary_play_conditional_metrics': play_summary(records, rng=rng, repetitions=repetitions)}


def load_records(root, coverage, protocol):
    by_seat = {(int(row['replay_id']), int(row['seat'])): row for row in coverage['seats']}
    unique = {}
    for line in (root / 'seat_metrics.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = row['replay_id'], row['seat']
        if row['checkpoint_sha256'] != protocol['checkpoint_sha256']:
            raise ValueError('Mixed checkpoint SHA in inference seat log')
        if key in unique and unique[key] != row:
            raise ValueError('Nonidentical duplicate seat metric record')
        unique[key] = row
    result = []
    for key, row in sorted(unique.items()):
        cov = by_seat[key]
        if row['partition'] != cov['partition'] or sorted(row['deck']) != sorted(cov['deck']):
            raise ValueError('Coverage and inference seat identity disagree')
        with np.load(root / 'seats' / f'replay_{key[0]:06d}_seat{key[1]}.npz', allow_pickle=False) as source:
            timing = {name: np.array(source['timing_' + name], copy=True)
                for name in ('labels', 'probabilities', 'choice')}
            plays = {name: np.array(source['play_' + name], copy=True) for name in (*PLAY_KEYS, 'cards', 'costs')}
        if len(timing['labels']) != row['valid_rows'] or len(plays['cards']) != row['play_rows']:
            raise ValueError('Seat raw metric array count differs from reported supervision counts')
        if int(timing['labels'].sum()) != row['action_rows'] or int(timing['choice'].sum()) != row['choice_rows']:
            raise ValueError('Seat raw timing labels differ from summary counts')
        if any(not np.isfinite(value).all() for value in (*timing.values(), *plays.values())):
            raise ValueError('Raw diagnostic metric contains a nonfinite value')
        if len(plays['tile_ce']) and abs(plays['tile_ce'].mean() - row['head_metrics']['position_loss']) > 1e-4:
            raise ValueError('Raw conditional placement CE disagrees with trainer-compatible loss')
        result.append({'row': row, 'coverage': cov, 'timing': timing, 'plays': plays})
    return result


def shared_card_standardization(train, target):
    """Descriptive equal-card comparison; does not control within-card states."""
    def by_card(records):
        result = defaultdict(lambda: defaultdict(list))
        for record in records:
            plays = record['plays']
            for card in np.unique(plays['cards']):
                pick = plays['cards'] == card
                for key in ('tile_ce', 'target_entropy', 'prior_ce_0.01', 'prior_ce_1.0', 'slot_correct'):
                    result[int(card)][key].append(plays[key][pick])
        return {card: {key: np.concatenate(parts) for key, parts in metrics.items()}
            for card, metrics in result.items()}
    left, right = by_card(train), by_card(target)
    common = sorted(left.keys() & right.keys())
    if not common:
        return {'shared_cards': 0, 'available': False}
    def mean_metrics(values):
        result = {key: float(np.mean([values[card][key].mean() for card in common]))
            for key in ('tile_ce', 'target_entropy', 'prior_ce_0.01', 'prior_ce_1.0', 'slot_correct')}
        result['model_minus_prior_alpha_0.01'] = result['tile_ce'] - result['prior_ce_0.01']
        result['model_minus_prior_alpha_1.0'] = result['tile_ce'] - result['prior_ce_1.0']
        return result
    a, b = mean_metrics(left), mean_metrics(right)
    return {'available': True, 'shared_cards': len(common),
        'train_labels': sum(len(left[card]['tile_ce']) for card in common),
        'target_labels': sum(len(right[card]['tile_ce']) for card in common),
        'weighting': 'Equal weight per shared physical card identity; within each card pool its play rows',
        'train_equal_card': a, 'target_equal_card': b,
        'target_minus_train_equal_card': {key: b[key] - a[key] for key in a},
        'card_label_counts': [{'card_id': card, 'train': len(left[card]['tile_ce']),
            'target': len(right[card]['tile_ce'])} for card in common],
        'limitation': 'Descriptive standardization only; differing state, cost, card form, opponents, and small per-card counts remain'}


def plot_summary(summary, path):
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parts = [part for part in PARTITIONS if summary['partitions'][part].get('available')]
    if not parts:
        return
    x = np.arange(len(parts))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), dpi=150)
    for shift, key, label, color in [(-.23, 'tile_ce', 'Model', '#2463a6'),
            (0, 'prior_ce_0.01', 'Train card prior (.01)', '#e18c36'),
            (.23, 'prior_ce_1.0', 'Train card prior (1)', '#53a479')]:
        values = [summary['partitions'][part]['ordinary_play_conditional_metrics']['equal_nonempty_seat_means'][key] for part in parts]
        axes[0].bar(x + shift, values, .23, label=label, color=color)
    floors = [summary['partitions'][part]['ordinary_play_conditional_metrics']['equal_nonempty_seat_means']['target_entropy'] for part in parts]
    axes[0].plot(x, floors, 'ko--', markersize=4, label='Soft-target entropy')
    axes[0].set(ylabel='Conditional placement CE (nats)', title='Placement: lower is better')
    axes[0].legend(fontsize=7)
    slot = [summary['partitions'][part]['ordinary_slot_accuracy_equal_seat'] for part in parts]
    uniform = [summary['partitions'][part]['ordinary_play_conditional_metrics']['equal_nonempty_seat_means']['uniform_slot_accuracy'] for part in parts]
    axes[1].bar(x - .16, np.asarray(slot) * 100, .32, label='Model', color='#2463a6')
    axes[1].bar(x + .16, np.asarray(uniform) * 100, .32, label='Uniform legal slot', color='#b8b8b8')
    axes[1].set(ylabel='Ordinary card slot accuracy (%)', title='Card selection: higher is better')
    axes[1].legend(fontsize=8)
    auc = [summary['partitions'][part]['timing_pooled_row_weighted']['choice_auroc'] for part in parts]
    axes[2].bar(x, auc, .55, color='#2463a6')
    axes[2].axhline(.5, color='#888888', linestyle='--', label='Random rank')
    axes[2].set(ylabel='Choice-only timing AUROC', ylim=(0, 1), title='Timing: forced WAIT excluded')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, parts)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    step = summary['checkpoint_progress']['optimizer_steps']
    fig.suptitle(f'Frozen state-BC checkpoint {step} | offline cross-game evaluation', fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--partition-audit', type=Path, default=Path(__file__).resolve().parents[1] /
        'artifacts/bc_diagnostic_20261110/supervision/bc100/audit.json')
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('At least 100 bootstrap repetitions required')
    protocol = json.loads((args.input / 'protocol.json').read_text())
    completion_path = args.input / 'completion.json'
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    if not args.allow_partial and not (completion.get('completed') and completion.get('full_dataset')):
        raise ValueError('Full inference completion required; use --allow-partial only for smoke summaries')
    if completion.get('checkpoint_sha256_after') not in (None, protocol['checkpoint_sha256']):
        raise ValueError('Checkpoint SHA changed during frozen inference')
    coverage = json.loads(args.coverage.read_text())
    if coverage['registry_provenance']['dataset_npz_sha256'] != protocol['source_sha256']:
        raise ValueError('Coverage audit is not bound to the inference source dataset')
    prior = json.loads((args.input / 'train_prior.json').read_text())
    audit = json.loads(args.partition_audit.read_text())
    priors = {part: audit['stats'][part + '/resampled']['valid_nonwait'] /
        audit['stats'][part + '/resampled']['valid_rows'] for part in PARTITIONS}
    records = load_records(args.input, coverage, protocol)
    grouped = {part: [record for record in records if record['row']['partition'] == part] for part in PARTITIONS}
    if not args.allow_partial:
        for part in PARTITIONS:
            if len(grouped[part]) != 2 * protocol['split_games'][part]:
                raise ValueError('Incomplete partition seats in completed inference')
    rng = np.random.default_rng(args.seed)
    target = protocol['training_config']['timing_target_action_prior']
    common = dict(rng=rng, repetitions=args.bootstrap)
    summary = {'checkpoint_progress': protocol['checkpoint_progress'],
        'checkpoint_sha256': protocol['checkpoint_sha256'], 'input_directory': str(args.input.resolve()),
        'full_dataset': completion.get('full_dataset', False), 'actual_seats': len(records),
        'aggregation_notes': {
            'trainer_equal_seat': 'Each chronological episode-seat has equal weight, matching logged validation objective',
            'conditional_card_nll': 'Ground-truth ACT labels, including abilities; not end-to-end joint action accuracy',
            'ordinary_slot_accuracy': 'Ground-truth ordinary play rows; argmax among four ordinary card slots',
            'placement': 'Ground-truth card slot conditional placement distribution; not predicted-card or native battle quality',
            'timing': 'Pooled row-weighted ranking and NLL plus equal-seat metrics. Primary ranking is choice-only',
            'timing_prior': 'Parent-partition global action prior; deck subgroup composition does not refit the objective',
            'bootstrap': f'{args.bootstrap} percentile repetitions, seed {args.seed}; independent unit is complete game',
            'prior': 'Both alpha=.01 and alpha=1 baselines fixed before this evaluation; no validation tuning',
            'frequencies': 'Card groups use valid training ordinary-play count only; cost uses recorded hand cost',
            'limitations': ['Single frozen step2274 snapshot, not necessarily latest cloud checkpoint',
                'Same original source and revision: tests cross-game and cross-deck, not cross-source robustness',
                'Train is resubstitution; card marginal priors are also fitted on all training labels',
                'Heldout was selected by eval deck presence in either seat, so both heldout seats are not necessarily unseen decks',
                'Labels are offline event-preserving 0.4s augmentation; this does not establish native battle win rate',
                'Raw seen/unseen deck differences are confounded by card composition, costs and game states',
                'Subgroups and multiple confidence intervals are exploratory; no multiple-comparison correction']},
        'partition_action_priors': priors,
        'partitions': {part: summarize_group(grouped[part], priors[part], target, **common) for part in PARTITIONS},
        'deck_groups': {}, 'card_frequency_groups': {}, 'cost_groups': {}, 'generalization_gaps': {},
        'shared_card_standardization': {}, 'validation_monitoring_groups': {}}
    fixed16_games = {5, 13, 15, 23, 33, 37, 54, 58}
    summary['validation_monitoring_groups'] = {
        'fixed16_monitored': summarize_group([record for record in grouped['validation']
            if record['row']['replay_id'] in fixed16_games], priors['validation'], target, **common),
        'remaining45_games': summarize_group([record for record in grouped['validation']
            if record['row']['replay_id'] not in fixed16_games], priors['validation'], target, **common)}
    for part in ('validation', 'heldout'):
        selections = {
            'physical_deck_seen_train': lambda cov: cov['physical_deck_seen_in_train'],
            'physical_deck_unseen_train': lambda cov: not cov['physical_deck_seen_in_train'],
            'nearest_train_deck_shared7': lambda cov: cov['max_shared_cards_with_train_deck'] == 7,
            'nearest_train_deck_shared_le6': lambda cov: cov['max_shared_cards_with_train_deck'] <= 6,
        }
        summary['deck_groups'][part] = {name: summarize_group(
            [record for record in grouped[part] if predicate(record['coverage'])], priors[part], target, **common)
            for name, predicate in selections.items()}
        summary['generalization_gaps'][part + '_minus_train'] = {
            key: independent_gap(grouped['train'], grouped[part], key, **common)
            for key in ('loss', 'timing_loss', 'conditional_slot_loss', 'position_loss')}
        summary['shared_card_standardization'][part + '_vs_train'] = shared_card_standardization(grouped['train'], grouped[part])
    card_counts = {int(key): int(value) for key, value in prior['card_counts'].items()}
    for part in PARTITIONS:
        values = grouped[part]
        frequencies = [np.asarray([card_counts.get(int(card), 0) for card in record['plays']['cards']]) for record in values]
        summary['card_frequency_groups'][part] = {name: play_summary(values,
            [(freq >= lower) & (freq <= upper) for freq in frequencies], **common)
            for name, (lower, upper) in FREQUENCY_BINS.items()}
        summary['cost_groups'][part] = {
            'cost_lt5': play_summary(values, [record['plays']['costs'] < 5 for record in values], **common),
            'cost_ge5': play_summary(values, [record['plays']['costs'] >= 5 for record in values], **common)}
    summary['resources'] = {'peak_reported_rss_gib': max((record['row']['rss_gib'] for record in records), default=None),
        'peak_reported_vram_gib': max((record['row']['vram_gib'] for record in records), default=None),
        'frozen_inference_seconds_sum': sum(record['row']['seconds'] for record in records)}
    checks = {}
    fixed = summary['validation_monitoring_groups']['fixed16_monitored']
    if fixed.get('seats') == 16:
        # Independent prior diagnostics preceded this frozen inference and used
        # the same immutable source labels and legal masks.
        checks['fixed16_prior_alpha_0.01_delta'] = (
            fixed['ordinary_play_conditional_metrics']['equal_nonempty_seat_means']['prior_ce_0.01'] - 4.8741724225923235)
        if protocol['checkpoint_sha256'] == '5944b51fcb0dd03ea4bf1680fe78a990e84024b59b2ef5fe50cd914d7abde2a7':
            checks['fixed16_loss_delta_vs_independent_calibration'] = fixed['trainer_equal_seat_head_metrics']['loss'] - 6.729520200869956
            checks['fixed16_tile_ce_delta_vs_independent_calibration'] = fixed['trainer_equal_seat_head_metrics']['position_loss'] - 5.117333528585732
    val = summary['partitions']['validation']
    if val.get('seats') == 106:
        checks['all_validation_prior_alpha_0.01_delta'] = (
            val['ordinary_play_conditional_metrics']['equal_nonempty_seat_means']['prior_ce_0.01'] - 4.958566390667232)
    checks['prior_deltas_pass_tolerance_1e-8'] = all(abs(value) < 1e-8
        for key, value in checks.items() if '_prior_' in key)
    checks['model_deltas_pass_tolerance_0.01'] = all(abs(value) < .01
        for key, value in checks.items() if '_calibration' in key)
    checks['precision_note'] = ('This evaluation enables TF32 high matmul precision; earlier calibration did not. '
        'About .003 nats fixed16 loss difference is a recomputation precision difference, not a weight update.')
    summary['independent_reference_verification'] = checks
    destination = args.out or args.input / 'summary.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if args.plot:
        plot_summary(summary, destination.with_suffix('.png'))
    print(json.dumps({'summary': str(destination), 'seats': len(records),
        'full_dataset': summary['full_dataset'], 'partition_seats': {key: len(value) for key, value in grouped.items()}}))


if __name__ == '__main__':
    main()
