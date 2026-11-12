"""CPU-only matched-experiment summary of local state-BC capacity probes.

Reads compact JSON results, never a checkpoint/feature cache. Only finished arms
enter the main table; optional partial mode records outstanding work explicitly.
Comparisons retain each experiment's own initialization, precision and exposure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PARTITIONS = ('train', 'validation', 'heldout')
METRICS = ('ce', 'entropy', 'excess_ce', 'within2', 'total_loss', 'card_nll',
    'timing_loss', 'slot_accuracy')
KNOWN_CHECKPOINT = '5944b51fcb0dd03ea4bf1680fe78a990e84024b59b2ef5fe50cd914d7abde2a7'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def nullable_mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def cluster_interval(values, game_ids, *, repetitions, rng):
    """Equal-seat paired effect; resample complete games with both seats kept."""
    if not len(values):
        return {'estimate': None, 'ci95': None, 'games': 0, 'seats': 0}
    grouped = defaultdict(list)
    for game, value in zip(game_ids, values):
        grouped[int(game)].append(float(value))
    sums = np.asarray([sum(value) for value in grouped.values()])
    counts = np.asarray([len(value) for value in grouped.values()])
    result = {'estimate': float(sums.sum()/counts.sum()), 'ci95': None,
        'equal_game_estimate': float((sums/counts).mean()), 'equal_game_ci95': None,
        'games': len(sums), 'seats': len(values)}
    if len(sums) > 1:
        index = rng.integers(0, len(sums), (repetitions, len(sums)))
        boot = sums[index].sum(-1)/counts[index].sum(-1)
        result['ci95'] = list(map(float, np.quantile(boot, [.025, .975])))
        game_boot = (sums/counts)[index].mean(-1)
        result['equal_game_ci95'] = list(map(float, np.quantile(game_boot, [.025, .975])))
    return result


def paired_effect(before, after, *, repetitions, rng):
    if before is None or after is None:
        return {'available': False, 'reason': 'No same-experiment initial evaluation for this partition'}
    a = {(row['replay_id'], row['seat']): row for row in before['seats']}
    b = {(row['replay_id'], row['seat']): row for row in after['seats']}
    if a.keys() != b.keys():
        raise ValueError('Paired comparison has different game-seat identities')
    if any(a[key]['plays'] != b[key]['plays'] for key in a):
        raise ValueError('Paired comparison has different ordinary-play label counts')
    report = {'available': True, 'direction': 'after minus before; negative CE/loss is improvement',
        'bootstrap_unit': 'Complete game; selected seats remain together', 'metrics': {}}
    for metric in METRICS:
        keys = [key for key in sorted(a) if a[key].get(metric) is not None and b[key].get(metric) is not None]
        if keys:
            report['metrics'][metric] = cluster_interval([b[key][metric]-a[key][metric] for key in keys],
                [key[0] for key in keys], repetitions=repetitions, rng=rng)
    return report


def normalize_head(point):
    output = {}
    for part in PARTITIONS:
        seats = []
        for row in point.get('seats', []):
            if row['partition'] != part:
                continue
            seats.append({'replay_id': int(row['replay_id']), 'seat': int(row['seat']),
                'plays': int(row['plays']), **{metric: row.get(metric) for metric in METRICS}})
        if seats:
            output[part] = {'seats': seats, 'games': len({row['replay_id'] for row in seats}),
                'plays': sum(row['plays'] for row in seats),
                'mean': {key: nullable_mean([row.get(key) for row in seats]) for key in METRICS}}
    return output


def normalize_objective(point, entropy_reference):
    output = {}
    for part in PARTITIONS:
        if part not in point:
            continue
        seats = []
        for row in point[part]['per_seat']:
            key = int(row['replay_id']), int(row['seat'])
            entropy = entropy_reference.get(key)
            ce = row.get('position_loss')
            seats.append({'replay_id': key[0], 'seat': key[1], 'plays': int(row['play_rows']),
                'ce': ce, 'entropy': entropy, 'excess_ce': None if entropy is None or ce is None else ce-entropy,
                'within2': None, 'total_loss': row.get('loss'), 'card_nll': row.get('conditional_slot_loss'),
                'timing_loss': row.get('timing_loss'), 'slot_accuracy': row.get('slot_accuracy')})
        if seats:
            output[part] = {'seats': seats, 'games': len({row['replay_id'] for row in seats}),
                'plays': sum(row['plays'] for row in seats),
                'mean': {key: nullable_mean([row.get(key) for row in seats]) for key in METRICS},
                'entropy_source': 'Same-cohort immutable soft targets from head baseline; not a new model measurement',
                'within2_unavailable': 'Whole-model result schema does not record this metric'}
    return output


def compact_parts(parts):
    return {part: {'games': value['games'], 'seats': len(value['seats']), 'plays': value['plays'],
        **value['mean']} for part, value in parts.items()}


def head_exposure(steps, rows, batch):
    rounds, partial = divmod(steps, math.ceil(rows/batch))
    return rounds*rows + min(rows, partial*batch)


def curve_rows(points, *, normalizer, rows=None, batch=None):
    curves = []
    for point in points:
        step = int(point.get('step', point.get('update', 0)))
        curves.append({'step': step, 'gt_play_labels_exposed': head_exposure(step, rows, batch) if rows else None,
            'partitions': compact_parts(normalizer(point))})
    return curves


def validate_identity(protocol, cohort_path, expected_checkpoint):
    if protocol.get('checkpoint_sha256') != expected_checkpoint:
        raise ValueError('Mixed or unexpected checkpoint in capacity experiment')
    if protocol.get('cohort_sha256') not in (None, sha256(cohort_path)):
        raise ValueError('Experiment was not run on the frozen cohort JSON')


def summarize_head(path, *, cohort_path, checkpoint, repetitions, rng, entropy_reference):
    data = json.loads(path.read_text())
    protocol = data['protocol']
    validate_identity(protocol, cohort_path, checkpoint)
    completion_path = path.parent/'completion.json'
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    if completion.get('checkpoint_sha256_after') not in (None, checkpoint):
        raise ValueError('Source checkpoint changed during experiment')
    arms = data.get('arms', [])
    if not isinstance(arms, list):
        raise ValueError('Expected head results arms list')
    rows, batch, target = int(protocol['training_rows']), int(protocol['batch_size']), int(protocol['steps'])
    full_head = 'full-placement' in protocol.get('scope', '') or path.parent.name.startswith('fullhead')
    data_prior = 'prior' in path.parent.name or any('prior' in arm['variant'] for arm in arms)
    family = 'data_prior_initialization' if data_prior else 'full_placement_head' if full_head else 'final_mlp_only'
    baseline_path = path.parent/'baseline.json'
    baseline_point = json.loads(baseline_path.read_text()) if baseline_path.exists() else arms[0]['metrics'][0] if arms else {}
    baseline = normalize_head(baseline_point)
    for value in baseline.values():
        for seat in value['seats']:
            key = seat['replay_id'], seat['seat']
            old = entropy_reference.setdefault(key, seat['entropy'])
            if old is not None and seat['entropy'] is not None and abs(old-seat['entropy']) > 1e-5:
                raise ValueError('Same-cohort soft-target entropy changed across head experiments')
    experiment = {'name': path.parent.name, 'family': family, 'result_path': str(path.resolve()),
        'result_sha256': sha256(path), 'complete': bool(completion.get('complete')),
        'scope': protocol.get('scope'), 'steps': target, 'batch_size': batch,
        'training_play_labels': rows, 'planned_play_exposures': head_exposure(target, rows, batch),
        'planned_action_epochs': head_exposure(target, rows, batch)/rows,
        'source_readout_version': protocol.get('source_readout_version', 2 if checkpoint == KNOWN_CHECKPOINT else None),
        'float32_matmul_precision': protocol.get('float32_matmul_precision', 'Earlier readout run used high'),
        'training_seconds_all_arms_observed': completion.get('seconds'),
        'extraction': {key: protocol.get('extraction', {}).get(key) for key in
            ('bytes', 'blocks', 'reused_from', 'raw_capture_retained')},
        'timing_qualification': 'Not an uncontended benchmark; GPU overlap and one-time extraction prevent cross-experiment speed claims',
        'baseline': compact_parts(baseline), 'candidates': [], 'pending_arms': []}
    internals = []
    observed_keys = {(arm['variant'],arm['learning_rate']) for arm in arms}
    for variant in protocol.get('variants') or []:
        for rate in protocol.get('learning_rates') or []:
            if (variant,rate) not in observed_keys:
                experiment['pending_arms'].append({'variant':variant,'learning_rate':rate,'last_step':None})
    for arm in arms:
        points = sorted(arm.get('metrics', []), key=lambda point: point['step'])
        initial = next((point for point in points if point['step'] == 0), None)
        endpoint = next((point for point in points if point['step'] >= target), None)
        if initial is None or endpoint is None:
            experiment['pending_arms'].append({'variant': arm['variant'], 'learning_rate': arm['learning_rate'],
                'last_step': points[-1]['step'] if points else None})
            continue
        before, after = normalize_head(initial), normalize_head(endpoint)
        candidate = {'id': f'{path.parent.name}:{arm["variant"]}:lr{arm["learning_rate"]:g}',
            'variant': arm['variant'], 'learning_rate': arm['learning_rate'], 'completed': True,
            'trainable_parameters': arm.get('trainable_parameters'), 'step': endpoint['step'],
            'gt_play_labels_exposed': head_exposure(endpoint['step'], rows, batch),
            'equivalent_action_epochs': head_exposure(endpoint['step'], rows, batch)/rows,
            'initial': compact_parts(before), 'final': compact_parts(after),
            'delta_vs_experiment_baseline': {part: paired_effect(baseline.get(part), after.get(part),
                repetitions=repetitions, rng=rng) for part in PARTITIONS},
            'delta_vs_own_initial': {part: paired_effect(before.get(part), after.get(part),
                repetitions=repetitions, rng=rng) for part in PARTITIONS},
            'curve': curve_rows(points, normalizer=normalize_head, rows=rows, batch=batch),
            'initial_function_ce_delta_vs_baseline': {part: before[part]['mean']['ce']-baseline[part]['mean']['ce']
                for part in before if part in baseline},
            'data_initialization': arm.get('data_initialization'),
            'initial_weight_audit': arm.get('initial_weight_audit'),
            'spatial_readout_initial': arm.get('spatial_readout_initial'),
            'spatial_readout_final': arm.get('spatial_readout_final'),
            'same_lr_original_comparison': None}
        experiment['candidates'].append(candidate)
        internals.append((candidate, before, after))
    for candidate, before, after in internals:
        original = next((item for item in internals if item[0]['variant'] == 'original' and
            item[0]['learning_rate'] == candidate['learning_rate'] and item[0]['step'] == candidate['step']), None)
        if original and candidate['variant'] != 'original':
            candidate['same_lr_original_comparison'] = {'reference_id': original[0]['id'],
                'note': 'Same experiment, original head, rate and exposure; after minus original endpoint',
                'partitions': {part: paired_effect(original[2].get(part), after.get(part),
                    repetitions=repetitions, rng=rng) for part in PARTITIONS}}
    extension_path = path.parent/'winner_extension.json'
    if extension_path.exists():
        extension = json.loads(extension_path.read_text())
        source = next((item for item in internals if item[0]['variant'] == extension['variant'] and
            item[0]['learning_rate'] == extension['learning_rate']), None)
        points = extension.get('metrics', [])
        planned = int(protocol.get('winner_extension_target_steps', target))
        done = source is not None and bool(points) and int(points[-1]['step']) >= planned
        experiment['winner_extension'] = {'completed': done, 'selection': extension.get('selection'),
            'selected_at_step': extension.get('selected_at_step'), 'target_steps': planned,
            'warning': 'Validation-selected longer continuation; not matched exposure against shorter arms'}
        if done:
            last, original_before = normalize_head(points[-1]), source[1]
            source_points = next(arm['metrics'] for arm in data['arms']
                if arm['variant'] == extension['variant'] and arm['learning_rate'] == extension['learning_rate'])
            continuation_points = [point for point in source_points
                if point['step'] < extension['selected_at_step']] + points
            experiment['winner_extension'].update({'variant': extension['variant'],
                'learning_rate': extension['learning_rate'], 'step': points[-1]['step'],
                'gt_play_labels_exposed': head_exposure(points[-1]['step'], rows, batch),
                'equivalent_action_epochs': head_exposure(points[-1]['step'], rows, batch)/rows,
                'final': compact_parts(last), 'delta_vs_own_initial': {part:
                    paired_effect(original_before.get(part), last.get(part), repetitions=repetitions, rng=rng)
                    for part in PARTITIONS},
                'curve': curve_rows(continuation_points, normalizer=normalize_head, rows=rows, batch=batch)})
    return experiment


def summarize_objectives(path, *, cohort_path, checkpoint, entropy_reference, repetitions, rng):
    data = json.loads(path.read_text())
    protocol = data['protocol']
    validate_identity(protocol, cohort_path, checkpoint)
    if data.get('checkpoint_sha256_after') not in (None, checkpoint):
        raise ValueError('Whole-model probe changed original checkpoint')
    cohort = json.loads(cohort_path.read_text())
    exposure_lookup = {(row['replay_id'], seat['seat']): seat
        for row in cohort['partitions']['train']['games'] for seat in row['seats']}
    experiment = {'name': path.parent.name, 'family': 'whole_model_objectives',
        'result_path': str(path.resolve()), 'result_sha256': sha256(path),
        'complete': bool(data.get('completed')), 'scope': protocol.get('scope'),
        'steps': protocol['steps'], 'learning_rate': protocol['learning_rate'],
        'source_readout_version': 2 if checkpoint == KNOWN_CHECKPOINT else None,
        'candidates': [], 'pending_arms': [],
        'metric_limitations': 'Within2 absent from source; entropy borrowed only from same immutable cohort labels. Heldout initial absent.',
        'gradient_alignment': data.get('gradient_alignment', [])}
    normalize = lambda point: normalize_objective(point, entropy_reference)
    internals = {}
    for name in protocol.get('arms',[]):
        if name not in data.get('arms',{}):
            experiment['pending_arms'].append({'variant':name,'last_step':None})
    for name, arm in data.get('arms', {}).items():
        points = sorted(arm.get('learning_curve', []), key=lambda point: point['update'])
        initial = next((point for point in points if point['update'] == 0), None)
        endpoint = next((point for point in points if point['update'] >= protocol['steps']), None)
        if initial is None or endpoint is None or not arm.get('seconds', 0):
            experiment['pending_arms'].append({'variant': name,
                'last_step': points[-1]['update'] if points else None})
            continue
        before, after = normalize(initial), normalize(endpoint)
        attempts = arm.get('updates', [])
        used = [exposure_lookup[(row['replay_id'],row['seat'])] for row in attempts]
        order = [(row['replay_id'],row['seat'],bool(row['stepped'])) for row in attempts]
        candidate = {'id': path.parent.name+':'+name, 'variant': name, 'completed': True,
            'step': endpoint['update'], 'learning_rate': protocol['learning_rate'],
            'trainable_parameters': 182795519 if checkpoint == KNOWN_CHECKPOINT else None,
            'full_seat_passes': endpoint['update']/16,
            'attempts': len(attempts), 'rejected_attempts': sum(not row['stepped'] for row in attempts),
            'input_frames_all_attempts': sum(row['input_frames'] for row in used),
            'gt_play_labels_all_attempts': sum(row['ordinary_play_labels'] for row in used),
            'exposure_order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            'seconds_observed': arm['seconds'], 'peak_vram_gib': arm.get('peak_vram_gib'),
            'timing_qualification': 'GPU contention may differ between arms; not an uncontended throughput benchmark',
            'grad_norm_median': nullable_mean([]),
            'initial': compact_parts(before), 'final': compact_parts(after),
            'delta_vs_own_initial': {part: paired_effect(before.get(part), after.get(part),
                repetitions=repetitions, rng=rng) for part in PARTITIONS},
            'curve': curve_rows(points, normalizer=normalize)}
        norms = [row['grad_norm'] for row in attempts if row.get('grad_norm') is not None]
        candidate['grad_norm_median'] = float(np.median(norms)) if norms else None
        candidate['grad_norm_max'] = max(norms) if norms else None
        candidate['raw_norm_gt1_fraction'] = float(np.mean(np.asarray(norms)>1)) if norms else None
        experiment['candidates'].append(candidate)
        internals[name] = (candidate, before, after)
    experiment['matched_endpoint_comparisons'] = []
    for left, right, interpretation in [('joint_saved','joint_fresh','Optimizer state changes; objective fixed'),
            ('joint_fresh','tile_fresh','Objective changes; both have fresh Adam')]:
        if left in internals and right in internals:
            a, b = internals[left], internals[right]
            experiment['matched_endpoint_comparisons'].append({'before_arm': left, 'after_arm': right,
                'interpretation': interpretation, 'identical_exposure_order': a[0]['exposure_order_sha256']==b[0]['exposure_order_sha256'],
                'partitions': {part: paired_effect(a[2].get(part),b[2].get(part),repetitions=repetitions,rng=rng)
                    for part in PARTITIONS}})
    return experiment


def create_plot(summary, path):
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),dpi=150)
    color = plt.get_cmap('tab10')
    plotted_families=set()
    for experiment in summary['experiments']:
        family = experiment['family']
        ax = axes[1,0] if family=='whole_model_objectives' else axes[0,0] if family=='final_mlp_only' else axes[0,1]
        for index,candidate in enumerate(experiment['candidates']):
            curve = [point for point in candidate['curve'] if 'validation' in point['partitions']]
            if not curve: continue
            x = [point['step']/16 if family=='whole_model_objectives' else point['gt_play_labels_exposed']/677 for point in curve]
            y = [point['partitions']['validation']['ce'] for point in curve]
            label = candidate['variant']+(f' {candidate["learning_rate"]:g}' if candidate.get('learning_rate') else '')
            ax.plot(x,y,marker='.',label=label,color=color(index%10))
            final=candidate['final']
            if 'train' in final and 'validation' in final:
                marker = {'whole_model_objectives':'^','final_mlp_only':'o','full_placement_head':'s','data_prior_initialization':'D'}[family]
                family_index={'final_mlp_only':0,'full_placement_head':1,'data_prior_initialization':2,'whole_model_objectives':3}[family]
                family_label={'final_mlp_only':'Final MLP','full_placement_head':'Full placement',
                    'data_prior_initialization':'Data initialization','whole_model_objectives':'Whole model'}[family]
                axes[1,1].scatter(final['train']['ce'],final['validation']['ce'],marker=marker,
                    color=color(family_index),s=35,alpha=.8,label=family_label if family not in plotted_families else None)
                plotted_families.add(family)
        extension = experiment.get('winner_extension',{})
        if extension.get('completed'):
            curve = extension['curve']
            ax.plot([point['gt_play_labels_exposed']/677 for point in curve],
                [point['partitions']['validation']['ce'] for point in curve],linestyle='--',color='black',
                label='Validation-selected longer continuation')
    settings=[('Final MLP only','GT-play-label exposure / 677'),
        ('Full placement / data initialization','GT-play-label exposure / 677'),
        ('Whole model: joint / tile objectives','Successful seat updates / 16'),
        ('Completed endpoints (different experiments)','Train equal-seat placement CE')]
    for axis,(title,xlabel) in zip(axes.flat,settings):
        axis.set(title=title,xlabel=xlabel,ylabel='Validation equal-seat placement CE')
        axis.grid(alpha=.2)
        if axis.lines:axis.legend(fontsize=6)
    if plotted_families:axes[1,1].legend(fontsize=7)
    fig.suptitle('Local BC capacity probes | each curve uses its own baseline | offline 8/8/8 games',fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report_markdown(summary):
    lines=['# BC 本地容量/充分学习试验汇总（草稿）','',
        '状态：'+('部分结果；仅下表已完成分支可用于讨论。' if summary['partial'] else '已读取全部已发现正式试验的完成结果。'),'',
        '**原 checkpoint 的 readout_version=2，已经具有空间投影/归一化、卡牌/历史 FiLM 和空间门；这些不是本次新加入的功能。**','',
        '所有比较使用同一8训练局、8验证局、8留出局，各保留双座位。它们是诊断集，不能替代完整53验证局、65留出局或闭环对战。CE/2格内命中均条件于真值卡槽。','',
        '表中每项 loss/CE 为座位等权均值；H为固定软目标熵，≤2为2格内命中率；ΔCE 为最终−该试验原 baseline，越负越好。不同试验的精度/初始化及训练范围不同，应先看各自初值与相同曝光下的配对差。', '']
    def num(value,digits=4):return 'NA' if value is None else f'{value:.{digits}f}'
    def tile_triple(parts,part):return ' / '.join(num(parts.get(part,{}).get(key)) for key in ('ce','entropy','within2'))
    for experiment in summary['experiments']:
        lines.extend(['## '+experiment['name']+' / '+experiment['family'],'',str(experiment.get('scope','')),'',
            '| 分支 / LR | 曝光 | train CE / H / ≤2 | val CE / H / ≤2 | held CE | val ΔCE [95%CI] |',
            '|---|---:|---|---|---:|---|'])
        base=experiment.get('baseline')
        if base:
            lines.append('| 原 baseline | 0 | '+tile_triple(base,'train')+' | '+tile_triple(base,'validation')+
                ' | '+num(base.get('heldout',{}).get('ce'))+' | — |')
        for candidate in experiment['candidates']:
            final=candidate['final']
            delta=candidate.get('delta_vs_experiment_baseline',candidate['delta_vs_own_initial'])['validation']
            stat=delta.get('metrics',{}).get('ce',{})
            ci=stat.get('ci95')
            effect=num(stat.get('estimate'))+(' ['+', '.join(num(value) for value in ci)+']' if ci else '')
            exposure=str(candidate.get('gt_play_labels_exposed',candidate.get('gt_play_labels_all_attempts','NA')))+' play labels'
            label=candidate['variant']+' / '+num(candidate.get('learning_rate'),6)
            lines.append('| '+label+' | '+exposure+' | '+tile_triple(final,'train')+' | '+tile_triple(final,'validation')+
                ' | '+num(final.get('heldout',{}).get('ce'))+' | '+effect+' |')
        lines.append('')
        if experiment['family']=='whole_model_objectives':
            lines.extend(['whole-model 各自从step0比较；heldout未记录初值，因此不能给heldout的同路径baseline Δ。within2未记录，记NA；熵仅引用同cohort固定标签。','',
                '| 分支 | train 总loss→ | val 总loss→ | val 条件选牌NLL→ | train 选牌准确率→ | val 选牌准确率→ |',
                '|---|---:|---:|---:|---:|---:|'])
            for candidate in experiment['candidates']:
                def change(part,key):return num(candidate['initial'].get(part,{}).get(key))+' → '+num(candidate['final'].get(part,{}).get(key))
                lines.append('| '+candidate['variant']+' | '+change('train','total_loss')+' | '+change('validation','total_loss')+
                    ' | '+change('validation','card_nll')+' | '+change('train','slot_accuracy')+' | '+change('validation','slot_accuracy')+' |')
            lines.append('')
        extension=experiment.get('winner_extension',{})
        if extension.get('completed'):
            lines.extend(['验证选择的继续训练：'+extension['variant']+'，到 '+str(extension['step'])+' 步 / '+
                str(extension['gt_play_labels_exposed'])+' 次出牌标签曝光。其曝光多于主表，不能归因为同预算架构胜利。',
                '最终 train/val/heldout CE：'+ ' / '.join(num(extension['final'].get(p,{}).get('ce')) for p in PARTITIONS)+'。',''])
        if experiment['family']=='data_prior_initialization':
            lines.extend(['该分支改变初始化函数。相对原baseline的收益包括训练集牌种位置统计的直接帮助；必须同时看其自身step0→末值，不能把先验初始化收益归作新增表达能力。',''])
    lines.extend(['## 比较边界','',
        '- 配对 bootstrap 以完整对局为单位、保留双座位，'+str(summary['bootstrap']['repetitions'])+'次、seed='+str(summary['bootstrap']['seed'])+'。8局区间较宽，多个探索性比较未作多重比较校正。',
        '- 模型容量证据要求充分训练下能否拟合；泛化收益要求独立集改善。只训练最终MLP不能排除冻结attention/表示的限制。',
        '- 端点架构比较只在同一试验、同LR和相同曝光内解释；whole-model的saved/fresh状态与joint/tile目标分别比较。',
        '- GPU并发污染墙钟计时；head读取/提取缓存是一次性开销。不同试验之间不能按记录秒数宣布提速，缓存训练吞吐不能充当端到端吞吐。',
        '- 数据先验、验证选择的长续训与原初始化短测分开报告；heldout不参与winner选择，但这仍是已观察的小诊断集。',
        '- 本地离线改善不等于L4满足0.4秒或战力提高，新候选仍需L4全流程延迟与完整对战验收。','',
        '原始输入/完成状态、曝光、曲线和逐game区间见[summary.json]('+summary['summary_path'].replace('\\','/')+')；本汇总不加载模型、不保存拟合权重。',''])
    return '\n'.join(lines)


def cpu_self_test():
    stat=cluster_interval([1.,3.,10.],[0,0,1],repetitions=1000,rng=np.random.default_rng(3))
    assert stat['estimate']==14/3 and stat['equal_game_estimate']==6 and stat['ci95']==[2.,10.]
    assert head_exposure(400,677,16)==6301
    a={'seats':[{'replay_id':1,'seat':0,'plays':2,'ce':4.}, {'replay_id':1,'seat':1,'plays':3,'ce':6.},
        {'replay_id':2,'seat':0,'plays':1,'ce':5.}, {'replay_id':2,'seat':1,'plays':4,'ce':7.}]}
    b={'seats':[{**row,'ce':row['ce']-.25} for row in a['seats']]}
    effect=paired_effect(a,b,repetitions=1000,rng=np.random.default_rng(4))['metrics']['ce']
    assert effect['estimate']==-.25 and effect['ci95']==[-.25,-.25]
    return {'passed':True,'device':'cpu','head400_label_exposure':6301,'game_pairing_preserved':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/bc_capacity_20261111'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(cpu_self_test()));return
    if args.bootstrap<100:parser.error('Use at least100 bootstrap repetitions')
    root=args.root.resolve();cohort_path=root/'cohort.json';cohort=json.loads(cohort_path.read_text())
    checkpoint=cohort['reference_checkpoint_sha256']
    output=(args.out or root/'summary.json').resolve()
    report=(args.report or Path(__file__).resolve().parents[1]/'docs/BC_CAPACITY_LOCAL_REPORT_DRAFT_20261111.md').resolve()
    if os.name=='nt' and any(path.drive.upper()!='G:' for path in (root,output,report)):
        raise ValueError('All local capacity artifacts and reports must remain on G:')
    # All local caches remain beside the selected G-drive artifacts.
    cache=root/'tmp';cache.mkdir(exist_ok=True)
    os.environ['TEMP']=str(cache);os.environ['TMP']=str(cache)
    os.environ['MPLCONFIGDIR']=str(root/'mplconfig')
    rng=np.random.default_rng(args.seed);experiments=[];entropy_reference={};ignored=[]
    files=sorted(root.glob('*/results.json'))+sorted(root.glob('*/result.json'))
    files=[path for path in files if 'smoke' not in path.parent.name]
    head_files=[];objective_files=[]
    for path in files:
        data=json.loads(path.read_text())
        scope=data.get('protocol',{}).get('scope','')
        games=data.get('protocol',{}).get('games')
        if 'ordinary-slot-selection' in scope or (games and
                any(set(games.get(part,[])) != set(cohort['partitions'][part]['game_ids']) for part in PARTITIONS)):
            ignored.append(str(path)+' (different task or cohort; excluded from8/8/8 matched table)')
            continue
        if isinstance(data.get('arms'),list):head_files.append(path)
        elif isinstance(data.get('arms'),dict):objective_files.append(path)
        else:ignored.append(str(path))
    # Read final-MLP first so fixed-label entropy can be reused by whole-model.
    head_files.sort(key=lambda path:(path.parent.name!='readout_only',path.parent.name))
    for path in head_files:
        experiments.append(summarize_head(path,cohort_path=cohort_path,checkpoint=checkpoint,
            repetitions=args.bootstrap,rng=rng,entropy_reference=entropy_reference))
    for path in objective_files:
        experiments.append(summarize_objectives(path,cohort_path=cohort_path,checkpoint=checkpoint,
            entropy_reference=entropy_reference,repetitions=args.bootstrap,rng=rng))
    incomplete=[experiment['name'] for experiment in experiments if not experiment['complete'] or experiment['pending_arms']]
    found={experiment['family'] for experiment in experiments}
    missing=sorted({'final_mlp_only','full_placement_head','data_prior_initialization','whole_model_objectives'}-found)
    if not args.allow_partial and (incomplete or missing):
        raise ValueError('Final summary requires completed planned families; incomplete='+str(incomplete)+' missing='+str(missing))
    summary={'format':'crrl-bc-capacity-summary-v1','partial':bool(incomplete or missing),'summary_path':str(output),
        'checkpoint_sha256':checkpoint,'source_readout_version':2 if checkpoint==KNOWN_CHECKPOINT else None,
        'existing_function_note':'Reference checkpoint readout-v2 already has spatial normalization/projection, FiLM and gate; no new-FiLM claim',
        'cohort_sha256':sha256(cohort_path),'cohort_games':{part:cohort['partitions'][part]['game_ids'] for part in PARTITIONS},
        'bootstrap':{'repetitions':args.bootstrap,'seed':args.seed,'unit':'whole game; double seats clustered'},
        'experiments':experiments,'incomplete_experiments':incomplete,'missing_families':missing,'ignored_inputs':ignored,
        'limits':['Diagnostic8/8/8 games do not replace full53/65 evaluation',
            'No cross-experiment wall-time speed comparison: GPU concurrency and one-time feature extraction',
            'Only completed arms enter candidates; absent metrics remain null',
            'Paired endpoint intervals are exploratory, with no multiplicity adjustment']}
    atom_json(output,summary)
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(report_markdown(summary),encoding='utf-8')
    if args.plot:create_plot(summary,output.with_suffix('.png'))
    print(json.dumps({'summary':str(output),'report':str(report),'partial':summary['partial'],
        'completed_candidates':sum(len(experiment['candidates']) for experiment in experiments),
        'incomplete':incomplete,'missing_families':missing}))


if __name__=='__main__':main()
