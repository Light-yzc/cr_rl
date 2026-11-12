"""CPU-only long-fit/validation-temperature summary; no cache or model reads."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np

from summarize_state_bc_capacity import atom_json, normalize_head, compact_parts, paired_effect, sha256


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/bc_capacity_20261111'))
    parser.add_argument('--bootstrap',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261111)
    args=parser.parse_args(argv);root=args.root.resolve();long=root/'rank1_long10'
    if os.name=='nt' and root.drive.upper()!='G:':raise ValueError('All artifacts remain on G:')
    temp=root/'tmp';os.environ['TEMP']=os.environ['TMP']=str(temp);os.environ['MPLCONFIGDIR']=str(root/'mplconfig')
    protocol=json.loads((long/'protocol.json').read_text());extension=json.loads((long/'winner_extension.json').read_text())
    completion=json.loads((long/'completion.json').read_text())
    calibration=json.loads((root/'temperature_calibration.json').read_text())
    if not completion.get('complete') or calibration['heldout_used_for_selection']:raise ValueError('Incomplete or heldout-selected results')
    if calibration['source_checkpoint_sha256']!=protocol['checkpoint_sha256']:raise ValueError('Calibration source mismatch')
    source=normalize_head(json.loads((long/'baseline_final_full.json').read_text()))
    priors=json.loads((long/'prior_baselines_final_full.json').read_text())['variants']
    prior=normalize_head(next(row for row in priors if row['alpha']==.01))
    raw=normalize_head(extension['best_validation_metrics']);calibrated=normalize_head(calibration['final'])
    early=json.loads((root/'full_train/winner_extension_rank1.json').read_text())
    three=normalize_head(early['best_validation_metrics'])
    def paired(a,b):return {part:paired_effect(a[part],b[part],repetitions=args.bootstrap,
        rng=np.random.default_rng(args.seed+i)) for i,part in enumerate(('train','validation','heldout'))}
    delta_cal=paired(raw,calibrated)
    if any(abs(delta_cal[part]['metrics']['within2']['estimate'])>1e-12 for part in delta_cal):
        raise ValueError('Positive scalar temperature changed argmax geometric metric')
    all_points=json.loads((long/'results.json').read_text())['arms'][0]['metrics'][:-1]+extension['metrics']
    curves=[]
    for point in all_points:
        curves.append({'step':point['step'],'action_epoch':point['step']/protocol['action_steps_per_epoch'],
            'partitions':compact_parts(normalize_head(point))})
    summary={'format':'crrl-bc-long-fit-summary-v1','completed':True,
        'source_checkpoint_sha256':protocol['checkpoint_sha256'],'source_checkpoint_progress':protocol['checkpoint_progress'],
        'variant':extension['variant'],'learning_rate':extension['learning_rate'],
        'best_validation_epoch':extension['best_validation_epoch'],'final_training_epoch':curves[-1]['action_epoch'],
        'epoch_selection':'Minimum full53 validation equal-seat CE; heldout never picks epoch or temperature',
        'temperature_selection':calibration['selection'],'selected_temperature':calibration['selected_temperature'],
        'temperature_grid':calibration['grid_predeclared'],'argmax_unchanged_verified':True,
        'source_baseline':compact_parts(source),'fixed_prior_alpha001':compact_parts(prior),
        'rank1_3epoch':compact_parts(three),'longfit_uncalibrated':compact_parts(raw),
        'longfit_validation_calibrated':compact_parts(calibrated),
        'calibrated_minus_source':paired(source,calibrated),'calibrated_minus_prior':paired(prior,calibrated),
        'calibrated_minus_3epoch':paired(three,calibrated),'calibration_delta':delta_cal,'curve':curves,
        'bootstrap':{'repetitions':args.bootstrap,'seed':args.seed,'unit':'Whole game, two seats clustered'},
        'completion':completion,'calibration_sha256':sha256(root/'temperature_calibration.json'),
        'limits':['Same development validation reused for candidate/epoch/temperature selection; intervals exploratory',
            'Longfit exposure differs from3epoch; report separately','Temperature changes confidence not representation or argmax',
            'Local frozen step2274, not current cloud latest; offline GT-card conditional placement, no battle claim']}
    coverage=json.loads((root.parent/'bc100_generalization_20261111/coverage.json').read_text())
    coverage_seats={(row['replay_id'],row['seat']):row for row in coverage['seats']}
    summary['deck_generalization_subgroups']={}
    for part in ('validation','heldout'):
        for seen in (True,False):
            def subset(parts):
                seats=[row for row in parts[part]['seats'] if coverage_seats[(row['replay_id'],row['seat'])]['physical_deck_seen_in_train']==seen]
                return {'seats':seats,'mean':{key:float(np.mean([row[key] for row in seats])) for key in ('ce','entropy','within2')},
                    'games':len({row['replay_id'] for row in seats})}
            a,b=subset(source),subset(calibrated)
            summary['deck_generalization_subgroups'][part+('_seen' if seen else '_unseen')]={
                'games':a['games'],'seats':len(a['seats']),'baseline':a['mean'],'final':b['mean'],
                'paired_delta':paired_effect(a,b,repetitions=args.bootstrap,rng=np.random.default_rng(args.seed)),
                'note':'Descriptive card-composition/state confounding remains; physical deck identity only'}
    summary['high_cost_subgroup_unavailable']='Final decoder JSON retains seat metrics, not per-play predictions; cannot infer high-cost placement metrics from deck composition'
    atom_json(long/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),dpi=150)
    epoch=[point['action_epoch'] for point in curves]
    for part in ('train','validation'):
        axes[0].plot(epoch,[point['partitions'][part]['ce'] for point in curves],marker='o',label=part)
    axes[0].axvline(summary['best_validation_epoch'],linestyle='--',color='#888',label='Validation-selected epoch')
    axes[0].set(xlabel='Action-label epochs',ylabel='Equal-seat placement CE',title='Long fit: training improves; validation rebounds')
    grid=calibration['grid_validation_results']
    axes[1].plot([row['temperature'] for row in grid],
        [row['summary']['validation']['equal_seat']['ce'] for row in grid],marker='o',color='#3475ac')
    axes[1].axvline(calibration['selected_temperature'],linestyle='--',color='#888')
    axes[1].set(xlabel='Scalar temperature',ylabel='Full53 validation CE',title='Temperature selected on validation only')
    for axis in axes:axis.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(long/'summary.png');plt.close(fig)
    def metric(parts,p,k):return f"{parts[p][k]:.4f}"
    def ci(comparison,part,key='ce'):
        row=comparison[part]['metrics'][key];return f"{row['estimate']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
    lines=['# 充分拟合与置信度校准结果（2026-11-11）','',
        '本轮候选：训练牌种位置初始化＋64通道实体scatter/CNN残差，冻结183M主干，仅拟合落点头，LR3e−4。来源为本地step2274。', '',
        '最长训练到10个action-label epoch；只按53局验证CE选择epoch8，末epoch10已反弹。随后在预先固定温度网格[.5,1,1.5,2,3]中只用同验证集选τ=1.5，再对65局留出末次评分。没有按heldout挑epoch或τ。','',
        '| 模型 | train CE | val CE | held CE | val ≤2 | held ≤2 |','|---|---:|---:|---:|---:|---:|']
    for name,parts in [('源模型',summary['source_baseline']),('纯训练牌种prior',summary['fixed_prior_alpha001']),
            ('同rank1 3epoch',summary['rank1_3epoch']),('epoch8原置信度',summary['longfit_uncalibrated']),
            ('epoch8＋τ1.5',summary['longfit_validation_calibrated'])]:
        lines.append('| '+name+' | '+' | '.join(metric(parts,p,k) for p,k in
            [('train','ce'),('validation','ce'),('heldout','ce'),('validation','within2'),('heldout','within2')])+' |')
    lines.extend(['','整局配对5000次bootstrap，最终校准模型−对照：','',
        '| 对照 | val ΔCE [95%CI] | held ΔCE [95%CI] | held Δ≤2 [95%CI] |','|---|---|---|---|'])
    for name,key in [('源模型','calibrated_minus_source'),('固定牌种prior','calibrated_minus_prior'),('rank1 3epoch','calibrated_minus_3epoch')]:
        lines.append('| '+name+' | '+ci(summary[key],'validation')+' | '+ci(summary[key],'heldout')+' | '+ci(summary[key],'heldout','within2')+' |')
    lines.extend(['','温度只改变概率尖锐程度；argmax及2格命中完全不变，不能称空间表达进一步提高。长训练提升验证CE，heldout相对3epoch的差异须按配对区间说明，不能借heldout反选旧epoch。','',
        '验证集反复用于结构/epoch/温度开发，区间是探索性证据，未校正选择偏差。不能据此承诺实战胜率；还需闭环旧BC/old对战。所有总loss/选牌“完全不变”声明须依独立raw-logits parity，单凭参数冻结只作理论预期。','',
        '[完整汇总](G:/Code/cr_rl/artifacts/bc_capacity_20261111/rank1_long10/summary.json)',''])
    report=Path(__file__).resolve().parents[1]/'docs/BC_LONG_FIT_RESULT_20261111.md';report.write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'summary':str(long/'summary.json'),'best_epoch':summary['best_validation_epoch'],
        'temperature':summary['selected_temperature'],'held_ce':summary['longfit_validation_calibrated']['heldout']['ce']}))


if __name__=='__main__':main()
