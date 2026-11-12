"""CPU-only full175/53/65 placement-head experiment summary, separate from pilots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from summarize_state_bc_capacity import atom_json, normalize_head, compact_parts, paired_effect, head_exposure, sha256


PARTITIONS=('train','validation','heldout')


def stable_pair(before,after,*,seed,repetitions):
    return paired_effect(before,after,rng=np.random.default_rng(seed),repetitions=repetitions)


def all_pairs(before,after,*,seed,repetitions):
    return {part:stable_pair(before.get(part),after.get(part),seed=seed+index,repetitions=repetitions)
        for index,part in enumerate(PARTITIONS)}


def check_point(point,protocol):
    normalized=normalize_head(point)
    for part,value in normalized.items():
        expected=set(map(int,protocol['games'][part]))
        found={row['replay_id'] for row in value['seats']}
        if expected!=found or len(value['seats'])!=2*len(expected):
            raise ValueError('Full split evaluation is missing games or seats: '+part)
    return normalized


def curves(points,protocol):
    output=[]
    rows=int(protocol['training_rows']);batch=int(protocol['batch_size'])
    for point in points:
        step=int(point['step']);parts=check_point(point,protocol)
        exposure=head_exposure(step,rows,batch)
        output.append({'step':step,'play_labels_exposed':exposure,'action_epochs':exposure/rows,
            'partitions':compact_parts(parts)})
    return output


def plot_full(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),dpi=150)
    labels=[];ce=[];near=[]
    for candidate in summary['screening_candidates']:
        labels.append(candidate['variant']+'\n'+f"{candidate['learning_rate']:g}")
        ce.append(candidate['final']['validation']['ce']);near.append(candidate['final']['validation']['within2']*100)
    x=np.arange(len(labels));axes[0].bar(x,ce,color='#3475ac');axes[1].bar(x,near,color='#4b9b74')
    baseline=summary['baseline'].get('validation',{})
    if baseline:
        axes[0].axhline(baseline['ce'],color='#888',linestyle='--',label='Original baseline')
        axes[1].axhline(baseline['within2']*100,color='#888',linestyle='--')
    axes[0].set(ylabel='Equal-seat placement CE',title='Same 1-epoch exposure: lower is better')
    axes[1].set(ylabel='Placement within2 (%)',title='Geometry: higher is better')
    for axis in axes[:2]:axis.set_xticks(x,labels,rotation=45,ha='right',fontsize=7)
    for candidate in summary['finalists']:
        curve=candidate['curve']
        axes[2].plot([p['action_epochs'] for p in curve],
            [p['partitions']['validation']['ce'] for p in curve],marker='o',markersize=3,
            label=candidate['variant']+' '+f"{candidate['learning_rate']:g}")
        best=candidate.get('selected_validation_best',{})
        if best:axes[2].scatter(best['action_epochs'],best['partitions']['validation']['ce'],marker='*',s=100)
    axes[2].set(xlabel='Action-label epochs',ylabel='Full53 validation CE',title='Validation-selected continuations')
    for axis in axes:
        axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    if labels:axes[0].legend(fontsize=7)
    if summary['finalists']:axes[2].legend(fontsize=7)
    fig.suptitle('Full175 train / 53 validation / 65 heldout | frozen representations | final heldout never selects',fontsize=11)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def report_full(summary):
    def num(value):return 'NA' if value is None else f'{value:.4f}'
    def effect(comparison,part,key='ce'):
        stat=comparison.get(part,{}).get('metrics',{}).get(key,{})
        if not stat:return 'NA'
        return num(stat['estimate'])+' ['+', '.join(num(v) for v in stat['ci95'])+']'
    lines=['# 全量 BC 落点头对照（2026-11-11）','',
        '状态：'+('结果尚未完整，主表仅显示完成的筛选分支。' if summary['partial'] else '筛选、继续训练及末次留出评估均已完成。'),'',
        '175训练局11248次出牌标签；53验证局用于筛选；65留出局只在模型/epoch选择结束后评分。与8/8/8先导试验分开解释。实体和时序表征冻结，完整落点头可训练。', '',
        '**本轮验证集选择的候选是训练牌种位置初始化＋实体空间CNN残差、冻结主干的head阶段，LR=3e−4；不是继续无限扩大183M时序主干。** 这是已测候选中的离线结果，不是所有架构的全局最优或实战胜率结论。', '',
        '**原readout_version=2已有FiLM/空间门。** CNN改变空间残差表达；牌种位置先验改变初始化。两类影响分别与匹配对照比较。', '',
        '| 同1epoch分支 / LR | train CE | val CE | val ≤2 | val ΔCE对原baseline [95%CI] |',
        '|---|---:|---:|---:|---|']
    for candidate in summary['screening_candidates']:
        final=candidate['final']
        lines.append('| '+candidate['variant']+' / '+f"{candidate['learning_rate']:g}"+' | '+num(final['train']['ce'])+
            ' | '+num(final['validation']['ce'])+' | '+num(final['validation']['within2'])+' | '+
            effect(candidate['delta_vs_original_baseline'],'validation')+' |')
    lines.extend(['','相同LR/曝光的结构增量：','',
        '| 比较（后者−前者） | val ΔCE [95%CI] | val Δ≤2 [95%CI] |',
        '|---|---|---|'])
    for comparison in summary['matched_screening_comparisons']:
        lines.append('| '+comparison['after_variant']+' − '+comparison['before_variant']+' / '+f"{comparison['learning_rate']:g}"+
            ' | '+effect(comparison['partitions'],'validation')+' | '+effect(comparison['partitions'],'validation','within2')+' |')
    lines.extend(['','## 验证选择的最终模型','',
        '| 分支 | 选中的epoch | train CE | val CE | held CE | held ≤2 | held ΔCE对原 [95%CI] | held ΔCE对同alpha先验 [95%CI] |',
        '|---|---:|---:|---:|---:|---:|---|---|'])
    for candidate in summary['finalists']:
        best=candidate.get('selected_validation_best')
        if not best:continue
        final=best['partitions'];held=final.get('heldout',{})
        lines.append('| '+candidate['variant']+' / '+f"{candidate['learning_rate']:g}"+' | '+num(best['action_epochs'])+
            ' | '+num(final.get('train',{}).get('ce'))+' | '+num(final.get('validation',{}).get('ce'))+
            ' | '+num(held.get('ce'))+' | '+num(held.get('within2'))+' | '+
            effect(candidate['delta_vs_original_baseline'],'heldout')+' | '+
            effect(candidate.get('delta_vs_matching_fixed_prior',{}),'heldout')+' |')
    resources=summary.get('resources',{})
    latency=summary.get('l4_latency',{})
    lines.extend(['','## 资源、边界与后续','',
        '原模型 '+str(resources.get('base_model_parameters','NA'))+' 参数，新增 '+str(resources.get('selected_added_parameters','NA'))+
        '（约'+num(resources.get('added_fraction',0.)*100)+'%）。本次所有筛选/继续训练及验证的记录墙钟 '+num(resources.get('seconds_excluding_extraction'))+
        ' 秒；不包含独立特征提取/打包/加载启动，不作为端到端速度倍数。', '',
        '独立L4压力夹具：900次已预热完整决策，5种结构均低于400ms；prior＋CNN最大约'+num(latency.get('selected_structure_max_ms'))+
        'ms。这是随机/零输出权重的结构耗时夹具，不是拟合模型战力或冷启动承诺。','',
        '本实验源自本地冻结step2274，不是每次变化的云端最新权重。云端若已继续训练，应验证迁移起点，不能直接把该旧分支当精确续训。','',
        '为确认充分学习，rank1另有最多10epoch的继续拟合试验正在进行；本表仅是已完成的3epoch公平比较，不把尚未完成的长拟合写成结果，也不与本表曝光混合。','',
        '只按53局验证挑选rank1及其epoch；rank2在heldout上CE更好也不据此反选。与纯prior相比，rank1的heldout CE配对区间跨0，几何2格命中改善更稳定，不能只呈现有利指标。','',
        '配对区间使用完整对局重采样，保留同局双座位；'+str(summary['bootstrap']['repetitions'])+
        '次，固定seed。选择多候选/epoch后区间属于探索性证据，未校正选择偏差。', '',
        '时机/选牌未直接训练，理论上沿用冻结主干及原头；只有独立logits parity通过后才称数值不变。落点指标条件于真值卡槽，不等于闭环实战胜率。', '',
        '训练墙钟有并发负载和特征提取一次性成本，不据此比较端到端速度。L4单独性能夹具仅验证已预热架构耗时，随机权重不能证明战力。', '',
        '[完整汇总]('+summary['summary_path'].replace('\\','/')+')',''])
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('artifacts/bc_capacity_20261111/full_train'))
    parser.add_argument('--out',type=Path);parser.add_argument('--report',type=Path)
    parser.add_argument('--bootstrap',type=int,default=5000);parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--allow-partial',action='store_true');parser.add_argument('--plot',action='store_true')
    args=parser.parse_args(argv)
    if args.bootstrap<100:parser.error('At least100 bootstrap repetitions required')
    root=args.input.resolve();output=(args.out or root/'summary.json').resolve()
    report=(args.report or Path(__file__).resolve().parents[1]/'docs/BC_FULL_TRAIN_RESULT_20261111.md').resolve()
    if os.name=='nt' and any(path.drive.upper()!='G:' for path in (root,output,report)):raise ValueError('Outputs must stay on G:')
    temporary=root/'tmp';temporary.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temporary)
    os.environ['MPLCONFIGDIR']=str(root/'.mplconfig')
    data=json.loads((root/'results.json').read_text());protocol=data['protocol']
    if {part:len(protocol['games'][part]) for part in PARTITIONS}!={'train':175,'validation':53,'heldout':65}:
        raise ValueError('This summary is exclusively full175/53/65; do not mix pilot cohorts')
    baseline_path=root/'baseline_final_full.json'
    baseline=check_point(json.loads((baseline_path if baseline_path.exists() else root/'baseline.json').read_text()),protocol)
    summary={'format':'crrl-bc-full-train-summary-v1','summary_path':str(output),
        'protocol':{key:protocol.get(key) for key in ('checkpoint_sha256','checkpoint_progress','source_readout_version',
            'training_rows','steps','batch_size','learning_rates','variants','selection_policy','winner_selection',
            'heldout_selection','restore_best_validation_epoch','action_steps_per_epoch','winner_extension_target_steps')},
        'bootstrap':{'repetitions':args.bootstrap,'seed':args.seed,'unit':'Whole game; both seats clustered'},
        'baseline':compact_parts(baseline),'screening_candidates':[],'pending_arms':[],
        'matched_screening_comparisons':[],'finalists':[],'fixed_training_prior_baselines':[]}
    target=int(protocol['steps']);rows=int(protocol['training_rows']);batch=int(protocol['batch_size'])
    by_key={};expected={(variant,rate) for variant in protocol['variants'] for rate in protocol['learning_rates']}
    for arm in data['arms']:
        key=arm['variant'],arm['learning_rate'];points=sorted(arm['metrics'],key=lambda value:value['step'])
        endpoint=next((point for point in points if point['step']>=target),None)
        initial=next((point for point in points if point['step']==0),None)
        if endpoint is None or initial is None:
            summary['pending_arms'].append({'variant':key[0],'learning_rate':key[1]});continue
        before,after=check_point(initial,protocol),check_point(endpoint,protocol)
        candidate={'id':key[0]+':lr'+str(key[1]),'variant':key[0],'learning_rate':key[1],
            'step':endpoint['step'],'play_labels_exposed':head_exposure(endpoint['step'],rows,batch),
            'action_epochs':head_exposure(endpoint['step'],rows,batch)/rows,'trainable_parameters':arm.get('trainable_parameters'),
            'initial':compact_parts(before),'final':compact_parts(after),'data_initialization':arm.get('data_initialization'),
            'delta_vs_own_initial':all_pairs(before,after,seed=args.seed,repetitions=args.bootstrap),
            'delta_vs_original_baseline':all_pairs(baseline,after,seed=args.seed+11,repetitions=args.bootstrap),
            'curve':curves(points,protocol)}
        summary['screening_candidates'].append(candidate);by_key[key]=(arm,candidate,before,after);expected.discard(key)
    summary['pending_arms'].extend({'variant':variant,'learning_rate':rate} for variant,rate in sorted(expected))
    for rate in protocol['learning_rates']:
        for left,right,meaning in [('original','spatial_cnn','Same initial function; added spatial residual'),
                ('card_prior_001','card_prior_cnn_001','Same data initialization; added spatial residual')]:
            if (left,rate) in by_key and (right,rate) in by_key:
                summary['matched_screening_comparisons'].append({'before_variant':left,'after_variant':right,
                    'learning_rate':rate,'meaning':meaning,'partitions':all_pairs(by_key[(left,rate)][3],
                    by_key[(right,rate)][3],seed=args.seed+22,repetitions=args.bootstrap)})
    prior_path=root/'prior_baselines_final_full.json';prior_parts={}
    if prior_path.exists():
        for prior in json.loads(prior_path.read_text())['variants']:
            parts=check_point(prior,protocol);prior_parts[float(prior['alpha'])]=parts
            summary['fixed_training_prior_baselines'].append({'alpha':prior['alpha'],
                'training_data_only':prior['training_data_only'],'partitions':compact_parts(parts)})
    extension_files=sorted(root.glob('winner_extension_rank*.json'))
    if not extension_files and (root/'winner_extension.json').exists():extension_files=[root/'winner_extension.json']
    for path in extension_files:
        extension=json.loads(path.read_text());key=extension['variant'],extension['learning_rate']
        if key not in by_key:raise ValueError('Finalist was not a completed screening arm')
        arm,screen,before,_=by_key[key]
        points=[point for point in arm['metrics'] if point['step']<extension['selected_at_step']]+extension['metrics']
        candidate={'variant':key[0],'learning_rate':key[1],'selection_rank':extension.get('selection_rank'),
            'selection':extension['selection'],'selected_at_step':extension['selected_at_step'],
            'source_file':path.name,'source_sha256':sha256(path),'curve':curves(points,protocol)}
        selected=extension.get('best_validation_metrics')
        if selected is None and points and points[-1]['step']>=protocol['winner_extension_target_steps']:selected=points[-1]
        if selected:
            parts=check_point(selected,protocol);exposure=head_exposure(selected['step'],rows,batch)
            candidate['selected_validation_best']={'step':selected['step'],'action_epochs':exposure/rows,
                'play_labels_exposed':exposure,'partitions':compact_parts(parts),
                'note':'Epoch selected only by validation; full heldout scored after restoring its decoder'}
            candidate['delta_vs_original_baseline']=all_pairs(baseline,parts,seed=args.seed+33,repetitions=args.bootstrap)
            candidate['delta_vs_own_initial']=all_pairs(before,parts,seed=args.seed+44,repetitions=args.bootstrap)
            if key[0].startswith('card_prior'):
                alpha=.01 if key[0].endswith('001') else 1.
                if alpha in prior_parts:
                    candidate['delta_vs_matching_fixed_prior']=all_pairs(prior_parts[alpha],parts,
                        seed=args.seed+55,repetitions=args.bootstrap)
            candidate['heldout_complete']='heldout' in parts
        summary['finalists'].append(candidate)
    completion_path=root/'completion.json';completion=json.loads(completion_path.read_text()) if completion_path.exists() else {}
    if completion.get('checkpoint_sha256_after') not in (None,protocol['checkpoint_sha256']):raise ValueError('Original checkpoint SHA changed')
    needed=int(protocol.get('extend_top_k',1));complete_finalists=sum(candidate.get('heldout_complete',False) for candidate in summary['finalists'])
    summary['partial']=not(completion.get('complete') and not summary['pending_arms'] and
        complete_finalists>=needed and 'heldout' in baseline and prior_parts)
    summary['completion']=completion
    added=1_410_626
    summary['resources']={'base_model_parameters':182_795_519,'selected_added_parameters':added,
        'total_selected_architecture_parameters':182_795_519+added,'added_fraction':added/182_795_519,
        'seconds_excluding_extraction':completion.get('seconds'),
        'wall_clock_scope':'All screening/continuation/validation after extraction and baseline; not end-to-end'}
    latency_path=root.parent/'l4_latency_actual/summary.json'
    if latency_path.exists():
        latency=json.loads(latency_path.read_text())
        selected_latency=next((row for row in latency['variants'] if row['variant']=='card_prior_cnn_001'),{})
        summary['l4_latency']={'source':str(latency_path),'measurements':latency['measurements'],
            'all_measured_under400ms':latency['all_passed_400ms'],
            'selected_structure_max_ms':selected_latency.get('max_warm_ms'),
            'selected_structure_p99_max_ms':selected_latency.get('max_case_p99_ms'),
            'scope':latency['scope'],'limitations':latency['limitations']}
    summary['logit_parity_independently_verified']=False
    summary['parity_note']='Frozen encoder/card/time imply unchanged functions; await independent raw-logit parity before a numeric no-change claim'
    if summary['partial'] and not args.allow_partial:raise ValueError('Full protocol/heldout not finished; use --allow-partial for screening-only report')
    atom_json(output,summary);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(report_full(summary),encoding='utf-8')
    if args.plot:plot_full(summary,output.with_suffix('.png'))
    print(json.dumps({'summary':str(output),'partial':summary['partial'],'screened':len(summary['screening_candidates']),
        'finalists':len(summary['finalists']),'pending':len(summary['pending_arms'])}))


if __name__=='__main__':main()
