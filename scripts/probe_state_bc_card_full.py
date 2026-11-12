"""Full175/53/65 ordinary-card head probe using the existing packed raw cache.

Only context/card features and one minibatch of entity tokens enter CUDA. The
799MB packed file is mmap-read directly, never copied or reserialized. Ability
weights stay frozen; all labels are ordinary plays, so this remains a diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from probe_state_bc_card_capacity import CardHead, WideSlots, EntityAttentionCardHead, load_small_head, sha, write_json, METRICS


PARTITIONS=('train','validation','heldout')


class PackedCardInputs:
    def __init__(self,cache,dataset,cohort,checkpoint_sha,device):
        self.cache=cache.resolve();self.index=json.loads((self.cache/'index.json').read_text())
        protocol=json.loads((self.cache.parent/'protocol.json').read_text())
        if self.index['checkpoint_sha256']!=checkpoint_sha or self.index['source_protocol_sha256']!=sha(self.cache.parent/'protocol.json'):
            raise ValueError('Packed cache has wrong checkpoint or protocol')
        wanted={int(rid):part for part in PARTITIONS for rid in cohort['partitions'][part]['game_ids']}
        if {part:len(cohort['partitions'][part]['game_ids']) for part in PARTITIONS}!={'train':175,'validation':53,'heldout':65}:
            raise ValueError('Full actor experiment needs all immutable293 games')
        manifest=json.loads(dataset.read_text());meta_path=dataset.parent/manifest['metadata']
        if sha(meta_path)!=manifest['metadata_sha256']:raise ValueError('Metadata checksum changed')
        metadata=json.loads(meta_path.read_text());h=metadata['schema']['header_fields']
        costs=np.asarray([i for i,name in enumerate(h) if name=='hand_cost']);source={}
        if len(costs)!=4:raise ValueError('Expected four cost fields')
        for record in manifest['shards']:
            rid=int(record['replay_offset'])
            if rid not in wanted:continue
            path=dataset.parent/record['path']
            if sha(path)!=record['sha256']:raise ValueError('Full source shard changed')
            with np.load(path,allow_pickle=False) as arrays:
                labels,flags,seats,ticks=(arrays[k] for k in ('actions','flags','seats','ticks'))
                legal=arrays['masks'];obs=arrays['observations']
                valid=(flags==0)&legal[np.arange(len(labels)),labels].astype(bool)
                for seat in (0,1):
                    all_rows=np.flatnonzero(seats==seat);all_rows=all_rows[np.argsort(ticks[all_rows],kind='stable')]
                    selected=all_rows[valid[all_rows]&(labels[all_rows]>0)&(labels[all_rows]<2305)]
                    slots=(labels[selected]-1)//576;tiles=(labels[selected]-1)%576
                    tile_masks=legal[selected,1:2305].reshape(-1,4,576)
                    source[(rid,seat)]={'slots':slots,'tiles':tiles,
                        'legal6':np.concatenate((tile_masks.any(-1),legal[selected,2305:2307]),-1).astype(bool),
                        'tile_legal':tile_masks[np.arange(len(slots)),slots].astype(bool),
                        'cost':obs[selected,costs[slots]].astype(np.int64),
                        'episode_frame_indices':np.searchsorted(all_rows,selected)}
            del obs,legal
        binary=(self.cache/self.index['bin']['path']).resolve()
        if binary.parent!=self.cache or binary.stat().st_size!=self.index['bin']['bytes'] or sha(binary)!=self.index['bin']['sha256']:
            raise ValueError('Packed binary path,size orchecksum changed')
        self.memory=np.memmap(binary,mode='r',dtype=np.uint8);self.blocks=self.index['blocks']
        context=[];cards=[];slots_all=[];legal_all=[];cost_all=[];self.info=[];counters=defaultdict(int)
        self.lookup=[]
        for block_index,record in enumerate(self.blocks):
            meta=record['metadata'];key=int(meta['replay_id']),int(meta['seat'])
            if wanted.get(key[0])!=meta['partition']:raise ValueError('Packed split differs from original full cohort')
            n=record['tensors']['slots']['shape'][0];start=counters[key];stop=start+n;ref=source[key];s=slice(start,stop)
            for name in ('slots','tiles','episode_frame_indices'):
                value=self.read(block_index,name)
                if not np.array_equal(value,ref[name][s]):raise ValueError('Packed actor/source alignment failed: '+name)
            if not np.array_equal(self.read(block_index,'legal'),ref['tile_legal'][s]):raise ValueError('GT tile mask drift')
            context.append(torch.from_numpy(self.read(block_index,'context').astype(np.float32)))
            cards.append(torch.from_numpy(self.read(block_index,'card_features').astype(np.float32)))
            slots_all.append(torch.from_numpy(ref['slots'][s].astype(np.int64)))
            legal_all.append(torch.from_numpy(ref['legal6'][s].copy()))
            cost_all.append(torch.from_numpy(ref['cost'][s].copy()))
            self.lookup.extend((block_index,row) for row in range(n))
            self.info.extend({'replay_id':key[0],'seat':key[1],'partition':meta['partition']} for _ in range(n))
            counters[key]=stop
        if any(counters[key]!=len(value['slots']) for key,value in source.items()):raise ValueError('Missing original labels')
        self.inputs={'context':torch.cat(context).to(device),'card_features':torch.cat(cards).to(device),
            'slots':torch.cat(slots_all).to(device),'legal6':torch.cat(legal_all).to(device),'costs':torch.cat(cost_all).to(device)}
        self.partitions={part:np.asarray([i for i,row in enumerate(self.info) if row['partition']==part],dtype=np.int64)
            for part in PARTITIONS}
        if {part:len(value) for part,value in self.partitions.items()}!={'train':11248,'validation':3847,'heldout':4955}:
            raise ValueError('Unexpected full ordinary play counts')
        if not self.inputs['legal6'].gather(1,self.inputs['slots'][:,None]).all():raise ValueError('Illegal GT card')
        self.costs=self.inputs['costs'].cpu().numpy()
        self.provenance={'checkpoint_sha256':checkpoint_sha,'source_protocol_sha256':sha(self.cache.parent/'protocol.json'),
            'packed_binary_sha256':self.index['bin']['sha256'],'source_dataset_sha256':self.index['source_sha256'],
            'cohort_rows':{part:len(value) for part,value in self.partitions.items()},'blocks':len(self.blocks),
            'copied_to_gpu_bytes':sum(value.numel()*value.element_size() for value in self.inputs.values()),
            'disk_copy_created':False,'alignment':'Every block GTslot,tile,episode_frame_index and exact GTslot legal mask match original game arrays'}
        del source,context,cards,slots_all,legal_all,cost_all,metadata;gc.collect()

    def read(self,index,name):
        spec=self.blocks[index]['tensors'][name];dtype=np.dtype(spec['dtype']);shape=tuple(spec['shape'])
        if int(np.prod(shape))*dtype.itemsize!=spec['nbytes']:raise ValueError('Packed tensor shape invalid')
        view=np.ndarray(shape,dtype=dtype,buffer=self.memory,offset=spec['offset'])
        # One small block copy; context lifetime never depends on mmap views.
        return np.array(view,copy=True)

    def forward(self,head,indices):
        device=self.inputs['context'].device;idx=torch.as_tensor(indices,device=device)
        context,card=self.inputs['context'][idx],self.inputs['card_features'][idx]
        if not head.requires_entities:return head(context,card)
        grouped=defaultdict(list)
        for output,index in enumerate(indices):
            block,row=self.lookup[int(index)];grouped[block].append((output,row))
        count=max(self.blocks[key]['tensors']['entities']['shape'][1] for key in grouped)
        tokens=torch.zeros(len(indices),count,head.card_dim);mask=torch.zeros(len(indices),count,dtype=torch.bool)
        for block,pairs in grouped.items():
            positions=torch.tensor([pair[0] for pair in pairs]);rows=torch.tensor([pair[1] for pair in pairs])
            entity=torch.from_numpy(self.read(block,'entities'));present=torch.from_numpy(self.read(block,'entity_mask'))
            n=entity.shape[1];tokens[positions,:n]=entity.index_select(0,rows).float();mask[positions,:n]=present.index_select(0,rows)
        return head(context,card,tokens.to(device),mask.to(device))

    def close(self):self.memory._mmap.close()


def evaluate(head,data,include_heldout=False,batch_size=256):
    head.eval();parts={};per_seat=[]
    with torch.inference_mode():
        for part in PARTITIONS:
            if part=='heldout' and not include_heldout:continue
            indices=data.partitions[part];packets=[]
            for start in range(0,len(indices),batch_size):
                chosen=indices[start:start+batch_size];index=torch.as_tensor(chosen,device='cuda')
                logits=data.forward(head,chosen).float();legal=data.inputs['legal6'][index];target=data.inputs['slots'][index]
                lp6=logits.masked_fill(~legal,-1e9).log_softmax(-1);lp4=logits[:,:4].masked_fill(~legal[:,:4],-1e9).log_softmax(-1)
                rows=torch.arange(len(chosen),device='cuda')
                packets.append(torch.stack((-lp6[rows,target],-lp4[rows,target],
                    (lp4.argmax(-1)==target).float(),(lp6.argmax(-1)==target).float()),-1).cpu())
            metrics=torch.cat(packets).numpy();seats=defaultdict(list)
            for local,index in enumerate(indices):
                row=data.info[int(index)];seats[(row['replay_id'],row['seat'])].append(local)
            seat_rows=[]
            for (rid,seat),local in sorted(seats.items()):
                row={'partition':part,'replay_id':rid,'seat':seat,'plays':len(local),
                    **{key:float(metrics[local,j].mean()) for j,key in enumerate(METRICS)}}
                seat_rows.append(row);per_seat.append(row)
            high=data.costs[indices]>=5
            parts[part]={'games':len({rid for rid,seat in seats}),'seats':len(seats),'plays':len(indices),
                'equal_seat':{key:float(np.mean([row[key] for row in seat_rows])) for key in METRICS},
                'play_weighted':{key:float(metrics[:,j].mean()) for j,key in enumerate(METRICS)},
                'high_cost_ge5':{'plays':int(high.sum()),**{key:float(metrics[high,j].mean()) for j,key in enumerate(METRICS)}}}
    return {'summary':parts,'seats':per_seat}


def batches_by_epoch(indices,batch_size,epochs,seed):
    rng=np.random.default_rng(seed);output=[]
    for _ in range(epochs):
        order=rng.permutation(indices);output.extend(order[start:start+batch_size] for start in range(0,len(order),batch_size))
    return output


def train_range(head,optimizer,data,batches,start,stop,record,args,*,best=None):
    for index in range(start,stop):
        chosen=batches[index];rows=torch.as_tensor(chosen,device='cuda');head.train();optimizer.zero_grad(set_to_none=True)
        logits=data.forward(head,chosen);loss=nn.functional.cross_entropy(
            logits.masked_fill(~data.inputs['legal6'][rows],-1e9),data.inputs['slots'][rows])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full actor loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_((p for p in head.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
        optimizer.step()
        step=index+1
        if step%args.steps_per_epoch==0:
            metrics={'step':step,'epoch':step/args.steps_per_epoch,**evaluate(head,data)}
            record['metrics'].append(metrics)
            value=metrics['summary']['validation']['equal_seat']['ordinary4_nll']
            if best is not None and value<best['score']:
                best.update(score=value,step=step,weights={name:value.detach().clone() for name,value in head.state_dict().items()})
            print(json.dumps({'phase':'full_actor_eval','variant':record['variant'],'learning_rate':record['learning_rate'],
                'step':step,'summary':metrics['summary'],'last_gradient_norm':float(norm)}),flush=True)


def summarize_completed(out,*,repetitions=5000,seed=20261111):
    """CPU-only compact paired-game inference and chart; no packed cache load."""
    result=json.loads((out/'results.json').read_text())
    if not result.get('completed'):raise ValueError('Wait for full actor experiment completion')
    def compare(before,after):
        report={}
        for part in PARTITIONS:
            a={(r['replay_id'],r['seat']):r for r in before['seats'] if r['partition']==part}
            b={(r['replay_id'],r['seat']):r for r in after['seats'] if r['partition']==part}
            if not a or not b:continue
            if a.keys()!=b.keys() or any(a[k]['plays']!=b[k]['plays'] for k in a):raise ValueError('Actor paired identities differ')
            grouped=defaultdict(list)
            for k in sorted(a):grouped[k[0]].append([b[k][m]-a[k][m] for m in METRICS])
            values=np.asarray([np.mean(rows,axis=0) for rows in grouped.values()]);rng=np.random.default_rng(seed)
            draws=rng.integers(0,len(values),(repetitions,len(values)));boot=values[draws].mean(1)
            report[part]={metric:{'delta_after_minus_before':float(values[:,j].mean()),
                'ci95':list(map(float,np.quantile(boot[:,j],[.025,.975]))),'games':len(values)} for j,metric in enumerate(METRICS)}
        return report
    summary={'completed':True,'protocol':result['protocol'],'baseline':result['baseline_final_full']['summary'],
        'bootstrap':{'repetitions':repetitions,'seed':seed,'unit':'Whole original game, both seats retained'},
        'screening':[],'finalists':[],'same_epoch_comparisons':[]}
    lookup={}
    for arm in result['arms']:
        endpoint=arm['metrics'][-1];lookup[(arm['variant'],arm['learning_rate'])]=arm
        summary['screening'].append({'variant':arm['variant'],'learning_rate':arm['learning_rate'],
            'epoch':endpoint['epoch'],'summary':endpoint['summary'],
            'paired_vs_baseline':compare(result['baseline_final_full'],endpoint)})
    for rate in result['protocol']['learning_rates']:
        for variant in ('wide_residual','entity_attention'):
            summary['same_epoch_comparisons'].append({'after_variant':variant,'before_variant':'original','learning_rate':rate,
                'paired':compare(lookup[('original',rate)]['metrics'][-1],lookup[(variant,rate)]['metrics'][-1])})
    for finalist in result['finalists']:
        selected=finalist['best_validation_metrics'];summary['finalists'].append({'variant':finalist['variant'],
            'learning_rate':finalist['learning_rate'],'selection_rank':finalist['selection_rank'],
            'epoch':selected['epoch'],'summary':selected['summary'],
            'paired_vs_baseline':compare(result['baseline_final_full'],selected)})
    write_json(out/'summary.json',summary)
    root=out.parent;os.environ['MPLCONFIGDIR']=str(root/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),dpi=150)
    for finalist in result['finalists']:
        screening=lookup[(finalist['variant'],finalist['learning_rate'])]
        points=screening['metrics'][:-1]+finalist['metrics']
        label=finalist['variant']+' '+f"{finalist['learning_rate']:g}"
        axes[0].plot([p['epoch'] for p in points],[p['summary']['validation']['equal_seat']['ordinary4_nll'] for p in points],marker='o',label=label)
        axes[1].plot([p['epoch'] for p in points],[p['summary']['validation']['equal_seat']['slot_accuracy']*100 for p in points],marker='o',label=label)
    axes[0].set(ylabel='Full53 validation ordinary4 NLL',title='Card ranking NLL (lower is better)')
    axes[1].set(ylabel='Full53 validation slot accuracy (%)',title='Argmax card accuracy')
    for axis in axes:axis.set_xlabel('Ordinary-play label epochs');axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.suptitle('Frozen-representation ordinary-card diagnostic | ability weights frozen',fontsize=10)
    fig.tight_layout();fig.savefig(out/'summary.png');plt.close(fig)
    lines=['# 全量普通选牌头诊断（2026-11-11）','','全175训练局11248条普通出牌；53验证局选候选/epoch；65留出局末次评分。主干冻结，ability参数冻结；只含普通play，不含WAIT或能力正例，不是完整actor训练。','',
        '| 1epoch变体/LR | train4NLL | val4NLL | val slotacc |','|---|---:|---:|---:|']
    for arm in summary['screening']:
        q=arm['summary'];lines.append('| '+arm['variant']+'/'+f"{arm['learning_rate']:g}"+' | '+
            f"{q['train']['equal_seat']['ordinary4_nll']:.5f} | {q['validation']['equal_seat']['ordinary4_nll']:.5f} | {q['validation']['equal_seat']['slot_accuracy']*100:.3f}% |")
    lines.extend(['','验证选出的较长训练（不同曝光，单独解释）：','','| 候选 | epoch | val4NLL | val acc | held4NLL | held acc | held Δ4NLL [95%CI] |','|---|---:|---:|---:|---:|---:|---|'])
    for arm in summary['finalists']:
        v=arm['summary']['validation']['equal_seat'];h=arm['summary']['heldout']['equal_seat'];d=arm['paired_vs_baseline']['heldout']['ordinary4_nll']
        lines.append('| '+arm['variant']+' | '+str(arm['epoch'])+' | '+f"{v['ordinary4_nll']:.5f} | {v['slot_accuracy']*100:.3f}% | {h['ordinary4_nll']:.5f} | {h['slot_accuracy']*100:.3f}% | {d['delta_after_minus_before']:.5f} [{d['ci95'][0]:.5f}, {d['ci95'][1]:.5f}] |")
    lines.extend(['','结论：全数据下原/宽/实体读出均可小幅改善普通4类NLL，但准确率提升很小；同1epoch实体读出相对原头差异很小。未证明global-pool瓶颈是选牌主因，也未证明能力不足已排除。不能把这个仅ordinary标签的小头fit直接部署。','',
        '6类NLL的改善同时包含slot相对ability质量变化；因此普通4类NLL和acc为主要解释指标。所有区间整局bootstrap5000，探索性多比较未校正。','',
        '没有写第二套特征bin或任何拟合权重。峰PyTorch allocated约'+f"{result['peak_allocated_gib']:.3f}"+'GiB；共享rawcache由根任务统一回收。',''])
    (Path(__file__).resolve().parents[1]/'docs/BC_CARD_FULL_RESULT_20261111.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path('artifacts/bc_capacity_20261111')
    parser.add_argument('--checkpoint',type=Path,default=Path('artifacts/bc100_rootcause_20261110/current_state.pt'))
    parser.add_argument('--cache',type=Path,default=root/'full_raw/feature_cache')
    parser.add_argument('--dataset',type=Path,default=root/'full_data/dataset.manifest.json')
    parser.add_argument('--cohort',type=Path,default=root/'full_cohort.json')
    parser.add_argument('--out',type=Path,default=root/'card_full')
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--eval-batch-size',type=int,default=256)
    parser.add_argument('--learning-rates',default='3e-5,3e-4');parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=20261111);args=parser.parse_args(argv)
    # A standalone CPU summarizer is intentionally selected via an existing
    # completed output, so no cache/model is touched on a repeated invocation.
    if (args.out/'results.json').exists() and json.loads((args.out/'results.json').read_text()).get('completed'):
        summary=summarize_completed(args.out);print(json.dumps({'phase':'cpu_summary','screening':len(summary['screening']),'finalists':len(summary['finalists'])}));return
    if min(args.batch_size,args.eval_batch_size,args.width)<1:parser.error('Positive dimensions needed')
    if os.name=='nt' and args.out.resolve().drive.upper()!='G:':raise ValueError('Keep all local artifacts on G:')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Use a fresh experiment output')
    temp=args.out/'tmp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp.resolve())
    if not torch.cuda.is_available():raise RuntimeError('Existing CUDA environment required')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(args.seed)
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(2*1024**3/total,0)
    checkpoint_sha=sha(args.checkpoint);cohort=json.loads(args.cohort.read_text())
    if checkpoint_sha!=cohort['reference_checkpoint_sha256']:raise ValueError('Wrong actor source checkpoint')
    data=PackedCardInputs(args.cache,args.dataset,cohort,checkpoint_sha,'cuda')
    original,source=load_small_head(args.checkpoint);original=original.cuda().eval()
    batches=batches_by_epoch(data.partitions['train'],args.batch_size,3,args.seed)
    args.steps_per_epoch=math.ceil(len(data.partitions['train'])/args.batch_size)
    rates=[float(value) for value in args.learning_rates.split(',')]
    protocol={'scope':'full175 ordinary-card-selection diagnostic; frozen representations, ability frozen; not full actor training',
        **data.provenance,**source,'cohort_sha256':sha(args.cohort),'batch_size':args.batch_size,
        'steps_per_epoch':args.steps_per_epoch,'screening_action_epochs':1,'continuation_action_epochs':3,
        'variants':['original','wide_residual','entity_attention'],'learning_rates':rates,'seed':args.seed,
        'selection':'Lowest full53 validation ordinary4 NLL, plus highest ordinary slot accuracy alternative; no heldout selection',
        'objective':'Conditional6 nonWAIT CE on ordinary labels only; report ordinary4 metrics to expose ability mass effects',
        'allocator_limit_gib':2,'new_disk_feature_cache':False,'fitted_weights_saved':False,
        'precision':'FP32 small heads highest matmul; frozen original raw contexts/tokens',
        'limits':['No ability positive labels; ability weights frozen','Offline true-play-time labels, no WAIT decision task',
            'Same1epoch initial screening; continuations have more exposure and are reported separately']}
    write_json(args.out/'protocol.json',protocol)
    baseline=evaluate(original,data);write_json(args.out/'baseline.json',baseline)
    results={'protocol':protocol,'baseline':baseline,'arms':[]};live={};started=time.perf_counter()
    def chosen_keys():
        ranked=sorted(results['arms'],key=lambda arm:arm['metrics'][-1]['summary']['validation']['equal_seat']['ordinary4_nll'])
        selected=[(ranked[0]['variant'],ranked[0]['learning_rate'])]
        best_acc=sorted(ranked,key=lambda arm:(-arm['metrics'][-1]['summary']['validation']['equal_seat']['slot_accuracy'],
            arm['metrics'][-1]['summary']['validation']['equal_seat']['ordinary4_nll']))
        selected.extend((arm['variant'],arm['learning_rate']) for arm in best_acc
            if (arm['variant'],arm['learning_rate']) not in selected)
        return selected[:2]
    try:
        for rate in rates:
            for variant in protocol['variants']:
                torch.manual_seed(args.seed);head=deepcopy(original)
                if variant=='wide_residual':head.slot_actor=WideSlots(head.slot_actor,head.context_dim+head.card_dim,args.width)
                if variant=='entity_attention':head=EntityAttentionCardHead(head)
                head.ability_actor.requires_grad_(False)
                with torch.no_grad():
                    chosen=np.arange(32);index=torch.as_tensor(chosen,device='cuda')
                    torch.testing.assert_close(data.forward(head,chosen),original(data.inputs['context'][index],data.inputs['card_features'][index]),
                        atol=1e-6,rtol=1e-6)
                optimizer=torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),lr=rate,weight_decay=.01,fused=True)
                record={'variant':variant,'learning_rate':rate,'trainable_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
                    'metrics':[{'step':0,'epoch':0,**baseline}]}
                train_range(head,optimizer,data,batches,0,args.steps_per_epoch,record,args)
                record['completed_screening']=True;results['arms'].append(record);live[(variant,rate)]=(head,optimizer,record)
                keep=set(chosen_keys());live={key:value for key,value in live.items() if key in keep}
                write_json(args.out/'results.json',results);del head,optimizer;gc.collect();torch.cuda.empty_cache()
        results['finalists']=[]
        for rank,key in enumerate(chosen_keys(),1):
            head,optimizer,record=live.pop(key)
            score=record['metrics'][-1]['summary']['validation']['equal_seat']['ordinary4_nll']
            best={'score':score,'step':args.steps_per_epoch,'weights':{name:value.detach().clone() for name,value in head.state_dict().items()}}
            extension={'variant':key[0],'learning_rate':key[1],'selection_rank':rank,'metrics':[record['metrics'][-1]]}
            train_range(head,optimizer,data,batches,args.steps_per_epoch,3*args.steps_per_epoch,extension,args,best=best)
            head.load_state_dict(best['weights'],strict=True)
            extension['best_validation_metrics']={'step':best['step'],'epoch':best['step']/args.steps_per_epoch,**evaluate(head,data,include_heldout=True)}
            results['finalists'].append(extension);write_json(args.out/'results.json',results)
            del head,optimizer,best;gc.collect();torch.cuda.empty_cache()
        results['baseline_final_full']=evaluate(original,data,include_heldout=True)
        results['completed']=True;results['seconds']=time.perf_counter()-started
        results['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        write_json(args.out/'results.json',results)
        print(json.dumps({'phase':'full_actor_complete','seconds':results['seconds'],'peak_allocated_gib':results['peak_allocated_gib']}))
    finally:data.close()
    summarize_completed(args.out)


if __name__=='__main__':main()
