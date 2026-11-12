"""Bounded ordinary-card selection diagnostic on frozen chronological features.

No full model is constructed. Only small slot/ability tensors are mmap-copied
from the existing checkpoint. Ability weights stay frozen: this cohort contains
ordinary-play labels, not positive ability examples. Fitted heads are never saved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


PARTITIONS = ('train','validation','heldout')
METRICS = ('conditional6_nll','ordinary4_nll','slot_accuracy','conditional6_accuracy')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class CardHead(nn.Module):
    requires_entities=False
    def __init__(self,context_dim=1280,card_dim=256,hidden=640):
        super().__init__()
        self.context_dim,self.card_dim=context_dim,card_dim
        self.slot_actor=nn.Sequential(nn.LayerNorm(context_dim+card_dim),
            nn.Linear(context_dim+card_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.ability_actor=nn.Linear(context_dim,2)
        self.ability_actor.requires_grad_(False)

    def forward(self,context,card_features):
        contexts=context[:,None].expand(-1,4,-1)
        cards=self.slot_actor(torch.cat((contexts,card_features),-1)).squeeze(-1)
        return torch.cat((cards,self.ability_actor(context)),-1)


class WideSlots(nn.Module):
    def __init__(self,original,dimensions,width=1024):
        super().__init__()
        self.original=original
        device=next(original.parameters()).device
        self.branch=nn.Sequential(nn.LayerNorm(dimensions,device=device),
            nn.Linear(dimensions,width,device=device),nn.GELU(),
            nn.Linear(width,width,device=device),nn.GELU(),nn.Linear(width,1,device=device))
        nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)

    def forward(self,features):return self.original(features)+self.branch(features)


class EntityAttentionCardHead(nn.Module):
    """Four card-conditioned queries read the CURRENT frozen entity tokens.

    Original six nonWAIT logits are preserved at zero-output initialization.
    Only the four slot logits receive residuals; ability parameters stay frozen.
    This tests a direct entity readout, not a larger temporal/global MLP.
    """
    requires_entities=True

    def __init__(self,original,heads=4):
        super().__init__()
        self.original=original
        self.context_dim,self.card_dim=original.context_dim,original.card_dim
        dimensions=self.context_dim+self.card_dim;device=next(original.parameters()).device
        self.query=nn.Sequential(nn.LayerNorm(dimensions,device=device),
            nn.Linear(dimensions,self.card_dim,device=device))
        self.entity_norm=nn.LayerNorm(self.card_dim,device=device)
        self.attention=nn.MultiheadAttention(self.card_dim,heads,batch_first=True,dropout=0.,device=device)
        self.readout=nn.Sequential(nn.LayerNorm(2*self.card_dim,device=device),
            nn.Linear(2*self.card_dim,self.card_dim,device=device),nn.GELU(),nn.Linear(self.card_dim,1,device=device))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
        if self.added_parameter_count()>=1_500_000:raise ValueError('Entity branch exceeds1.5M added parameters')

    @property
    def ability_actor(self):return self.original.ability_actor

    def added_parameter_count(self):
        return sum(p.numel() for name,p in self.named_parameters() if not name.startswith('original.'))

    def forward(self,context,card_features,entities,entity_mask):
        base=self.original(context,card_features)
        if entities.ndim!=3 or entity_mask.shape!=entities.shape[:2] or entities.shape[-1]!=self.card_dim:
            raise ValueError('Encoded entity tokens and masks must align')
        mask=entity_mask.bool();tokens=self.entity_norm(torch.where(mask[...,None],entities.float(),0.))
        # A dummy is visible only for an empty entity list; valid lists are
        # unaffected. It prevents all-masked attention from producing NaNs.
        tokens=torch.cat((tokens,tokens.new_zeros(len(context),1,self.card_dim)),1)
        mask=torch.cat((mask,~mask.any(-1,keepdim=True)),-1)
        query=self.query(torch.cat((context[:,None].expand(-1,4,-1),card_features),-1))
        attended,_=self.attention(query,tokens,tokens,key_padding_mask=~mask,need_weights=False)
        residual=self.readout(torch.cat((query,attended),-1)).squeeze(-1)
        return base+torch.cat((residual,residual.new_zeros(len(context),2)),-1)


def load_small_head(checkpoint_path):
    if not hasattr(np,'_core'):
        sys.modules.setdefault('numpy._core',np.core)
        sys.modules.setdefault('numpy._core.multiarray',np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric',np.core.numeric)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
    config=checkpoint['config'];context_dim=checkpoint['temporal_config']['width'];card_dim=config['entity_dim']
    head=CardHead(context_dim,card_dim,context_dim//2)
    head.slot_actor.load_state_dict({key.removeprefix('slot_actor.'):value
        for key,value in checkpoint['model'].items() if key.startswith('slot_actor.')},strict=True)
    head.ability_actor.load_state_dict({key.removeprefix('ability_actor.'):value
        for key,value in checkpoint['model'].items() if key.startswith('ability_actor.')},strict=True)
    metadata={'checkpoint_progress':dict(checkpoint['progress']),
        'context_dim':context_dim,'card_dim':card_dim,
        'dataset_fingerprint':checkpoint['dataset_fingerprint'],'source_readout_version':config['readout_version']}
    del checkpoint;gc.collect()
    return head,metadata


def prepare_inputs(args):
    """Align cached ordinary-play rows with their original labels and masks."""
    cohort=json.loads(args.cohort.read_text());manifest=json.loads(args.dataset.read_text())
    source_protocol=json.loads((args.feature_cache.parent/'protocol.json').read_text())
    checkpoint_sha=sha(args.checkpoint)
    if checkpoint_sha!=cohort['reference_checkpoint_sha256'] or source_protocol['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('Actor source checkpoint differs from frozen cache/cohort')
    if source_protocol.get('cohort_sha256')!=sha(args.cohort):
        raise ValueError('Raw feature cache belongs to another cohort')
    metadata_path=args.dataset.parent/manifest['metadata']
    if sha(metadata_path)!=manifest['metadata_sha256']:raise ValueError('Dataset metadata changed')
    metadata=json.loads(metadata_path.read_text());fields=metadata['schema']['header_fields']
    cost_columns=np.asarray([index for index,name in enumerate(fields) if name=='hand_cost'])
    if len(cost_columns)!=4:raise ValueError('Expected four recorded hand costs')
    del metadata;gc.collect()
    wanted={int(rid):part for part in PARTITIONS for rid in cohort['partitions'][part]['game_ids']}
    source={}
    for record in manifest['shards']:
        rid=int(record['replay_offset'])
        if rid not in wanted:continue
        path=args.dataset.parent/record['path']
        if sha(path)!=record['sha256']:raise ValueError('Cohort shard checksum changed')
        # Exactly one compressed game is decoded at a time. No whole dataset,
        # model tensor set, optimizer state or entity history is materialized.
        with np.load(path,allow_pickle=False) as arrays:
            actions,flags,seats,ticks=(arrays[key] for key in ('actions','flags','seats','ticks'))
            masks=arrays['masks'];observations=arrays['observations']
            valid=(flags==0)&masks[np.arange(len(actions)),actions].astype(bool)
            for seat in (0,1):
                rows=np.flatnonzero((seats==seat)&valid&(actions>0)&(actions<2305))
                rows=rows[np.argsort(ticks[rows],kind='stable')]
                slots=(actions[rows]-1)//576;tiles=(actions[rows]-1)%576
                card_legal=masks[rows,1:2305].reshape(-1,4,576).any(-1)
                nonwait=np.concatenate((card_legal,masks[rows,2305:2307].astype(bool)),-1)
                if not nonwait[np.arange(len(rows)),slots].all():raise ValueError('True card is not legal')
                source[(rid,seat)]={'slots':slots.astype(np.int64),'tiles':tiles.astype(np.int64),
                    'legal6':nonwait,'tile_legal':masks[rows,1:2305].reshape(-1,4,576)[np.arange(len(rows)),slots],
                    'costs':observations[rows,cost_columns[slots]].astype(np.int64),'partition':wanted[rid]}
        del observations,masks,actions,flags,seats,ticks
    counters=defaultdict(int);collected=defaultdict(list);row_info=[];entity_blocks=[]
    files=sorted(args.feature_cache.glob('block_*.pt'))
    if not files:raise ValueError('Raw feature cache was already reclaimed or is absent')
    for path in files:
        block=torch.load(path,map_location='cpu',weights_only=False)
        key=int(block['replay_id']),int(block['seat'])
        if key not in source:raise ValueError('Cache contains a game outside frozen cohort')
        reference=source[key];start=counters[key];stop=start+len(block['slots']);selection=slice(start,stop)
        if (reference['partition']!=block['partition'] or
                not np.array_equal(block['slots'].numpy(),reference['slots'][selection]) or
                not np.array_equal(block['tiles'].numpy(),reference['tiles'][selection]) or
                not np.array_equal(block['legal'].bool().numpy(),reference['tile_legal'][selection])):
            raise ValueError('Chronological cache/GT slot/tile/legal-mask alignment failed')
        collected['context'].append(block['context'].float().clone())
        collected['card_features'].append(block['card_features'].float().clone())
        collected['slots'].append(block['slots'].long().clone())
        collected['legal6'].append(torch.from_numpy(reference['legal6'][selection].copy()).bool())
        collected['costs'].append(torch.from_numpy(reference['costs'][selection].copy()).long())
        block_index=len(entity_blocks)
        if args.include_entities:
            if block['entities'].shape[:2]!=block['entity_mask'].shape or len(block['entities'])!=stop-start:
                raise ValueError('Raw entity cache/mask rows differ')
            entity_blocks.append({'entities':block['entities'].clone(),'entity_mask':block['entity_mask'].bool().clone()})
        row_info.extend({'replay_id':key[0],'seat':key[1],'partition':reference['partition'],
            'entity_block':block_index,'entity_row':row} for row in range(stop-start))
        counters[key]=stop
        del block
    if any(counters[key]!=len(value['slots']) for key,value in source.items()):
        raise ValueError('One or more original ordinary-play labels were omitted')
    values={key:torch.cat(value) for key,value in collected.items()}
    if not all(torch.isfinite(value).all() for value in values.values()):raise FloatingPointError('Nonfinite cached actor input')
    expected={part:cohort['partitions'][part]['summary']['ordinary_play_labels'] for part in PARTITIONS}
    actual={part:sum(row['partition']==part for row in row_info) for part in PARTITIONS}
    if actual!=expected:raise ValueError('Unexpected ordinary-play counts')
    entity_bytes=sum(value.numel()*value.element_size() for block in entity_blocks for value in block.values())
    payload={'inputs':values,'row_info':row_info,'entity_blocks':entity_blocks,'provenance':{'checkpoint_sha256':checkpoint_sha,
        'cohort_sha256':sha(args.cohort),'source_cache_protocol_sha256':sha(args.feature_cache.parent/'protocol.json'),
        'dataset_manifest_sha256':sha(args.dataset),'rows':actual,'cache_blocks':len(files),
        'bytes':sum(value.numel()*value.element_size() for value in values.values())+entity_bytes,
        'entity_bytes':entity_bytes,'entities_retained':args.include_entities,
        'alignment':'Per replay/seat chronological GT ordinary-play order; slots, tiles and exact GT tile masks equal',
        'ability_labels':0,'cache_contains_weights':False}}
    staged=args.out/'prepared_inputs.pt.tmp'
    torch.save(payload,staged);staged.replace(args.out/'prepared_inputs.pt')
    write_json(args.out/'preparation.json',payload['provenance'])
    return payload


def forward_rows(head,inputs,info,indices,entity_blocks):
    """Pad/gather only one minibatch; never pad all1880 rows on the GPU."""
    index=torch.as_tensor(indices,device=inputs['context'].device)
    context,card=inputs['context'][index],inputs['card_features'][index]
    if not head.requires_entities:return head(context,card)
    if not entity_blocks:raise ValueError('Entity-attention arm requires the aligned raw token blocks')
    groups=defaultdict(list)
    for output_row,source_row in enumerate(indices):
        metadata=info[int(source_row)];groups[metadata['entity_block']].append((output_row,metadata['entity_row']))
    count=max(entity_blocks[key]['entities'].shape[1] for key in groups)
    tokens=torch.zeros(len(indices),count,head.card_dim,dtype=torch.float32)
    mask=torch.zeros(len(indices),count,dtype=torch.bool)
    for key,pairs in groups.items():
        positions=torch.tensor([pair[0] for pair in pairs]);rows=torch.tensor([pair[1] for pair in pairs])
        block=entity_blocks[key];length=block['entities'].shape[1]
        tokens[positions,:length]=block['entities'].index_select(0,rows).float()
        mask[positions,:length]=block['entity_mask'].index_select(0,rows)
    return head(context,card,tokens.to(context.device),mask.to(context.device))


def evaluate(head,inputs,info,batch_size=128,entity_blocks=None):
    head.eval();packets=[]
    with torch.inference_mode():
        for start in range(0,len(info),batch_size):
            end=min(start+batch_size,len(info));s=slice(start,end)
            logits=forward_rows(head,inputs,info,np.arange(start,end),entity_blocks).float()
            legal=inputs['legal6'][s];target=inputs['slots'][s]
            lp6=logits.masked_fill(~legal,-1e9).log_softmax(-1)
            lp4=logits[:,:4].masked_fill(~legal[:,:4],-1e9).log_softmax(-1)
            rows=torch.arange(end-start,device=logits.device)
            packets.append(torch.stack((-lp6[rows,target],-lp4[rows,target],
                (lp4.argmax(-1)==target).float(),(lp6.argmax(-1)==target).float()),-1).cpu())
    all_metrics=torch.cat(packets).numpy();costs=inputs['costs'].cpu().numpy()
    result={'summary':{},'seats':[]}
    for part in PARTITIONS:
        indices=np.asarray([i for i,row in enumerate(info) if row['partition']==part])
        per_seat=[]
        for rid,seat in sorted({(info[i]['replay_id'],info[i]['seat']) for i in indices}):
            selected=np.asarray([i for i in indices if info[i]['replay_id']==rid and info[i]['seat']==seat])
            row={'partition':part,'replay_id':rid,'seat':seat,'plays':len(selected),
                **{key:float(all_metrics[selected,j].mean()) for j,key in enumerate(METRICS)}}
            result['seats'].append(row);per_seat.append(row)
        grouped={}
        for name,choose in [('cost_lt5',costs[indices]<5),('cost_ge5',costs[indices]>=5)]:
            selected=indices[choose]
            grouped[name]={'plays':len(selected),**{key:float(all_metrics[selected,j].mean())
                if len(selected) else None for j,key in enumerate(METRICS)}}
        result['summary'][part]={'games':len({info[i]['replay_id'] for i in indices}),
            'seats':len(per_seat),'plays':len(indices),
            'play_weighted':{key:float(all_metrics[indices,j].mean()) for j,key in enumerate(METRICS)},
            'equal_seat':{key:float(np.mean([row[key] for row in per_seat])) for key in METRICS},
            'cost_groups_play_weighted':grouped}
    return result


def cpu_self_test():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261111)
        actor=CardHead(8,4,6);wide=deepcopy(actor);wide.slot_actor=WideSlots(wide.slot_actor,12,16)
        context=torch.randn(5,8);cards=torch.randn(5,4,4)
        assert torch.equal(actor(context,cards),wide(context,cards))
        legal=torch.ones(5,6,dtype=torch.bool);legal[0,1]=False;target=torch.tensor([0,1,2,3,0])
        logits=wide(context,cards);loss=nn.functional.cross_entropy(logits.masked_fill(~legal,-1e9),target)
        loss.backward()
        assert wide.ability_actor.weight.grad is None
        assert wide.slot_actor.branch[-1].weight.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wide.parameters())
        entity=EntityAttentionCardHead(deepcopy(actor),heads=2)
        tokens=torch.randn(5,7,4);mask=torch.ones(5,7,dtype=torch.bool);mask[:,-2:]=False
        assert torch.equal(actor(context,cards),entity(context,cards,tokens,mask))
        entity(context,cards,tokens,mask).square().mean().backward()
        assert entity.readout[-1].weight.grad.abs().sum()>0
        assert entity.ability_actor.weight.grad is None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in entity.parameters())
        with torch.no_grad():entity.readout[-1].weight.fill_(.1)
        entity.zero_grad(set_to_none=True)
        entity(context,cards,tokens,mask).square().mean().backward()
        assert entity.query[-1].weight.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in entity.parameters())
        bad=tokens.clone();bad[:,-2:]=float('nan')
        assert torch.equal(entity(context,cards,tokens,mask),entity(context,cards,bad,mask))
        empty=entity(context,cards,torch.empty(5,0,4),torch.empty(5,0,dtype=torch.bool))
        assert torch.isfinite(empty).all()
        return {'passed':True,'device':'cpu','initial_function_exact':True,'ability_frozen':True,
            'finite_gradients':True,'entity_attention_initial_exact':True,'masked_entities_invariant':True,'empty_entities_finite':True}


def summarize_results(out):
    """Small CPU-only whole-game paired intervals; no model/cache load."""
    result=json.loads((out/'results.json').read_text());rng=np.random.default_rng(20261111)
    def compare(before,after):
        report={}
        for part in PARTITIONS:
            a={(row['replay_id'],row['seat']):row for row in before['seats'] if row['partition']==part}
            b={(row['replay_id'],row['seat']):row for row in after['seats'] if row['partition']==part}
            if a.keys()!=b.keys():raise ValueError('Actor paired identities differ')
            grouped=defaultdict(list)
            for key in sorted(a):
                if a[key]['plays']!=b[key]['plays']:raise ValueError('Actor paired labels differ')
                grouped[key[0]].append([b[key][metric]-a[key][metric] for metric in METRICS])
            # Both seats are present, so equal-game and equal-seat means agree.
            values=np.asarray([np.mean(rows,axis=0) for rows in grouped.values()])
            index=rng.integers(0,len(values),(5000,len(values)))
            boot=values[index].mean(1)
            report[part]={metric:{'after_minus_before':float(values[:,j].mean()),
                'ci95':list(map(float,np.quantile(boot[:,j],[.025,.975]))),'games':len(values)}
                for j,metric in enumerate(METRICS)}
        return report
    summary={'completed':result.get('completed'),'bootstrap_repetitions':5000,'seed':20261111,
        'unit':'Complete game; both seats retained together','arms':[]}
    for arm in result['arms']:
        if not arm.get('completed'):continue
        row={'variant':arm['variant'],'learning_rate':arm['learning_rate'],
            'initial':arm['metrics'][0]['summary'],'final':arm['metrics'][-1]['summary'],
            'paired_vs_initial':compare(arm['metrics'][0],arm['metrics'][-1])}
        original=next((other for other in result['arms'] if other['variant']=='original' and
            other['learning_rate']==arm['learning_rate']),None)
        if original and arm['variant']!='original':row['paired_vs_same_lr_original']=compare(original['metrics'][-1],arm['metrics'][-1])
        summary['arms'].append(row)
    write_json(out/'paired_summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('artifacts/bc100_rootcause_20261110/current_state.pt'))
    parser.add_argument('--feature-cache',type=Path,default=Path('artifacts/bc_capacity_20261111/readout_only/feature_cache'))
    parser.add_argument('--dataset',type=Path,default=Path('artifacts/bc_capacity_20261111/data/dataset.manifest.json'))
    parser.add_argument('--cohort',type=Path,default=Path('artifacts/bc_capacity_20261111/cohort.json'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/bc_capacity_20261111/card_capacity'))
    parser.add_argument('--steps',type=int,default=400);parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--eval-every',type=int,default=50);parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--learning-rates',default='3e-5,3e-4');parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--variants',default='original,wide_residual')
    parser.add_argument('--append-results',action='store_true')
    parser.add_argument('--gpu-memory-limit-gib',type=float,choices=(.5,.75),default=.5)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--keep-input-cache',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:print(json.dumps(cpu_self_test()));return
    if args.summarize_only:
        summary=summarize_results(args.out);print(json.dumps({'phase':'cpu_summary','arms':len(summary['arms'])}));return
    if not(args.prepare_only or args.run_gpu):parser.error('Choose --prepare-only or explicit --run-gpu')
    variants=args.variants.split(',');args.include_entities='entity_attention' in variants
    if any(value not in ('original','wide_residual','entity_attention') for value in variants):parser.error('Unknown actor arm')
    if min(args.steps,args.batch_size,args.eval_every,args.width)<1:parser.error('Positive sizes required')
    if os.name=='nt' and args.out.resolve().drive.upper()!='G:':raise ValueError('All local outputs must remain on G:')
    args.out.mkdir(parents=True,exist_ok=True);temporary=args.out/'tmp';temporary.mkdir(exist_ok=True)
    os.environ['TEMP']=str(temporary.resolve());os.environ['TMP']=str(temporary.resolve())
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(args.seed)
    prepared=args.out/'prepared_inputs.pt'
    payload=torch.load(prepared,map_location='cpu',weights_only=False) if prepared.exists() else prepare_inputs(args)
    if payload['provenance']['cohort_sha256']!=sha(args.cohort):raise ValueError('Prepared features belong to another cohort')
    if args.include_entities and not payload.get('entity_blocks'):raise ValueError('Rebuild the prepared cache with entities for this arm')
    if args.prepare_only:
        print(json.dumps({'phase':'cpu_prepared',**payload['provenance']}));return
    existing_path=args.out/'results.json'
    if existing_path.exists() and not args.append_results:raise ValueError('Results exist; use --append-results only for new matched arms')
    if not torch.cuda.is_available():raise RuntimeError('Existing CUDA required; install nothing')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit_gib*1024**3/total,0)
    original,source=load_small_head(args.checkpoint)
    if source['checkpoint_progress']['optimizer_steps']!=2274:raise ValueError('Expected frozen step2274 actor')
    values={key:value.to('cuda') for key,value in payload['inputs'].items()};info=payload['row_info']
    provenance=payload['provenance'];entity_blocks=payload.get('entity_blocks');del payload;gc.collect()
    original=original.cuda().eval();original.ability_actor.requires_grad_(False)
    train=np.asarray([i for i,row in enumerate(info) if row['partition']=='train'])
    rng=np.random.default_rng(args.seed);batches=[]
    while len(batches)<args.steps:
        order=rng.permutation(train)
        batches.extend(order[start:start+args.batch_size] for start in range(0,len(order),args.batch_size))
    batches=batches[:args.steps];rates=[float(value) for value in args.learning_rates.split(',')]
    protocol={'scope':'ordinary-slot-selection diagnostic; frozen context/card representations; ability head frozen',
        **provenance,**source,'steps':args.steps,'batch_size':args.batch_size,'learning_rates':rates,
        'ordinary_play_exposures':sum(map(len,batches)),'equivalent_action_epochs':sum(map(len,batches))/len(train),
        'seed':args.seed,'precision':'FP32 heads, highest matmul; cached representations unchanged',
        'optimizer':'Fresh fused AdamW, weight_decay .01, global clip1; no checkpoint optimizer loaded',
        'loss':'Conditional six-class nonWAIT CE on ordinary labels only; frozen ability logits remain competitors',
        'allocator_limit_gib':args.gpu_memory_limit_gib,'allocator_limit_note':'PyTorch allocator; CUDA context/library memory is separate',
        'gpu':torch.cuda.get_device_name(0),'save_weights':False,
        'limitations':['No positive ability examples, so not full-actor retraining or a deployable fit',
            '8/8/8 diagnostic games, not full53/65 or native battles',
            'Concurrent GPU use invalidates end-to-end wall-time speed comparisons']}
    baseline=evaluate(original,values,info)
    if existing_path.exists():
        result=json.loads(existing_path.read_text());previous=result['protocol']
        for key in ('checkpoint_sha256','cohort_sha256','steps','batch_size','seed','ordinary_play_exposures','learning_rates'):
            if previous[key]!=protocol[key]:raise ValueError('Append protocol differs: '+key)
        error=max(abs(baseline['summary'][part]['equal_seat'][key]-result['baseline']['summary'][part]['equal_seat'][key])
            for part in PARTITIONS for key in METRICS)
        if error>1e-6:raise ValueError('Append baseline recomputation drift')
        result.setdefault('extension_runs',[]).append({'variants':variants,'protocol':protocol,'baseline_max_error':error})
    else:
        write_json(args.out/'protocol.json',protocol);write_json(args.out/'baseline.json',baseline)
        result={'protocol':protocol,'baseline':baseline,'arms':[]}
    result['completed']=False;write_json(existing_path,result);started=time.perf_counter()
    for rate in rates:
        for variant in variants:
            if any(arm['variant']==variant and arm['learning_rate']==rate and arm.get('completed') for arm in result['arms']):continue
            torch.manual_seed(args.seed);head=deepcopy(original)
            if variant=='wide_residual':head.slot_actor=WideSlots(head.slot_actor,head.context_dim+head.card_dim,args.width)
            if variant=='entity_attention':head=EntityAttentionCardHead(head)
            head.ability_actor.requires_grad_(False)
            parameters=[p for p in head.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=rate,weight_decay=.01,fused=True)
            with torch.no_grad():
                picked=np.arange(min(32,len(info)));pick=torch.as_tensor(picked,device='cuda')
                torch.testing.assert_close(forward_rows(head,values,info,picked,entity_blocks),
                    original(values['context'][pick],values['card_features'][pick]),rtol=1e-6,atol=1e-6)
            arm={'variant':variant,'learning_rate':rate,'trainable_parameters':sum(p.numel() for p in parameters),
                'added_parameters':head.added_parameter_count() if variant=='entity_attention' else None,
                'metrics':[{'step':0,**evaluate(head,values,info,entity_blocks=entity_blocks)}]}
            for step,indices in enumerate(batches,1):
                index=torch.as_tensor(indices,device='cuda');head.train();optimizer.zero_grad(set_to_none=True)
                logits=forward_rows(head,values,info,indices,entity_blocks)
                loss=nn.functional.cross_entropy(logits.masked_fill(~values['legal6'][index],-1e9),values['slots'][index])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
                if step%args.eval_every==0 or step==args.steps:
                    point={'step':step,'last_loss':float(loss),'last_gradient_norm':float(norm),**evaluate(head,values,info,entity_blocks=entity_blocks)}
                    arm['metrics'].append(point)
                    print(json.dumps({'phase':'card_capacity','variant':variant,'learning_rate':rate,
                        'step':step,'summary':point['summary']}),flush=True)
            arm['completed']=True;result['arms'].append(arm);write_json(args.out/'results.json',result)
            del head,optimizer;gc.collect();torch.cuda.empty_cache()
    result['completed']=True;elapsed=time.perf_counter()-started
    result['seconds']=result.get('seconds',0.)+elapsed
    if args.append_results:result['extension_runs'][-1]['seconds']=elapsed
    result['peak_allocated_gib']=max(result.get('peak_allocated_gib',0.),torch.cuda.max_memory_allocated()/1024**3)
    if not args.keep_input_cache:prepared.unlink(missing_ok=True)
    result['input_cache_reclaimed']=not prepared.exists();write_json(args.out/'results.json',result)
    summarize_results(args.out)
    print(json.dumps({'phase':'complete','arms':len(result['arms']),'seconds':result['seconds'],
        'peak_allocated_gib':result['peak_allocated_gib'],'input_cache_reclaimed':result['input_cache_reclaimed']}))


if __name__=='__main__':main()
