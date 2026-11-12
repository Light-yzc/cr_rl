"""Bounded local causal probes of a frozen state-BC checkpoint; never publish weights."""
from __future__ import annotations
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.training.state_bc import ReplaySequenceDataset,run_state_episode


def module_group(name):
    if name.startswith('frame.'):return 'frame'
    if name.startswith(('temporal.','to_temporal.')):return 'temporal'
    return 'heads_and_context'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--parent-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=64)
    args=parser.parse_args()
    if not 1<=args.steps<=96:raise ValueError('Use 1..96 bounded diagnostic updates')
    if not torch.cuda.is_available():raise RuntimeError('An existing CUDA environment is required')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(20261110)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    import hashlib
    assert checkpoint['dataset_fingerprint']==hashlib.sha256(args.parent_manifest.read_bytes()).hexdigest()
    settings=checkpoint['training_config']
    dataset=ReplaySequenceDataset(args.dataset,args.out.parent/'.probe-scratch',ram_limit_gib=6)
    model=state_model_from_checkpoint(checkpoint,'cuda');model.frame_microbatch=32
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=settings['weight_decay'],fused=True)
    saved={**checkpoint['optimizer'],'param_groups':[{**g,'lr':0.,'fused':True,'foreach':False} for g in checkpoint['optimizer']['param_groups']]}
    optimizer.load_state_dict(saved)
    # Free the mapped parent tensors as soon as the live GPU copies exist.
    for parameter,values in optimizer.state.items():
        if isinstance(values.get('step'),torch.Tensor):values['step']=values['step'].to(parameter.device,dtype=torch.float32)
    progress=checkpoint['progress'];del checkpoint,saved;gc.collect()
    adapter=CrForgeObservationAdapter(dataset.schema,model.cfg.vocab_size,entity_bucket=16)
    common=dict(chunk_size=settings['chunk_size'],device='cuda',precision='fp16',
        position_coef=settings['position_coef'],tile_label_sigma=settings['tile_label_sigma'],
        timing_objective=settings['timing_objective'],timing_target_action_prior=settings['timing_target_action_prior'],
        max_grad_norm=settings['max_grad_norm'],ram_limit_gib=6,defer_metrics=True)
    partitions={key:('train' if key in {252,112,206,227,253} else 'validation') for key in dataset.replay_ids}
    # Keep original whole-training prior for all repeated-fit updates.
    train_prior=float(settings['sampled_action_prior'])
    validation_prior=4023/57493
    report={'checkpoint_progress':progress,'gpu':torch.cuda.get_device_name(0),'steps':args.steps,
       'scope':'Diagnostic only; no parameter file, optimizer or result is published back to training',
       'training_cases':[[252,0],[112,0]],'control_cases':[[5,0],[13,0]],'input_counterfactuals':[],
       'fixed_fit':[],'gradients':[],'gate_stats':{}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    original_clip=torch.nn.utils.clip_grad_norm_
    gradient_capture=[]
    def capture_clip(parameters,max_norm,**kwargs):
        totals={}
        for name,parameter in model.named_parameters():
            if parameter.grad is None:continue
            group=module_group(name)
            value=parameter.grad.detach().float().square().sum()
            totals[group]=totals.get(group,0)+value
        gradient_capture.append({key:float(value.sqrt().cpu()) for key,value in totals.items()})
        return original_clip(parameters,max_norm,**kwargs)
    torch.nn.utils.clip_grad_norm_=capture_clip
    def one(replay,seat,*,training=False,adapter_override=None,prior=None):
        part=partitions[replay]
        iterator=dataset.episodes(partitions,part,epoch=0,seed=20261110)
        try:
            for episode in iterator:
                if episode.replay_id==replay and episode.seat==seat:
                    options={**common,'sampled_action_prior':prior or (train_prior if part=='train' else validation_prior)}
                    if training:
                        scaler=torch.amp.GradScaler('cuda',init_scale=32)
                        return run_state_episode(model,episode,adapter_override or adapter,optimizer=optimizer,scaler=scaler,**options)
                    return run_state_episode(model,episode,adapter_override or adapter,**options)
        finally:iterator.close()
        raise ValueError('Missing diagnostic seat')
    def summaries(cases):
        rows=[one(game,seat) for game,seat in cases]
        keys=['loss','timing_loss','conditional_slot_loss','position_loss','act_recall','slot_accuracy','tile_mean_distance']
        return {key:sum(row[key] for row in rows)/len(rows) for key in keys}
    # Zero-LR counterfactual on the exact spike seat. Only large buff columns change.
    class BoundedBuffAdapter:
        schema=adapter.schema
        def __call__(self,observations):
            header,entities,mask=adapter(observations)
            for index,name in enumerate(dataset.schema.entity_fields):
                if name.startswith('buff_ms'):
                    values=entities[...,index]
                    entities[...,index]=np.where(values>65504.,36.,values)
            return header,entities,mask
    try:
        for name,wrapped in [('original',adapter),('near_permanent_buff_to_360_seconds',BoundedBuffAdapter())]:
            gradient_capture.clear();row=one(253,0,training=True,adapter_override=wrapped)
            result={'variant':name,'metrics':row,'group_grad_norm':gradient_capture[-1]}
            report['input_counterfactuals'].append(result);save()
            print('counterfactual',name,'loss',row['loss'],'norm',row['grad_norm'],flush=True)
        # Current trained KDA gates sampled without modifying any input.
        captured={};hooks=[]
        for name,module in model.named_modules():
            if name.endswith(('.decay','.beta','.gate')) and name.startswith('temporal.'):
                def capture(_module,_inputs,output,name=name):
                    value=output.detach().float().flatten();stride=max(1,value.numel()//128)
                    captured.setdefault(name,[]).append(value[::stride][:128].cpu())
                hooks.append(module.register_forward_hook(capture))
        one(252,0)
        for hook in hooks:hook.remove()
        for name,values in captured.items():
            raw=torch.cat(values)
            value=(-torch.nn.functional.softplus(raw)).exp() if name.endswith('.decay') else raw.sigmoid()
            report['gate_stats'][name]={'quantiles':torch.quantile(value,torch.tensor([0.,.01,.5,.99,1.])).tolist(),
                'fraction_below_0_01':float((value<.01).float().mean()),'fraction_above_0_99':float((value>.99).float().mean()),'samples':len(value)}
        report['fixed_fit'].append({'update':0,'train':summaries(report['training_cases']),'control':summaries(report['control_cases'])});save()
        # Counterfactual Adam moments were updated at lr0. Restore the original
        # optimizer before the fit arm; this does not duplicate any CPU model.
        parent=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        saved={**parent['optimizer'],'param_groups':[{**g,'lr':3e-5,'fused':True,'foreach':False} for g in parent['optimizer']['param_groups']]}
        optimizer.load_state_dict(saved)
        for parameter,values in optimizer.state.items():
            if isinstance(values.get('step'),torch.Tensor):values['step']=values['step'].to(parameter.device,dtype=torch.float32)
        del parent,saved;gc.collect()
        for step in range(1,args.steps+1):
            gradient_capture.clear();case=report['training_cases'][(step-1)%len(report['training_cases'])]
            row=one(*case,training=True)
            report['gradients'].append({'update':step,'loss':row['loss'],'grad_norm':row['grad_norm'],'group_grad_norm':gradient_capture[-1],
                'optimizer_stepped':row['optimizer_stepped'],'overflow':row['amp_overflow_skipped']})
            if step%16==0 or step==args.steps:
                result={'update':step,'train':summaries(report['training_cases']),'control':summaries(report['control_cases'])}
                report['fixed_fit'].append(result);save();print('fixed_fit',json.dumps(result),flush=True)
        report['rss_note']='6GiB RSS guard, mmap checkpoint + direct meta-to-CUDA model; no CPU model copy'
        save();print('diagnosis_saved',args.out,flush=True)
    finally:
        torch.nn.utils.clip_grad_norm_=original_clip


if __name__=='__main__':main()
