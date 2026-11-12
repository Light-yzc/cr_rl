"""Real V100/FP32-frame, random-history dedup gate; never writes production weights."""
import copy
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import HistoryValueNetwork, ModelConfig
from cr_rl.training.fp16 import backward_loss, configure_optimizer_amp
from cr_rl.training.host_memory import detach_optimizer_counters, release_host_memory
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import _optimizer_step, autocast_for, forward_inputs, move_inputs

os.environ['CRRL_AMP_DTYPE']='fp16'
torch.set_num_threads(4)
root=Path(__file__).resolve().parents[1]
run=root/'artifacts/local_dual_v100'
out=root/'artifacts/local_dual_v100_checks/v100_dedup_benchmark_v2.json'
state=torch.load(run/'ppo_latest.pt',mmap=True,map_location='cpu',weights_only=False)
cfg=ModelConfig(**state['critic_config'])
assert cfg.frame_encoder_fp32
source_update=state['updates']
env=CrForgeVector(1,1,independent=True)
adapter=CrForgeObservationAdapter(env.vector.schema,env.vector.vocabulary_size,entity_bucket=16)
env.close()
with np.load(run/'rollouts'/f'rollout_{source_update:06d}.npz') as r:
    obs,masks=r['obs'],r['masks']
    eligible=r['learner_mask'].astype(bool)&r['valid'].astype(bool)
    episode_steps=r['episode_steps']
    old_values=r['old_values']
index=adapter.schema.header['entity_count'][0]
rows=[(int(obs[t,l,s,index]),int(t),int(l),int(s)) for t,l,s in np.argwhere(eligible)
      if t>=cfg.context_len-1 and episode_steps[t,l]>=cfg.context_len-1]
rng=np.random.default_rng(20261109)
scenarios={
    'random64':[rows[i] for i in rng.choice(len(rows),64,replace=False)],
    'dense64':sorted(rows,reverse=True)[:64],
}
report=dict(source_update=source_update,frame_encoder_fp32=cfg.frame_encoder_fp32,
            history=cfg.context_len,effective_batch=64,microbatch=16,
            gates=dict(output_max_abs=0.01,gradient_cosine_min=0.999,parameter_max_abs=1e-5),
            scope='Same raw histories, three nonzero Adam steps; random case represents sampling; dense case stress only',scenarios={})


def gradient_cosine(current, reference):
    totals=torch.zeros(3,device=current.device,dtype=torch.float64)
    for left in range(0,current.numel(),2**20):
        a=current[left:left+2**20].double()
        b=reference[left:left+2**20].to(current.device).double()
        totals[0].add_(torch.dot(a,b))
        totals[1].add_(torch.dot(a,a))
        totals[2].add_(torch.dot(b,b))
    return float(totals[0]/(totals[1]*totals[2]).sqrt().clamp_min(1e-30))


def save():
    temp=out.with_suffix('.tmp.json')
    temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out)


for scenario,chosen in scenarios.items():
    traces={lane:SimpleNamespace(observations=list(obs[:,lane]),masks=list(masks[:,lane]))
            for lane in range(obs.shape[1])}
    refs=[(traces[lane],t,seat) for _,t,lane,seat in chosen]
    ordinary=PreparedHistoryBatches(refs,cfg.context_len,adapter,'cuda:0')
    packed=copy.copy(ordinary);packed.deduplicate_frames=True;packed.metadata=dict(ordinary.metadata)
    targets=torch.tensor([old_values[t,lane,seat]+0.01 for _,t,lane,seat in chosen],device='cuda:1')
    model=HistoryValueNetwork(cfg).to('cuda:1')
    baseline_parameters=None
    baseline_predictions=[]
    baseline_gradients=[]
    case=[]
    for mode in ('baseline','dedup','baseline_repeat','dedup_repeat'):
        model.load_state_dict(state['critic'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        optimizer.load_state_dict(state['critic_optimizer']);detach_optimizer_counters(optimizer)
        configure_optimizer_amp(optimizer,'cuda:1',state.get('critic_amp_scaler'))
        prepared=packed if mode.startswith('dedup') else ordinary
        prepared.metadata['frames_requested']=prepared.metadata['frames_encoded']=0
        optimizer.zero_grad(set_to_none=True)
        model.train();torch.manual_seed(819)
        # Exercise backward kernels before timed scenarios, without touching weights or Adam.
        for _ in range(2):
            inputs=move_inputs(prepared(np.arange(16)),'cuda:1')
            with autocast_for('cuda:1'):
                loss=.5*(forward_inputs(model,inputs).float()-targets[:16]).square().mean()
            backward_loss(loss*16,optimizer);optimizer.zero_grad(set_to_none=True)
        del inputs,loss
        torch.cuda.synchronize();torch.cuda.synchronize(1)
        prepared.metadata['frames_requested']=prepared.metadata['frames_encoded']=0
        torch.cuda.reset_peak_memory_stats(0);torch.cuda.reset_peak_memory_stats(1)
        elapsed=0.;output_error=0.;cosines=[];norms=[]
        try:
            for step in range(3):
                torch.cuda.synchronize(1);began=time.perf_counter()
                predictions=[]
                for left in range(0,64,16):
                    indices=np.arange(left,left+16)
                    inputs=move_inputs(prepared(indices),'cuda:1')
                    with autocast_for('cuda:1'):
                        prediction=forward_inputs(model,inputs).float()
                        loss=.5*(prediction-targets[indices]).square().mean()
                    backward_loss(loss*16,optimizer)
                    predictions.append(prediction.detach())
                torch.cuda.synchronize(1);elapsed+=time.perf_counter()-began
                prediction=torch.cat(predictions)
                grads=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])
                if mode=='baseline':
                    baseline_predictions.append(prediction.cpu().clone())
                    baseline_gradients.append(grads.to('cuda:0'))
                    cosines.append(1.)
                else:
                    output_error=max(output_error,float((prediction.cpu()-baseline_predictions[step]).abs().max()))
                    cosines.append(gradient_cosine(grads,baseline_gradients[step]))
                del grads,prediction,predictions,inputs,loss
                torch.cuda.synchronize(1);began=time.perf_counter()
                norm=_optimizer_step(model,optimizer,64,.5)
                torch.cuda.synchronize(1);elapsed+=time.perf_counter()-began
                if norm is None:
                    raise FloatingPointError('AMP overflow in isolated probe; no matched three-step result')
                norms.append(norm)
            parameters={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if baseline_parameters is None:baseline_parameters=parameters
            error=max(float((parameters[k]-baseline_parameters[k]).abs().max()) for k in parameters)
            row=dict(mode=mode,seconds=elapsed,samples_per_second=192/elapsed,output_max_abs=output_error,
                     gradient_cosine_min=min(cosines),parameter_max_abs=error,
                     numeric_gate=error<=1e-5 and output_error<=.01 and min(cosines)>=.999,
                     nonzero_adam_steps=len(norms),grad_norms=norms,
                     frames_requested=prepared.metadata['frames_requested'],frames_encoded=prepared.metadata['frames_encoded'],
                     storage_device=str(prepared.storage_device),peak_primary_gib=torch.cuda.max_memory_allocated(1)/2**30,
                     peak_data_gpu_gib=torch.cuda.max_memory_allocated(0)/2**30)
            del parameters
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
            row=dict(mode=mode,error=str(error),numeric_gate=False,nonzero_adam_steps=len(norms))
        case.append(row);report['scenarios'][scenario]=case;save();print(json.dumps(dict(scenario=scenario,**row)),flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        release_host_memory()
        if 'error' in row:break
    del model,ordinary,packed,traces,refs,targets,baseline_parameters,baseline_gradients,baseline_predictions
    gc.collect();torch.cuda.empty_cache()
print(json.dumps(report),flush=True)
