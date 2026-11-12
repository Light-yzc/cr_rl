"""Local/CUDA precision probe; no data mutation, checkpoint or optimizer restore."""
import argparse
import json
from pathlib import Path
import time
import torch
from cr_rl.models.state_temporal import StateTemporalConfig,StateTemporalCore,KDAState,GQAState

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--widths',default='768,1024,1280');args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261110)
    results=[]
    for width in map(int,args.widths.split(',')):
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        cfg=StateTemporalConfig(width=width,heads=width//64,kv_heads=width//256,head_dim=64)
        with torch.device('cuda'):model=StateTemporalCore(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5)
        scaler=torch.amp.GradScaler('cuda',init_scale=32.)
        state=model.init_state(1,'cuda',torch.float16)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            _,state=model.prefill(torch.randn(1,1200,width,device='cuda'),state)
        row={'width':width,'prefix':1200,'steps':[]}
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            # An optimizer update expires memory: rebuild its entire prefix.
            if i:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    _,state=model.prefill(torch.randn(1,1200,width,device='cuda'))
            with torch.autocast('cuda',dtype=torch.float16):
                output,state=model.prefill(torch.randn(1,32,width,device='cuda'),state)
                target=torch.randn_like(output)
                loss=(output.float()-target.float()).square().mean()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert torch.isfinite(loss) and torch.isfinite(output).all()
            matrix_max=max(s.matrix.abs().max().item() for s in state.layers if isinstance(s,KDAState))
            assert all(torch.isfinite(s.matrix).all() for s in state.layers if isinstance(s,KDAState))
            assert all(torch.isfinite(s.key).all() and torch.isfinite(s.value).all()
                       for s in state.layers if isinstance(s,GQAState))
            state=state.detach();scaler.step(optimizer);scaler.update()
            row['steps'].append({'loss':loss.item(),'unclipped_grad_norm':norm.item(),
                                 'scaler':scaler.get_scale(),'state_abs_max':matrix_max})
        row.update(seconds=time.perf_counter()-started,
                   peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                   parameters=sum(p.numel() for p in model.parameters()))
        results.append(row);print(json.dumps(row),flush=True)
        del model,optimizer,scaler,state,output,loss,norm,target
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'fp16_autocast':True,
        'fp32_kda_state':True,'results':results},indent=2),encoding='utf-8')

if __name__=='__main__':main()
