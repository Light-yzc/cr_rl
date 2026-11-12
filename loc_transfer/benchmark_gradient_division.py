"""Compare elementwise vs foreach gradient division with real Actor parameter tensor shapes."""
import gc
import json
import time
from pathlib import Path

import torch

root=Path(__file__).resolve().parents[1]
run=root/'artifacts/local_dual_v100'
torch.set_num_threads(2)
state=torch.load(run/'actor_export.pt',mmap=True,map_location='cpu',weights_only=False)
shapes=[v.shape for v in state['model'].values() if v.is_floating_point()]
del state;gc.collect()
generator=torch.Generator(device='cuda:0').manual_seed(819)
source=[torch.randn(s,device='cuda:0',generator=generator) for s in shapes]
results=[]
for samples in (64,58):
    for mode in ('loop','foreach'):
        work=[v.clone() for v in source]
        if mode=='loop':
            for v in work:v.div_(samples)
        else:torch._foreach_div_(work,samples)
        torch.cuda.synchronize()
        maximum = 0.0
        if mode=='loop':expected=work
        else:
            maximum=max(float((a-b).abs().max()) for a,b in zip(expected,work))
        elapsed=[]
        for repeat in range(4):
            torch.cuda.synchronize();began=time.perf_counter()
            # Reciprocal-free alternating operations avoid cumulative underflow.
            for iteration in range(100):
                if mode=='loop':
                    for v in work:v.div_(samples)
                    for v in work:v.mul_(samples)
                else:
                    torch._foreach_div_(work,samples)
                    torch._foreach_mul_(work,samples)
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-began)
        results.append(dict(samples=samples,mode=mode,seconds=elapsed,tensor_count=len(work), max_abs_diff_vs_loop=maximum, bitwise_equal_to_loop=maximum==0))
        del work
report=dict(source='Real Actor export parameter shapes; synthetic gradients; actual gradients/Adam checked separately', arithmetic_scope='Raw division diagnostic only; production acceptance requires actual multi-step parameter gate <=1e-5', results=results)
(root/'artifacts/local_dual_v100_checks/gradient_division_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
