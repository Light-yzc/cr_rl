from pathlib import Path
import json
for p in sorted(Path('/content/crrl_training/runs').glob('async_profile_*/summary.json'))[-2:]:
 d=json.loads(p.read_text());print(p.parent.name,json.dumps({k:d.get(k) for k in ('update','seconds','samples_per_second','metrics','optimizer_steps','critic_optimizer_steps','actor_overflow_steps','critic_overflow_steps','peak_cuda_allocated_gib')}))
print('DRIVE',Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun/latest.json').read_text())
