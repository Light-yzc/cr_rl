import json,os,signal,time
from pathlib import Path
out=Path('/content/crrl_training/runs/longrun');w=Path('/content/crrl_inbox_upgrade')
backup=Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun/latest.json')
summary=json.loads((out/'summary.json').read_text());drive=json.loads(backup.read_text());age=time.time()-(out/'progress.jsonl').stat().st_mtime
print('HANG_AUDIT',dict(update=summary['update'],drive=drive['update'],no_progress_seconds=age))
if age>180 and summary['update']==980 and drive['update']>=980:
 import torch
 ckpt=torch.load(out/'ppo_latest.pt',map_location='cpu',mmap=True,weights_only=False);assert int(ckpt['updates'])==980;del ckpt
 for pid,needle in ((92169,b'/crrl_deploy_inbox_no_valid.py'),(58045,b'/train_selfplay.py')):
  p=Path(f'/proc/{pid}/cmdline')
  if p.exists() and needle in p.read_bytes():os.kill(pid,signal.SIGKILL);print('TERMINATED_VERIFIED_PID',pid)
 (w/'recovered_native_hang.json').write_text(json.dumps(dict(update=980,reason='native_lane_stall_after_completed_checkpoint',discarded_partial_update=981)))
else:print('No forced stop performed')
