import os,signal,json
from pathlib import Path
out=Path('/content/crrl_training/runs/longrun')
pid=int((out/'pipeline.pid').read_text());assert b'/cr_train_pipeline.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGINT)
print('GRACEFUL_STOP_REQUESTED',pid)
