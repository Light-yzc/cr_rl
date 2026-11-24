"""Small read-only status for the resident opponent cache upgrade."""
import json
import subprocess
from pathlib import Path
root=Path('/content/crrl_training')
work=Path('/content/crrl_cache_upgrade')
memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
print('HOST_RAM_AVAILABLE',memory.get('MemAvailable'))
for name in ('upgrade.log','result.json'):
    p=work/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2600));print(name,f.read().decode(errors='replace'))
launch=root/'runs/longrun/launch.json'
if launch.exists():
    args=json.loads(launch.read_text())
    print('CACHE_ARGS',{k:args[args.index(k)+1] for k in ('--resume','--opponent-cache-size','--training-opponent-cache-size')})
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=path.read_bytes().split(b'\0')
    except OSError: continue
    if any(a.endswith((b'/crrl_raise_opponent_cache.py',b'/train_selfplay.py')) for a in args):
        print('PROCESS',path.parent.name)
        print(subprocess.run(['ps','-p',path.parent.name,'-o','stat=,etime=,time=,wchan='],capture_output=True,text=True).stdout.strip())
        if b'/content/crrl_training/scripts/train_selfplay.py' in args:
            print('PROCESS_IO',Path('/proc',path.parent.name,'io').read_text())
progress=root/'runs/longrun/progress.jsonl'
if progress.exists():
    with progress.open('rb') as f:
        f.seek(max(0,progress.stat().st_size-16000))
        for line in f.read().splitlines()[-4:]:
            try:r=json.loads(line)
            except ValueError:continue
            print('EVENT',{k:r[k] for k in ('phase','update','reason','ran') if k in r})
