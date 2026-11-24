import json
from pathlib import Path
root=Path('/content/crrl_training');out=root/'runs/longrun';work=Path('/content/crrl_inbox_upgrade')
for name in ('upgrade.log','result.json'):
 p=work/name
 if p.exists():print(name,p.read_text()[-2200:])
args=json.loads((out/'launch.json').read_text());print('VALID_INTERVAL',args[args.index('--validation-interval')+1]);print('AWBC_INTERVAL',args[args.index('--awbc-interval')+1])
for name in ('awbc_inbox_state.status.json',):
 p=root/'prepared'/name
 if p.exists():print(name,p.read_text())
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=path.read_bytes().split(b'\0')
 except OSError:continue
 if any(a.endswith((b'/train_selfplay.py',b'/sync_awbc_inbox.py')) for a in args):print('PROCESS',path.parent.name,[a.decode() for a in args if a.endswith(b'.py')])
print('LATEST_UPDATE',json.loads((out/'summary.json').read_text())['update'])
