"""Read-only, small-file probe; no model loading or training changes."""
import json
import time
from pathlib import Path

root = Path('/content/crrl_training')
out = root / 'runs/longrun'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


processes = {'trainer': [], 'receiver': [], 'controller': []}
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().split(b'\0')
    except OSError:
        continue
    for role, name in [('trainer', b'/train_selfplay.py'),
                       ('receiver', b'/sync_awbc_inbox.py'),
                       ('controller', b'/cr_train_pipeline.py')]:
        if any(arg.endswith(name) for arg in args):
            processes[role].append(int(path.parent.name))
summary = read(out / 'summary.json')
activity = [p.stat().st_mtime for p in
            [out / 'summary.json', out / 'progress.jsonl', out / 'console.log'] if p.exists()]
report = dict(processes=processes, update=summary.get('update'),
              activity_age_seconds=time.time() - max(activity) if activity else None,
              samples_per_second=summary.get('samples_per_second'),
              faulted_samples=summary.get('faulted_samples'),
              real_games=len(list((root / 'sessions').glob('*/result.json'))),
              receiver_status=read(root / 'prepared/awbc_inbox_state.status.json'))
print('FLYWHEEL_HEALTH_JSON ' + json.dumps(report), flush=True)
