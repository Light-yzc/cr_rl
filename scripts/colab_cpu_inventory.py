"""Bounded, read-only inventory for a mounted CPU preparation runtime."""
import json
import shutil
from pathlib import Path

root = Path('/content/drive/MyDrive')
print(json.dumps({'drive_mounted': root.is_dir(),
                  'local_free_gib': round(shutil.disk_usage('/content').free/1024**3, 2)}))
if not root.is_dir():
    raise RuntimeError('Mount Drive first')
print('MYDRIVE_TOP', json.dumps([x.name for x in root.iterdir()][:60]))
for name in ('cr_rl_ppo', 'cr_rl_live_ppo', 'cr_rl', 'crrl', 'cr_rl_live_awbc'):
    folder = root/name
    if not folder.is_dir():
        continue
    print('FOLDER', name)
    level = [folder]
    for depth in range(3):
        next_level = []
        for directory in level[:30]:
            for path in list(directory.iterdir())[:80]:
                if path.is_file() and (path.suffix in ('.pt', '.zip') or
                                      'manifest' in path.name or path.name == 'batch.json'):
                    print(json.dumps({'path':str(path.relative_to(root)),
                                      'bytes':path.stat().st_size}))
                elif path.is_dir() and path.name not in ('wandb', 'cache', '.git', 'seats'):
                    next_level.append(path)
        level = next_level
