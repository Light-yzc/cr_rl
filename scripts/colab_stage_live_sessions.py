"""Join transport chunks once, persist one ZIP, and extract on SSD."""
import json
from pathlib import Path
import shutil
import zipfile

root = Path('/content/crrl_training')
archive = Path('/content/live_sessions.zip')
with archive.with_suffix('.partial').open('wb') as target:
    for i in range(16):
        with Path(f'/content/live_part_{i:03d}').open('rb') as source:
            shutil.copyfileobj(source,target,4*1024**2)
archive.with_suffix('.partial').replace(archive)
backup = Path('/content/drive/MyDrive/cr_rl_ppo/preparation/ssd_snapshots/live_sessions.zip')
temp = backup.with_suffix('.partial')
shutil.copyfile(archive,temp)
if temp.stat().st_size != archive.stat().st_size: raise IOError('Incomplete ZIP copy')
temp.replace(backup)
target = root/'sessions'
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if not (target/name).resolve().is_relative_to(target.resolve()): raise ValueError('Unsafe ZIP path')
    z.extractall(target)
print(json.dumps(dict(games=len(list(target.glob('*/result.json'))),
    local_path=str(target),archive=str(backup),checksum_scans=False)))
