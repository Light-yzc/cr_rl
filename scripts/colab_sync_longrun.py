"""Sync uploaded code and stage immutable ZIP snapshots on the GPU SSD."""
import json
import os
from pathlib import Path
import shutil
import zipfile

root = Path('/content/crrl_training')
drive = Path('/content/drive/MyDrive/cr_rl_ppo')

def extract(archive_path, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        for name in z.namelist():
            if not (target/name).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(target)

extract('/content/crrl_training_update.zip', root)
shutil.copyfile('/content/crrl_training_update.zip', drive/'preparation/crrl_training_update.zip')
mode = os.environ.get('CRRL_SNAPSHOT_MODE', 'smoke')
report_path = drive/'preparation/ssd_snapshots'/f'{mode}_snapshot.json'
report = json.loads(report_path.read_text())
destination = root/'datasets'/report['snapshot_id']
if not (destination/'dataset.manifest.json').is_file():
    archive = root/'datasets'/Path(report['archive']).name
    archive.parent.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < report['bytes']*3 + 12*1024**3:
        raise RuntimeError('Insufficient SSD headroom')
    print('COPY_ZIP', report['bytes'], flush=True)
    temporary = archive.with_suffix('.partial')
    shutil.copyfile(report['archive'], temporary)
    if temporary.stat().st_size != report['bytes']:
        raise IOError('Incomplete archive copy')
    temporary.replace(archive)
    extract(archive, destination)
(root/f'prepared/{mode}_dataset.json').write_text(json.dumps(dict(report, local_path=str(destination))))
print(json.dumps(dict(stage='ssd_ready', mode=mode, dataset=str(destination),
    games=report['games'], checksum_scans=False)), flush=True)
