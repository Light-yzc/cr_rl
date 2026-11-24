"""SSD staging for restartable Colab jobs. One copy, one extraction, no hashes."""
import json
from pathlib import Path
import shutil
import zipfile

from cr_rl.training.fast_publication import copy_once


def extract_zip(archive, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (target/name).resolve().is_relative_to(target.resolve()):
                raise ValueError('Archive path escapes target')
        z.extractall(target)


def stage_zip(source, target, *, archive_dir=None):
    source, target = Path(source), Path(target)
    marker = target/'.staged.json'
    if marker.exists(): return target
    archive_dir = Path(archive_dir or target.parent/'archives')
    archive_dir.mkdir(parents=True,exist_ok=True)
    archive = archive_dir/source.name
    if not archive.exists():
        if shutil.disk_usage(archive_dir).free < source.stat().st_size*3+12*1024**3:
            raise RuntimeError('Insufficient SSD space for ZIP and extraction')
        copy_once(source,archive)
    extract_zip(archive,target)
    marker.write_text(json.dumps(dict(source=str(source),bytes=source.stat().st_size,
                                      checksum_scans=False)))
    return target
