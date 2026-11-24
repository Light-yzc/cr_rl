"""Validate only new recordings, then publish consistent Drive and live SSD copies."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

content=Path('/content')
root=content/'crrl_training'
sessions=root/'sessions'
drive=content/'drive/MyDrive/cr_rl_ppo/preparation/ssd_snapshots/live_sessions.zip'
baseline=root/'archives/live_sessions.zip'
if not baseline.is_file(): baseline=content/'live_sessions.zip'
sys.path.insert(0,str(root/'src'))
from cr_rl.client.rl_recording import load_episode

receipt=json.loads((content/'crrl_live_delta_receipt.json').read_text())
with tempfile.TemporaryDirectory(prefix='awbc_admit_',dir=content) as folder:
    folder=Path(folder)
    delta=folder/'delta.zip'
    with delta.open('wb') as dst:
        for part in receipt['parts']:
            p=content/part['name']
            if p.stat().st_size!=part['bytes']: raise IOError('Incomplete delta part')
            with p.open('rb') as src: shutil.copyfileobj(src,dst,4*1024**2)
    pending=folder/'sessions';pending.mkdir()
    games=set(receipt['games'])
    with zipfile.ZipFile(delta) as z:
        if set(json.loads(z.read('delta.json'))['games'])!=games:
            raise ValueError('Transport inventory and archive differ')
        for member in z.infolist():
            if member.filename=='delta.json': continue
            name=Path(member.filename)
            if (name.parts[0] not in games or
                    not (pending/name).resolve().is_relative_to(pending.resolve())):
                raise ValueError('Unexpected archive path')
            z.extract(member,pending)
    accepted=[];rejected={}
    for name in sorted(games):
        if (sessions/name).exists(): continue
        try:
            episode=load_episode(pending/name,purpose='awbc')
            if not episode['actor_keep'].any(): raise ValueError('No usable Actor labels')
            accepted.append(name)
            del episode
        except (ValueError,KeyError,OSError,EOFError) as exc:
            rejected[name]=str(exc)
    if accepted:
        merged=folder/'live_sessions.zip'
        with zipfile.ZipFile(baseline) as z:
            inventories=[n for n in z.namelist() if n.startswith('delta_inventory_')]
            head=max(inventories,key=lambda n:int(n.rsplit('_',1)[1].split('.')[0])) if inventories else 'snapshot.json'
            previous=json.loads(z.read(head))['games']
        existing={p.parent.name for p in sessions.glob('*/result.json')}
        if not existing.issubset(previous):
            raise RuntimeError('SSD contains newer sessions absent from the durable baseline')
        shutil.copyfile(baseline,merged)
        # Do not add a second snapshot.json entry when appending; keep the current
        # complete inventory in the versioned extension.
        added_to_archive=[name for name in accepted if name not in previous]
        generation=len(previous)+len(added_to_archive)
        with zipfile.ZipFile(merged,'a',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name in added_to_archive:
                for p in (pending/name).rglob('*'):
                    if p.is_file(): z.write(p,p.relative_to(pending).as_posix())
            if added_to_archive:
                z.writestr(f'delta_inventory_{generation}.json',json.dumps(dict(games=previous+added_to_archive)))
        temporary=drive.with_suffix('.partial')
        shutil.copyfile(merged,temporary)
        if temporary.stat().st_size!=merged.stat().st_size: raise IOError('Incomplete Drive copy')
        temporary.replace(drive)
        merged.replace(baseline)
        for name in accepted: os.replace(pending/name,sessions/name)
    print(json.dumps(dict(added=len(accepted),rejected=rejected,
          ssd_games=len(list(sessions.glob('*/result.json'))),drive_zip_bytes=drive.stat().st_size,
          checksum_scans=False)),flush=True)
