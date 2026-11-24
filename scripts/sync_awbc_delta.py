"""Package and upload only new finalized recordings relative to a baseline ZIP."""
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

root=Path(__file__).resolve().parents[1]
work=root/'artifacts/colab_cpu_prep_20261117'
source=root/'artifacts/nulls_live/preset_60_20261117'
with zipfile.ZipFile(work/'live_sessions.zip') as base:
    known=set(json.loads(base.read('snapshot.json'))['games'])
with tempfile.TemporaryDirectory(prefix='awbc_delta_',dir=work) as folder:
    folder=Path(folder)
    archive=folder/'delta.zip'
    added=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for run in sorted(source.iterdir()):
            if not run.is_dir() or run.name in known or not (run/'result.json').is_file():
                continue
            manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
            if not manifest.get('collection_complete') or not manifest.get('outcome_verified'):
                continue
            paths=[run/n for n in ('manifest.json','result.json','terminal.json','events.jsonl')]
            paths+=list((run/'diagnostics').glob('frames_*.jsonl.gz'))
            if len(paths)<5 or not all(p.is_file() for p in paths):
                continue
            for p in paths: z.write(p,p.relative_to(source).as_posix())
            added.append(run.name)
        z.writestr('delta.json',json.dumps(dict(games=added)))
    def upload(path,name):
        for _ in range(3):
            result=subprocess.run(['colab','upload','-s','cr-train-l4-1117',str(path),
                                   '/content/'+name],capture_output=True,timeout=75)
            if result.returncode==0: return
        raise RuntimeError('Upload failed: '+name)
    parts=[]
    with archive.open('rb') as stream:
        while block:=stream.read(8*1024**2):
            part=folder/f'crrl_live_delta_{len(parts):03d}'
            part.write_bytes(block)
            upload(part,part.name)
            parts.append(dict(name=part.name,bytes=len(block)))
    receipt=folder/'crrl_live_delta_receipt.json'
    receipt.write_text(json.dumps(dict(parts=parts,games=added)),encoding='utf8')
    upload(receipt,receipt.name)
    print(json.dumps(dict(baseline=len(known),new_games=len(added),
                          uploaded_bytes=archive.stat().st_size)),flush=True)
