"""Snapshot completed labeled live sessions in one pass, excluding duplicate exports."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
source = root/'artifacts/nulls_live/preset_60_20261117'
out = root/'artifacts/colab_cpu_prep_20261117/live_sessions.zip'
games = []
with zipfile.ZipFile(out.with_suffix('.partial'),'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
    for run in sorted(source.iterdir()):
        if not run.is_dir() or not (run/'result.json').is_file(): continue
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf8'))
        if not manifest.get('collection_complete') or not manifest.get('outcome_verified'): continue
        paths = [run/n for n in ('manifest.json','result.json','terminal.json','events.jsonl')]
        paths += list((run/'diagnostics').glob('frames_*.jsonl.gz'))
        if not all(p.is_file() for p in paths) or len(paths)<5: continue
        for path in paths:
            archive.write(path, path.relative_to(source).as_posix())
        games.append(run.name)
    archive.writestr('snapshot.json',json.dumps(dict(games=games,checksum_scans=False)))
out.with_suffix('.partial').replace(out)
print(json.dumps(dict(path=str(out),games=len(games),bytes=out.stat().st_size,checksum_scans=False)))
