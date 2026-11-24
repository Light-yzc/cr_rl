"""One-pass allowlisted code ZIP; no checksum or archive reread passes."""
import argparse
import json
from pathlib import Path
import zipfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def build(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out.with_suffix('.partial'), 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=1) as archive:
        for directory, suffixes in [('src',{'.py'}),('scripts',{'.py','.sh','.ps1'}),
                ('configs',{'.json','.yaml'}),('tests',{'.py'}),
                ('notebooks',{'.ipynb'}),('docs',{'.md'})]:
            for p in (ROOT/directory).rglob('*'):
                if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:
                    archive.write(p, p.relative_to(ROOT).as_posix())
                    count += 1
        archive.write(ROOT/'pyproject.toml','pyproject.toml')
        revision = subprocess.check_output(['git','-C',str(ROOT/'.cache/vendor/crforge'),
                                             'rev-parse','HEAD'],text=True).strip()
        archive.writestr('.cache/vendor/crforge/PACKAGED_REVISION', revision+'\n')
    out.with_suffix('.partial').replace(out)
    print(json.dumps(dict(path=str(out),files=count,bytes=out.stat().st_size,checksum_scans=False)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    build(p.parse_args().out)
