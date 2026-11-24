"""Package current training code and pinned runtime, without checkpoints/data."""
import hashlib
import json
import zipfile
from pathlib import Path

from build_colab_ppo_upload import SECRET

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    files = {}
    for directory, suffixes in [('src', {'.py'}), ('scripts', {'.py', '.sh', '.ps1'}),
                                ('configs', {'.json', '.yaml'}), ('tests', {'.py'}),
                                ('notebooks', {'.ipynb'}), ('docs', {'.md'})]:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:
                body = path.read_bytes()
                if SECRET.search(body):
                    raise ValueError('Credential-like content in '+str(path.relative_to(ROOT)))
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ('pyproject.toml', 'README.md'):
        files[name] = ROOT/name
    vendor = ROOT/'.cache/vendor/crforge'
    for directory in ('python', 'env/build/install/env'):
        for path in (vendor/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and not any(
                    part.endswith('.egg-info') for part in path.parts):
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ('LICENSE', 'NOTICE', 'crforge-data.lock'):
        files['.cache/vendor/crforge/'+name] = vendor/name
    for path in (Path.home()/'.crforge/tables/16.635.3').rglob('*'):
        if path.is_file():
            files['assets/tables/16.635.3/'+path.name] = path
    records = []
    partial = output.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            body = path.read_bytes()
            archive.writestr(name, body)
            records.append(dict(path=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
        archive.writestr('CODE_MANIFEST.json', json.dumps(records, indent=2))
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():
            raise IOError('ZIP CRC verification failed')
    partial.replace(output)
    print(json.dumps(dict(path=str(output), bytes=output.stat().st_size, files=len(records),
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
