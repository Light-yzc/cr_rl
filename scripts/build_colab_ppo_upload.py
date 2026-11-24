"""Stream an explicit, credential-free PPO handoff into one verified ZIP64 archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'build', '.gradle',
            'node_modules', 'wandb'}
SECRET = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{30,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,})')


def sources():
    pairs = {}
    def add(path, target=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        target = target or path.relative_to(ROOT).as_posix()
        if path.suffix in {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.sh', '.ps1', '.ipynb'}:
            if SECRET.search(path.read_bytes()):
                raise ValueError(f'Credential-like value in selected text file: {target}')
        if target in pairs and pairs[target] != path:
            raise ValueError(f'Duplicate bundle path: {target}')
        pairs[target] = path
    for directory, suffixes in [('src', {'.py'}), ('scripts', {'.py', '.sh', '.ps1', '.java', '.c'}),
                                 ('configs', {'.yaml', '.yml', '.json'}), ('tests', {'.py'})]:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in suffixes and not (set(path.parts) & EXCLUDED):
                add(path)
    for name in ['pyproject.toml', 'README.md', '.gitignore',
                 'docs/PPO_LIVE_FAILURE_CURRICULUM_DRAFT_20261116.md',
                 'docs/LIVE_POLICY_BEHAVIOR_DIAGNOSIS_20261116.md',
                 'docs/NULLS_ACTOR_DIAGNOSIS_20261115.md',
                 'docs/COLAB_PPO_UPLOAD_20261116.md',
                 'notebooks/colab_ppo_prepare.ipynb']:
        add(ROOT / name)
    # Preserve original paths so the new single-GPU runner can rebase explicitly.
    for name in ['artifacts/ppo_v3_v100/ppo_latest.pt',
                 'artifacts/ppo_v3_v100/actor_best_val.pt',
                 'artifacts/ppo_v3_v100_bak_u480/actor_export.pt',
                 'artifacts/ppo_v3_v100_bak_u275/actor_export.pt',
                 'artifacts/ppo_prep_20261113/best.pt',
                 'artifacts/ppo_v3_v100/run_config.json',
                 'artifacts/ppo_v3_v100/best_validation.json',
                 'artifacts/ppo_v3_v100/summary.json']:
        add(ROOT / name)
    vendor = ROOT / '.cache/vendor/crforge'
    # No Gradle downloads or rebuild needed: ship the pinned compiled JVM runtime.
    for subtree in ['python', 'env/build/install/env']:
        for path in (vendor / subtree).rglob('*'):
            if path.is_file() and not (set(path.relative_to(vendor / subtree).parts) & EXCLUDED):
                if not any(part.endswith('.egg-info') for part in path.parts):
                    add(path)
    for name in ['LICENSE', 'NOTICE', 'README.md', 'crforge-data.lock']:
        add(vendor / name)
    tables = Path.home() / '.crforge/tables/16.635.3'
    for path in tables.rglob('*'):
        if path.is_file():
            add(path, 'assets/tables/16.635.3/' + path.relative_to(tables).as_posix())
    for run in ['play_20261115_210255', 'play_20261116_221405']:
        folder = ROOT / 'artifacts/nulls_live' / run
        # Small forensic materials only; no raw memory/large observation recordings.
        for name in ['events.jsonl', 'manifest.json', 'behavior_head_ablations.json',
                     'behavior_sensitivity.json', 'spell_guard_recheck.json',
                     'sampling_controller_replay.json']:
            if (folder / name).is_file():
                add(folder / name)
    return sorted(pairs.items())


def build(output):
    pairs = sources()
    total = sum(p.stat().st_size for _, p in pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < total + 2 * 1024**3:
        raise RuntimeError('Not enough space for an uncompressed ZIP plus 2GiB reserve')
    partial = output.with_suffix('.zip.partial')
    if partial.exists():
        raise FileExistsError(partial)
    started = time.monotonic()
    manifest = {'format': 'crrl-colab-upload-v1', 'root': 'cr_rl',
                'purpose': 'Preparation bundle; new single-GPU PPO launcher/curriculum still pending.',
                'files': []}
    done = 0
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for target, path in pairs:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open('cr_rl/' + target, 'w', force_zip64=True) as dest:
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    dest.write(block)
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed during packaging: {target}')
            manifest['files'].append({'path': target, 'bytes': before.st_size, 'sha256': digest.hexdigest()})
            done += before.st_size
            if before.st_size > 100_000_000:
                print(json.dumps({'stage': 'packed', 'file': target, 'bytes_done': done,
                                  'bytes_total': total, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
        manifest['total_file_bytes'] = total
        try:
            manifest['workspace_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError:
            manifest['workspace_commit'] = None
        manifest['note'] = 'Working-tree file hashes are authoritative; includes uncommitted fixes.'
        body = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        archive.writestr('cr_rl/BUNDLE_MANIFEST.json', body)
    print('Verifying every archive member CRC...', flush=True)
    with zipfile.ZipFile(partial) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f'Bad archive member: {broken}')
    digest = hashlib.sha256()
    with partial.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    partial.rename(output)
    output.with_suffix('.zip.sha256').write_text(digest.hexdigest() + '  ' + output.name + '\n')
    output.with_suffix('.manifest.json').write_bytes(body)
    print(json.dumps({'stage': 'complete', 'zip': str(output), 'bytes': output.stat().st_size,
                      'sha256': digest.hexdigest(), 'files': len(pairs),
                      'seconds': round(time.monotonic()-started, 1)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.out.resolve())
