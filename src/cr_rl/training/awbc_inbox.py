"""Immutable completed-game batches for a Drive inbox and disposable local SSD.

Transport performs metadata/size checks, without file hash scans. The existing
AWBC importer remains responsible for full observation/action/reward validation.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


FORMAT = 'crrl-awbc-batch-v1'


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value):
        raise ValueError('Invalid batch/session name')
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def completed_files(run):
    run = Path(run)
    component(run.name)
    names = ('manifest.json', 'result.json', 'terminal.json', 'events.jsonl')
    paths = [run/name for name in names]
    if not all(p.is_file() and not p.is_symlink() for p in paths):
        raise ValueError('Missing finalized recording files')
    manifest, result, terminal = [json.loads((run/name).read_text(encoding='utf-8'))
                                  for name in names[:3]]
    if (not manifest.get('collection_complete') or not manifest.get('outcome_verified')
            or manifest.get('stop_reason') != 'battle_end'
            or not result.get('outcome_verified') or not terminal.get('clock_ended')
            or result.get('outcome') != manifest.get('outcome')
            or result.get('session_id') != run.name or terminal.get('session_id') != run.name):
        raise ValueError('Recording is incomplete or has no consistent terminal result')
    seal=run/'collection_sealed.json'
    if seal.is_file():
        marker=json.loads(seal.read_text(encoding='utf-8'))
        if (marker.get('format')!='crrl-sealed-session-v1' or marker.get('manifest')!=manifest
                or marker.get('result')!=result):
            raise ValueError('Recording metadata changed after sealing')
        paths.append(seal)
    else:
        # Legacy automatic recordings finish with a verified lobby event.
        # Never read a newly finalized result while its writer is still returning to menus.
        with (run/'events.jsonl').open('rb') as stream:
            stream.seek(max(0,(run/'events.jsonl').stat().st_size-8192))
            tail=stream.read().splitlines()
        if not tail or json.loads(tail[-1]).get('event')!='menu_lobby_verified':
            raise ValueError('Recording writer has not sealed or completed its lobby return')
    chunks = sorted((run/'diagnostics').glob('frames_*.jsonl.gz'))
    if not chunks or any(not p.is_file() or p.is_symlink() for p in chunks):
        raise ValueError('Missing finalized frame chunks')
    return paths + chunks


def pack_batch(runs, archive, batch_id):
    batch_id = component(batch_id)
    runs = list(map(Path, runs))
    if not runs or len({run.name for run in runs}) != len(runs):
        raise ValueError('Empty batch or duplicate session names')
    paths = [(run, completed_files(run)) for run in runs]
    metadata = {str(p): p.read_bytes() for run, _ in paths
                for p in (run/'manifest.json', run/'result.json')}
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for run, files in paths:
                for p in files:
                    z.write(p, run.name + '/' + p.relative_to(run).as_posix())
        if any(Path(p).read_bytes() != before for p, before in metadata.items()):
            raise ValueError('Recording metadata changed while packaging')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(format=FORMAT, batch_id=batch_id, archive=batch_id+'.zip',
                bytes=archive.stat().st_size, games=[run.name for run in runs])


def publish_batch(source, inbox, receipt):
    """Commit the small ready receipt last; uploads have unique immutable names."""
    inbox = Path(inbox)
    batch = component(receipt['batch_id'])
    if receipt.get('format') != FORMAT or receipt.get('archive') != batch+'.zip':
        raise ValueError('Invalid batch receipt')
    source = Path(source)
    if source.stat().st_size != receipt['bytes']:
        raise ValueError('Incomplete uploaded archive')
    inbox.mkdir(parents=True, exist_ok=True)
    ready = inbox/(batch+'.ready.json')
    if ready.exists():
        if json.loads(ready.read_text()) != receipt:
            raise ValueError('Attempted to change a committed batch')
        if (inbox/receipt['archive']).stat().st_size != receipt['bytes']:
            raise ValueError('Committed archive size changed')
        return
    target = inbox/receipt['archive']
    temporary = target.with_name(target.name+'.partial')
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != receipt['bytes']:
        raise OSError('Incomplete Drive upload')
    temporary.replace(target)
    atomic_json(ready, receipt)


def _extract_batch(archive, staging, receipt):
    games = receipt['games']
    if not games or len(games) > 256 or len(set(games)) != len(games):
        raise ValueError('Invalid game roster')
    games = set(map(component, games))
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist()) > 20000:
            raise ValueError('Too many ZIP members')
        names = set()
        total = 0
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if ('\\' in info.filename or not parts or parts[0] not in games
                    or PurePosixPath(info.filename).as_posix() != info.filename
                    or PurePosixPath(info.filename).is_absolute() or '..' in parts
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.filename in names):
                raise ValueError('Unsafe or duplicate ZIP member')
            allowed = (len(parts) == 2 and parts[1] in
                       ('manifest.json', 'result.json', 'terminal.json', 'events.jsonl','collection_sealed.json'))
            allowed = allowed or (len(parts) == 3 and parts[1] == 'diagnostics'
                       and re.fullmatch(r'frames_[A-Za-z0-9_.-]+\.jsonl\.gz', parts[2]))
            if not allowed:
                raise ValueError('Unexpected ZIP member')
            names.add(info.filename)
            total += info.file_size
            if total > 4*1024**3:
                raise ValueError('Batch exceeds extraction budget')
        z.extractall(staging)
    for game in games:
        completed_files(staging/game)


def ingest_ready_batches(inbox, sessions, state_path, *, max_batches=4):
    """Extract outside sessions, then rename each whole game into view atomically."""
    inbox, sessions, state_path = Path(inbox), Path(sessions), Path(state_path)
    sessions.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'imported': {}}
    report = dict(imported_batches=0, added_games=0, duplicate_games=0, errors=[])
    waiting = [p for p in sorted(inbox.glob('*.ready.json'))
               if p.name[:-len('.ready.json')] not in state['imported']]
    offset = state.get('retry_cursor',0) % len(waiting) if waiting else 0
    pending = (waiting[offset:]+waiting[:offset])[:max_batches]
    # A few damaged batches must not permanently block every later recording.
    state['retry_cursor'] = (offset+len(pending)) % len(waiting) if waiting else 0
    for ready in pending:
        try:
            receipt = json.loads(ready.read_text())
            batch = component(receipt['batch_id'])
            if (ready.name != batch+'.ready.json' or receipt.get('format') != FORMAT
                    or receipt.get('archive') != batch+'.zip'
                    or not isinstance(receipt.get('bytes'), int)
                    or not 0 < receipt['bytes'] <= 2*1024**3):
                raise ValueError('Invalid ready receipt')
            source = inbox/receipt['archive']
            if source.stat().st_size != receipt['bytes']:
                raise ValueError('Committed archive is incomplete')
            # A sibling directory keeps partial result.json files out of rglob().
            with tempfile.TemporaryDirectory(prefix='.awbc_stage_', dir=sessions.parent) as tmp:
                staging = Path(tmp)
                archive = staging/'batch.zip'
                shutil.copyfile(source, archive)
                if archive.stat().st_size != receipt['bytes']:
                    raise OSError('Incomplete SSD copy')
                _extract_batch(archive, staging, receipt)
                # Check all name collisions before exposing any new directory.
                for game in receipt['games']:
                    target = sessions/game
                    if target.exists():
                        if target.is_symlink() or any((target/name).read_bytes() !=
                            (staging/game/name).read_bytes() for name in ('manifest.json', 'result.json')):
                            raise ValueError('Conflicting existing recording: '+game)
                for game in receipt['games']:
                    target = sessions/game
                    if target.exists():
                        report['duplicate_games'] += 1
                    else:
                        (staging/game).rename(target)
                        report['added_games'] += 1
            state['imported'][batch] = dict(games=receipt['games'], bytes=receipt['bytes'])
            atomic_json(state_path, state)
            report['imported_batches'] += 1
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            report['errors'].append(dict(batch=ready.name, reason=str(error)))
    if pending:
        atomic_json(state_path, state)
    return report
