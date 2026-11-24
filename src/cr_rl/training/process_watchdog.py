"""Low-overhead liveness checks and restoration of a complete local save pair."""
import json
import os
from pathlib import Path
import time


class ProgressWatchdog:
    def __init__(self, paths, timeout, now=None):
        self.paths = list(map(Path, paths))
        self.timeout = timeout
        self.last_activity = time.monotonic() if now is None else now
        self.signature = self._signature()

    @staticmethod
    def _dir_signature(directory: Path):
        try:
            subdirs = sorted(p for p in directory.iterdir() if p.is_dir())
        except FileNotFoundError:
            return None
        target = subdirs[-1] if subdirs else directory
        entries = []
        for p in sorted(target.rglob("*")):
            try:
                if p.is_file():
                    st = p.stat()
                    entries.append((str(p.relative_to(directory)), st.st_size, st.st_mtime_ns))
            except FileNotFoundError:
                continue
        return tuple(entries)

    def _signature(self):
        values = []
        for path in self.paths:
            try:
                info = path.stat()
                if path.is_dir():
                    values.append(self._dir_signature(path))
                else:
                    values.append((info.st_size, info.st_mtime_ns))
            except FileNotFoundError:
                values.append(None)
        return tuple(values)

    def stalled(self, now=None):
        now = time.monotonic() if now is None else now
        signature = self._signature()
        if signature != self.signature:
            self.signature = signature
            self.last_activity = now
        return self.timeout > 0 and now-self.last_activity >= self.timeout


def restore_local_pair(out):
    """A process must have exited before this is called. No large tensor reads."""
    out = Path(out)
    pairs = sorted(p for p in (out/'checkpoints').glob('update_*.pt')
                   if p.with_suffix('.pool.json').is_file() and p.stat().st_size > 0)
    if not pairs:
        raise RuntimeError('No complete local checkpoint/pool pair to recover')
    checkpoint = pairs[-1]
    pool_path = checkpoint.with_suffix('.pool.json')
    pool = json.loads(pool_path.read_text())
    for entry in pool['entries']:
        source = Path(entry['checkpoint'])
        source = source if source.is_absolute() else out/source
        if not source.is_file():
            raise RuntimeError('Checkpoint opponent dependency is missing')
    temporary = out/'watchdog_restore.tmp.pt'
    temporary.unlink(missing_ok=True)
    os.link(checkpoint, temporary)
    temporary.replace(out/'ppo_latest.pt')
    temporary = out/'opponent_pool.json.partial'
    temporary.write_text(json.dumps(pool, indent=2))
    temporary.replace(out/'opponent_pool.json')
    return int(checkpoint.stem.removeprefix('update_'))
