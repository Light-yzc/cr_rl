"""Upload new completed live games in bounded ZIP batches while Colab trains."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cr_rl.training.awbc_inbox import atomic_json, completed_files, exclusive_lock, pack_batch
from cr_rl.training.colab_transport import run_bounded


def colab(arguments, *, expected=None):
    # Never echo Colab transport errors: some CLI versions include proxy credentials.
    for attempt in range(3):
        try:
            result = run_bounded(['colab', *arguments], timeout=100)
            if result.returncode == 0 and (expected is None or expected.encode() in result.stdout):
                return
        except subprocess.TimeoutExpired:
            pass
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError('Colab transfer failed after 3 attempts; pending batch retained')


def upload(archive, receipt, session):
    batch = receipt['batch_id']
    with tempfile.TemporaryDirectory(prefix='transfer_', dir=archive.parent) as tmp:
        tmp = Path(tmp)
        parts = []
        with archive.open('rb') as stream:
            while block := stream.read(8*1024**2):
                part = tmp/f'part_{len(parts):03d}'
                part.write_bytes(block)
                remote = f'/content/crrl_awbc_{batch}_{len(parts):03d}.part'
                colab(['upload', '-s', session, str(part), remote])
                parts.append(remote)
        remote_archive = '/content/crrl_awbc_'+batch+'.zip'
        helper = tmp/'commit.py'
        helper.write_text(
            "import sys,json,shutil\nfrom pathlib import Path\n"
            "sys.path.insert(0,'/content/crrl_training/src')\n"
            "from cr_rl.training.awbc_inbox import publish_batch\n"
            f"parts={parts!r}\narchive=Path({remote_archive!r})\n"
            "with archive.open('wb') as target:\n"
            "    for part in parts:\n"
            "        with Path(part).open('rb') as source:shutil.copyfileobj(source,target,4*1024**2)\n"
            f"receipt=json.loads({json.dumps(receipt)!r})\n"
            "publish_batch(archive,Path('/content/drive/MyDrive/cr_rl_ppo/preparation/awbc_inbox'),receipt)\n"
            "for part in parts:Path(part).unlink(missing_ok=True)\n"
            "archive.unlink(missing_ok=True)\nprint('BATCH_COMMITTED',receipt['batch_id'])\n",
            encoding='utf-8')
        colab(['exec', '-s', session, '-f', str(helper), '--timeout', '30'],
              expected='BATCH_COMMITTED '+batch)


def cycle(args, state):
    args.outbox.mkdir(parents=True, exist_ok=True)
    # Retry the same immutable batch after failures, rather than sending another copy.
    for pending in sorted(args.outbox.glob('*.receipt.json')):
        receipt = json.loads(pending.read_text())
        archive = args.outbox/receipt['archive']
        upload(archive, receipt, args.session)
        state['known'].update({name: receipt['batch_id'] for name in receipt['games']})
        atomic_json(args.outbox/'upload_state.json', state)
        print(json.dumps(dict(event='batch_uploaded', batch=receipt['batch_id'],
                              games=len(receipt['games']), bytes=receipt['bytes'])), flush=True)
        archive.unlink(missing_ok=True)
        pending.unlink()
    runs = []
    seen = set()
    for manifest in sorted(args.source.rglob('manifest.json')):
        run = manifest.parent
        if run.name in state['known'] or run.name in seen:
            continue
        try:
            completed_files(run)
        except (OSError, ValueError, KeyError):
            continue
        seen.add(run.name)
        runs.append(run)
    flush=args.outbox/'FLUSH'
    minimum=args.batch_games if args.loop and not flush.exists() else 1
    if len(runs) < minimum:
        if flush.exists() and not runs:
            flush.rename(flush.with_name('FLUSH_done_'+str(time.time_ns())))
        return False
    batch = time.strftime('batch_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:12]
    archive = args.outbox/(batch+'.zip')
    receipt = pack_batch(runs[:args.batch_games], archive, batch)
    atomic_json(args.outbox/(batch+'.receipt.json'), receipt)
    # The next cycle uploads this durable pending batch, including after a restart.
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--session', default='cr-train-l4-1117')
    parser.add_argument('--outbox', type=Path, default=ROOT/'artifacts/awbc_outbox')
    parser.add_argument('--baseline-zip', type=Path)
    parser.add_argument('--batch-games', type=int, default=10)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if not args.source.is_dir() or not 1 <= args.batch_games <= 256 or args.interval < 1:
        parser.error('Existing source, 1..256 batch games and positive interval required')
    args.outbox.mkdir(parents=True, exist_ok=True)
    state_path = args.outbox/'upload_state.json'
    with exclusive_lock(args.outbox/'uploader.lock'):
        state = json.loads(state_path.read_text()) if state_path.exists() else {'known': {}}
        if args.baseline_zip:
            with zipfile.ZipFile(args.baseline_zip) as z:
                for name in json.loads(z.read('snapshot.json'))['games']:
                    state['known'].setdefault(name, 'existing_baseline')
            atomic_json(state_path, state)
        while True:
            try:
                more = cycle(args, state)
            except (OSError, ValueError, RuntimeError) as error:
                print(json.dumps(dict(event='upload_retry_pending', reason=str(error))), flush=True)
                if not args.loop:
                    raise SystemExit(1)
                more = False
            if more:
                continue
            if not args.loop:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
