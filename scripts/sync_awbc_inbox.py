"""CPU-only background Drive → SSD import of committed AWBC game batches."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cr_rl.training.awbc_inbox import atomic_json, exclusive_lock, ingest_ready_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inbox', type=Path, required=True)
    parser.add_argument('--sessions', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('Interval must be at least one second')
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with exclusive_lock(args.state.with_suffix('.lock')):
        while not stopped:
            try:
                report = ingest_ready_batches(args.inbox, args.sessions, args.state)
            except Exception as error:
                report = dict(imported_batches=0, added_games=0,
                              errors=[dict(reason=type(error).__name__+': '+str(error))])
            atomic_json(args.state.with_suffix('.status.json'), dict(time=time.time(), **report))
            if report['imported_batches'] or report['errors'] or not args.loop:
                print(json.dumps(report), flush=True)
            if not args.loop:
                break
            deadline = time.monotonic() + args.interval
            while not stopped and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
