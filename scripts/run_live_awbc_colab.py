"""Supervise local Colab AWBC and copy stable, hash-verified checkpoints to Drive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def verified_copy(source,target):
    before=source.stat()
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.partial')
    digest=hashlib.sha256()
    with source.open('rb') as src,temporary.open('wb') as dst:
        for block in iter(lambda:src.read(4*1024**2),b''):
            digest.update(block);dst.write(block)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        return False
    received=file_sha256(temporary)
    if received!=digest.hexdigest():raise IOError(f'Backup checksum mismatch: {source.name}')
    os.replace(temporary,target)
    receipt=target.with_suffix(target.suffix+'.sha256')
    receipt.write_text(received,encoding='ascii')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if not a.command:p.error('Pass training command after --')
    command=a.command[1:] if a.command[0]=='--' else a.command
    a.work.mkdir(parents=True,exist_ok=True);a.backup.mkdir(parents=True,exist_ok=True)
    # Restore only complete verified files; local features can always be rebuilt.
    for source in a.backup.iterdir():
        if not source.is_file() or source.suffix not in ('.pt','.json','.jsonl'):continue
        checksum=source.with_suffix(source.suffix+'.sha256')
        target=a.work/source.name
        if checksum.exists() and not target.exists():
            actual=file_sha256(source)
            if actual!=checksum.read_text().strip():raise IOError(f'Bad backup: {source.name}')
            shutil.copyfile(source,target)
    synced={}
    def sync():
        for source in sorted(a.work.iterdir()):
            if not source.is_file() or source.suffix not in ('.pt','.json','.jsonl'):continue
            signature=(source.stat().st_size,source.stat().st_mtime_ns)
            if synced.get(source.name)==signature:continue
            if verified_copy(source,a.backup/source.name):synced[source.name]=signature
    with (a.work/'console.log').open('a',encoding='utf8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
                sync()
                path=a.work/'metrics.jsonl'
                if path.exists():
                    lines=path.read_text().splitlines()
                    if lines:print(lines[-1],flush=True)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
            raise
        finally:
            if process.poll() is not None:sync()
    if process.returncode:
        print((a.work/'console.log').read_text(errors='replace')[-5000:])
        raise SystemExit(process.returncode)


if __name__=='__main__':main()
