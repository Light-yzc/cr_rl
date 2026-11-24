"""Download only a committed Actor via resumable streaming, never full PPO state."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cr_rl.training.colab_transport import run_bounded


def cli_python():
    candidates=[Path(os.environ.get('APPDATA',''))/'uv/tools/google-colab-cli/Scripts/python.exe',
                Path.home()/'.local/share/uv/tools/google-colab-cli/bin/python']
    return next((str(p) for p in candidates if p.is_file()),None)


def prepare(session):
    # Pin a single committed Actor on the runtime's SSD while Drive generations rotate.
    source='''import json,shutil,time
from pathlib import Path
d=Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun')
r=json.loads((d/'latest.json').read_text())
if not r.get('actor'):raise RuntimeError('Latest complete checkpoint has no Actor export yet; retry after the next save')
source=(d/r['actor']).resolve()
if not source.is_relative_to(d.resolve()):raise ValueError('Unsafe Actor path')
out=Path('/content/crrl_actor_transfer')/('u'+str(r['update']))
out.mkdir(parents=True,exist_ok=True)
receipt=None
cached=out/'receipt.json'
if cached.exists():
    candidate=json.loads(cached.read_text())
    if candidate.get('parts') and candidate.get('update')==r['update'] and candidate.get('bytes')==r['actor_bytes'] and sum(p['bytes'] for p in candidate['parts'])==r['actor_bytes'] and all(Path(p['remote_path']).is_file() and Path(p['remote_path']).stat().st_size==p['bytes'] for p in candidate.get('parts',[])):
        receipt=candidate
if receipt is None:
    parts=[]
    with source.open('rb') as stream:
        index=0
        while block:=stream.read(8*1024**2):
            part=out/f'part_{index:03d}.bin'
            if not part.exists() or part.stat().st_size!=len(block):
                temporary=part.with_suffix('.partial');temporary.write_bytes(block);temporary.replace(part)
            parts.append(dict(remote_path=str(part),bytes=len(block)))
            index+=1
    if sum(p['bytes'] for p in parts)!=r['actor_bytes']:raise OSError('Incomplete Actor staging')
    receipt=dict(update=r['update'],bytes=r['actor_bytes'],parts=parts,actor_only=True)
    temporary=cached.with_suffix('.partial');temporary.write_text(json.dumps(receipt));temporary.replace(cached)
temporary=out.parent/'latest.partial';temporary.write_text(json.dumps(receipt));temporary.replace(out.parent/'latest.json')
# Keep two immutable transport versions, always under this owned SSD cache.
directories=sorted((p for p in out.parent.iterdir() if p.is_dir() and p.name.startswith('u') and p.name[1:].isdigit()),key=lambda p:int(p.name[1:]))
for stale in directories[:-2]:
    if stale.resolve().is_relative_to(out.parent.resolve()):shutil.rmtree(stale)
print('ACTOR_READY',r['update'])
'''
    with tempfile.TemporaryDirectory(prefix='actor_request_') as tmp:
        helper=Path(tmp)/'prepare.py';helper.write_text(source,encoding='utf-8')
        for _ in range(3):
            try:
                result=run_bounded([sys.executable,'-c','from colab_cli.cli import app; app()',
                                       'exec','-s',session,'-f',str(helper),'--timeout','30'],
                                      timeout=100)
                if result.returncode==0 and b'ACTOR_READY ' in result.stdout:
                    return
                text=(result.stdout+result.stderr).decode(errors='replace')
                print(json.dumps(dict(event='actor_stage_retry',exit_code=result.returncode,
                    causes=[key for key in ('Connection was lost','SyntaxError','FileNotFoundError',
                            'RuntimeError','PermissionError','no Actor export','ImportError') if key in text])),flush=True)
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(event='actor_stage_retry',cause='command_timeout')),flush=True)
            time.sleep(2)
    raise RuntimeError('Actor staging unavailable; training is not modified')


class RemoteFiles:
    def __init__(self, session):
        import requests
        from colab_cli.common import state
        self.session=state.get_session(state.resolve_session(session))
        self.http=requests.Session()

    def get(self,path,**kwargs):
        return self.http.get(self.session.url.rstrip('/')+'/files/'+quote(path.strip('/'),safe='/'),
            params={'authuser':'0','colab-runtime-proxy-token':self.session.token},
            timeout=(10,45),**kwargs)

    def receipt(self):
        with self.get('/content/crrl_actor_transfer/latest.json') as response:
            response.raise_for_status()
            return response.json()


def download_file(remote, receipt, destination):
    temporary=destination.with_suffix('.partial')
    expected=receipt['bytes']
    parts=receipt['parts']
    if (not receipt.get('actor_only') or not 0<expected<2*1024**3
            or sum(p['bytes'] for p in parts)!=expected):
        raise ValueError('Invalid Actor-only transfer')
    offset=temporary.stat().st_size if temporary.exists() else 0
    boundaries={0};total=0
    for part in parts:
        total+=part['bytes'];boundaries.add(total)
    if offset not in boundaries:
        temporary.unlink(missing_ok=True);offset=0
    position=0
    for index,part in enumerate(parts):
        position+=part['bytes']
        if position<=offset:continue
        part_start=temporary.stat().st_size if temporary.exists() else 0
        for attempt in range(4):
            try:
                # Small binary parts avoid both giant-file proxy failures and
                # the 33% base64 overhead of the notebook contents API.
                with remote.get(part['remote_path'],stream=True) as response:
                    response.raise_for_status()
                    pieces=[];received=0
                    for piece in response.iter_content(4*1024**2):
                        received+=len(piece)
                        if received>part['bytes']:raise ValueError('Oversized Actor chunk')
                        pieces.append(piece)
                    block=b''.join(pieces)
                if len(block)!=part['bytes']:raise ValueError('Truncated Actor chunk')
                with temporary.open('ab') as target:
                    if target.write(block)!=len(block):raise OSError('Short Actor chunk write')
                break
            except Exception:
                if temporary.exists():
                    with temporary.open('r+b') as target:target.truncate(part_start)
                if attempt==3:raise RuntimeError('Actor transfer paused; completed chunks retained') from None
                time.sleep(2)
        if (index+1)%16==0:
            print(json.dumps(dict(event='actor_transfer',update=receipt['update'],
                                 received=position,total=expected)),flush=True)
    if temporary.stat().st_size!=expected or not zipfile.is_zipfile(temporary):
        raise ValueError('Incomplete Actor export')
    temporary.replace(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='cr-train-l4-1117')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/awbc_flywheel/models')
    parser.add_argument('--cli-python')
    args=parser.parse_args()
    if importlib.util.find_spec('colab_cli') is None:
        interpreter=args.cli_python or cli_python()
        if not interpreter:
            parser.error('Pass --cli-python pointing to the installed Colab CLI environment')
        return subprocess.call([interpreter,str(Path(__file__).resolve()),*sys.argv[1:]])
    args.out.mkdir(parents=True,exist_ok=True)
    from cr_rl.training.awbc_inbox import exclusive_lock
    with exclusive_lock(args.out/'download.lock'):
        prepare(args.session)
        remote=RemoteFiles(args.session);receipt=remote.receipt()
        update=int(receipt['update'])
        destination=args.out/f'actor_u{update:06d}.pt'
        changed=not destination.exists()
        if not destination.exists() or destination.stat().st_size!=receipt['bytes']:
            download_file(remote,receipt,destination)
            changed=True
        pointer=dict(update=receipt['update'],bytes=receipt['bytes'],actor_only=True,checkpoint=str(destination.resolve()),downloaded_at=time.time())
        temporary=args.out/'latest_actor.partial';temporary.write_text(json.dumps(pointer,indent=2))
        temporary.replace(args.out/'latest_actor.json')
    print(json.dumps(dict(**pointer,downloaded=changed)),flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        # requests exceptions can contain the Colab proxy token in their URL.
        print(json.dumps(dict(event='actor_download_failed',error=type(error).__name__,
            locations=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name)
                       for frame in traceback.extract_tb(error.__traceback__)],
            message='Retry safely; no full checkpoint was downloaded')),flush=True)
        raise SystemExit(1)
