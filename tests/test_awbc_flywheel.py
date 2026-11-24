import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_actor_chunk_download_resumes_without_fetching_finished_chunks(tmp_path,monkeypatch):
    module=load_script('download_latest_actor')
    buffer=BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('actor/data.pkl',b'actor-only-placeholder')
    data=buffer.getvalue();chunks=[data[:40],data[40:80],data[80:]]
    calls=[]
    class Response:
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield self.data
    def get(url,**kwargs):
        index=int(url.rsplit('/',1)[-1]);calls.append(index)
        return Response(chunks[index])
    remote=SimpleNamespace(get=get)
    target=tmp_path/'actor.pt';target.with_suffix('.partial').write_bytes(chunks[0])
    receipt=dict(bytes=len(data),actor_only=True,update=12,
                 parts=[dict(bytes=len(block),remote_path=f'/content/{i}') for i,block in enumerate(chunks)])
    module.download_file(remote,receipt,target)
    assert calls==[1,2]
    assert target.read_bytes()==data
    assert not target.with_suffix('.partial').exists()


def test_actor_cache_keeps_active_and_latest_but_prunes_other_finished_versions(tmp_path):
    module=load_script('run_awbc_flywheel')
    for update in (1,2,3):(tmp_path/f'actor_u{update:06d}.pt').write_bytes(b'actor')
    current=tmp_path/'actor_u000003.pt'
    (tmp_path/'latest_actor.json').write_text(json.dumps(dict(checkpoint=str(current),bytes=5,actor_only=True)))
    untouched=tmp_path/'other.pt';untouched.write_bytes(b'untouched')
    module.prune_actors(tmp_path,str(tmp_path/'actor_u000001.pt'))
    assert (tmp_path/'actor_u000001.pt').exists()
    assert not (tmp_path/'actor_u000002.pt').exists()
    assert current.exists() and untouched.exists()


def test_resuming_batch_keeps_its_frozen_actor_when_new_candidate_exists(tmp_path):
    module=load_script('run_awbc_flywheel')
    models=tmp_path/'models';models.mkdir()
    old=models/'actor_u000001.pt';old.write_bytes(b'old')
    new=models/'actor_u000002.pt';new.write_bytes(b'new')
    (models/'latest_actor.json').write_text(json.dumps(dict(checkpoint=str(new),update=2,bytes=3,actor_only=True)))
    out=tmp_path/'run';out.mkdir()
    active=dict(checkpoint=str(old),folder=str(out/'sessions/batch_one'),update=1)
    (out/'flywheel.json').write_text(json.dumps(dict(plan='2:10,1:10',cycles=0,active=active)))
    args=SimpleNamespace(out=out,models=models,plan='2:10,1:10',refresh_minutes=60,refresh_now=True)
    wheel=module.Flywheel(args)
    assert wheel.state['active']['checkpoint']==str(old)
    assert not wheel.initial_wait
    wheel.uploader=SimpleNamespace(poll=lambda:None)
    wheel.next_download=0
    wheel.spawn=lambda *a:pytest.fail('Unconsumed candidate was replaced by another download')
    wheel.tick()
    assert wheel.downloader is None and wheel.next_download>0


def test_bounded_transport_kills_launcher_descendants(tmp_path):
    psutil=pytest.importorskip('psutil')
    from cr_rl.training.colab_transport import run_bounded
    pid_path=tmp_path/'child.pid'
    code=('import subprocess,sys,time,pathlib;'
          "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
          f'pathlib.Path({str(pid_path)!r}).write_text(str(p.pid));time.sleep(30)')
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable,'-c',code],timeout=2)
    assert pid_path.exists()
    pid=int(pid_path.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
