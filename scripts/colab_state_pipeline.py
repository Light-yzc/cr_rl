"""One L4: latency gate, causal data rebuild, bounded BC, paired frozen evaluation.

Run as a detached subprocess after mounting Drive. No keepalive interaction or
extra VM is created. Failure status is published for the supervising agent.
"""
from __future__ import annotations
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path('/content/state_global')
DRIVE=Path('/content/drive/MyDrive/cr_rl/runs/state_global_183m_040s_20261110')
OLD=Path('/content/drive/MyDrive/cr_rl/runs/bc_salvage_fp16_20261109')
ENV={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def publish(phase,**extra):
    row={'phase':phase,'time_unix':time.time(),'pid':os.getpid(),**extra}
    atomic(ROOT/'status.json',row)
    if DRIVE.exists():atomic(DRIVE/'status.json',row)
    print(json.dumps(row),flush=True)

def run(command,log,phase,timeout=3600):
    publish(phase,command=command)
    with (ROOT/log).open('w',buffering=1) as output:
        child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+timeout
        while child.poll() is None:
            if time.monotonic()>deadline:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise TimeoutError(phase)
            time.sleep(2)
        if child.returncode:raise RuntimeError(f'{phase} exit {child.returncode}; see {log}')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def copy_atomic(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.uploading')
    shutil.copy2(source,temporary)
    if temporary.stat().st_size!=source.stat().st_size:raise RuntimeError('Incomplete Drive copy')
    temporary.replace(destination)

def source_download():
    import requests
    target=ROOT/'input/IL_Replay_replays.parquet'
    expected='8f99ddd7b6f460c3efaf6873b4727e746a9e9edd8c796d120084aaaa68b4e261'
    if not target.exists():
        with requests.get('https://drive.usercontent.google.com/download',
                params={'id':'1NldB0NKgILdmE4VI2WeflwaKMdJauMuC','export':'download','confirm':'t'},
                stream=True,timeout=90) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type',''):raise RuntimeError('Source download returned HTML')
            with target.with_suffix('.part').open('wb') as stream:
                for block in response.iter_content(1024**2):stream.write(block)
        target.with_suffix('.part').replace(target)
    if sha256(target)!=expected:raise RuntimeError('Original source parquet SHA mismatch')

def preserve_old_state(source,destination):
    import torch
    current=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    keys=['model_architecture_version','config','temporal_config','decision_ticks','model','progress']
    small={key:current[key] for key in keys}
    temporary=destination.with_suffix('.uploading')
    torch.save(small,temporary);temporary.replace(destination)
    del small,current

def main():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive must already be mounted')
    DRIVE.mkdir(parents=True,exist_ok=True)
    progress_path=ROOT/'pipeline_progress.json'
    if not progress_path.exists() and (DRIVE/progress_path.name).exists():
        copy_atomic(DRIVE/progress_path.name,progress_path)
    progress=(json.loads(progress_path.read_text()) if progress_path.exists() else
              {'completed_epoch':0,'evaluated':[],'previous_for_epoch':0})
    def commit():
        atomic(progress_path,progress);copy_atomic(progress_path,DRIVE/progress_path.name)
    copy_atomic(ROOT/'source_manifest.json',DRIVE/'source_manifest.json')
    copy_atomic(Path('/content/candidate_source.tar.gz'),DRIVE/'candidate_source.tar.gz')
    gate=json.loads((ROOT/'results/bench1280.json').read_text())
    if not gate.get('l4_gate_verified') or gate['width']!=1280:raise RuntimeError('L4 400ms admission failed')
    copy_atomic(ROOT/'results/bench1280.json',DRIVE/'latency_admission.json')
    publish('latency_admitted',parameters=gate['parameter_count'])
    missing=[package for module,package in [('jpype','jpype1'),('pyarrow','pyarrow'),
        ('yaml','PyYAML'),('pytest','pytest'),('gymnasium','gymnasium>=1.0,<2'),
        ('pettingzoo','pettingzoo>=1.24,<2'),('zstandard','zstandard>=0.23,<1'),('msgpack','msgpack')]
             if importlib.util.find_spec(module) is None]
    if missing:run([sys.executable,'-m','pip','install','-q',*missing],'dependencies.log','dependencies',600)
    java=Path('/usr/lib/jvm/java-21-openjdk-amd64')
    if not java.exists():
        run(['apt-get','update','-qq'],'apt_update.log','java_package_index',300)
        run(['apt-get','install','-y','-qq','openjdk-21-jdk-headless'],'java.log','java_install',600)
    ENV['JAVA_HOME']=str(java);ENV['PATH']=str(java/'bin')+':'+os.environ['PATH']
    run([sys.executable,'scripts/setup_simulator.py'],'simulator_setup.log','simulator_setup',1200)
    vendor=ROOT/'.cache/vendor/crforge'
    ENV['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(vendor/'python')
    run([sys.executable,'-m','pytest','tests/test_state_temporal.py','tests/test_state_policy.py',
         'tests/test_state_bc.py','tests/test_interval_rebuild.py','tests/test_state_evaluation.py',
         '-q','-p','no:cacheprovider'],'remote_tests.log','remote_tests',300)
    (ROOT/'input').mkdir(exist_ok=True)
    old_model=ROOT/'input/old_bc.pt'
    if not old_model.exists():copy_atomic(OLD/'ablation_prior020/bc.pt',old_model)
    if sha256(old_model)!='a196b85348f4fe6b64ffc8451dccd3656123860afd0278074ebb520cd9bcc3e1':
        raise RuntimeError('Old BC checkpoint differs from frozen parent')
    metadata=ROOT/'input/old_metadata.json'
    if not metadata.exists():copy_atomic(OLD/'data/human_bc_reconstructed.json',metadata)
    source_download()
    rebuilt=ROOT/'rebuilt_040'
    if not (rebuilt/'dataset.manifest.json').exists() and (DRIVE/'dataset/dataset.manifest.json').exists():
        restored=json.loads((DRIVE/'dataset/dataset.manifest.json').read_text())
        for record in restored['shards']:
            copy_atomic(DRIVE/'dataset'/record['path'],rebuilt/record['path'])
            if sha256(rebuilt/record['path'])!=record['sha256']:raise RuntimeError('Restored shard hash mismatch')
        for name in ['interval_summary.json','interval_protocol.json',restored['metadata']]:
            copy_atomic(DRIVE/'dataset'/name,rebuilt/name)
        copy_atomic(DRIVE/'dataset/dataset.manifest.json',rebuilt/'dataset.manifest.json')
    elif rebuilt.exists() and not (rebuilt/'dataset.manifest.json').exists():
        if rebuilt.resolve().parent!=ROOT.resolve() or rebuilt.name!='rebuilt_040':
            raise RuntimeError('Refusing incomplete rebuild cleanup outside owned output')
        marker=rebuilt/'interval_protocol.json'
        if not marker.exists():raise RuntimeError('Incomplete rebuild lacks ownership marker; inspect manually')
        if json.loads(marker.read_text()).get('decision_ticks')!=8:raise RuntimeError('Rebuild marker differs')
        shutil.rmtree(rebuilt)
    rebuild=[sys.executable,'scripts/rebuild_bc_interval.py','--metadata',str(metadata),
       '--source-parquet',str(ROOT/'input/IL_Replay_replays.parquet'),'--source-root',str(ROOT),
       '--decision-ticks','8','--limit','0','--simulator-install',str(vendor/'env/build/install/env'),
       '--tables',str(Path.home()/'.crforge/tables/16.635.3'),'--out',str(rebuilt)]
    if not (rebuilt/'dataset.manifest.json').exists():run(rebuild,'rebuild.log','rebuilding_040',3600)
    admission=json.loads((rebuilt/'interval_summary.json').read_text())
    if (not admission.get('all_selected_source_commands_preserved',False)
            or not admission.get('training_ready',False)):
        raise RuntimeError('User requires every source action retained; partial/legacy datasets cannot resume')
    if not admission.get('training_ready') or admission['accepted_games']<150:
        raise RuntimeError('Too few verified complete-source games; do not start formal training')
    for name in ['interval_summary.json','interval_protocol.json','dataset_metadata.json']:
        copy_atomic(rebuilt/name,DRIVE/'dataset'/name)
    for source in (rebuilt/'shards').glob('*.npz'):copy_atomic(source,DRIVE/'dataset/shards'/source.name)
    for source in (rebuilt/'audit').glob('*.json'):copy_atomic(source,DRIVE/'dataset/audit'/source.name)
    # Publish the commit marker only after every referenced shard is available.
    copy_atomic(rebuilt/'dataset.manifest.json',DRIVE/'dataset/dataset.manifest.json')
    publish('dataset_admitted',games=admission['accepted_games'],rows=admission['rows'])
    training=ROOT/'training';previous=ROOT/'previous_state.pt'
    if not (training/'latest.pt').exists() and (DRIVE/'training/latest.pt').exists():
        for name in ['latest.pt','run_config.json','split.json','metrics.jsonl']:
            if (DRIVE/'training'/name).exists():copy_atomic(DRIVE/'training'/name,training/name)
    if not previous.exists() and (DRIVE/previous.name).exists():copy_atomic(DRIVE/previous.name,previous)
    base=[sys.executable,'scripts/train_state_bc.py',str(rebuilt/'dataset.manifest.json'),
       '--out',str(training),'--backup-dir',str(DRIVE/'training'),'--device','cuda','--precision','fp16',
       '--temporal-width','1280','--temporal-heads','20','--kv-heads','5','--chunk-size','32',
       '--frame-microbatch','8','--decision-ticks','8','--learning-rate','3e-5','--position-coef','1.0',
       '--timing-target-action-prior','.20','--tile-label-sigma','1.0','--frame-encoder-fp32',
       '--split-registry',str(ROOT/'fixtures/split_registry.json'),'--ram-limit-gib','40',
       '--checkpoint-every-episodes','32','--validation-every-episodes','64',
       '--max-validation-episodes','16']
    for epoch in range(1,4):
        latest=training/'latest.pt'
        if progress['completed_epoch']<epoch:
            if latest.exists():
                import torch
                saved=torch.load(latest,map_location='cpu',mmap=True,weights_only=False)
                saved_progress=saved['progress'];del saved
                if saved_progress['epoch']>=epoch:
                    progress['completed_epoch']=epoch;commit()
                else:
                    if epoch>1 and progress['previous_for_epoch']!=epoch:
                        if saved_progress['epoch']!=epoch-1 or saved_progress['next_episode']!=0:
                            raise RuntimeError('Missing frozen previous epoch; do not overwrite it with mid-epoch weights')
                        preserve_old_state(latest,previous);copy_atomic(previous,DRIVE/previous.name)
                        progress['previous_for_epoch']=epoch;commit()
                    command=[*base,'--resume',str(latest),'--epochs',str(epoch)]
                    run(command,f'train_epoch{epoch}.log',f'training_epoch_{epoch}',7200)
            else:
                if epoch!=1:raise RuntimeError('Missing latest checkpoint for continued stage')
                run([*base,'--init-from-bc',str(old_model),'--epochs','1'],
                    'train_epoch1.log','training_epoch_1',7200)
            progress['completed_epoch']=epoch;commit()
        eval_targets=[('bc',old_model)] + ([('old',previous)] if epoch>1 else [])
        for tag,opponent in eval_targets:
            evaluation_tag=f'epoch{epoch}_{tag}'
            if evaluation_tag in progress['evaluated']:continue
            if progress['completed_epoch']!=epoch:raise RuntimeError('Cannot evaluate an old label with newer weights')
            destination=ROOT/'evaluation'/f'epoch{epoch}_{tag}'
            if destination.exists() and not (destination/'evaluation_summary.json').exists():
                if destination.resolve().parent!=(ROOT/'evaluation').resolve():raise RuntimeError('Invalid evaluation cleanup path')
                shutil.rmtree(destination)
            command=[sys.executable,'scripts/evaluate_state_policy.py','--policy',str(latest),
                '--opponent',str(opponent),'--decks','configs/decks_eval.yaml','--pairs','4',
                '--pair-start',str((epoch-1)*4),'--envs','2','--device','cuda','--precision','fp16',
                '--out',str(destination)]
            if not (destination/'evaluation_summary.json').exists():
                run(command,f'eval_epoch{epoch}_{tag}.log',f'evaluating_epoch_{epoch}_{tag}',3600)
            for source in destination.iterdir():
                if source.is_file():copy_atomic(source,DRIVE/'evaluation'/destination.name/source.name)
            observed=json.loads((destination/'evaluation.json').read_text())
            if not observed.get('candidate_inference_budget_passed',False):
                raise RuntimeError('Candidate exceeded 400ms in actual evaluation; stop expansion')
            progress['evaluated'].append(evaluation_tag);commit()
        publish('epoch_completed',epoch=epoch)
    publish('complete',epochs=3,evaluation_games=40)

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    import fcntl
    lock=(ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception as error:
        publish('failed',error_type=type(error).__name__,error=str(error))
        traceback.print_exc();raise
