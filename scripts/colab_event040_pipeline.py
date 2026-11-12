"""One-L4 continuation on event-preserving delayed BC supervision at 0.4s."""
from __future__ import annotations
import hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path('/content/state_event040')
DRIVE=Path('/content/drive/MyDrive/cr_rl/runs/state_global_event040_20261110')
ENV={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python'),
     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','JAVA_HOME':'/usr/lib/jvm/java-21-openjdk-amd64'}

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.uploading')
    shutil.copy2(source,temporary)
    if temporary.stat().st_size!=source.stat().st_size:raise RuntimeError('Incomplete copy')
    temporary.replace(destination)

def publish(phase,**fields):
    row={'phase':phase,'pid':os.getpid(),'time_unix':time.time(),**fields}
    atomic(ROOT/'status.json',row);atomic(DRIVE/'status.json',row)
    print(json.dumps(row),flush=True)

def run(command,name,timeout=7200):
    publish(name,command=command)
    with (ROOT/(name+'.log')).open('w',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        while child.poll() is None:
            if time.monotonic()-started>timeout:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise TimeoutError(name)
            time.sleep(3)
    if child.returncode:raise RuntimeError(f'{name} exit {child.returncode}; see {name}.log')

def freeze_weights(source,destination):
    import torch
    state=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    keep=['model_architecture_version','config','temporal_config','decision_ticks','model','progress']
    weights={key:state[key] for key in keep}
    temporary=destination.with_name(destination.name+'.uploading')
    torch.save(weights,temporary);temporary.replace(destination)
    del weights,state


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def discard_uncommitted_resample(target):
    """Remove only this task's interrupted, never-committed resample output."""
    target=Path(target)
    if not target.exists() or (target/'dataset.manifest.json').exists():return
    expected=ROOT.resolve()/'event040_data'
    if target.resolve()!=expected or target.is_symlink():
        raise RuntimeError('Refusing incomplete resample cleanup outside owned output')
    allowed={'shards','audit','.source_arrays','resampling_summary.json',
             'resampling_summary.json.tmp','dataset_metadata.json','dataset_metadata.json.tmp',
             'dataset.manifest.json.tmp'}
    if any(child.name not in allowed or child.is_symlink() for child in target.iterdir()):
        raise RuntimeError('Incomplete resample contains unrecognized files; inspect manually')
    shutil.rmtree(target)


def checkpoint_progress(path):
    import torch
    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    saved=dict(checkpoint['progress']);del checkpoint
    if any(not isinstance(saved.get(key),int) or isinstance(saved[key],bool) or saved[key]<0
           for key in ('epoch','next_episode','optimizer_steps')):
        raise RuntimeError('Invalid continuation checkpoint progress')
    return saved


def require_completed_epoch(path,epoch):
    saved=checkpoint_progress(path)
    if saved['epoch']!=epoch or saved['next_episode']!=0:
        raise RuntimeError(f"Evaluation epoch {epoch} differs from completed checkpoint: {saved}")
    return saved


def verify_evaluation(destination,policy,opponent,epoch):
    """A cached result is reusable only for the exact frozen weights/protocol."""
    record=destination/'evaluation.json'
    summary=json.loads((destination/'evaluation_summary.json').read_text())
    observed=json.loads(record.read_text())
    if summary.get('evaluation_sha256')!=file_sha256(record):
        raise RuntimeError('Cached evaluation summary does not match its evidence')
    if (observed.get('policy_sha256')!=file_sha256(policy)
            or observed.get('opponent_sha256')!=file_sha256(opponent)):
        raise RuntimeError('Cached evaluation belongs to different frozen weights')
    if (observed.get('decision_ticks')!=8 or observed.get('pair_count')!=4
            or observed.get('pair_start')!=(epoch-1)*4
            or not observed.get('candidate_inference_budget_passed')):
        raise RuntimeError('Cached evaluation protocol differs from this epoch')

def main():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive is not mounted')
    DRIVE.mkdir(parents=True,exist_ok=True)
    copy(ROOT/'source_manifest.json',DRIVE/'source_manifest.json')
    copy(Path('/content/event040_source.tar.gz'),DRIVE/'source.tar.gz')
    old=ROOT/'original025/dataset.manifest.json'
    target=ROOT/'event040_data'
    if not (target/'dataset.manifest.json').exists() and (DRIVE/'dataset/dataset.manifest.json').exists():
        restore=json.loads((DRIVE/'dataset/dataset.manifest.json').read_text())
        for record in restore['shards']:
            copy(DRIVE/'dataset'/record['path'],target/record['path'])
        for source in (DRIVE/'dataset').iterdir():
            if source.is_file() and source.name!='dataset.manifest.json':copy(source,target/source.name)
        copy(DRIVE/'dataset/dataset.manifest.json',target/'dataset.manifest.json')
    if not (target/'dataset.manifest.json').exists():
        discard_uncommitted_resample(target)
        run([sys.executable,'scripts/resample_bc_interval.py','--source-manifest',str(old),
             '--decision-ticks','8','--out',str(target)],'event_resample',1800)
    manifest=json.loads((target/'dataset.manifest.json').read_text())
    metadata=json.loads((target/manifest['metadata']).read_text())
    if metadata.get('decision_ticks')!=8 or not metadata.get('training_ready'):
        raise RuntimeError('Event-preserving 0.4s dataset did not pass its integrity gate')
    publish('event_dataset_ready',rows=manifest['rows'],games=manifest['replays'],
            strategy='merge_WAIT_causal_delay_FIFO_preserve_original_pre_action_labels')
    # Publish data after all action hashes/counts pass; commit manifest last.
    for source in target.rglob('*'):
        if source.is_file() and source.name!='dataset.manifest.json':
            copy(source,DRIVE/'dataset'/source.relative_to(target))
    copy(target/'dataset.manifest.json',DRIVE/'dataset/dataset.manifest.json')
    run([sys.executable,'scripts/benchmark_state_policy.py','--probe','fixtures/probe.npz',
        '--metadata','fixtures/probe_parent_metadata.pt','--state-checkpoint','input/parent_state.pt',
        '--width','1280','--steps','30','--warmup','5','--precision','amp_fp16',
        '--entity-count','200','--out',str(ROOT/'latency.json')],'latency_admission',600)
    latency=json.loads((ROOT/'latency.json').read_text())
    if not latency.get('l4_gate_verified'):raise RuntimeError('Current L4/weights failed 400ms gate')
    copy(ROOT/'latency.json',DRIVE/'latency_admission.json')
    train=ROOT/'training';previous=ROOT/'previous_state.pt'
    progress_path=ROOT/'pipeline_progress.json'
    if not progress_path.exists() and (DRIVE/progress_path.name).exists():copy(DRIVE/progress_path.name,progress_path)
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'epochs':0,'evaluated':[],'previous_for':0}
    def commit():atomic(progress_path,progress);copy(progress_path,DRIVE/progress_path.name)
    if not (train/'latest.pt').exists() and (DRIVE/'training/latest.pt').exists():
        for name in ['latest.pt','run_config.json','split.json','metrics.jsonl']:
            if (DRIVE/'training'/name).exists():copy(DRIVE/'training'/name,train/name)
    if not previous.exists():
        if (DRIVE/previous.name).exists():copy(DRIVE/previous.name,previous)
        elif progress['previous_for']>1:
            raise RuntimeError('Missing frozen previous-epoch opponent; cannot substitute stage parent')
        else:freeze_weights(ROOT/'input/parent_state.pt',previous);copy(previous,DRIVE/previous.name)
    base=[sys.executable,'scripts/train_state_bc.py',str(target/'dataset.manifest.json'),
      '--out',str(train),'--backup-dir',str(DRIVE/'training'),'--device','cuda','--precision','fp16',
      '--temporal-width','1280','--temporal-heads','20','--kv-heads','5','--decision-ticks','8',
      '--chunk-size','32','--frame-microbatch','8','--learning-rate','3e-5','--position-coef','1.0',
      '--tile-label-sigma','1.0','--timing-target-action-prior','.20','--frame-encoder-fp32',
      '--split-registry','fixtures/split_registry.json','--ram-limit-gib','40',
      '--checkpoint-every-episodes','32','--validation-every-episodes','64','--max-validation-episodes','16']
    for epoch in range(1,4):
        latest=train/'latest.pt'
        if progress['epochs']<epoch:
            if latest.exists():
                saved=checkpoint_progress(latest)
                if saved['epoch']>epoch:
                    raise RuntimeError('Checkpoint is ahead of pipeline progress; do not relabel newer weights')
                if saved['epoch']<epoch:
                    if epoch>1 and progress['previous_for']!=epoch:
                        if saved['epoch']!=epoch-1 or saved['next_episode']!=0:
                            raise RuntimeError('Cannot replace previous opponent with unfinished current epoch')
                        freeze_weights(latest,previous);copy(previous,DRIVE/previous.name)
                        progress['previous_for']=epoch;commit()
                    run([*base,'--resume',str(latest),'--epochs',str(epoch)],f'train_epoch{epoch}')
            else:
                if epoch!=1:raise RuntimeError('Missing continuation checkpoint')
                run([*base,'--init-from-state','input/parent_state.pt',
                     '--init-state-dataset-manifest','input/parent_dataset/dataset.manifest.json',
                     '--epochs','1'],'train_epoch1')
            progress['epochs']=epoch;commit()
        for tag,opponent in [('bc',ROOT/'input/old_bc.pt'),('old',previous)]:
            key=f'epoch{epoch}_{tag}'
            if key in progress['evaluated']:continue
            if progress['epochs']!=epoch:raise RuntimeError('Evaluation epoch does not match current weights')
            require_completed_epoch(latest,epoch)
            destination=ROOT/'evaluation'/key
            if destination.exists() and not (destination/'evaluation_summary.json').exists():
                if destination.resolve().parent!=(ROOT/'evaluation').resolve():raise RuntimeError('Invalid owned evaluation path')
                shutil.rmtree(destination)
            if not (destination/'evaluation_summary.json').exists():
                run([sys.executable,'scripts/evaluate_state_policy.py','--policy',str(latest),
                     '--opponent',str(opponent),'--decks','configs/decks_eval.yaml','--pairs','4',
                     '--pair-start',str((epoch-1)*4),'--envs','2','--device','cuda','--precision','fp16',
                     '--out',str(destination)],f'eval_{key}',3600)
            verify_evaluation(destination,latest,opponent,epoch)
            for source in destination.iterdir():
                if source.is_file():copy(source,DRIVE/'evaluation'/key/source.name)
            progress['evaluated'].append(key);commit()
    publish('complete',epochs=3,evaluation_games=48)

if __name__=='__main__':
    import fcntl
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception as error:
        publish('failed',error=str(error));traceback.print_exc();raise
