"""Small read-only experiment status, plus durable small reports only."""
import json
from pathlib import Path
import shutil

root=Path('/content/crrl_army_elixir_1117')
out=root/'artifacts/army_elixir'
for path in [root/'experiment.log',out/'tests.log']:
    if path.exists():
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-1800));print(path.name,f.read().decode(errors='replace'))
for directory in out.iterdir() if out.exists() else []:
    if not directory.is_dir(): continue
    summary=directory/'summary.json'
    if summary.exists():
        data=json.loads(summary.read_text())
        metrics=data.get('metrics',{})
        print(directory.name,json.dumps(dict(update=data.get('update'),
            actor_steps=data.get('optimizer_steps'),kl=metrics.get('approx_kl'),
            critic_loss=metrics.get('value_loss'),seconds=data.get('seconds'),
            coverage=data.get('army_elixir_diagnostics',{}))))
    log=directory/'train.log'
    if log.exists() and not (directory/'exit.json').exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-700));print('TAIL',directory.name,f.read().decode(errors='replace'))
    result=directory/'evaluation/result.json'
    if result.exists():
        data=json.loads(result.read_text())
        print('EVAL',directory.name,json.dumps({k:data[k] for k in ('games','wins','losses','draws')}))
backup=Path('/content/drive/MyDrive/cr_rl_ppo/preparation/army_elixir_results')
backup.mkdir(exist_ok=True)
for path in out.rglob('*.json') if out.exists() else []:
    if path.name.startswith('simulator_') or path.name.startswith('match_'): continue
    if path.stat().st_size>2*1024**2: continue
    target=backup/path.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target)
