"""Generate the CPU preparation and explicitly selected GPU smoke notebook."""
import json
from pathlib import Path


def main():
    cells = []
    def markdown(text):
        cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text):
        compile(text,'notebook-cell','exec')
        cells.append(dict(cell_type='code',metadata={},source=text.splitlines(True),
                          execution_count=None,outputs=[]))
    markdown('''# PPO / strict BC / live AWBC preparation
Run preparation on CPU. Drive contains one full PPO checkpoint; code and public BC sources are prepared separately.
No training starts by default. Auxiliary BC interleaving and temperature variants still require implementation and acceptance.
''')
    code('''from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/cr_rl_ppo')
ROOT = Path('/content/crrl_training')
assert (DRIVE/'ppo_latest.pt').is_file(), 'Missing full PPO checkpoint'
''')
    code('''import hashlib, json, shutil, subprocess, sys, os, zipfile
bundle = DRIVE/'preparation/crrl_training_code.zip'
local_bundle = Path('/content/crrl_training_code.zip')
assert bundle.is_file(), 'Missing prepared code bundle'
shutil.copyfile(bundle, local_bundle)
with zipfile.ZipFile(local_bundle) as archive:
    records = {r['path']:r for r in json.loads(archive.read('CODE_MANIFEST.json'))}
    name = 'scripts/prepare_colab_training_cpu.py'
    body = archive.read(name)
    assert hashlib.sha256(body).hexdigest() == records[name]['sha256']
bootstrap = Path('/content/prepare_crrl_cpu.py')
bootstrap.write_bytes(body)
subprocess.run([sys.executable, str(bootstrap)], check=True)
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python')
''')
    code('''report = ROOT/'prepared/single_checkpoint_preparation.json'
if not report.exists():
    subprocess.run([sys.executable,'-u','scripts/prepare_single_ppo_checkpoint.py',
        '--checkpoint',str(DRIVE/'ppo_latest.pt'),'--out',str(ROOT/'prepared')], check=True)
prepared = json.loads(report.read_text())
print({k:prepared[k] for k in ['generation','actor_fingerprint','historical_policy_count','training_started']})
''')
    markdown('''## Public BC
Full-source preparation downloads all 52 immutable shards and reconstructs them one at a time on CPU.
Its output remains training_ready=false until deduplication, context quality and model contract acceptance. Do not repeatedly launch the worker.
''')
    code('''import runpy
runpy.run_path(str(ROOT/'scripts/colab_public_bc_probe.py'), run_name='__main__')
START_BC_SCAN = False
if START_BC_SCAN:
    subprocess.run([sys.executable,'scripts/colab_full_quality_bc.py'], check=True)
else:
    print('New BC worker disabled; existing reports:')
    full_status = ROOT/'public_bc_full/state.json'
    if full_status.exists():
        status = json.loads(full_status.read_text())
        print({k:v for k,v in status.items() if k not in ['completed','downloaded']})
    for path in sorted((ROOT/'public_bc').glob('native_probe_*/quality_summary.json')):
        print(path.parent.name, json.loads(path.read_text()))
''')
    markdown('''## Explicit GPU stages
Switch runtime to GPU, then rerun preparation. `ppo-smoke` performs one small PPO update; `awbc` needs at least 12 qualified, result-labeled real sessions in Drive sessions/.
No such real sessions are currently present on this account. PPO+BC combined training is not enabled by this notebook.
''')
    code('''STAGE = 'none'  # none / ppo-smoke / awbc
USE_AWBC_PLAYBUFFER = True  # Checks only at update boundaries; insufficient data skips.
if STAGE != 'none':
    command = [sys.executable,'scripts/launch_prepared_pipeline.py','--root',str(ROOT),'--stage',STAGE]
    if STAGE == 'ppo-smoke' and USE_AWBC_PLAYBUFFER:
        command.append('--awbc-playbuffer')
    subprocess.run(command+['--plan-only'], check=True)
    # W&B credentials stay in Colab Secrets; never put them into notebook source.
    try:
        from google.colab import userdata
        os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
    except Exception:
        pass
    if os.environ.get('WANDB_API_KEY'):
        command.append('--wandb')
    subprocess.run(command, check=True)
else:
    print('Preparation only; no training launched.')
''')
    path = Path(__file__).resolve().parents[1]/'notebooks/colab_three_pipelines.ipynb'
    path.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(
        display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),
        indent=1)+'\n',encoding='utf8')
    print(path)


if __name__=='__main__':
    main()
