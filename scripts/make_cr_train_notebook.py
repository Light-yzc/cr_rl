"""Thin notebook: durable bootstrap + shared preparation/controller scripts."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
cells=[]
def markdown(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
markdown('''# PPO + BC + AWBC
Drive 保留代码包、数据 ZIP、固定起点 Actor、最新完整 PPO 和必要对手池；SSD 随实例销毁，可重建。
没有额外文件哈希扫描。BC 为原生重建的候选辅助数据，尚未证明专家水平；AWBC 通过 Value/独立留出门控才更新 Actor。
新圣水价值奖励仍属实验，长期训练默认保留 U929 已验证的旧奖励。所有配置在 Drive 的 `preparation/cr_train_config.json`。
''')
code('''from pathlib import Path
from google.colab import drive
import shutil, zipfile, sys, runpy
DRIVE = Path('/content/drive/MyDrive/cr_rl_ppo')
if not DRIVE.is_dir(): drive.mount('/content/drive')
ROOT = Path('/content/crrl_training')
ROOT.mkdir(exist_ok=True)
for name, needed in [('crrl_training_code.zip', not (ROOT/'.cache/vendor/crforge').is_dir()), ('crrl_training_update.zip', True)]:
    if not needed: continue
    local = Path('/content')/name
    shutil.copyfile(DRIVE/'preparation'/name, local)
    with zipfile.ZipFile(local) as archive:
        if any(not (ROOT/n).resolve().is_relative_to(ROOT.resolve()) for n in archive.namelist()): raise ValueError('Unsafe ZIP')
        archive.extractall(ROOT)
sys.path.insert(0, str(ROOT/'src'))
runpy.run_path(str(ROOT/'scripts/prepare_cr_train.py'), run_name='__main__')
''')
code('''import os, json
config = json.loads((ROOT/'prepared/cr_train_config.json').read_text())
if config.get('wandb', True):
    from google.colab import userdata
    if not os.environ.get('WANDB_API_KEY'):
        os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
    print('W&B credential ready')
else:
    print('W&B disabled; training progress is recorded locally')
# 凭据仅在进程环境中，不写入 notebook 输出、配置或 ZIP。
''')
code('''import subprocess
subprocess.run([sys.executable, str(ROOT/'scripts/cr_train_pipeline.py'), 'start'], check=True)
''')
code('''import time, json
from IPython.display import clear_output
from cr_rl.training.colab_pipeline import progress
config = json.loads((ROOT/'prepared/cr_train_config.json').read_text())
try:
    while True:
        clear_output(wait=True)
        print(json.dumps(progress(ROOT, config['run_name']), indent=2, ensure_ascii=False))
        exit_file = ROOT/'runs'/config['run_name']/'exit.json'
        if exit_file.exists():
            print(exit_file.read_text())
            break
        time.sleep(15)
except KeyboardInterrupt:
    print('停止显示；后台训练仍在继续。要停止训练，运行下一格。')
''')
code('''# 需要停止训练时取消下一行的注释：等待保存完整 checkpoint 并完成 Drive 备份。
# subprocess.run([sys.executable, str(ROOT/'scripts/cr_train_pipeline.py'), 'stop'], check=True)
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',name='python3'),
             accelerator='GPU',colab=dict(name='cr_train.ipynb')),nbformat=4,nbformat_minor=5)
out=root/'notebooks/cr_train.ipynb'
out.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf8')
print(out)
