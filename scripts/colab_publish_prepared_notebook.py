"""Publish the prepared notebook beside the checkpoint without copying weights."""
from pathlib import Path
import shutil

root = Path('/content/crrl_training')
drive = Path('/content/drive/MyDrive/cr_rl_ppo')
for name in ('colab_three_pipelines.ipynb','colab_live_awbc.ipynb'):
    shutil.copyfile(root/'notebooks'/name,drive/name)
shutil.copyfile(root/'docs/COLAB_THREE_PIPELINES_PREPARATION_20261117.md',
                drive/'preparation/TRAINING_PLAN.md')
for name in ('AWBC_PLAYBUFFER_PIPELINE_20261117.md','LIVE_AWBC_COLAB.md',
             'NULLS_AUTOMATIC_TERMINAL_REWARD_20261117.md'):
    shutil.copyfile(root/'docs'/name,drive/'preparation'/name)
print('Published preparation notebook, AWBC notebook and training plan to MyDrive/cr_rl_ppo/')
