"""Inspect only resource/config availability; never print credentials."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

report=dict(cpus=os.cpu_count(),free_gib=shutil.disk_usage('/content').free/1024**3,
    wandb_installed=importlib.util.find_spec('wandb') is not None,
    wandb_env_available=bool(os.environ.get('WANDB_API_KEY')))
try:
    from google.colab import userdata
    value=userdata.get('WANDB_API_KEY')
    if value:
        os.environ['WANDB_API_KEY']=value
        report['wandb_secret_available']=True
except Exception as e:
    report['wandb_secret_available']=False
    report['wandb_secret_error_type']=type(e).__name__
print(json.dumps(report))
