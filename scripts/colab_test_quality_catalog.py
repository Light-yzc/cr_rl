import os
from pathlib import Path
import subprocess
import sys

root=Path('/content/crrl_training')
result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_quality_bc_catalog.py',
    '--basetemp=/content/crrl_training/catalog_validation_tmp'],cwd=root,
    env=dict(os.environ,PYTHONPATH=str(root/'src')),timeout=90)
if result.returncode:raise RuntimeError('Quality catalog regression failed')
