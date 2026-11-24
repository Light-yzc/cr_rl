import os
from pathlib import Path
import subprocess
import sys

root=Path('/content/crrl_training')
result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_terminal_outcome.py',
    'tests/test_live_rl_recording.py','--basetemp=/content/crrl_training/terminal_outcome_tests'],
    cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),timeout=120,capture_output=True,text=True)
print(result.stdout);print(result.stderr)
if result.returncode:raise RuntimeError('Terminal result regression failed')
