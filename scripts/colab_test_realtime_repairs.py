import os
from pathlib import Path
import subprocess
import sys

root=Path('/content/crrl_training')
command=[sys.executable,'-m','pytest','-q','tests/test_realtime_recording_repairs.py',
    'tests/test_live_rl_recording.py','tests/test_live_control.py','tests/test_client_diagnostics.py',
    'tests/test_awbc_playbuffer.py','tests/test_live_awbc_cli.py',
    '--basetemp=/content/crrl_training/realtime_repair_tests']
result=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+
    str(root/'.cache/vendor/crforge/python'),OMP_NUM_THREADS='2'),capture_output=True,text=True,timeout=300)
print(result.stdout);print(result.stderr)
if result.returncode:raise RuntimeError('Real-time recorder/loader regression failed')
