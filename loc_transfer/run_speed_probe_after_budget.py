"""Wait for the configured training budget, then probe idle V100s in isolation."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
while subprocess.run(['systemctl','--user','is-active','--quiet','crrl-v100.service']).returncode==0:
    time.sleep(10)
summary=json.loads((root/'artifacts/local_dual_v100/summary.json').read_text())
assert summary['completed_training_games_total']>=1258, 'Training stopped before its budget; inspect failure before profiling'
for script in ('benchmark_gradient_division.py','benchmark_foreach_critic.py'):
    subprocess.run([sys.executable,str(root/'loc_transfer'/script)],check=True)
