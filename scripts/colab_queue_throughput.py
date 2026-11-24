"""Run L4 throughput comparisons only after the combined pipeline has passed."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

work=Path('/content/crrl_pipeline_validation')
root=Path('/content/crrl_training')
if '--worker' in sys.argv:
    lock=(work/'throughput.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (work/'result.json').is_file():
        pid=int((work/'validation.pid').read_text())
        path=Path(f'/proc/{pid}/cmdline')
        try: alive=b'/content/crrl_queue_pipeline.py\0' in path.read_bytes()
        except OSError: alive=False
        if not alive: raise RuntimeError('Pipeline validation exited without a result')
        time.sleep(15)
    result=json.loads((work/'result.json').read_text())
    if result['exit_code']!=0: raise RuntimeError('Pipeline validation failed; throughput sweep not started')
    report=json.loads((root/'runs/awbc-smoke/awbc_smoke_report.json').read_text())
    bc=json.loads((root/'runs/awbc-smoke/expert_bc_latest_report.json').read_text())
    if not report['passed'] or not bc.get('actor_attempted'):
        raise RuntimeError('Both auxiliary phases must exercise Actor optimization before profiling')
    subprocess.run([sys.executable,'-u',str(root/'scripts/benchmark_colab_pipeline.py')],
                   cwd=root,check=True)
else:
    (work/'validation.pid').write_text(str(int(os.environ['CRRL_VALIDATION_PID'])))
    with (work/'throughput.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','/content/crrl_queue_throughput.py','--worker'],
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,log=str(work/'throughput.log'))))
