"""Validate replay integration on the existing CPU VM, without training weights."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/content/crrl_training')
work=root/'awbc_validation';work.mkdir(exist_ok=True)
if '--worker' in sys.argv:
    if importlib.util.find_spec('pytest') is None:
        subprocess.run([sys.executable,'-m','pip','install','--quiet','pytest'],check=True)
    env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'),
             OMP_NUM_THREADS='2')
    command=[sys.executable,'-m','pytest','-q','tests/test_awbc_playbuffer.py',
        'tests/test_live_awbc.py','tests/test_live_awbc_cli.py','tests/test_telemetry.py',
        'tests/test_awbc_colab_backup.py',
        'tests/test_state_ppo_v3.py::test_episode_trace_cross_update_burn_in_and_truncation_bootstrap',
        'tests/test_state_ppo_v3.py::test_behavior_model_sync_and_per_update_kl_and_validation_preservation',
        '--basetemp='+str(work/'pytest_tmp')]
    result=subprocess.run(command,cwd=root,env=env,timeout=600)
    (work/'result.json').write_text(json.dumps({'exit_code':result.returncode,'command':command}))
    raise SystemExit(result.returncode)
else:
    with (work/'tests.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(root/'scripts/colab_test_awbc_playbuffer.py'),'--worker'],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'test_pid':process.pid,'log':str(work/'tests.log')}))
