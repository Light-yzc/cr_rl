"""Prepare a debugger for saving an idle Actor/Critic from a stuck collector."""
import subprocess
import shutil
from pathlib import Path
import json
import sys
work=Path('/content/crrl_cache_upgrade')
if '--worker' in sys.argv:
    if not shutil.which('gdb'):
        subprocess.run(['apt-get','update','-qq'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['apt-get','install','-y','-qq','gdb'],check=True,stdout=subprocess.DEVNULL)
    (work/'debugger_ready.json').write_text(json.dumps(dict(gdb=shutil.which('gdb'))))
else:
    with (work/'debugger_setup.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'/content/crrl_prepare_rescue.py','--worker'],
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Debugger setup PID',p.pid)
