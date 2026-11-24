"""Read a trainer's Python stack to diagnose delayed graceful shutdown."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
pid='26990'
command=Path('/proc')/pid/'cmdline'
if not command.exists() or b'/train_selfplay.py\0' not in command.read_bytes():
    print('Old trainer no longer active')
else:
    if not shutil.which('py-spy'):
        subprocess.run([sys.executable,'-m','pip','install','--quiet','py-spy'],check=True,
                       stdout=subprocess.DEVNULL)
    result=subprocess.run(['py-spy','dump','--pid',pid],capture_output=True,text=True,timeout=20)
    print(result.stdout[:5500])
    if result.returncode: print('STACK_PROBE_ERROR',result.stderr[-1500:])
    if shutil.which('jcmd'):
        result=subprocess.run(['jcmd',pid,'Thread.print'],capture_output=True,text=True,timeout=20)
        print('JAVA_STATUS',result.returncode,result.stderr[:1000])
        blocks=result.stdout.split('\n\n')
        Path('/content/crrl_cache_upgrade/java_threads.txt').write_text(result.stdout)
        for block in blocks:
            if 'org.crforge' in block and ('RUNNABLE' in block or 'java.lang.Thread.State' in block):
                print('JAVA',block[:6500])
