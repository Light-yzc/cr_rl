import subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cr_rl.training.colab_transport import run_bounded
try:
 r=run_bounded(['colab']+sys.argv[1:],timeout=90)
 if r.returncode:print('COLAB_COMMAND_FAILED',r.returncode,'connection lost' if b'Connection was lost' in r.stdout+r.stderr else 'retry required');sys.exit(1)
 print(r.stdout.decode('utf-8',errors='replace'))
except subprocess.TimeoutExpired:print('COLAB_COMMAND_TIMEOUT');sys.exit(1)
