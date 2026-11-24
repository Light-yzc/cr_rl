"""Bounded Colab API transfers with retries; no content hashes."""
from pathlib import Path
import json
import subprocess
import time

root = Path(__file__).resolve().parents[1]
source = root/'artifacts/colab_cpu_prep_20261117/live_sessions.zip'
with source.open('rb') as stream:
    index = 0
    while block := stream.read(8*1024**2):
        path = source.parent/f'live_part_{index:03d}'
        path.write_bytes(block)
        for attempt in range(4):
            result = subprocess.run(['colab','upload','-s','cr-train-l4-1117',str(path),
                 '/content/'+path.name],capture_output=True,text=True)
            if result.returncode == 0: break
            print(json.dumps(dict(part=index,retry=attempt+1)),flush=True)
            time.sleep(2)
        else: raise RuntimeError('Chunk upload failed; part '+str(index))
        print(json.dumps(dict(part=index,bytes=len(block),uploaded=True)),flush=True)
        index += 1
print(json.dumps(dict(parts=index,total_bytes=source.stat().st_size,complete=True)),flush=True)
