"""Inspect immutable public source manifest before downloading replay shards."""
import json
from pathlib import Path
from urllib.request import urlopen

revision = '059d43a02138a34b1b3009cc2acc7630fb99a638'
url = f'https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/{revision}/manifest.json'
with urlopen(url, timeout=45) as response:
    body = response.read(8*1024**2+1)
if len(body) > 8*1024**2:
    raise ValueError('Unexpectedly large public manifest')
root = Path('/content/crrl_training/public_bc')
root.mkdir(exist_ok=True)
(root/'source_manifest.json').write_bytes(body)
manifest = json.loads(body)
files = [row for row in manifest['files'] if row['path'].startswith('replays/')]
print(json.dumps({'revision':revision,'source_shards':len(files),'first_shards':files[:2]}))
