import json,shutil,time
from pathlib import Path
d=Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun')
r=json.loads((d/'latest.json').read_text())
if not r.get('actor'):raise RuntimeError('Latest complete checkpoint has no Actor export yet; retry after the next save')
source=(d/r['actor']).resolve()
if not source.is_relative_to(d.resolve()):raise ValueError('Unsafe Actor path')
out=Path('/content/crrl_actor_transfer')/('u'+str(r['update']))
out.mkdir(parents=True,exist_ok=True)
target=out/'actor_export.pt'
if not target.exists() or target.stat().st_size!=r['actor_bytes']:
    temporary=target.with_suffix('.partial');shutil.copyfile(source,temporary)
    if temporary.stat().st_size!=r['actor_bytes']:raise OSError('Incomplete Actor staging')
    temporary.replace(target)
receipt=dict(update=r['update'],bytes=r['actor_bytes'],remote_path=str(target),actor_only=True)
temporary=out.parent/'latest.partial';temporary.write_text(json.dumps(receipt));temporary.replace(out.parent/'latest.json')
print('ACTOR_READY',r['update'])
