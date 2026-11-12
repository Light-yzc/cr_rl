from pathlib import Path
import subprocess, sys
root=Path(__file__).resolve().parents[1]
run=(root/'artifacts/local_dual_v100').resolve()
out=root/'artifacts/local_dual_v100_checks/strength_178'
pin=run/'strength_probe_178.pt'
try:
 for name,opponent in [('anchor', root/'artifacts/v100_handoff/opponents/opponent_000000.pt'),('historical139',root/'artifacts/v100_handoff/opponents/opponent_1073741963.pt')]:
  subprocess.run([sys.executable,str(root/'scripts/evaluate_policy.py'),'--policy',str(pin),'--opponent',str(opponent),'--decks',str(root/'configs/decks_expansion12.yaml'),'--out',str(out/name),'--pairs','4','--pair-start','10','--sampling-seed','20261108','--pairing-seed','20261108','--simulator-seed-base','8100000','--device','cuda:0','--frame-cache','--forced-wait-fastpath'],check=True)
finally:
 pin.unlink(missing_ok=True)
