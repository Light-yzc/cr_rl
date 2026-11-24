"""Show bounded native-BC/model-contract audit output."""
from pathlib import Path

work=Path('/content/crrl_training/public_bc_full')
report=work/'model_contract_audit.json'
log=work/'contract_audit.log'
if report.exists():
    print(report.read_text())
elif log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-3000))
        print(stream.read().decode(errors='replace'))
