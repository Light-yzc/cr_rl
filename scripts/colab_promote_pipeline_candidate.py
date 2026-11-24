"""Atomically replace the queued archive before the worker opens it."""
import os
from pathlib import Path

work = Path('/content/crrl_pipeline_validation')
incoming = Path('/content/crrl_pipeline_candidate_next.zip')
if not (work/'validation.log').exists() and not (work/'result.json').exists():
    os.replace(incoming,Path('/content/crrl_pipeline_candidate.zip'))
    print('Queued archive updated')
else:
    print('Validation already started; revised archive retained for the next handoff')
