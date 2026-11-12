#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v java >/dev/null || { echo 'Install openjdk-21-jdk-headless first' >&2; exit 1; }
if [[ ! -x .venv-v100/bin/python ]]; then uv venv .venv-v100 --python 3.12; fi
uv pip install --python .venv-v100/bin/python --index-url https://pypi.org/simple \
  -e . 'torch==2.6.0' 'tensorboard==2.21.0' pytest ruff
.venv-v100/bin/python scripts/setup_simulator.py
CUDA_VISIBLE_DEVICES="${GPU_IDS:-1,2}" .venv-v100/bin/python -m cr_rl.cli doctor
