#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${COLAB_SESSION:-crrl-l4-20261102b}"
ARCHIVE="/tmp/cr-rl-lab-src.tar.gz"
trap 'colab stop -s "$SESSION" >/dev/null 2>&1 || true' EXIT
cd "$ROOT"
tar -czf "$ARCHIVE" \
  --exclude='./.git' --exclude='./.venv' --exclude='./.cache' \
  --exclude='./artifacts/*' --exclude='./.pytest_cache' --exclude='./.ruff_cache' \
  --exclude='*/__pycache__' .
colab upload -s "$SESSION" "$ARCHIVE" /content/cr-rl-lab-src.tar.gz
colab exec -s "$SESSION" --timeout 900 -f scripts/colab_remote_smoke.py
colab download -s "$SESSION" /content/colab_smoke_results.tar.gz "$ROOT/artifacts/colab_smoke_results.tar.gz"
rm -rf "$ROOT/artifacts/colab_selfplay_smoke"
tar -xzf "$ROOT/artifacts/colab_smoke_results.tar.gz" -C "$ROOT/artifacts"
echo "Colab results: $ROOT/artifacts/colab_selfplay_smoke"
