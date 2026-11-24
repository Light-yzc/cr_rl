#!/usr/bin/env bash
set -euo pipefail

# Default v3 entrypoint strictly targets the 2x Tesla V100 GPUs (GPU_IDS=1,2).
# RTX 4090 (GPU 0) must not participate in training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export GPU_IDS="${GPU_IDS:-1,2}"
if [[ ",${GPU_IDS}," == *",0,"* ]]; then
  echo "Refusing GPU_IDS=${GPU_IDS}: RTX 4090 (GPU 0) is excluded; use Dual-V100 GPU_IDS=1,2." >&2
  exit 1
fi
exec bash "$ROOT/loc_transfer/run_v100_v3.sh" "$@"
