#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv-v100/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-1,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_DIR="${RUN_DIR:-$ROOT/artifacts/local_dual_v100}"
export RUN_DIR
RESUME_PATH="${RESUME_PATH:-$RUN_DIR/ppo_latest.pt}"
# The local restore directory may already contain a verified checkpoint.  The
# full mmap/hash verification is intentionally opt-out because it rereads a
# multi-gigabyte HDD file before every restart; the trainer still validates and
# loads the checkpoint itself.
if [[ "${SKIP_RESUME_PREPARE:-0}" != "1" ]]; then
  "$PYTHON" loc_transfer/prepare_resume.py
fi
ARGS=(--v2 --device cuda:0 --critic-device cuda:1 --amp-dtype "${AMP_DTYPE:-fp16}"
  --resume "$RESUME_PATH" --out "$RUN_DIR"
  --envs "${ENVS:-4}" --env-threads "${ENV_THREADS:-4}"
  --torch-threads "${OMP_NUM_THREADS}" --torch-interop-threads 1 --steps "${STEPS:-3072}" --updates "${UPDATES:-600}"
  --epochs 2 --critic-epochs 1 --batch-size "${BATCH:-8}"
  --actor-batch-size "${ACTOR_BATCH:-8}" --critic-batch-size "${CRITIC_BATCH:-16}"
  --grad-accumulation "${ACCUM:-8}" --critic-model same --critic-mode independent
  --critic-warmup-updates 2 --critic-warmup-games 4
  --gamma .9995 --gae-lambda .993 --critic-gae-lambda .995
  --tower-shaping --princess-reward .05 --king-reward .05 --crown-reward .15
  --overflow-penalty 0 --reference-kl-coef .01 --target-kl .02
  --target-card-kl .015 --target-tile-kl .06 --max-initial-logp-error .05
  --actor-choice-only --frame-encoder-fp32 --role-league --disable-holding-prior
  --compact-host-memory --balanced-critic --dual-rollout --incremental-inference --forced-wait-fastpath --cache-reference --defer-scalar-metrics
  --card-entropy-coef .001 --tile-entropy-coef .001
  # Keep the rollout cache large to avoid re-reading historical checkpoints;
  # optimize() trims it to TRAINING_OPPONENT_CACHE_SIZE before Critic work.
  --opponent-cache-size "${OPPONENT_CACHE_SIZE:-8}"
  --training-opponent-cache-size "${TRAINING_OPPONENT_CACHE_SIZE:-2}" --keep-checkpoints 2 --keep-periodic-checkpoints 0
  --history-snapshot-interval 20000 --min-training-games 1258
  --decks configs/decks_expansion12.yaml)
exec "$PYTHON" -u scripts/train_selfplay.py "${ARGS[@]}" "$@"
