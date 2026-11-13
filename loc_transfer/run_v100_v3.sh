#!/usr/bin/env bash
set -euo pipefail

# V100 launch for the explicit-memory v3 PPO driver.
# V100 uses FP16 autocast; recurrent matrices remain FP32 inside the model.
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-2}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the v3 BC best.pt}
RESUME=${RESUME:-}
DECKS=${DECKS:-configs/decks_ppo46.yaml}
OUT=${OUT:-artifacts/ppo_v3_v100}
ENVS=${ENVS:-2}
STEPS=${STEPS:-8}
UPDATES=${UPDATES:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-5600}
VALIDATION_PAIRS=${VALIDATION_PAIRS:-5}
FRAME_MICROBATCH=${FRAME_MICROBATCH:-8}

# Keep the simulator extension visible when launching from a clean shell.
export PYTHONPATH="${PWD}/src:${PWD}/.cache/vendor/crforge/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "$RESUME" ]]; then
  SOURCE_ARGS=(--resume "$RESUME")
else
  SOURCE_ARGS=(--init-from-bc "$CHECKPOINT")
fi

exec torchrun --standalone --nproc_per_node="$NPROC" scripts/train_selfplay.py \
  --v3 --device cuda --precision fp16 \
  "${SOURCE_ARGS[@]}" --decks "$DECKS" \
  --envs "$ENVS" --steps "$STEPS" --updates "$UPDATES" \
  --frame-microbatch "$FRAME_MICROBATCH" \
  --tensorboard-dir "${TENSORBOARD_DIR:-$OUT/tensorboard}" \
  --epochs "${EPOCHS:-2}" --learning-rate "${LEARNING_RATE:-1e-6}" \
  --entropy-coef "${ENTROPY_COEF:-0.0001}" --clip-coef "${CLIP_COEF:-0.2}" \
  --value-coef "${VALUE_COEF:-0.5}" --gae-lambda "${GAE_LAMBDA:-0.95}" \
  --gamma "${GAMMA:-0.999}" --max-initial-logp-error "${MAX_INITIAL_LOGP_ERROR:-0.1}" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --validation-interval "$VALIDATION_INTERVAL" --validation-pairs "$VALIDATION_PAIRS" \
  --bc-anchor "$CHECKPOINT" --out "$OUT"
