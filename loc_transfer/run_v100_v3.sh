#!/usr/bin/env bash
set -euo pipefail

# V100 launch for the explicit-memory v3 PPO driver.
# V100 uses FP16 autocast; recurrent matrices remain FP32 inside the model.
PYTHON=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the v3 BC best.pt}
DECKS=${DECKS:-configs/decks_ppo46.yaml}
OUT=${OUT:-artifacts/ppo_v3_v100}
ENVS=${ENVS:-2}
STEPS=${STEPS:-1024}
UPDATES=${UPDATES:-100}

exec "$PYTHON" -u scripts/train_selfplay.py \
  --v3 --device cuda --precision fp16 \
  --init-from-bc "$CHECKPOINT" --decks "$DECKS" \
  --envs "$ENVS" --steps "$STEPS" --updates "$UPDATES" \
  --epochs "${EPOCHS:-2}" --learning-rate "${LEARNING_RATE:-1e-6}" \
  --entropy-coef "${ENTROPY_COEF:-0.0001}" --clip-coef "${CLIP_COEF:-0.2}" \
  --value-coef "${VALUE_COEF:-0.5}" --gae-lambda "${GAE_LAMBDA:-0.95}" \
  --gamma "${GAMMA:-0.999}" --out "$OUT"
