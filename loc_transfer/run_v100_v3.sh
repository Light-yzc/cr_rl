#!/usr/bin/env bash
set -euo pipefail

# Dual-V100 (NVLink, CUDA_VISIBLE_DEVICES=2,1) max-throughput / low-RAM v3 PPO launcher.
# RTX 4090 (GPU 0) is strictly excluded from training.
# Single-process topology:
#   cuda:0 (physical V100#2, better cooling): Actor rollout/TBPTT, Ref cache, PeerCritic (during warmup & eval only)
#   cuda:1 (physical V100#1): Independent Critic, Frozen Opponent rollout, post-Critic NVLink PeerActor
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv-v100/bin/python" ]]; then
  PYTHON="$ROOT/.venv-v100/bin/python"
else
  PYTHON="python3"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-2,1}"
if [[ ",${CUDA_VISIBLE_DEVICES}," == *",0,"* ]]; then
  echo "Refusing GPU_IDS=${CUDA_VISIBLE_DEVICES}: RTX 4090 (GPU 0) is excluded; use Dual-V100 GPU_IDS=2,1." >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-$ROOT/artifacts/ppo_v3_v100}"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/ppo_prep_20261113/best_with_v3_critic_v2.pt}"
BC_ANCHOR="${BC_ANCHOR:-$ROOT/artifacts/ppo_prep_20261113/best.pt}"
RESUME="${RESUME:-}"
DECKS="${DECKS:-$ROOT/configs/decks_augmented.yaml}"
TARGET_UPDATES="${TARGET_UPDATES:-250000}"

if [[ -n "$RESUME" ]]; then
  RESUME_CKPT="$RESUME"
  SOURCE_ARGS=(--resume "$RESUME" --expand-decks "$DECKS")
elif [[ -f "$RUN_DIR/ppo_latest.pt" ]]; then
  RESUME_CKPT="$RUN_DIR/ppo_latest.pt"
  SOURCE_ARGS=(--resume "$RUN_DIR/ppo_latest.pt" --expand-decks "$DECKS")
else
  RESUME_CKPT=""
  SOURCE_ARGS=(--init-from-bc "$CHECKPOINT")
fi

if [[ -z "${UPDATES:-}" ]]; then
  if [[ -n "$RESUME_CKPT" && -f "$RESUME_CKPT" ]]; then
    DONE_UPDATES="$("$PYTHON" -c '
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
print(int(ckpt.get("updates", 0)))
' "$RESUME_CKPT")"
    UPDATES=$(( TARGET_UPDATES - DONE_UPDATES ))
    if (( UPDATES <= 0 )); then
      echo "[run_v100_v3] target updates ($TARGET_UPDATES) already reached ($DONE_UPDATES); exiting 0."
      exit 0
    fi
    echo "[run_v100_v3] resuming from update $DONE_UPDATES; scheduling remaining $UPDATES updates toward target $TARGET_UPDATES."
  else
    UPDATES="$TARGET_UPDATES"
  fi
fi

ARGS=(
  --v3
  --device cuda:0
  --critic-device cuda:1
  --opponent-device cuda:1
  --balanced-critic
  --amp-dtype "${AMP_DTYPE:-fp16}"
  --precision "${PRECISION:-fp16}"
  "${SOURCE_ARGS[@]}"
  --bc-anchor "${BC_ANCHOR}"
  --out "$RUN_DIR"
  --decks "$DECKS"
  --envs "${ENVS:-48}"
  --env-threads "${ENV_THREADS:-20}"
  --torch-threads "${OMP_NUM_THREADS}"
  --torch-interop-threads 1
  --steps "${STEPS:-384}"
  --updates "${UPDATES}"
  --epochs "${EPOCHS:-2}"
  --critic-epochs "${CRITIC_EPOCHS:-2}"
  --batch-size "${BATCH:-8}"
  --actor-batch-size "${ACTOR_BATCH:-4}"
  --critic-batch-size "${CRITIC_BATCH:-4}"
  --grad-accumulation "${ACCUM:-8}"
  --actor-effective-batch-size "${ACTOR_EFF:-512}"
  --critic-effective-batch-size "${CRITIC_EFF:-2048}"
  --tbptt-chunk "${TBPTT_CHUNK:-128}"
  --frame-microbatch "${FRAME_MICROBATCH:-64}"
  --critic-mode independent
  --critic-warmup-updates "${CRITIC_WARMUP_UPDATES:-8}"
  --critic-warmup-games "${CRITIC_WARMUP_GAMES:-4}"
  --learning-rate "${LEARNING_RATE:-2.5e-6}"
  --critic-learning-rate "${CRITIC_LEARNING_RATE:-3e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --gamma "${GAMMA:-0.9995}"
  --gae-lambda "${GAE_LAMBDA:-0.985}"
  --critic-gae-lambda "${CRITIC_GAE_LAMBDA:-0.992}"
  --clip-coef "${CLIP_COEF:-0.2}"
  --value-coef "${VALUE_COEF:-0.5}"
  --card-entropy-coef "${CARD_ENTROPY_COEF:-0.003}"
  --tile-entropy-coef "${TILE_ENTROPY_COEF:-0.0005}"
  --reference-kl-coef "${REFERENCE_KL_COEF:-0.02}"
  --anchor-kl-coef "${ANCHOR_KL_COEF:-0.002}"
  --target-kl "${TARGET_KL:-0.02}"
  --target-card-kl "${TARGET_CARD_KL:-0.015}"
  --target-tile-kl "${TARGET_TILE_KL:-0.06}"
  --max-initial-logp-error "${MAX_INITIAL_LOGP_ERROR:-0.002}"
  --adaptive-actor-lr
  --adaptive-actor-lr-min "${ADAPTIVE_ACTOR_LR_MIN:-1e-6}"
  --adaptive-actor-lr-max "${ADAPTIVE_ACTOR_LR_MAX:-5e-6}"
  --override-adaptive-actor-lr
  --reserve-to-wait
  --gradient-checkpointing
  --temporal-checkpoint-mode "${TEMPORAL_CHECKPOINT_MODE:-block}"
  --dual-rollout
  --historical-cohort-size "${HISTORICAL_COHORT_SIZE:-2}"
  --historical-cohort-updates "${HISTORICAL_COHORT_UPDATES:-2}"
  --focal-decks "${FOCAL_DECKS:-giant_sparky_v1,user_giant_sparky_pump_plain,hog_cycle_v1}"
  --focal-deck-prob "${FOCAL_DECK_PROB:-0.50}"
  --army-reward "${ARMY_REWARD:-0.015}"
  --actor-choice-only
  --tower-shaping
  --princess-reward "${PRINCESS_REWARD:-0.05}"
  --king-reward "${KING_REWARD:-0.05}"
  --crown-reward "${CROWN_REWARD:-0.15}"
  --overflow-penalty "${OVERFLOW_PENALTY:-0.0002}"
  --allow-reward-migration
  --role-league
  --current-opponent-fraction "${CURRENT_OPPONENT_FRACTION:-0.60}"
  --historical-opponent-fraction "${HISTORICAL_OPPONENT_FRACTION:-0.25}"
  --anchor-opponent-fraction "${ANCHOR_OPPONENT_FRACTION:-0.15}"
  --pool-capacity "${POOL_CAPACITY:-20}"
  --history-snapshot-interval "${HISTORY_SNAPSHOT_INTERVAL:-75000}"
  --opponent-cache-size "${OPPONENT_CACHE_SIZE:-4}"
  --training-opponent-cache-size "${TRAINING_OPPONENT_CACHE_SIZE:-3}"
  --rollout-save-interval "${ROLLOUT_SAVE_INTERVAL:-0}"
  --keep-rollouts "${KEEP_ROLLOUTS:-96}"
  --compact-host-memory
  --cache-reference
  --keep-checkpoints "${KEEP_CHECKPOINTS:-2}"
  --keep-periodic-checkpoints "${KEEP_PERIODIC_CHECKPOINTS:-1}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-800}"
  --validation-interval "${VALIDATION_INTERVAL:-0}"
  --validation-pairs "${VALIDATION_PAIRS:-20}"
  --tensorboard-dir "${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"
  --progress-jsonl "${PROGRESS_JSONL:-$RUN_DIR/progress.jsonl}"
)

if [[ "${PPO_ACTOR_FP32:-1}" == "1" ]]; then
  ARGS+=(--ppo-actor-fp32)
fi

if [[ "${CACHE_STATE_FRAMES:-1}" == "1" ]]; then
  ARGS+=(--cache-state-frames)
fi

if [[ "${FULL_EPISODE_BPTT:-1}" == "1" ]]; then
  ARGS+=(--full-episode-bptt)
fi

if [[ -n "${VALIDATION_DECKS:-}" ]]; then
  ARGS+=(--validation-decks "${VALIDATION_DECKS}")
fi

if [[ -n "${MIN_TRAINING_GAMES:-}" ]]; then
  ARGS+=(--min-training-games "${MIN_TRAINING_GAMES}")
fi

mkdir -p "$RUN_DIR"
echo "$$" > "$RUN_DIR/trainer.pid"
exec "$PYTHON" -u scripts/train_selfplay.py "${ARGS[@]}" "$@"
