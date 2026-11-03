# Clash Royale RL Lab

An experiment-first, shared multi-deck Clash Royale agent: human replay BC -> PPO -> historical-opponent self-play. Formal training starts from BC; random-initialized PPO is reserved for pipeline acceptance. The simulator is the training backend, not a claim of perfect parity with every live-client mechanic.

## Verified training status — 2026-11-04

The active model is the 121M architecture-v2 multi-deck BC policy on Colab Linux/L4.
Stage 1 completed at optimizer step 2616; Stage 2 is training on a frozen dataset
with 134,377 training card-play labels. Full-period replay reconstruction remains
in progress. This is not yet a completed RL training run.

See the [latest progress and recovery notes](docs/TRAINING_PROGRESS_20261104.md)
and the [validation I/O repair](docs/VALIDATION_IO_REPAIR_20261104.md).
The repair passed 113 tests and real-model output comparisons. Training updates,
W&B uploads and a new Drive checkpoint have been independently verified.
PPO promotion still requires the paired multi-deck strength and timing gates.

Datasets, model weights, credentials, private logs and local replay artifacts are
excluded from Git. Historical `artifacts/...` references identify local or Drive
evidence, not files included in this repository.

The human-data collection, held-out deck split, BC, weights-only transfer and Colab backup workflow is in [BC_TO_PPO.md](docs/BC_TO_PPO.md). The implemented network is explained in [MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md).

New PPO runs use an [independent Transformer critic](docs/INDEPENDENT_CRITIC.md),
with complete-game value warmup, separate optimizers and a policy-only export.
Existing shared PPO checkpoints retain their original architecture on resume.
The [audit follow-up](docs/AUDIT_FOLLOWUP_20261103.md) records the selected tile
fix, prefix-safe BC data and remaining PPO blockers. Optional
[tower-potential shaping](docs/REWARD_SHAPING_20261103.md) preserves raw match
outcomes and handles terminal and time-limit boundaries separately.

## Current backend decision

Primary: [crforge](https://github.com/voonhous/crforge), pinned at `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`. Its compatibility notes name client 16.635.1 and data 16.635.3 (also shipped with client 16.635.7). It provides a 50 ms tick Java battle core, a Python/JVM vector environment, replay labelling, action masks, and recorded-battle conformance tests. This is the strongest current candidate for fidelity, but the exact eight-card deck and Sparky mechanics still need local acceptance tests.

Secondary comparison: [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym), pinned in `configs/backends.yaml`. Do not mix data from the two engines in one training run.

## Linux training (primary)

A [matched placement-head trial](docs/TILE_HEAD_FIX_20261103.md) tests query
residuals and an FP32 placement decoder while preserving legacy checkpoint
behavior, BC optimizer state and the BF16 backbone.

The active [cycle-aligned BC stages](docs/CYCLE_BC_CALIBRATION_20261103.md)
continue the architecture-v2 policy, cover every prepared training action label,
incorporate new data between frozen stages and use bounded streaming caches.
The earlier [full-coverage switch](docs/ALL_BC_SWITCH_20261103.md) is retained as
a historical record of the preceding 119M run.

The expanded Cochon123 + IL_Replay BC round, measured L4 batch/kernel tuning,
cloud-only dataset storage and recovery procedure are in
[FORMAL_BC_20261103.md](docs/FORMAL_BC_20261103.md).

The current 119M human-BC to multi-deck PPO run is described in [BC_PPO_RUN_20261103.md](docs/BC_PPO_RUN_20261103.md), including data-quality limits and the corrected `bc_v3/*` metrics. The default league contains 38 distinct training decks; another 10 are held out. Colab Drive backups, PPO metric definitions and runtime-lifetime limits are documented in [TRAINING_MONITORING.md](docs/TRAINING_MONITORING.md). Policy metrics use learner-sample axes and PPO diagnostics use optimizer-step axes.

On Ubuntu, install Java 21, Git and uv, then bootstrap the pinned simulator:

```bash
sudo apt-get update && sudo apt-get install -y openjdk-21-jdk-headless git
python3 -m pip install uv
bash scripts/bootstrap_linux.sh
uv run python scripts/validate_training.py --device cuda
```

The legacy shared-critic acceptance suite runs two tiny-model updates with complete games, moves the run directory and resumes against historical opponents, then runs two general-model updates with the full 128-frame context. It checks legal actions, finite losses, learner-only samples, terminal rewards and optimizer restoration. Results, logs and checkpoints are in `artifacts/linux_validation/`; use a fresh `--out` for another acceptance run. `--skip-general --device cpu` is available for a CPU integration check.

Start PPO after BC and acceptance (the generated league file contains multiple compatible decks):

```bash
uv run python scripts/train_selfplay.py --model production --device cuda \
  --init-from-bc artifacts/bc.pt --decks artifacts/league_train.yaml \
  --envs 2 --steps 1536 --epochs 2 --batch-size 1 --grad-accumulation 8 \
  --updates 100 --out artifacts/general_run
# Continue from the same run directory:
uv run python scripts/train_selfplay.py --model production --device cuda \
  --decks artifacts/league_train.yaml \
  --envs 2 --steps 1536 --epochs 2 --batch-size 1 --grad-accumulation 8 \
  --updates 100 --out artifacts/general_run --resume artifacts/general_run/ppo_latest.pt
```

Resume restores weights, optimizer and RNG state, and checks simulator/data
compatibility. The v2 training driver keeps one self-play process alive across
updates, so ongoing simulator episodes continue at update boundaries. Episodes
restart after a process/runtime restart. Keep the **whole run directory**,
including `opponent_pool.json` and its referenced snapshots, when transferring
machines; pool paths are relative to that directory. The opponent pool is bounded
and snapshot admission is controlled by step and match gates. Short rollouts can
end before a terminal reward; the independent critic requires complete-game
warmup. See [the independent critic](docs/INDEPENDENT_CRITIC.md),
[reward shaping](docs/REWARD_SHAPING_20261103.md) and the
[current handoff requirements](docs/TRAINING_PROGRESS_20261104.md).

## Local quick start (macOS)

Requirements: Java 21, `uv`, Python 3.12. Run `./scripts/bootstrap_macos.sh`, then `uv run python -m cr_rl.cli doctor`, `uv run python -m cr_rl.cli smoke --episodes 4`, `uv run python -m cr_rl.cli model-check --size tiny`, `uv run python -m cr_rl.cli collect --steps 256 --envs 4 --out artifacts/random_rollout.npz`, `uv run python scripts/benchmark_sim.py --steps 200`, and `uv run pytest`. The bootstrap clones the pinned simulator into `.cache/vendor/`, builds its JVM environment, and fetches the explicitly pinned game data from the game's asset CDN. This is an opt-in network operation. Review the script first. It does not install or modify the official Clash Royale client.

## Colab GPU acceptance (Windows / Linux / macOS)

With the `colab` CLI authenticated, run `python scripts/colab_launch.py --gpu L4` (or `--gpu T4`). The launcher allocates a new runtime, uploads source, builds the simulator and runs the Linux acceptance suite above. It downloads logs/checkpoints into `artifacts/<session>/linux_validation/` and stops its runtime on exit. Java and PyTorch are installed on the remote Linux VM; the local launcher only needs Python 3.12+ and the Colab CLI. `bash scripts/colab_launch.sh --gpu L4` is the shell equivalent. Acceptance proves the training pipeline, not policy strength.

`scripts/train_ppo.py --updates N` performs N consecutive rollout/update cycles and resumes model, optimizer and RNG state from `ppo_latest.pt` between cycles. `metrics.jsonl` records one summary per update. Use `--resume PATH` to continue an existing run and `--keep-history` to retain lightweight model snapshots.

## Population self-play (new)

For the larger general-policy experiment, `production` has 119.54M parameters and `general_300m` has 309.23M. Both have now passed L4 full-history forward/backward/AdamW capacity probes with a frozen behavior model and opponent resident. At microbatch 4, measured peak allocated memory was 6.75 / 10.98 GiB and optimization throughput 9.98 / 8.25 samples/s. These are network probes, not online learning or win-rate results. Use 119M for the first formal larger baseline and 309M as a direct comparison; see [the capacity report](docs/MODEL_CAPACITY_20261103.md) for commands and limits. Existing defaults remain unchanged.

Use `uv run python scripts/train_selfplay.py --model general --envs 4 --steps 1536 --epochs 2 --batch-size 4 --grad-accumulation 8 --updates 100 --device auto --out artifacts/selfplay`. This runner trains one randomized learner seat against frozen opponents sampled per environment from a PFSP checkpoint pool, rotates through validated Giant-Sparky / Hog-cycle / Graveyard-control decks, saves two-seat trajectory schema v3 rollouts, and records per-update metrics/checkpoints. Resume with `--resume artifacts/selfplay/ppo_latest.pt`. The recommended starting preset is `general` (~34.2M parameters); `compact` (~16.6M) and `large` (~46.8M) are capacity ablations. See `docs/SELFPLAY_RESEARCH_AND_DESIGN.md` before scaling up. `train_ppo.py` remains a legacy single-policy PPO smoke runner; use `train_selfplay.py` for self-play experiments.

## Deck v1

Giant, Sparky (`ZapMachine` internal row), MiniPekka, Musketeer, Minions, Knight, Zap, Arrows. Names and costs are checked against the selected simulator's live catalogue at runtime; no card numeric IDs or elixir costs are duplicated in Python. Card levels and tower level are a single config value for the first experiments.

## Project map

- `src/cr_rl/`: schemas, simulator adapter, model, BC/PPO, rollouts, opponent pool.
- `configs/`: backend pins, deck, model and training settings.
- `scripts/`: reproducible bootstrap and experiment entry points.
- `tests/`: schema, masks, causal attention, action likelihood and rollout tests.
- `docs/`: simulator acceptance, data protocol and experiment log.
- `artifacts/`: ignored local runs/checkpoints/replays.

## Compute policy

The M4 MacBook Air (10-core, 16 GB unified memory) is for CPU/MPS correctness checks, simulator integration, small-model overfit tests and short rollouts. It is not the 120M training host. `tiny` and `small` presets share the production implementation; the `production` preset follows the plan's 4x256 entity encoder + 16x768 causal temporal trunk and must be benchmarked on CUDA before training. Keep raw observations/actions and policy-version metadata so later GPU runs can resume/replay cleanly.

## Fidelity boundary

The simulator's own versioned reference battles are evidence for covered scenarios only. Our exact deck needs dedicated regression fixtures: Sparky charge/retarget/stun/reset/projectile timing, Giant body-blocking, Zap, Minions air targeting, card cycle, overtime, and simultaneous action fairness. Any unsupported mechanic or missing game-table row blocks large-scale training rather than being silently approximated.
# cr_rl
