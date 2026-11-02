# Clash Royale RL Lab

An experiment-first Clash Royale agent: human replay BC (optional) -> PPO -> historical-opponent self-play. The simulator is the training backend, not a claim of perfect parity with every live-client mechanic. Client deployment is intentionally out of scope for the first milestone.

## Current backend decision

Primary: [crforge](https://github.com/voonhous/crforge), pinned at `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`. Its compatibility notes name client 16.635.1 and data 16.635.3 (also shipped with client 16.635.7). It provides a 50 ms tick Java battle core, a Python/JVM vector environment, replay labelling, action masks, and recorded-battle conformance tests. This is the strongest current candidate for fidelity, but the exact eight-card deck and Sparky mechanics still need local acceptance tests.

Secondary comparison: [RoyaleGym](https://github.com/RoyaleGym/RoyaleGym), pinned in `configs/backends.yaml`. Do not mix data from the two engines in one training run.

## Local quick start (macOS)

Requirements: Java 21, `uv`, Python 3.12. Run `./scripts/bootstrap_macos.sh`, then `uv run python -m cr_rl.cli doctor`, `uv run python -m cr_rl.cli smoke --episodes 4`, `uv run python -m cr_rl.cli model-check --size tiny`, `uv run python -m cr_rl.cli collect --steps 256 --envs 4 --out artifacts/random_rollout.npz`, `uv run python scripts/benchmark_sim.py --steps 200`, and `uv run pytest`. The bootstrap clones the pinned simulator into `.cache/vendor/`, builds its JVM environment, and fetches the explicitly pinned game data from the game's asset CDN. This is an opt-in network operation. Review the script first. It does not install or modify the official Clash Royale client.

## Colab GPU smoke

With the `colab` CLI authenticated and an L4 session allocated, run `COLAB_SESSION=<session-name> ./scripts/colab_launch.sh`. The launcher uploads a source-only archive, installs Java 21 and the pinned crforge build/data on the Colab VM, runs the simulator doctor, performs a 128-step tiny PPO update on CUDA, checks a production-model forward pass, and downloads checkpoints/metrics into `artifacts/colab_smoke/`. This is a pipeline/GPU compatibility test, not a strength benchmark. The launcher stops the Colab session on exit to avoid idle compute charges.

`scripts/train_ppo.py --updates N` performs N consecutive rollout/update cycles and resumes model, optimizer and RNG state from `ppo_latest.pt` between cycles. `metrics.jsonl` records one summary per update. Use `--resume PATH` to continue an existing run and `--keep-history` to retain lightweight model snapshots.

## Population self-play (new)

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
