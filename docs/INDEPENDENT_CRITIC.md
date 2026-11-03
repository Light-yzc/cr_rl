# Independent Transformer Critic (v2)

New `train_selfplay.py` runs default to independent actor and critic networks.
BC and legacy shared PPO keep their compatible weight layout. The new actor
removes the value readout while preserving the BC action distribution (including
v2 Tile Head flags `tile_query_residual` and `tile_head_fp32`). The critic
owns its entity/history Transformers, embeddings and value readout. Compatible BC
encoder tensors are copied by value; no parameter storage is shared.

For the production preset, the deployable actor has 119,246,085 parameters and the
independent critic has 118,057,985. The removed shared value head had 295,681
parameters. Total trainable capacity is about 237M, while deployment carries only
the actor. Frozen rollout/opponent copies consume additional training memory.

## Fixed rollout targets & v2 dense reward shaping

Collection and final-state bootstrapping finish before any optimizer runs. Store
`log pi_old`, `V_old`, and next-state old values. Compute GAE once, then compute
`returns = raw_advantages + V_old` before normalizing actor advantages. GAE runs
without gradients; optimization detaches and copies its targets. They remain fixed
through all actor/value epochs. A separate full old-critic copy is unnecessary in
this synchronous schedule, since the old predictions have already been saved.
Asynchronous collection would require explicit critic snapshots/versioning.

True terminals bootstrap zero. Time limits bootstrap the final observation but
stop the GAE trace before reset. Simulator faults are excluded. Value regression
uses half-MSE, not value clipping; a future clipped variant must also anchor to
the saved `V_old`.

**v2 Reward Shaping & GAE Horizon Upgrade**:
- Instead of relying purely on sparse terminal win/loss rewards ($\pm 1.0$) with a short GAE horizon ($\gamma=0.999, \lambda=0.95 \Rightarrow \sim 4.9\text{s}$), v2 feeds the critic and GAE a **zero-sum composite reward** computed directly from the `crforge` observation header at each 250ms step:
  $$r_t = r_t^{\text{terminal}} + \big[\Phi_{\text{hp}}(s_t) - \Phi_{\text{hp}}(s_{t-1})\big] + w_{\text{crown}}\big[\Delta \text{crowns}_t - \Delta \text{crowns\_enemy}_t\big] + r_t^{\text{overflow}}$$
  with default weights $w_{\text{princess}} = 0.15$, $w_{\text{king}} = 0.20$, $w_{\text{crown}} = 0.15$, and elixir overflow penalty $r_t^{\text{overflow}} = -0.001$ when `elixir >= 99000` and `action == WAIT`.
- GAE hyperparameters are extended to **$\gamma = 0.9995, \lambda = 0.98$** (effective horizon $\approx 48.8\text{ steps} \approx 12.2\text{s}$), enabling the independent critic to rapidly learn intermediate board valuation and propagate credit back to troop deployments 5–12 seconds earlier.

## Warmup and optimization (v2)

The actor stays frozen for at least two successful critic-fitting rollout rounds
and four complete fault-free games. Warmup uses discounted returns (including v2
dense tower-HP and crown shaping) for complete episodes. To avoid discarding 30–40%
of rollout steps in unfinished games at the end of a rollout window, v2 runs
`train_selfplay.py --updates N` in a **single persistent process** so ongoing
matches continue seamlessly across update boundaries rather than being reset by
per-update subprocesses.

After warmup, actor and critic use separate optimizers and training loops. Critic
fitting completes its own budget even if actor KL exceeds its threshold. The
actor guard uses exact action-distribution reverse KL to the frozen behavior policy.
**In v2, KL early-stopping is evaluated over full gradient-accumulation steps
(effective batch size $\ge 64$, e.g. `batch_size=4~8, grad_accumulation=16~32`)**
rather than aborting on single 4-sample microbatch noise spikes. Value loss cannot
backpropagate into actor parameters.

```bash
uv run python scripts/train_selfplay.py --model production --device cuda \
  --init-from-bc artifacts/bc_selected.pt --decks configs/decks.yaml \
  --critic-mode independent --critic-model same --critic-learning-rate 1e-4 \
  --critic-epochs 4 --critic-warmup-updates 2 --critic-warmup-games 4 \
  --envs 2 --steps 1536 --epochs 2 --batch-size 4 --grad-accumulation 4 \
  --updates 5 --out artifacts/independent_run \
  --progress-jsonl artifacts/independent_run/progress.jsonl
```

`--critic-model same` copies the matching BC encoder dimensions. A production actor
can instead use `--critic-model general_300m`; only shape-compatible tensors transfer
when dimensions differ, and actual copied counts are reported. Both currently
require the same history length. Benchmark capacity and learning before selecting
the larger option.

## Recovery, deployment, monitoring

`ppo_latest.pt` contains both networks, both optimizers, warmup progress, league
configuration and RNG state. Resume automatically uses its saved critic mode and
dimensions. Explicitly incompatible modes/configurations are rejected. Old shared
PPO checkpoints resume in shared mode; start a new BC-initialized run to migrate.
Simulator episodes restart at each rollout/process boundary.

`actor_export.pt` contains policy weights/config and simulator pins, without any
critic or optimizer weights, and loads with `scripts/evaluate_policy.py`.
Training checkpoints use `training_architecture_version=3`; the actor layout has
`model_architecture_version=2, policy_only=true`. Drive publication verifies the
full training checkpoint plus `exports/update_NNNNNN.actor.pt` for each generation.

`critic/value_loss`, `critic/grad_norm` and `critic/learning_rate` use an independent
`charts/critic_optimizer_steps` axis. Actor updates use `charts/optimizer_steps`.
Warmup counts and `value/actor_enabled` describe the schedule. Actor updates do not
emit a fabricated zero value loss.

```bash
uv run python scripts/validate_separate_critic.py \
  --out artifacts/independent_acceptance --device cuda
# Add --model production --init-from-bc PATH to test the existing BC actor.
# Add --drive /content/drive/MyDrive/cr_rl/runs/NAME for verified backups.
```

Independent acceptance checks full-game warmup with unchanged actor weights,
Drive-style publication/restoration to another directory, resumed actor/value
updates and policy-only export. Its one-round/one-game warmup is a test budget.
Existing `validate_training.py` / `validate_bc_transfer.py` explicitly exercise the
legacy shared path. Unit tests also force actor KL stopping, check critic progress,
and check fixed GAE targets after old-value tensors are changed.
