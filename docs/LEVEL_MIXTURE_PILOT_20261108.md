# Global level-mixture pilot

The pinned `crforge` fresh-battle API has one card level and one tower level in
`LadderConfig`; it does not accept per-seat or per-card levels. The supported
curriculum therefore assigns a global level pair to each independent simulator
lane. Both seats in a lane share that pair, while different lanes can run
different pairs in parallel. This approximates level variation without writing
private replay state or silently changing the simulator protocol.

Example fresh pilot:

```bash
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=src python scripts/train_selfplay.py \
  --v2 --envs 8 --env-threads 8 \
  --level-mixture 10 11 12 \
  --tower-level-mixture 10 11 12 \
  --out artifacts/level_mixture_pilot_20261108 \
  --updates 3
```

For the distribution you described, keep a mostly uniform base schedule and
perturb a small, reproducible fraction of lanes:

```bash
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=src python scripts/train_selfplay.py \
  --v2 --envs 8 --env-threads 8 \
  --level-mixture 10 11 12 \
  --level-jitter-prob 0.15 --level-jitter 2 \
  --init-from-bc path/to/bc.pt \
  --out artifacts/level_jitter_pilot_20261108 --updates 3
```

Here the main lanes use the uniform base levels `10/11/12` with both seats
equal; roughly 15% of lanes receive a `base±2` level. The actual realized lane
schedule and seed are saved in `level_profile`.

The schedule is assigned cyclically to lanes, so with eight lanes the profile
is `10,11,12,10,11,12,10,11`. The default remains the configured `11/11` in
`configs/deck.yaml`. Start a level-mixture run in a fresh output directory;
initializing from the level-11 BC weights is allowed, but resuming an existing
checkpoint requires the same level profile. Checkpoint and rollout metadata
contain `level_profile`, and a resumed pilot must match it exactly.

This is a generalization pilot, not per-card/per-seat level support. Compare
fixed `11/11` and mixed `10/11/12` runs on the same held-out seeds, with seat
swap and separate win-rate/return summaries by level lane before promoting a
mixture into the long run.
