# State PPO v3 handoff

The v3 recurrent PPO update is implemented in `src/cr_rl/training/state_ppo.py`.
Each PPO sequence starts from a fresh `PolicyMemory`, advances frames in order,
and recomputes log-probabilities and values before applying the clipped ratio
loss. This preserves the 8-tick (0.4 second) state contract and the learned
placement head.

Local smoke:

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/smoke_state_ppo.py `
  --checkpoint artifacts/ppo_prep_20261113/best.pt `
  --device cuda --precision fp16 --steps 2
```

The smoke passed on the local CUDA GPU with finite losses, ratios near one, and
no AMP overflow. V100 should use the same `--precision fp16` path. Do not use
BF16 on V100. Keep recurrent matrices in FP32 and use CUDA autocast FP16 for
the forward/loss path, as the v3 model already does.

`src/cr_rl/training/state_selfplay.py` is the v3 native rollout driver. It
records chronological raw frames, masks, actions, old log-probabilities, values,
rewards and episode boundaries for both seats, then calls `state_ppo_update`.
`scripts/train_selfplay.py --v3` routes directly to it, and
`scripts/run_colab_training.py --v3` plus `loc_transfer/run_v100_v3.sh` provide
the Colab/V100 launch paths. Checkpoint backup, progress JSONL and optional
W&B logging are enabled on the v3 driver.

The real crforge smoke completed one simulator rollout and one PPO update on
the local CUDA GPU. It produced two v3 sequences, two samples, a finite loss,
and `artifacts/ppo_v3_smoke_real/ppo_latest.pt`. Forced-WAIT entropy was zero
for that smoke, as expected; the synthetic smoke covers nonzero categorical
entropy. Illegal flat actions are excluded from the entropy calculation.

The legacy `selfplay_v2.py` path remains available only for architecture-v2
checkpoints. It is never selected for a v3 checkpoint.
