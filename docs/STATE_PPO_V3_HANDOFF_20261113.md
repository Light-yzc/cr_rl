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

This is the PPO update core and numerical smoke. The old `train_selfplay.py`
and `selfplay_v2.py` rollout supervisors still produce architecture-v2
sliding-window trajectories; they must not be pointed at this v3 checkpoint.
The next integration step is a v3 native rollout collector that records the
chronological state inputs, action, old log-probability, value, reward, and
terminal flag for each seat, then calls `state_ppo_update`.
