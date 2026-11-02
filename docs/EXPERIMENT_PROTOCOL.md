# Experiment protocol

1. **M0 simulator:** boot JVM, resolve deck from catalogue, run parity scenarios and deterministic smoke tests. No large training before acceptance.
2. **M1 data/rollouts:** raw observations, masks, actions, execution/refusal, rewards, done/truncated, seeds, policy version and opponent ID. Persist sharded Parquet/NPZ with run metadata.
3. **M2 model:** first overfit a tiny synthetic batch; test causal masking, legal-action masking, action log-prob round trips and episode-boundary reset. Then benchmark small and production presets.
4. **M3 BC:** import replay labels only after verifying replay version, deck, action time and seat observation. Split by whole match/player; don't leak both sides of one replay across splits.
5. **M4 PPO:** begin against scripted opponents; terminal win/loss reward first, then compare potential-based tower-HP shaping. Store old log-prob/value and recompute with exact same history window.
6. **M5 self-play:** opponent pool with latest, historical and scripted agents; maintain a fixed held-out evaluation pool and cross-play matrix.

The simulator's own `label_replays` and reference fixtures are useful starting points. Human BC is optional for getting the first PPO pipeline running. Never train PPO on arbitrary stale human trajectories as if they were on-policy rollouts.
