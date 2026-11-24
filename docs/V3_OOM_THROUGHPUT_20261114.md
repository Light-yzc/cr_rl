# V3 OOM and throughput repair — 2026-11-14

The active run is `artifacts/ppo_v3_v100`. Critic backward previously exhausted
the second V100's 15.77 GiB VRAM. History replicas, full-episode recurrent
prefixes and training activations share that card. A smaller critic microbatch
reduces activation peaks without reducing PPO's configured effective batch.

## Applied changes

- Launcher Actor/Critic microbatches: 6 / 4, previously 4 / 12.
- Effective batches remain Actor 256 / Critic 2048; 24 environments, 384 steps,
  two epochs, recurrent burn-in, TBPTT 32, independent critic, GAE, KL guards,
  anchor KL, role mixture and PFSP remain enabled with their prior settings.
- `slice_memory` avoids a CUDA scalar read for untrimmed slices. Frozen
  opponent groups transfer their position vector once and supply exact bounds
  to individual slices. KV tensors are trimmed before advanced indexing,
  avoiding allocation/copy of the unused capacity for every lane.
- Graceful shutdown skips a duplicate checkpoint when the current update has
  already been saved. Previously an update on a checkpoint boundary could
  synchronously write the same 4.1 GiB checkpoint twice on shutdown.
- Optional immutable opponent read cache on Intel's **data partition**:
  `/dev/nvme0n1p4`, mounted at `/run/media/regen/本地磁盘`.
  `CRRL_OPPONENT_DISK_CACHE=/run/media/regen/本地磁盘/cr_rl_active/frozen_opponent_cache_v3`.
  Cache cap: 6 GiB; LRU eviction; source size/mtime/path invalidation; 2 GiB
  free-space headroom before a copy; originals remain on HDD. No writes were
  made to Intel's Win system partition `/dev/nvme0n1p1`, or GALAX.
- Automatic validation stays disabled, as requested; trajectory win-rate
  logging remains active. RTX 4090 is excluded from training.

## Verification and observed performance

Related CPU tests: 32 passed, 8 deselected. Targeted bounded-cache and
memory-slicing parity tests: 2 passed. `git diff --check` passed for changed
files. GPU validation here consists of live training, not the GPU unit suite.

Training resumed from saved update 136. Three consecutive full updates:

| Update | Wall seconds | Learner samples/s | Actor seconds | Critic seconds | Peak V100 1 / 2 GiB |
|---|---:|---:|---:|---:|---:|
| 137 | 91.69 | 100.51 | 27.54 | 12.43 | 9.96 / 10.93 |
| 138 | 97.46 | 94.57 | 24.24 | 11.45 | 9.29 / 11.34 |
| 139 | 84.49 | 109.08 | 24.14 | 11.75 | 9.41 / 11.46 |

These updates had no OOM or process restart. Actor overflow skips: 0; critic
overflow skips: 0, 1, 1 respectively. Skips are still recorded, so this is not
an assertion that all numerical issues disappeared. Initial chosen log-prob
error at update 137 was 0.00219.

Aggregate throughput for these 27,648 learner transitions is approximately
101 samples/s. Previous updates 133–135 averaged approximately 93 samples/s.
The difference is about 9%, but trajectories, prefix lengths, entity counts
and disk-cache state differ: this is a short operational comparison, not a
controlled benchmark or evidence of maximum achievable throughput. Rollout
and opponent loading still dominate wall time. Long-run OOM freedom is not
established by three updates.

The current launcher target remains 1200 updates and checkpoint interval 8;
this repair did not change that existing run's training budget or save cadence.
