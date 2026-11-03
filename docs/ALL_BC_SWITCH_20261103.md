# Switch from sampled BC to full action-label coverage

The switch was armed at old epoch 7. Epoch 8 was published to Drive and verified
before the old sampled worker was retired. Its SHA-256 is
`e406228d4601a06c507e5de01b16754e152f3b5ce1251f58da23ddfe08ae856e`.

`run_all_bc.py` now trains immutable snapshots of every currently published shard
from the original 381-record dataset and the growing broad dataset. New shards
enter the next full-pass stage, at an epoch checkpoint boundary. Source records
are not deduplicated. Each source/game identity has a stable train/validation
assignment across these stages; evaluation deck combinations remain excluded.
The legacy sampled run used a different validation split, so the new within-run
validation split should not be described as never seen by any earlier model.

The first snapshot contains 1,421 records and 2,233,092 history rows. After deck
exclusions and validation splitting, its training labels include 33,225 card plays
and 672 abilities. The full pass covers all 33,897 action labels once and samples
33,897 WAIT labels, for 67,794 training draws. Each epoch records exact action
coverage. This is all prepared data, not all upstream raw records: conversion
continues in parallel over the supplied 11,592 candidates.

The model and AdamW state continue from old optimizer step 1,368. The new
[W&B run](https://wandb.ai/weidemingdeguge-test/cr-rl/runs/8zuqhx5i)
has passed step 1,408. A full model/optimizer/RNG/within-epoch sampling checkpoint
was verified in Drive at:

`MyDrive/cr_rl/runs/119m_all_bc_20261103/stage_0001/bc/bc_latest.pt`

Checkpoint SHA-256 at step 1,408:
`3ab5a6de083a1e5584bb9ad920f332c859602b86561494c72681c0944523d657`.
The latest file is replaced atomically every 128 optimizer steps, and complete
epoch checkpoints are retained separately. At a partial-epoch resume, retain the
same immutable dataset manifest, batch size, gradient accumulation and split flags.
`source_locations.json` maps frozen snapshot files to their existing Drive copies.

The streaming loader uses at most 8 GiB for decompressed observations/masks, plus
small scalar arrays and training indices. It keeps compressed source data and
expands only the currently accessed shards. Sampling groups batches by shard to
avoid repeatedly inflating the whole corpus. There are no complete local downloads.

Acceptance on real replay shards demonstrated all of the following:

- 32 merged records preserve both copies without deduplication.
- All 540 training action labels are seen exactly once.
- Resume at step 2 reaches the same step 135 as uninterrupted training, with
  bitwise-identical model weights and optimizer tensors on CPU.
- Carrying the optimizer into a new BC stage advances the counter from 135 to 137.
- The cache eviction test verifies bounded state-file storage and stable returned
  arrays after eviction. The Linux unit suite has 38 passing tests.

Observed training blocks after startup take 37-46 seconds for 1,536 draws
(approximately 33-41 draws/s); startup validation and decompression took several
minutes. The current snapshot is a roughly half-hour full pass after startup,
with I/O-dependent variance. A complete pass over the final corpus can take hours;
do not extrapolate the old 16,384-draw epoch duration to full-data BC.

PPO is considered after source processing finishes and a full BC pass on the
final snapshot completes. The pipeline writes `ppo_readiness.json` then, requesting
held-out BC metrics and paired multi-deck battle evaluation. Loss alone is not a
policy-strength gate. Initial battle screening is not a statistically reliable
win-rate estimate. If evaluation is acceptable, first freeze the actor and warm
the independent Transformer critic using complete-game returns (at least two
successful rounds and four complete games); PPO actor updates follow that warmup.
The current worker does not silently start PPO without this evaluation.

The real Colab monitoring cell supervises the data and BC workers for up to 12
hours, releases the runtime only after verified completion/Drive flush, and retains
published shards/checkpoints on interruption. This is a work budget, not a Colab
runtime-lifetime guarantee.
