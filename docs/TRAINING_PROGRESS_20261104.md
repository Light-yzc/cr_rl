# Training progress and overnight handoff — 2026-11-04

This is a verified progress snapshot, not a claim that BC or RL training has finished.
The latest verified optimizer step, Drive checkpoint, data counts and inspection
time are recorded in the pre-shutdown snapshot at the end of this document.

## Active training

- Model: architecture-v2 multi-deck policy, 121,179,142 parameters, 128-frame history.
- Platform: Colab Linux/L4; run `121m_cycle_bc_20261103`, Stage 2.
- W&B: [Stage 2](https://wandb.ai/weidemingdeguge-test/cr-rl/runs/40willzs).
- Stage 1 completed one full action-label coverage epoch at optimizer step 2616.
- Stage 2 contains 9,463 supervised replay identities after cross-source deduplication.
  Its training partition contains 134,377 card plays and 872 ability labels.
- Across all valid partitions, legal unmasked card-play labels increased from 74,559
  to 199,325. Retained trajectories reaching 120 seconds increased from 9.50% to 34.95%.
- Cycle-aligned data is primary; bounded-omission and strict sources are fallbacks.
  The producer continues the Cochon123 and IL_Replay candidate pools. It has not
  finished processing the entire corpus. Longer replay prefixes do not establish
  exact fidelity to the original human battle state.

BC keeps balanced timing loss separate from conditional card and tile losses.
It excludes ambiguous WAIT labels immediately around human actions, uses legal-tile
Gaussian supervision, BF16 backbone computation and an FP32 placement head.
The current microbatch is 16 with six-way gradient accumulation (effective batch 96).

## Engineering health

The former Stage 2 worker spent over twenty minutes on initial validation and
quick-cohort preparation without a gradient update. Its stack was doing whole-shard
NPZ decompression. A live process was not evidence of healthy training throughput.

The [validation I/O repair](VALIDATION_IO_REPAIR_20261104.md) now persists only exact
validation histories: 5,119 distinct validation rows, 427,389 history rows and about
11.18 GiB of state storage. Selected histories, masks, model log-probabilities and
value outputs matched the original path. All reported metrics on the 1,024-row
balanced and 4,096-row natural validation cohorts matched exactly across the switch.

After switching, optimizer steps advanced beyond 2616, training GPU utilization
reached 100% in an observation, and repeated 768-row quick validations took about
5.44 seconds. Drive step 2688 was independently verified:

```text
bytes: 1476783701
sha256: b03c2d9563a5272fa2479c516e7b520c094dbed7be439b1202e4e04e1fc3aa99
```

W&B's server confirmed new optimizer steps in the existing run. The Linux suite
passes 113 tests. The full source tree compiles. Publication also checks the exact
local source in an isolated Linux workspace before committing it.

## Learning and PPO status

Stage 1's natural conditional validation score was essentially flat: 6.6613 to
6.6620. Card-slot accuracy improved slightly while placement within two tiles
declined slightly. Engineering recovery is not proof of convergence.

The earlier step-1920 candidate, with a WAIT bias selected on separate tuning games,
scored 8–8 against early step237 and 11–5 against conditional step896 over eight
paired deck matchups each. It did not pass promotion, and some held-out games
still accumulated too much full-elixir waiting. PPO has not started.

The queued Stage 2 follow-up audits coverage, waits for its completed and verified
checkpoint, then performs natural timing calibration and action-window diagnostics.
The queued evaluator tests the fixed bias grid, freezes the chosen bias before
final games, and uses fresh pair indices 13–20 for two 16-game baseline comparisons.
It audits actual native successful placements, costs, pre/post elixir and direct
WAIT decisions. A request in a replay is not automatically a successful placement.

PPO still requires full-data BC readiness and paired evaluation approval. Its v2
implementation uses an independent Transformer critic, complete-game critic warmup,
fixed old values/log-probabilities and GAE targets, separate optimizers, persistent
simulator episodes between updates and actor-only exports. Reward shaping uses
`gamma * Phi(next) - Phi(current)`, terminal potential zero, with optional elixir
overflow penalties; gamma is 0.9995 and GAE lambda is 0.98.

## Where to inspect and recover

On the current Colab VM, the project is `/content/cr-rl-lab`. Under its
`artifacts/121m_cycle_bc_20261103/` inspect:

- `status.json`, `pipeline.json`, `worker.json`: controller, current stage and process handles.
- `stage_0002/train.log`, `bc.quick.jsonl`, `bc.data.json`: live updates and frozen sample counts.
- `stage_0002_followup/status.json`: coverage/calibration follow-up.
- `stage_0002_evaluation/status.json`: timing and paired-game evaluation.
- `validation_io_repair/runtime_verification.json`: actual update, W&B and Drive readback proof.

Drive backups are rooted at `MyDrive/cr_rl/`:

- `runs/121m_cycle_bc_20261103/`: checkpoint generations, stage metadata, reports and source archives.
- `datasets/bc_cycle_20261103/`: published compressed shards, manifests and source cursors.

The data producer, BC controller and follow-ups are independent remote process
groups, reparented to the Colab VM's init process. They do not rely on the local PC
remaining powered on. The kernel monitors actual process handles and published
status. This does not bypass Colab runtime lifetime or quota limits.

After a VM loss, mount Drive in the new runtime, restore the matching source and
frozen dataset manifests, verify checkpoint and shard hashes, and reconstruct the
controller's jobs/state paths before resuming. Preserve optimizer, RNG, epoch sample
indices, split registry and source cursors. Simulator episodes restart after a VM
restart. Do not reuse an expired runtime's PIDs or start duplicate supervisors based
only on a disconnected client. The `colab_v2_*` one-off launchers retained in this
repository document earlier sessions; the current entrypoints are `run_all_bc.py`,
`followup_cycle_stage.py`, `evaluate_cycle_stage.py` and the gated `run_v2_handoff.py`.

Credentials, raw datasets, checkpoints, logs and replay artifacts are deliberately
excluded from Git. Historical documents may reference local/Drive evidence paths
that are not included in a clone.

## Pre-shutdown verification - 2026-11-04T00:47:07+08:00

- BC Stage 2: optimizer step **3008**, 37,632 sampled rows covered in this epoch.
- Drive checkpoint: step **2944**, 1,476,825,173 bytes; independent SHA verification: **True**.
- Checkpoint SHA-256: `561f638c4d687dc1727d2731e988066efc2693326437859acec55fbc1ae2a906`.
- Cycle-aligned producer: **7,486** reconstructed games, **251,725** valid play labels; phase `building`. These are producer totals, separate from frozen Stage 2 training labels.
- Trainer, controller, producer and follow-ups were live; remote supervisors were reparented to VM init, with independent sessions.
- Exact publication source: 113 Linux tests passed, Ruff passed, Python compilation passed, shell syntax passed; no token-pattern matches in candidate files or 266 historical Git blobs.
- Continue BC and the queued calibration/evaluation flow. PPO is **not started**, and learning-strength gates remain unpassed.
