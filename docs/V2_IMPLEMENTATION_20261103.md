# v2 implementation and runtime evidence — 2026-11-03

## Scope and status

`plan.md` §14.3 remains the acceptance scope. The model, data, reward and persistent
PPO changes are implemented and have targeted acceptance evidence below. Fresh
production BC is running on Colab L4. Formal production critic warmup and PPO are
pending BC evaluation; tiny acceptance runs do not establish production strength.

## Requirement map

| Requirement | Implementation | Verification |
|---|---|---|
| Four static-plus-dynamic entity queries; four pooling ablations; early deck context | `models/policy.py`, representation version 2 | `tests/test_representation_v2.py`, real production CUDA probe |
| Spatial tile queries, query residual, FP32 head, legal Gaussian labels | `models/policy.py`, `training/bc.py` | `tests/test_tile_head.py`, real production CUDA optimizer steps |
| Discrete entity IDs, buff embeddings, 16 flag bits, shared card embeddings | `models/policy.py` | Representation tests cover discrete identities, bit sensitivity and header card groups |
| Prefix cutoff, replay deduplication, ability sampling, prior correction | `data/replay_supervision.py`, `bc_sampling.py`, `bc_snapshot.py` | Data/sampling tests; published prefix dataset and immutable snapshot |
| Independent critic with frozen actor warmup and fixed rollout targets | `training/separate.py`, `episode_trace.py`, `selfplay_v2.py` | Separate-critic tests; persistent complete-game acceptance; BC-to-PPO handoff |
| Correct discounted potential shaping, terminal boundary, overflow option | `training/rewards.py`, v2 runner | `tests/test_reward_shaping.py`; persistent rollout telemetry |
| KL over accumulated batches, persistent episodes, both on-policy seats | `training/selfplay_v2.py`, Colab wrapper | 32-update acceptance with 31 continuous update boundaries, complete games and real actor/critic steps |
| Incremental entity encoding with bounded frame cache | `models/frame_cache.py`, `training/inference.py` | Frame cache equivalence and invalidation tests; production CUDA comparison |
| Sparse opponent admission, bounded pool and deck-conditioned EMA | `opponents/league.py`, `admission.py` | Governance tests; complete paired-game admission and resumed update |
| Verified Drive persistence and resume | `training/persistence.py`, `scripts/validate_v2_resume.py` | BC mid-epoch model/optimizer bitwise equality; Drive PPO update 32 restored into update 33 |

Paths in the map are relative to `src/cr_rl/` unless prefixed with `tests/` or
`scripts/`. The full suite passed 72 tests in 8.22 seconds before the live-loss
calibration gate and detached monitor were added. Those latest edits have not
been rerun through the suite.

## Actual acceptance artifacts

Colab root: `/content/cr-rl-lab/artifacts/`. Corresponding reports and checkpoints
are backed up under `MyDrive/cr_rl/runs/`.

- `v2_final_acceptance_20261103/status.json`: completed and passed. Includes
  production representation probe and batch tuning.
- `v2_persistent_acceptance_20261103/report.json`: 32 updates, two terminal games,
  20 actor steps, 33 critic steps, 31 persistent update boundaries, zero faults.
- `v2_resume_acceptance_20261103_r2/report.json`: passed; BC model and optimizer
  bitwise equality after resume, baseline preservation, exact actor handoff,
  actual Drive restoration and candidate complete-game gate.
- Production model: 121,179,142 parameters. Probe used 128-frame windows and
  executed forward/backward/optimizer steps; no production PPO learning claim.

The independent critic owns its parameters and optimizer. `old_value`, old joint
log probabilities, advantages and return targets are fixed during each PPO update.
Actor KL stopping does not stop the critic optimizer. Simulator episode state is
continuous inside a process, but a process restart resets simulator episodes;
checkpoint recovery is not advertised as an exact simulator-state restoration.

## Formal BC configuration and data

Run: `121m_v2_bc_20261103`, W&B run ID `d5230948`.

- Source: `MyDrive/cr_rl/datasets/bc_prefix_20261103`.
- Snapshot: 5,429 prefix-safe replays, 887,872 rows, 340 compressed shards.
- Split: 3,710 training replays, 985 validation replays; another 734 replays
  excluded by held-out deck rules. Training has 10,205 card plays and 20 ability
  labels. Ability coverage is still limited and must be reported as such.
- All training action labels included each epoch; WAIT and ability strata sampled
  with importance correction. Correction uses measured class priors, not a fixed
  `+3.79` WAIT logit. Do not apply a second inference-time prior correction.
- Fresh representation version 2, dynamic four-query pooling, no old BC parent,
  zero inherited optimizer steps, eight planned epochs, 128-frame history.
- Batch 16, accumulation 6, effective batch 96; fused optimizer; automatic SDPA;
  backbone LR `1e-4`, tile LR `3e-4`, Gaussian sigma 1.0.
- Natural validation: 4,096 samples. Balanced conditional diagnostics: 1,024.
  Evaluate hard target placement NLL separately from the soft training objective.
- Checkpoint every 128 optimizer steps and at epoch boundaries, with verified
  Drive copies. First publication was independently read and SHA-256 verified
  from Drive at optimizer step 128: `bc_latest.pt`, 1,458,242,389 bytes,
  SHA-256 `b417c92dfbcabb3462969ff9dfad5b0d458af054fb58bdfeaedf90cb808dad22`.
  Proof: `first_checkpoint_verification.json` in the run directory on Colab/Drive.

Drive-to-Colab hard links failed with `EXDEV` before any optimization. The snapshot
writer now falls back only for `EXDEV` to a verified copy; other I/O errors still
fail. Cross-filesystem regression tests passed and the real snapshot was created.
This failure is retained in the append-only supervisor log; current status and
the actual worker process determine whether the replacement run is running.

## Performance and result boundaries

The L4 tuner measured 83.3 samples/s with ordinary entity counts; the 200-entity
stress case measured 13.3 samples/s and 19.26 GiB peak allocation. Forced Flash
Attention was unavailable for the current attention mask, so the formal run uses
automatic kernel selection. A warmed interval of the actual BC run, optimizer
steps 32 through 64, processed 3,072 samples in about 63.5 seconds (~48.4 samples/s).
The initial baseline evaluation and shard warming are excluded from that interval.

Frame caching demonstrably reduces repeated entity encoding, but a speedup of
5–10× has **not** been established. On the short, mostly empty prefix probe,
cached inference was slower; BF16 cached/full joint log probabilities differed
by up to 0.00135. FP32 equivalence tests pass at their documented tolerance.

At optimizer step 64 the batch loss was ~1.186, versus ~2.503 at step 16. This is
training progress only; the completed-epoch held-out comparison is reported below.
Natural initialization global loss was 3.1522; balanced placement NLL was 5.4867,
conditional slot accuracy 22.83%, and conditional tile L1 distance 13.09 cells.
These are the fixed baselines for the new architecture, not old-v1 comparisons.

### Completed epochs 1–2 (474 optimizer steps at last epoch summary)

All 10,225 unique training action labels were included. The run advanced into
epoch 2 after evaluating and saving epoch 1.

| Metric | Initialization | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Balanced-set conditional slot accuracy | 22.83% | 42.72% | 42.52% |
| Balanced-set conditional tile exact accuracy | 0.79% | 11.22% | 11.22% |
| Balanced-set tile distance within 2 cells | 5.51% | 25.79% | 25.79% |
| Balanced-set conditional tile NLL | 5.4867 | 5.2210 | 4.3222 |
| Balanced-set mean tile L1 distance | 13.0945 | 12.8839 | 12.8780 |
| Balanced-set global loss | 2.8562 | 3.0725 | 4.1403 |
| Natural-set global loss | 3.1522 | 1.1791 | 0.9785 |
| Natural-set mean policy non-WAIT probability | 69.63% | 7.10% | 0.415% |
| Natural-set non-WAIT probability on action rows | 90.10% | 11.93% | 0.704% |

Natural label non-WAIT frequency is 1.76%. Epoch 1 substantially overpredicts
actions (7.10% average); epoch 2 swings to underprediction (0.415%). On epoch 2
action rows the mean is only 0.704%, versus 0.410% on WAIT rows. This is a real
WAIT-collapse warning: natural global loss keeps falling because WAIT dominates,
while balanced global loss worsens from 3.07 to 4.14. Tile NLL improves sharply,
but exact and within-two-cell rates plateau and mean placement distance barely
changes. Hierarchical argmax play recall remains zero for both epochs; rollout
samples the policy, so the metric is not itself the play rate. Full-action exact
accuracy is likewise dominated by WAIT labels. The policy is **not ready for a
PPO handoff based on natural loss alone**.

v2 checkpoint selection ranks eligible completed epochs by natural validation
`global_loss`, requiring improvement over initialization in both natural loss
and balanced placement NLL, plus nonzero conditional slot accuracy. It also
requires mean predicted non-WAIT probability between 0.5× and 2× the held-out
label frequency, and higher non-WAIT probability on action rows than on WAIT
rows. This prevents a falling majority-class loss from selecting an almost
always-WAIT policy. The last checkpoint remains available for resume. Selection
is an imitation gate; closed-loop paired games are still required before
claiming better play.

## Remaining work

### Validation cadence update

Training losses are recorded every optimizer step. The original validation series
ran once per epoch (~237 optimizer steps), which was too coarse for monitoring.
`train_bc.py` now supports `--validation-every-steps 32`: fixed 256-row balanced
and 512-row natural monitoring cohorts are prepared in CPU memory, evaluated
every 32 optimizer steps, and logged as `bc_quick/*` and `bc_quick_natural/*`.
Their axes use `bc/optimizer_steps`. Full epoch validation remains separate, now
also using the optimizer-step axis. Small-cohort results are monitoring signals;
checkpoint selection continues to use the complete epoch validation results.

Real-data tiny resume acceptance compared monitoring-on/off runs: final model
and optimizer tensors were bitwise equal, initialization baselines were retained,
and validation points occurred at each requested optimizer step. Proof:
`v2_step_validation_acceptance_20261103/report.json` and the formal run's Drive
`step_validation/acceptance.json`.

The concurrent acceptance test exposed a shared decompression-cache race and
interrupted the original BC worker. Cache defaults now belong to each output
directory; the formal run uses its own `.bc_arrays`. Training was restored from
the verified step-1185 checkpoint with model/optimizer state and the same W&B run
ID; the pre-failure log's last recorded optimizer step was 1232. The replacement
worker's initial PID is 47085, supervised by recovery PID 47086, with handoff
suffix `_handoff_stepval`. `monitor/step_validation_switch.json` records the
actual resume command and checkpoint. The previously recorded controller PIDs
below describe the earlier workflow and are superseded by this recovery.

The first production quick-validation point at optimizer step 1216 was confirmed
in both W&B history and `Drive/bc/bc.quick.jsonl`. Its two cohorts took 3.72 seconds
to evaluate in total. The fixed natural monitoring subset has 16 action labels
out of 512 rows; its sample frequency differs from the larger natural validation
set, so monitoring and full-validation series remain separately named.

`scripts/run_v2_handoff.py --bc-run 121m_v2_bc_20261103 --ppo-updates 6`
is running on Colab as a detached controller. The first controller was stopped
while waiting, before it could launch PPO, because it had loaded the earlier
natural-loss-only selector. Its replacement uses the action-calibration gate
and is waiting for BC completion. A separate `monitor_v2_bc.py` records rolling
training loss, validation metrics, action-rate calibration and Drive checkpoint
freshness in `loss_health.json` and its JSONL history. Warnings do not fabricate
progress or stop BC.

After successful BC completion, the controller selects a qualifying epoch,
evaluates four paired-seat games against a seeded untrained v2 baseline, then
launches one persistent production PPO process for six updates. The critic
warmup requires two rounds and four completed games before actor updates. After
the requested updates it checks actual actor/critic steps, complete games,
episode continuity, Drive file hashes, and four PPO-versus-BC evaluation games.
These comparisons check numerical/legal operation and record outcomes, not a
statistical strength threshold. Any failed stage is recorded and stops the
handoff. Its source archive is
`121m_v2_bc_20261103_handoff_health/pipeline_source.tar.gz`.

1. Compare completed-epoch conditional placement, slot prediction, natural NLL
   and false-action probabilities against the fixed initialization baseline.
2. Select a qualifying epoch; evaluate complete multi-deck games, then run the
   production independent critic warmup and persistent dense-reward PPO.
3. Keep §14.3's full BC → critic → PPO training item open until those stages
   actually run and their checkpoints, rewards and update metrics are verified.
