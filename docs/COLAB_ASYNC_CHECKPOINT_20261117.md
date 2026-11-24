# L4 batching and asynchronous checkpoint writes

## Deployed configuration

The longrun was stopped normally and saved at U940 before this change. It resumed
from that checkpoint after the native simulator tests below, on the single L4
runtime `cr-train-l4-1117`.

| Setting | Previous | Deployed |
| --- | ---: | ---: |
| Environments | 8 | 16 |
| Frame microbatch | 64 | 128 |
| Actor / Critic batch | 2 / 2 | 4 / 4 |
| Rollout / optimization opponent caches | 4 / 4 | 4 / 4 |
| Checkpoint interval, PPO updates | 25 | 5 |
| Local serialization | Synchronous | Asynchronous |
| Drive publication | Asynchronous | Asynchronous |

W&B remains disabled. The AWBC and experimental BC inputs and intervals are
unchanged. The reward mode is still `uniform`.

## Consistency and bounded work

`--async-checkpoint` is opt-in in `train_selfplay.py`; the shared Colab launcher
enables it by default, with a five-update interval. Other callers retain the
synchronous default unless they pass the flag.

The training thread first copies model, optimizer, scaler, RNG and progress state
into an independent CPU snapshot. CPU tensors are copied too, so the worker never
reads mutable Adam counters or live model tensors. Device-to-host staging still
takes a short synchronous interval at the update boundary. Measured staging was
approximately 3.5–5.6 seconds in these tests.

A single bounded writer serializes the checkpoint and actor export to SSD in the
background. Completed files replace their temporary files atomically. Disk errors
propagate to the trainer. The previous latest checkpoint stays intact if writing
fails before commit. The worker has no model or simulator reference.

During the following rollout, completion is polled every 64 steps. The trainer
finalizes retention and freezes immutable file links before handing publication
to the existing bounded Drive worker. Publication uses `--no-backup-hash` and
commits the Drive `latest.json` pointer only after all required files are copied.
It does not add checkpoint-sized hash scans or per-step disk writes. There is at
most one staged checkpoint and one active Drive publication. If either worker
falls behind, bounded backpressure can wait rather than grow an unlimited queue.

Validation and orderly shutdown wait for the required save. A final pending save
is drained before closing the publisher. Abrupt runtime destruction can still
lose data that has not finished uploading.

## Verification

Local tests: **8 passed**, covering frozen state versus subsequent parameter and
optimizer mutation, bounded staging, disk-error propagation, previous-checkpoint
preservation, publication consistency and command construction.

Two native simulator profiles resumed the same saved U940 checkpoint and each
ran two full PPO updates. Both used frame microbatch 128, Actor/Critic batch 4,
opponent caches 4/4, and checkpoint interval 1 to exercise background writing
across an update boundary. Auxiliary BC/AWBC and Drive publication were disabled
for these isolated throughput measurements; existing data was not changed.

| Second-update metric | 8 environments | 16 environments |
| --- | ---: | ---: |
| Learner samples / second | 46.44 | 75.25 |
| Peak allocated CUDA memory, GiB | 10.09 | 10.47 |
| Initial log-probability error | 0.000219 | 0.000497 |
| Actor overflow steps | 0 | 0 |
| Critic overflow steps | 1 | 0 |
| Faulted samples | 0 | 0 |
| Process exit | 0 | 0 |

The 8-environment Critic overflow was skipped by AMP. The selected 16-environment
profile completed its second update without overflow. Actual `nvidia-smi` memory
usage during the profiles was approximately 13.2–13.3 GB; allocated tensor memory
and total driver/reserved memory are different measurements. GPU utilization
reached 100% at observed instants; this is not an average-utilization claim.

The figures above are short throughput tests, not evidence of improved playing
strength. Different environment counts collect different games and sample totals.
The original U940 checkpoint was kept as the formal restart point; profiling
updates were not promoted into the longrun.

## Persistence and recovery

The deployed launch arguments were read back and confirmed, including
`--async-checkpoint --async-publication --no-backup-hash --checkpoint-interval 5`.
Exactly one formal trainer was active at verification (PID 58045).

The first resumed formal update U941 subsequently completed at 55.68 learner
samples/second (including cold-start effects), with zero Actor overflow steps,
one Critic AMP overflow step skipped, and zero faulted samples. Drive's committed
`latest.json` advanced to `generations/update_000941` while that same trainer
continued running. This verifies the local asynchronous save → Drive publication
chain in the formal run. The 75.25 figure above is the warmed second profile
update, not the measured first formal update.

Drive holds the matching local configuration in `preparation/cr_train_config.json`,
the updated shared code in `preparation/crrl_training_update.zip`, and the profile
report in `preparation/async_checkpoint_performance_20261117.json`. Existing thin
notebook bootstrapping reuses those durable files after SSD loss. No new notebook
copy of the training loop is needed.

The original native simulator hang described in
`COLAB_LONGRUN_CACHE_RECOVERY_20261117.md` is not fixed by these changes.
