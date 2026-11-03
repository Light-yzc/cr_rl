# Placement-head fix and matched BC trial

The placement decoder now supports a query residual and FP32 head arithmetic.
The existing tile queries, card features and history context are preserved in the
scoring path through `attention_output + query`. No parameters are added.

Two `ModelConfig` fields record behavior in every checkpoint:

- `tile_query_residual`: default false for legacy checkpoints.
- `tile_head_fp32`: default false; true disables autocast for query construction,
  placement attention and scoring. The BC trainer uses highest FP32 matmul
  precision while retaining BF16 autocast for the backbone.

`train_bc.py --init-from-bc OLD --upgrade-tile-head --carry-optimizer` explicitly
migrates old BC weights. `--resume` inherits saved head options and cannot silently
change them. Config comparisons normalize missing legacy fields to false. The
feature layout and state-dict keys remain architecture v2; head behavior is fully
specified by the saved config.

An optional `--tile-learning-rate 3e-4` creates a separate placement parameter
group while the backbone remains at `--learning-rate 3e-5`. Named optimizer
parameter lists preserve Adam moments and counters across the group split.
Shared card embeddings retain the backbone rate. Exact resume restores both
groups. New head options are also retained by BC-to-PPO initialization, historical
opponent loading and actor-only exports.

Acceptance on real replay data passed: migrate an old BC checkpoint, train all
540 available action labels, resume from step 136, and reach step 270 with
bitwise-identical CPU model weights and optimizer state to uninterrupted training.
The continued optimizer reaches step 272 without a reset; actor export and PPO
initialization retain both head flags. The unit suite has 41 passing tests.

`run_tile_head_trial.py` safely retires the existing BC queue after the next
verified Drive checkpoint and runs two full-model training cases from that exact
parent. Both use the immutable stage-1 dataset, 128-frame histories, batch 8,
gradient accumulation 12, seed 20261102, 8,192 sampled training draws and the same
1,024-row validation selection. Dataset, sample-order and validation-row hashes
must match. The control retains its original head and rate; the fixed case enables
the residual, FP32 placement head and separate placement rate. This evaluates the
combined fix, not each factor independently.

Results are published to
`MyDrive/cr_rl/runs/119m_tile_fix_trial_20261103/comparison.json`.
The original trial's selection file has been written and verified on Drive;
`fixed` is selected and full-data BC is running. New trial invocations select
automatically against matched-data and metric guards, unless manual selection is
explicitly requested. Source conversion proceeds concurrently. The short matched
trial does not replace full action-label coverage in the subsequent pipeline.
Actual results and the new prefix-data path are recorded in
[the audit follow-up](AUDIT_FOLLOWUP_20261103.md).

New validation fields include `card_loss`, `position_loss` and `global_loss`.
`global_loss` aggregates card CE over all validation rows and conditional tile CE
over all validation plays, avoiding dependence on how WAIT-only minibatches are
grouped. The legacy `loss` field remains for chart compatibility. Compare tile L1,
within-two-tiles accuracy and exact tile accuracy alongside the loss components;
these are imitation metrics, not evidence of improved battle win rate.
