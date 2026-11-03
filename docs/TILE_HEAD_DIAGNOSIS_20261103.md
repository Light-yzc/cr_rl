# Placement-head diagnosis

This investigation does not change the running BC actor, optimizer or hyperparameters.
`scripts/diagnose_tile_head.py` runs a bounded overfit ablation using the verified
old epoch-8 checkpoint and 16 real legal play labels from its first replay shard.
The full 119M encoder is frozen. All cases use the same frozen features and
886,529 trainable placement-head parameters. There are 6-17 valid entities per
sample; this is not a synthetic empty-arena example.

The existing decoder computes a tile/slot/card/history query, cross-attends to
current entity tokens, and sends only the attention output to the scoring MLP.
The query has no direct residual path into that MLP. The alternative adds the
query to the attention output, without adding parameters.

At initialization the original head's FP32 spatial-logit standard deviation is
about 1.46e-5, and its conditional cross entropy is essentially the legal-uniform
baseline (5.41027). BF16 head arithmetic produces spatial differences of roughly
1.28e-3 instead. This motivates checking numerical precision rather than assuming
that a flat placement loss proves insufficient overall model capacity.

Results after 200 fixed-batch updates at head learning rate 3e-4:

| Placement head | Arithmetic | Cross entropy | Exact tile accuracy | Mean tile L1 |
|---|---|---:|---:|---:|
| Original attention-only | BF16 | 1.76357 | 50.00% | 6.4375 |
| Original attention-only | FP32 | 0.22063 | 93.75% | 0.3125 |
| Query residual | BF16 | 0.93654 | 56.25% | 4.8125 |
| Query residual | FP32 | 0.06535 | 100.00% | 0.0000 |

The same experiment also includes the formal head learning rate, 3e-5. At that
rate the original head reaches loss 5.014 (BF16) or 5.027 (FP32), while the residual
head reaches 5.138 (BF16) or 4.635 (FP32). Thus the residual alone is not a universal
loss improvement at every precision/rate. With BF16 the residual increases the
initial tile-query gradient norm from 0.000334 to 0.012594, about 37.8 times.

The frozen encoder features were generated with the existing BF16 backbone for
every case; FP32 here refers only to subsequent head arithmetic. Head-only
learning rates and loss scaling differ from full-model BC, so 3e-4 is not a
recommendation to raise the entire actor's learning rate tenfold.

This is evidence of a placement readout/optimization/numerical bottleneck. It does
not establish its contribution to the full-corpus plateau, nor prove improved
validation or battle strength. A larger held-out comparison is required before
changing the running architecture. Increasing the full actor to 300M is not the
first remedy supported by these results.

The importer seat-coordinate transforms match the pinned simulator's
`Schema.toSeatX`/`toSeatY`. That rules out an obvious implementation mismatch in
those flips, not every possible raw-replay or reconstructed-state alignment issue.

Full results and the exact diagnostic script are archived under
`MyDrive/cr_rl/runs/119m_all_bc_20261103/diagnostics/tile_head/`.
