# Four-checkpoint placement warmup → joint BC

The full notebook now requests four regular saves (512 training seats each),
then unfreezes entity encoder, temporal core, card/WAIT heads and learned placement
decoder. All active BC parameters use LR 1e-5. Existing placement Adam moments,
GradScaler, RNG, train/validation split and exact data cursor are preserved.
Newly unfrozen parameters obtain Adam moments lazily on their first gradient.
The phase and ordered optimizer groups are checkpointed for exact resume.

The live user runtime had advanced beyond four saves while this change was
prepared. Handoff uses its latest complete checkpoint rather than replaying data.
The original BC source checkpoint, stored location initialization and trained
decoder all remain intact. No second GPU is allocated.

BC now evaluates the labelled hand slot's placement head only: attention queries
2304 → 576 per placement-labelled frame, CNN passes four → one. Card/WAIT logits
and all-frame temporal state advancement are unchanged. Validation and inference
continue evaluating all four hand slots. CPU tests compare full loss and every
parameter gradient; monitoring exact-action accuracy during training uses the
selective placement representation and should not substitute for validation.

L4 comparison on the same checkpoint and eight 32-frame sequences, full-model
forward/backward FP16 (256 frames, warmup excluded, no optimizer update):

| Path | Median seconds | Peak allocated GiB |
| --- | ---: | ---: |
| Four hand slots | 0.20250 | 3.0868 |
| Labelled slot | 0.18733 | 2.9937 |

This microbenchmark measures a 1.081x speedup (8.1% greater throughput). It excludes
Drive checkpoint publication, data preparation and validation. It does not mean
joint training is faster than the previous frozen-trunk phase.

Final CPU suite: 172 passed, one CUDA-dependent test skipped; actual L4 FP16
forward/backward also ran successfully. Numerical losses were 4.63004504 and
4.63004513 in the comparison. Live progress and transfer receipts are saved with
the run as joint_handoff_receipt.json and joint_gt_benchmark.json.

The existing browser's monitor and original W&B identity are reattached after
handoff. Preparation logs may contain ordinary stderr lines (including the old
controller's expected termination traceback); the preparation reader now skips
non-JSON text. The committed metric JSONL reader retains strict validation.
