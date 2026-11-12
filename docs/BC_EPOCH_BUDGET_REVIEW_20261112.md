# BC epoch budget review — 2026-11-12

Recommendation: consider 24 epochs as a conditional total upper bound, with
checks at 8, 12, 16, 20 and 24. The current evidence does not justify committing
to 36 epochs. No training budget/configuration was changed during this review.

W&B run: https://wandb.ai/weidemingdeguge-test/cr-rl/runs/014fcb2dcb75

At inspection the run was near the end of epoch 5, optimizer step 12,759,
constant LR 1e-5. Sampled epoch train losses declined approximately
5.74 → 5.58 → 5.43 → 5.23 (epochs 2–5, epoch 5 incomplete). This establishes
continued training-set fitting; it does not establish continued generalization.

The four independent epoch validation losses are 5.8695, 5.3346, 5.7834 and
5.5278. Epochs 1–4 were replayed into W&B once; duplicates were removed by
(validation epoch, optimizer step). Every validation draws 16 different games
from a 426-game pool. Its noisy minimum should not determine a long budget or
model selection on its own. Use a frozen 64-game subset for paired checkpoint
comparison; keep all data outside train and do not fit priors/calibration on it.

Continue beyond a checkpoint only if the fixed-validation placement loss,
card/timing metrics and relevant action diagnostics support improvement.
If fixed validation deteriorates while train loss decreases, select the best
validated checkpoint and reassess generalization rather than using all 24.
Only consider 36 if epochs 20–24 still improve fixed validation meaningfully.

Based on the earliest original timestamps of validation events (not replay
timestamps), an epoch has taken about 2.8 L4-hours including validation/storage.
Relative to the current total target of eight epochs, 24 adds approximately
45 hours, and 36 adds approximately 78 hours, assuming unchanged throughput.
These are runtime estimates, not a promise of uninterrupted Colab availability.
There are 2,556 group updates per epoch: approximately 61,344 total at epoch24
and 92,016 at epoch36. W&B event Step is not the optimizer update count.
