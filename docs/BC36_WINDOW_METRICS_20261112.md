# 36-epoch continuation and count-weighted training metrics

User-requested total target: 36 epochs, including epochs already completed.
This is the same full-finetune run, with LR 1e-5, existing head/trunk Adam,
GradScaler, RNG, data split and exact checkpoint cursor. The controller requires
an explicit allow-epoch-extension flag and rejects unrelated contract changes
or horizon reduction. Constant LR does not restart a warmup/decay schedule.

Training logging now emits one window per 32 successful optimizer updates,
usually 256 episode seats with batch eight. Checkpoint, validation and epoch
boundaries flush a shorter window. Overflow events flush immediately. Logging
does not change sampling, effective batch, forward/backward or optimizer updates.

Metrics use their correct denominators:

- slot_accuracy: sum(correct card-slot predictions) / sum(play labels).
- tile_mean_distance: sum(tile distances) / sum(play labels).
- wait_accuracy, act_recall and exact_accuracy: their corresponding WAIT,
  action and valid-row counts.
- No-play windows omit slot_accuracy and tile_mean_distance; they do not
  introduce an artificial zero.
- Losses remain equal-seat averages, matching the existing BC objective.
- grad_norm and group_seconds average each optimizer group once.

The same scalar values remain under train/* for compatibility. New
train_window/* series contain only the corrected aggregates, giving a clean
plot without old per-seat spikes or visualization smoothing. The x axis is
trainer/optimizer_steps. Window play counts and seat/update counts are logged
so the sample size remains visible. Historical per-seat events are retained.

Validation sampling and validation loss/model selection are unchanged by this
logging change. Future fixed-game comparisons are a separate evaluation change.
