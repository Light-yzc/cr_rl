# V100 PPO sampling and replay consistency

The learner's streaming sampler and whole-episode PPO forward used different
batch shapes and different PyTorch attention paths. A mismatch still occurred
after the guard was relaxed to 0.10: update 763 logged 0.06049. The guard is
now 0.002 by default for the V100 launcher.

On four complete trajectories (3,106 frames, checkpoint 763), holding weights
fixed gave these maximum selected-action log-prob differences between a
streaming forward and a training sequence forward:

| Arithmetic | Maximum difference |
| --- | ---: |
| Original mixed precision | 0.018649 |
| FP32 with the eval fast path enabled | 0.026869 |
| FP32 with the same attention path in rollout and training | 0.000066 |

The learner, its NVLink replica and its frozen behavior reference now use
FP32 arithmetic. The independent critic and frozen opponents retain their
mixed-precision configuration. The behavior reference cache checks the
recorded chosen-action log-prob for every trainable frame in full-episode
mode, before any Actor updates and before the cache is compressed to FP16.
The check against the actual gradient-enabled Actor forward is also retained.
The PPO ratio still uses the probabilities recorded at sampling time.

Two isolated 48-lane V100 smoke updates produced 68,750 learner frames.
Maximum initial Actor errors were 0.0000701 and 0.0000119, with no Actor
overflow or KL stop, and one Critic AMP overflow skip. Peak allocated GPU
memory was 7.37 GiB and peak trainer RSS was 4.14 GiB. Smoke throughput was
234–245 learner frames/s; the isolated run did not restore the full historical
opponent pool, so compare production throughput separately.
The resumed production update 764, with three frozen opponent groups, passed
both consistency checks at a maximum of 0.0000861. It processed 34,444 learner
frames at 228.7 frames/s, with zero overflows or KL stops and peak allocation of
7.33/7.87 GiB on logical CUDA devices 0/1.

On this sample, the WAIT gate's mean and maximum absolute output were
0.00226 and 0.00760 logits, and the PPO gradient into it was nonzero. The
card and tile entropy gradient norms were about 0.14% and 0.04% of the
parameter-space PPO gradient norm, respectively. This is one
sample, so it does not establish whether a lower entropy coefficient
improves wins. Entropy coefficients, gate learning rate, focal decks and
the training opponent mix are held constant for this numerical repair.

Use `scripts/diagnose_ppo_consistency.py` with a matching checkpoint and a
trusted replay bundle containing `version`, `segments`, `schema` and
`vocabulary_size`. Segments are complete `PersistentStateSelfplay.collect()`
episodes. Compare `baseline`, `fp32` and `unified`; use `--gradients` for a
parameter-space comparison. The `vs_steps_*` fields compare like-for-like
arithmetic; `vs_rollout_*` on a replay collected before the fix compares
against old FP16 probabilities and is not the new consistency check.
