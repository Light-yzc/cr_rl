# BC opening-cycle repair and resumed training ? 2026-11-03

The previous multi-deck evaluation established that the full-elixir stall was
absent in 12 tested games, but did not establish stronger play: step 896 went
3?3 against step 444 and 2?4 against early step 237. It did not justify treating
the first two-game 2?0 result as a general strength improvement.

## Reconstruction error and repair

`import_il_replay.py` previously reset crforge with a synthetic random seed and
used its opening hand directly. The source event tables do not provide a verified
opening hand. Many recorded human cards therefore were absent from that hand,
causing early omissions and prefix truncation even before combat divergence.

The optional `--infer-opening-cycle` implementation enumerates ordinary opening
hand sets and queue orders, choosing a plausible deal that explains a prefix of
recorded card identities (at least eight events when available, at most 32 used
for inference). It preserves uncertainty counts in replay metadata. This is a
latent-state reconstruction, **not recovery of the original human hand**.

The adapter reads the pinned native shuffle permutation, inversely permutes the
input deck and evolution/hero slot flags, and resets on the same seed. It checks
that the resulting native deal matches the inferred card order. It never changes
an in-progress battle's hand. Generated native replay setup therefore contains
the actual runtime deck and can be read by the existing labeller/viewer. The
original source card mapping and canonical game identity remain unchanged.

Forced/omitted opening-card rules and source sequences that cannot satisfy the
minimum ordinary-cycle prefix fall back to the seeded deal. Champion cycle
exceptions, levels, unit mechanics and missing commands are not claimed fixed.
The third-omission cutoff and neighborhood supervision masks remain enabled.

## Real-source experiments

| Cohort | Matched games | Seeded accepted commands | Cycle-aligned accepted commands |
| --- | ---: | ---: | ---: |
| Cochon pilot intersection | 7 | 73 | 390 |
| Independent IL_Replay cohort | 32 | 192 | 1,190 |
| First published production-source intersection | 125 | 1,429 | 5,613 |

The two Cochon branches each accepted eight games, but only seven identities
were shared; the table deliberately compares that intersection. In IL_Replay,
the baseline was the existing same-source/same-seed reconstruction for the exact
32 selected records. Games without an omission cutoff increased from 1 to 6.
One Cochon record expanded from three commands and a 23.5-second cutoff to all
45 source commands. These are data-availability results, not policy-strength or
original-state-fidelity results.

## Training continuation and storage

The replacement producer is `artifacts/bc_cycle_20261103`, with versioned shards
and metadata backed up to `MyDrive/cr_rl/datasets/bc_cycle_20261103`. Raw source
parquets were verified while copying from their existing Drive location into
Colab scratch; no raw dataset or model weights were downloaded to the PC.

The prior producer and controller were stopped after the new published source
passed the matched-data audit. Its latest valid source snapshot was preserved
as fallback data. BC continued from the verified Drive checkpoint at optimizer
step **1408**, including optimizer state, in `121m_cycle_bc_20261103`.

The first frozen stage contains 8,928 supervised game identities (13,511 raw
records before duplicate exclusion), 4,169,072 raw frame rows, and **52,176 valid
training card-play labels plus 485 ability labels**. These are snapshot counts;
the full source producer is still running. Same-game splits are preserved by the
existing split registry, and cycle-aligned records take precedence over old ones.

W&B: https://wandb.ai/weidemingdeguge-test/cr-rl/runs/0198t7vi

Full-source processing, a final full-data BC pass, complete-game evaluation,
independent critic warmup and PPO remain required. No new-model strength claim
has been made before those checks.

## Metrics, selection and throughput

Validation now reports separate balanced ACT/WAIT NLL, conditional card NLL,
hard placement NLL and their conditional validation score (metric version 4).
A real-data tiny Linux run completed an optimizer step, quick validation and
checkpoint save; it verified that the score equals the recorded components.

Conditional checkpoint selection uses component improvement, placement
improvement and an ACT-versus-WAIT score gap. It does not impose the former
natural-prior action-rate gate on a deliberately balanced timing objective.
Natural-distribution metrics remain available for diagnosis. The staged handoff
resolves the actual final checkpoint and stage-specific Drive epoch backups.
These changes do not bypass the complete-game checks or critic warmup.

The entire Linux unit suite passed **86 tests**, covering the new cycle solver,
deck/slot-flag permutation, conditional selection, staged checkpoint resolution
and scratch-cache space reservation, alongside prior v2 tests.

The new stage's expanded observation/mask arrays were measured at approximately
118 GiB. The former 8 GiB limit was a **Colab SSD decompression cache**, not GPU
memory or a RAM allocation. A checkpoint-boundary tuning worker has been started
to resume with an upper bound of 128 GiB and preserve 16 GiB of free scratch space.
It evicts only disposable decompressed arrays, keeping source NPZs intact. This
tuning was still awaiting its first saved checkpoint when this note was written;
no throughput improvement is claimed until applied and measured. Large caches
remain on Colab scratch and are not copied to Drive or the PC.
