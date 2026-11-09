# BC data collection and PPO cold-start plan

## Finding from the latest branch

The current pipeline is reproducible, but its supervision is only an
approximation:

`IL_Replay/Cochon event tables -> crforge 16.635.3 open-loop reconstruction -> BC -> PPO`

The source tables contain card-play events and coordinates, not the original
arena state, hidden hand, exact simulator version, or the client's legal-action
mask. A missed command can therefore make every later reconstructed frame
wrong. The importer correctly masks omitted windows and can cut at the first
omission, but broad mode currently accepts a replay after one usable command
(`--min-plays 1`). Those one-command prefixes are valid for pipeline smoke tests
but are unsafe as the main PPO warm start.

The latest frozen BC evidence is consistent with a cold-start problem rather
than a demonstrated capacity limit: conditional slot accuracy is about 38%,
tile-within-two is about 18%, and the natural validation set has weak action
separation (the repaired checkpoint's greedy play recall is 0 on the small
natural sample). The PPO pilot improved behaviour such as waiting, but did not
pass a strength gate. More PPO updates on the same BC distribution are not a
substitute for better state/action supervision.

The requested branch `codex/training-game-budget` is five commits ahead of
`main` (latest `81f9374`). Its BC work adds a shared replay registry across
source shards and documents KataCR and TV Royale as visual-data candidates.
The registry change is now applied to the local rebuild entry point. The visual
datasets remain audit candidates because they expose image-derived fields and
screen coordinates rather than the pinned 7024-int simulator state and legal
mask; the branch has not trained BC on either candidate.

The branch also adds PPO-side controls (`894857e`, `ca56c4f`): Actor updates
can use only rows with a genuine action choice, card and tile KL budgets are
separated, and a run can stop after a checkpointed number of complete learner
games. These address PPO sample-use and early-KL-stop risks; they do not repair
state/action mismatch in reconstructed BC data. Its 64-game BC comparison
(33--31) is inconclusive and is not evidence that either BC or PPO is strong.

## Dataset tiers

Every imported replay must be assigned a quality tier in metadata. The tier is
computed after native labelling, from the retained prefix rather than the raw
event count:

| Tier | Admission rule | Use |
| --- | --- | --- |
| A | both seats, complete 250 ms history grid, at least 60 s retained horizon, at least 8 legal card/ability labels, no collision, and no unbounded suffix | Main BC train/validation |
| B | at least 30 s retained horizon and at least 4 legal labels, with a bounded prefix cutoff | Train with a 0.25--0.5 sample weight; never use as the only validation evidence |
| C | anything shorter, one-seat, collision-heavy, or with fewer than 4 labels | Audit and coverage accounting only; exclude from PPO initialization |

The thresholds are a starting protocol, not hidden labels. They must be recorded
in the manifest and changed only with a new dataset version. A replay may still
be preserved in the archive when it is C; it simply cannot silently contribute
the same weight as a long, two-seat prefix.

## Collection protocol

1. **Source snapshot.** Pin the upstream revision, source file SHA-256, parser
   version, simulator commit, game tables, level normalization, and mode. Keep
   raw source files immutable in Drive. Do not call the number of candidates the
   number of training games.
2. **Canonical identity.** Compute the canonical game signature before shard
   publication and use one registry across IL, Cochon, and every resumed shard.
   `freeze_datasets(..., deduplicate=True)` remains a final safety net; it is not
   a replacement for a shared import registry.
3. **Reconstruction.** Run opening-cycle inference only when its candidate is
   unique and covers the declared prefix. Otherwise mark the deal as seeded in
   provenance. Stop at the first omitted command (or the declared bounded
   omission budget), mask its decision window and retain the prefix only.
4. **Native audit.** For each retained prefix, verify legal unmasked labels,
   duplicate `(replay,tick,seat)` rows, both-seat coverage, contiguous decision
   ticks, and the retained horizon. Save rejection counts and tier counts next
   to the manifest.
5. **Split.** Split by canonical game identity and deck/form combination before
   sampling. Keep all seats and all temporal windows from a game in one split.
   Reserve entire deck combinations for a held-out set. Do not use a random row
   split to claim generalisation.
6. **Sampling.** Train on all A labels once per epoch; sample B labels with the
   recorded quality weight; sample WAIT/play/ability strata separately. Keep a
   natural-prior validation set and a balanced diagnostic set. Report early,
   middle, late, and overtime label coverage separately.

## BC stages before PPO

The next BC run should be a new immutable snapshot, not a continuation that
silently mixes tiers:

1. **Action warm-up:** A-tier data only, WAIT/card/ability head, one or two
   epochs. Check legality, slot accuracy, action calibration, and action-window
   recall.
2. **Placement stage:** add the tile head and legal-tile Gaussian targets. Track
   tile-within-two, mean L1, and per-card/per-lane breakdowns; do not select on
   global loss alone.
3. **Timing calibration:** use balanced sampling for gradient density but apply
   the natural prior through importance weights or a recorded WAIT logit offset.
   Validate sampled closed-loop play counts (target 18--28 successful plays in a
   normal three-minute game) and false actions per WAIT second.
4. **Promotion gate:** require improvement over the frozen BC baseline on a
   larger held-out set, no degradation in natural action calibration, and a
   multi-deck, role-swapped simulator check. A BC loss decrease alone is not a
   PPO gate. If any component fails, keep PPO disabled and retain the candidate
   only for diagnosis.

## Real client trajectory track

The replay reconstruction track is useful for bootstrapping and simulator
regression, but it cannot close the state-fidelity gap. In parallel, collect a
small verified in-memory stream from the user's Android emulator:

- read-only snapshots of header, entities, towers, elixir, hand/deck and public
  opponent cards;
- ADB touch/action events with a monotonic timestamp and selected card/tile;
- a 250 ms decision boundary with the preceding state snapshot, legal mask
  (when available), and the executed result;
- raw JSONL/Parquet plus parser/version/device/game-build provenance; never
  overwrite raw captures during conversion.

Start with 50--100 user-played games on one ordinary deck. Align action events
to the nearest decision boundary, reject ambiguous or missing alignments, and
use the verified subset for validation and BC fine-tuning. The first goal is not
to replace the broad corpus; it is to measure how much the reconstructed states
change action/placement metrics and to provide a trustworthy cold-start anchor.

## PPO handoff

Until the A-tier BC promotion gate and the real-client/simulator state audit are
complete, do not increase PPO scale or add more reward terms. Once they pass,
export the actor, warm an independent critic on complete games, then run a small
league with fixed BC and frozen historical opponents. Compare against the same
BC checkpoint with role-swapped deck pairs. Keep the previous stable policy and
all failed candidates; never promote a checkpoint merely because it has more
learner samples or a higher rolling self-play win rate.

## Current blockers

- The published broad corpus is a bounded supplied subset, not the full upstream
  corpus.
- Import/build does not yet attach the A/B/C quality tier or quality weights to
  each replay.
- The current PPO pilot was allowed from a BC checkpoint that had not passed the
  formal BC promotion gate.
- No verified official-client memory/action capture has been converted into the
  training schema.

These blockers are the reason the next run should be a quality-tiered BC run,
followed by a gated PPO handoff, rather than a larger continuation of the
current PPO pilot.
