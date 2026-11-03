# Broad human-replay BC data expansion

The current training architecture remains simulator-reconstructed observations ->
BC -> PPO. This expansion changes data admission, not the actor architecture.

The initial 381-record round used restrictive ranked/level/coverage filters. Broad
admission is now the default for `import_il_replay.py` and `build_bc_shards.py`:

- Accept every source mode and source card level that fits the 1v1 interface.
- Normalize reconstruction to level 11; retain original card levels and mode in
  replay metadata. Observations remain approximate, including special-mode rules.
- Keep a replay if at least one command can be reconstructed. There is no minimum
  whole-game coverage and no whole-game collision rejection.
- Exclude omitted-command decision windows from supervision using flag 65536.
  Otherwise a human play that could not execute would incorrectly become WAIT.
  Keep all observation frames to preserve history spacing.
- Preserve the existing native labeller flags and legal-action checks.
- Add no cross-source deduplication. Map six missing public card names to pinned
  internal table rows: Executioner/AxeMan, Magic Archer/EliteArcher, Flying
  Machine/DartBarrell, Rune Giant/GiantBuffer, Void/DarkMagic and Spirit
  Empress/MergeMaiden. These rows exist in the pinned backend tables.

The previous subset can be reproduced with `build_bc_shards.py --strict-subset`,
or importer flags `--ranked-only --source-level 16 --min-coverage .9 --min-plays 20
--reject-collisions`. An existing strict build cannot be resumed as a broad build;
use a new output directory.

The source tables currently in Colab contain 10,000 IL_Replay candidate records
and 1,592 cleaned Cochon123 records. This is not the complete upstream corpus.
An audit of the 10,000 IL records found 8,715 Ranked and 1,285 other-mode records;
both sides have one player in every record. Source card levels range from 6 to 16.
No additional mode filtering is applied.

Use `build_bc_shards.py --target 0` to scan all supplied candidates. Compressed
16-game shards, metadata, source files and reconstructed replays are checksum
verified into `MyDrive/cr_rl/datasets/broad_bc_20261103/`. Expanded data remains in
Colab. Each published manifest is resumable; local raw downloads are not required.

The first integration smoke imported 16 IL records with coverage .33-.89 and
source levels 14-16 that the previous coverage rule would reject. It kept 21,402
history rows and recorded 191 omitted commands, whose decision windows are
excluded from loss. A unit test covers duplicate omissions, seat separation,
five-tick window alignment and preservation of native flags.

The original sampled BC round was retired at its verified epoch-8 checkpoint.
[ALL_BC_SWITCH_20261103.md](ALL_BC_SWITCH_20261103.md) records the new full-coverage
training stages, streaming cache, exact within-epoch recovery and PPO readiness
criteria. New data enters immutable snapshots at epoch checkpoint boundaries.
The real monitoring cell supervises data and training for up to 12 hours and
releases the runtime only after completion and Drive flush. Runtime lifetime is
not guaranteed. Full source conversion and final full-corpus training are ongoing.
