# Ordinary and upgraded deck curriculum

The screenshot deck was absent from the existing configured pools. Added
`user_giant_sparky_pump_plain` to `configs/decks_all51.yaml` with these eight cards:
Wizard, DarkPrince, ZapMachine, Elixir Collector, Fireball, Giant, Rage, MiniPekka.
All eight slot flags are zero: ordinary forms, no evolution or hero skills.
Card level is a separate simulator setting; slot flags do not set card levels.

`configs/decks_augmented.yaml` preserves all existing entries and appends supported
form variants. It contains 242 named configurations after augmentation of the
52-entry source. Generation groups matching eight-card sets into families and
adds an ordinary version if missing, then at most three new upgraded candidates
per family: one evolution, one hero, and one evolution plus a different hero.
Existing upgraded configurations are retained.

Only alternate rows explicitly linked by `EvolvedSpells` and enabled in the pinned
16.635.3 tables are eligible. Unused/hidden placeholder rows are excluded. This
does not establish full mechanic parity with the private or official server.

For the screenshot deck, appended variants are:

- `__evo`: evolved Wizard.
- `__hero`: hero Wizard.
- `__evo_hero`: evolved Wizard and hero DarkPrince.

The v3 trainer now honors `deck_sampling.mode: family_balanced`. Both seats draw
independently: first uniformly choose a family, then ordinary/upgraded with
probabilities 0.5/0.5 when both exist, then uniformly choose a configuration in
that category. Families without upgraded candidates use ordinary with probability
one. Thus ordinary configurations retain at least half the sampling mass despite
the additional upgraded configurations. Pools without this metadata retain the
previous uniform deck sampling and RNG consumption.

The dual-V100 v3 launcher defaults to `configs/decks_augmented.yaml`; its existing
resume path passes `--expand-decks`. Previous names/order/cards/forms remain at
the beginning of the pool. Existing checkpoints do not acquire new skills merely
by changing this configuration. No training was started for this edit.

Regenerate and verify from the repository root, using the existing Python/JDK and
the simulator Python package on PYTHONPATH:

```powershell
$env:PYTHONPATH = 'G:/Code/cr_rl/src;G:/Code/cr_rl/.cache/vendor/crforge/python'
$env:JAVA_HOME = 'F:/JDK'
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/build_deck_augmentation.py
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/smoke_deck_forms.py
```

Validation: seven augmentation/expansion unit tests passed; all 242 configurations
reset and ran 120 randomly sampled legal-action decisions in self-play without a
simulator fault (`artifacts/deck_forms/smoke.json`). This is interface acceptance,
not full-match evaluation or exhaustive ability/evolution execution coverage.
The current real-client controller still masks hero abilities off. Use the
ordinary deck when testing that controller; skill-button support is separate.
