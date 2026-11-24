# Nulls private-server policy runner

Selection update (2026-11-16): the runner now defaults to `policy_sample`, drawing
from the legal joint policy, including WAIT, cards/tiles and enabled abilities.
Confidence thresholds and tile temperature apply only to the legacy `confidence`
mode. The spell target guard still vetoes sampled damage spells with no detected
target, so actual execution remains a guarded policy. `proposed_action` and its
probability are logged alongside the final selection; all execution gates and
receipts remain active. This fixes greedy card starvation, not learned tactics.
Use `-SelectionMode confidence` to reproduce the old controller for comparison.

Skill work status (2026-11-16): the user has switched back to a deck without active
skills. Ordinary control is the default. Skill selection/confirmation code has
unit coverage, but the current calibration profile remains `validated=false`:
no live skill confirmation was completed. Do not enable it as if it were verified.
`-EnableAbilities` is explicit opt-in and requires a validated matching profile.
The runner also holds an OS process lock to prevent two controllers from sending
commands concurrently, including when started directly without PowerShell.

Run from PowerShell in `G:\Code\cr_rl`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_nulls_policy.ps1
```

Keep MuMu Android 12 instance 1 open, with Nulls logged in, root/ADB enabled,
900x1600 resolution and serial `127.0.0.1:16416`. The already deployed read-only
helper `/data/local/tmp/crrl-readonly-pages` must remain executable.
The optimized touch helper is already deployed on this instance. To rebuild it
using the existing JDK/d8 without downloads:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_nulls_touch.ps1
```

The runner defaults to a persistent Android InputManager touch transport with a
20ms inter-tap gap. If startup cannot initialize the helper, `auto` logs a fallback
to shell input. A partial failed touch command is never automatically resent.
`--touch-transport shell` selects the old path; `persistent` refuses startup fallback.
Wait for `ready` after `warming_up`, then enter a private-server battle manually.
The runner detects the battle and controls only battle actions, not menus.
Do not manually play cards concurrently: confirmation assumes exclusive control.
Ctrl+C stops the runner. It also stops at battle end, 600 seconds or 100 sends.
Run the same command again for the next battle. It refuses a second controller.

Read-only test (no taps):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_nulls_policy.ps1 -DryRun
```

Logs are in a new `artifacts/nulls_live/play_YYYYMMDD_HHMMSS` directory:

- `console.log`: terminal output.
- `events.jsonl`: structured decisions, sends and confirmations.
- `crash.log`: Python tracebacks and native crash diagnostics, including startup failures.
- `diagnostics/frames_*.jsonl.gz`: compressed memory snapshots, exact model observations,
  legal action IDs and full action probabilities for replay/debugging. The complete
  bounded session is retained in 32MiB uncompressed segments for training export.
- `decision.hand`: current four cards, in slot order.
- `card`, `decoded`: chosen card and board column/row, or WAIT.
- `wait_probability`, `slot_probabilities`: WAIT and each card's total action probability.
- `selected_action_probability`: probability of this exact card plus tile action.
- `tile_probability_given_card`: tile probability conditioned on the chosen card.
- `top_actions`: top five exact actions and probabilities.
- `entropy`: legal joint action entropy in nats; `max_action_probability`: largest joint probability.
- `read_ms`, `inference_ms`, `state_age_ms`, `tick_gap`: latency and missed-frame diagnostics.
- `reason`, `legal_play_tiles`, `legal_before_gate`, `clock_running`, `tower_hp`,
  `pending_slot`: distinguish policy WAIT from mask closure, opening gates, stale
  actions and pending confirmations. Ten seconds of WAIT emits `no_play_warning`.
- `waiting_for_battle` / `clock_stalled`: five-second heartbeat while no frames advance.
- `stopped.reason`: battle end, time limit, send limit, Ctrl+C or exception.
- `manifest.json`, `sampling_rng.pt`, `source.zip`: model identity, initial sampling
  RNG and Python source provenance. See `docs/NULLS_ACTOR_DIAGNOSIS_20261115.md`
  for export commands, label limitations and the Sparky investigation.
- `submitted` only means taps were sent. A `receipt` with `confirmed` requires
  both increased played count and rotation of the expected hand slot.

State older than 400ms is skipped. Screenshots are not policy input.
Abilities remain disabled. Unknown adapter fields and game/simulator differences
remain limitations; interface success does not establish competitive performance.

## Mid-match WAIT regression fixed, 2026-11-15

The raw `clock_running` byte changed to false at tick2651 in
`play_20261115_204948`, although tick/time continued advancing and the battle
had not ended. The old controller incorrectly treated this byte as an opening
gate and erased all play actions. `BattleClockGate` now accepts either a true
raw running flag or strictly increasing tick AND time, after tick8. Explicit
end-of-match evidence always blocks play. An initial unflagged frame must wait
for another observation to demonstrate progress.

Offline replay of 396 saved frames reproduced all model observations exactly
(zero mismatches). It identified 119 incorrectly gated frames; 107 had playable
actions restored. This establishes the mask fix, not how the model would choose
actions after taking a different trajectory. Eighteen client regression tests
passed. Restart an older running controller to load the fix.

Reproduce without GPU, ADB or touches:

```powershell
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/replay_nulls_diagnostics.py artifacts/nulls_live/play_20261115_204948
```

The source of the raw running byte's semantic change is not established; it is
retained in diagnostics but no longer used as an independent veto. Optional
unverified observation fields remain UNKNOWN. They must not be filled with
invented defaults as a remedy for this gate bug.

Suggested ordinary-form first deck, matching `giant_sparky_v1` in
`configs/decks_expansion6.yaml`: Giant, Sparky, Mini P.E.K.K.A, Musketeer,
Minions, Knight, Zap, Arrows. Leave evolution/hero slots unused for the first test.

## Measured interface run, 2026-11-15

`artifacts/nulls_live/run002/events.jsonl`, v3 update697, local RTX4090 FP16,
MuMu training camp: 81 decisions, four sends, four confirmed plays, no timed-out
plays. Excluding the first cold inference, read latency 15–234ms, inference
16–281ms, total state age 62–328ms. The cold first inference took 28.1 seconds
and its action was skipped. Startup warmup was subsequently added. These are
4090 measurements, not L4 measurements. The training opponent had lower tower
levels, so the match result cannot establish model strength.
