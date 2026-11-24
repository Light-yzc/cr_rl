# Real-client actor diagnosis and training collection

## Observed session

Source: `artifacts/nulls_live/play_20261115_210255`.
345 inference frames over 137.8 seconds; 18 submitted actions, 17 confirmed and
one timed out. Sparky stayed in hand throughout; no confirmed Sparky play.
The final controller exit was an unmapped-page EIO, not a policy decision.

Sparky had legal tiles on 11 frames. Seven of these frames were pending another
command, leaving only four opportunities to submit a fresh command. At tick98
the sampled action was Sparky, but the previous Giant command was still pending,
so the controller did not execute it. That Giant command later timed out. A
second Giant request at tick155 was confirmed; after tick171 recorded elixir
never exceeded 56160 (5.616), below Sparky's six-elixir cost.

This is a combination of command/confirmation losses and resource spending,
not evidence that the model cannot recognize Sparky. Do not interpret the eleven
frames as eleven independent chances to execute the card.

## Same-observation recurrent policy comparison

`scripts/analyze_nulls_actor.py` ran all three actors on the entire saved sequence,
including WAIT frames, without clicks or changing any weights:

| Checkpoint | Mean Sparky probability when legal | Mean WAIT probability with any play legal |
| --- | --- | --- |
| PPO697 | 19.15% | 83.95% |
| PPO184 | 2.21% | 95.83% |
| BC5112 | 4.23% | 92.56% |

These are distributions on another policy's fixed trajectory, including pending
command frames. They are not battle scores, free-command-only decisions, or
counterfactual trajectories. The PPO697 rerun matches its recorded Sparky
probabilities. Reverting to PPO184 is not supported as a Sparky fix by this test.

The deployed actor directly samples its distribution; it does not perform online
lookahead. The separate PPO value network supplies training targets/advantages,
not a runtime search over card sequences. Turning on a value output in inference
does not change this actor's logits. Poor resource planning could involve learned
advantages/rewards, coverage and client/simulator observation shift; this session
cannot uniquely identify one of these causes. Training update697 also reported
requested mean card cost 2.62, with 42 of 691 requests costing at least five, which
supports inspecting high-cost coverage but does not prove critic failure.

## Reader repair

`UnmappedGamePageError` distinguishes EIO/EFAULT page responses from broken
transport framing and permissions. A failed snapshot is discarded without
inference or actions; the next attempt starts a fresh page cache. Five consecutive
failures stop explicitly. All batch responses are consumed even when one page
fails, keeping the next request framed correctly. Component-array traversal now
only applies to the verified character/tower layouts, not projectiles/areas.
This prevents following an unverified object layout; the precise origin of the
reported invalid address is not proven. No relaxation of freshness limits.

## Collection on the next controller launch

- Complete compressed frame sequence retained for each bounded session (maximum
  600 seconds), split into 32MiB uncompressed segments. No within-session deletion.
- Raw parsed snapshots, exact observations/masks, all action probabilities, chosen
  log-probability, decision IDs and read/inference timestamps.
- Submissions and confirmations joined by decision ID. Pending/rejected/stale
  actions must not be used as successful action labels.
- `manifest.json`: checkpoint SHA256/metadata, arguments, package versions,
  completeness and unknown outcome. `sampling_rng.pt`: initial sampling RNG states.
- `source.zip`: Python source snapshot. The 743MB checkpoint is referenced by hash
  rather than duplicated; preserve that checkpoint for reproducing inference.
- No claim of an exact server replay: observations are partial/non-atomic, some
  internal fields remain UNKNOWN, and the actual server cannot be deterministically
  reset to this observation stream. No fabricated terminal reward or outcome.

Export a collected session:

```powershell
$env:PYTHONPATH='G:/Code/cr_rl/src'
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/export_nulls_training.py artifacts/nulls_live/play_20261115_210255
```

This session exported 345 frames: 17 confirmed-play labels, 121 policy-WAIT
labels, 207 unlabeled frames. All states remain in sequence for recurrent replay.
The labels imitate the collecting policy, not an expert; do not blindly fine-tune
on them. Rewards/returns remain null, `ppo_eligible=false`, and outcome is unknown.
Old logs lack the new provenance manifest; those metadata cannot be reconstructed
retroactively with certainty. Group train/validation splits by whole match.

Validation: 22 focused client/export tests passed, including draining a failed
page batch before a subsequent successful read, projectile component guards,
confirmed versus timed-out labels, and preventing false legacy-tail alignment.
Live bounded retry recovery still requires another real-game observation; these
are unit and offline replay results, not a claim of a new successful match.
