# Trajectory-to-simulator repair protocol

This protocol implements an explicit repair path and a separate BC experiment for event replays. It is
not a claim that repaired states match the original human battle.

The importer keeps three outcomes separate:

- `exact`: the requested card, time and placement are accepted by the native
  simulator mask;
- `repaired`: an explicitly logged time snap, nearest legal tile snap, or live
  hand swap was needed;
- `rejected`: the event cannot be executed under the current state.

The default BC path remains exact/prefix-safe. The repair path is opt-in:

```bash
python scripts/import_il_replay.py source.parquet \
  --infer-opening-cycle --max-omissions 0 \
  --repair-hand --max-hand-repairs -1 \
  --repair-tile --max-tile-shift 2 \
  --repair-time --max-delay-ticks 200 \
  --max-source-rows 128 --limit 16 --min-plays 20 --min-coverage 1 \
  --registry artifacts/bc_repaired_20261109/registry.json \
  --out artifacts/bc_repaired_20261109
```

Every accepted replay records `repair_records`, `repair_summary`, and
`repair_is_exact_bc=false` in metadata. A repaired replay must not be merged
silently into the exact BC snapshot.

Repair semantics:

1. **Opening cycle:** search and reorder the deck before reset. This is the
   preferred repair because it preserves the simulator's normal hand cycle.
2. **Time:** use the next 5-tick boundary (causal ceiling), not a past state.
   The default alignment tolerance is four 20 Hz ticks. With `--repair-time`,
   retain a per-seat FIFO and retry closed masks up to `--max-delay-ticks`
   (default 200 ticks = 10 seconds from the original source event). Later source
   events of that seat cannot overtake an awaiting native confirmation or retry.
3. **Placement:** if enabled, choose the nearest legal tile for the same hand
   slot, bounded by Manhattan distance. The requested and executed actions are
   both recorded.
4. **Hand:** if enabled, swap a requested card from the live queue into an
   available hand slot, preserving the displaced card in the queue. This
   mutates the running simulator and therefore marks all subsequent data as
   repaired. `--max-hand-repairs -1` enables an unlimited per-game swap budget;
   use a positive value for a bounded ablation. Pending/empty hand slots are
   protected, and the runtime deck index is resolved after opening reordering.
5. **Elixir and other closed masks:** never add elixir or invent a legal action.
   A hand swap may expose the requested card, but it cannot override an
   insufficient-elixir or rule mask. The repair scheduler waits and recomputes
   the mask within its time budget; only irrecoverable or exhausted events stop
   the strict repaired prefix.

6. **Skills:** parse `ability_source_candidates` when `card_key` is null. Match
   all source candidates to native champion slots and require a unique legal
   slot. Record the inferred attribution; do not choose the first candidate.
7. **Execution:** `NativeReplayObserver` pairs every submitted command name with
   the actual native card/skill outcome. Retry declared transient failures within
   the original event deadline. Refused or unconfirmed submissions are masked
   out of BC labels. Native match decision is detected before its delayed
   `vector.dones` end screen; events after it are counted as incomplete.

`dataset.manifest.json` points to `human_bc_reconstructed.npz` and its checked
metadata. The live per-game `.live.npz` files and `.repair.json` sidecars retain
state/action and command evidence. The final dataset is accepted by the existing
sharded BC loader. Tail WAITs beyond the known source horizon receive
`GENERATED_WAIT_TAIL`, so they remain history frames but are not human targets.
Use small collection shards (for example 16 accepted games) to bound the final
concatenation memory on a laptop.

The resulting completion rate is an engineering metric for the repair policy,
not a human-state fidelity metric. Compare repaired BC against the old BC only
with the same fixed deck pairs, simulator seeds, seat swaps, and native replay
audit. Report exact, repaired, rejected, repair count, and first-repair tick
separately.

## Dataset isolation and an executable BC experiment

The manifest/metadata field `dataset_kind` is `exact` or `repaired`. Here
`exact` means no runtime repair policy was enabled/applied; it still describes
simulator-reconstructed states, not verified original human states. A repair
run may include games that needed no intervention, but the entire experimental
dataset remains `repaired` so that a source-wide training decision is explicit.

`freeze_datasets` and `prepare_clean_bc_run.py` reject repaired data by default.
They also recognize legacy provenance (`repair_is_exact_bc=false`, enabled
hand/tile/time repair flags, nonempty repair records, or repair counts), so a
missing `dataset_kind` cannot hide an intervention. A declaration of `exact`
with such evidence is an error. Exact and repaired source datasets cannot be
merged by the snapshot freezer, even with opt-in; prepare a dedicated repaired
dataset instead. The canonical source-game identity remains the same across
experimental variants for split consistency, while each variant gets its own
dataset directory and registry.

The user-requested repaired BC trial is explicitly supported:

```bash
python scripts/prepare_clean_bc_run.py /content/bc_repaired_20261109 \
  --allow-repaired --min-a-games 2 \
  --out /content/bc_repair_experiment_20261109 \
  --init-from-bc /content/frozen_old_bc.pt --epochs 2
```

This opt-in selects paired, continuous native observation grids containing legal
action labels. It retains A/B/C diagnostics but does not require repaired games
to pass the exact-prefix A-tier duration/count gate. Two whole games is only the
minimum to populate the existing train/validation split, not evidence of model
quality. Use as many usable repaired games as the bounded collection budget
produces. Training commands remain reviewable in `clean_run.json`; execute that
command to actually train the candidate. Do not label this experimental
snapshot `clean_snapshot=true` or silently add it to a later exact run.

## Event/state ledger required for trustworthy labels

Every source event needs a stable `(canonical_game_id, source_event_index)` and
one terminal status, including events skipped after a simulator terminal:

- original source tick, side, event kind, card identity/form, original world
  coordinates and normalized seat coordinates;
- scheduled tick, actual submission tick, native run tick when observed,
  delay, tile displacement, and the FIFO/collision resolution used;
- requested card/action/tile separately from submitted action and confirmed
  executed card/action/tile; the current native hand-slot index belongs to the
  submission state and must never be reused from the source deck index;
- pre-action observation/mask row or an unambiguous row index and hash; hand,
  queue, elixir and pending actions before/after any state intervention;
- acceptance/refusal code, whether the command was submitted and whether it
  actually ran, repair categories, first repair tick and inherited repaired-state
  status for all later rows of both seats;
- a final unexecuted reason (`delay_budget`, `collision`, `terminal_before_event`,
  `fault`, `missing_card`, etc.) for every source event that did not execute.

`mask[action]=1` and incrementing a Python `used` counter prove only a legal
request. Deployment/ability delay can still leave a command pending or refused
when it is due to run. Report source events, submissions, native executions and
legal unmasked BC labels as separate totals.

For live hand interventions, capture the actual repaired run's observation and
mask before the action. The ordinary native replay records seed, decks, plays
and abilities; it does not record Python reflection calls to `hand.rearrange`.
Relabelling that replay without reapplying the intervention can recreate combat
plays while producing different hand/mask states. Either use online rows as the
training authority, or replay an explicit intervention ledger and compare every
saved observation/mask hash. A replay viewer file alone does not validate these
labels.

## Scheduling and trajectory completion

Use a stable per-seat FIFO on the model's 250 ms decision grid, with at most one
action per seat per decision. Both seats at the same decision are submitted
together. A short, bounded delay can resolve source timing, insufficient-elixir
or pending-slot mismatches without inventing elixir. Recompute the mask and hand
slot at the actual submission tick. Do not discard the second event merely
because two events snap to the same boundary, and do not let a later event
overtake an earlier delayed event on that seat.

Record time snap and nonnegative delay separately. Nearest-boundary rounding can
move an event earlier than the source tick; a causal scheduling option uses the
next boundary instead. Whatever policy is chosen must be fixed in the repair
configuration and evaluated with the same policy in all collection arms.

The original importer stops immediately after the last source event. That can
leave its last commands pending and is not a complete trajectory. Drain queued
commands for a bounded period and, if a source battle end is known, simulate to
that declared horizon or native terminal. If no source end is known, report
`source_events_consumed` separately from `simulator_terminal`; a generated WAIT
tail can be retained for diagnostics without claiming human WAIT supervision.
If the simulator ends early, mark every remaining source event
`terminal_before_event`; preserve the usable prefix and count the failure rather
than inventing states beyond terminal or calling the match complete.

The immediate acceptance target is a bounded repaired collection that produces
usable aligned state/action rows, followed by a real BC candidate trained on
those rows and a fixed paired comparison against the old BC. Reconstruction
uncertainty is reported alongside results; it is not a reason to indefinitely
block this experiment behind exact-client parity or expert-level metrics.

## Fixed-cohort CPU benchmark

```bash
python scripts/benchmark_replay_repairs.py source.parquet \
  --out artifacts/repair_benchmark_new \
  --count 100 --start-row 16 \
  --source-revision 059d43a02138a34b1b3009cc2acc7630fb99a638 \
  --allow-hand-repair --max-hand-repairs -1 \
  --max-tile-shift 50 --max-delay-ticks 200
```

The three arms are seeded exact replay, opening-cycle inference, and explicit
hand/tile/time repair. Exact/cycle execute source ticks at 20 Hz; repair is
collected on the model's 4 Hz grid. This measures the complete policies rather
than attributing their difference to a single independent intervention. All
arms use the same source rows, source seeds and native outcome confirmation.
The benchmark records execution file hashes before/after every arm and fails
if code changes, a process fails, or an arm times out. It reports raw source and
compatible-attempt denominators, source command coverage, native confirmations,
long-source subsets, source-horizon reach and early native decisions separately.
Tile distance 50 deliberately allows arena-wide nearest-legal repair in this
engineering experiment; distance 2 is the conservative default. Report the
observed displacement/hand-swap/delay rates when assessing BC quality.

See [verified repair results](TRAJ_REPAIR_RESULTS_20261109.md) for measured rates
and the next training comparison. This implementation does not by itself prove
that repaired BC is stronger than the previous BC checkpoint.

## 用户接受的正常提前终局收集（最新）

保留两个完成字段：`all_source_events_executed`严格要求所有原始事件均原生成功；
`source_or_native_terminal_completed`还允许模拟器已经正常判出胜负。后者不表示源事件
后半局也被执行。使用以下显式筛选收集可用对局，排除中途技能失败等未完成前缀：

```bash
python scripts/import_il_replay.py source.parquet \
  --infer-opening-cycle --repair-hand --max-hand-repairs -1 \
  --repair-tile --max-tile-shift 50 --repair-time --max-delay-ticks 200 \
  --seed 20261109 --require-source-or-terminal \
  --max-source-rows 128 --limit 16 --min-plays 20 --min-coverage 0 \
  --registry artifacts/bc_terminal_repaired/registry.json \
  --out artifacts/bc_terminal_repaired
```

正常终局需要native match_ended/decided、实际winner以及无fault/truncated；时间已到但
仍处于同冠决胜过程不会被误收。保留的终局末尾未确认请求仍mask掉，不能训练为成功动作。
技能关联现在使用native来源deck index和card candidates，不用卡牌名/单位名的字符串近似。
技能失败记录保存native槽位、单位、状态、冷却/次数、来源卡和匹配诊断。
