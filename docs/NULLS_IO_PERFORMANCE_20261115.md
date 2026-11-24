# Read and touch transport optimization

Changes apply on the next invocation of `scripts/start_nulls_policy.ps1`.
No model weights, sampling temperature or freshness threshold were changed.

## Reads

Cache only logical table-row metadata (name/SCID and card cost), keyed by data
pointer within a match. Clear on battle object changes or tick rewind. Never
retain live entity, HP, hand, elixir or timer pages across polls. Global pointer
pages are fetched together. Keep character component traversal restricted to
verified character/tower layouts. EIO/EFAULT invalidates the entire snapshot;
bounded fresh-poll retries remain active. This reduces read span and unverified
pointer traversal, but does not eliminate races with live game allocation.

Real MuMu Android12 training-camp A/B: alternate cached/uncached reads and reverse
order each iteration; 57 usable samples per mode after warmup:

| Metric | Uncached | Cached |
| --- | --- | --- |
| Median snapshot | 37.20ms | 33.00ms |
| P95 snapshot | 55.71ms | 52.13ms |
| Median transferred bytes | 368640 | 299008 |
| Median ADB exchanges | 16 | 16 |

About 11% lower median latency and 19% lower bytes in this short scene. Neither
latency nor race reduction is guaranteed in a large battle. No page failures
occurred in this benchmark. Raw results: `artifacts/nulls_live/io_benchmark.json`.
Static fake-memory comparisons verify unchanged complete observations and fresh
HP updates after caching.

## Touches

`scripts/native/CrrlTouch.java` runs once through Android `app_process` and uses
InputManager events, the same system input route as shell input. The calibrated
hand/board tap pair has 8ms press duration and a 20ms inter-tap gap. It does not
hook the game or modify its memory. Host transport remains synchronized; no
retries after a partially failed sequence. All bridge freshness, expected-card,
screen bounds and single-pending checks remain in place.

Eight harmless screen-corner pairs per mode, no game actions:

| Metric | Shell pair | Persistent pair |
| --- | --- | --- |
| Median host round trip | 67.73ms | 41.31ms |
| P95 host round trip | 81.72ms | 48.82ms |

The earlier 50ms-gap prototype was slower than shell at the median and was not
selected. Raw final comparison: `artifacts/nulls_live/touch_benchmark.json`.

Then `perf_live001` exercised the actual actor with the persistent transport:
89 decisions, five submitted plays, five confirmations, no page errors or
confirmation timeouts. Host touch calls measured 31–47ms. The run stopped at
battle end. A cold shape still produced one stale action, which was skipped by
the unchanged freshness guard.

Sending is not confirmation: median send-to-confirm was 1.33s in this run versus
1.11s in the prior different match. These are not controlled matched scenarios;
there is no evidence of improved total server acceptance latency. The transport
improvement is real but does not remove game/network/deploy delay or the 400ms
decision cadence. Do not shorten the pending lock based on touch acknowledgments.

The helper DEX (4180 bytes) is deployed to this MuMu instance. Rebuild without
downloads using `scripts/build_nulls_touch.ps1`. In `auto`, helper startup failure
logs a fallback to shell before any game taps. Runtime failure is never silently
resent through another transport. Force the old path with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_nulls_policy.ps1 -TouchTransport shell
```

25 focused tests passed, including cached/dynamic state equality, fast-transport
framing, partial-failure behavior, freshness checks and pending-lock preservation.
The bounded live check is interface evidence, not win-rate or long-session proof.
