# Flywheel health monitoring and emergency shutdown

## Active mode: notify only; shutdown removed

The automatic shutdown feature was revoked by the user on 2026-11-18.
The monitoring script no longer has shutdown commands or shutdown flags.
It never creates STOP in response to low memory, disconnection, or errors.
Available commit is recorded as a metric only. Failures leave recordings and
logs available for diagnosis and recovery. No automatic PC shutdown occurs.
Historical `shutdown_pending.json` describes the earlier incident only.

## Every-30-minute notification monitor

Windows scheduled task: `CR_RL_AWBC_Health_30min`.

Runs as the current interactive Windows user, every 30 minutes, without a
continuously running Python worker. It uses Python `-S` (no Torch import),
ignores overlapping invocations, and has a five-minute execution limit.
It starts after a missed trigger when possible, but does not wake the PC.

Checks:

- Configured emulator ADB connection and local supervisor/uploader processes.
- Completed games, uploaded game count, current game and trajectory freshness.
- Windows available commit, recorded for diagnosis only.
- Colab trainer/controller/receiver processes, update and log freshness.
- Game/update progress compared with the previous half-hour check.

Stale local trajectory threshold: 15 minutes. Stale cloud activity: 20 minutes.
Colab connection errors are retried once and reported as unknown connectivity,
not as confirmed training crashes. An intentional local `STOP` suppresses
missing-emulator/local-worker warnings; cloud monitoring continues.

Normal checks write local JSON only. Anomalies additionally use Windows
`msg.exe` to notify the logged-in user. This is a local Windows notification,
not an automatic message to the Codex conversation. Notification success is
recorded in `latest.json` when attempted. The monitor never restarts the
emulator, changes policy parameters, swaps Actors, or interrupts training.

Reports:

- `artifacts/awbc_flywheel/live/health/latest.json`
- `artifacts/awbc_flywheel/live/health/history.jsonl`

Reinstall/update the same task (does not create a second task):

```powershell
& .\scripts\register_awbc_monitor.ps1
```

Check once immediately:

```powershell
Start-ScheduledTask -TaskName CR_RL_AWBC_Health_30min
```

Disable monitoring without stopping collection or Colab:

```powershell
Disable-ScheduledTask -TaskName CR_RL_AWBC_Health_30min
```
