param([Parameter(Mandatory=$true)][int]$Collector,[Parameter(Mandatory=$true)][string]$Folder)
$ErrorActionPreference = 'Stop'
$process = Get-Process -Id $Collector -ErrorAction SilentlyContinue
if (!$process) { exit 1 }
do {
    $finished = $process.WaitForExit(1800000)
    try {
        $batch = Get-Content -LiteralPath (Join-Path $Folder 'batch.json') -Raw | ConvertFrom-Json
        $counts = @{}
        foreach ($entry in $batch.completed.PSObject.Properties) { $counts[[string]$entry.Value.slot]++ }
        $events = if ($batch.inflight) { Join-Path (Join-Path $Folder $batch.inflight.name) 'events.jsonl' }
        $age = if ($events -and (Test-Path -LiteralPath $events)) {
            [int]((Get-Date)-(Get-Item -LiteralPath $events).LastWriteTime).TotalSeconds
        } else { $null }
        $state = if ($finished) { 'exited' } elseif ($null -ne $age -and $age -gt 720) { 'stalled' } else { 'running' }
        $errors = Get-Content -LiteralPath (Join-Path $Folder 'stable.err.log') -Tail 5 -ErrorAction SilentlyContinue
        $row = [pscustomobject]@{time=(Get-Date).ToString('o');status=$state;counts=$counts;last_event_age_seconds=$age;inflight=$batch.inflight;error=($errors -join ' ')}
        $row | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath (Join-Path $Folder 'health_30min.jsonl') -Encoding UTF8
        if ($finished) { $row | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Folder 'stable_exit.json') -Encoding UTF8 }
    } catch {
        $_.ToString() | Add-Content -LiteralPath (Join-Path $Folder 'health_monitor_error.log')
    }
} while (!$finished)
