param(
    [int]$WaitForProcess = 0,
    [string]$OutputDirectory = 'artifacts/nulls_live/preset_60_20261117'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$folder = [IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
if ($WaitForProcess) {
    $old = Get-Process -Id $WaitForProcess -ErrorAction SilentlyContinue
    if ($old) { $old.WaitForExit() }
}
if (Test-Path -LiteralPath (Join-Path $folder 'STOP')) { exit 0 }
if ($WaitForProcess) {
    $previous = Get-Content -LiteralPath (Join-Path $folder 'batch.json') -Raw | ConvertFrom-Json
    $finishedSession = $false
    if ($previous.inflight) {
        $manifestPath = Join-Path (Join-Path $folder $previous.inflight.name) 'manifest.json'
        if (Test-Path -LiteralPath $manifestPath) {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $finishedSession = $manifest.collection_complete -and $manifest.stop_reason -eq 'battle_end'
        }
    }
    if (!$finishedSession -and ($previous.inflight -or @($previous.completed.PSObject.Properties).Count -lt 60)) {
        'Previous batch exited before completion; review its error before resuming.' |
            Set-Content -LiteralPath (Join-Path $folder 'loop_handoff_error.txt')
        exit 1
    }
}
$env:PYTHONPATH = Join-Path $root 'src'
$pythonExe = 'F:/python/python.exe'
& $pythonExe -u scripts/collect_nulls_presets.py --plan '2:30,1:30' --out $folder --repeat 1>> (Join-Path $folder 'loop.log') 2>> (Join-Path $folder 'loop.err.log')
$status = $LASTEXITCODE
$state = Get-Content -LiteralPath (Join-Path $folder 'batch.json') -Raw | ConvertFrom-Json
[pscustomobject]@{time=(Get-Date).ToString('o');exit_code=$status;completed=@($state.completed.PSObject.Properties).Count;inflight=$state.inflight} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $folder 'loop_exit.json') -Encoding UTF8
exit $status
