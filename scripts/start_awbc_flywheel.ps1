param(
    [string]$Plan = '2:10,1:10',
    [int]$RefreshMinutes = 60,
    [int]$UploadBatchGames = 10,
    [ValidateRange(0, 1024)]
    [double]$MinFreeCommitGiB = 6,
    [string]$OutputDirectory = 'artifacts/awbc_flywheel/live',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$folder = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$prefix = [IO.Path]::GetFullPath($projectRoot) + [IO.Path]::DirectorySeparatorChar
if (!$folder.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside the project' }
[IO.Directory]::CreateDirectory($folder) | Out-Null
$stopMarker = Join-Path $folder 'STOP'
if (Test-Path -LiteralPath $stopMarker) {
    if (!$Resume) { throw 'STOP exists. Use -Resume to archive that marker and resume.' }
    Move-Item -LiteralPath $stopMarker -Destination (Join-Path $folder ('STOP_archived_' + (Get-Date -Format 'yyyyMMdd_HHmmss')))
}
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_awbc_flywheel.py*'
}
if ($running) { throw "A flywheel already runs: $($running.ProcessId -join ', ')" }
foreach ($name in @('console.log','error.log')) {
    $previous = Join-Path $folder $name
    if (Test-Path -LiteralPath $previous) {
        Move-Item -LiteralPath $previous -Destination (Join-Path $folder ($name + '.previous_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fffffff')))
    }
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:CUDA_MODULE_LOADING = 'LAZY'
$arguments = @('-u', ('"{0}"' -f (Join-Path $projectRoot 'scripts/run_awbc_flywheel.py')),
    '--out', ('"{0}"' -f $folder), '--plan', $Plan, '--refresh-minutes', "$RefreshMinutes",
    '--upload-batch-games', "$UploadBatchGames", '--min-free-commit-gib', "$MinFreeCommitGiB", '--refresh-now')
$job = Start-Process -FilePath 'F:/python/python.exe' -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $folder 'console.log') `
    -RedirectStandardError (Join-Path $folder 'error.log') -PassThru
[pscustomobject]@{pid=$job.Id;folder=$folder;plan=$Plan;refresh_minutes=$RefreshMinutes;upload_batch_games=$UploadBatchGames} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'launcher.json') -Encoding UTF8
Write-Host "Flywheel PID $($job.Id); logs: $folder"
Write-Host 'It keeps collecting until STOP is created; Ctrl+C in this launching shell does not stop the hidden worker.'
