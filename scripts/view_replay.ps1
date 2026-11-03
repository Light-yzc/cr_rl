param(
    [string]$Replay = "",
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Replay)) {
    $latestReplay = Join-Path $projectRoot "artifacts/replay_viewer/latest_replay.json"
    if (Test-Path -LiteralPath $latestReplay) {
        $Replay = (Get-Content -LiteralPath $latestReplay -Raw | ConvertFrom-Json).replay
    } else {
        $Replay = "artifacts/bc_multideck_20261103/evaluation/all_matches.jsonl"
    }
}
$viewerRoot = Join-Path $projectRoot ".cache/replay-viewer"
$vendorRoot = Join-Path $viewerRoot "crforge"
$libraryRoot = Join-Path $vendorRoot "desktop/build/install/desktop/lib"
$tablesRoot = Join-Path $viewerRoot "tables"
$expectedCommit = "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6"

if (!(Test-Path -LiteralPath (Join-Path $viewerRoot "UPSTREAM_COMMIT"))) {
    throw "Viewer bundle is missing from .cache/replay-viewer. See docs/REPLAY_VIEWER.md."
}
if ((Get-Content -LiteralPath (Join-Path $viewerRoot "UPSTREAM_COMMIT") -Raw).Trim() -ne $expectedCommit) {
    throw "Viewer revision differs from the training simulator."
}
if (![System.IO.Path]::IsPathRooted($Replay)) {
    $Replay = Join-Path $projectRoot $Replay
}
$replayPath = (Resolve-Path -LiteralPath $Replay).Path
$annotatedReplay = Join-Path (Join-Path (Split-Path -Parent $replayPath) 'viewer') (Split-Path -Leaf $replayPath)
if ((Split-Path -Leaf $replayPath) -match '^match_\d+\.json$' -and (Test-Path -LiteralPath $annotatedReplay)) {
    $replayPath = $annotatedReplay
}
if (!(Test-Path -LiteralPath (Join-Path $tablesRoot "16.635.3"))) {
    throw "The pinned 16.635.3 game tables are missing."
}

$javaExe = (Get-Command java.exe -ErrorAction Stop).Source
$javawExe = Join-Path (Split-Path -Parent $javaExe) "javaw.exe"
if (!(Test-Path -LiteralPath $javawExe)) { $javawExe = $javaExe }

$labelsBuilt = Join-Path $viewerRoot 'policy_labels_v1.built'
if (!(Test-Path -LiteralPath $labelsBuilt)) {
    & python (Join-Path $projectRoot 'scripts/patch_replay_viewer.py')
    if ($LASTEXITCODE -ne 0) { throw 'Viewer policy-label patch failed.' }
    $Build = $true
}
if ($Build -or !(Test-Path -LiteralPath $libraryRoot)) {
    $previousJavaHome = $env:JAVA_HOME
    try {
        $env:JAVA_HOME = Split-Path -Parent (Split-Path -Parent $javaExe)
        Push-Location $vendorRoot
        try {
            & .\gradlew.bat :desktop:installDist --console=plain --no-daemon
            if ($LASTEXITCODE -ne 0) { throw "Desktop viewer build failed." }
            Set-Content -LiteralPath $labelsBuilt -Value 'policy-labels-v1' -Encoding ASCII
        } finally { Pop-Location }
    } finally { $env:JAVA_HOME = $previousJavaHome }
}

$arguments = @(
    "-Dcrforge.tablesRoot=$tablesRoot",
    "-Dcrforge.dataVersion=16.635.3",
    "-Dcrforge.lockVersion=16.635.3",
    "-cp", (Join-Path $libraryRoot "*"),
    "org.crforge.desktop.DesktopLauncher", "--replay", $replayPath
)
# Quote native-process arguments rather than interpreting file paths as shell code.
$argumentString = ($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
$logRoot = Join-Path $projectRoot "artifacts/replay_viewer"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logRoot "viewer_$stamp.stdout.log"
$stderrPath = Join-Path $logRoot "viewer_$stamp.stderr.log"

# This is the interactive window the user asked to watch.
$viewerProcess = Start-Process -FilePath $javawExe -ArgumentList $argumentString `
    -WorkingDirectory $vendorRoot -WindowStyle Normal -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$record = @{
    pid = $viewerProcess.Id
    replay = $replayPath
    upstream_commit = $expectedCommit
    data_version = "16.635.3"
    stdout = $stdoutPath
    stderr = $stderrPath
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot "viewer_process.json") -Encoding UTF8
Write-Output "Replay viewer started (PID $($viewerProcess.Id)). SPACE pauses, +/- changes speed, [ and ] switch matches."
