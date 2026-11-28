param(
    [int]$Seconds = 600,
    [int]$MaxPlays = 100,
    [ValidateRange(0,500)]
    [int]$Matches = 0,
    [string]$Device = 'cuda',
    [string]$PythonExecutable = '',
    [string]$AdbExecutable = '',
    [string]$Serial = '127.0.0.1:16416',
    [string]$Checkpoint = '',
    [ValidateSet('auto','persistent','shell')]
    [string]$TouchTransport = 'shell',
    [ValidateSet('hazard','policy_sample','confidence','argmax')]
    [string]$SelectionMode = 'hazard',
    [double]$PlayThreshold = 0.45,
    [double]$SlotThreshold = 0.28,
    [double]$TileTemperature = 0.0,
    [switch]$EnableAbilities,
    [switch]$NoResultPrompt,
    [switch]$AutoBattle,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = $PythonExecutable
if (!$pythonExe) { $pythonExe = $env:CRRL_PYTHON }
if (!$pythonExe) {
    $candidates = @(
        (Join-Path $projectRoot '.venv-v100/Scripts/python.exe'),
        (Join-Path $projectRoot '.venv/Scripts/python.exe'),
        'F:/python/python.exe',
        'F:/yzc/Miniconda/envs/cs285/python.exe'
    )
    $pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (!$pythonExe) {
    $cmdPy = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPy) { $pythonExe = $cmdPy.Source }
}
if (!$pythonExe) { throw 'No existing Python found; set CRRL_PYTHON or pass -PythonExecutable.' }
if (!(Test-Path -LiteralPath $pythonExe)) { throw "Python missing: $pythonExe" }

$adbExe = $AdbExecutable
if (!$adbExe) { $adbExe = $env:CRRL_ADB }
if (!$adbExe) {
    $adbCmd = Get-Command adb -ErrorAction SilentlyContinue
    if ($adbCmd) { $adbExe = $adbCmd.Source }
}
if (!$adbExe) {
    $adbExe = @('G:/sdk/platform-tools/adb.exe', "$env:LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe") |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (!$adbExe) { $adbExe = 'adb' }

if (!$Checkpoint) {
    $ckptCandidates = @(
        (Join-Path $projectRoot 'artifacts/g4_heavy_mcts/actor_best_val.pt'),
        (Join-Path $projectRoot 'artifacts/g4_heavy_mcts/actor_latest.pt'),
        (Join-Path $projectRoot 'artifacts/ppo_v3_v100/actor_export.pt')
    )
    $Checkpoint = $ckptCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (!$Checkpoint) { throw 'No actor checkpoint found; run colab_g4_ops.py download-actors or pass -Checkpoint.' }
}
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_nulls_policy\.py' }
if ($existing) {
    Write-Host "Controller already active (PID $($existing.ProcessId -join ', ')). Enter a battle; no second controller was started."
    exit 0
}
$env:PYTHONPATH = "$(Join-Path $projectRoot 'src');$(Join-Path $projectRoot 'scripts')"
$env:PYTHONIOENCODING = 'utf-8'
if ($Matches -eq 1) {
    $runDir = Join-Path $projectRoot ('artifacts/nulls_live/play_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
    $argsList = @('-u', 'scripts/run_nulls_policy.py', '--out', $runDir,
                  '--serial', $Serial, '--adb', $adbExe,
                  '--checkpoint', $checkpointPath,
                  '--seconds', "$Seconds", '--max-plays', "$MaxPlays", '--device', $Device,
                  '--touch-transport', $TouchTransport,
                  '--selection-mode', $SelectionMode,
                  '--play-threshold', "$PlayThreshold", '--slot-threshold', "$SlotThreshold",
                  '--tile-temperature', "$TileTemperature")
    if (!$DryRun) { $argsList += '--execute' }
    if ($EnableAbilities) { $argsList += '--enable-abilities' }
    if ($AutoBattle) { $argsList += '--auto-battle' }
    if (!$NoResultPrompt -and !$DryRun) { $argsList += '--prompt-result' }
    Write-Host "Log directory: $runDir"
    Write-Host "Match 1/1"
    Write-Host "Python: $pythonExe; ADB: $adbExe ($Serial); checkpoint: $checkpointPath"
    Write-Host 'Wait for ready, then enter a Nulls private-server battle. Ctrl+C stops control.'
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExe @argsList 2>&1 | Tee-Object -FilePath (Join-Path $runDir 'console.log')
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($LASTEXITCODE -ne 0) { throw "Policy controller exited with code $LASTEXITCODE; see $runDir" }
} else {
    $outRoot = Join-Path $projectRoot 'artifacts/nulls_live'
    New-Item -ItemType Directory -Path $outRoot -Force | Out-Null
    $argsList = @('-u', 'scripts/run_nulls_policy.py',
                  '--out-root', $outRoot, '--matches', "$Matches",
                  '--serial', $Serial, '--adb', $adbExe,
                  '--checkpoint', $checkpointPath,
                  '--seconds', "$Seconds", '--max-plays', "$MaxPlays", '--device', $Device,
                  '--touch-transport', $TouchTransport,
                  '--selection-mode', $SelectionMode,
                  '--play-threshold', "$PlayThreshold", '--slot-threshold', "$SlotThreshold",
                  '--tile-temperature', "$TileTemperature")
    if (!$DryRun) { $argsList += '--execute' }
    if ($EnableAbilities) { $argsList += '--enable-abilities' }
    if ($AutoBattle) { $argsList += '--auto-battle' }
    if (!$NoResultPrompt -and !$DryRun) { $argsList += '--prompt-result' }
    $label = if ($Matches -eq 0) { 'unlimited' } else { "$Matches" }
    Write-Host "Persistent single-load mode ($label matches); root: $outRoot"
    Write-Host "Python: $pythonExe; ADB: $adbExe ($Serial); checkpoint: $checkpointPath"
    Write-Host 'Model loads ONCE into GPU VRAM and reuses across all matches (<0.1s reset between battles).'
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExe @argsList 2>&1 | Tee-Object -FilePath (Join-Path $outRoot 'persistent_console.log')
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($LASTEXITCODE -ne 0) { throw "Policy controller exited with code $LASTEXITCODE; see $outRoot" }
}
