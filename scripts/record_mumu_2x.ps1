param(
    [ValidateSet("auto", "window", "adb")]
    [string]$Mode = "auto",
    [int]$Width = 360,
    [int]$Height = 640,
    [double]$Speed = 2.0,
    [int]$Crf = 30,
    [double]$Seconds = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Py = if (Test-Path "$RepoRoot\.venv-v100\Scripts\python.exe") {
    "$RepoRoot\.venv-v100\Scripts\python.exe"
} elseif (Test-Path "F:\yzc\Miniconda\envs\cs285\python.exe") {
    "F:\yzc\Miniconda\envs\cs285\python.exe"
} else {
    "python"
}

Write-Host "=== MuMu 2x Compact Video Recorder (${Width}x${Height}, ${Speed}x, CRF=${Crf}) ===" -ForegroundColor Cyan
Write-Host "Videos save to: $RepoRoot\artifacts\videos\" -ForegroundColor Green
Write-Host "Press Ctrl+C when the battle finishes to finalize the MP4!" -ForegroundColor Yellow

& $Py "$RepoRoot\scripts\record_mumu_2x.py" `
    --mode $Mode `
    --width $Width `
    --height $Height `
    --speed $Speed `
    --crf $Crf `
    --seconds $Seconds
