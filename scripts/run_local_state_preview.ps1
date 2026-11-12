param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Pairs = 1,
    [ValidateSet(1,2,4)][int]$Envs = 1,
    [ValidateSet('fp16','fp32')][string]$Precision = 'fp16'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'F:/yzc/Miniconda/envs/cs285/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw "Torch environment not found: $python" }
$env:JAVA_HOME = 'F:/JDK'
$env:PYTHONPATH = "$root/src;$root/.cache/vendor/crforge/python"
$opponent = (Resolve-Path -LiteralPath $Checkpoint).Path
& $python (Join-Path $root 'scripts/evaluate_state_policy.py') `
    --policy $opponent --opponent $opponent --decks (Join-Path $root 'configs/decks_ppo46.yaml') `
    --out (Join-Path $root $Out) --pairs $Pairs --envs $Envs --device cuda `
    --precision $Precision --no-native-audit
if ($LASTEXITCODE -ne 0 -and !(Test-Path -LiteralPath (Join-Path $root $Out 'evaluation.json'))) {
    throw "Preview failed with exit code $LASTEXITCODE"
}
