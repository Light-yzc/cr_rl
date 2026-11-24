param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$artifactRoot = Join-Path $workspace 'artifacts'
$activeRun = Join-Path $artifactRoot 'ppo_v3_v100'
$retained = @(
    'ppo_v3_v100/ppo_latest.pt',
    'ppo_v3_v100/ppo_best_val.pt',
    'ppo_v3_v100/actor_export.pt',
    'ppo_v3_v100/actor_best_val.pt',
    'ppo_v3_v100_bak_u480/actor_export.pt',
    'ppo_v3_v100_bak_u275/actor_export.pt',
    'ppo_prep_20261113/best.pt'
) | ForEach-Object { (Resolve-Path -LiteralPath (Join-Path $artifactRoot $_)).Path }
$pool = Get-Content -LiteralPath (Join-Path $activeRun 'opponent_pool.json') -Raw | ConvertFrom-Json
$activeOpponents = @($pool.entries | ForEach-Object {
    $name = [IO.Path]::GetFileName($_.checkpoint)
    (Resolve-Path -LiteralPath (Join-Path $activeRun ('opponents/' + $name))).Path
})
$retained += $activeOpponents
$allowedRoots = @(
    (Join-Path $activeRun 'checkpoints'),
    (Join-Path $activeRun 'opponents'),
    (Join-Path $artifactRoot 'ppo_v3_v100_bak_u480'),
    (Join-Path $artifactRoot 'ppo_v3_v100_bak_u275'),
    'E:\cr_rl_cleanup_archive_20261116'
) | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') + '\' }
$candidates = @()
foreach ($name in @('ppo_v3_v100_bak_u480','ppo_v3_v100_bak_u275')) {
    $old = Join-Path $artifactRoot $name
    $candidates += Get-ChildItem -LiteralPath $old -File -Filter '*.pt'
    foreach ($sub in @('checkpoints','opponents','rollouts')) {
        $directory = Join-Path $old $sub
        if (Test-Path -LiteralPath $directory) {
            $candidates += Get-ChildItem -LiteralPath $directory -File -Recurse
        }
    }
}
$candidates += Get-ChildItem -LiteralPath (Join-Path $activeRun 'checkpoints') -File -Filter '*.pt'
$candidates += Get-ChildItem -LiteralPath (Join-Path $activeRun 'opponents') -File -Filter '*.pt'
$abandonedArchive = 'E:\cr_rl_cleanup_archive_20261116'
if (Test-Path -LiteralPath $abandonedArchive) {
    $candidates += Get-ChildItem -LiteralPath $abandonedArchive -File -Recurse
}
$files = @($candidates | Where-Object { $_.FullName -notin $retained } | Sort-Object FullName -Unique)
foreach ($file in $files) {
    $resolved = (Resolve-Path -LiteralPath $file.FullName).Path
    if (!(@($allowedRoots | Where-Object {
        $resolved.StartsWith($_, [StringComparison]::OrdinalIgnoreCase)
    }).Count)) { throw "Path outside explicit cleanup scope: $resolved" }
    if ($resolved -in $retained) { throw "Retained checkpoint in deletion plan: $resolved" }
}
$report = Join-Path $artifactRoot 'colab_upload_20261116'
New-Item -ItemType Directory -Path $report -Force | Out-Null
$files | Select-Object FullName,Length | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $report 'manual_cleanup_plan.json') -Encoding UTF8
Write-Host "Selected obsolete files: $($files.Count). Current latest/best, BC and active league files retained."
Write-Host 'Logical file sizes include hard links; actual recovered space may be smaller.'
if (!$Apply) {
    Write-Host 'Preview only. Review manual_cleanup_plan.json, then run with -Apply to delete.'
    exit 0
}
$beforeG = (Get-PSDrive G).Free
$beforeE = (Get-PSDrive E).Free
foreach ($file in $files) { Remove-Item -LiteralPath $file.FullName }
[pscustomobject]@{
    deleted_files=$files.Count
    recovered_G_bytes=(Get-PSDrive G).Free-$beforeG
    recovered_E_bytes=(Get-PSDrive E).Free-$beforeE
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'manual_cleanup_result.json')
Write-Host 'Cleanup complete; details: manual_cleanup_result.json'
