param([string]$OutputDirectory = 'artifacts/awbc_flywheel/live')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$folder = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$prefix = [IO.Path]::GetFullPath($projectRoot) + [IO.Path]::DirectorySeparatorChar
if (!$folder.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside the project' }
if (!(Test-Path -LiteralPath $folder)) { throw 'No flywheel output directory exists' }
[IO.File]::WriteAllText((Join-Path $folder 'STOP'),(Get-Date).ToString('o'))
Write-Host 'Stop requested. The current battle finishes, then remaining completed games upload.'
