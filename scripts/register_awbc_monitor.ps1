$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskName = 'CR_RL_AWBC_Health_30min'
$mode = '--notify'
$interval = 30
$action = New-ScheduledTaskAction -Execute 'F:\python\python.exe' `
    -Argument ('-S "{0}" {1}' -f (Join-Path $PSScriptRoot 'monitor_awbc_flywheel.py'),$mode) `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes($interval) `
    -RepetitionInterval (New-TimeSpan -Minutes $interval)
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 12) -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Description ('Emulator/traj/Colab health; mode: '+$mode) -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object TaskName,LastRunTime,NextRunTime,LastTaskResult
