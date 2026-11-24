$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$buildDir = Join-Path $projectRoot 'artifacts/nulls_live/touch_build'
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
& 'F:/JDK/bin/javac.exe' -source 8 -target 8 -d $buildDir scripts/native/CrrlTouch.java
if ($LASTEXITCODE -ne 0) { throw 'Touch helper Java compilation failed' }
& 'G:/sdk/build-tools/36.1.0/d8.bat' --min-api 23 --output $buildDir (Join-Path $buildDir 'CrrlTouch.class')
if ($LASTEXITCODE -ne 0) { throw 'Touch helper DEX compilation failed' }
& 'G:/sdk/platform-tools/adb.exe' -s 127.0.0.1:16416 push (Join-Path $buildDir 'classes.dex') /data/local/tmp/crrl-touch.dex
if ($LASTEXITCODE -ne 0) { throw 'Touch helper deployment failed' }
