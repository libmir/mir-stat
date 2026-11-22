$ErrorActionPreference = 'Stop'
$master = Join-Path $env:RUNNER_TEMP $env:CACHE_DIR
$pr = Join-Path $env:RUNNER_TEMP $env:PR_CACHE_DIR
$null = [IO.Directory]::CreateDirectory($master)
$null = [IO.Directory]::CreateDirectory($pr)
$previous = @(Get-ChildItem -LiteralPath $pr -Filter 'ircache_*.obj' -File -Name)
$baseline = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(Get-ChildItem -LiteralPath $master -Filter 'ircache_*.obj' -File -Name))

# LDC reads one directory. Drop PR objects now supplied by master and move
# the rest beside the baseline for the duration of the build.
foreach ($name in $previous) {
    $path = Join-Path $pr $name
    if ($baseline.Contains($name)) {
        [IO.File]::Delete($path)
    } else {
        [IO.File]::Move($path, (Join-Path $master $name))
    }
}
$env:DFLAGS = "-cache=$master"
& "$PSScriptRoot/windows-test.ps1" unittest-release -AllowPackage
if ($LASTEXITCODE) { exit $LASTEXITCODE }

$current = @(Get-ChildItem -LiteralPath $master -Filter 'ircache_*.obj' -File -Name |
    Where-Object { !$baseline.Contains($_) })
foreach ($name in $current) {
    [IO.File]::Move((Join-Path $master $name), (Join-Path $pr $name))
}
$changed = [bool](Compare-Object $previous $current)
Write-Host "Windows LDC cache: $($baseline.Count) master objects, $($previous.Count) previous PR objects, $($current.Count) current PR objects."
"changed=$($changed.ToString().ToLowerInvariant())" >> $env:GITHUB_OUTPUT
