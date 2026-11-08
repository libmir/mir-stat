param(
    [Parameter(Mandatory)][string]$Build,
    [switch]$AllowPackage
)

$buildArgs = @('test', "--build=$Build")
$mode = 'sequential per-file'
try {
    $machine = Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 5 -ErrorAction Stop
    $cpus = [int]$machine.NumberOfLogicalProcessors
    $memory = $machine.TotalPhysicalMemory
    # DUB uses all logical CPUs for parallel compilation on Windows (dub#3177).
    # Reserve 2 GiB for the host and budget 2.5 GiB for each compiler worker.
    if ($cpus -gt 1 -and $memory -ge (2GB + $cpus * 2.5GB)) {
        $mode = 'parallel per-file'
    } elseif ($AllowPackage -and $cpus -eq 1 -and $memory -ge 20GB) {
        $mode = 'package'
    }
} catch {
    Write-Warning 'Runner capacity unavailable; using sequential per-file compilation.'
}
if ($mode -ne 'package') {
    $buildArgs += '--build-mode=singleFile'
}
if ($mode -eq 'parallel per-file') {
    $buildArgs += '--parallel'
}
Write-Host "Windows $Build compilation: $mode."
& "$PSScriptRoot/windows-resources.ps1" Run dub @buildArgs
exit $LASTEXITCODE
