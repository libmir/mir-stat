param([Parameter(Mandatory)][string]$Build)

# DUB sizes its Windows worker pool from the system CPU count. Use this guard
# until DUB supports an explicit worker limit: https://github.com/dlang/dub/issues/3177
$buildArgs = @('test', "--build=$Build", '--build-mode=singleFile')
try {
    $machine = Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 5 -ErrorAction Stop
    $cpus = [int]$machine.NumberOfLogicalProcessors
    if ($cpus -ge 2 -and $cpus -le 4 -and $machine.TotalPhysicalMemory -ge 12GB) {
        $buildArgs += '--parallel'
        Write-Output "Windows DMD per-file compilation: up to $cpus workers."
    } else {
        Write-Output "Windows DMD per-file compilation: sequential fallback ($cpus CPUs)."
    }
} catch {
    Write-Warning 'Runner capacity unavailable; using sequential per-file compilation.'
}
& "$PSScriptRoot/windows-resources.ps1" Run dub @buildArgs
exit $LASTEXITCODE
