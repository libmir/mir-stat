# Best-effort Windows CI diagnostics. Do not print environment or command arguments.
# Use unbound arguments so native switches cannot bind to helper parameters.
$Mode = if ($args.Count) { [string]$args[0] } else { '' }
$Command = @($args | Select-Object -Skip 1)
if ($Mode -notin @('Report', 'Run')) {
    Write-Warning 'Usage: windows-resources.ps1 {Report|Run COMMAND [ARGS...]}'
    exit 2
}

$ErrorActionPreference = 'Stop'
# Preserve native exit codes even if the caller enables native error promotion.
$PSNativeCommandUseErrorActionPreference = $false

$snapshot = {
    Write-Host ('Memory snapshot {0:yyyy-MM-ddTHH:mm:ssZ}' -f [DateTime]::UtcNow)
    try {
        $os = Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5
        Write-Host ('Physical RAM: total {0:N0} MiB, free {1:N0} MiB; virtual memory: total {2:N0} MiB, free {3:N0} MiB' -f
            ($os.TotalVisibleMemorySize / 1024), ($os.FreePhysicalMemory / 1024),
            ($os.TotalVirtualMemorySize / 1024), ($os.FreeVirtualMemory / 1024))
        # Per-process working sets can share pages and are not an aggregate peak.
        Get-Process -ErrorAction SilentlyContinue |
            Sort-Object WorkingSet64 -Descending |
            Select-Object -First 10 Id, ProcessName,
                @{Name = 'WorkingSetKiB'; Expression = { [long]($_.WorkingSet64 / 1KB) }},
                @{Name = 'PrivateBytesKiB'; Expression = { [long]($_.PrivateMemorySize64 / 1KB) }} |
            Format-Table -AutoSize | Out-String -Width 180 | Write-Host
    } catch {
        Write-Warning 'Windows memory snapshot unavailable.'
    }
}

if ($Mode -eq 'Report') {
    Write-Host '::group::Windows runner resources'
    try {
        Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5 |
            Select-Object Caption, Version, BuildNumber, OSArchitecture |
            Format-List | Out-String | Write-Host
        Get-CimInstance Win32_Processor -OperationTimeoutSec 5 |
            Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
            Format-Table -AutoSize | Out-String -Width 180 | Write-Host
        Write-Host "PowerShell: $($PSVersionTable.PSVersion); runtime-visible processors: $([Environment]::ProcessorCount)"
        $current = [Diagnostics.Process]::GetCurrentProcess()
        try {
            Write-Host ('PowerShell processor affinity mask: 0x{0:X}' -f $current.ProcessorAffinity.ToInt64())
        } finally {
            $current.Dispose()
        }
        Get-CimInstance Win32_PageFileUsage -OperationTimeoutSec 5 |
            Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage |
            Format-Table -AutoSize | Out-String -Width 180 | Write-Host
        Write-Host 'Page-file sizes and usage above are MiB. Host memory figures do not describe Windows job-object limits.'
    } catch {
        Write-Warning 'Some Windows runner resource information is unavailable.'
    }
    & $snapshot
    Write-Host '::endgroup::'
    exit 0
}

if (!$Command -or $Command.Count -eq 0) {
    Write-Warning 'Missing native command for resource monitoring.'
    exit 2
}

& $snapshot
Write-Host 'Working sets and private bytes are per-process samples, not the combined peak of the build.'
$stopSignal = [Threading.ManualResetEventSlim]::new($false)
$monitor = $null
$clock = $null
$status = 127
try {
    try {
        # A thread job shares the stop signal and streams output immediately.
        # It creates no separate monitoring process that could be left behind.
        $monitor = Start-ThreadJob -StreamingHost $Host -ArgumentList $snapshot.ToString(), $stopSignal -ScriptBlock {
            param($snapshotSource, $stopSignal)
            $ErrorActionPreference = 'Stop'
            $takeSnapshot = [scriptblock]::Create($snapshotSource)
            while (!$stopSignal.Wait(30000)) {
                & $takeSnapshot
            }
        }
    } catch {
        Write-Warning 'Periodic Windows memory monitoring unavailable; continuing with the command.'
    }
    try {
        $native = Get-Command -Name $Command[0] -CommandType Application -ErrorAction Stop
        $commandArgs = @($Command | Select-Object -Skip 1)
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & $native.Source @commandArgs
        $status = $LASTEXITCODE
    } catch {
        # Avoid including command arguments in diagnostics for launch failures.
        Write-Warning 'Unable to start the requested native command.'
        $status = 127
    }
} finally {
    if ($null -ne $clock) { $clock.Stop() }
    $stopSignal.Set()
    try {
        if ($null -ne $monitor) {
            # Signaling first wakes the timed wait, so cleanup need not wait 30s.
            Stop-Job -Job $monitor -ErrorAction SilentlyContinue
            Remove-Job -Job $monitor -Force -ErrorAction SilentlyContinue
        }
    } finally {
        $stopSignal.Dispose()
    }
    if ($null -ne $clock) { Write-Host ('Elapsed: {0:F3} s' -f $clock.Elapsed.TotalSeconds) }
    & $snapshot
}
Write-Host "Test command exit status: $status"
exit $status
