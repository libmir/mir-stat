#!/usr/bin/env bash
# Best-effort Linux CI diagnostics. Do not print environment or command arguments.
set -uo pipefail

report_cgroups() {
    local group path name
    group=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null) || true
    if [[ -z "$group" || ! -r /sys/fs/cgroup/cgroup.controllers ]]; then
        echo 'Cgroup v2 resource information unavailable.'
        return 0
    fi
    path="/sys/fs/cgroup${group%/}"
    # Ancestors may impose limits even when the leaf reports "max".
    while [[ "$path" == /sys/fs/cgroup || "$path" == /sys/fs/cgroup/* ]]; do
        for name in cpu.max cpuset.cpus.effective memory.max memory.swap.max memory.current memory.peak memory.events; do
            if [[ -r "$path/$name" ]]; then
                printf '%s\n' "$path/$name"
                cat "$path/$name" || true
            fi
        done
        [[ "$path" == /sys/fs/cgroup ]] && break
        path=$(dirname "$path")
    done
}

snapshot() {
    date -u '+Memory snapshot %Y-%m-%dT%H:%M:%SZ' || true
    free -m || true
    # RSS is KiB per process, not the combined peak of concurrent compilers.
    # Show executable names only; command arguments may contain sensitive data.
    ps -eo pid,ppid,rss,comm --sort=-rss | head -n 11 || true
}

report() {
    echo '::group::Linux runner resources'
    uname -srmo || true
    cat /etc/os-release || true
    lscpu || true
    printf 'Available CPUs: '; nproc || true
    printf 'Installed CPUs: '; nproc --all || true
    taskset -pc "$$" || true
    free -m || true
    swapon --show || true
    ulimit -a || true
    cat /proc/self/cgroup || true
    report_cgroups
    echo 'Cgroup memory counters describe the enclosing group, including earlier job steps.'
    echo '::endgroup::'
}

monitor_memory() {
    local sleep_pid=
    trap 'if [[ -n "$sleep_pid" ]]; then kill "$sleep_pid" 2>/dev/null || true; wait "$sleep_pid" 2>/dev/null || true; fi; exit 0' TERM INT
    while true; do
        snapshot
        sleep 30 &
        sleep_pid=$!
        wait "$sleep_pid" || true
        sleep_pid=
    done
}

run_monitored() {
    if (( $# == 0 )); then
        echo 'Missing command for resource monitoring.' >&2
        return 2
    fi
    local monitor_pid status
    # Report effective settings after any caller-supplied taskset/ulimit.
    taskset -pc "$$" || true
    printf 'Virtual-memory limit (KiB, per process): '; ulimit -v || true
    monitor_memory &
    monitor_pid=$!
    cleanup() {
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo 'Timing maximum RSS is not an aggregate peak across concurrent compilers.'
    if [[ -x /usr/bin/time ]]; then
        # An explicit format omits the command arguments printed by time -v.
        if /usr/bin/time -f 'Elapsed: %e s\nUser CPU: %U s\nSystem CPU: %S s\nMaximum RSS (not aggregate): %M KiB\nMajor page faults: %F\nMinor page faults: %R\nSwaps: %W\nTimed command exit status: %x' "$@"; then
            status=0
        else
            status=$?
        fi
    else
        echo '/usr/bin/time unavailable; running without timing statistics.'
        if "$@"; then status=0; else status=$?; fi
    fi
    cleanup
    trap - EXIT INT TERM
    snapshot
    report_cgroups
    printf 'Test command exit status: %s\n' "$status"
    return "$status"
}

kernel_diagnostics() {
    free -m || true
    # Kernel logs may be unavailable, especially after runner loss.
    sudo -n dmesg --ctime | grep -Ei 'out of memory|oom|killed process|memory cgroup' | tail -n 80 || true
}

case "${1:-}" in
    report) report ;;
    run) shift; run_monitored "$@" ;;
    kernel) kernel_diagnostics ;;
    *) echo 'Usage: linux-resources.sh {report|run COMMAND [ARGS...]|kernel}' >&2; exit 2 ;;
esac
