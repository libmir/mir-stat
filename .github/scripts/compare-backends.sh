#!/usr/bin/env bash
# Run only in a fresh, disposable checkout with dependencies already fetched.
set -euo pipefail

case "${BACKEND:?Select BACKEND in the workflow matrix}" in
  dub) tests=(dub test) ;;
  redub)
    redub --version
    # v1.35.1's per-module test strategy emits multiple generated mains.
    # Keep the full suite; use its supported single-invocation build strategy.
    # It also drops this SDL build type's buildOptions. Supply the same LDC
    # flags explicitly for the root package; retain recipe versions/dependencies.
    tests=(env 'DFLAGS=-unittest -g -d-debug -lowmem' redub test --build-strategy=full-once)
    ;;
  ninja)
    echo 'Ninja is disabled: released DUB lacks the generator; development DUB drops unittest/debug build options.' >&2
    echo 'See .github/backend-comparison.md before enabling this backend.' >&2
    exit 1
    ;;
  *) echo "Unknown backend: $BACKEND" >&2; exit 1 ;;
esac

# Match recipe flags and versions; scheduling/cache granularity is backend-specific.
args=(--compiler=ldc2 --config=library --build=unittest)
mkdir -p backend-results
summary="$PWD/backend-results/summary.md"
module=source/mir/stat/distribution/normal.d
backup=$(mktemp)
cp "$module" "$backup"
cp dub.selections.json backend-results/dub.selections.json
cleanup() {
  cp "$backup" "$module"
  rm -f "$backup"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$summary" >> "$GITHUB_STEP_SUMMARY"
  fi
}
trap cleanup EXIT
cat > "$summary" <<EOF
### $BACKEND — Linux LDC, library / unittest

| Operation | Wall seconds | Exit status |
| --- | ---: | ---: |
EOF

measure() {
  local label=$1 status
  shift
  echo "::group::$BACKEND: $label"
  # Keep failure status through tee; publish even failed operation timings.
  set +e
  /usr/bin/time -f '%e' -o "backend-results/$label.seconds" "$@" 2>&1 | tee "backend-results/$label.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "| $label | $(tail -n 1 "backend-results/$label.seconds") | $status |" | tee -a "$summary"
  echo "::endgroup::"
  if (( status != 0 )); then return "$status"; fi
  # Redub writes only used packages; DUB also resolves other configurations.
  # Every selected version must match the shared resolution, and both ordinary
  # library dependencies must remain present. Ignore unused BLAS/benchmark entries.
  python3 - <<'PY'
import json
from pathlib import Path
expected = json.loads(Path('backend-results/dub.selections.json').read_text())['versions']
actual = json.loads(Path('dub.selections.json').read_text())['versions']
assert {'mir-algorithm', 'mir-core'} <= actual.keys(), 'Missing library dependencies'
assert actual.items() <= expected.items(), f'Dependency versions changed: {expected} -> {actual}'
PY
}

# Fresh CI jobs have no compiled package artifacts and use no persistent cache.
# Both test commands compile, link AND run. Also time the already-built test
# executable so readers can judge execution overhead without subtracting timings.
# Reject missing test/debug modes before any timing can be mistaken for success.
cat >> "$module" <<'D'

version (unittest) {} else static assert(0, "CI requires unittests");
version (mir_stat_test) {} else static assert(0, "CI requires mir_stat_test");
debug {} else static assert(0, "CI requires debug mode");
D
measure cold-build-and-test "${tests[@]}" "${args[@]}"
measure no-change-build-and-test "${tests[@]}" "${args[@]}"
measure test-execution-only ./mir-stat-test-library

# Change content, not just mtime. A compiler-emitted marker proves this module
# was compiled again, rather than reusing the cold-build object.
before=$(sha256sum "$module")
cat >> "$module" <<'D'

pragma(msg, "mir-stat CI incremental compilation verified");
D
if [[ "$before" == "$(sha256sum "$module")" ]]; then
  echo 'Incremental probe did not change source content' >&2
  exit 1
fi
measure incremental-build-and-test "${tests[@]}" "${args[@]}"
grep -Fq 'mir-stat CI incremental compilation verified' backend-results/incremental-build-and-test.log
cat >> "$summary" <<'EOF'

Cold/no-change/incremental times include compilation, linking, and test execution.
The execution-only row reruns the already-built test binary. Dependencies were
downloaded before timing; no compiled artifacts were restored. DUB uses its
default package build mode; Redub uses full-once (package-level reuse).
Shared runners and different scheduling make these exploratory observations,
not controlled performance benchmarks. See the attached logs for rebuild scope.
EOF
