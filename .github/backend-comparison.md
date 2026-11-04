# Build backend comparison

The `backends` job in `workflows/d.yml` runs on Linux with the same resolved
stable LDC as the existing CI. Edit `matrix.backend: [dub, redub]` to select
backends; there are no manual workflow inputs. Each backend fails independently
(`fail-fast: false`). Both jobs use at most two allowed CPUs to bound compiler
workers. No existing job or required-check name was changed.

To disable the entire comparison, change the `backend-dependencies` job's
`if: ${{ true }}` to `if: ${{ false }}`. Its dependent `backends` job is then
skipped automatically, including installations and builds. Set it back to
`true` to re-enable the comparison. Existing CI jobs are unaffected. Keep the
backend matrix nonempty; use this switch rather than an empty backend list.

Ninja is **disabled pending upstream fixes**, following the implementation
review. It is not currently a working third backend. Adding `ninja` to the list
fails explicitly rather than silently reporting untested code as passing.

## Scope and commands

The comparison resolves `library` dependencies once with DUB and distributes
`dub.selections.json` to all jobs. Each job fetches those sources before timing,
uses `--compiler=ldc2 --config=library --build=unittest`, and verifies that its
selected dependency versions match. The recipe supplies `mir_stat_test`, unittests,
debug mode, debug information, and LDC's `-lowmem`. `dub.sdl` is unchanged.

Redub resolves from installed package versions and rewrites the selections file;
it does not use that file as a DUB-style lock. Fresh runners fetch only the DUB
selection, and the helper compares parsed selection data after every operation,
failing if Redub chooses different versions. Redub omits unused configurations'
dependencies from its output; missing BLAS/benchmark entries are allowed, but
`mir-algorithm` and `mir-core` must both remain present. Do not reuse a populated package
cache without revisiting this constraint.

The helper script exists to share timing, failure propagation, source restoration,
and diagnostic capture between local validation and CI:

* DUB: `dub test`. The default package build mode remains the baseline.
* Redub: `DFLAGS='-unittest -g -d-debug -lowmem' redub test --build-strategy=full-once`.
  The workflow installs the
  pinned upstream Linux x86-64 binary from release **v1.35.1**. See the
  [upstream installation/help documentation](https://github.com/MrcSnm/redub)
  and [CLI options](https://github.com/MrcSnm/redub/blob/v1.35.1/source/redub/cli/dub.d).

Cold/no-change/incremental measurements include compilation, linking, **and test
execution**. Both backends use their real test command; compilation alone never
counts as passing. Redub's test command has no build-only mode, and local
validation found that both `redub build` and `redub test` omit this SDL build
type's `buildOptions`. The Redub-only `DFLAGS` compatibility setting explicitly
restores the same LDC unittest/debug/debug-info/low-memory flags as DUB, without
editing the recipe. Keep it in sync with the recipe's ordinary unittest type;
do not extend it to other configurations without validating their flags.
A separate execution-only row reruns `./mir-stat-test-library`,
the test binary produced by both commands, to expose test overhead. Do not simply
subtract this warmed execution time from the other rows.

Redub v1.35.1's `incremental-parallel` test strategy failed locally with duplicate
generated `main` and `__ModuleInfo` symbols. The comparison therefore explicitly
uses `full-once`, retaining the full tests. This measures package-level reuse,
not Redub's advertised per-module incremental test compilation. Revisit the
strategy only after upstream fixes and equivalent-test validation.

The validated Redub executable reports 77 tested modules versus DUB's 75.
Symbol comparison found no DUB unittest symbols missing from Redub, but found
additional dependency unittest symbols, including `mir.algebraic` and
`mir.format`. Thus the full project suite is retained, but test scope is not
strictly identical; include this extra work when interpreting timings.

Cold means a fresh CI checkout/runner with fetched package sources, no restored
compiled artifacts, and no preceding project build. The immediate no-change
build exercises backend caches. Compile-time flag assertions are added to
`source/mir/stat/distribution/normal.d` before the cold build. The incremental
build appends a compile-time message; its content hash must change and the
message must appear in compiler output. An EXIT trap restores the original
bytes even after a failed build. The full tests run with the probe present,
verifying the requested test/debug versions. No numerical implementation
changes are made.

Logs and wall-clock timings are uploaded as `backend-<backend>-logs` and timings
are added to the job summary, including failures. Inspect compilation logs to
inspect package reuse and full rebuilds; neither active path demonstrates
module-level incremental test compilation. Filesystem caches,
parallel scheduling, and shared GitHub runners affect measurements; these are
initial observations, not a controlled performance ranking.

## Ninja findings and re-enablement

DUB **v1.40.0** source and the bundled **v1.42.0** CLI inspected do not contain a
Ninja generator. Development revision `1fca61757450c02146050db08591101f873537b3` does.
Its [generator implementation](https://github.com/dlang/dub/blob/1fca61757450c02146050db08591101f873537b3/source/dub/generators/ninja.d)
has correctness limitations for this comparison:

* `dub generate ninja --compiler=ldc2 --config=unittest --build=unittest`
  generates a test executable target, but does not translate recipe build options
  into unittest/debug/debug-info compiler flags. It retains version identifiers
  and remaining raw dflags, which is insufficient to run the same tests.
* Its regeneration command is just `dub generate ninja`: the compiler,
  configuration, and build-type arguments are lost on regeneration.
* Local LDC 1.43.0 successfully compiled and linked the generated graph, despite
  the missing flags. A successful Ninja build therefore does not establish that
  the tests were compiled or executed.

Do not patch generated files or change the library recipe to hide these issues.
After upstream fixes land, install `ninja-build`, pin a DUB version containing
them, generate using the explicit options above, and inspect `ninja -t commands` for matching flags and
dependency handling. Build the generated test executable target with Ninja and
**execute it explicitly**; Ninja's build target alone does not run tests. Add
that verified path to the helper and enable `ninja` in the matrix. Include graph
generation in cold timing, report its cost, and check regeneration keeps options.

## Existing coverage retained

`d.yml` retains Linux/Windows stable and newer beta DMD/LDC jobs for general
coverage, release tests, and DIP1000 lifetime tests. Linux also runs OpenBLAS
coverage and the stable-LDC optimized slow numerical subpackage. Stable compiler
jobs test benchmark helpers/cases and smoke-test release benchmark CLI modes.
Ubuntu 26.04 retains GDC 15 standard tests with per-file compilation; optimized
GDC remains blocked by the previously documented mir-algorithm issue. There is
no macOS job in this checkout.

`docs.yml` retains DMD Ddoc validation, HTML generation, and LDC 1.43.0 Meson/Ddoc
example tests. No documentation workflow changes are needed.

The recipe additionally defines slow/extreme numerical, DIP1008, combined DIP,
benchmark, and multiple BLAS/LAPACK/MKL configurations. These are not all existing
CI jobs. The new comparison intentionally covers only ordinary unittests; it
does not establish Redub/Ninja compatibility with those specialized settings,
coverage output, subpackages, or native linking. Existing DUB coverage remains
the authority for configurations it already exercises.

## Validation

Local WSL Ubuntu checks used LDC 1.43.0, DUB 1.42.0, and Redub 1.35.1:

* YAML parsed successfully; a structural comparison confirmed every original
  job remained unchanged. New Bash blocks and the helper passed syntax checks.
* DUB and Redub (with the compatibility settings above) passed their full test
  commands, no-change runs, separate executable runs, and incremental runs.
* Both no-change runs reported cached targets. Both incremental runs emitted
  the required compiler marker and passed the test/debug flag assertions.
* Source restoration was checked byte-for-byte after successful runs and an
  earlier failed Redub probe.
* Development DUB generated and linked a Ninja test target, but omitted required
  flags; the disabled helper branch was verified to fail explicitly.

Local iterations reused dependency artifacts and ran on a Windows-mounted
filesystem; their timing numbers are not fresh-run performance evidence. The
clean GitHub runner installs, shared-selection artifact transfer, summary upload,
and dynamically resolved compiler version still need a GitHub Actions run.

## Follow-up

Verify results on GitHub runners before widening configurations or platforms.
Resolve Ninja correctness upstream first. Evaluate persistent caching separately;
any later cache must distinguish backend, OS, compiler, configuration, and
dependency selections, and a cache miss must still produce a correct build.
