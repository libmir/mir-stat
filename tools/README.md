# Documentation and example tests

The `Documentation` GitHub Actions workflow builds HTML on pull requests,
pushes to `master`, and manual runs. Download `documentation-html` from a
successful run and open `index.html`. It does not publish a website.

The workflow has three independent jobs: the existing DUB/DMD Ddoc validation,
the HTML preview, and Meson library/example tests. The Ddoc job retains
`dub build --build=docs-check --compiler=dmd` with `dmd-latest`. A failure
in one job does not prevent the other jobs from running.

On Linux, install LDC and ldmd2 **1.43.0**, Git, g++, Ninja, and Python 3.9+ with
pip. From the repository root, use a fresh output directory:

```sh
python3 tools/build_docs.py build/docs
python3 -m http.server --directory build/docs/site 8000
```

The driver clones build inputs and overlays tracked working-tree edits,
including staged additions and deletions. It retains logs, inputs, isolated
Python dependencies, and a version manifest without modifying the checkout.
Meson 0.61.1, semver 2.13.0, and the circle-dlang asset revision are pinned.
HTML compilation treats warnings as errors. Release tags are required by
the template; both Meson jobs fetch full history and tags.

mir-algorithm 3.22.4 and mir-core 1.7.4 are pinned to exact Git commits.
The Meson build follows the default DUB library configuration. Optional BLAS,
benchmark, slow-numerics, and lifetime tests remain available through DUB.

## Library and extracted-example tests

The `Meson D and documentation example tests` job in the same workflow runs
the D library suite and extracted documentation examples with `mir_stat_test` and the DIP1008 compiler option:

```sh
python3 tools/build_docs.py build/meson-tests --tests-with-examples
```

Internal documentation examples are compiled in generated modules within their
source package, so package-restricted APIs remain accessible without making them
public. The extraction adapter leaves public API examples in external modules,
preserves the extracted bodies and line directives, and does not skip examples.
Both CI systems use `prepare_doc_examples.py` to wire this adapter into the Meson
template. CircleCI defines the build job locally because the published Mir orb
has no extraction hook; it retains the orb's container and `web` workspace for
the existing tag-only upload job.

The template builds HTML alongside the extracted example executables.
Every test configured by Meson must run successfully. The driver requires
the library test and at least one extracted-example suite, rejects skipped
tests, and does not hard-code the example count. Build and test logs are
uploaded as `meson-test-logs` even on failure.

The driver creates the pinned extractor's default `out` directory before
starting parallel example extraction. Single-file results go to stdout, but
the extractor still creates this directory and otherwise races on first use.
All build modes retain two Ninja build jobs. The shared Meson configuration uses a timeout multiplier
of 40, allowing 20 minutes for the full numerical library suite instead of the
template's default 30 seconds in both GitHub Actions and CircleCI. Tests still
fail if they exceed that limit.

Use `--tests-only` for library tests without HTML or extracted examples.
Run `python3 tools/test_build_docs.py` for driver regression checks.

The existing compiler matrix and CircleCI release publishing remain separate.
An HTML build does not verify every link or the external online example runner.

## Slow numerical references

Run `dub test mir-stat:slow-numerics --build=unittest-optimized` for only the
slow numerical tests with DMD or LDC. The focused package imports the five
modules containing these tests and compiles their required mir-stat implementation,
without building or running
unrelated ordinary tests. Add `--compiler=ldc2` to select LDC. Optimization and
inlining retain assertions, contracts, and bounds checks. Omit the build option
for an unoptimized run.

The slow suite includes the 256- and 512-bit centered binomial log-mass sweeps,
512-bit log-factorial and Poisson log-mass sweeps, and the existing large-count
binomial and negative-binomial tests. These use `mir_stat_test_slow_numerics`
independently of `mir_stat_test`; their reference generators preserve the guards.
Ordinary tests retain the smaller internal sweeps, public API references at all
four widths, and the other Fp boundary and regression tests.

Use `dub test --build=unittest-slow-numerics` to run ordinary and slow tests
together through the full library build. When adding slow tests in another
module, also import it in `tests/slow_numerics/suite.d`.

CI runs the focused slow numerical suite once per pull request or push to
`master`, in the Linux `ldc-latest` job. Ordinary coverage, release, and lifetime
tests retain their full compiler and operating-system matrices. The slow run
keeps DUB build and test markers in its logs to separate compilation from execution.
