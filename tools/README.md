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

Run `dub test --build=unittest-slow-numerics` to include the expensive internal
Fp reference sweeps: 256- and 512-bit centered binomial log masses, and 512-bit
log-factorials and Poisson log masses. These use `mir_stat_test_slow_numerics`
in addition to `mir_stat_test`; their reference generators preserve the guards.
Ordinary tests retain the smaller internal sweeps, public API references at all
four widths, and the other Fp boundary and regression tests.
