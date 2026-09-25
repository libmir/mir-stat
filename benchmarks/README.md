# Statistical benchmarks

Run from the repository root:

```sh
dub run mir-stat:benchmarks --build=release --compiler=ldc2 -- --function=correlation --transform=standardize --algorithms=all --iterations=10000 --rounds=10 --warmup=10 --size=1000 --seed=5489
```

Use `--function=all` (the default) to run skewness, kurtosis, covariance, and
correlation. Use `--help` to list options. DMD is also supported with
`--compiler=dmd`. Use a release build for performance measurements.

Each comparison uses `double` and
`Summation.fast`. Iterations and size default to 10000 and 1000, respectively.
`--rounds` defaults to 10 measured rounds; `--warmup` defaults to 10 untimed
iterations once per function before its first measured round. `--iterations`
is per algorithm per measured round, so the defaults execute 100000 measured
calls and 10 warm-up calls per selected algorithm. Set `--rounds=1 --warmup=0`
for a single round without warm-up. Rounds and iterations must be positive;
zero warm-up is allowed.

Skewness requires at least three observations, kurtosis four, and covariance
and correlation two. Each result records the round's mean computed value and
total elapsed time, excluding input generation, preparation, reference checks,
and warm-up. Each algorithm accumulates native clock ticks across calls before
converting the round total to `Duration`, avoiding per-call rounding to 100 ns.
Warm-up still runs correctness checks.
Inputs are generated once per iteration and shared through read-only views
across all algorithms. `--seed` selects a local MT19937-64 engine's seed
(default 5489); it does not use or modify the thread-local random engine.
Running a function alone or as part of `all` uses the same seed. The generator
runs continuously through warm-up and all measured rounds; it is not reseeded
at round boundaries. Reproduction therefore requires the same seed, warm-up,
iteration count, compiler, build and dependency versions. Measured values
are reproducible under those conditions; elapsed times are not. Bitwise
agreement across platforms is not promised.

Select the input transformation independently of the algorithms:

- `--transform=none` (default): leave generated inputs unchanged.
- `--transform=center`: subtract each input's mean, without scaling variance.
- `--transform=standardize`: calculate sample z-scores for each input.

Select `--algorithms=all` (default) or a comma-separated list, for example:

```sh
dub run mir-stat:benchmarks --build=release --compiler=ldc2 -- --function=correlation --transform=standardize --algorithms=twoPass,assumeStandardized
```

`all` runs every algorithm valid for the transformation. `assumeZeroMean`
requires `center` or `standardize`; `assumeStandardized` requires
`standardize` and is only a correlation algorithm. Explicitly requesting an
incompatible or unknown algorithm fails before any measurements start.
With `--function=all`, an explicit algorithm list must be valid for every
function. Duplicates, empty list items, and combining `all` with names are
rejected. The first round uses enumeration order, regardless of list order.
Each subsequent round cyclically shifts the first selected algorithm by one;
disabled algorithms take no positions. The order stays fixed within a round.
Every algorithm takes each position once per complete cycle of rounds, which
reduces position bias but does not eliminate all order or carryover effects.
Results list round and execution position alongside function and transformation.

The general inputs use `x = 2 + 2*z1` and `y = -1 + z1 + 3*z2`, where `z1`
and `z2` are independent standard normal draws. Preparation costs are excluded
when centering or standardizing: these comparisons model data already transformed,
not an end-to-end speedup from choosing a shortcut on arbitrary data.

Preparation and reference calculations run outside timing. Every result is
checked against the three-pass (skewness/kurtosis) or two-pass
(covariance/correlation) implementation using precise summation. Checks reject
nonfinite values and errors exceeding `1e-10 + 1e-10 * abs(reference)`. This
mixed tolerance accommodates rounding and near-zero statistics for these
normal inputs; it is not a general accuracy guarantee. A failed check stops
the run with a nonzero exit status.

This replaces the old `unittest-perf` configuration/build. The shared-input
loop, centering, and reference checks change cache state and the timing
baseline, so old timings are not directly comparable. Round timings are retained
individually; the runner does not perform significance tests or claim speedups.

## CSV output

Use `--format=csv` for one row per algorithm per measured round. For example:

```sh
dub run --quiet mir-stat:benchmarks --build=release --compiler=ldc2 -- --function=correlation --transform=standardize --algorithms=twoPass,assumeStandardized --rounds=10 --warmup=10 --iterations=10000 --format=csv > results.csv
```

`--quiet` suppresses DUB progress messages. Text is the default format. CSV
includes function, transform, algorithm, round, execution position, iterations,
input size, seed, warm-up count, mean value, and total elapsed nanoseconds.
Round and position are one-based. Nanoseconds are the output unit, not a promise
of clock resolution. Means use 17 significant digits for double round trips.
Rows are emitted in enumeration order within each round; use `position` to
recover execution order.

Pair algorithms within the same function, transform, seed, warm-up, input size,
iteration count, and round. Each such pair used identical inputs. Repeated
rounds from one process are not independent process-level replications; retain
that distinction in subsequent statistical analysis. Output occurs after a
function's measured rounds, keeping printing out of the round sequence.

## Layout

- `source/mir/stat/benchmark/cases.d` defines cases and returns results.
- `source/mir/stat/benchmark/runner.d` handles command-line options and output.
- `source/mir/stat/benchmark/report.d` formats per-round CSV rows.
- `mir.math.internal.benchmark` remains the shared helper module in the library.

Case interfaces use `package(mir.stat)` visibility. The executable depends on
the local parent package; benchmark code and `mir-random` are not added to the
normal library configuration.

## Tests

```sh
dub test mir-stat:benchmarks --compiler=dmd
dub test mir-stat:benchmarks --compiler=ldc2
dub test --config=unittest-benchmark --compiler=dmd
```

The subpackage tests run small cases and check results and input validation,
without asserting timing thresholds. The parent configuration tests the shared
benchmark helpers alongside the normal library tests.

CI runs these tests with stable DMD and LDC on Linux and Windows. Short release
runs exercise all functions and applicable algorithms with untransformed,
centered, and standardized inputs, including CSV output. These are correctness
checks; CI does not compare timings or enforce performance thresholds.
