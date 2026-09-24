# Statistical benchmarks

Run from the repository root:

```sh
dub run mir-stat:benchmarks --build=release --compiler=ldc2 -- --function=correlation --transform=standardize --algorithms=all --iterations=10000 --size=1000 --seed=5489
```

Use `--function=all` (the default) to run skewness, kurtosis, covariance, and
correlation. Use `--help` to list options. DMD is also supported with
`--compiler=dmd`. Use a release build for performance measurements.

Each comparison uses `double` and
`Summation.fast`. Iterations and size default to 10000 and 1000, respectively.
Skewness requires at least three observations, kurtosis four, and covariance
and correlation two. Output reports the mean computed value and total elapsed
time for each algorithm, excluding random input generation and preparation.
Inputs are generated once per iteration and shared through read-only views
across all algorithms. `--seed` selects a local MT19937-64 engine's seed
(default 5489); it does not use or modify the thread-local random engine.
Running a case alone or as part of `all` uses the same seed. Values are
reproducible with the same compiler, build and dependency versions; elapsed
times are not, and bitwise agreement across platforms is not promised.

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
rejected. Algorithms run in their enumeration order, regardless of list order.
The output labels each function, transformation, and selected algorithm.

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
baseline, so old timings are not directly comparable. Algorithms still run
in a fixed order, without warm-up or repeated measurement rounds. Results
are exploratory measurements, not statistical evidence of a speedup.

## Layout

- `source/mir/stat/benchmark/cases.d` defines cases and returns results.
- `source/mir/stat/benchmark/runner.d` handles command-line options and output.
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
