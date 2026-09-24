# Statistical benchmarks

Run from the repository root:

```sh
dub run mir-stat:benchmarks --build=release --compiler=ldc2 -- --case=correlation --iterations=10000 --size=1000
```

Use `--case=all` (the default) to run skewness, kurtosis, covariance, and
correlation. Use `--help` to list options. DMD is also supported with
`--compiler=dmd`. Use a release build for performance measurements.

Each case compares every algorithm in its enumeration, using `double` and
`Summation.fast`. Iterations and size default to 10000 and 1000, respectively.
Skewness requires at least three observations, kurtosis four, and covariance
and correlation two. Output reports the mean computed value and total elapsed
time for each algorithm, excluding random input generation and preparation.
Correlation inputs are sample-standardized outside timing.

This replaces the old `unittest-perf` configuration/build. Timing methodology
is otherwise unchanged: algorithms run in a fixed order with separately
generated random inputs. Results are exploratory measurements, not paired
comparisons or performance assertions.

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
