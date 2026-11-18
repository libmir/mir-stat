# Histogram construction benchmark

From the repository root, build and run one case with:

```sh
dub run --root=benchmarks/histogram --build=release --compiler=ldc2 -- --allocation=gc --size=1024 --iterations=10000
```

Use `--allocation=rc` or `--allocation=custom` for reference-counted storage or
Mallocator. Add `--variable-axis` for variable boundaries. The default is a
regular axis. DMD is also supported with `--compiler=dmd`. Run each case in a
fresh process and repeat runs to assess timing variability.

The benchmark creates short-lived histograms with four bins over [0, 4).
Inputs repeat 0, 0.25, ..., 3.75. All cases use uint counts. Input generation
and reference-counted boundary allocation occur before timing. Each case
performs 128 untimed warmup constructions, checking every bin against its
expected count, then requests a collection before measurement.

Construction time includes axis construction, count allocation, initialization,
insertion, the external observer call, checksum calculation, and ordinary
reclamation. RC destruction and custom disposal occur within each iteration.
Automatic GC retains its normal policy. The GC case also requests a final
collection, reported separately. Total time is the sum of those two phases.
A collection request does not guarantee that conservative GC finds every
allocation unreachable. GC counters and times cover the process; run without
other workloads in the same process. Reported GC time is already included in
elapsed time and must not be added again.

The observer is built as a separate static library. The runner sees only its
external declaration. Do not enable LTO: the opaque call is needed to prevent
the compiler from eliminating the allocation under measurement. It borrows
the count pointer for the duration of the call and never retains it.

Compare revisions within the same case using identical compiler versions,
flags, dependency versions, sizes, and iteration counts. Record CPU affinity
and runtime GC options as well. These measurements do not establish a general
ranking of allocation strategies and do not cover retained histograms.

For a quick check, use sizes 0, 1, 16, and 1024 with 100 iterations for each
allocation strategy and each axis type. Zero iterations and invalid allocation
names are rejected. Input size is limited to uint.max, and checksum overflow
is rejected before allocation. Ordinary unit tests do not run this benchmark.
