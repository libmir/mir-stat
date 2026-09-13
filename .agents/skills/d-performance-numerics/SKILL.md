---
name: d-performance-numerics
description: Investigate D performance, allocation, compile-time cost, code generation, SIMD/vectorization, benchmarking, or numerical correctness and floating-point behavior. Use only when performance or numerical behavior is central to the task; do not load for ordinary D implementation work.
---

# D Performance and Numerics

Measure or define the relevant contract before changing code. Do not trade away correctness, safety, genericity, or maintainability for an unmeasured optimization.

## Performance triage

Classify the dominant cost first: compiler wall time/template growth/CTFE; runtime CPU; allocation/GC; code size; cache/memory-layout behavior; missed inlining/vectorization; or checking overhead.

Investigate in this order when applicable:

1. accidental allocation or materialization;
2. asymptotic or repeated work;
3. copies, conversions, and temporary objects;
4. memory layout, locality, and stride-one traversal;
5. template-instantiation or CTFE growth;
6. generated code, inlining, and vectorization;
7. instruction-level changes.

Use diagnostics supported by the installed D compiler rather than assuming a fixed version. Relevant DMD facilities may include `-vgc` for GC allocations, `-ftime-trace` for compile-time behavior, `-vtemplates=list-instances` for template-instantiation analysis, and `-vasm` for generated assembly. With LDC, use the corresponding optimization/codegen/IR facilities when appropriate. Verify options with the installed compiler.

Benchmark with stable inputs and comparable compiler version, flags, target CPU, build mode, and environment. Prefer the project's existing benchmark harness. Run enough iterations to distinguish signal from noise, and report both the baseline and changed result. Do not retain substantial complexity for an unmeasured or unstable gain.

For specialized fast paths, state the exact precondition and preserve a correct general path unless the API intentionally narrows its accepted inputs.

## Numerical contract

Before changing a numerical algorithm, establish its supported domain and types, expected accuracy, boundary behavior, NaN/Inf/signed-zero behavior when observable, overflow/underflow/subnormal concerns, accumulator precision, and whether evaluation order or bitwise reproducibility is promised.

Do not use exact equality for general computed floating-point results unless exactness is an invariant. Select absolute, relative, mixed, or ULP-based comparisons according to the algorithm and scale. Do not hide errors behind arbitrary loose tolerances.

For reductions, dot products, statistics, and parallel/vectorized computations, account for reassociation changing low-order bits. Distinguish mathematical correctness, acceptable floating-point error, and bitwise reproducibility.

Tests should target relevant numerical regimes: zero and signed zero, tiny and large finite values, NaN/Inf where accepted, regime boundaries, degenerate or empty inputs, ill-conditioned cases, and deterministic seeds/invariants for randomized algorithms. Avoid flaky one-sample statistical assertions.

## Completion

For performance work, record the bottleneck and before/after evidence. For numerical work, record the intended error and edge-case contract. Re-run representative configurations and relevant compiler/build modes. If results suggest an optimizer/compiler miscompile or compiler-version regression rather than an algorithmic issue, switch to `d-compiler-investigation`.
