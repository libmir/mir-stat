# Linear algebra used by mir-stat

This package provides the matrix operations used by mir-stat, including the
building blocks for Mahalanobis distance. It is currently **internal**: its
operations are available to `mir.stat` and its subpackages, not to application
code as a general-purpose BLAS API. The shared `TrianglePolicy` and
`DiagonalPolicy` enums are public exceptions: they can be imported through
this package, `mir.stat.distance`, or `mir.stat`. Mahalanobis functions use the
same `TrianglePolicy`; there is no separate distance-specific triangle enum.

For application-level distance calculations, use `mir.stat.distance`.
This README explains the internal package's capabilities and design choices.
[package.d](package.d) and the individual modules contain the detailed function
contracts and executable examples.

## What is supported

| Operation | Inputs | Result |
| --- | --- | --- |
| `mtimes` | Matrix × matrix, matrix × vector, vector × matrix | Matrix or vector |
| `mtimes` | Vector × vector | Dot-product value |
| `mtimesTriangular` | Triangular matrix on the left | Matrix or vector |
| `mtimesTriangularRight` | Triangular matrix on the right | Matrix or vector |
| `quadraticForm` | Vector and general or symmetric matrix | `xᵀ A x` |
| `quadraticFormWithWorkspace` | Same inputs and a separate vector workspace | Compensated or explicitly fast `xᵀ A x` |
| `choleskyDecompose` | Real symmetric positive-definite matrix | Upper or lower Cholesky factor |
| `choleskySolve` | Existing Cholesky factor and vector or matrix right-hand side | Solution of `A X = B` |

Multiplication accepts matching floating-point or complex element types.
Complex products do **not** conjugate either operand. Quadratic forms,
Cholesky decomposition, and Cholesky solves currently accept matching
`float`, `double`, or `real` element types. There is no implicit mixed-type
promotion.

Inputs use Mir ndslices. Borrowed and reference-counted slices, read-only
inputs, computed inputs, and general strides are supported where the operation
only reads them. Output and in-place arguments must provide writable storage.
Transposing a slice changes its logical shape and indexing without copying.

`mtimes` provides `C = A B`; it does not yet expose full GEMM scaling
`C = α A B + β C`. Quadratic forms do not require positive definiteness and
can legitimately return negative results. Cholesky decomposition does require
positive definiteness; no regularization or pseudoinverse is performed.

## Choosing storage ownership

Input ownership and result ownership are independent. Passing an RC matrix
does not force an RC result.

| Style | Example | Who manages storage? |
| --- | --- | --- |
| GC | `mtimes(a, b)` | D's garbage collector |
| Reference counting | `rcMtimes(a, b)` | Released with the last owning reference |
| Custom allocator | `makeMtimes(allocator, a, b)` | Caller disposes `result.field` with that allocator |
| Caller output | `mtimesInto(a, b, output)` | Caller supplies and retains the buffer |

The same naming applies to triangular products, Cholesky decomposition, and
Cholesky solves. Two-vector products and quadratic forms return a numerical
value, without allocating result storage. Custom-allocator factories release
their allocation if computation throws before returning the result.

Allocated GC/custom results are contiguous pointer-backed slices; RC results
are contiguous RC-backed slices. Their types do not inherit the input's
strides or ownership. A custom result's pointer-backed type does not make its
storage GC-managed.

For internal code that wants only one allocation strategy, import `gc`, `rc`,
or `custom` from this package. These convenience modules expose the existing
names and overloads; they do not add wrappers. Import views, policies,
quadratic forms, and caller-output/in-place operations separately as needed.

## Matrix structure and views

`SymmetricView` and `TriangularView`, constructed with `symmetricView` and
`triangularView`, attach structural information without copying elements.

- `TrianglePolicy.upper` or `.lower` selects the stored **logical** triangle.
- A symmetric view reflects that triangle across the diagonal. For complex
  elements this is symmetry, not conjugate/Hermitian symmetry.
- A triangular view supplies zeros outside its selected triangle.
- `DiagonalPolicy.unit` supplies ones on a triangular matrix's diagonal
  without reading stored diagonal entries. `.nonUnit` reads the stored values.

`DiagonalPolicy` is not a diagonal-matrix representation. A standalone
`DiagonalView` is outside the current scope.

Declare triangular structure once, then use the usual multiplication function:

```d
auto upper = triangularView!(TrianglePolicy.upper)(a);
auto left = mtimes(upper, b);  // Uses the left triangular product.
auto right = mtimes(b, upper); // Uses the right triangular product.
auto lowerProduct = mtimes(upper.transposed, b);
```

`a` must be square; `b` is an ordinary matrix or vector ndslice of matching
element type and conforming dimensions. The view does not copy the matrix or
inspect its entries to discover structure. Entries outside the selected
triangle are ignored, even if they contain NaNs. A transposed view reverses
its triangle policy automatically.

The same dispatch is available for every result-storage choice:

```d
auto owned = rcMtimes(upper, b);
auto allocated = makeMtimes(allocator, b, upper);
scope(exit) allocator.dispose(allocated.field);
mtimesInto(upper, b, output); // Reuses separate caller-provided storage.
```

Custom allocation requires the usual allocator and `dispose` imports. Select
an implicit unit diagonal with
`triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(a)`; stored diagonal
entries are then ignored as well. The explicitly named `mtimesTriangular` and
`mtimesTriangularRight` families remain available for callers supplying the
raw matrix and structural options directly.

Symmetric views are accepted by quadratic forms and Cholesky decomposition.
Triangular views are accepted by multiplication when the other operand is an
ordinary ndslice. Combining two structural views is not currently an overload.
These descriptors do not certify positive definiteness or identify a matrix
as a valid Cholesky factor.

Views are read-only through this interface. They retain an RC owner when the
input has one; views over borrowed storage remain borrowed. Their `storage`
property exposes read-only square indexing. For dense inputs this includes
unused stored entries; for packed inputs absent entries read as zero.
Their `transposed` property retains ownership and reverses the triangle policy.
The caller must keep borrowed storage and computed-iterator dependencies alive.

## Packed triangular and symmetric storage

Packed stairs store `n * (n + 1) / 2` entries rather than a full square matrix.
Use `stairs!"-"(n)` for upper rows of lengths `n, n-1, ..., 1`, or
`stairs!"+"(n)` for lower rows of lengths `1, 2, ..., n`.

```d
import mir.ndslice.topology: stairs;
auto packed = data.stairs!"-"(n);
auto result = mtimes(packed, x); // Bare stairs mean triangular, stored diagonal.
auto same = mtimesTriangular(packed, x);
auto right = mtimes(x, packed);
auto unit = triangularView!(DiagonalPolicy.unit)(packed);
auto withUnitDiagonal = mtimes(unit, x);
auto symmetric = symmetricView(packed); // Same storage, different meaning.
auto q = quadraticForm(x, symmetric);
```

Direct calls and views share implementations across GC, RC, custom-allocator,
and caller-output variants. No dense input temporary is created. The view
retains the packed iterator and its ownership; transpose remains a view.
The packed direction determines the triangle, so it is not repeated as an
option. Symmetric packed views also work with Cholesky decomposition, producing
a dense factor. Packed output/in-place factorization is not provided.

This support accepts the contiguous outer slice returned by the packed-data
`stairs` overload. A sequence of row views produced by `stairs` on a dense
matrix is a different representation; use the original matrix with a structural
view instead. Arbitrarily reordered or incomplete stairs rows are not a square
packed matrix. Construction of pointer-backed stairs in mir-algorithm currently
requires `@system`; arithmetic attributes depend on the underlying iterator.

## Quadratic-form workspace and fast arithmetic

The existing `quadraticForm(x, a)` remains allocation-free and compensated by
default. `quadraticFormWithWorkspace(x, a, work)` additionally stores `a * x`
in a separate writable vector and retains compensated reductions by default.
Workspace must be disjoint from the inputs and have the same length and element
type as `x`.

```d
import mir.math.sum: Summation;
auto q = quadraticFormWithWorkspace!(Summation.fast)(x, a, work);
auto symmetricQ = quadraticFormWithWorkspace!(Summation.fast)(
    x, symmetricView(a), work);
```

Explicit fast mode permits GEMV plus DOT for general matrices, SYMV plus DOT
for dense symmetric views, and SPMV plus DOT for packed symmetric views. Native
D implementations handle unsupported types/layouts and builds without BLAS.
Fast mode does not promise compensated accuracy or identical rounding across
backends. It is a choice for callers willing to trade that accuracy for speed,
not a change to the default numerical contract. Results are not clamped.

## Factorization and reuse

Upper factors satisfy `A = Uᵀ U`; lower factors satisfy `A = L Lᵀ`.
The selected input triangle is read, and successful decomposition zeros the
unused output triangle.

`choleskyDecomposeInto` preserves the input and writes a separate buffer.
`choleskyDecomposeInPlace` reuses writable input. Both throw on numerical
failure. `tryCholeskyDecomposeInPlace` instead returns zero on success or a
one-based failed pivot index. Failure can leave a partial factorization that
must not be used as a valid factor.

`choleskySolve`, `choleskySolveInto`, and `choleskySolveInPlace` reuse a factor
without forming an inverse. Each column of a matrix right-hand side is one
system. The caller is responsible for supplying a valid factor; solves do not
repeat the decomposition's positive-definiteness checks.

## Native D and BLAS-accelerated implementations

**No BLAS installation is required for the default build.** Multiplication,
triangular operations, factorization, and solves have native D implementations.
They use ordinary arithmetic loops; the compiler may vectorize those loops.
The benchmark configuration for this build is named `native-d`.

With the optional BLAS/LAPACK configuration, eligible layouts and types use
native routines such as GEMM, TRMM, triangular solves, and POTRF. Unsupported
layouts or types continue through the native D fallback. The choice is
made by the compiled configuration and storage checks, not by a runtime DLL
loader. A binary linked to BLAS still needs its native library at runtime.

**The no-workspace `quadraticForm` does not call BLAS.** It uses compensated summation
for each row and the outer reduction by default. The symmetric overload reads
only the selected stored triangle but evaluates every logical row. Its split
loops preserve summation order. The faster experimental single-triangle
reduction in the benchmark suite is not a library API: flattening the sum can
lose useful row-wise cancellation and introduce overflow.

Packed triangular products use TPMV per output column when eligible, including
transpose for right-side products. Packed storage saves input memory but is not
guaranteed to beat dense TRMM for multiple right-hand sides; benchmark the
expected sizes and backend.

The transposed triangular matrix paths depend on the corrections in
[mir-blas PR #12](https://github.com/libmir/mir-blas/pull/12). Resolving that
dependency is separate from the availability of the built-in implementations.
Backend selection and summation order can change floating-point rounding.

## Preconditions and errors

Shapes and finite-input requirements are checked with assertions where
specified. Assertions may be disabled in release builds; callers must satisfy
the documented preconditions. Output elements must not overlap one another.
Most caller-output operations also forbid overlap with inputs. The exception
is `choleskySolveInto`, which permits the output to be exactly the same logical
view as the right-hand side. Partial overlap is not supported or detected.

General matrix arithmetic can overflow or lose accuracy through cancellation,
even with finite inputs. Decomposition failures and exceptions from computed
inputs follow the individual function's documented partial-output rules.

## Tests and benchmarks

Run the focused built-in implementation tests from the repository root:

```sh
dub test --build=unittest-linear-algebra --compiler=ldc2
```

Add `--config=openblas` to exercise eligible native paths with the required
native libraries and fixed dependency installed.

The version declarations

```d
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;
```

enable this module's guarded tests when either the full test suite or the BLAS
test suite is selected. The focused build also defines
`mir_stat_test_linearAlgebra` directly. These are compile-time **test switches**;
they do not enable BLAS or select an arithmetic implementation.

The [benchmark guide](../../../../../benchmarks/README.md#linear-algebra)
explains reproducible comparisons of dense/triangular multiplication,
quadratic forms, and the four Mahalanobis metric representations. Measurements
separate precomputation from per-call work and compare allocating calls with
reusable workspaces.
