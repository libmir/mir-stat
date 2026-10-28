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
| `crossprod` | One or two matrices | `Aᵀ A` or `Aᵀ B` |
| `tcrossprod` | One or two matrices | `A Aᵀ` or `A Bᵀ` |
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

The same naming applies to cross-products, triangular products, Cholesky decomposition, and
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

## Cross-products with explicit matrix shapes

`crossprod(a)` forms the column Gram matrix `Aᵀ A`; `tcrossprod(a)` forms the
row Gram matrix `A Aᵀ`. For a matrix with `m` rows and `n` columns, these return
`n × n` and `m × m` matrices respectively. Two-input forms compute `Aᵀ B`
(equal input row counts) or `A Bᵀ` (equal input column counts).

These functions require **matrix shapes**, including for vector-like data:

```d
double[2] values = [2, 3];
auto column = values[].sliced(2, 1);
auto row = values[].sliced(1, 2);
assert(crossprod(column) == [[13]]);
assert(tcrossprod(column) == [[4, 6], [6, 9]]);
assert(crossprod(row) == [[4, 6], [6, 9]]);
assert(tcrossprod(row) == [[13]]);
```

Results remain matrices, even when the shape is `1 × 1`. A bare rank-one slice
is rejected. The names describe logical transposes, independently of storage
order. Complex values are **not conjugated**; these are symmetric products,
not Hermitian products. No symmetry is assumed for two-input calls.

Choose `crossprod` / `tcrossprod` for GC storage, `rcCrossprod` / `rcTcrossprod`
for RC storage, `makeCrossprod` / `makeTcrossprod` for custom allocation, or
`crossprodInto` / `tcrossprodInto` for caller output. All four styles accept
one or two matrix inputs. Custom callers dispose the result's field using the
same allocator. `Into` output must be disjoint from both inputs and have
nonoverlapping elements; the two read-only inputs may overlap each other.

Self-products compute one triangle using SYRK where eligible, then mirror it
to fill the whole output. Other layouts/types use a native D triangle kernel.
Two-input calls reuse `mtimes`; use the one-input form to explicitly request
the self-product optimization. The functions allocate no hidden input copies
or scratch buffers. Empty contraction dimensions produce zeros. Reduction
order and BLAS implementations can affect rounding and performance.

The operations live in `crossproducts.d`, are re-exported by the package and
the allocation-specific convenience modules, and remain package-restricted.

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

Symmetric views are accepted by `mtimes` on either side, quadratic forms, and Cholesky decomposition.
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

### Logical slice conversion

Both structural views expose a read-only `asSlice` property. It reads the
logical matrix: symmetric views reflect their selected triangle, and triangular
views supply zeros and optional unit diagonal entries. It never reads unused
storage to supply those values. The adapter copies no elements and preserves
RC ownership; borrowed storage and computed dependencies must remain alive.

~~~d
double[4] data = [double.nan, 3, double.nan, double.nan];
auto upper = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(
    data[].sliced(2, 2));
auto logical = upper.asSlice;
assert(logical == [[1, 3], [0, 1]]);
auto dense = logical.slice; // Independent writable GC copy.
auto owned = logical.rcslice; // Independent writable RC copy.
~~~

Use `makeSlice(allocator, logical)` for a custom copy, and dispose its `field`
with the same allocator. Raw `view.storage` is different: it exposes the
underlying square indexing and can include unused or uninitialized entries.

The property participates in Mir's existing `hasAsSlice` convention.
`view.toSlice` and forwarding operations such as `view.map!"a * 2"` consume
logical values without requiring an explicit `.asSlice` at the call site.
The result is an ordinary logical slice; it no longer carries structural
metadata for algorithm selection. Pass the original view to `mtimes`,
quadratic forms, and other structure-aware operations to retain their
specialized paths. Mutation requires an explicit writable copy.

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

## Fast precision-matrix Mahalanobis distance

Public distance functions keep their existing compensated defaults. For a known
inverse covariance matrix, select fast arithmetic explicitly:

```d
import mir.stat.distance;
import mir.math.sum: Summation;
auto squared = mahalanobisSquared!(MahalanobisMatrix.inverseCovariance,
    TrianglePolicy.upper, Summation.fast)(x, center, precision);
auto reused = mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovariance,
    TrianglePolicy.upper, Summation.fast)(x, center, precision, difference, product);
```

The allocating call needs two temporary vectors; RC and custom-allocator variants
provide the same option. The workspace call allocates nothing and requires two
separate vectors, disjoint from all inputs. The first holds `x - center`; the
second holds the precision-matrix product. Existing covariance and factor-based
paths are unchanged. Fast mode trades compensated accuracy for backend-dependent
arithmetic; it does not change the positive-definite metric precondition.

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


## Symmetric multiplication

Use the same multiplication functions after declaring symmetry:

```d
auto a = symmetricView!(TrianglePolicy.upper)(matrix);
auto left = mtimes(a, observations);
auto right = mtimes(observations.transposed, a);
mtimesInto(a, observations, output);
```

The unselected triangle is never read. The ordinary operand may be a matrix
or vector. GC, RC, custom allocation, and caller-output variants all dispatch
on the view. Complex symmetry reflects entries without conjugating them.

Eligible dense matrix products use SYMM when the ordinary input and output
have compatible storage orientations. Eligible mixed orientations use a blocked
calculation: GEMM handles stored off-diagonal blocks and their reflections,
while small diagonal blocks use native D arithmetic. This avoids a full dense
copy and requires no additional workspace. Real vectors use SYMV. Packed symmetric storage uses SPMV per column, because
standard BLAS has no packed SYMM operation. Other types and layouts use native
D arithmetic. Native D loops split at the diagonal to avoid per-entry triangle
selection; complete packed views use direct stored coordinates instead of
reconstructing square indices. No dense copy of the symmetric input is created.

Declaring symmetry is not a universal speedup: it supplies the selected-triangle
contract. SYMV can beat GEMV, while SYMM and GEMM may perform similarly. Packed
storage and mixed layouts can remain slower than dense matched layouts.

For eligible `float`/`double` inputs, an ordinary operand with one column uses
the vector kernel while keeping a rank-two output. A right-side product with
one row follows the same rule. No workspace is needed for this specialization.

For matrix products with different orientations, provide layout workspace:

```d
auto a = symmetricView(metric);
auto b = observations.transposed;
mtimesInto(a, b, output, work);
```

`work` has the ordinary operand/output's shape and element type. When BLAS can
use it, workspace matching `output` receives a copy of `b` before SYMM;
workspace matching `b` receives the product before it is copied into `output`.
The right-side overload uses the same convention. All writable buffers must be
mutually disjoint and disjoint from inputs, with nonoverlapping elements.

Workspace may be unused: already-compatible layouts, single-column real
products, and unsupported backend/type/storage combinations keep the ordinary
path. Its contents are unspecified afterward. No heap storage is allocated,
no full symmetric matrix is materialized, and no timing-based dispatch is used.
The existing three-argument `mtimesInto` remains allocation-free without workspace.


## Selecting observation axes

Use the ordinary names for a single vector or an explicit selection of vectors.
Use the `Batch` names for a plain matrix with observations in columns:

```d
auto one = mahalanobisSquared(observation, center, metric); // Scalar
auto many = mahalanobisSquaredBatch(columns, center, metric); // Vector
auto fromRows = mahalanobisSquared(rows.alongDim!1, center, metric);
auto sameRows = mahalanobisSquared(rows.byDim!0, center, metric);
```

Both selectors come from `mir.ndslice.topology`:

| Observations | `alongDim` selection | Equivalent `byDim` selection |
|---|---|---|
| Columns of a matrix | `matrix.alongDim!0` | `matrix.byDim!1` |
| Rows of a matrix | `matrix.alongDim!1` | `matrix.byDim!0` |

`alongDim` names the coordinate dimension inside each observation. `byDim`
names the dimension indexing observations. Their meanings are complementary
for a matrix, not interchangeable for the same numeric axis. Negative indices
are supported: `alongDim!(-1)` selects rows and `alongDim!(-2)` selects columns.

The selected observations retain the original matrix's storage and ownership.
Internally, `unpack` plus a transpose view recovers a matrix whose columns are
observations. No elements are copied. The adapter forwards that whole matrix
to the corresponding `Batch` implementation, keeping one covariance
factorization and eligible TRSM/TRMM/GEMM/SYMM batch calls. In contrast, mapping
the single-observation function invokes it repeatedly.

Axis meaning is independent of row-major or column-major storage. A transpose
changes logical axes, so choose axes on the view actually passed. One selected
observation returns a one-element vector; an ordinary vector returns a scalar.
Empty selections return empty results. Sliced, reversed, strided, and const
selections preserve their logical order. Arbitrary arrays of separate vectors
and higher-rank selections are outside the initial adapter contract.

The allocation options are the same for explicit batches and selected views:

| Ownership | Explicit batch | Selected observations |
|---|---|---|
| GC | `mahalanobisSquaredBatch` | `mahalanobisSquared` |
| RC | `rcMahalanobisSquaredBatch` | `rcMahalanobisSquared` |
| Custom | `makeMahalanobisSquaredBatch` | `makeMahalanobisSquared` |
| Caller output/scratch | `mahalanobisSquaredBatchWithWorkspace` | `mahalanobisSquaredWithWorkspace` |

Unsquared distances have the same variants. Internal quadratic forms provide
`quadraticFormBatch`, `rcQuadraticFormBatch`, and `makeQuadraticFormBatch`, with
selected counterparts `quadraticForm`, `rcQuadraticForm`, and `makeQuadraticForm`.
`quadraticFormInto` writes selected results; `quadraticFormWithWorkspace` also
takes selected scratch. Explicit batches use `quadraticFormBatchInto` for both.

Select workspace axes explicitly too; workspace storage orientation may differ:

```d
mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(
    rows.alongDim!1, center, factor, results, scratchColumns.byDim!1);

quadraticFormWithWorkspace!(Summation.fast)(
    rows.byDim!0, symmetricView(metric), values, scratchColumns.alongDim!0);
```

Each selected scratch view must have the input's observation count and vector
length. Output remains an ordinary vector. Covariance needs an extra ordinary
square factor workspace; fast precision needs an extra selected product
workspace. Inputs, output, and scratch obey the same disjointness requirements
as their explicit batch counterparts. The adapters themselves allocate nothing.

## Batches of quadratic forms and distances

A batch has shape **(dimension, observation count)**: each column is one
observation. A single metric applies to all columns. Results are vectors with
one entry per column; observations are not compared pairwise.

For internal quadratic forms:

```d
auto values = quadraticFormBatch(observations, symmetricView(metric));
quadraticFormBatchInto(observations, symmetricView(metric), values);
quadraticFormBatchInto!(Summation.fast)(
    observations, symmetricView(metric), values, productWorkspace);
```

The default computes compensated row and final reductions independently for
each column. Explicit fast mode with a matrix workspace computes all products
at once using GEMM for general matrices or SYMM for eligible symmetric views,
then reduces column dots. It permits backend-dependent rounding. The allocating
`quadraticFormBatch`, `rcQuadraticFormBatch`, and `makeQuadraticFormBatch` variants
allocate this workspace only for fast mode; other policies allocate just the
result. These names are also available through the corresponding allocation
convenience modules.

The public distance module provides `mahalanobisBatch` and
`mahalanobisSquaredBatch`, plus `rc`, `make`, and `WithWorkspace` variants:

```d
auto distances = mahalanobisBatch!(MahalanobisMatrix.cholesky)(
    observations, center, covarianceFactor);
mahalanobisSquaredBatchWithWorkspace!(MahalanobisMatrix.cholesky)(
    observations, center, covarianceFactor, results, centeredWorkspace);
mahalanobisSquaredBatchWithWorkspace!(MahalanobisMatrix.inverseCovariance,
    TrianglePolicy.upper, Summation.fast)(
    observations, center, precision, results, centeredWorkspace, productWorkspace);
```

The center is one vector shared by the batch. `results` has one element per
column, and both centered/product workspaces have the observation matrix's
shape. Starting from covariance requires an additional square factor workspace
instead of a product workspace; the covariance is factorized once per batch.
Prepared covariance factors use a single TRSM across columns, and precision
factors use TRMM. Their final scaled column norms preserve the unsquared
function's range. Precision matrices use compensated quadratic forms unless
fast arithmetic is explicitly selected. Every path has a native D fallback.

Caller-owned output and scratch buffers must be mutually disjoint and disjoint
from inputs, with no overlapping writable elements. GC and RC variants manage
output ownership; custom variants release scratch but transfer result ownership
to the caller.

The benchmark runner's `quadraticBatch` and `mahalanobisBatch` families compare
32 columns per invocation with repeated single-column calculations using the
same summation policy and reusable workspace. `--size` selects the dimension;
reported time is per whole batch, not per observation. Both storage orientations
are available through `--layout`. Setup, factorization of prepared metrics,
and analytic-reference validation occur outside the timed operations.
