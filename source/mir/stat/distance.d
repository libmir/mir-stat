/++
Mahalanobis distance calculations with a supplied metric.

Import `mir.stat.distance` directly or through `mir.stat`. The distance APIs
are public; the underlying linear algebra operations remain internal.
`TrianglePolicy.upper` (the default) or `.lower` selects the stored logical
triangle independently of the supplied metric representation.

For difference `d = x - center` and positive-definite covariance `S`, squared
distance is `d.transpose * inverse(S) * d`. Supply `S`, its Cholesky factor,
its inverse (the precision matrix), or a Cholesky factor of its inverse through
`CovarianceRepresentation`. A precision Cholesky factor is not the inverse of a
covariance Cholesky factor: their multiplication orientations differ.

For an upper covariance factor `S = U.transpose * U`, solve `U.transpose * z = d`.
For a lower factor `S = L * L.transpose`, solve `L * z = d`.
For an upper precision factor `P = R.transpose * R`, multiply `z = R * d`;
for a lower factor `P = Q * Q.transpose`, multiply `z = Q.transpose * d`.
The distance is the Euclidean norm of `z`; no matrix inverse is formed.

For an already centered vector, internal callers can import `quadraticForm`
from `mir.stat.internal.linearAlgebra.quadratic`; it allocates no workspace.
Pass a `symmetricView` to select one matrix triangle.
Quadratic forms also accept indefinite matrices and may return negative values.

API selection:
$(BOOKTABLE ,
    $(TR $(TH Input) $(TH Entry point) $(TH Result))
    $(TR $(TD One observation vector)
         $(TD `mahalanobis` or `mahalanobisSquared`)
         $(TD Scalar))
    $(TR $(TD Supported rectangular `byDim` or `alongDim` selection of observation vectors)
         $(TD Ordinary function names)
         $(TD One eagerly calculated result per selected observation))
    $(TR $(TD Plain matrix with observations in columns)
         $(TD Explicit `Batch` functions)
         $(TD One eagerly calculated result per column))
)
Selection order determines result order. For a matrix, `byDim!1` / `alongDim!0`
selects columns and `byDim!0` / `alongDim!1` selects rows. Selections must be
rectangular collections of observation vectors; arbitrary nested or jagged
collections and higher-rank selections are not accepted.

For vector calls, GC/RC/custom allocation choices concern scratch storage.
For allocating batch and selection calls, they concern the result as well as
scratch. `WithWorkspace` variants use caller-provided output and scratch
buffers (vector calls still return a scalar).

The same observations can be selected in either orientation:
------
import mir.stat.distance;
import mir.ndslice.slice: sliced;
import mir.ndslice.dynamic: transposed;
import mir.ndslice.topology: byDim, alongDim;

double[6] data = [2, 3, 4, 0, 0, 6];
double[4] covariance = [4, 0, double.nan, 9];
double[2] center = [0, 0];
auto rows = data[].sliced(3, 2);
auto columns = rows.transposed;
auto metric = covariance[].sliced(2, 2);
auto byRows = mahalanobisSquared(rows.byDim!0, center[], metric);
auto byColumns = mahalanobisSquared(columns.alongDim!0, center[], metric);
assert(byRows == [2, 4, 4] && byColumns == byRows);
assert(mahalanobisSquaredBatch(columns, center[], metric) == byRows);
------

Retain a covariance factor and select `CovarianceRepresentation.covarianceCholesky`
to reuse it across calls without repeating factorization. The other
representations remain `covariance`, `inverseCovariance`, and
`inverseCovarianceCholesky`. The last means a Cholesky factor of the precision
matrix, not simply the inverse of a covariance factor.

Choose `mahalanobis` or `mahalanobisSquared` for GC scratch storage, their `rc`
variants for temporary reference-counted storage, `make` variants for a custom
allocator, or `WithWorkspace` variants for reusable caller storage. Vector calls return
scalars; their ownership choices apply only to temporary workspaces. Covariance input
needs a matrix and vector workspace; precomputed forms normally need one vector.
Precision-matrix calls can opt into `Summation.fast` as a third template argument
(after the representation and triangle). This uses two vectors and permits BLAS
SYMV plus DOT. GC, RC, custom allocation, and a two-vector `WithWorkspace`
overload are available. Existing calls keep their compensated behavior.

For example, consider 100 test runs with three temperature readings per run.
Selecting rows of a 100-by-3 matrix with `rows.byDim!0` writes 100 distances,
one per run. Each dense batch needs room for 300 centered readings
(100 runs times 3 coordinates). The metric determines the additional storage:

$(TABLE
    $(TR $(TH Dense metric) $(TH Additional workspace) $(TH Use))
    $(TR $(TD Prepared covariance or precision factor) $(TD None)
         $(TD Transform the centered readings in place.))
    $(TR $(TD Covariance) $(TD 3-by-3 factor matrix)
         $(TD Factor the covariance once for all 100 runs.))
    $(TR $(TD Precision with default compensated summation) $(TD None required)
         $(TD Evaluate one quadratic form per run.))
    $(TR $(TD Precision with `Summation.fast`) $(TD 300 product elements)
         $(TD Multiply the precision matrix by the centered readings.))
)

Row and column selections are views. The batch can use a BLAS triangular solve
or multiply for prepared factors, or a symmetric matrix multiply for fast
precision, when the layout permits. To use less observation scratch, process a
few runs per call and reuse a prepared factor. Select scratch axes independently
of input axes. For row selections, a square factor workspace is still an ordinary
matrix, while product scratch is an observation selection. A single selected
observation gives one output entry. Negative axis indices follow Mir conventions.

Internal callers may supply `diagonalView(diagonal)` for any representation.
It stores only the diagonal vector; triangle selection has no effect on values.
Covariance and factor forms scale coordinates and use scaled norms; precision
uses the selected quadratic-form summation. A batch shares each covariance
square root across observations. Nonpositive covariance entries throw the same
factorization exception as dense covariance, including in release builds.
These calls need only one vector workspace
or one observation-shaped batch workspace, including covariance and fast
precision batches. Explicit two-vector precision calls still fill both buffers;
extra covariance-factor workspace, if supplied, still receives a dense factor.
No diagonal constructor is re-exported by this public module. The view carries
structure only; CovarianceRepresentation still determines what its entries represent.

Inputs are same-type real float, double or real vectors and matrices. Vectors
may also be dynamic arrays. Only the selected matrix triangle is read; full
covariance/precision matrices are assumed symmetric. Inputs must be finite,
and supplied metrics must be positive definite. Preconditions use assertions;
precomputed matrices and factors are not revalidated for positive definiteness.
Empty vectors with a zero-by-zero metric have zero distance.

For every metric representation, each centered coordinate is computed as
$(D x[i] - center[i]) in the input element type. This subtraction can overflow
even when the mathematical distance is representable. The same centering rule
applies to single observations, batches, and observation selections.

Covariance and factor forms use a single whitening solve or multiplication,
followed by a scaled Euclidean norm. Distance is computed without first squaring
that norm. The direct inverse-covariance form evaluates a compensated quadratic
form; cancellation can lose accuracy or produce a negative squared result,
which is not clamped. Its distance is then NaN. Intermediate arithmetic can
overflow even for finite inputs. No regularization or pseudoinverse is applied.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distance;

public import mir.stat.internal.linearAlgebra.types: CovarianceRepresentation, TrianglePolicy, DiagonalPolicy;
public import mir.stat.internal.mahalanobis:
    mahalanobis,
    rcMahalanobis,
    makeMahalanobis,
    mahalanobisWithWorkspace,
    mahalanobisBatch,
    rcMahalanobisBatch,
    makeMahalanobisBatch,
    mahalanobisBatchWithWorkspace,
    mahalanobisSquared,
    rcMahalanobisSquared,
    makeMahalanobisSquared,
    mahalanobisSquaredWithWorkspace,
    mahalanobisSquaredBatch,
    rcMahalanobisSquaredBatch,
    makeMahalanobisSquaredBatch,
    mahalanobisSquaredBatchWithWorkspace;
