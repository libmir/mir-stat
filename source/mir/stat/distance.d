/++
Mahalanobis distance calculations with a supplied metric.

Import `mir.stat.distance` directly or through `mir.stat`. The distance APIs
are public; the underlying linear algebra operations remain internal.
`TrianglePolicy.upper` (the default) or `.lower` selects the stored logical
triangle independently of the supplied metric representation.

For difference `d = x - center` and positive-definite covariance `S`, squared
distance is `d.transpose * inverse(S) * d`. Supply `S`, its Cholesky factor,
its inverse (the precision matrix), or a Cholesky factor of its inverse.
The label identifies which matrix the caller has supplied.

$(BOOKTABLE ,
    $(TR $(TH Supplied matrix) $(TH Metric argument) $(TH Squared distance from raw storage))
    $(TR $(TD Covariance S) $(TD `covarianceView(s)`) $(TD `mahalanobisSquaredFromCovariance`))
    $(TR $(TD Precision P) $(TD `precisionView(p)`) $(TD `mahalanobisSquaredFromPrecision`))
    $(TR $(TD Factor of S) $(TD `covarianceCholeskyView(f)`) $(TD `mahalanobisSquaredFromCovarianceCholesky`))
    $(TR $(TD Factor of P) $(TD `precisionCholeskyView(f)`) $(TD `mahalanobisSquaredFromPrecisionCholesky`))
)

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
auto metric = covarianceView(covariance[].sliced(2, 2));
auto byRows = mahalanobisSquared(rows.byDim!0, center[], metric);
auto byColumns = mahalanobisSquared(columns.alongDim!0, center[], metric);
assert(byRows == [2, 4, 4] && byColumns == byRows);
assert(mahalanobisSquaredBatch(columns, center[], metric) == byRows);
------

Retain a labeled Cholesky factor to reuse it across calls. The ordinary GC
functions accept raw matrices or matching labels through the explicit names in
the table. Use `mahalanobisFromPrecision`, for example, for the distance and
`mahalanobisSquaredFromPrecision` for its square. A raw input defaults to the
upper triangle. A labeled input supplies its own triangle.

Choose `mahalanobis` or `mahalanobisSquared` for GC scratch storage, their `rc`
variants for temporary reference-counted storage, `make` variants for a custom
allocator, or `WithWorkspace` variants for reusable caller storage. Vector calls return
scalars; their ownership choices apply only to temporary workspaces. Covariance input
needs a matrix and vector workspace; precomputed forms normally need one vector.
Precision-matrix calls can opt into `Summation.fast` as a template argument
on a call with a precision view. This uses two vectors and permits BLAS
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

public import mir.stat.covariance_views: CovarianceMetric, CovarianceRepresentation, TrianglePolicy,
    covarianceView, covarianceCholeskyView, precisionView, precisionCholeskyView;
public import mir.stat.internal.linearAlgebra.types: DiagonalPolicy;

private import mir.stat.covariance_views: isCovarianceMetric;
private import mir.math.sum: Summation;
private import std.conv: to;
private import mir.stat.internal.linearAlgebra.allocation: Allocation, AllocatedSlice;
private import mir.stat.internal.linearAlgebra.types: ObservationColumns, isObservationSelection, observationColumns,
    Element, vector, matrix, compatible, asView;
private import impl = mir.stat.internal.mahalanobis;

version(mir_stat_test) version = mir_stat_test_distance;
version(mir_stat_test_linearAlgebra) version = mir_stat_test_distance;
version(mir_stat_test_blas) version = mir_stat_test_distance;

private enum acceptsPolicy(M, Summation summation) = isCovarianceMetric!M &&
    (summation == Summation.kbn || M.form == CovarianceRepresentation.inverseCovariance);

// Factor scratch is a square matrix; precision product scratch follows the selected observation axes.
private auto selectedExtra(CovarianceRepresentation form, E)(return scope auto ref E extra)
    if ((form == CovarianceRepresentation.covariance && matrix!E) ||
        (form == CovarianceRepresentation.inverseCovariance && isObservationSelection!E))
{
    static if (form == CovarianceRepresentation.inverseCovariance) return observationColumns(extra);
    else return extra.asView;
}

// Show only interpretations accepted with this policy and these workspaces.
private template missingMetricMessage(alias accepts, string covarianceAlternative = "")
{
    enum covariance = accepts!covarianceView ? " For covariance, pass covarianceView(metric)" ~ covarianceAlternative ~ "." : "";
    enum precision = accepts!precisionView ? " For precision, pass precisionView(metric)." : "";
    enum covarianceFactor = accepts!covarianceCholeskyView ? " For a covariance factor, pass covarianceCholeskyView(metric)." : "";
    enum precisionFactor = accepts!precisionCholeskyView ? " For a precision factor, pass precisionCholeskyView(metric)." : "";
    enum choices = covariance ~ precision ~ covarianceFactor ~ precisionFactor;
    enum missingMetricMessage = choices.length ? "Specify the matrix representation." ~ choices
        : "No labeled metric supports these arguments. Check the observation, center, workspace, and summation arguments.";
}

/++
Evaluate one distance using GC-managed storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X mahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.evaluate!(Allocation.gc, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Reuse a prepared covariance factor.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    assert(mahalanobis(x[], center[], metric) == 2);
}

/++
Evaluate distances for selected observations using GC-managed storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A GC-managed vector of n distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
mahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, mahalanobisBatch!summation(observationColumns(x), center, metric)))
{
    return mahalanobisBatch!summation(observationColumns(x), center, metric);
}

/// Select the two observations by row.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    auto result = mahalanobis(data[].sliced(2, 1).byDim!0, center[], metric);
    assert(result == [2, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, mahalanobis!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts, " or call mahalanobisFromCovariance(x, center, metric)"));
}

/++
Evaluate distance from a covariance matrix.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k covariance matrix or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
auto mahalanobisFromCovariance(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobis(x, center, covarianceView!triangle(metric))))
{
    return mahalanobis(x, center, covarianceView!triangle(metric));
}

/// ditto
auto mahalanobisFromCovariance(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobis(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.covariance,
        "mahalanobisFromCovariance expects a covariance matrix. Received " ~ to!string(M.form) ~ ".");
    return mahalanobis(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [4];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisFromCovariance(x[], center[], matrix) == 2);
    assert(mahalanobisFromCovariance(x[], center[], covarianceView(matrix)) == 2);
}

/++
Evaluate distance from a precision matrix.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k precision matrix or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisFromPrecision(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobis(x, center, precisionView!triangle(metric))))
{
    return mahalanobis(x, center, precisionView!triangle(metric));
}

/// ditto
auto mahalanobisFromPrecision(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobis(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovariance,
        "mahalanobisFromPrecision expects a precision matrix. Received " ~ to!string(M.form) ~ ".");
    return mahalanobis(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.25];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisFromPrecision(x[], center[], matrix) == 2);
    assert(mahalanobisFromPrecision(x[], center[], precisionView(matrix)) == 2);
}

/++
Evaluate distance from a Cholesky factor of covariance.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k Cholesky factor of covariance or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisFromCovarianceCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobis(x, center, covarianceCholeskyView!triangle(metric))))
{
    return mahalanobis(x, center, covarianceCholeskyView!triangle(metric));
}

/// ditto
auto mahalanobisFromCovarianceCholesky(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobis(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.covarianceCholesky,
        "mahalanobisFromCovarianceCholesky expects a Cholesky factor of covariance. Received " ~ to!string(M.form) ~ ".");
    return mahalanobis(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [2];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisFromCovarianceCholesky(x[], center[], matrix) == 2);
    assert(mahalanobisFromCovarianceCholesky(x[], center[], covarianceCholeskyView(matrix)) == 2);
}

/++
Evaluate distance from a Cholesky factor of precision.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k Cholesky factor of precision or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisFromPrecisionCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobis(x, center, precisionCholeskyView!triangle(metric))))
{
    return mahalanobis(x, center, precisionCholeskyView!triangle(metric));
}

/// ditto
auto mahalanobisFromPrecisionCholesky(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobis(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovarianceCholesky,
        "mahalanobisFromPrecisionCholesky expects a Cholesky factor of precision. Received " ~ to!string(M.form) ~ ".");
    return mahalanobis(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.5];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisFromPrecisionCholesky(x[], center[], matrix) == 2);
    assert(mahalanobisFromPrecisionCholesky(x[], center[], precisionCholeskyView(matrix)) == 2);
}

/++
Evaluate one distance using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X rcMahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.evaluate!(Allocation.rc, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Evaluate a precision matrix using reference-counted scratch.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.25];
    auto metric = precisionView(entries[].sliced(1, 1));
    assert(rcMahalanobis(x[], center[], metric) == 2);
}

/++
Evaluate distances for selected observations using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A reference-counted vector of n distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, rcMahalanobisBatch!summation(observationColumns(x), center, metric)))
{
    return rcMahalanobisBatch!summation(observationColumns(x), center, metric);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto rcMahalanobis(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, rcMahalanobis!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate one distance using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X makeMahalanobis(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    return impl.evaluate!(Allocation.custom, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Use a custom allocator for temporary storage.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    double[1] x = [4], center = [0], entries = [4];
    auto metric = covarianceView(entries[].sliced(1, 1));
    assert(makeMahalanobis(Mallocator.instance, x[], center[], metric) == 2);
}

/++
Evaluate distances for selected observations using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch and result storage.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A custom-allocated vector of n distances in observation order.
    Zero-dimensional observations have distance zero.
    Dispose the returned slice field with allocator after its last use.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMahalanobis(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, makeMahalanobisBatch!summation(allocator, observationColumns(x), center, metric)))
{
    return makeMahalanobisBatch!summation(allocator, observationColumns(x), center, metric);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto makeMahalanobis(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, makeMahalanobis!summation(allocator, x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate one distance using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix with KBN summation, or a Cholesky
        factor of covariance or precision. Other precision reductions need a product vector.
        The selected triangle must be finite and represent a positive-definite metric.
    work = Writable vector of k centered coordinates.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    The distance in the input element type.
    Zero-dimensional observations have distance zero.
+/
Element!X mahalanobisWithWorkspace(Summation summation = Summation.kbn, X, C, M, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && is(Element!X == Element!W) &&
        M.form != CovarianceRepresentation.covariance && summation == Summation.kbn)
{
    return impl.prepared!(false, M.form, M.triangle)(
        x.asView, center.asView, metric.storage.asView, work.asView);
}

/// Reuse a scratch vector with a prepared precision factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.5];
    double[1] scratch;
    auto metric = precisionCholeskyView(entries[].sliced(1, 1));
    assert(mahalanobisWithWorkspace(x[], center[], metric, scratch[]) == 2);
}

/++
Evaluate one distance using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    work = Writable vector of k centered coordinates.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    The distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X mahalanobisWithWorkspace(Summation summation = Summation.kbn, X, C, M, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && is(Element!X == Element!W) &&
        (M.form == CovarianceRepresentation.covariance && matrix!E ||
         M.form == CovarianceRepresentation.inverseCovariance && vector!E) && is(Element!X == Element!E))
{
    static if (M.form == CovarianceRepresentation.covariance)
        return impl.covarianceDistance!(false, M.triangle)(
            x.asView, center.asView, metric.storage.asView, work.asView, extra.asView);
    else
        return impl.precisionWorkspace!(false, M.triangle, summation)(
            x.asView, center.asView, metric.storage.asView, work.asView, extra.asView);
}

/// Supply both the centered vector and the precision product vector.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.sum: Summation;
    double[1] x = [4], center = [0], entries = [0.25];
    double[1] scratch, product;
    auto metric = precisionView(entries[].sliced(1, 1));
    assert(mahalanobisWithWorkspace!(Summation.fast)(x[], center[], metric, scratch[], product[]) == 2);
}

/++
Evaluate distances for selected observations using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix (KBN by default), or a Cholesky
        factor of covariance or precision. A fast precision reduction also needs product storage.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n distances in observation order.
    work = Writable observation selection matching the lengths and count of x.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one distance per observation to result.
    Zero-dimensional observations have distance zero.
+/
void mahalanobisWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X && isObservationSelection!W &&
        __traits(compiles, mahalanobisBatchWithWorkspace!summation(
            observationColumns(x), center, metric, result, observationColumns(work))))
{
    mahalanobisBatchWithWorkspace!summation(
        observationColumns(x), center, metric, result, observationColumns(work));
}

/++
Evaluate distances for selected observations using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n distances in observation order.
    work = Writable observation selection matching the lengths and count of x.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one distance per observation to result.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails, before any result entry is written.
+/
void mahalanobisWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X && isObservationSelection!W &&
        __traits(compiles, mahalanobisBatchWithWorkspace!summation(
            observationColumns(x), center, metric, result, observationColumns(work), selectedExtra!(M.form)(extra))))
{
    mahalanobisBatchWithWorkspace!summation(
        observationColumns(x), center, metric, result, observationColumns(work), selectedExtra!(M.form)(extra));
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisWithWorkspace(Summation summation = Summation.kbn, X, C, M, W...)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric, scope auto ref W buffers)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, mahalanobisWithWorkspace!summation(x, center, label(metric), buffers));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate distances for observation columns using GC-managed storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A GC-managed vector of n distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
mahalanobisBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.batchAllocated!(Allocation.gc, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Evaluate two observation columns with a prepared factor.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    auto result = mahalanobisBatch(data[].sliced(1, 2), center[], metric);
    assert(result == [2, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, mahalanobisBatch!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate distances for observation columns using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A reference-counted vector of n distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMahalanobisBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.batchAllocated!(Allocation.rc, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Keep a reference-counted vector of column results.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [0.25];
    auto metric = precisionView(entries[].sliced(1, 1));
    auto result = rcMahalanobisBatch(data[].sliced(1, 2), center[], metric);
    assert(result == [2, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto rcMahalanobisBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, rcMahalanobisBatch!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate distances for observation columns using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch and result storage.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A custom-allocated vector of n distances in observation order.
    Zero-dimensional observations have distance zero.
    Dispose the returned slice field with allocator after its last use.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMahalanobisBatch(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    return impl.batchAllocated!(Allocation.custom, false, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Release the result with the allocator that created it.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [4];
    auto metric = covarianceView(entries[].sliced(1, 1));
    auto result = makeMahalanobisBatch(Mallocator.instance, data[].sliced(1, 2), center[], metric);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [2, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto makeMahalanobisBatch(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, makeMahalanobisBatch!summation(allocator, x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate distances for observation columns using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix (KBN by default), or a Cholesky
        factor of covariance or precision. A fast precision reduction also needs product storage.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n distances in observation order.
    work = Writable k-by-n matrix for centered observations.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one distance per observation to result.
    Zero-dimensional observations have distance zero.
+/
void mahalanobisBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)) &&
        vector!R && matrix!W && is(Element!X == Element!R) && is(Element!X == Element!W) &&
        M.form != CovarianceRepresentation.covariance && summation != Summation.fast)
{
    impl.preparedBatch!(false, M.form, M.triangle, summation)(
        x.asView, center.asView, metric.storage.asView, result.asView, work.asView);
}

/// Write one result per column using caller-owned scratch.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [0.5];
    double[2] result, scratch;
    auto metric = precisionCholeskyView(entries[].sliced(1, 1));
    mahalanobisBatchWithWorkspace(data[].sliced(1, 2), center[], metric, result[], scratch[].sliced(1, 2));
    assert(result == [2, 1]);
}

/++
Evaluate distances for observation columns using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n distances in observation order.
    work = Writable k-by-n matrix for centered observations.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one distance per observation to result.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails, before any result entry is written.
+/
void mahalanobisBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)) &&
        vector!R && matrix!W && is(Element!X == Element!R) && is(Element!X == Element!W) &&
        matrix!E && (M.form == CovarianceRepresentation.covariance || M.form == CovarianceRepresentation.inverseCovariance) && is(Element!X == Element!E))
{
    impl.batchWithExtra!(false, M.form, M.triangle, summation)(
        x.asView, center.asView, metric.storage.asView, result.asView, work.asView, extra.asView);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, W...)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric, scope auto ref W buffers)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, mahalanobisBatchWithWorkspace!summation(x, center, label(metric), buffers));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate one squared distance using GC-managed storage.

For a normal observation, see the probability ellipsoid example in
$(MREF mir,stat,distribution,multivariate_normal) for converting a squared
distance from the population mean into a tail probability.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X mahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.evaluate!(Allocation.gc, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Reuse a prepared covariance factor.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    assert(mahalanobisSquared(x[], center[], metric) == 4);
}

/// Test whether a point lies in the 95% probability ellipse for two correlated measurements.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.math.common: fabs;
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.chi2: chi2CDF, chi2CCDF, chi2InvCDF;

    double[2] point = [4, 2], mean = [0, 0];
    double[4] lower = [2, 0, 1, 1]; // Lower Cholesky factor of the covariance.
    auto metric = covarianceCholeskyView!(TrianglePolicy.lower)(lower[].sliced(2, 2));
    const d2 = mahalanobisSquared(point[], mean[], metric);
    const cutoff = chi2InvCDF(0.95, 2);
    assert(d2 == 4 && d2 <= cutoff); // The squared-radius cutoff is about 5.99.

    assert(fabs(chi2CDF(d2, 2) - 0.864665) < 1e-6);
    assert(fabs(chi2CCDF(d2, 2) - 0.135335) < 1e-6);
}

/++
Evaluate squared distances for selected observations using GC-managed storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A GC-managed vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
mahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, mahalanobisSquaredBatch!summation(observationColumns(x), center, metric)))
{
    return mahalanobisSquaredBatch!summation(observationColumns(x), center, metric);
}

/// Select the two observations by row.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    auto result = mahalanobisSquared(data[].sliced(2, 1).byDim!0, center[], metric);
    assert(result == [4, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, mahalanobisSquared!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts, " or call mahalanobisSquaredFromCovariance(x, center, metric)"));
}

/++
Evaluate squared distance from a covariance matrix.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k covariance matrix or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
auto mahalanobisSquaredFromCovariance(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobisSquared(x, center, covarianceView!triangle(metric))))
{
    return mahalanobisSquared(x, center, covarianceView!triangle(metric));
}

/// ditto
auto mahalanobisSquaredFromCovariance(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobisSquared(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.covariance,
        "mahalanobisSquaredFromCovariance expects a covariance matrix. Received " ~ to!string(M.form) ~ ".");
    return mahalanobisSquared(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [4];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisSquaredFromCovariance(x[], center[], matrix) == 4);
    assert(mahalanobisSquaredFromCovariance(x[], center[], covarianceView(matrix)) == 4);
}

/++
Evaluate squared distance from a precision matrix.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k precision matrix or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisSquaredFromPrecision(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobisSquared(x, center, precisionView!triangle(metric))))
{
    return mahalanobisSquared(x, center, precisionView!triangle(metric));
}

/// ditto
auto mahalanobisSquaredFromPrecision(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobisSquared(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovariance,
        "mahalanobisSquaredFromPrecision expects a precision matrix. Received " ~ to!string(M.form) ~ ".");
    return mahalanobisSquared(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.25];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisSquaredFromPrecision(x[], center[], matrix) == 4);
    assert(mahalanobisSquaredFromPrecision(x[], center[], precisionView(matrix)) == 4);
}

/++
Evaluate squared distance from a Cholesky factor of covariance.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k Cholesky factor of covariance or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisSquaredFromCovarianceCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobisSquared(x, center, covarianceCholeskyView!triangle(metric))))
{
    return mahalanobisSquared(x, center, covarianceCholeskyView!triangle(metric));
}

/// ditto
auto mahalanobisSquaredFromCovarianceCholesky(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobisSquared(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.covarianceCholesky,
        "mahalanobisSquaredFromCovarianceCholesky expects a Cholesky factor of covariance. Received " ~ to!string(M.form) ~ ".");
    return mahalanobisSquared(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [2];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisSquaredFromCovarianceCholesky(x[], center[], matrix) == 4);
    assert(mahalanobisSquaredFromCovarianceCholesky(x[], center[], covarianceCholeskyView(matrix)) == 4);
}

/++
Evaluate squared distance from a Cholesky factor of precision.

Params:
    triangle = Triangle to read from raw storage, upper by default.
        A labeled input supplies its own triangle.
    x = Vector of k finite coordinates or a rectangular selection of n such vectors.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Dense k-by-k Cholesky factor of precision or matching labeled view.
        Its selected entries must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type, or a GC-managed vector in selection order.
    Zero-dimensional observations have distance zero.
+/
auto mahalanobisSquaredFromPrecisionCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && __traits(compiles, mahalanobisSquared(x, center, precisionCholeskyView!triangle(metric))))
{
    return mahalanobisSquared(x, center, precisionCholeskyView!triangle(metric));
}

/// ditto
auto mahalanobisSquaredFromPrecisionCholesky(X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, mahalanobisSquared(x, center, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovarianceCholesky,
        "mahalanobisSquaredFromPrecisionCholesky expects a Cholesky factor of precision. Received " ~ to!string(M.form) ~ ".");
    return mahalanobisSquared(x, center, metric);
}

/// Supply the matrix directly or pass its matching view.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.5];
    auto matrix = entries[].sliced(1, 1);
    assert(mahalanobisSquaredFromPrecisionCholesky(x[], center[], matrix) == 4);
    assert(mahalanobisSquaredFromPrecisionCholesky(x[], center[], precisionCholeskyView(matrix)) == 4);
}

/++
Evaluate one squared distance using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X rcMahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.evaluate!(Allocation.rc, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Evaluate a precision matrix using reference-counted scratch.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.25];
    auto metric = precisionView(entries[].sliced(1, 1));
    assert(rcMahalanobisSquared(x[], center[], metric) == 4);
}

/++
Evaluate squared distances for selected observations using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A reference-counted vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, rcMahalanobisSquaredBatch!summation(observationColumns(x), center, metric)))
{
    return rcMahalanobisSquaredBatch!summation(observationColumns(x), center, metric);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto rcMahalanobisSquared(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, rcMahalanobisSquared!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate one squared distance using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    The squared distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X makeMahalanobisSquared(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)))
{
    return impl.evaluate!(Allocation.custom, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Use a custom allocator for temporary storage.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    double[1] x = [4], center = [0], entries = [4];
    auto metric = covarianceView(entries[].sliced(1, 1));
    assert(makeMahalanobisSquared(Mallocator.instance, x[], center[], metric) == 4);
}

/++
Evaluate squared distances for selected observations using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch and result storage.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A custom-allocated vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.
    Dispose the returned slice field with allocator after its last use.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMahalanobisSquared(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X &&
        __traits(compiles, makeMahalanobisSquaredBatch!summation(allocator, observationColumns(x), center, metric)))
{
    return makeMahalanobisSquaredBatch!summation(allocator, observationColumns(x), center, metric);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto makeMahalanobisSquared(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, makeMahalanobisSquared!summation(allocator, x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate one squared distance using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix with KBN summation, or a Cholesky
        factor of covariance or precision. Other precision reductions need a product vector.
        The selected triangle must be finite and represent a positive-definite metric.
    work = Writable vector of k centered coordinates.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    The squared distance in the input element type.
    Zero-dimensional observations have distance zero.
+/
Element!X mahalanobisSquaredWithWorkspace(Summation summation = Summation.kbn, X, C, M, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && is(Element!X == Element!W) &&
        M.form != CovarianceRepresentation.covariance && summation == Summation.kbn)
{
    return impl.prepared!(true, M.form, M.triangle)(
        x.asView, center.asView, metric.storage.asView, work.asView);
}

/// Reuse a scratch vector with a prepared precision factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [4], center = [0], entries = [0.5];
    double[1] scratch;
    auto metric = precisionCholeskyView(entries[].sliced(1, 1));
    assert(mahalanobisSquaredWithWorkspace(x[], center[], metric, scratch[]) == 4);
}

/++
Evaluate one squared distance using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Vector of k finite coordinates.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    work = Writable vector of k centered coordinates.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    The squared distance in the input element type.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
Element!X mahalanobisSquaredWithWorkspace(Summation summation = Summation.kbn, X, C, M, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && is(Element!X == Element!W) &&
        (M.form == CovarianceRepresentation.covariance && matrix!E ||
         M.form == CovarianceRepresentation.inverseCovariance && vector!E) && is(Element!X == Element!E))
{
    static if (M.form == CovarianceRepresentation.covariance)
        return impl.covarianceDistance!(true, M.triangle)(
            x.asView, center.asView, metric.storage.asView, work.asView, extra.asView);
    else
        return impl.precisionWorkspace!(true, M.triangle, summation)(
            x.asView, center.asView, metric.storage.asView, work.asView, extra.asView);
}

/// Supply both the centered vector and the precision product vector.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.sum: Summation;
    double[1] x = [4], center = [0], entries = [0.25];
    double[1] scratch, product;
    auto metric = precisionView(entries[].sliced(1, 1));
    assert(mahalanobisSquaredWithWorkspace!(Summation.fast)(x[], center[], metric, scratch[], product[]) == 4);
}

/++
Evaluate squared distances for selected observations using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix (KBN by default), or a Cholesky
        factor of covariance or precision. A fast precision reduction also needs product storage.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n squared distances in observation order.
    work = Writable observation selection matching the lengths and count of x.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one squared distance per observation to result.
    Zero-dimensional observations have distance zero.
+/
void mahalanobisSquaredWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X && isObservationSelection!W &&
        __traits(compiles, mahalanobisSquaredBatchWithWorkspace!summation(
            observationColumns(x), center, metric, result, observationColumns(work))))
{
    mahalanobisSquaredBatchWithWorkspace!summation(
        observationColumns(x), center, metric, result, observationColumns(work));
}

/++
Evaluate squared distances for selected observations using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Rectangular selection of n observations of k finite coordinates each.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n squared distances in observation order.
    work = Writable observation selection matching the lengths and count of x.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one squared distance per observation to result.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails, before any result entry is written.
+/
void mahalanobisSquaredWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && isObservationSelection!X && isObservationSelection!W &&
        __traits(compiles, mahalanobisSquaredBatchWithWorkspace!summation(
            observationColumns(x), center, metric, result, observationColumns(work), selectedExtra!(M.form)(extra))))
{
    mahalanobisSquaredBatchWithWorkspace!summation(
        observationColumns(x), center, metric, result, observationColumns(work), selectedExtra!(M.form)(extra));
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisSquaredWithWorkspace(Summation summation = Summation.kbn, X, C, M, W...)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric, scope auto ref W buffers)
    if (matrix!M && (vector!X || isObservationSelection!X))
{
    enum accepts(alias label) = __traits(compiles, mahalanobisSquaredWithWorkspace!summation(x, center, label(metric), buffers));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate squared distances for observation columns using GC-managed storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A GC-managed vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
mahalanobisSquaredBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.batchAllocated!(Allocation.gc, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Evaluate two observation columns with a prepared factor.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [2];
    auto metric = covarianceCholeskyView(entries[].sliced(1, 1));
    auto result = mahalanobisSquaredBatch(data[].sliced(1, 2), center[], metric);
    assert(result == [4, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisSquaredBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, mahalanobisSquaredBatch!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate squared distances for observation columns using reference-counted storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A reference-counted vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMahalanobisSquaredBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return impl.batchAllocated!(Allocation.rc, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Keep a reference-counted vector of column results.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [0.25];
    auto metric = precisionView(entries[].sliced(1, 1));
    auto result = rcMahalanobisSquaredBatch(data[].sliced(1, 2), center[], metric);
    assert(result == [4, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto rcMahalanobisSquaredBatch(Summation summation = Summation.kbn, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, rcMahalanobisSquaredBatch!summation(x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate squared distances for observation columns using custom-allocated storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    allocator = Allocator for scratch and result storage.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either.
        The selected triangle must be finite and represent a positive-definite metric.

Returns:
    A custom-allocated vector of n squared distances in observation order.
    Zero-dimensional observations have distance zero.
    Dispose the returned slice field with allocator after its last use.

Throws:
    Exception if covariance factorization fails.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMahalanobisSquaredBatch(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)))
{
    return impl.batchAllocated!(Allocation.custom, true, M.form, M.triangle, summation)(
        allocator, x, center, metric.storage);
}

/// Release the result with the allocator that created it.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [4];
    auto metric = covarianceView(entries[].sliced(1, 1));
    auto result = makeMahalanobisSquaredBatch(Mallocator.instance, data[].sliced(1, 2), center[], metric);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [4, 1]);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto makeMahalanobisSquaredBatch(Summation summation = Summation.kbn, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, makeMahalanobisSquaredBatch!summation(allocator, x, center, label(metric)));
    static assert(0, missingMetricMessage!(accepts));
}

/++
Evaluate squared distances for observation columns using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense precision matrix (KBN by default), or a Cholesky
        factor of covariance or precision. A fast precision reduction also needs product storage.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n squared distances in observation order.
    work = Writable k-by-n matrix for centered observations.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one squared distance per observation to result.
    Zero-dimensional observations have distance zero.
+/
void mahalanobisSquaredBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)) &&
        vector!R && matrix!W && is(Element!X == Element!R) && is(Element!X == Element!W) &&
        M.form != CovarianceRepresentation.covariance && summation != Summation.fast)
{
    impl.preparedBatch!(true, M.form, M.triangle, summation)(
        x.asView, center.asView, metric.storage.asView, result.asView, work.asView);
}

/// Write one result per column using caller-owned scratch.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [4, 2];
    double[1] center = [0], entries = [0.5];
    double[2] result, scratch;
    auto metric = precisionCholeskyView(entries[].sliced(1, 1));
    mahalanobisSquaredBatchWithWorkspace(data[].sliced(1, 2), center[], metric, result[], scratch[].sliced(1, 2));
    assert(result == [4, 1]);
}

/++
Evaluate squared distances for observation columns using caller-supplied storage.

Params:
    summation = Precision quadratic-form reduction, compensated KBN by default.
    x = Matrix of k finite coordinates by n observation columns.
    center = Vector of k finite reference coordinates of the same floating-point type.
    metric = Labeled dense covariance or precision matrix of shape k-by-k.
        The selected triangle must be finite and represent a positive-definite metric.
    result = Writable vector of n squared distances in observation order.
    work = Writable k-by-n matrix for centered observations.
    extra = For covariance, a writable k-by-k workspace containing the Cholesky factor on success
        and possibly a partial factor on failure. For precision, a product workspace
        matching the shape and observation convention of work.
        All writable elements must be disjoint from each other and from the inputs.

Returns:
    void. Writes one squared distance per observation to result.
    Zero-dimensional observations have distance zero.

Throws:
    Exception if covariance factorization fails, before any result entry is written.
+/
void mahalanobisSquaredBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, R, W, E)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (acceptsPolicy!(M, summation) && impl.batchCompatible!(X, C, typeof(metric.storage)) &&
        vector!R && matrix!W && is(Element!X == Element!R) && is(Element!X == Element!W) &&
        matrix!E && (M.form == CovarianceRepresentation.covariance || M.form == CovarianceRepresentation.inverseCovariance) && is(Element!X == Element!E))
{
    impl.batchWithExtra!(true, M.form, M.triangle, summation)(
        x.asView, center.asView, metric.storage.asView, result.asView, work.asView, extra.asView);
}

// Diagnose a missing metric label before instantiating the distance kernels.
auto mahalanobisSquaredBatchWithWorkspace(Summation summation = Summation.kbn, X, C, M, W...)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric, scope auto ref W buffers)
    if (matrix!M && matrix!X)
{
    enum accepts(alias label) = __traits(compiles, mahalanobisSquaredBatchWithWorkspace!summation(x, center, label(metric), buffers));
    static assert(0, missingMetricMessage!(accepts));
}

// A stored lower precision matrix and its prepared factor must give the same
// distances, including when input rows and workspace columns have different strides.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: byDim;
    import mir.stat.internal.linearAlgebra.cholesky: choleskyDecompose;
    import std.math: fabs;
    double[4] data = [4, double.nan, 2, 2];
    double[6] observations = [1, 1, 0, 0, 1, 0], scratch, products;
    double[3] values;
    double[2] center = [0, 0];
    const metric = precisionView!(TrianglePolicy.lower)(data[].sliced(2, 2));
    auto factor = choleskyDecompose(metric);
    auto rows = observations[].sliced(3, 2);
    auto selected = rows.byDim!0;
    auto work = scratch[].sliced(2, 3).byDim!1;
    auto product = products[].sliced(3, 2).byDim!0;
    assert(mahalanobisSquaredFromPrecision!(TrianglePolicy.lower)(selected, center[], data[].sliced(2, 2)) == [10, 0, 4]);
    assert(mahalanobisSquaredFromPrecision(selected, center[], metric) == [10, 0, 4]);
    mahalanobisSquaredWithWorkspace!(Summation.fast)(selected, center[], metric, values[], work, product);
    assert(values == [10, 0, 4]);
    mahalanobisSquaredWithWorkspace(selected, center[], factor, values[], work);
    assert(fabs(values[0] - 10) < 1e-13 && values[1] == 0 && values[2] == 4);
    auto distances = rcMahalanobisBatch(rows.transposed, center[], factor);
    assert(fabs(distances[0] * distances[0] - 10) < 1e-13 && distances[1] == 0 && distances[2] == 2);

    auto raw = data[].sliced(2, 2);
    static assert(!__traits(compiles, mahalanobisSquared(rows[0], center[], raw)));
    static assert(!__traits(compiles, mahalanobisSquaredBatch(rows.transposed, center[], raw)));
    static assert(!__traits(compiles, mahalanobisSquaredFromCovariance(selected, center[], metric)));
    static assert(!__traits(compiles, mahalanobisSquaredFromPrecision!(TrianglePolicy.upper)(selected, center[], metric)));
    static assert(!__traits(compiles, mahalanobisSquared!(Summation.fast)(rows[0], center[], factor)));
    static assert(!__traits(compiles, mahalanobisSquaredWithWorkspace!(
        Summation.fast)(selected, center[], metric, values[], work)));
}

version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [1, 1], center = [0, 0], work, product;
    double[4] data = [4, 2, double.nan, 2];
    auto raw = data[].sliced(2, 2);
    enum accepts(alias label) = __traits(compiles,
        mahalanobisSquaredWithWorkspace!(Summation.fast)(x[], center[], label(raw), work[], product[]));
    static assert(missingMetricMessage!accepts ==
        "Specify the matrix representation. For precision, pass precisionView(metric).");
    static assert(!__traits(compiles, mahalanobisSquaredWithWorkspace!(
        Summation.fast)(x[], center[], raw, work[], product[])));
    assert(mahalanobisSquaredWithWorkspace!(Summation.fast)(
        x[], center[], precisionView(raw), work[], product[]) == 10);
}

version(mir_stat_test_distance)
@safe unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.exception: assertThrown;
    double[4] singular = [4, 2, double.nan, 1], factor;
    double[6] data = [1, 2, 3, 4, 5, 6], scratch;
    double[2] center = [0, 0];
    double[3] values = [7, 8, 9];
    auto covariance = covarianceView(singular[].sliced(2, 2));
    auto square = factor[].sliced(2, 2);
    assertThrown!Exception(mahalanobisBatchWithWorkspace(
        data[].sliced(2, 3), center[], covariance, values[], scratch[].sliced(2, 3), square));
    assert(values == [7, 8, 9]);
    assertThrown!Exception(mahalanobisSquaredWithWorkspace(
        data[].sliced(3, 2).byDim!0, center[], covariance, values[], scratch[].sliced(3, 2).byDim!0, square));
    assert(values == [7, 8, 9]);
}

// The typed dispatch must preserve the one-vector default for scalar precision
// and the single shared factor allocation for an entire covariance batch.
version(mir_stat_test_distance)
@system unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.exception: assertThrown;

    struct CountingAllocator
    {
        size_t allocations, releases;
        void[] allocate(size_t bytes)
        {
            ++allocations;
            return Mallocator.instance.allocate(bytes);
        }
        bool deallocate(void[] memory)
        {
            ++releases;
            return Mallocator.instance.deallocate(memory);
        }
    }
    double[1] center = [0], entries = [1];
    double[3] data = [2, 3, 4];
    auto rows = data[].sliced(3, 1);
    auto raw = entries[].sliced(1, 1);
    auto precision = precisionView(raw);
    CountingAllocator allocator;
    assert(makeMahalanobisSquared(allocator, rows[0], center[], precision) == 4);
    assert(allocator.allocations == 1 && allocator.releases == 1);
    assert(makeMahalanobisSquared!(Summation.fast)(allocator, rows[0], center[], precision) == 4);
    assert(allocator.allocations == 3 && allocator.releases == 3);

    auto covariance = covarianceView(raw);
    auto result = makeMahalanobisSquared(allocator, rows.byDim!0, center[], covariance);
    assert(result == [4, 9, 16]);
    assert(allocator.allocations == 6 && allocator.releases == 5);
    allocator.dispose(result.field);
    entries[0] = 0;
    assertThrown!Exception(makeMahalanobisSquared(allocator, rows.byDim!0, center[], covariance));
    assert(allocator.allocations == 9 && allocator.releases == 9);
}
