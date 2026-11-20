/++
Multivariate normal log-density for real observations and a positive-definite
covariance or precision matrix. Precision is the inverse of covariance.

For example, a test rig records inlet, chamber, and outlet temperatures in
degrees Celsius on each of 100 runs. Each run is one observation with three
coordinates in that order. The second run might yield `[12, 21, 35]` against
a mean of `[10, 20, 30]`. A batch call returns one log density per run under
the same three-dimensional normal distribution.

Supply a Cholesky factor of either matrix to reuse it across
calls. In zero dimensions the log-density is zero.
For a single observation and a raw matrix, $(LREF tryMultivariateNormalLPDF),
$(LREF rcTryMultivariateNormalLPDF), $(LREF makeTryMultivariateNormalLPDF), and
$(LREF tryMultivariateNormalLPDFWithWorkspace) return a
$(LREF MultivariateNormalLPDFResult). Check its `failedPivot` field
to see whether factorization succeeded. With custom iterators or allocators,
function attributes such as `nothrow` depend on the operations they supply.

With one run per row, pass `rows.byDim!0` and a 100-by-3 scratch matrix
selected as `workRows.byDim!0`. The function writes the second run's log
density to `result[1]`. A transpose view presents the rows to the batch
routine as columns while keeping each run's readings together in row-major
storage. For a 3-by-100 matrix with one run per column, pass the matrix
directly to the corresponding `Batch` function.

The scratch matrix holds the centered readings. Subtracting the mean from the
second run writes `[2, 1, 5]` into its scratch row. The batch then transforms
all 100 scratch rows in place using the shared Cholesky factor. With a
compatible layout, BLAS can perform this step in one triangular solve or
multiply. A raw covariance or precision matrix is factored once, and all
100 log densities use the same log determinant.

The full batch requires 300 observation scratch elements (100 runs × 3 readings per run).
Processing ten runs per call requires 30. Each call can reuse the prepared factor and computes
its log determinant once. A full batch gives BLAS a larger operation. The
faster choice depends on the dimensions, storage layout, and backend.

The covariance and precision inputs accept dense square matrix views, including strided slices.
Only the selected triangle is read. Upper factors represent a matrix
as U.transpose * U, and lower factors represent it as L * L.transpose.
As in `mir.stat.distance`, subtracting the mean in the input
element type can overflow before the factor is applied.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.multivariate_normal;

version(mir_stat_test) version = mir_stat_test_multivariate_normal;
version(mir_stat_test_blas) version = mir_stat_test_multivariate_normal;

public import mir.stat.internal.linearAlgebra.types: CovarianceRepresentation, TrianglePolicy;
private import mir.stat.internal.linearAlgebra.types: isObservationSelection, ObservationColumns, observationColumns;
private import mir.stat.distance: asView, compatible, matrix, vector, Element, mahalanobisWithWorkspace;
private import mir.stat.internal.linearAlgebra.cholesky: tryCholeskyDecomposeInto, logDeterminantFromCholesky;
private import mir.stat.internal.linearAlgebra.allocation: Allocation, allocate;

/// Whether the caller supplied a Cholesky factor.
private enum prepared(CovarianceRepresentation form) =
    form == CovarianceRepresentation.covarianceCholesky ||
    form == CovarianceRepresentation.inverseCovarianceCholesky;

/++
Evaluate the multivariate normal log density using GC-managed scratch.
For covariance S and Mahalanobis distance d, the result is
$(D -0.5 * (d * d + k * log(2 * PI) + log(det(S)))).
A precision matrix contributes its log determinant with the opposite sign.

Params:
    form = Representation of metric (covariance by default).
    triangle = Triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X multivariateNormalLPDF(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && compatible!(X, C, M))
{
    int allocator;
    return evaluate!(Allocation.gc, form, triangle)(allocator, x.asView, mean.asView, metric.asView);
}

/// Equivalent covariance and precision inputs give the same normalized density.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.constant: LOGSQRT2PI;
    import std.math: fabs, log, sqrt;
    double[2] x = [1, 2], mean = [0, 0];
    double[4] cov = [4, 2, double.nan, 5], cf = [2, 1, double.nan, 2];
    double[4] prec = [0.3125, -0.125, double.nan, 0.25];
    double[4] pf = [sqrt(5.0) / 4, -0.5 / sqrt(5.0), double.nan, 1 / sqrt(5.0)];
    const expected = -2 * double(LOGSQRT2PI) - log(4.0) - 0.8125 / 2;
    alias R = CovarianceRepresentation;
    assert(fabs(multivariateNormalLPDF(x[], mean[], cov[].sliced(2, 2)) - expected) < 1e-14);
    assert(fabs(rcMultivariateNormalLPDF!(R.covarianceCholesky)(x[], mean[], cf[].sliced(2, 2)) - expected) < 1e-14);
    assert(fabs(rcMultivariateNormalLPDF!(R.inverseCovariance)(x[], mean[], prec[].sliced(2, 2)) - expected) < 1e-14);
    assert(fabs(multivariateNormalLPDF!(R.inverseCovarianceCholesky)(x[], mean[], pf[].sliced(2, 2)) - expected) < 1e-14);
}

/// Supply the lower Cholesky factor of a precision matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.constant: LOGSQRT2PI;
    import std.math: fabs, log;

    double[2] x = [1, 2], mean = [0, 0];
    double[4] lower = [2, double.nan, 1, 3];
    // L.transpose * x = [4, 6] and det(L) = 6.
    const result = multivariateNormalLPDF!(CovarianceRepresentation.inverseCovarianceCholesky,
        TrianglePolicy.lower)(x[], mean[], lower[].sliced(2, 2));
    assert(fabs(result - (log(6.0) - 2 * double(LOGSQRT2PI) - 26)) < 1e-14);
}

/++
Evaluate the multivariate normal log density using reference-counted scratch.

Params:
    form = Representation of metric (covariance by default).
    triangle = Triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X rcMultivariateNormalLPDF(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && compatible!(X, C, M))
{
    int allocator;
    return evaluate!(Allocation.rc, form, triangle)(allocator, x.asView, mean.asView, metric.asView);
}

/// Evaluate a covariance factor using reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;

    double[1] x = [3], mean = [1], factor = [2];
    const result = rcMultivariateNormalLPDF!(CovarianceRepresentation.covarianceCholesky)(
        x[], mean[], factor[].sliced(1, 1));
    assert(fabs(result - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);
}

/++
Evaluate the multivariate normal log density using a caller-selected allocator.

Params:
    form = Representation of metric (covariance by default).
    triangle = Triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.
    allocator = Allocator for scratch storage, which is released on return or exception.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X makeMultivariateNormalLPDF(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && compatible!(X, C, M))
{
    return evaluate!(Allocation.custom, form, triangle)(allocator, x.asView, mean.asView, metric.asView);
}

/// Custom allocation also supports raw precision input and empty observations.
version(mir_stat_test_multivariate_normal)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    alias R = CovarianceRepresentation;
    float[1] x = [3], mean = [1], precision = [0.25f];
    const value = makeMultivariateNormalLPDF!(R.inverseCovariance)(
        Mallocator.instance, x[], mean[], precision[].sliced(1, 1));
    static assert(is(typeof(value) == const(float)));
    assert(fabs(value - normalLPDF(3f, 1f, 2f)) < 4 * float.epsilon);
    assert(makeMultivariateNormalLPDF(Mallocator.instance, x[0 .. 0], mean[0 .. 0],
        precision[0 .. 0].sliced(0, 0)) == 0);
}

/// Use a custom allocator for scratch with a prepared precision factor.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs;

    float[1] x = [3], mean = [1], factor = [0.5f];
    const result = makeMultivariateNormalLPDF!(CovarianceRepresentation.inverseCovarianceCholesky)(
        Mallocator.instance, x[], mean[], factor[].sliced(1, 1));
    assert(fabs(result - normalLPDF(3f, 1f, 2f)) < 4 * float.epsilon);
}

/// Allocate one work vector and, for a raw metric, one factor matrix.
private auto evaluate(Allocation policy, CovarianceRepresentation form, TrianglePolicy triangle,
    bool reportFailure = false,
    A, X, C, M)(ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
{
    import std.experimental.allocator: dispose;
    assert(x.length == mean.length && metric.length!0 == x.length && metric.length!1 == x.length);
    auto work = allocate!(policy, Element!X)(allocator, [x.length]);
    scope(exit) static if (policy == Allocation.custom) allocator.dispose(work.field);
    static if (prepared!form)
        return multivariateNormalLPDFWithWorkspace!(form, triangle)(x, mean, metric, work);
    else
    {
        auto factor = allocate!(policy, Element!X)(allocator, metric.shape);
        scope(exit) static if (policy == Allocation.custom) allocator.dispose(factor.field);
        static if (reportFailure)
            return tryMultivariateNormalLPDFWithWorkspace!(form, triangle)(x, mean, metric, work, factor);
        else
            return multivariateNormalLPDFWithWorkspace!(form, triangle)(x, mean, metric, work, factor);
    }
}

private T logDensityFromDistance(CovarianceRepresentation form, T)(T distance, T dimensionTerm, T halfLogDet)
{
    // Halve before squaring so the negative half-square can remain finite.
    const base = (-T(0.5) * distance) * distance - dimensionTerm;
    static if (form == CovarianceRepresentation.covarianceCholesky)
        return base - halfLogDet;
    else
        return base + halfLogDet;
}

private T scalarLogDensity(CovarianceRepresentation form, T, F)(T distance, F factor)
{
    import mir.stat.constant: LOGSQRT2PI;
    return logDensityFromDistance!form(distance, T(factor.length!0) * T(LOGSQRT2PI),
        T(0.5) * logDeterminantFromCholesky(factor));
}

/++
Evaluate the log density from a prepared factor using caller-owned scratch.
The elements of work must be disjoint from each other and from all inputs.

Params:
    form = Either covarianceCholesky or inverseCovarianceCholesky.
    triangle = Triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    factor = Dense k-by-k Cholesky factor of the covariance or precision matrix,
        with the same element type as x. The selected triangle must be finite
        and its diagonal entries positive.
    work = Writable vector of k elements of the same type as x for the centered
        and transformed observation.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.
+/
Element!X multivariateNormalLPDFWithWorkspace(CovarianceRepresentation form, TrianglePolicy triangle = TrianglePolicy.upper,
    X, C, M, W)(scope auto ref X x, scope auto ref C mean, scope auto ref M factor, scope auto ref W work)
    if (prepared!form && matrix!M && compatible!(X, C, M) && vector!W && is(Element!X == Element!W))
{
    auto factorView = factor.asView;
    auto observation = x.asView;
    const distance = mahalanobisWithWorkspace!(form, triangle)(observation, mean, factorView, work);
    return scalarLogDensity!form(distance, factorView);
}

/// Reuse a lower covariance factor for successive observations.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.constant: LOGSQRT2PI;
    import std.math: fabs, log;
    double[2] mean = [1, -2], scratch;
    double[4] storage = [2, double.nan, 1, 3];
    auto factor = storage[].sliced(2, 2);
    const atMean = -2 * double(LOGSQRT2PI) - log(6.0);

    // The second observation is mean + L * [1, 2], so its squared distance is 5.
    double[2] x = [3, 5];
    enum form = CovarianceRepresentation.covarianceCholesky;
    const first = multivariateNormalLPDFWithWorkspace!(form, TrianglePolicy.lower)(
        mean[], mean[], factor, scratch[]);
    const second = multivariateNormalLPDFWithWorkspace!(form, TrianglePolicy.lower)(
        x[], mean[], factor, scratch[]);
    assert(fabs(first - atMean) < 1e-14);
    assert(fabs(second - (atMean - 2.5)) < 1e-14);
}

/++
Factor a raw covariance or precision matrix and evaluate the log density.
Both scratch buffers must have nonoverlapping elements and be disjoint from
one another and from all inputs.

Params:
    form = Either covariance (default) or inverseCovariance.
    triangle = Triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense, positive-definite k-by-k matrix of the same element type as x
        with finite entries in the selected triangle.
    work = Writable vector of k elements of the same type as x for the centered
        and transformed observation.
    factorWork = Writable matrix of the same shape and element type as metric.
        On success its selected triangle holds the Cholesky factor.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization fails. In that case factorWork may
    contain a partial factorization.
+/
Element!X multivariateNormalLPDFWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(scope auto ref X x, scope auto ref C mean,
    scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (!prepared!form && matrix!M && compatible!(X, C, M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    import mir.exception: enforce;
    const result = tryMultivariateNormalLPDFWithWorkspace!(form, triangle)(x, mean, metric, work, factorWork);
    enforce!"Cholesky factorization failed"(result.failedPivot == 0);
    return result.logDensity;
}

/// Supply a work vector and a factor matrix to evaluate a raw covariance.
version(mir_stat_test_multivariate_normal)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;

    double[1] x = [3], mean = [1], covariance = [4];
    double[1] work, factorStorage;
    auto factorWork = factorStorage[].sliced(1, 1);
    const result = multivariateNormalLPDFWithWorkspace(
        x[], mean[], covariance[].sliced(1, 1), work[], factorWork);
    assert(fabs(result - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);
}

// Halving the distance before squaring preserves a finite log-density in the far tail.
version(mir_stat_test_multivariate_normal)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [0x1p512], mean = [0], diagonal = [1], work;
    auto factor = diagonal[].sliced(1, 1);
    enum form = CovarianceRepresentation.covarianceCholesky;
    assert(multivariateNormalLPDFWithWorkspace!form(x[], mean[], factor, work[]) == -0x1p1023);
}

/++
Evaluate selected rows or columns using a prepared Cholesky factor.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    form = covarianceCholesky or inverseCovarianceCholesky.
    triangle = Selected matrix triangle, upper by default.
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    factor = Dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n log densities in selection order.
    work = Writable selection of n scratch vectors of length k.

Returns:
    void. Writes one log density per selected observation.
+/
void multivariateNormalLPDFWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        multivariateNormalLPDFBatchWithWorkspace!(form, triangle)(
            ObservationColumns!X.init, mean, factor, result, ObservationColumns!W.init)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    multivariateNormalLPDFBatchWithWorkspace!(form, triangle)(columns, mean, factor, result, scratch);
}

/// Select two row observations and use one precision factor for both.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[2] data = [1, 3], values, scratch;
    double[1] mean = [1], factor = [0.5];
    auto rows = data[].sliced(2, 1);
    auto workRows = scratch[].sliced(2, 1);
    multivariateNormalLPDFWithWorkspace!(CovarianceRepresentation.inverseCovarianceCholesky)(
        rows.byDim!0, mean[], factor[].sliced(1, 1), values[], workRows.byDim!0);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(data[j], 1.0, 2.0)) < 1e-14);
}

/++
Result of evaluating a log density from a raw covariance or precision matrix.
Check `failedPivot` first. If it is zero, use `logDensity`.
For an empty observation both fields are zero.

Params:
    T = Floating-point element type of the observation.
+/
struct MultivariateNormalLPDFResult(T)
{
    /// Natural log density in the input element type, or NaN on factorization failure.
    T logDensity;
    /++
    Zero when Cholesky factorization succeeds. A value from 1 through the
    matrix dimension identifies the position where it reports numerical
    failure. For example, `failedPivot == 2` identifies the second
    diagonal position, which has index 1 in a D matrix.

    A pivot is the diagonal value after accounting for earlier factor
    entries. It must be positive before taking its square root. Singular
    or indefinite matrices can fail this check. A nonfinite value computed
    during factorization can also cause failure.
    +/
    size_t failedPivot;
}

/++
Evaluate the log density using caller-owned scratch and report Cholesky failure.
With ordinary floating-point slices for inputs and scratch, the call is `nothrow @nogc`.
Each workspace must have nonoverlapping elements and be disjoint from the
other workspace and from all inputs.

Params:
    form = Raw matrix representation, either covariance (default) or inverseCovariance.
    triangle = Logical triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x and finite
        entries in the selected triangle.
    work = Writable scratch vector of k elements of the same type as x.
    factorWork = Writable matrix of the same shape and element type as metric.
        On success its selected triangle holds the factor. On failure it may
        contain a partial factorization.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) tryMultivariateNormalLPDFWithWorkspace(
    CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref W work, scope auto ref F factorWork)
    if (!prepared!form && matrix!M && compatible!(X, C, M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    alias Result = MultivariateNormalLPDFResult!(Element!X);
    auto observation = x.asView;
    auto center = mean.asView;
    auto matrixView = metric.asView;
    auto workView = work.asView;
    auto factor = factorWork.asView;
    static if (form == CovarianceRepresentation.covariance)
    {
        import mir.stat.distance: tryCovarianceDistanceWithWorkspace;
        Element!X distance;
        const pivot = tryCovarianceDistanceWithWorkspace!(false, triangle)(
            observation, center, matrixView, distance, workView, factor);
        if (pivot) return Result(Element!X.nan, pivot);
        return Result(scalarLogDensity!(CovarianceRepresentation.covarianceCholesky)(distance, factor), 0);
    }
    else
    {
        assert(observation.length == center.length && observation.length == workView.length);
        assert(matrixView.length!0 == observation.length && matrixView.length!1 == observation.length);
        const pivot = tryCholeskyDecomposeInto!triangle(matrixView, factor);
        if (pivot) return Result(Element!X.nan, pivot);
        enum factorForm = CovarianceRepresentation.inverseCovarianceCholesky;
        return Result(multivariateNormalLPDFWithWorkspace!(factorForm, triangle)(
            observation, center, factor, workView), 0);
    }
}

/// A one-dimensional precision of one quarter gives a standard deviation of two.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    float[1] x = [3], mean = [1], precision = [0.25f], work, factor;
    const result = tryMultivariateNormalLPDFWithWorkspace!(CovarianceRepresentation.inverseCovariance)(
        x[], mean[], precision[].sliced(1, 1), work[], factor[].sliced(1, 1));
    static assert(is(typeof(result.logDensity) == const(float)));
    assert(result.failedPivot == 0);
    assert(fabs(result.logDensity - normalLPDF(3f, 1f, 2f)) < 4 * float.epsilon);
}

version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import std.math: PI, fabs, isNaN, log;
    double[2] x = [1, 2], mean = [0, 0];
    double[2] work;
    double[4] factor, data = [4, double.nan, 2, 5];
    auto metric = data[].sliced(2, 2);
    // det(metric) = 16. The quadratic form is 13/16 for covariance and 32 for precision.
    const logTwoPi = log(2 * double(PI));
    static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.inverseCovariance])
    {{
        static if (form == CovarianceRepresentation.covariance)
            const expected = -logTwoPi - log(4.0) - 0.5 * (13.0 / 16);
        else
            const expected = -logTwoPi + log(4.0) - 0.5 * 32;
        const result = tryMultivariateNormalLPDFWithWorkspace!(form, TrianglePolicy.lower)(
            x[], mean[], metric, work[], factor[].sliced(2, 2));
        assert(result.failedPivot == 0);
        assert(fabs(result.logDensity - expected) < 1e-14);
    }}
    data[3] = 1;
    const singular = tryMultivariateNormalLPDFWithWorkspace!(CovarianceRepresentation.covariance, TrianglePolicy.lower)(
        x[], mean[], metric, work[], factor[].sliced(2, 2));
    assert(singular.failedPivot == 2 && isNaN(singular.logDensity));
    assertThrown!Exception(multivariateNormalLPDF!(CovarianceRepresentation.covariance, TrianglePolicy.lower)(
        x[], mean[], metric));
    const empty = tryMultivariateNormalLPDFWithWorkspace(x[0 .. 0], mean[0 .. 0],
        data[0 .. 0].sliced(0, 0), work[0 .. 0], factor[0 .. 0].sliced(0, 0));
    assert(empty.failedPivot == 0 && empty.logDensity == 0);
    x[0] = double.max;
    const tail = tryMultivariateNormalLPDFWithWorkspace(x[0 .. 1], mean[0 .. 1],
        data[0 .. 1].sliced(1, 1), work[0 .. 1], factor[0 .. 1].sliced(1, 1));
    assert(tail.failedPivot == 0 && tail.logDensity == -double.infinity);
}

/++
Evaluate selected rows or columns using one shared covariance or precision matrix.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    form = covariance (default) or inverseCovariance.
    triangle = Selected matrix triangle, upper by default.
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    metric = Dense k-by-k matrix with finite selected triangle.
    result = Writable vector of n log densities in selection order.
    work = Writable selection of n scratch vectors of length k.
    factorWork = Writable k-by-k factor workspace. May contain a partial factor on failure.

Returns:
    Zero on success or the one-based failed pivot. Failure leaves result untouched.
+/
size_t tryMultivariateNormalLPDFWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        tryMultivariateNormalLPDFBatchWithWorkspace!(form, triangle)(
            ObservationColumns!X.init, mean, metric, result, ObservationColumns!W.init, factorWork)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    return tryMultivariateNormalLPDFBatchWithWorkspace!(form, triangle)(columns, mean, metric, result, scratch, factorWork);
}

/// Evaluate the inlet and chamber temperatures for three test runs.
/// Each row contains the first two readings in the module's temperature example.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: PI, fabs, log;
    double[6] data = [10, 20, 12, 21, 10, 24], scratch;
    double[2] mean = [10, 20];
    double[4] covariance = [4, 2, 2, 5], factor;
    double[3] logDensities;
    auto rows = data[].sliced(3, 2);
    auto workRows = scratch[].sliced(3, 2);
    const pivot = tryMultivariateNormalLPDFWithWorkspace(rows.byDim!0, mean[],
        covariance[].sliced(2, 2), logDensities[], workRows.byDim!0, factor[].sliced(2, 2));
    assert(pivot == 0);
    // The first run is at the mean. The covariance determinant is 16.
    const atMean = -log(8 * double(PI));
    assert(fabs(logDensities[0] - atMean) < 1e-14);
    // The second run's [12, 21] centers to [2, 1] and transforms to [1, 0].
    assert(fabs(logDensities[1] - (atMean - 0.5)) < 1e-14);
    // The third run's transformed row is [0, 2].
    assert(fabs(logDensities[2] - (atMean - 2)) < 1e-14);
}

private enum batchBuffers(X, C, M, R, W) = matrix!X && matrix!M && vector!C &&
    vector!R && matrix!W && is(Element!X == Element!C) && is(Element!X == Element!M) &&
    is(Element!X == Element!R) && is(Element!X == Element!W);

private void batchLogDensities(CovarianceRepresentation form, R, F)(R values, F factor)
{
    import mir.stat.constant: LOGSQRT2PI;
    alias T = Element!R;
    const dimensionTerm = T(factor.length!0) * T(LOGSQRT2PI);
    const halfLogDet = T(0.5) * logDeterminantFromCholesky(factor);
    foreach (j; 0 .. values.length)
        values[j] = logDensityFromDistance!form(values[j], dimensionTerm, halfLogDet);
}

/++
Write one log density per observation column using a shared Cholesky factor.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covarianceCholesky or inverseCovarianceCholesky (factor of the precision matrix).
    triangle = Triangle containing the factor, upper by default.
    x = Dense k-by-n matrix of finite observations. Column j is observation j.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    factor = Dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n log densities in column order.
    work = Writable k-by-n scratch for centering and transforming the observations.

Returns:
    void. Writes each log density to result. In zero dimensions each value is zero.
+/
void multivariateNormalLPDFBatchWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (prepared!form && batchBuffers!(X, C, M, R, W))
{
    import mir.stat.distance: mahalanobisBatchWithWorkspace;
    auto values = result.asView;
    auto columns = x.asView;
    auto factorView = factor.asView;
    mahalanobisBatchWithWorkspace!(form, triangle)(columns, mean, factorView, values, work);
    batchLogDensities!form(values, factorView);
}

/// Two columns, each with one coordinate and standard deviation two.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[2] x = [1, 3], values, scratch;
    double[1] mean = [1], factor = [2];
    multivariateNormalLPDFBatchWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(
        x[].sliced(1, 2), mean[], factor[].sliced(1, 1), values[], scratch[].sliced(1, 2));
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(x[j], 1.0, 2.0)) < 1e-14);
}

/++
Factor a covariance or precision matrix once, then write one log density per column.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covariance (default) or inverseCovariance.
    triangle = Triangle to read, upper by default.
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    metric = Dense k-by-k matrix with finite entries in the selected triangle.
    result = Writable vector of n log densities in column order. A failed factorization
        leaves every entry untouched.
    work = Writable k-by-n scratch for centering and transforming the observations.
    factorWork = Writable k-by-k scratch. Its selected triangle holds the factor on
        success and may contain a partial factor on failure.

Returns:
    Zero on success, or the one-based failed Cholesky pivot. The matrix is checked
    even for an empty batch. In zero dimensions each log density is zero.
+/
size_t tryMultivariateNormalLPDFBatchWithWorkspace(
    CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if ((form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        batchBuffers!(X, C, M, R, W) && matrix!F && is(Element!X == Element!F))
{
    auto columns = x.asView;
    auto matrixView = metric.asView;
    auto factor = factorWork.asView;
    static if (form == CovarianceRepresentation.covariance)
    {
        import mir.stat.distance: tryCovarianceDistanceBatchWithWorkspace;
        auto values = result.asView;
        const pivot = tryCovarianceDistanceBatchWithWorkspace!(false, triangle)(
            columns, mean.asView, matrixView, values, work.asView, factor);
        if (pivot) return pivot;
        batchLogDensities!(CovarianceRepresentation.covarianceCholesky)(values, factor);
    }
    else
    {
        assert(columns.length!0 == mean.asView.length && columns.length!0 == matrixView.length!0);
        assert(columns.shape == work.asView.shape && columns.length!1 == result.asView.length);
        const pivot = tryCholeskyDecomposeInto!triangle(matrixView, factor);
        if (pivot) return pivot;
        multivariateNormalLPDFBatchWithWorkspace!(CovarianceRepresentation.inverseCovarianceCholesky, triangle)(
            columns, mean, factor, result, work);
    }
    return 0;
}

/// A singular covariance fails before either output is written.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] x = [1, 2, 3, 4], metric = [1, 1, double.nan, 1], scratch, factor;
    double[2] mean = [0, 0], values = [7, 8];
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(x[].sliced(2, 2), mean[],
        metric[].sliced(2, 2), values[], scratch[].sliced(2, 2), factor[].sliced(2, 2)) == 2);
    assert(values == [7, 8]);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim;
    import std.math: fabs;
    alias R = CovarianceRepresentation;

    double[6] data = [3, 7, 1, -2, 6, 9], scratch;
    double[4] lower = [4, double.nan, 2, 5], factor;
    double[2] mean = [1, -2], vectorWork;
    double[3] values;
    auto rows = data[].sliced(3, 2);
    auto workColumns = scratch[].sliced(2, 3);
    static foreach (form; [R.covariance, R.inverseCovariance])
    {{
        const pivot = tryMultivariateNormalLPDFWithWorkspace!(form, TrianglePolicy.lower)(
            rows.alongDim!1, mean[], lower[].sliced(2, 2), values[],
            workColumns.alongDim!0, factor[].sliced(2, 2));
        assert(pivot == 0);
        enum preparedForm = form == R.covariance ? R.covarianceCholesky : R.inverseCovarianceCholesky;
        foreach (j; 0 .. values.length)
        {
            auto expected = multivariateNormalLPDFWithWorkspace!(preparedForm, TrianglePolicy.lower)(
                rows[j], mean[], factor[].sliced(2, 2), vectorWork[]);
            assert(fabs(values[j] - expected) <= 1e-13 * (1 + fabs(expected)));
        }
    }}
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [0x1p512], mean = [0], metric = [1], work, factor, result;
    auto unit = metric[].sliced(1, 1);
    auto f = factor[].sliced(1, 1);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(x[].sliced(1, 1), mean[], unit,
        result[], work[].sliced(1, 1), f) == 0);
    assert(result[0] == -0x1p1023);

    auto noCoordinates = x[0 .. 0].sliced(0, 1);
    auto emptyFactor = factor[0 .. 0].sliced(0, 0);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noCoordinates, mean[0 .. 0],
        metric[0 .. 0].sliced(0, 0), result[], work[0 .. 0].sliced(0, 1), emptyFactor) == 0);
    assert(result[0] == 0);

    auto noObservations = x[0 .. 0].sliced(1, 0);
    auto emptyWork = work[0 .. 0].sliced(1, 0);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noObservations, mean[], unit,
        result[0 .. 0], emptyWork, f) == 0);
    unit[0, 0] = 0;
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noObservations, mean[], unit,
        result[0 .. 0], emptyWork, f) == 1);
}

/++
Evaluate the log density using GC-managed scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow`.

Params:
    form = Raw matrix representation, either covariance (default) or inverseCovariance.
    triangle = Logical triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) tryMultivariateNormalLPDF(
    CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (!prepared!form && matrix!M && compatible!(X, C, M))
{
    int allocator;
    return evaluate!(Allocation.gc, form, triangle, true)(allocator, x.asView, mean.asView, metric.asView);
}

/// The first diagonal gives a factor entry of 2. The second pivot is 1 - (2/2) * (2/2) = 0.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    double[2] x = [1, 2], mean = [0, 0];
    double[4] covariance = [4, 2, double.nan, 1];
    const result = tryMultivariateNormalLPDF(x[], mean[], covariance[].sliced(2, 2));
    assert(result.failedPivot == 2 && isNaN(result.logDensity));
}

/++
Evaluate the log density using reference-counted scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow @nogc`.

Params:
    form = Raw matrix representation, either covariance (default) or inverseCovariance.
    triangle = Logical triangle to read, upper by default.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) rcTryMultivariateNormalLPDF(
    CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (!prepared!form && matrix!M && compatible!(X, C, M))
{
    int allocator;
    return evaluate!(Allocation.rc, form, triangle, true)(allocator, x.asView, mean.asView, metric.asView);
}

/// Evaluate a raw covariance using reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs, isNaN;

    double[1] x = [3], mean = [1], covariance = [4];
    auto metric = covariance[].sliced(1, 1);
    const result = rcTryMultivariateNormalLPDF(x[], mean[], metric);
    assert(result.failedPivot == 0);
    assert(fabs(result.logDensity - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = rcTryMultivariateNormalLPDF(x[], mean[], metric);
    assert(singular.failedPivot == 1 && isNaN(singular.logDensity));
}

/++
Evaluate the log density using a caller-selected scratch allocator and report Cholesky failure.
The call is `nothrow @nogc` with ordinary floating-point slices and
`Mallocator`. With other allocators or iterators, the inferred attributes depend
on their operations.

Params:
    form = Raw matrix representation, either covariance (default) or inverseCovariance.
    triangle = Logical triangle to read, upper by default.
    allocator = Allocator used to obtain and release temporary storage.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) makeTryMultivariateNormalLPDF(
    CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (!prepared!form && matrix!M && compatible!(X, C, M))
{
    return evaluate!(Allocation.custom, form, triangle, true)(allocator, x.asView, mean.asView, metric.asView);
}

/// Evaluate a raw precision using Mallocator for scratch.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs, isNaN;

    double[1] x = [3], mean = [1], precision = [0.25];
    auto metric = precision[].sliced(1, 1);
    enum form = CovarianceRepresentation.inverseCovariance;
    const result = makeTryMultivariateNormalLPDF!form(Mallocator.instance, x[], mean[], metric);
    assert(result.failedPivot == 0);
    assert(fabs(result.logDensity - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = makeTryMultivariateNormalLPDF!form(Mallocator.instance, x[], mean[], metric);
    assert(singular.failedPivot == 1 && isNaN(singular.logDensity));
}
