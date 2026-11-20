/++
Multivariate normal log-density for one real observation and a positive-definite
covariance or precision matrix. Precision is the inverse of covariance.
Supply a Cholesky factor of either matrix to reuse it across
calls. In zero dimensions the log-density is zero.
For a raw matrix, $(LREF tryMultivariateNormalLPDF),
$(LREF rcTryMultivariateNormalLPDF), $(LREF makeTryMultivariateNormalLPDF), and
$(LREF tryMultivariateNormalLPDFWithWorkspace) return a
$(LREF MultivariateNormalLPDFResult). Check its `failedPivot` field
to see whether factorization succeeded. With custom iterators or allocators,
function attributes such as `nothrow` depend on the operations they supply.
These functions accept dense square matrix views, including strided slices.
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
    import mir.stat.constant: LOGSQRT2PI;
    alias T = Element!X;
    auto factorView = factor.asView;
    auto observation = x.asView;
    const distance = mahalanobisWithWorkspace!(form, triangle)(observation, mean, factorView, work);
    const halfLogDet = T(0.5) * logDeterminantFromCholesky(factorView);
    const base = (-T(0.5) * distance) * distance - T(observation.length) * T(LOGSQRT2PI);
    static if (form == CovarianceRepresentation.covarianceCholesky)
        return base - halfLogDet;
    else
        return base + halfLogDet;
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
    assert(observation.length == center.length && observation.length == workView.length);
    assert(matrixView.length!0 == observation.length && matrixView.length!1 == observation.length);
    const failedPivot = tryCholeskyDecomposeInto!triangle(matrixView, factor);
    if (failedPivot)
        return Result(Element!X.nan, failedPivot);
    enum factorForm = form == CovarianceRepresentation.covariance
        ? CovarianceRepresentation.covarianceCholesky : CovarianceRepresentation.inverseCovarianceCholesky;
    const logDensity = multivariateNormalLPDFWithWorkspace!(factorForm, triangle)(observation, center, factor, workView);
    return Result(logDensity, 0);
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

/// The first diagonal gives a factor entry of 2. The second pivot is 1 - (2/2)Â² = 0.
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
