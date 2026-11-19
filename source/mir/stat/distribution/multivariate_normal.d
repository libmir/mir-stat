/++
Multivariate normal log-density for one real observation and a positive-definite
covariance or precision matrix. Precision is the inverse of covariance.
Supply a Cholesky factor of either matrix to reuse it across
calls. In zero dimensions the log-density is zero.
These functions accept dense square matrix views, including strided slices.
Only the selected triangle is read. Upper factors represent a matrix
as U.transpose * U, and lower factors represent it as L * L.transpose.
As in $(D mir.stat.distance), subtracting the mean in the input
element type can overflow before the factor is applied.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.multivariate_normal;

version(mir_stat_test) version = mir_stat_test_multivariate_normal;
version(mir_stat_test_blas) version = mir_stat_test_multivariate_normal;

public import mir.stat.internal.linearAlgebra.types: CovarianceRepresentation, TrianglePolicy;
private import mir.stat.distance: asView, compatible, matrix, vector, Element, mahalanobisWithWorkspace;
private import mir.stat.internal.linearAlgebra.cholesky: choleskyDecomposeInto, logDeterminantFromCholesky;
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
private Element!X evaluate(Allocation policy, CovarianceRepresentation form, TrianglePolicy triangle,
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
    auto observation = x.asView;
    auto center = mean.asView;
    auto matrixView = metric.asView;
    auto workView = work.asView;
    auto factor = factorWork.asView;
    assert(observation.length == center.length && observation.length == workView.length);
    assert(matrixView.length!0 == observation.length && matrixView.length!1 == observation.length);
    choleskyDecomposeInto!triangle(matrixView, factor);
    enum factorForm = form == CovarianceRepresentation.covariance
        ? CovarianceRepresentation.covarianceCholesky : CovarianceRepresentation.inverseCovarianceCholesky;
    return multivariateNormalLPDFWithWorkspace!(factorForm, triangle)(observation, center, factor, workView);
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
