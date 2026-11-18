/++
Multivariate normal log-density for one real observation and a positive-definite
covariance or precision matrix. Supply a Cholesky factor to reuse it across
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
Evaluate a multivariate normal log-density using GC-managed temporary storage.
The observation and mean are equal-length vectors of finite floating-point values.
The selected metric triangle must be finite. The metric is a square
matrix of the same order and element type. Raw covariance and precision inputs
are factored once per call and must be positive definite. Supplied factors
must have a positive finite diagonal. A precision factor is a Cholesky factor
of the precision matrix. Singular covariance is outside the supported domain.
The zero-dimensional log-density is zero.

For an observation with k coordinates, the result is minus one half of
the squared Mahalanobis distance plus the normalization term. That term
is -k * log(2 * PI) / 2 minus half the covariance log-determinant.
For a precision factor the log-determinant contribution changes sign.
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

/// Use reference-counted scratch storage.
Element!X rcMultivariateNormalLPDF(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && compatible!(X, C, M))
{
    int allocator;
    return evaluate!(Allocation.rc, form, triangle)(allocator, x.asView, mean.asView, metric.asView);
}

/// Evaluate a raw covariance in @nogc code using reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;

    double[1] x = [3], mean = [1], covariance = [4];
    const result = rcMultivariateNormalLPDF(x[], mean[], covariance[].sliced(1, 1));
    assert(fabs(result - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);
}

/// Use caller-selected allocation for scratch. All storage is disposed on return or exception.
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

/// Evaluate from a supplied factor using one disjoint writable work vector of observation length.
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
@safe pure @nogc unittest
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

/// Evaluate from a raw metric using a work vector and a writable factor matrix
/// of matching shape. Both buffers must be disjoint from each other and all inputs.
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
