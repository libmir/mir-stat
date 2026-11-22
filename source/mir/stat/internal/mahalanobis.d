/++
Mahalanobis distance kernels and structural matrix adapters.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.mahalanobis;

version(mir_stat_test) version = mir_stat_test_distance;
version(mir_stat_test_linearAlgebra) version = mir_stat_test_distance;
version(mir_stat_test_blas) version = mir_stat_test_distance;

/// Shared matrix policies; the algebra operations themselves remain internal.
public import mir.stat.internal.linearAlgebra.types: CovarianceRepresentation, TrianglePolicy, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.views: DiagonalView, DiagonalInfo, diagonalView;
import mir.stat.internal.linearAlgebra.quadratic: quadraticForm, quadraticFormWithWorkspace;
import mir.math.sum: Summation;
import mir.stat.internal.linearAlgebra.views: symmetricView;
import mir.stat.internal.linearAlgebra.allocation;
import mir.stat.internal.linearAlgebra.types: asView, Element, vector, matrix, compatible;

/// Reuse a covariance factor, a precision matrix, or a precision factor.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs, sqrt;
    double[2] x = [1, 2], center = [0, 0];
    double[4] covarianceData = [4, 2, double.nan, 5];
    double[4] precisionData = [0.3125, -0.125, double.nan, 0.25];
    auto covariance = covarianceData[].sliced(2, 2);
    auto precision = precisionData[].sliced(2, 2);
    // Factors may be supplied by another library or retained from earlier work.
    double[4] covarianceFactorData = [2, 1, double.nan, 2];
    double[4] precisionFactorData = [sqrt(5.0)/4, -1/(2*sqrt(5.0)), double.nan, 1/sqrt(5.0)];
    auto covarianceFactor = covarianceFactorData[].sliced(2, 2);
    auto precisionFactor = precisionFactorData[].sliced(2, 2);
    auto fromCovariance = rcMahalanobisSquared(x[], center[], covariance);
    auto fromFactor = rcMahalanobisSquared!(CovarianceRepresentation.covarianceCholesky)(x[], center[], covarianceFactor);
    auto fromPrecision = rcMahalanobisSquared!(CovarianceRepresentation.inverseCovariance)(x[], center[], precision);
    auto fromPrecisionFactor = rcMahalanobisSquared!(CovarianceRepresentation.inverseCovarianceCholesky)(x[], center[], precisionFactor);
    assert(fabs(fromCovariance - 0.8125) < 1e-14);
    assert(fabs(fromFactor - fromCovariance) < 1e-14);
    assert(fabs(fromPrecision - fromCovariance) < 1e-14);
    assert(fabs(fromPrecisionFactor - fromCovariance) < 1e-14);
    // Keep the same factor for a later observation; no decomposition is repeated.
    double[2] next = [2, 4];
    auto nextSquared = rcMahalanobisSquared!(CovarianceRepresentation.covarianceCholesky)(
        next[], center[], covarianceFactor);
    assert(nextSquared == 3.25);
    assert(covarianceFactor[0,0] == 2 && covarianceFactor[0,1] == 1 &&
        covarianceFactor[1,1] == 2);
}

/// Select lower-triangle storage without importing internal linear algebra.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.stat.internal.mahalanobis: mahalanobisSquared, CovarianceRepresentation, TrianglePolicy;
    import mir.ndslice.slice: sliced;
    double[2] x = [1, 2], center = [0, 0];
    double[4] covariance = [4, double.nan, 2, 5];
    assert(mahalanobisSquared!(CovarianceRepresentation.covariance, TrianglePolicy.lower)(
        x[], center[], covariance[].sliced(2,2)) == 0.8125);
}

/++
Compute Mahalanobis distance using GC-managed temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X mahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    int allocator;
    return evaluate!(Allocation.gc, false, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    auto result = mahalanobis(values[].sliced, zero[].sliced, covariance[].sliced(2,2));
    assert(result > 1.414 && result < 1.415);
}

/++
Compute Mahalanobis distance using reference-counted temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X rcMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    int allocator;
    return evaluate!(Allocation.rc, false, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    auto result = rcMahalanobis(values[].sliced, zero[].sliced, covariance[].sliced(2,2));
    assert(result > 1.414 && result < 1.415);
}

/++
Compute Mahalanobis distance using custom-allocator temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
    allocator = Allocator supporting `makeUninitSlice` and `dispose`. All temporary storage is disposed before return or exception.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X makeMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    return evaluate!(Allocation.custom, false, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    import std.experimental.allocator.mallocator: Mallocator;
    auto result = makeMahalanobis(Mallocator.instance, values[].sliced, zero[].sliced, covariance[].sliced(2,2));
    assert(result > 1.414 && result < 1.415);
}

/++
Compute Mahalanobis squared distance using GC-managed temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X mahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    int allocator;
    return evaluate!(Allocation.gc, true, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    assert(mahalanobisSquared(values[].sliced, zero[].sliced, covariance[].sliced(2,2)) == 2);
}

/++
Compute Mahalanobis squared distance using reference-counted temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X rcMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    int allocator;
    return evaluate!(Allocation.rc, true, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    assert(rcMahalanobisSquared(values[].sliced, zero[].sliced, covariance[].sliced(2,2)) == 2);
}

/++
Compute Mahalanobis squared distance using custom-allocator temporary storage.
Params:
    form = Representation of `metric`; defaults to covariance.
    triangle = Logical matrix triangle containing the metric or factor.
    x = Observation vector.
    center = Center vector of the same length and real element type.
    metric = Square matrix of matching order and element type.
    allocator = Allocator supporting `makeUninitSlice` and `dispose`. All temporary storage is disposed before return or exception.
Returns:
    A scalar of the input element type. Inputs are not modified.

Covariance input is factorized on each call and throws on numerical failure.
Precomputed forms avoid factorization. See the module documentation for factor
orientation, input preconditions, and numerical behavior.
+/
Element!X makeMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (compatible!(X,C,M))
{
    return evaluate!(Allocation.custom, true, form, triangle)(allocator, x, center, metric);
}

/// A diagonal covariance scales each coordinate by its standard deviation.
version(mir_stat_test_distance)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [3,4], zero = [0,0];
    double[4] covariance = [9,0,double.nan,16];
    import std.experimental.allocator.mallocator: Mallocator;
    assert(makeMahalanobisSquared(Mallocator.instance, values[].sliced, zero[].sliced, covariance[].sliced(2,2)) == 2);
}

/++
Compute distance with reusable workspace and a precomputed metric.
`form` must explicitly select a precomputed representation, except that internal
diagonal-view covariance calls also use this single-vector overload. Inputs follow
`mahalanobis`. `work` must be a writable vector of length `x.length` and the same
element type. Its elements must not overlap each other or any input. No
allocation is performed. Its contents are overwritten and unspecified afterward.
Input iterator exceptions may leave partial workspace contents.
+/
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work)
    if ((form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W))
{
    return prepared!(false, form, triangle)(x.asView, center.asView, metric.asView, work.asView);
}

/++
Evaluate one observation with a raw covariance and caller-owned storage.
On success, the function allocates no temporary storage.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covariance (default).
    triangle = Triangle to read, upper by default.
    x = Observation of k finite coordinates.
    center = Reference vector of k finite coordinates.
    metric = Positive-definite k-by-k covariance with finite selected triangle.
    work = Writable vector of k elements for centering and transforming x.
    factorWork = Writable k-by-k Cholesky factor storage. All inputs and
        workspaces have the same floating-point element type.

Returns:
    The Mahalanobis distance in the input element type, zero for k = 0.

Throws:
    Exception on Cholesky failure. factorWork may contain a partial factor.
+/
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == CovarianceRepresentation.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    import mir.exception: enforce;
    Element!X result;
    const pivot = tryCovarianceDistanceWithWorkspace!(false, triangle)(
        x.asView, center.asView, metric.asView, result, work.asView, factorWork.asView);
    enforce!"Cholesky factorization failed"(pivot == 0);
    return result;
}

/// Reuse vector storage with an already computed covariance Cholesky factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [3,4], center = [0,0], work;
    double[4] factor = [3,0,double.nan,4];
    auto result = mahalanobisWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(
        x[].sliced, center[].sliced, factor[].sliced(2,2), work[].sliced);
    assert(result > 1.414 && result < 1.415);
}

/++
Compute squared distance with reusable workspace and a precomputed metric.
`form` must explicitly select a precomputed representation, except that internal
diagonal-view covariance calls also use this single-vector overload. Inputs follow
`mahalanobis`. `work` must be a writable vector of length `x.length` and the same
element type. Its elements must not overlap each other or any input. No
allocation is performed. Its contents are overwritten and unspecified afterward.
Input iterator exceptions may leave partial workspace contents.
+/
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work)
    if ((form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W))
{
    return prepared!(true, form, triangle)(x.asView, center.asView, metric.asView, work.asView);
}

/++
Evaluate one observation with a raw covariance and caller-owned storage.
On success, the function allocates no temporary storage.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covariance (default).
    triangle = Triangle to read, upper by default.
    x = Observation of k finite coordinates.
    center = Reference vector of k finite coordinates.
    metric = Positive-definite k-by-k covariance with finite selected triangle.
    work = Writable vector of k elements for centering and transforming x.
    factorWork = Writable k-by-k Cholesky factor storage. All inputs and
        workspaces have the same floating-point element type.

Returns:
    The squared Mahalanobis distance in the input element type, zero for k = 0.

Throws:
    Exception on Cholesky failure. factorWork may contain a partial factor.
+/
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == CovarianceRepresentation.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    import mir.exception: enforce;
    Element!X result;
    const pivot = tryCovarianceDistanceWithWorkspace!(true, triangle)(
        x.asView, center.asView, metric.asView, result, work.asView, factorWork.asView);
    enforce!"Cholesky factorization failed"(pivot == 0);
    return result;
}

/// Reuse vector storage with an already computed covariance Cholesky factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [3,4], center = [0,0], work;
    double[4] factor = [3,0,double.nan,4];
    auto result = mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(
        x[].sliced, center[].sliced, factor[].sliced(2,2), work[].sliced);
    assert(result == 2);
}

/// Supply both workspaces when starting from a covariance matrix.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [1, 2], center = [0, 0], work;
    double[4] covariance = [4, 2, double.nan, 5], factorWork;
    auto result = mahalanobisSquaredWithWorkspace(x[], center[],
        covariance[].sliced(2, 2), work[], factorWork[].sliced(2, 2));
    assert(result == 0.8125);
}

/// Factor a raw covariance and write the distance only after successful factorization.
package(mir.stat) size_t tryCovarianceDistanceWithWorkspace(bool squared, TrianglePolicy triangle,
    X, C, M, T, W, F)(X x, C center, M metric, ref T result, W work, F factor)
{
    import mir.stat.internal.linearAlgebra.cholesky: tryCholeskyDecomposeInto;
    check(x, center, metric, work);
    const pivot = tryCholeskyDecomposeInto!triangle(metric, factor);
    if (pivot) return pivot;

    static if (DiagonalInfo!M.valid)
    {
        import mir.ndslice.topology: diagonal;
        auto preparedFactor = diagonalView(factor.diagonal);
    }
    else alias preparedFactor = factor;
    result = prepared!(squared, CovarianceRepresentation.covarianceCholesky, triangle)(
        x, center, preparedFactor, work);
    return 0;
}

/// Assert matching observation, center, workspace, and square metric dimensions.
private void check(X,C,M,W)(X x, C center, M metric, W work)
{
    assert(x.length == center.length && work.length == x.length,
        "Incompatible Mahalanobis vector lengths");
    assert(metric.length!0 == x.length && metric.length!1 == x.length,
        "Incompatible Mahalanobis matrix shape");
}

/// Allocate temporary workspaces according to policy and release custom storage on success or failure.
private Element!X evaluate(Allocation policy, bool squared, CovarianceRepresentation form,
    TrianglePolicy triangle, Allocator, X,C,M)(ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    auto xv = x.asView;
    auto cv = center.asView;
    auto mv = metric.asView;
    check(xv, cv, mv, xv);
    alias T = Element!X;
    auto work = allocate!(policy,T)(allocator, xv.shape);
    scope(exit)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(work.field);
        }
    }
    static if (DiagonalInfo!M.valid)
        return diagonalDistance!(squared,form)(xv,cv,mv,work);
    else static if (form == CovarianceRepresentation.covariance)
    {
        auto factor = allocate!(policy,T)(allocator, mv.shape);
        scope(exit)
        {
            static if (policy == Allocation.custom)
            {
                import std.experimental.allocator: dispose;
                allocator.dispose(factor.field);
            }
        }
        static if (squared) return mahalanobisSquaredWithWorkspace!(form,triangle)(xv,cv,mv,work,factor);
        else return mahalanobisWithWorkspace!(form,triangle)(xv,cv,mv,work,factor);
    }
    else
        return prepared!(squared,form,triangle)(xv,cv,mv,work);
}

/// Evaluate a precomputed metric using a quadratic form, triangular solve, or triangular multiplication.
private Element!X prepared(bool squared, CovarianceRepresentation form, TrianglePolicy triangle, X,C,M,W)(
    X x, C center, M metric, W work)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static if (DiagonalInfo!M.valid)
    {
        return diagonalDistance!(squared,form)(x,center,metric,work);
    }
    else
    {
        import std.math: isFinite, sqrt;
        static assert(form != CovarianceRepresentation.covariance);
        check(x,center,metric,work);
        alias T = Element!X;
        foreach (i; 0 .. x.length)
        {
            auto a = x[i];
            auto b = center[i];
            assert(isFinite(a) && isFinite(b), "Mahalanobis vectors must be finite");
            work[i] = a - b;
        }
        static if (form == CovarianceRepresentation.inverseCovariance)
        {
            foreach (i; 0 .. metric.length!0) foreach (j; 0 .. metric.length!1)
                if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                    assert(isFinite(metric[i,j]), "Mahalanobis metric must be finite");
            auto result = quadraticForm(work, symmetricMetric!triangle(metric));
            static if (squared) return result;
            else return sqrt(result);
        }
        else
        {
            import mir.ndslice.topology: universal;
            import mir.ndslice.dynamic: transposed;
            import mir.stat.internal.linearAlgebra.triangular: triangularVectorInPlace;
            foreach (i; 0 .. metric.length!0) foreach (j; 0 .. metric.length!1)
                if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                    assert(isFinite(metric[i,j]), "Mahalanobis factor must be finite");
            static if (form == CovarianceRepresentation.covarianceCholesky)
            {
                // S=U' U => solve U' z=d. S=L L' => solve L z=d.
                static if (triangle == TrianglePolicy.upper) auto a = metric.universal.transposed;
                else auto a = metric.universal;
                triangularVectorInPlace!(true,TrianglePolicy.lower)(a,work);
            }
            else static if (form == CovarianceRepresentation.inverseCovarianceCholesky)
            {
                // P=R' R => z=R d. P=Q Q' => z=Q' d.
                static if (triangle == TrianglePolicy.lower) auto a = metric.universal.transposed;
                else auto a = metric.universal;
                triangularVectorInPlace!(false,TrianglePolicy.upper)(a,work);
            }
            else static assert(0, "Unknown Mahalanobis matrix representation");
            return scaledNorm!squared(work);
        }

    }
}

/// Compute the norm or squared norm using a scaled sum of squares, propagating NaNs and infinities.
private Element!W scaledNorm(bool squared, W)(W work)
{
    import std.math: fabs, sqrt;
    alias T = Element!W;
    T scale = 0, sum = 1;
    bool infinite;
    foreach (i; 0 .. work.length)
    {
        auto v = fabs(work[i]);
        if (v != v) return T.nan;
        if (v == T.infinity) { infinite = true; continue; }
        if (v == 0) continue;
        if (v > scale)
        {
            auto ratio = scale / v;
            sum = 1 + sum * ratio * ratio;
            scale = v;
        }
        else
        {
            auto ratio = v / scale;
            sum += ratio * ratio;
        }
    }
    if (infinite) return T.infinity;
    static if (squared) return scale * (scale * sum);
    else return scale * sqrt(sum);
}

version(mir_stat_test_distance)
{
    private import std.meta: AliasSeq;
    // Generate separate test functions to bound optimization work per combination.
    static foreach (T; AliasSeq!(float,double,real))
    static foreach (triangle; [TrianglePolicy.upper,TrianglePolicy.lower])
    static foreach (form; [CovarianceRepresentation.covariance,CovarianceRepresentation.covarianceCholesky,
        CovarianceRepresentation.inverseCovariance,CovarianceRepresentation.inverseCovarianceCholesky])
    @safe pure @nogc unittest
    {
        import std.meta: AliasSeq;
        import std.math: sqrt, fabs;
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: universal, map;
        import mir.ndslice.dynamic: transposed, strided, reversed;
        import mir.ndslice.allocation: mininitRcslice;
        foreach (transposeStorage; [false,true])
        foreach (reverseWork; [false,true])
        {
            T[6] data = T(-99);
            auto metric = data[].sliced(2,3)[0..2,0..2].universal;
            if (transposeStorage) metric = metric.transposed;
            static if (form == CovarianceRepresentation.covariance)
            {
                metric[0,0] = 4; metric[1,1] = 5;
                metric[0,1] = metric[1,0] = 2;
            }
            else static if (form == CovarianceRepresentation.covarianceCholesky)
            {
                metric[0,0] = 2; metric[1,1] = 2;
                metric[0,1] = metric[1,0] = 1;
            }
            else static if (form == CovarianceRepresentation.inverseCovariance)
            {
                metric[0,0] = T(5)/16; metric[1,1] = T(1)/4;
                metric[0,1] = metric[1,0] = T(-1)/8;
            }
            else
            {
                metric[0,0] = sqrt(T(5))/4; metric[1,1] = 1/sqrt(T(5));
                metric[0,1] = metric[1,0] = -1/(2*sqrt(T(5)));
            }
            metric[triangle == TrianglePolicy.upper ? 1 : 0, triangle == TrianglePolicy.upper ? 0 : 1] = T.nan;
            // d=[1,2], S=[[4,2],[2,5]], exact quadratic form is 13/16.
            T[4] xs = [T(4),T(-99),T(6),T(-99)];
            T[2] cs = [3,4];
            auto x = xs[].sliced.universal.strided(2);
            auto center = cs[].sliced;
            T[5] ws = T(-99);
            auto work = ws[1..4].sliced.universal.strided(2);
            if (reverseWork) work = work.reversed;
            T squared, distance;
            static if (form == CovarianceRepresentation.covariance)
            {
                T[16] fs = T(-99);
                auto f = fs[].sliced(4,4).strided!(0,1)(2,2).transposed;
                squared = mahalanobisSquaredWithWorkspace!(form,triangle)(x,center,metric,work,f);
                distance = mahalanobisWithWorkspace!(form,triangle)(x,center,metric,work,f);
            }
            else
            {
                squared = mahalanobisSquaredWithWorkspace!(form,triangle)(x,center,metric,work);
                distance = mahalanobisWithWorkspace!(form,triangle)(x,center,metric,work);
            }
            assert(fabs(squared - T(13)/16) < 32*T.epsilon);
            assert(fabs(distance - sqrt(T(13)/16)) < 32*T.epsilon);
            assert(fabs(rcMahalanobisSquared!(form,triangle)(x,center,metric) - squared) < 32*T.epsilon);
            assert(fabs(rcMahalanobis!(form,triangle)(x,center,metric) - distance) < 32*T.epsilon);
            assert(ws[0] == -99 && ws[2] == -99 && ws[4] == -99);
            assert(xs[0] == 4 && xs[2] == 6 && cs[0] == 3 && cs[1] == 4);
            assert(data[2] == -99 && data[5] == -99);
            auto computed = metric.map!((T value) { assert(value == value); return value; });
            assert(fabs(rcMahalanobisSquared!(form,triangle)(x,center,computed) - squared) < 32*T.epsilon);
            auto owner = mininitRcslice!T(2,2);
            owner[] = metric;
            const constOwner = owner;
            assert(fabs(rcMahalanobisSquared!(form,triangle)(x,center,constOwner) - squared) < 32*T.epsilon);
            assert(rcMahalanobisSquared!(form,triangle)(center,center,metric) == 0);
            auto empty = mininitRcslice!T(0);
            auto emptyMetric = mininitRcslice!T(0,0);
            assert(rcMahalanobisSquared!(form,triangle)(empty,empty,emptyMetric) == 0);
            assert(rcMahalanobis!(form,triangle)(empty,empty,emptyMetric) == 0);
        }
    }
}

version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs, isFinite;
    double[4] identity = [1,0,double.nan,1];
    double[2] x = [1e200,1e200], center = [0,0], work;
    auto a = identity[].sliced(2,2);
    // Vector arrays are accepted directly; matrices remain ndslices.
    auto distance = mahalanobisWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(x[],center[],a,work[]);
    assert(isFinite(distance) && fabs(distance/1e200 - 1.4142135623730951) < 1e-15);
    assert(mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(x[],center[],a,work[]) == double.infinity);
    x[] = 1e-200;
    distance = mahalanobisWithWorkspace!(CovarianceRepresentation.inverseCovarianceCholesky)(x[],center[],a,work[]);
    assert(distance > 0 && fabs(distance/1e-200 - 1.4142135623730951) < 1e-15);
    assert(mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.inverseCovarianceCholesky)(x[],center[],a,work[]) == 0);
}

version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    struct CountingAllocator
    {
        size_t allocations, releases;
        /// Allocate test storage while counting allocation attempts.
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        /// Release test storage while counting releases.
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[2] x = [1,2], center = [0,0];
    double[4] data = [4,2,double.nan,5];
    auto metric = data[].sliced(2,2);
    assert(makeMahalanobisSquared(allocator,x[],center[],metric) == 0.8125);
    assert(allocator.allocations == 2 && allocator.releases == 2);
    data[] = [2,1,double.nan,2];
    assert(makeMahalanobisSquared!(CovarianceRepresentation.covarianceCholesky)(allocator,x[],center[],metric) == 0.8125);
    assert(allocator.allocations == 3 && allocator.releases == 3);
    auto failure = new Exception("input failed");
    /// Throw the test exception to verify cleanup after an input iterator fails.
    double fail(double value) { throw failure; }
    assertThrown!Exception(makeMahalanobisSquared!(CovarianceRepresentation.covarianceCholesky)(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 4 && allocator.releases == 4);
    assertThrown!Exception(makeMahalanobisSquared(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 6 && allocator.releases == 6);
    data[] = [1,2,double.nan,1];
    assertThrown!Exception(makeMahalanobisSquared(allocator,x[],center[],metric));
    assert(allocator.allocations == 8 && allocator.releases == 8);
    assertThrown!AssertError(makeMahalanobisSquared(allocator,x[0..1],center[],metric));
    assert(allocator.allocations == 8);
    data[] = [0.3125,-0.125,double.nan,0.25];
    enum inverse = CovarianceRepresentation.inverseCovariance;
    assert(makeMahalanobisSquared!(inverse,TrianglePolicy.upper,Summation.fast)(allocator,x[],center[],metric) >= 0);
    assert(allocator.allocations == 10 && allocator.releases == 10);
    assertThrown!Exception(makeMahalanobisSquared!(inverse,TrianglePolicy.upper,Summation.fast)(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 12 && allocator.releases == 12);
    double[1] wrongWork;
    assertThrown!AssertError(mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(x[],center[],metric,wrongWork[]));
}

version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import std.math: fabs, sqrt;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
    // L = [[2,0,0],[1,3,0],[-1,2,4]], d = L * [1,-2,3].
    // S = L*L' and the independently calculated inverse are supplied below.
    // Therefore d' * inverse(S) * d = 1 + 4 + 9 = 14.
    double[3] x = [5, -1, 12], center = [3, 4, 5], work;
    double[9] covariance = [4, double.nan, double.nan,
        2, 10, double.nan, -2, 5, 21];
    double[9] lower = [2, double.nan, double.nan,
        1, 3, double.nan, -1, 2, 4];
    double[9] inverse = [185.0/576, double.nan, double.nan,
        -13.0/144, 5.0/36, double.nan, 5.0/96, -1.0/24, 1.0/16];
    static foreach (triangle; [TrianglePolicy.lower, TrianglePolicy.upper])
    {{
        auto covarianceView = covariance[].sliced(3,3).universal;
        auto factorView = lower[].sliced(3,3).universal;
        auto precisionView = inverse[].sliced(3,3).universal;
        static if (triangle == TrianglePolicy.upper)
        {
            covarianceView = covarianceView.transposed;
            factorView = factorView.transposed;
            precisionView = precisionView.transposed;
        }
        auto precisionFactor = rcCholeskyDecompose!triangle(precisionView);
        static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.covarianceCholesky,
            CovarianceRepresentation.inverseCovariance, CovarianceRepresentation.inverseCovarianceCholesky])
        {{
            static if (form == CovarianceRepresentation.covariance) auto metric = covarianceView;
            else static if (form == CovarianceRepresentation.covarianceCholesky) auto metric = factorView;
            else static if (form == CovarianceRepresentation.inverseCovariance) auto metric = precisionView;
            else auto metric = precisionFactor;
            auto squared = rcMahalanobisSquared!(form, triangle)(x[], center[], metric);
            auto distance = rcMahalanobis!(form, triangle)(x[], center[], metric);
            assert(fabs(squared - 14) < 128 * double.epsilon * 14);
            assert(fabs(distance - sqrt(14.0)) < 128 * double.epsilon * sqrt(14.0));
        }}
    }}
}

version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: Slice;
    import std.experimental.allocator.mallocator: Mallocator;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[] x, center, work;
        Slice!(T*, 2) metric, factorWork;
        static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.covarianceCholesky,
            CovarianceRepresentation.inverseCovariance, CovarianceRepresentation.inverseCovarianceCholesky])
        {
            static assert(is(typeof(mahalanobis!form(x, center, metric)) == T));
            static assert(is(typeof(mahalanobisSquared!form(x, center, metric)) == T));
            static assert(is(typeof(rcMahalanobis!form(x, center, metric)) == T));
            static assert(is(typeof(rcMahalanobisSquared!form(x, center, metric)) == T));
            static assert(is(typeof(makeMahalanobis!form(Mallocator.instance, x, center, metric)) == T));
            static assert(is(typeof(makeMahalanobisSquared!form(Mallocator.instance, x, center, metric)) == T));
            static if (form == CovarianceRepresentation.covariance)
            {
                static assert(is(typeof(mahalanobisWithWorkspace!form(x, center, metric, work, factorWork)) == T));
                static assert(is(typeof(mahalanobisSquaredWithWorkspace!form(x, center, metric, work, factorWork)) == T));
            }
            else
            {
                static assert(is(typeof(mahalanobisWithWorkspace!form(x, center, metric, work)) == T));
                static assert(is(typeof(mahalanobisSquaredWithWorkspace!form(x, center, metric, work)) == T));
            }
        }
    }}
    static assert(!__traits(compiles, mahalanobis(1, 2, 3)));
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X mahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.gc, false, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X rcMahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.rc, false, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X makeMahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    return precisionAllocated!(Allocation.custom, false, triangle, summation)(
        allocator, x, center, metric);
}

/++
Precision-matrix overload with separate centered-input and product workspaces.
`form` must be `CovarianceRepresentation.inverseCovariance`. Both writable vectors must
have the input element type and length, contain nonoverlapping elements, and
be disjoint from each other and every input. Shapes are asserted; overlap is
not checked. No allocation occurs. On success `work` contains `x - center` and
`productWork` contains `metric * work`; iterator exceptions may leave either
partially written. The default is compensated; explicit `Summation.fast` permits
SYMV and DOT with backend-dependent rounding. Other input contracts match
`mahalanobis`. Negative squared results are not clamped.
+/
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) &&
        vector!W && vector!P && is(Element!X == Element!W) && is(Element!X == Element!P))
{
    return precisionWorkspace!(false, triangle, summation)(
        x.asView, center.asView, metric.asView, work.asView, productWork.asView);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X mahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.gc, true, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X rcMahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.rc, true, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`CovarianceRepresentation.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X makeMahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M))
{
    return precisionAllocated!(Allocation.custom, true, triangle, summation)(
        allocator, x, center, metric);
}

/++
Precision-matrix overload with separate centered-input and product workspaces.
`form` must be `CovarianceRepresentation.inverseCovariance`. Both writable vectors must
have the input element type and length, contain nonoverlapping elements, and
be disjoint from each other and every input. Shapes are asserted; overlap is
not checked. No allocation occurs. On success `work` contains `x - center` and
`productWork` contains `metric * work`; iterator exceptions may leave either
partially written. The default is compensated; explicit `Summation.fast` permits
SYMV and DOT with backend-dependent rounding. Other input contracts match
`mahalanobis`. Negative squared results are not clamped.
+/
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) &&
        vector!W && vector!P && is(Element!X == Element!W) && is(Element!X == Element!P))
{
    return precisionWorkspace!(true, triangle, summation)(
        x.asView, center.asView, metric.asView, work.asView, productWork.asView);
}

/// Allocate disjoint precision work vectors and release custom storage on every exit.
private Element!X precisionAllocated(Allocation policy, bool squared, TrianglePolicy triangle,
    Summation summation, Allocator, X, C, M)(ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    auto xv = x.asView;
    auto cv = center.asView;
    auto mv = metric.asView;
    check(xv, cv, mv, xv);
    auto work = allocate!(policy, Element!X)(allocator, xv.shape);
    scope(exit)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(work.field);
        }
    }
    auto productWork = allocate!(policy, Element!X)(allocator, xv.shape);
    scope(exit)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(productWork.field);
        }
    }
    return precisionWorkspace!(squared, triangle, summation)(xv, cv, mv, work, productWork);
}

/// Center observations and evaluate a precision quadratic form through caller workspace.
private Element!X precisionWorkspace(bool squared, TrianglePolicy triangle, Summation summation, X,C,M,W,P)(
    X x, C center, M metric, W work, P productWork)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static if (DiagonalInfo!M.valid)
    {
        import mir.math.common: sqrt;
        import std.math: isFinite;
        check(x,center,metric,work);
        assert(productWork.length == x.length);
        foreach(i; 0 .. x.length)
        {
            assert(isFinite(x[i]) && isFinite(center[i]));
            assert(isFinite(metric[i,i]) && metric[i,i] > 0);
            work[i] = x[i] - center[i];
        }
        auto result = quadraticFormWithWorkspace!summation(work,metric,productWork);
        static if (squared) return result;
        else return sqrt(result);
    }
    else
    {
        import std.math: isFinite, sqrt;
        check(x, center, metric, work);
        assert(productWork.length == x.length, "Incorrect Mahalanobis product workspace length");
        foreach (i; 0 .. x.length)
        {
            auto a = x[i];
            auto b = center[i];
            assert(isFinite(a) && isFinite(b), "Mahalanobis vectors must be finite");
            work[i] = a - b;
        }
        foreach (i; 0 .. metric.length!0) foreach (j; 0 .. metric.length!1)
            if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                assert(isFinite(metric[i,j]), "Mahalanobis metric must be finite");
        auto result = quadraticFormWithWorkspace!summation(work, symmetricMetric!triangle(metric), productWork);
        static if (squared) return result;
        else return sqrt(result);

    }
}

/// Reuse two vectors to opt into BLAS-compatible precision arithmetic.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.sum: Summation;
    double[2] x = [2, 3], center = [1, 1], difference, product;
    double[4] data = [0.3125, -0.125, double.nan, 0.25];
    auto precision = data[].sliced(2, 2);
    auto result = mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.inverseCovariance,
        TrianglePolicy.upper, Summation.fast)(x[], center[], precision, difference[], product[]);
    assert(result == 0.8125);
    assert(difference == [1, 2] && product == [0.0625, 0.375]);
}

version(mir_stat_test_distance)
@system unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs, sqrt;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed, reversed;
    import std.experimental.allocator.mallocator: Mallocator;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    {{
        T[2] x = [2,3], center = [1,1], work, product;
        T[4] data = [0.3125, -0.125, T.nan, 0.25];
        auto p = data[].sliced(2,2).universal;
        static if (triangle == TrianglePolicy.lower) p = p.transposed;
        enum form = CovarianceRepresentation.inverseCovariance;
        enum fast = Summation.fast;
        assert(mahalanobisSquared!(form,triangle,fast)(x[],center[],p) == T(0.8125));
        assert(rcMahalanobisSquared!(form,triangle,fast)(x[],center[],p) == T(0.8125));
        assert(makeMahalanobisSquared!(form,triangle,fast)(Mallocator.instance,x[],center[],p) == T(0.8125));
        assert(fabs(mahalanobis!(form,triangle,fast)(x[],center[],p) - sqrt(T(0.8125))) < 8*T.epsilon);
        assert(fabs(rcMahalanobis!(form,triangle,fast)(x[],center[],p) - sqrt(T(0.8125))) < 8*T.epsilon);
        assert(fabs(makeMahalanobis!(form,triangle,fast)(Mallocator.instance,x[],center[],p) - sqrt(T(0.8125))) < 8*T.epsilon);
        assert(mahalanobisSquaredWithWorkspace!(form,triangle)(x[],center[],p,work[],product[]) == T(0.8125));
        assert(fabs(mahalanobisWithWorkspace!(form,triangle,fast)(x[],center[],p,work[],product[]) - sqrt(T(0.8125))) < 8*T.epsilon);
        auto xv = x[].sliced.reversed;
        auto cv = center[].sliced.reversed;
        auto expected = mahalanobisSquared!(form,triangle)(xv,cv,p);
        assert(mahalanobisSquaredWithWorkspace!(form,triangle,fast)(xv,cv,p,work[],product[]) == expected);
        assert(mahalanobisSquaredWithWorkspace!(form,triangle,fast)(x[0..0],center[0..0],data[0..0].sliced(0,0),work[0..0],product[0..0]) == 0);
    }}
}

/// Whether column observations, center, and metric have matching real element types.
private enum batchCompatible(X, C, M) = matrix!X && vector!C && (matrix!M || DiagonalInfo!M.valid) &&
    is(Element!X == Element!C) && is(Element!X == Element!M);

/// Validate batch shapes before touching caller-owned output or scratch storage.
private void checkBatch(X, C, M, R, W)(X x, C center, M metric, R result, W work)
{
    assert(center.length == x.length!0 && metric.length!0 == x.length!0 &&
        metric.length!1 == x.length!0, "Incompatible Mahalanobis batch dimensions");
    assert(result.length == x.length!1 && work.shape == x.shape,
        "Incompatible Mahalanobis batch output or workspace");
}

/// Factor a raw covariance and write the distances in observation order on success.
package(mir.stat) size_t tryCovarianceDistanceBatchWithWorkspace(bool squared, TrianglePolicy triangle,
    Summation summation = Summation.kbn, X, C, M, R, W, F)(X x, C center, M metric, R result, W work, F factor)
{
    import mir.stat.internal.linearAlgebra.cholesky: tryCholeskyDecomposeInto;
    checkBatch(x, center, metric, result, work);
    const pivot = tryCholeskyDecomposeInto!triangle(metric, factor);
    if (pivot) return pivot;

    static if (DiagonalInfo!M.valid)
    {
        import mir.ndslice.topology: diagonal;
        auto preparedFactor = diagonalView(factor.diagonal);
    }
    else alias preparedFactor = factor;
    preparedBatch!(squared, CovarianceRepresentation.covarianceCholesky, triangle, summation)(
        x, center, preparedFactor, result, work);
    return 0;
}

/// Center every observation, checking finite inputs and only the selected metric triangle.
private void centerBatch(TrianglePolicy triangle, X, C, M, W)(X x, C center, M metric, W work)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    import std.math: isFinite;
    foreach (i; 0 .. x.length!0)
    {
        assert(isFinite(center[i]), "Mahalanobis center must be finite");
        foreach (j; 0 .. x.length!1)
        {
            assert(isFinite(x[i, j]), "Mahalanobis observations must be finite");
            work[i, j] = x[i, j] - center[i];
        }
        static if (DiagonalInfo!M.valid)
            assert(isFinite(metric[i,i]) && metric[i,i] > 0, "Diagonal precision must be finite and positive");
        else
            foreach (j; 0 .. metric.length!1)
                if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                    assert(isFinite(metric[i, j]), "Mahalanobis metric must be finite");
    }
}

/// Evaluate a prepared batch after centering, using one triangular matrix operation when possible.
private void preparedBatch(bool squared, CovarianceRepresentation form, TrianglePolicy triangle,
    Summation summation, X, C, M, R, W)(X x, C center, M metric, R result, W work)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static if (DiagonalInfo!M.valid)
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        checkBatch(x,center,metric,result,work);
        import mir.math.common: sqrt;
        import std.math: isFinite;
        // Prepare each coordinate once, sharing its factor across observations.
        foreach (i; 0 .. x.length!0)
        {
            auto d = metric[i,i];
            assert(isFinite(d) && isFinite(center[i]), "Mahalanobis inputs must be finite");
            static if (form == CovarianceRepresentation.covariance)
            {
                import mir.exception: enforce;
                enforce!"Cholesky factorization failed"(d > 0);
            }
            else static if (form == CovarianceRepresentation.inverseCovariance)
                assert(d > 0, "Diagonal metric must be positive definite");
            else assert(d != 0, "Diagonal factor must be nonsingular");
            static if (form == CovarianceRepresentation.covariance) auto factor = sqrt(d);
            else auto factor = d;
            foreach (j; 0 .. x.length!1)
            {
                assert(isFinite(x[i,j]), "Mahalanobis observations must be finite");
                auto delta = x[i,j] - center[i];
                static if (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.covarianceCholesky)
                    work[i,j] = delta / factor;
                else static if (form == CovarianceRepresentation.inverseCovarianceCholesky)
                    work[i,j] = factor * delta;
                else static if (form == CovarianceRepresentation.inverseCovariance) work[i,j] = delta;
                else static assert(0, "Unknown Mahalanobis matrix representation");
            }
        }
        auto columns = work.universal.transposed;
        foreach (j; 0 .. result.length)
        {
            static if (form == CovarianceRepresentation.inverseCovariance)
            {
                auto value = quadraticForm!summation(columns[j], metric);
                static if (squared) result[j] = value;
                else result[j] = sqrt(value);
            }
            else result[j] = scaledNorm!squared(columns[j]);
        }
    }
    else
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        import std.math: sqrt;
        checkBatch(x, center, metric, result, work);
        centerBatch!triangle(x, center, metric, work);
        static if (form == CovarianceRepresentation.inverseCovariance)
        {
            import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
            quadraticFormBatchInto!summation(work, symmetricMetric!triangle(metric), result);
            static if (!squared) foreach (j; 0 .. result.length) result[j] = sqrt(result[j]);
        }
        else
        {
            import mir.stat.internal.linearAlgebra.triangular: triangularMatrixInPlace;
            static if (form == CovarianceRepresentation.covarianceCholesky)
            {
                static if (triangle == TrianglePolicy.upper) auto a = metric.universal.transposed;
                else auto a = metric.universal;
                triangularMatrixInPlace!(true, TrianglePolicy.lower)(a, work);
            }
            else static if (form == CovarianceRepresentation.inverseCovarianceCholesky)
            {
                static if (triangle == TrianglePolicy.lower) auto a = metric.universal.transposed;
                else auto a = metric.universal;
                triangularMatrixInPlace!(false, TrianglePolicy.upper)(a, work);
            }
            else static assert(0, "A prepared batch requires a factor or precision matrix");
            auto columns = work.universal.transposed;
            foreach (j; 0 .. result.length) result[j] = scaledNorm!squared(columns[j]);
        }

    }
}

/++
Compute distances for observation columns using one shared center and metric.
Returns a GC-managed vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
mahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    int allocator;
    return batchAllocated!(Allocation.gc, false, form, triangle, summation)(allocator, x, center, metric);
}

/++
Compute distances for observation columns using one shared center and metric.
Returns a reference-counted vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    int allocator;
    return batchAllocated!(Allocation.rc, false, form, triangle, summation)(allocator, x, center, metric);
}

/++
Compute distances for observation columns using one shared center and metric.
Returns a custom-allocated vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
Custom callers dispose the returned vector's field; temporary buffers are released.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    return batchAllocated!(Allocation.custom, false, form, triangle, summation)(allocator, x, center, metric);
}

/++
Write one distance per observation column into `result`, reusing `work`.
`result` has length `x.length!1`; `work` has shape `x.shape`. All writable
buffers must have nonoverlapping elements and be disjoint from each other and
all inputs. Shapes are asserted; overlap is not checked. No allocation occurs.
This overload accepts prepared factors or a compensated precision matrix.
Ordinary covariance and fast precision calls require the extra-workspace overload.
Internal diagonal-view calls accept every representation here without extra scratch.
Factor forms compute scaled norms directly, preserving the unsquared API's
range. Finite-input and positive-definite preconditions match the scalar API.
+/
void mahalanobisBatchWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        (form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) &&
        (!(form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast) || DiagonalInfo!M.valid))
{
    preparedBatch!(false, form, triangle, summation)(x.asView, center.asView, metric.asView, result.asView, work.asView);
}

/++
Write one distance per observation column using caller-owned storage.
On success, the function allocates no temporary storage.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covariance (default) or inverseCovariance.
    triangle = Triangle to read, upper by default.
    summation = Precision quadratic-form summation, kbn by default.
        With fast summation, eligible precision batches use BLAS.
    x = k-by-n matrix with one finite observation per column.
    center = Vector of k finite coordinates shared by all observations.
    metric = Positive-definite k-by-k matrix with finite selected triangle.
    result = Writable vector of n distances, in column order.
    work = Writable k-by-n scratch matrix for centered observations.
    extra = For covariance, a k-by-k workspace that receives the Cholesky
        factor. For precision, a k-by-n workspace that receives the matrix
        product. All buffers have the same floating-point element type.

Returns:
    void. Zero-dimensional observations have distance zero.

Throws:
    Exception on covariance factorization failure, before any result is written.
+/
void mahalanobisBatchWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance))
{
    auto xv = x.asView; auto cv = center.asView; auto mv = metric.asView;
    auto rv = result.asView; auto wv = work.asView; auto ev = extra.asView;
    static if (form == CovarianceRepresentation.covariance)
    {
        import mir.exception: enforce;
        const pivot = tryCovarianceDistanceBatchWithWorkspace!(false, triangle, summation)(
            xv, cv, mv, rv, wv, ev);
        enforce!"Cholesky factorization failed"(pivot == 0);
    }
    else
    {
        checkBatch(xv, cv, mv, rv, wv);
        import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
        import std.math: sqrt;
        assert(ev.shape == xv.shape);
        centerBatch!triangle(xv, cv, mv, wv);
        quadraticFormBatchInto!summation(wv, symmetricMetric!triangle(mv), rv, ev);
        foreach (j; 0 .. rv.length) rv[j] = sqrt(rv[j]);
    }
}

/++
Compute squared distances for observation columns using one shared center and metric.
Returns a GC-managed vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
mahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    int allocator;
    return batchAllocated!(Allocation.gc, true, form, triangle, summation)(allocator, x, center, metric);
}

/// Evaluate several observation columns while factorizing the covariance only once.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[6] observations = [1, 2, 0, 2, 4, 0];
    double[2] center = [0, 0];
    double[4] covariance = [4, 2, double.nan, 5];
    auto result = mahalanobisSquaredBatch(observations[].sliced(2, 3), center[], covariance[].sliced(2, 2));
    assert(result == [0.8125, 3.25, 0]);
}

/++
Compute squared distances for observation columns using one shared center and metric.
Returns a reference-counted vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    int allocator;
    return batchAllocated!(Allocation.rc, true, form, triangle, summation)(allocator, x, center, metric);
}

/++
Compute squared distances for observation columns using one shared center and metric.
Returns a custom-allocated vector of length `x.length!1`. Inputs share a real element type;
`x` has shape `(dimension, count)`, center has length `dimension`, and the metric
is square. The selected triangle and representation follow the single-observation
API. Covariance is factorized once per batch. Prepared covariance/precision
factors use TRSM/TRMM where eligible and stable scaled column norms.
Precision matrices retain compensated reductions by default; explicit
`Summation.fast` permits SYMM and column dots with an extra matrix workspace.
Native D fallbacks support general strides. Zero-dimensional observations have
zero distance; a batch with no columns returns an empty vector.
Custom callers dispose the returned vector's field; temporary buffers are released.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (batchCompatible!(X, C, M))
{
    return batchAllocated!(Allocation.custom, true, form, triangle, summation)(allocator, x, center, metric);
}

/++
Write one squared distance per observation column into `result`, reusing `work`.
`result` has length `x.length!1`; `work` has shape `x.shape`. All writable
buffers must have nonoverlapping elements and be disjoint from each other and
all inputs. Shapes are asserted; overlap is not checked. No allocation occurs.
This overload accepts prepared factors or a compensated precision matrix.
Ordinary covariance and fast precision calls require the extra-workspace overload.
Internal diagonal-view calls accept every representation here without extra scratch.
Factor forms compute scaled norms directly, preserving the unsquared API's
range. Finite-input and positive-definite preconditions match the scalar API.
+/
void mahalanobisSquaredBatchWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        (form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) &&
        (!(form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast) || DiagonalInfo!M.valid))
{
    preparedBatch!(true, form, triangle, summation)(x.asView, center.asView, metric.asView, result.asView, work.asView);
}

/++
Write one squared distance per observation column using caller-owned storage.
On success, the function allocates no temporary storage.
All writable elements must be disjoint from one another and from the inputs.

Params:
    form = covariance (default) or inverseCovariance.
    triangle = Triangle to read, upper by default.
    summation = Precision quadratic-form summation, kbn by default.
        With fast summation, eligible precision batches use BLAS.
    x = k-by-n matrix with one finite observation per column.
    center = Vector of k finite coordinates shared by all observations.
    metric = Positive-definite k-by-k matrix with finite selected triangle.
    result = Writable vector of n squared distances, in column order.
    work = Writable k-by-n scratch matrix for centered observations.
    extra = For covariance, a k-by-k workspace that receives the Cholesky
        factor. For precision, a k-by-n workspace that receives the matrix
        product. All buffers have the same floating-point element type.

Returns:
    void. Zero-dimensional observations have distance zero.

Throws:
    Exception on covariance factorization failure, before any result is written.
+/
void mahalanobisSquaredBatchWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance))
{
    auto xv = x.asView; auto cv = center.asView; auto mv = metric.asView;
    auto rv = result.asView; auto wv = work.asView; auto ev = extra.asView;
    static if (form == CovarianceRepresentation.covariance)
    {
        import mir.exception: enforce;
        const pivot = tryCovarianceDistanceBatchWithWorkspace!(true, triangle, summation)(
            xv, cv, mv, rv, wv, ev);
        enforce!"Cholesky factorization failed"(pivot == 0);
    }
    else
    {
        checkBatch(xv, cv, mv, rv, wv);
        import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
        import std.math: sqrt;
        assert(ev.shape == xv.shape);
        centerBatch!triangle(xv, cv, mv, wv);
        quadraticFormBatchInto!summation(wv, symmetricMetric!triangle(mv), rv, ev);

    }
}

/// Compute the same squared distances using factor scratch and product scratch.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import std.math: fabs;

    // Three runs in columns. Each column contains two temperature readings.
    double[6] readings = [10, 12, 10, 20, 21, 24], centered, products;
    double[2] center = [10, 20];
    double[4] covariance = [4, 2, 2, 5], precision = [0.3125, -0.125, -0.125, 0.25], factor;
    double[3] values;
    auto runs = readings[].sliced(2, 3);
    auto work = centered[].sliced(2, 3);
    auto factorWork = factor[].sliced(2, 2);
    auto productWork = products[].sliced(2, 3);

    mahalanobisSquaredBatchWithWorkspace(runs, center[], covariance[].sliced(2, 2), values[], work, factorWork);
    assert(values == [0, 1, 4]);
    mahalanobisSquaredBatchWithWorkspace!(CovarianceRepresentation.inverseCovariance, TrianglePolicy.upper, Summation.fast)(
        runs, center[], precision[].sliced(2, 2), values[], work, productWork);
    foreach (i, expected; [0, 1, 4])
        assert(fabs(values[i] - expected) < 1e-14);
}

/// Allocate batch output and scratch, releasing custom buffers independently on failure.
private AllocatedSlice!(policy, Element!X, 1) batchAllocated(Allocation policy, bool squared,
    CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(
    ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
{
    auto xv = x.asView;
    auto cv = center.asView;
    auto mv = metric.asView;
    alias T = Element!X;
    auto result = allocate!(policy, T)(allocator, [xv.length!1]);
    scope(failure) { static if (policy == Allocation.custom) {
        import std.experimental.allocator: dispose; allocator.dispose(result.field); } }
    auto work = allocate!(policy, T)(allocator, xv.shape);
    scope(exit) { static if (policy == Allocation.custom) {
        import std.experimental.allocator: dispose; allocator.dispose(work.field); } }
    static if (DiagonalInfo!M.valid)
        preparedBatch!(squared,form,triangle,summation)(xv,cv,mv,result,work);
    else static if (form == CovarianceRepresentation.covariance ||
        (form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast))
    {
        static if (form == CovarianceRepresentation.covariance) auto shape = mv.shape;
        else auto shape = xv.shape;
        auto extra = allocate!(policy, T)(allocator, shape);
        scope(exit) { static if (policy == Allocation.custom) {
            import std.experimental.allocator: dispose; allocator.dispose(extra.field); } }
        static if (squared) mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(xv, cv, mv, result, work, extra);
        else mahalanobisBatchWithWorkspace!(form, triangle, summation)(xv, cv, mv, result, work, extra);
    }
    else
    {
        static if (squared) mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(xv, cv, mv, result, work);
        else mahalanobisBatchWithWorkspace!(form, triangle, summation)(xv, cv, mv, result, work);
    }
    return result;
}


// Compare every representation, triangle, and layout with independent scalar calls.
version(mir_stat_test_distance)
{
    private import std.meta: AliasSeq;
    // Generate separate test functions to bound optimization work per combination.
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.covarianceCholesky,
        CovarianceRepresentation.inverseCovariance, CovarianceRepresentation.inverseCovarianceCholesky])
    static foreach (summation; [Summation.kbn, Summation.fast])
    @safe pure unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: universal, map;
        import mir.ndslice.dynamic: transposed, reversed;
        import std.meta: AliasSeq;
        import std.math: sqrt, fabs;
        foreach (tx;[false, true]) foreach (tm;[false, true])
        {
            T[6] xs, ws, ps; T[4] ms, fs; T[3] rs; T[2] cs = [3, 4];
            auto x = xs[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
            auto work = ws[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
            auto product = ps[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
            if (tx) { x = x.transposed; work = work.transposed; product = product.transposed; }
            auto metric = ms[].sliced(2, 2).universal;
            if (tm) metric = metric.transposed;
            static if (form == CovarianceRepresentation.covariance) { metric[0, 0] = 4; metric[0, 1] = metric[1, 0] = 2; metric[1, 1] = 5; }
            else static if (form == CovarianceRepresentation.covarianceCholesky) { metric[0, 0] = 2; metric[0, 1] = metric[1, 0] = 1; metric[1, 1] = 2; }
            else static if (form == CovarianceRepresentation.inverseCovariance) { metric[0, 0] = T(5)/16; metric[0, 1] = metric[1, 0] = T(-1)/8; metric[1, 1] = T(1)/4; }
            else { metric[0, 0] = sqrt(T(5))/4; metric[0, 1] = metric[1, 0] = -1/(2*sqrt(T(5))); metric[1, 1] = 1/sqrt(T(5)); }
            metric[triangle == TrianglePolicy.upper ? 1 : 0, triangle == TrianglePolicy.upper ? 0 : 1] = T.nan;
            foreach (i;0 .. 2) foreach (j;0 .. 3) x[i, j] = cs[i]+T((i+1)*j);
            auto result = rs[].sliced.universal.reversed;
            auto factor = fs[].sliced(2, 2);
            static if (form == CovarianceRepresentation.covariance)
                mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, cs[], metric, result, work, factor);
            else static if (form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast)
                mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, cs[], metric, result, work, product);
            else mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, cs[], metric, result, work);
            auto distances = rcMahalanobisBatch!(form, triangle, summation)(x, cs[], metric);
            auto squared = rcMahalanobisSquaredBatch!(form, triangle, summation)(x, cs[], metric);
            auto columns = x.transposed;
            foreach (j;0 .. 3) {
                auto reference = rcMahalanobisSquared!(form, triangle)(columns[j], cs[], metric);
                assert(fabs(result[j]-reference)<T(128)*T.epsilon);
                assert(fabs(squared[j]-reference)<T(128)*T.epsilon);
                assert(fabs(distances[j]-sqrt(reference))<T(128)*T.epsilon);
            }
            auto generic = x.map!(v=>v);
            assert(rcMahalanobisSquaredBatch!(form, triangle, summation)(generic, cs[], metric).length == 3);
        }
    }
}

version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isFinite;
    double[4] identity = [1, 0, double.nan, 1];
    double[2] x = [1e200, 1e200], center = [0, 0], work;
    double[1] result;
    mahalanobisBatchWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(x[].sliced(2, 1), center[], identity[].sliced(2, 2), result[], work[].sliced(2, 1));
    assert(isFinite(result[0]) && result[0]>1e200);
    double[0] empty;
    double[3] zeros;
    mahalanobisSquaredBatchWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(empty[].sliced(0, 3), empty[], empty[].sliced(0, 0), zeros[], empty[].sliced(0, 3));
    assert(zeros == [0, 0, 0]);
}

version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] a = [1, 0, double.nan, 1]; double[2] x = [2, 3], center = [0, 0];
    auto r = makeMahalanobisSquaredBatch!(CovarianceRepresentation.covarianceCholesky)(Mallocator.instance, x[].sliced(2, 1), center[], a[].sliced(2, 2));
    scope(exit) Mallocator.instance.dispose(r.field);
    assert(r[0] == 13);
}

version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    struct CountingAllocator {
        size_t allocations, releases;
        /// Count allocations made by the batch factory.
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        /// Count released scratch and result buffers.
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[4] a = [1, 0, double.nan, 1]; double[2] x = [2, 3], center = [0, 0];
    auto r = makeMahalanobisBatch(allocator, x[].sliced(2, 1), center[], a[].sliced(2, 2));
    assert(allocator.allocations == 3 && allocator.releases == 2);
    allocator.dispose(r.field);
    auto failure = new Exception("computed observation failed");
    /// Fail while centering after all three buffers have been allocated.
    double fail(double value) { throw failure; }
    assertThrown!Exception(makeMahalanobisSquaredBatch!(CovarianceRepresentation.inverseCovariance,
        TrianglePolicy.upper, Summation.fast)(allocator, x[].sliced(2, 1).map!fail, center[], a[].sliced(2, 2)));
    assert(allocator.allocations == allocator.releases);
}

import mir.stat.internal.linearAlgebra.types: isObservationSelection, ObservationColumns, observationColumns;

/++
Evaluate selected observation vectors using `mahalanobisBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `mahalanobisBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a GC-managed vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `mahalanobisBatch`.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
mahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        mahalanobisBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return mahalanobisBatch!(form, triangle, summation)(columns, center, metric);
}

/++
Evaluate selected observation vectors using `rcMahalanobisBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `rcMahalanobisBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a reference-counted vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `rcMahalanobisBatch`.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        rcMahalanobisBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return rcMahalanobisBatch!(form, triangle, summation)(columns, center, metric);
}

/++
Evaluate selected observation vectors using `makeMahalanobisBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `makeMahalanobisBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a custom-allocated vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `makeMahalanobisBatch`.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        makeMahalanobisBatch!(form, triangle, summation)(allocator, ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return makeMahalanobisBatch!(form, triangle, summation)(allocator, columns, center, metric);
}

/++
Write distances for selected observations using selected caller scratch.
`x` and `work` are rectangular observation selections made with `alongDim` or
`byDim`; they have equal observation counts and vector lengths, but may have
different storage orientations. `result` is an ordinary vector in selection order.
No allocation occurs. All writable elements must be nonoverlapping and all
writable buffers disjoint from one another and inputs, as in `mahalanobisBatchWithWorkspace`.
Prepared factors and compensated precision need only this workspace. Covariance
and fast precision use the extra-workspace overload.
+/
void mahalanobisWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
            result, ObservationColumns!W.init)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    mahalanobisBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch);
}

/++
Selected-observation workspace overload with additional caller storage.
For covariance, `extra` is an ordinary square factor workspace with metric shape.
For precision, `extra` is an observation selection for product workspace, with
the same observation count and vector length as `x`. Select its axes independently
of its storage orientation. Other contracts follow the one-workspace overload.
+/
void mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == CovarianceRepresentation.covariance && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, extra))) ||
         (form == CovarianceRepresentation.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    static if (form == CovarianceRepresentation.covariance)
        mahalanobisBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch, extra);
    else
    {
        auto product = observationColumns(extra);
        mahalanobisBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch, product);
    }
}

/++
Evaluate selected observation vectors using `mahalanobisSquaredBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `mahalanobisSquaredBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a GC-managed vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `mahalanobisSquaredBatch`.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
mahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        mahalanobisSquaredBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return mahalanobisSquaredBatch!(form, triangle, summation)(columns, center, metric);
}

/++
Evaluate selected observation vectors using `rcMahalanobisSquaredBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `rcMahalanobisSquaredBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a reference-counted vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `rcMahalanobisSquaredBatch`.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        rcMahalanobisSquaredBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return rcMahalanobisSquaredBatch!(form, triangle, summation)(columns, center, metric);
}

/++
Evaluate selected observation vectors using `makeMahalanobisSquaredBatch` in one batch.
`x.alongDim!0` or `x.byDim!1` selects matrix columns;
`x.alongDim!1` or `x.byDim!0` selects matrix rows. Negative dimension indices
follow Mir topology conventions. Selection order determines result order.
The center is one vector shared by all observations; the metric is unchanged.
This accepts rectangular Mir slice-of-slices views, not jagged collections.
A bare matrix requires the explicit `makeMahalanobisSquaredBatch` name.

The adapter changes descriptors only, preserving storage strides and ownership.
Covariance is factorized once; eligible factors use whole-batch TRSM/TRMM.
The result is a custom-allocated vector with one entry per selected observation,
including a one-element vector for a single selected observation. Allocation,
triangle, summation, numerical, and custom-disposal contracts follow `makeMahalanobisSquaredBatch`.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (isObservationSelection!X && __traits(compiles,
        makeMahalanobisSquaredBatch!(form, triangle, summation)(allocator, ObservationColumns!X.init, center, metric)))
{
    auto columns = observationColumns(x);
    return makeMahalanobisSquaredBatch!(form, triangle, summation)(allocator, columns, center, metric);
}

/++
Write distances for selected observations using selected caller scratch.
`x` and `work` are rectangular observation selections made with `alongDim` or
`byDim`; they have equal observation counts and vector lengths, but may have
different storage orientations. `result` is an ordinary vector in selection order.
No allocation occurs. All writable elements must be nonoverlapping and all
writable buffers disjoint from one another and inputs, as in `mahalanobisSquaredBatchWithWorkspace`.
Prepared factors and compensated precision need only this workspace. Covariance
and fast precision use the extra-workspace overload.
+/
void mahalanobisSquaredWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
            result, ObservationColumns!W.init)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch);
}

/++
Selected-observation workspace overload with additional caller storage.
For covariance, `extra` is an ordinary square factor workspace with metric shape.
For precision, `extra` is an observation selection for product workspace, with
the same observation count and vector length as `x`. Select its axes independently
of its storage orientation. Other contracts follow the one-workspace overload.
+/
void mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == CovarianceRepresentation.covariance && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, extra))) ||
         (form == CovarianceRepresentation.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    static if (form == CovarianceRepresentation.covariance)
        mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch, extra);
    else
    {
        auto product = observationColumns(extra);
        mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(columns, center, metric, result, scratch, product);
    }
}

/// Select observation rows or columns explicitly; every call uses one batch.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    import mir.ndslice.dynamic: transposed;
    double[6] rowData = [2, 3, 4, 0, 0, 6];
    double[2] center = [0, 0];
    double[4] covariance = [4, 0, double.nan, 9];
    auto rows = rowData[].sliced(3, 2);
    auto columns = rows.transposed;
    auto metric = covariance[].sliced(2, 2);
    // Batch names accept plain matrices with observations in columns.
    auto expected = mahalanobisSquaredBatch(columns, center[], metric);
    assert(expected == [2, 4, 4]);
    assert(mahalanobisSquared(rows.alongDim!1, center[], metric) == expected);
    assert(mahalanobisSquared(rows.byDim!0, center[], metric) == expected);
    assert(mahalanobisSquared(columns.alongDim!0, center[], metric) == expected);
    assert(mahalanobisSquared(columns.byDim!1, center[], metric) == expected);
    // Vector inputs still return a scalar; one selected observation returns a vector.
    auto scalar = mahalanobisSquared(rows[0], center[], metric);
    static assert(is(typeof(scalar) == double));
    assert(scalar == expected[0]);
    assert(mahalanobisSquared(rows.byDim!0[0 .. 1], center[], metric).length == 1);
    static assert(!__traits(compiles, mahalanobisSquared(rows, center[], metric)));
}

/// Select scratch axes independently; the square covariance-factor workspace is not selected.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    double[6] data = [2, 3, 4, 0, 0, 6], scratch;
    double[4] covariance = [4, 0, double.nan, 9], factor;
    double[2] center = [0, 0];
    double[3] result;
    auto observations = data[].sliced(3, 2).alongDim!1;
    // Scratch stores the same three observation vectors as columns.
    auto work = scratch[].sliced(2, 3).byDim!1;
    mahalanobisSquaredWithWorkspace(observations, center[], covariance[].sliced(2, 2),
        result[], work, factor[].sliced(2, 2));
    assert(result == [2, 4, 4]);
}

/// Fast precision mode takes selected product scratch in addition to selected centered scratch.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    import std.math: fabs;
    double[6] data = [2, 3, 4, 0, 0, 6], scratch, products;
    double[4] precision = [0.25, 0, double.nan, 1.0 / 9];
    double[2] center = [0, 0];
    double[3] result;
    mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.inverseCovariance,
        TrianglePolicy.upper, Summation.fast)(data[].sliced(3, 2).byDim!0,
            center[], precision[].sliced(2, 2), result[],
            scratch[].sliced(3, 2).alongDim!1, products[].sliced(2, 3).alongDim!0);
    foreach (j; 0 .. 3) assert(fabs(result[j] - (j == 0 ? 2 : 4)) < 1e-12);
}

// Every metric form and summation policy forwards to the existing batch implementation.
version(mir_stat_test_distance)
{
    // Generate separate test functions to bound optimization work per combination.
    static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.covarianceCholesky,
        CovarianceRepresentation.inverseCovariance, CovarianceRepresentation.inverseCovarianceCholesky])
    static foreach (summation; [Summation.kbn, Summation.fast])
    @safe pure unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: alongDim, byDim, universal;
        import mir.ndslice.dynamic: reversed, strided;
        import mir.ndslice.allocation: mininitRcslice;
        double[24] data;
        auto rows = data[].sliced(6, 4).universal.strided!(0, 1)(2, 2);
        rows[0,0] = 1; rows[0,1] = 2;
        rows[1,0] = 2; rows[1,1] = 4;
        rows[2,0] = 0; rows[2,1] = 0;
        const selected = rows.alongDim!(-1);
        double[4] identity = [1, 0, double.nan, 1];
        double[2] center = [0, 0];
        double[6] buffer, product;
        double[4] factor;
        double[3] result;
        auto metric = identity[].sliced(2, 2);
        auto expected = mahalanobisSquaredBatch!(form, TrianglePolicy.upper, summation)(
            observationColumns(selected), center[], metric);
        assert(rcMahalanobisSquared!(form, TrianglePolicy.upper, summation)(selected, center[], metric) == expected);
        assert(mahalanobis!(form, TrianglePolicy.upper, summation)(selected, center[], metric) ==
            mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
        assert(rcMahalanobis!(form, TrianglePolicy.upper, summation)(selected, center[], metric) ==
            mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
        auto output = result[].sliced.universal.reversed;
        auto work = buffer[].sliced(2, 3).byDim!1;
        auto extra = product[].sliced(3, 2).byDim!0;
        static if (form == CovarianceRepresentation.covariance)
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
        }
        else static if (form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast)
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, extra);
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, extra);
        }
        else
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work);
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work);
        }
        assert(output == mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
    }

    @safe pure unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: alongDim, byDim, universal;
        import mir.ndslice.dynamic: reversed, strided;
        import mir.ndslice.allocation: mininitRcslice;
        double[24] data;
        auto rows = data[].sliced(6, 4).universal.strided!(0, 1)(2, 2);
        rows[0,0] = 1; rows[0,1] = 2;
        rows[1,0] = 2; rows[1,1] = 4;
        rows[2,0] = 0; rows[2,1] = 0;
        const selected = rows.alongDim!(-1);
        double[4] identity = [1, 0, double.nan, 1];
        double[2] center = [0, 0];
        double[6] buffer, product;
        double[4] factor;
        double[3] result;
        auto tail = rows.byDim!0[0 .. 2].universal.reversed;
        assert(mahalanobisSquared(tail, center[], identity[].sliced(2,2)) == [20,5]);
        double[0] empty;
        assert(mahalanobisSquared(empty[].sliced(0,2).alongDim!1, center[], identity[].sliced(2,2)).length == 0);
        assert(mahalanobisSquared(empty[].sliced(2,0).byDim!1, center[], identity[].sliced(2,2)).length == 0);
        auto owner = mininitRcslice!double(3, 2);
        foreach (i; 0 .. 3) foreach (j; 0 .. 2) owner[i,j] = rows[i,j];
        const retained = owner.byDim!0;
        owner = typeof(owner).init;
        assert(rcMahalanobisSquared(retained, center[], identity[].sliced(2,2)) == [5,20,0]);
        double[][2] jagged;
        static assert(!__traits(compiles, mahalanobisSquared(jagged[], center[], identity[].sliced(2,2))));
        static assert(!__traits(compiles, mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(
            selected, center[], identity[].sliced(2,2), result[], buffer[].sliced(3,2))));
    }
}

/// Custom-allocation selected overloads retain caller ownership of the result.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    import mir.ndslice.dynamic: transposed;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [3, 4, 0, 0], metric = [1, 0, double.nan, 1];
    double[2] center = [0, 0];
    auto distances = makeMahalanobis(Mallocator.instance, data[].sliced(2,2).alongDim!1, center[], metric[].sliced(2,2));
    scope(exit) Mallocator.instance.dispose(distances.field);
    auto squared = makeMahalanobisSquared(Mallocator.instance, data[].sliced(2,2).byDim!0, center[], metric[].sliced(2,2));
    scope(exit) Mallocator.instance.dispose(squared.field);
    assert(distances == [5,0] && squared == [25,0]);
}

import mir.stat.internal.linearAlgebra.views: SymmetricView;

/++
Symmetric metrics infer their stored triangle from the view, including
transposed views. An explicitly supplied triangle must match that metadata.
These overloads accept covariance or inverse covariance, never Cholesky
factors. They borrow the selected storage for the call, preserving the usual
allocation, workspace, batch, and observation-axis contracts.
+/
Element!X mahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobis!(form, triangle)(x, center, metric.borrowStorage)))
{
    return mahalanobis!(form, triangle)(x, center, metric.borrowStorage);
}
/// ditto
Element!X rcMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobis!(form, triangle)(x, center, metric.borrowStorage)))
{
    return rcMahalanobis!(form, triangle)(x, center, metric.borrowStorage);
}
/// ditto
Element!X makeMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobis!(form, triangle)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobis!(form, triangle)(allocator, x, center, metric.borrowStorage);
}
/// ditto
Element!X mahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquared!(form, triangle)(x, center, metric.borrowStorage)))
{
    return mahalanobisSquared!(form, triangle)(x, center, metric.borrowStorage);
}
/// ditto
Element!X rcMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobisSquared!(form, triangle)(x, center, metric.borrowStorage)))
{
    return rcMahalanobisSquared!(form, triangle)(x, center, metric.borrowStorage);
}
/// ditto
Element!X makeMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Allocator, X, C, M)(
    auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobisSquared!(form, triangle)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobisSquared!(form, triangle)(allocator, x, center, metric.borrowStorage);
}
/// ditto
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric, scope auto ref W work)
    if ((form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work)))
{
    return mahalanobisWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work);
}
/// ditto
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == CovarianceRepresentation.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work, factorWork)))
{
    return mahalanobisWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work, factorWork);
}
/// ditto
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric, scope auto ref W work)
    if ((form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work)))
{
    return mahalanobisSquaredWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work);
}
/// ditto
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == CovarianceRepresentation.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work, factorWork)))
{
    return mahalanobisSquaredWithWorkspace!(form, triangle)(x, center, metric.borrowStorage, work, factorWork);
}
/// ditto
Element!X mahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
Element!X rcMahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
Element!X makeMahalanobis(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobis!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobis!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
Element!X mahalanobisWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) &&
        vector!W && vector!P && is(Element!X == Element!W) && is(Element!X == Element!P) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, work, productWork)))
{
    return mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, work, productWork);
}
/// ditto
Element!X mahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
Element!X rcMahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
Element!X makeMahalanobisSquared(CovarianceRepresentation form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobisSquared!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobisSquared!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
Element!X mahalanobisSquaredWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == CovarianceRepresentation.inverseCovariance && compatible!(X,C,M) &&
        vector!W && vector!P && is(Element!X == Element!W) && is(Element!X == Element!P) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, work, productWork)))
{
    return mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, work, productWork);
}
/// ditto
AllocatedSlice!(Allocation.gc, Element!X, 1) mahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisBatch!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobisBatch!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.rc, Element!X, 1) rcMahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobisBatch!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobisBatch!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.custom, Element!X, 1) makeMahalanobisBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobisBatch!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobisBatch!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
void mahalanobisBatchWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        (form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) &&
        (!(form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast) || DiagonalInfo!M.valid) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work)))
{
    return mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work);
}
/// ditto
void mahalanobisBatchWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra)))
{
    return mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra);
}
/// ditto
AllocatedSlice!(Allocation.gc, Element!X, 1) mahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredBatch!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobisSquaredBatch!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.rc, Element!X, 1) rcMahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobisSquaredBatch!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobisSquaredBatch!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.custom, Element!X, 1) makeMahalanobisSquaredBatch(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (batchCompatible!(X, C, M) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobisSquaredBatch!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobisSquaredBatch!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
void mahalanobisSquaredBatchWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        (form != CovarianceRepresentation.covariance || DiagonalInfo!M.valid) &&
        (!(form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast) || DiagonalInfo!M.valid) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work)))
{
    return mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work);
}
/// ditto
void mahalanobisSquaredBatchWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra)))
{
    return mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra);
}
/// ditto
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1) mahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        mahalanobisBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1) rcMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        rcMahalanobisBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobis!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1) makeMahalanobis(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        makeMahalanobisBatch!(form, triangle, summation)(allocator, ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobis!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobis!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
void mahalanobisWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
            result, ObservationColumns!W.init)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work)))
{
    return mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work);
}
/// ditto
void mahalanobisWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == CovarianceRepresentation.covariance && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
                result, ObservationColumns!W.init, extra))) ||
         (form == CovarianceRepresentation.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra)))
{
    return mahalanobisWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra);
}
/// ditto
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1) mahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        mahalanobisSquaredBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return mahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1) rcMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M)(
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        rcMahalanobisSquaredBatch!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, rcMahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage)))
{
    return rcMahalanobisSquared!(form, triangle, summation)(x, center, metric.borrowStorage);
}
/// ditto
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1) makeMahalanobisSquared(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, Allocator)(auto ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric)
    if (isObservationSelection!X && __traits(compiles,
        makeMahalanobisSquaredBatch!(form, triangle, summation)(allocator, ObservationColumns!X.init, center, metric.borrowStorage)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, makeMahalanobisSquared!(form, triangle, summation)(allocator, x, center, metric.borrowStorage)))
{
    return makeMahalanobisSquared!(form, triangle, summation)(allocator, x, center, metric.borrowStorage);
}
/// ditto
void mahalanobisSquaredWithWorkspace(CovarianceRepresentation form,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work)
    if (isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
            result, ObservationColumns!W.init)) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work)))
{
    return mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work);
}
/// ditto
void mahalanobisSquaredWithWorkspace(CovarianceRepresentation form = CovarianceRepresentation.covariance,
    TrianglePolicy triangle, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref const(SymmetricView!(triangle, M)) metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == CovarianceRepresentation.covariance && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
                result, ObservationColumns!W.init, extra))) ||
         (form == CovarianceRepresentation.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric.borrowStorage,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))) && (form == CovarianceRepresentation.covariance || form == CovarianceRepresentation.inverseCovariance) &&
        __traits(compiles, mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra)))
{
    return mahalanobisSquaredWithWorkspace!(form, triangle, summation)(x, center, metric.borrowStorage, result, work, extra);
}

// Structural metrics preserve axis selection, batch workspace, and summation dispatch.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim, universal;
    import mir.ndslice.dynamic: reversed, strided;
    import mir.ndslice.allocation: mininitRcslice;
    double[24] data;
    auto rows = data[].sliced(6, 4).universal.strided!(0, 1)(2, 2);
    rows[0,0] = 1; rows[0,1] = 2;
    rows[1,0] = 2; rows[1,1] = 4;
    rows[2,0] = 0; rows[2,1] = 0;
    const selected = rows.alongDim!(-1);
    double[4] identity = [1, 0, double.nan, 1];
    double[2] center = [0, 0];
    double[6] buffer, product;
    double[4] factor;
    double[3] result;
    static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.inverseCovariance])
    static foreach (summation; [Summation.kbn, Summation.fast])
    {{
        auto metric = symmetricView(identity[].sliced(2, 2));
        auto expected = mahalanobisSquaredBatch!(form, TrianglePolicy.upper, summation)(
            observationColumns(selected), center[], metric);
        assert(rcMahalanobisSquared!(form, TrianglePolicy.upper, summation)(selected, center[], metric) == expected);
        assert(mahalanobis!(form, TrianglePolicy.upper, summation)(selected, center[], metric) ==
            mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
        assert(rcMahalanobis!(form, TrianglePolicy.upper, summation)(selected, center[], metric) ==
            mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
        auto output = result[].sliced.universal.reversed;
        auto work = buffer[].sliced(2, 3).byDim!1;
        auto extra = product[].sliced(3, 2).byDim!0;
        static if (form == CovarianceRepresentation.covariance)
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
        }
        else static if (form == CovarianceRepresentation.inverseCovariance && summation == Summation.fast)
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, extra);
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, extra);
        }
        else
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work);
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work);
        }
        assert(output == mahalanobisBatch!(form, TrianglePolicy.upper, summation)(observationColumns(selected), center[], metric));
    }}
    auto tail = rows.byDim!0[0 .. 2].universal.reversed;
    assert(mahalanobisSquared(tail, center[], identity[].sliced(2,2)) == [20,5]);
    double[0] empty;
    assert(mahalanobisSquared(empty[].sliced(0,2).alongDim!1, center[], identity[].sliced(2,2)).length == 0);
    assert(mahalanobisSquared(empty[].sliced(2,0).byDim!1, center[], identity[].sliced(2,2)).length == 0);
    auto owner = mininitRcslice!double(3, 2);
    foreach (i; 0 .. 3) foreach (j; 0 .. 2) owner[i,j] = rows[i,j];
    const retained = owner.byDim!0;
    owner = typeof(owner).init;
    assert(rcMahalanobisSquared(retained, center[], identity[].sliced(2,2)) == [5,20,0]);
    double[][2] jagged;
    static assert(!__traits(compiles, mahalanobisSquared(jagged[], center[], identity[].sliced(2,2))));
    static assert(!__traits(compiles, mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covarianceCholesky)(
        selected, center[], identity[].sliced(2,2), result[], buffer[].sliced(3,2))));
}

/// Custom-allocation selected overloads retain caller ownership of the result.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    import mir.ndslice.dynamic: transposed;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [3, 4, 0, 0], metric = [1, 0, double.nan, 1];
    double[2] center = [0, 0];
    auto distances = makeMahalanobis(Mallocator.instance, data[].sliced(2,2).alongDim!1, center[], metric[].sliced(2,2));
    scope(exit) Mallocator.instance.dispose(distances.field);
    auto squared = makeMahalanobisSquared(Mallocator.instance, data[].sliced(2,2).byDim!0, center[], metric[].sliced(2,2));
    scope(exit) Mallocator.instance.dispose(squared.field);
    assert(distances == [5,0] && squared == [25,0]);
}

// Generic toSlice forwarding must not discard triangular metadata on metric inputs.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;
    double[2] x = [1,2], center;
    double[4] values = [2,double.nan,1,3];
    auto metric = triangularView!(TrianglePolicy.lower)(values[].sliced(2,2));
    static foreach (form; [CovarianceRepresentation.covariance, CovarianceRepresentation.covarianceCholesky,
        CovarianceRepresentation.inverseCovariance, CovarianceRepresentation.inverseCovarianceCholesky])
        static assert(!__traits(compiles, mahalanobisSquared!form(x[],center[],metric)));
}


/// Preserve explicit diagonal structure; otherwise select the ordinary matrix triangle.
private auto symmetricMetric(TrianglePolicy triangle, M)(return scope M metric)
{
    static if (DiagonalInfo!M.valid) return metric;
    else return symmetricView!triangle(metric);
}

/++
Evaluate independent-coordinate metrics in linear time. Precision entries use
the requested quadratic-form summation; covariance and factor representations
use scaled norms after coordinate scaling. No dense matrix is materialized.
+/
private Element!X diagonalDistance(bool squared, CovarianceRepresentation form,
    Summation summation = Summation.kbn, X,C,M,W)(X x, C center, M metric, W work)
{
    import mir.math.common: sqrt;
    import std.math: isFinite;
    check(x,center,metric,work);
    foreach(i; 0 .. x.length)
    {
        auto d = metric[i,i];
        assert(isFinite(d) && isFinite(x[i]) && isFinite(center[i]), "Mahalanobis inputs must be finite");
        static if (form == CovarianceRepresentation.covariance)
        {
            import mir.exception: enforce;
            enforce!"Cholesky factorization failed"(d > 0);
        }
        else static if (form == CovarianceRepresentation.inverseCovariance)
            assert(d > 0, "Diagonal metric must be positive definite");
        else assert(d != 0, "Diagonal factor must be nonsingular");
        auto delta = x[i] - center[i];
        static if (form == CovarianceRepresentation.covariance) work[i] = delta / sqrt(d);
        else static if (form == CovarianceRepresentation.covarianceCholesky) work[i] = delta / d;
        else static if (form == CovarianceRepresentation.inverseCovarianceCholesky) work[i] = d * delta;
        else static if (form == CovarianceRepresentation.inverseCovariance) work[i] = delta;
        else static assert(0, "Unknown Mahalanobis matrix representation");
    }
    static if (form == CovarianceRepresentation.inverseCovariance)
    {
        auto result = quadraticForm!summation(work,metric);
        static if (squared) return result;
        else return sqrt(result);
    }
    else return scaledNorm!squared(work);
}

// All representations, allocation families, and workspace layouts agree.
version(mir_stat_test_distance)
{
    // Generate separate test functions to bound optimization work per combination.
    static foreach(form; [CovarianceRepresentation.covariance,CovarianceRepresentation.covarianceCholesky,
        CovarianceRepresentation.inverseCovariance,CovarianceRepresentation.inverseCovarianceCholesky])
    static foreach(triangle; [TrianglePolicy.upper,TrianglePolicy.lower])
    static foreach(summation; [Summation.kbn,Summation.fast])
    @system unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: byDim, alongDim, universal;
        import mir.ndslice.dynamic: transposed, reversed;
        import mir.math.common: fabs;
        import std.experimental.allocator.mallocator: Mallocator;
        import std.experimental.allocator: dispose;
        import std.exception: assertThrown;
        import core.exception: AssertError;
        double[2] ds,x=[2,3],center=[0,0],ws,ps,rs;
        double[4] xs=[2,4,3,6],work,extra;
        static if(form==CovarianceRepresentation.covariance) ds=[4,9];
        else static if(form==CovarianceRepresentation.covarianceCholesky) ds=[2,3];
        else static if(form==CovarianceRepresentation.inverseCovariance) ds=[0.25,1.0/9];
        else ds=[0.5,1.0/3];
        auto metric=diagonalView(ds[].sliced);
        assert(fabs(rcMahalanobisSquared!(form,triangle)(x[],center[],metric)-2)<1e-14);
        assert(fabs(makeMahalanobisSquared!(form,triangle)(Mallocator.instance,x[],center[],metric)-2)<1e-14);
        auto batch=xs[].sliced(2,2);
        auto result=rs[].sliced.universal.reversed;
        auto scratch=work[].sliced(2,2).universal.transposed;
        mahalanobisSquaredBatchWithWorkspace!(form,triangle,summation)(batch,center[],metric,result,scratch);
        assert(fabs(result[0]-2)<1e-14 && fabs(result[1]-8)<1e-14);
        auto owned=makeMahalanobisSquaredBatch!(form,triangle,summation)(Mallocator.instance,batch,center[],metric);
        scope(exit) Mallocator.instance.dispose(owned.field);
        assert(fabs(owned[1]-8)<1e-14);
        auto distances=rcMahalanobisBatch!(form,triangle,summation)(batch,center[],metric);
        assert(fabs(distances[1]*distances[1]-8)<1e-13);
        mahalanobisSquaredWithWorkspace!(form,triangle,summation)(batch.byDim!1,center[],metric,
            result,scratch.alongDim!0);
        assert(fabs(result[1]-8)<1e-14);
        static if(form==CovarianceRepresentation.covariance || form==CovarianceRepresentation.inverseCovariance)
        {
            mahalanobisSquaredBatchWithWorkspace!(form,triangle,summation)(batch,center[],metric,result,scratch,extra[].sliced(2,2));
            assert(fabs(result[1]-8)<1e-14);
        }
        static if(form==CovarianceRepresentation.covariance)
        {
            assert(fabs(mahalanobisSquaredWithWorkspace!(form,triangle)(
                x[],center[],metric,ws[],extra[].sliced(2,2))-2)<1e-14);
            assert(fabs(mahalanobisWithWorkspace!(form,triangle)(
                x[],center[],metric,ws[],extra[].sliced(2,2))^^2-2)<1e-14);
        }
        static if(form==CovarianceRepresentation.inverseCovariance)
        {
            assert(fabs(mahalanobisSquaredWithWorkspace!(form,triangle,summation)(x[],center[],metric,ws[],ps[])-2)<1e-14);
            assert(fabs(rcMahalanobisSquared!(form,triangle,summation)(x[],center[],metric)-2)<1e-14);
            assert(fabs(makeMahalanobisSquared!(form,triangle,summation)(Mallocator.instance,x[],center[],metric)-2)<1e-14);
        }
    }

    @system unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: byDim, alongDim, universal;
        import mir.ndslice.dynamic: transposed, reversed;
        import mir.math.common: fabs;
        import std.experimental.allocator.mallocator: Mallocator;
        import std.experimental.allocator: dispose;
        import std.exception: assertThrown;
        import core.exception: AssertError;
        double[1] bad=[0],x=[1],zero=[0];
        assertThrown!Exception(mahalanobisSquared(x[],zero[],diagonalView(bad[].sliced)));
        double[0] empty;
        auto metric=diagonalView(empty[].sliced);
        assert(mahalanobisSquared(empty[],empty[],metric)==0);
        assert(mahalanobisSquaredBatch(empty[].sliced(0,2),empty[],metric)==[0,0]);
        double[1] huge=[double.max/4],one=[1];
        auto norm=mahalanobis!(CovarianceRepresentation.covarianceCholesky)(huge[],zero[],diagonalView(one[].sliced));
        assert(norm==huge[0]); // squaring first would overflow
    }
}

// Allocation counts protect the absence of dense scratch and cleanup on iterator failure.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.internal.linearAlgebra.quadratic: makeQuadraticFormBatch;
    import mir.stat.internal.linearAlgebra.products: makeMtimes;
    import mir.stat.internal.linearAlgebra.cholesky: makeCholeskyDecompose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    struct CountingAllocator
    {
        size_t allocations, releases, bytes;
        /// Count requested storage and delegate allocation to malloc.
        void[] allocate(size_t n)
        {
            ++allocations;
            bytes += n;
            return Mallocator.instance.allocate(n);
        }
        /// Count releases and delegate deallocation to malloc.
        bool deallocate(void[] memory)
        {
            ++releases;
            return Mallocator.instance.deallocate(memory);
        }
    }
    CountingAllocator allocator;
    double[2] diagonal = [4,9], x = [2,3], center = [0,0];
    double[6] observations = [2,4,6,3,6,9];
    auto metric = diagonalView(diagonal[].sliced);
    assert(makeMahalanobisSquared(allocator, x[], center[], metric) == 2);
    assert(allocator.allocations == 1 && allocator.releases == 1);
    auto batch = observations[].sliced(2,3);
    auto results = makeMahalanobisSquaredBatch(allocator, batch, center[], metric);
    assert(results == [2,8,18]);
    allocator.dispose(results.field);
    assert(allocator.allocations == 3 && allocator.releases == 3);
    // Only a vector, a result vector, and one observation-shaped scratch matrix.
    assert(allocator.bytes == (2 + 3 + 6) * double.sizeof);
    auto quadratic = makeQuadraticFormBatch!(Summation.fast)(allocator, batch, metric);
    allocator.dispose(quadratic.field);
    assert(allocator.allocations == 4 && allocator.releases == 4);
    assert(allocator.bytes == (2 + 3 + 6 + 3) * double.sizeof);
    auto failure = new Exception("diagonal iterator failed");
    /// Raise an input error after allocating result or scratch storage.
    double fail(double value) { throw failure; }
    auto failing = diagonalView(diagonal[].sliced.map!fail);
    assertThrown!Exception(makeMtimes(allocator, failing, batch));
    assertThrown!Exception(makeCholeskyDecompose(allocator, failing));
    assertThrown!Exception(makeMahalanobisSquaredBatch(allocator, batch, center[], failing));
    assert(allocator.allocations == 8 && allocator.releases == 8);
}

// Covariance factorization failures remain catchable across every entry path.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.exception: assertThrown;
    import std.experimental.allocator.mallocator: Mallocator;
    foreach (invalid; [0.0, -1.0])
    {
        double[2] values = [1, invalid], x = [1,2], center = [0,0], work, result;
        double[4] observations = [1,2,3,4], scratch, factor;
        auto metric = diagonalView(values[].sliced);
        auto batch = observations[].sliced(2,2);
        auto workspace = scratch[].sliced(2,2);
        auto factorWorkspace = factor[].sliced(2,2);
        assertThrown!Exception(mahalanobis(x[], center[], metric));
        assertThrown!Exception(mahalanobisSquared(x[], center[], metric));
        assertThrown!Exception(rcMahalanobisSquared(x[], center[], metric));
        assertThrown!Exception(makeMahalanobisSquared(Mallocator.instance, x[], center[], metric));
        assertThrown!Exception(mahalanobisWithWorkspace(x[], center[], metric, work[]));
        assertThrown!Exception(mahalanobisSquaredWithWorkspace(x[], center[], metric, work[]));
        assertThrown!Exception(mahalanobisSquaredWithWorkspace(x[], center[], metric, work[], factorWorkspace));
        assertThrown!Exception(mahalanobisBatch(batch, center[], metric));
        assertThrown!Exception(mahalanobisSquaredBatch(batch, center[], metric));
        assertThrown!Exception(rcMahalanobisSquaredBatch(batch, center[], metric));
        assertThrown!Exception(makeMahalanobisSquaredBatch(Mallocator.instance, batch, center[], metric));
        assertThrown!Exception(mahalanobisBatchWithWorkspace!(CovarianceRepresentation.covariance)(
            batch, center[], metric, result[], workspace));
        assertThrown!Exception(mahalanobisSquaredBatchWithWorkspace!(CovarianceRepresentation.covariance)(
            batch, center[], metric, result[], workspace));
        assertThrown!Exception(mahalanobisSquaredBatchWithWorkspace(
            batch, center[], metric, result[], workspace, factorWorkspace));
        assertThrown!Exception(mahalanobisSquared(batch.byDim!0, center[], metric));
        assertThrown!Exception(mahalanobisSquaredWithWorkspace!(CovarianceRepresentation.covariance)(
            batch.byDim!0, center[], metric, result[], workspace.byDim!0));
    }
}

// Allocating batches normalize asSlice-only wrappers just like workspace calls.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: Slice, sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: byDim, alongDim;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    struct MatrixWrapper
    {
        Slice!(double*, 2) data;
        /// Expose only the logical matrix, without shape or length forwarding.
        @property auto asSlice() { return data; }
    }
    struct VectorWrapper
    {
        Slice!(double*) data;
        /// Expose the center through the same conversion convention.
        @property auto asSlice() { return data; }
    }
    double[4] observations = [3, 0, 0, 4];
    double[4] identity = [1, 0, double.nan, 1];
    double[2] zero = [0, 0];
    auto x = MatrixWrapper(observations[].sliced(2, 2));
    auto metric = MatrixWrapper(identity[].sliced(2, 2));
    auto center = VectorWrapper(zero[].sliced);
    static foreach (form; [CovarianceRepresentation.covariance,
        CovarianceRepresentation.covarianceCholesky, CovarianceRepresentation.inverseCovariance,
        CovarianceRepresentation.inverseCovarianceCholesky])
    static foreach (summation; [Summation.kbn, Summation.fast])
    {{
        assert(mahalanobisSquaredBatch!(form, TrianglePolicy.upper, summation)(x, center, metric) == [9, 16]);
        assert(mahalanobisBatch!(form, TrianglePolicy.upper, summation)(x, center, metric) == [3, 4]);
        assert(rcMahalanobisSquaredBatch!(form, TrianglePolicy.upper, summation)(x, center, metric) == [9, 16]);
        assert(rcMahalanobisBatch!(form, TrianglePolicy.upper, summation)(x, center, metric) == [3, 4]);
        auto squared = makeMahalanobisSquaredBatch!(form, TrianglePolicy.upper, summation)(
            Mallocator.instance, x, center, metric);
        scope(exit) Mallocator.instance.dispose(squared.field);
        auto distances = makeMahalanobisBatch!(form, TrianglePolicy.upper, summation)(
            Mallocator.instance, x, center, metric);
        scope(exit) Mallocator.instance.dispose(distances.field);
        assert(squared == [9, 16] && distances == [3, 4]);
        // Selection adapters also accept a metric with no direct shape member.
        assert(mahalanobisSquared!(form, TrianglePolicy.upper, summation)(
            x.data.alongDim!0, center, metric) == [9, 16]);
        assert(rcMahalanobis!(form, TrianglePolicy.upper, summation)(
            x.data.transposed.byDim!0, center, metric) == [3, 4]);
    }}
}
