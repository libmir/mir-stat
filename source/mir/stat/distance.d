/++
Mahalanobis distance between an observation and a center with a supplied metric.

For difference `d = x - center` and positive-definite covariance `S`, squared
distance is `d.transpose * inverse(S) * d`. Supply `S`, its Cholesky factor,
its inverse (the precision matrix), or a Cholesky factor of its inverse through
`MahalanobisMatrix`. A precision Cholesky factor is not the inverse of a
covariance Cholesky factor: their multiplication orientations differ.

For an upper covariance factor `S = U.transpose * U`, solve `U.transpose * z = d`.
For a lower factor `S = L * L.transpose`, solve `L * z = d`.
For an upper precision factor `P = R.transpose * R`, multiply `z = R * d`;
for a lower factor `P = Q * Q.transpose`, multiply `z = Q.transpose * d`.
The distance is the Euclidean norm of `z`; no matrix inverse is formed.

For an already centered vector, `quadraticForm` is available directly and
allocates no workspace. Pass a `symmetricView` to select one matrix triangle.
Quadratic forms also accept indefinite matrices and may return negative values.

Choose `mahalanobis` or `mahalanobisSquared` for GC scratch storage, their `rc`
variants for temporary reference-counted storage, `make` variants for a custom
allocator, or `WithWorkspace` variants for reusable caller storage. All return
scalars; ownership choices apply only to temporary workspaces. Covariance input
needs a matrix and vector workspace; precomputed forms need only a vector.

Inputs are same-type real float, double or real vectors and matrices. Vectors
may also be dynamic arrays. Only the selected matrix triangle is read; full
covariance/precision matrices are assumed symmetric. Inputs must be finite,
and supplied metrics must be positive definite. Preconditions use assertions;
precomputed matrices and factors are not revalidated for positive definiteness.
Empty vectors with a zero-by-zero metric have zero distance.

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

version(mir_stat_test) version = mir_stat_test_distance;
version(mir_stat_test_linearAlgebra) version = mir_stat_test_distance;
version(mir_stat_test_blas) version = mir_stat_test_distance;

public import mir.math.internal.linearAlgebra.triangular: Triangle;
public import mir.math.internal.linearAlgebra.quadratic: quadraticForm;
import mir.math.internal.linearAlgebra.views: symmetricView;
import mir.math.internal.linearAlgebra.allocation;
import mir.ndslice.slice: toSlice, isSlice;
import std.traits: Unqual, isFloatingPoint;

/// Representation of the supplied covariance or precision matrix.
enum MahalanobisMatrix
{
    /// Symmetric positive-definite covariance S. Factorized on each call.
    covariance,
    /// Upper U with S = U.transpose * U, or lower L with S = L * L.transpose.
    cholesky,
    /// Symmetric positive-definite precision matrix P = inverse(S).
    inverseCovariance,
    /// Upper R with P = R.transpose * R, or lower Q with P = Q * Q.transpose.
    inverseCovarianceCholesky,
}

/// Reuse a covariance factor, a precision matrix, or a precision factor.
version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
    import std.math: fabs;
    double[2] x = [1, 2], center = [0, 0];
    double[4] covarianceData = [4, 2, double.nan, 5];
    double[4] precisionData = [0.3125, -0.125, double.nan, 0.25];
    auto covariance = covarianceData[].sliced(2, 2);
    auto precision = precisionData[].sliced(2, 2);
    auto covarianceFactor = rcCholeskyDecompose(covariance);
    auto precisionFactor = rcCholeskyDecompose(precision);
    auto fromCovariance = rcMahalanobisSquared(x[], center[], covariance);
    auto fromFactor = rcMahalanobisSquared!(MahalanobisMatrix.cholesky)(x[], center[], covarianceFactor);
    auto fromPrecision = rcMahalanobisSquared!(MahalanobisMatrix.inverseCovariance)(x[], center[], precision);
    auto fromPrecisionFactor = rcMahalanobisSquared!(MahalanobisMatrix.inverseCovarianceCholesky)(x[], center[], precisionFactor);
    assert(fabs(fromCovariance - 0.8125) < 1e-14);
    assert(fabs(fromFactor - fromCovariance) < 1e-14);
    assert(fabs(fromPrecision - fromCovariance) < 1e-14);
    assert(fabs(fromPrecisionFactor - fromCovariance) < 1e-14);
}

// Borrow ndslice storage directly; array inputs use the ordinary slice view.
// The public call retains the input owners for the lifetime of these views.
private auto asView(A)(return scope auto ref A value)
{
    static if (isSlice!A) return value.lightScope;
    else return value.toSlice;
}

private alias View(A) = typeof(A.init.asView);
private alias Element(A) = Unqual!(typeof(View!A.init._iterator[0]));
private enum vector(A) = is(typeof(A.init.asView)) && View!A.N == 1 && isFloatingPoint!(Element!A);
private enum matrix(A) = is(typeof(A.init.asView)) && View!A.N == 2 && isFloatingPoint!(Element!A);
private enum compatible(X, C, M) = vector!X && vector!C && matrix!M &&
    is(Element!X == Element!C) && is(Element!X == Element!M);

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
auto mahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M)(
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
auto rcMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M)(
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
auto makeMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, Allocator, X, C, M)(
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
auto mahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M)(
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
auto rcMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M)(
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
auto makeMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, Allocator, X, C, M)(
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
`form` must explicitly select a precomputed representation. Inputs follow
`mahalanobis`. `work` must be a writable vector of length `x.length` and the same
element type. Its elements must not overlap each other or any input. No
allocation is performed. Its contents are overwritten and unspecified afterward.
Input iterator exceptions may leave partial workspace contents.
+/
auto mahalanobisWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work)
    if (form != MahalanobisMatrix.covariance && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W))
{
    return prepared!(false, form, triangle)(x.asView, center.asView, metric.asView, work.asView);
}

/++
Covariance-input overload. `factorWork` is an additional writable square matrix
of the same shape and type as `metric`. Both workspaces must have nonoverlapping
elements and be disjoint from each other and all inputs. No allocation occurs
on success. Factorization failure throws and leaves partial workspace contents.
+/
auto mahalanobisWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == MahalanobisMatrix.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    auto a = metric.asView;
    auto f = factorWork.asView;
    check(x.asView, center.asView, a, work.asView);
    assert(f.shape == a.shape, "Incorrect Mahalanobis factor workspace shape");
    import mir.math.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
    choleskyDecomposeInto!triangle(a, f);
    return prepared!(false, MahalanobisMatrix.cholesky, triangle)(x.asView, center.asView, f, work.asView);
}

/// Reuse vector storage with an already computed covariance Cholesky factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [3,4], center = [0,0], work;
    double[4] factor = [3,0,double.nan,4];
    auto result = mahalanobisWithWorkspace!(MahalanobisMatrix.cholesky)(
        x[].sliced, center[].sliced, factor[].sliced(2,2), work[].sliced);
    assert(result > 1.414 && result < 1.415);
}

/++
Compute squared distance with reusable workspace and a precomputed metric.
`form` must explicitly select a precomputed representation. Inputs follow
`mahalanobis`. `work` must be a writable vector of length `x.length` and the same
element type. Its elements must not overlap each other or any input. No
allocation is performed. Its contents are overwritten and unspecified afterward.
Input iterator exceptions may leave partial workspace contents.
+/
auto mahalanobisSquaredWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M, W)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work)
    if (form != MahalanobisMatrix.covariance && compatible!(X,C,M) &&
        vector!W && is(Element!X == Element!W))
{
    return prepared!(true, form, triangle)(x.asView, center.asView, metric.asView, work.asView);
}

/++
Covariance-input overload. `factorWork` is an additional writable square matrix
of the same shape and type as `metric`. Both workspaces must have nonoverlapping
elements and be disjoint from each other and all inputs. No allocation occurs
on success. Factorization failure throws and leaves partial workspace contents.
+/
auto mahalanobisSquaredWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    Triangle triangle = Triangle.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == MahalanobisMatrix.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    auto a = metric.asView;
    auto f = factorWork.asView;
    check(x.asView, center.asView, a, work.asView);
    assert(f.shape == a.shape, "Incorrect Mahalanobis factor workspace shape");
    import mir.math.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
    choleskyDecomposeInto!triangle(a, f);
    return prepared!(true, MahalanobisMatrix.cholesky, triangle)(x.asView, center.asView, f, work.asView);
}

/// Reuse vector storage with an already computed covariance Cholesky factor.
version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [3,4], center = [0,0], work;
    double[4] factor = [3,0,double.nan,4];
    auto result = mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(
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

private void check(X,C,M,W)(X x, C center, M metric, W work)
{
    assert(x.length == center.length && work.length == x.length,
        "Incompatible Mahalanobis vector lengths");
    assert(metric.length!0 == x.length && metric.length!1 == x.length,
        "Incompatible Mahalanobis matrix shape");
}

private auto evaluate(Allocation policy, bool squared, MahalanobisMatrix form,
    Triangle triangle, Allocator, X,C,M)(ref Allocator allocator,
    scope auto ref X x, scope auto ref C center, scope auto ref M metric)
{
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
    static if (form == MahalanobisMatrix.covariance)
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

private auto prepared(bool squared, MahalanobisMatrix form, Triangle triangle, X,C,M,W)(
    X x, C center, M metric, W work)
{
    import std.math: isFinite, sqrt;
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    static assert(form != MahalanobisMatrix.covariance);
    check(x,center,metric,work);
    alias T = Element!X;
    foreach (i; 0 .. x.length)
    {
        auto a = x[i];
        auto b = center[i];
        assert(isFinite(a) && isFinite(b), "Mahalanobis vectors must be finite");
        work[i] = a - b;
    }
    static if (form == MahalanobisMatrix.inverseCovariance)
    {
        foreach (i; 0 .. metric.length!0) foreach (j; 0 .. metric.length!1)
            if (triangle == Triangle.upper ? j >= i : j <= i)
                assert(isFinite(metric[i,j]), "Mahalanobis metric must be finite");
        auto result = quadraticForm(work, symmetricView!triangle(metric));
        static if (squared) return result;
        else return sqrt(result);
    }
    else
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        import mir.math.internal.linearAlgebra.triangularVector: triangularVectorInPlace;
        foreach (i; 0 .. metric.length!0) foreach (j; 0 .. metric.length!1)
            if (triangle == Triangle.upper ? j >= i : j <= i)
                assert(isFinite(metric[i,j]), "Mahalanobis factor must be finite");
        static if (form == MahalanobisMatrix.cholesky)
        {
            // S=U' U => solve U' z=d. S=L L' => solve L z=d.
            static if (triangle == Triangle.upper) auto a = metric.universal.transposed;
            else auto a = metric.universal;
            triangularVectorInPlace!(true,Triangle.lower)(a,work);
        }
        else static if (form == MahalanobisMatrix.inverseCovarianceCholesky)
        {
            // P=R' R => z=R d. P=Q Q' => z=Q' d.
            static if (triangle == Triangle.lower) auto a = metric.universal.transposed;
            else auto a = metric.universal;
            triangularVectorInPlace!(false,Triangle.upper)(a,work);
        }
        else static assert(0, "Unknown Mahalanobis matrix representation");
        return scaledNorm!squared(work);
    }
}

private auto scaledNorm(bool squared, W)(W work)
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
@safe pure @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: sqrt, fabs;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    static foreach (T; AliasSeq!(float,double,real))
    static foreach (triangle; [Triangle.upper,Triangle.lower])
    static foreach (form; [MahalanobisMatrix.covariance,MahalanobisMatrix.cholesky,
        MahalanobisMatrix.inverseCovariance,MahalanobisMatrix.inverseCovarianceCholesky])
    {{
        foreach (transposeStorage; [false,true])
        foreach (reverseWork; [false,true])
        {
            T[6] data = T(-99);
            auto metric = data[].sliced(2,3)[0..2,0..2].universal;
            if (transposeStorage) metric = metric.transposed;
            static if (form == MahalanobisMatrix.covariance)
            {
                metric[0,0] = 4; metric[1,1] = 5;
                metric[0,1] = metric[1,0] = 2;
            }
            else static if (form == MahalanobisMatrix.cholesky)
            {
                metric[0,0] = 2; metric[1,1] = 2;
                metric[0,1] = metric[1,0] = 1;
            }
            else static if (form == MahalanobisMatrix.inverseCovariance)
            {
                metric[0,0] = T(5)/16; metric[1,1] = T(1)/4;
                metric[0,1] = metric[1,0] = T(-1)/8;
            }
            else
            {
                metric[0,0] = sqrt(T(5))/4; metric[1,1] = 1/sqrt(T(5));
                metric[0,1] = metric[1,0] = -1/(2*sqrt(T(5)));
            }
            metric[triangle == Triangle.upper ? 1 : 0, triangle == Triangle.upper ? 0 : 1] = T.nan;
            // d=[1,2], S=[[4,2],[2,5]], exact quadratic form is 13/16.
            T[4] xs = [T(4),T(-99),T(6),T(-99)];
            T[2] cs = [3,4];
            auto x = xs[].sliced.universal.strided(2);
            auto center = cs[].sliced;
            T[5] ws = T(-99);
            auto work = ws[1..4].sliced.universal.strided(2);
            if (reverseWork) work = work.reversed;
            T squared, distance;
            static if (form == MahalanobisMatrix.covariance)
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
    }}
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
    auto distance = mahalanobisWithWorkspace!(MahalanobisMatrix.cholesky)(x[],center[],a,work[]);
    assert(isFinite(distance) && fabs(distance/1e200 - 1.4142135623730951) < 1e-15);
    assert(mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(x[],center[],a,work[]) == double.infinity);
    x[] = 1e-200;
    distance = mahalanobisWithWorkspace!(MahalanobisMatrix.inverseCovarianceCholesky)(x[],center[],a,work[]);
    assert(distance > 0 && fabs(distance/1e-200 - 1.4142135623730951) < 1e-15);
    assert(mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovarianceCholesky)(x[],center[],a,work[]) == 0);
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
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[2] x = [1,2], center = [0,0];
    double[4] data = [4,2,double.nan,5];
    auto metric = data[].sliced(2,2);
    assert(makeMahalanobisSquared(allocator,x[],center[],metric) == 0.8125);
    assert(allocator.allocations == 2 && allocator.releases == 2);
    data[] = [2,1,double.nan,2];
    assert(makeMahalanobisSquared!(MahalanobisMatrix.cholesky)(allocator,x[],center[],metric) == 0.8125);
    assert(allocator.allocations == 3 && allocator.releases == 3);
    auto failure = new Exception("input failed");
    double fail(double value) { throw failure; }
    assertThrown!Exception(makeMahalanobisSquared!(MahalanobisMatrix.cholesky)(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 4 && allocator.releases == 4);
    assertThrown!Exception(makeMahalanobisSquared(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 6 && allocator.releases == 6);
    data[] = [1,2,double.nan,1];
    assertThrown!Exception(makeMahalanobisSquared(allocator,x[],center[],metric));
    assert(allocator.allocations == 8 && allocator.releases == 8);
    assertThrown!AssertError(makeMahalanobisSquared(allocator,x[0..1],center[],metric));
    assert(allocator.allocations == 8);
    double[1] wrongWork;
    assertThrown!AssertError(mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(x[],center[],metric,wrongWork[]));
}

version(mir_stat_test_distance)
@safe pure @nogc unittest
{
    import std.math: fabs, sqrt;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.math.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
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
    static foreach (triangle; [Triangle.lower, Triangle.upper])
    {{
        auto covarianceView = covariance[].sliced(3,3).universal;
        auto factorView = lower[].sliced(3,3).universal;
        auto precisionView = inverse[].sliced(3,3).universal;
        static if (triangle == Triangle.upper)
        {
            covarianceView = covarianceView.transposed;
            factorView = factorView.transposed;
            precisionView = precisionView.transposed;
        }
        auto precisionFactor = rcCholeskyDecompose!triangle(precisionView);
        static foreach (form; [MahalanobisMatrix.covariance, MahalanobisMatrix.cholesky,
            MahalanobisMatrix.inverseCovariance, MahalanobisMatrix.inverseCovarianceCholesky])
        {{
            static if (form == MahalanobisMatrix.covariance) auto metric = covarianceView;
            else static if (form == MahalanobisMatrix.cholesky) auto metric = factorView;
            else static if (form == MahalanobisMatrix.inverseCovariance) auto metric = precisionView;
            else auto metric = precisionFactor;
            auto squared = rcMahalanobisSquared!(form, triangle)(x[], center[], metric);
            auto distance = rcMahalanobis!(form, triangle)(x[], center[], metric);
            assert(fabs(squared - 14) < 128 * double.epsilon * 14);
            assert(fabs(distance - sqrt(14.0)) < 128 * double.epsilon * sqrt(14.0));
        }}
    }}
}
