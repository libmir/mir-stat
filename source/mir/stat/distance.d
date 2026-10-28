/++
Mahalanobis distance calculations with a supplied metric.

Import `mir.stat.distance` directly or through `mir.stat`. The distance APIs
are public; the underlying linear algebra operations remain internal.
`TrianglePolicy.upper` (the default) or `.lower` selects the stored logical
triangle independently of the supplied metric representation.

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

For an already centered vector, internal callers can import `quadraticForm`
from `mir.stat.internal.linearAlgebra.quadratic`; it allocates no workspace.
Pass a `symmetricView` to select one matrix triangle.
Quadratic forms also accept indefinite matrices and may return negative values.

Choose `mahalanobis` or `mahalanobisSquared` for GC scratch storage, their `rc`
variants for temporary reference-counted storage, `make` variants for a custom
allocator, or `WithWorkspace` variants for reusable caller storage. Vector calls return
scalars; their ownership choices apply only to temporary workspaces. Covariance input
needs a matrix and vector workspace; precomputed forms normally need one vector.
Precision-matrix calls can opt into `Summation.fast` as a third template argument
(after the representation and triangle). This uses two vectors and permits BLAS
SYMV plus DOT. GC, RC, custom allocation, and a two-vector `WithWorkspace`
overload are available. Existing calls keep their compensated behavior.

For multiple observations sharing one center and metric, use `mahalanobisBatch`
or `mahalanobisSquaredBatch`. Observations are columns of a `(dimension, count)`
matrix, and the result is a vector of length `count`. GC/RC/custom ownership
applies to the result as well as scratch. `BatchWithWorkspace` variants accept
caller output followed by matrix scratch. Covariance is factorized once per
batch; prepared factors use TRSM/TRMM where eligible. Fast precision batches
use an extra product matrix and permit SYMM plus column dots. All writable
buffers must be mutually disjoint and disjoint from inputs.


For explicit axis selection, pass a rectangular view made by `alongDim` or
`byDim` to the ordinary names. For a matrix, `alongDim!0` / `byDim!1` selects
columns; `alongDim!1` / `byDim!0` selects rows. Negative dimension indices follow
Mir conventions. The outer selection indexes observations and each inner
vector contains its coordinates. A selected observation always produces one
result entry; a vector input to the ordinary function still returns a scalar.
Bare matrices require the explicit `Batch` names, whose observations are columns.

Selection changes view metadata only. Adapters recover the whole matrix and
forward once to the batch implementation, preserving input ownership and
eligible BLAS matrix kernels. They do not map independent distance calls or
repeat covariance factorization per observation. Arbitrary jagged collections
and higher-rank selections are not supported by these adapters.

Storage order never determines observation axes. Select the axes of caller
scratch separately, with the same observation count and coordinate length as
the input selection. Output is an ordinary vector in selection order. Extra
covariance-factor scratch remains an ordinary square matrix; extra precision
product scratch uses an observation selection. All buffers retain the batch
API's disjointness and nonoverlapping-element requirements.

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

/// Shared matrix policies; the algebra operations themselves remain internal.
public import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.quadratic: quadraticForm, quadraticFormWithWorkspace;
import mir.math.sum: Summation;
import mir.stat.internal.linearAlgebra.views: symmetricView, isStructuralView;
import mir.stat.internal.linearAlgebra.allocation;
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
    auto fromFactor = rcMahalanobisSquared!(MahalanobisMatrix.cholesky)(x[], center[], covarianceFactor);
    auto fromPrecision = rcMahalanobisSquared!(MahalanobisMatrix.inverseCovariance)(x[], center[], precision);
    auto fromPrecisionFactor = rcMahalanobisSquared!(MahalanobisMatrix.inverseCovarianceCholesky)(x[], center[], precisionFactor);
    assert(fabs(fromCovariance - 0.8125) < 1e-14);
    assert(fabs(fromFactor - fromCovariance) < 1e-14);
    assert(fabs(fromPrecision - fromCovariance) < 1e-14);
    assert(fabs(fromPrecisionFactor - fromCovariance) < 1e-14);
}

/// Select lower-triangle storage without importing internal linear algebra.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.stat.distance: mahalanobisSquared, MahalanobisMatrix, TrianglePolicy;
    import mir.ndslice.slice: sliced;
    double[2] x = [1, 2], center = [0, 0];
    double[4] covariance = [4, double.nan, 2, 5];
    assert(mahalanobisSquared!(MahalanobisMatrix.covariance, TrianglePolicy.lower)(
        x[], center[], covariance[].sliced(2,2)) == 0.8125);
}

// Borrow ndslice storage directly; array inputs use the ordinary slice view.
// The public call retains the input owners for the lifetime of these views.
/// Borrow an ndslice or array as a view tied to the input lifetime; callers retain owning inputs.
private auto asView(A)(return scope auto ref A value)
{
    static if (isSlice!A) return value.lightScope;
    else return value.toSlice;
}

/// Borrowed ndslice descriptor used for array and slice inputs.
private alias View(A) = typeof(A.init.asView);
/// Unqualified floating-point element type of an observation or workspace.
private alias Element(A) = Unqual!(typeof(View!A.init._iterator[0]));
/// Whether an input can be borrowed as a real floating-point vector.
private enum vector(A) = is(typeof(A.init.asView)) && View!A.N == 1 && isFloatingPoint!(Element!A);
/// Whether an input can be borrowed as a real floating-point matrix.
private enum matrix(A) = !isStructuralView!A && is(typeof(A.init.asView)) && View!A.N == 2 && isFloatingPoint!(Element!A);
/// Whether observation, center, and metric have supported ranks and matching element types.
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
Element!X mahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Element!X rcMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Element!X makeMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Element!X mahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Element!X rcMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Element!X makeMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
`form` must explicitly select a precomputed representation. Inputs follow
`mahalanobis`. `work` must be a writable vector of length `x.length` and the same
element type. Its elements must not overlap each other or any input. No
allocation is performed. Its contents are overwritten and unspecified afterward.
Input iterator exceptions may leave partial workspace contents.
+/
Element!X mahalanobisWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W)(scope auto ref X x,
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
Element!X mahalanobisWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == MahalanobisMatrix.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    auto a = metric.asView;
    auto f = factorWork.asView;
    check(x.asView, center.asView, a, work.asView);
    assert(f.shape == a.shape, "Incorrect Mahalanobis factor workspace shape");
    import mir.stat.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
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
Element!X mahalanobisSquaredWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W)(scope auto ref X x,
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
Element!X mahalanobisSquaredWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, X, C, M, W, F)(scope auto ref X x,
    scope auto ref C center, scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (form == MahalanobisMatrix.covariance && compatible!(X,C,M) && vector!W && matrix!F &&
        is(Element!X == Element!W) && is(Element!X == Element!F))
{
    auto a = metric.asView;
    auto f = factorWork.asView;
    check(x.asView, center.asView, a, work.asView);
    assert(f.shape == a.shape, "Incorrect Mahalanobis factor workspace shape");
    import mir.stat.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
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

/// Assert matching observation, center, workspace, and square metric dimensions.
private void check(X,C,M,W)(X x, C center, M metric, W work)
{
    assert(x.length == center.length && work.length == x.length,
        "Incompatible Mahalanobis vector lengths");
    assert(metric.length!0 == x.length && metric.length!1 == x.length,
        "Incompatible Mahalanobis matrix shape");
}

/// Allocate temporary workspaces according to policy and release custom storage on success or failure.
private Element!X evaluate(Allocation policy, bool squared, MahalanobisMatrix form,
    TrianglePolicy triangle, Allocator, X,C,M)(ref Allocator allocator,
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

/// Evaluate a precomputed metric using a quadratic form, triangular solve, or triangular multiplication.
private Element!X prepared(bool squared, MahalanobisMatrix form, TrianglePolicy triangle, X,C,M,W)(
    X x, C center, M metric, W work)
{
    import std.math: isFinite, sqrt;
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
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
            if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                assert(isFinite(metric[i,j]), "Mahalanobis metric must be finite");
        auto result = quadraticForm(work, symmetricView!triangle(metric));
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
        static if (form == MahalanobisMatrix.cholesky)
        {
            // S=U' U => solve U' z=d. S=L L' => solve L z=d.
            static if (triangle == TrianglePolicy.upper) auto a = metric.universal.transposed;
            else auto a = metric.universal;
            triangularVectorInPlace!(true,TrianglePolicy.lower)(a,work);
        }
        else static if (form == MahalanobisMatrix.inverseCovarianceCholesky)
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
@safe pure @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: sqrt, fabs;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    static foreach (T; AliasSeq!(float,double,real))
    static foreach (triangle; [TrianglePolicy.upper,TrianglePolicy.lower])
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
    assert(makeMahalanobisSquared!(MahalanobisMatrix.cholesky)(allocator,x[],center[],metric) == 0.8125);
    assert(allocator.allocations == 3 && allocator.releases == 3);
    auto failure = new Exception("input failed");
    /// Throw the test exception to verify cleanup after an input iterator fails.
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
    data[] = [0.3125,-0.125,double.nan,0.25];
    enum inverse = MahalanobisMatrix.inverseCovariance;
    assert(makeMahalanobisSquared!(inverse,TrianglePolicy.upper,Summation.fast)(allocator,x[],center[],metric) >= 0);
    assert(allocator.allocations == 10 && allocator.releases == 10);
    assertThrown!Exception(makeMahalanobisSquared!(inverse,TrianglePolicy.upper,Summation.fast)(allocator,x[],center[],metric.map!fail));
    assert(allocator.allocations == 12 && allocator.releases == 12);
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
        static foreach (form; [MahalanobisMatrix.covariance, MahalanobisMatrix.cholesky,
            MahalanobisMatrix.inverseCovariance, MahalanobisMatrix.inverseCovarianceCholesky])
        {
            static assert(is(typeof(mahalanobis!form(x, center, metric)) == T));
            static assert(is(typeof(mahalanobisSquared!form(x, center, metric)) == T));
            static assert(is(typeof(rcMahalanobis!form(x, center, metric)) == T));
            static assert(is(typeof(rcMahalanobisSquared!form(x, center, metric)) == T));
            static assert(is(typeof(makeMahalanobis!form(Mallocator.instance, x, center, metric)) == T));
            static assert(is(typeof(makeMahalanobisSquared!form(Mallocator.instance, x, center, metric)) == T));
            static if (form == MahalanobisMatrix.covariance)
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
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X mahalanobis(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.gc, false, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X rcMahalanobis(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.rc, false, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X makeMahalanobis(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    return precisionAllocated!(Allocation.custom, false, triangle, summation)(
        allocator, x, center, metric);
}

/++
Precision-matrix overload with separate centered-input and product workspaces.
`form` must be `MahalanobisMatrix.inverseCovariance`. Both writable vectors must
have the input element type and length, contain nonoverlapping elements, and
be disjoint from each other and every input. Shapes are asserted; overlap is
not checked. No allocation occurs. On success `work` contains `x - center` and
`productWork` contains `metric * work`; iterator exceptions may leave either
partially written. The default is compensated; explicit `Summation.fast` permits
SYMV and DOT with backend-dependent rounding. Other input contracts match
`mahalanobis`. Negative squared results are not clamped.
+/
Element!X mahalanobisWithWorkspace(MahalanobisMatrix form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M) &&
        vector!W && vector!P && is(Element!X == Element!W) && is(Element!X == Element!P))
{
    return precisionWorkspace!(false, triangle, summation)(
        x.asView, center.asView, metric.asView, work.asView, productWork.asView);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X mahalanobisSquared(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.gc, true, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X rcMahalanobisSquared(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, X, C, M)(scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    int unused;
    return precisionAllocated!(Allocation.rc, true, triangle, summation)(
        unused, x, center, metric);
}

/++
Precision-matrix overload with an explicit summation policy. Supply
`MahalanobisMatrix.inverseCovariance`, a triangle, and `Summation.fast` to permit
BLAS acceleration. Two temporary vectors hold the centered input and its
matrix-vector product; their ownership follows this function's allocation
strategy. Other policies retain their selected reductions. Inputs and numerical
preconditions follow `mahalanobis`; fast mode does not promise compensated
accuracy or backend-independent rounding. Negative squared values are not clamped.
+/
Element!X makeMahalanobisSquared(MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(auto ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M))
{
    return precisionAllocated!(Allocation.custom, true, triangle, summation)(
        allocator, x, center, metric);
}

/++
Precision-matrix overload with separate centered-input and product workspaces.
`form` must be `MahalanobisMatrix.inverseCovariance`. Both writable vectors must
have the input element type and length, contain nonoverlapping elements, and
be disjoint from each other and every input. Shapes are asserted; overlap is
not checked. No allocation occurs. On success `work` contains `x - center` and
`productWork` contains `metric * work`; iterator exceptions may leave either
partially written. The default is compensated; explicit `Summation.fast` permits
SYMV and DOT with backend-dependent rounding. Other input contracts match
`mahalanobis`. Negative squared results are not clamped.
+/
Element!X mahalanobisSquaredWithWorkspace(MahalanobisMatrix form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, W, P)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref W work, scope auto ref P productWork)
    if (form == MahalanobisMatrix.inverseCovariance && compatible!(X,C,M) &&
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
    import std.math: isFinite, sqrt;
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
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
    auto result = quadraticFormWithWorkspace!summation(work, symmetricView!triangle(metric), productWork);
    static if (squared) return result;
    else return sqrt(result);
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
    auto result = mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovariance,
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
        enum form = MahalanobisMatrix.inverseCovariance;
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
private enum batchCompatible(X, C, M) = matrix!X && vector!C && matrix!M &&
    is(Element!X == Element!C) && is(Element!X == Element!M);

/// Validate batch shapes before touching caller-owned output or scratch storage.
private void checkBatch(X, C, M, R, W)(X x, C center, M metric, R result, W work)
{
    assert(center.length == x.length!0 && metric.length!0 == x.length!0 &&
        metric.length!1 == x.length!0, "Incompatible Mahalanobis batch dimensions");
    assert(result.length == x.length!1 && work.shape == x.shape,
        "Incompatible Mahalanobis batch output or workspace");
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
        foreach (j; 0 .. metric.length!1)
            if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                assert(isFinite(metric[i, j]), "Mahalanobis metric must be finite");
    }
}

/// Evaluate a prepared batch after centering, using one triangular matrix operation when possible.
private void preparedBatch(bool squared, MahalanobisMatrix form, TrianglePolicy triangle,
    Summation summation, X, C, M, R, W)(X x, C center, M metric, R result, W work)
{
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import std.math: sqrt;
    checkBatch(x, center, metric, result, work);
    centerBatch!triangle(x, center, metric, work);
    static if (form == MahalanobisMatrix.inverseCovariance)
    {
        import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
        quadraticFormBatchInto!summation(work, symmetricView!triangle(metric), result);
        static if (!squared) foreach (j; 0 .. result.length) result[j] = sqrt(result[j]);
    }
    else
    {
        import mir.stat.internal.linearAlgebra.triangular: triangularMatrixInPlace;
        static if (form == MahalanobisMatrix.cholesky)
        {
            static if (triangle == TrianglePolicy.upper) auto a = metric.universal.transposed;
            else auto a = metric.universal;
            triangularMatrixInPlace!(true, TrianglePolicy.lower)(a, work);
        }
        else static if (form == MahalanobisMatrix.inverseCovarianceCholesky)
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
mahalanobisBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
rcMahalanobisBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
makeMahalanobisBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Covariance and fast precision calls require the extra-workspace overload.
Factor forms compute scaled norms directly, preserving the unsquared API's
range. Finite-input and positive-definite preconditions match the scalar API.
+/
void mahalanobisBatchWithWorkspace(MahalanobisMatrix form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        form != MahalanobisMatrix.covariance &&
        !(form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast))
{
    preparedBatch!(false, form, triangle, summation)(x.asView, center.asView, metric.asView, result.asView, work.asView);
}

/++
Batch workspace overload with an extra matrix. For covariance, `extra` is square
with the metric's shape and receives the Cholesky factor. For a precision matrix,
`extra` has `x.shape` and receives the matrix product. Explicit fast precision
uses SYMM when layouts permit. Other summation policies retain their selected reductions.
All buffers must be mutually disjoint and disjoint from inputs; no allocation
occurs. Other contracts follow the one-workspace overload.
+/
void mahalanobisBatchWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == MahalanobisMatrix.covariance || form == MahalanobisMatrix.inverseCovariance))
{
    auto xv = x.asView; auto cv = center.asView; auto mv = metric.asView;
    auto rv = result.asView; auto wv = work.asView; auto ev = extra.asView;
    checkBatch(xv, cv, mv, rv, wv);
    static if (form == MahalanobisMatrix.covariance)
    {
        import mir.stat.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
        assert(ev.shape == mv.shape);
        choleskyDecomposeInto!triangle(mv, ev);
        preparedBatch!(false, MahalanobisMatrix.cholesky, triangle, summation)(xv, cv, ev, rv, wv);
    }
    else
    {
        import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
        import std.math: sqrt;
        assert(ev.shape == xv.shape);
        centerBatch!triangle(xv, cv, mv, wv);
        quadraticFormBatchInto!summation(wv, symmetricView!triangle(mv), rv, ev);
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
mahalanobisSquaredBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
rcMahalanobisSquaredBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
makeMahalanobisSquaredBatch(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
Covariance and fast precision calls require the extra-workspace overload.
Factor forms compute scaled norms directly, preserving the unsquared API's
range. Finite-input and positive-definite preconditions match the scalar API.
+/
void mahalanobisSquaredBatchWithWorkspace(MahalanobisMatrix form,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W &&
        is(Element!X == Element!R) && is(Element!X == Element!W) &&
        form != MahalanobisMatrix.covariance &&
        !(form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast))
{
    preparedBatch!(true, form, triangle, summation)(x.asView, center.asView, metric.asView, result.asView, work.asView);
}

/++
Batch workspace overload with an extra matrix. For covariance, `extra` is square
with the metric's shape and receives the Cholesky factor. For a precision matrix,
`extra` has `x.shape` and receives the matrix product. Explicit fast precision
uses SYMM when layouts permit. Other summation policies retain their selected reductions.
All buffers must be mutually disjoint and disjoint from inputs; no allocation
occurs. Other contracts follow the one-workspace overload.
+/
void mahalanobisSquaredBatchWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (batchCompatible!(X, C, M) && vector!R && matrix!W && matrix!E &&
        is(Element!X == Element!R) && is(Element!X == Element!W) && is(Element!X == Element!E) &&
        (form == MahalanobisMatrix.covariance || form == MahalanobisMatrix.inverseCovariance))
{
    auto xv = x.asView; auto cv = center.asView; auto mv = metric.asView;
    auto rv = result.asView; auto wv = work.asView; auto ev = extra.asView;
    checkBatch(xv, cv, mv, rv, wv);
    static if (form == MahalanobisMatrix.covariance)
    {
        import mir.stat.internal.linearAlgebra.cholesky: choleskyDecomposeInto;
        assert(ev.shape == mv.shape);
        choleskyDecomposeInto!triangle(mv, ev);
        preparedBatch!(true, MahalanobisMatrix.cholesky, triangle, summation)(xv, cv, ev, rv, wv);
    }
    else
    {
        import mir.stat.internal.linearAlgebra.quadratic: quadraticFormBatchInto;
        import std.math: sqrt;
        assert(ev.shape == xv.shape);
        centerBatch!triangle(xv, cv, mv, wv);
        quadraticFormBatchInto!summation(wv, symmetricView!triangle(mv), rv, ev);

    }
}

/// Allocate batch output and scratch, releasing custom buffers independently on failure.
private AllocatedSlice!(policy, Element!X, 1) batchAllocated(Allocation policy, bool squared,
    MahalanobisMatrix form, TrianglePolicy triangle, Summation summation, Allocator, X, C, M)(
    ref Allocator allocator, scope auto ref X x, scope auto ref C center, scope auto ref M metric)
{
    alias T = Element!X;
    auto result = allocate!(policy, T)(allocator, [x.length!1]);
    scope(failure) { static if (policy == Allocation.custom) {
        import std.experimental.allocator: dispose; allocator.dispose(result.field); } }
    auto work = allocate!(policy, T)(allocator, x.shape);
    scope(exit) { static if (policy == Allocation.custom) {
        import std.experimental.allocator: dispose; allocator.dispose(work.field); } }
    static if (form == MahalanobisMatrix.covariance ||
        (form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast))
    {
        static if (form == MahalanobisMatrix.covariance) auto shape = metric.shape;
        else auto shape = x.shape;
        auto extra = allocate!(policy, T)(allocator, shape);
        scope(exit) { static if (policy == Allocation.custom) {
            import std.experimental.allocator: dispose; allocator.dispose(extra.field); } }
        static if (squared) mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric, result, work, extra);
        else mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric, result, work, extra);
    }
    else
    {
        static if (squared) mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, center, metric, result, work);
        else mahalanobisBatchWithWorkspace!(form, triangle, summation)(x, center, metric, result, work);
    }
    return result;
}


// Compare every representation, triangle, and layout with independent scalar calls.
version(mir_stat_test_distance)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, reversed;
    import std.meta: AliasSeq;
    import std.math: sqrt, fabs;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    static foreach (form; [MahalanobisMatrix.covariance, MahalanobisMatrix.cholesky,
        MahalanobisMatrix.inverseCovariance, MahalanobisMatrix.inverseCovarianceCholesky])
    static foreach (summation; [Summation.kbn, Summation.fast])
    foreach (tx;[false, true]) foreach (tm;[false, true])
    {
        T[6] xs, ws, ps; T[4] ms, fs; T[3] rs; T[2] cs = [3, 4];
        auto x = xs[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
        auto work = ws[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
        auto product = ps[].sliced(tx ? 3 : 2, tx ? 2 : 3).universal;
        if (tx) { x = x.transposed; work = work.transposed; product = product.transposed; }
        auto metric = ms[].sliced(2, 2).universal;
        if (tm) metric = metric.transposed;
        static if (form == MahalanobisMatrix.covariance) { metric[0, 0] = 4; metric[0, 1] = metric[1, 0] = 2; metric[1, 1] = 5; }
        else static if (form == MahalanobisMatrix.cholesky) { metric[0, 0] = 2; metric[0, 1] = metric[1, 0] = 1; metric[1, 1] = 2; }
        else static if (form == MahalanobisMatrix.inverseCovariance) { metric[0, 0] = T(5)/16; metric[0, 1] = metric[1, 0] = T(-1)/8; metric[1, 1] = T(1)/4; }
        else { metric[0, 0] = sqrt(T(5))/4; metric[0, 1] = metric[1, 0] = -1/(2*sqrt(T(5))); metric[1, 1] = 1/sqrt(T(5)); }
        metric[triangle == TrianglePolicy.upper ? 1 : 0, triangle == TrianglePolicy.upper ? 0 : 1] = T.nan;
        foreach (i;0 .. 2) foreach (j;0 .. 3) x[i, j] = cs[i]+T((i+1)*j);
        auto result = rs[].sliced.universal.reversed;
        auto factor = fs[].sliced(2, 2);
        static if (form == MahalanobisMatrix.covariance)
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(x, cs[], metric, result, work, factor);
        else static if (form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast)
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

version(mir_stat_test_distance)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isFinite;
    double[4] identity = [1, 0, double.nan, 1];
    double[2] x = [1e200, 1e200], center = [0, 0], work;
    double[1] result;
    mahalanobisBatchWithWorkspace!(MahalanobisMatrix.cholesky)(x[].sliced(2, 1), center[], identity[].sliced(2, 2), result[], work[].sliced(2, 1));
    assert(isFinite(result[0]) && result[0]>1e200);
    double[0] empty;
    double[3] zeros;
    mahalanobisSquaredBatchWithWorkspace!(MahalanobisMatrix.cholesky)(empty[].sliced(0, 3), empty[], empty[].sliced(0, 0), zeros[], empty[].sliced(0, 3));
    assert(zeros == [0, 0, 0]);
}

version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] a = [1, 0, double.nan, 1]; double[2] x = [2, 3], center = [0, 0];
    auto r = makeMahalanobisSquaredBatch!(MahalanobisMatrix.cholesky)(Mallocator.instance, x[].sliced(2, 1), center[], a[].sliced(2, 2));
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
    assertThrown!Exception(makeMahalanobisSquaredBatch!(MahalanobisMatrix.inverseCovariance,
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
mahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
rcMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
makeMahalanobis(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
void mahalanobisWithWorkspace(MahalanobisMatrix form,
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
void mahalanobisWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == MahalanobisMatrix.covariance && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, extra))) ||
         (form == MahalanobisMatrix.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    static if (form == MahalanobisMatrix.covariance)
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
mahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
rcMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
makeMahalanobisSquared(MahalanobisMatrix form = MahalanobisMatrix.covariance,
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
void mahalanobisSquaredWithWorkspace(MahalanobisMatrix form,
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
void mahalanobisSquaredWithWorkspace(MahalanobisMatrix form = MahalanobisMatrix.covariance,
    TrianglePolicy triangle = TrianglePolicy.upper, Summation summation = Summation.kbn,
    X, C, M, R, W, E)(scope auto ref X x, scope auto ref C center, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref E extra)
    if (isObservationSelection!X && isObservationSelection!W &&
        ((form == MahalanobisMatrix.covariance && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, extra))) ||
         (form == MahalanobisMatrix.inverseCovariance && isObservationSelection!E && __traits(compiles,
            mahalanobisSquaredBatchWithWorkspace!(form, triangle, summation)(ObservationColumns!X.init, center, metric,
                result, ObservationColumns!W.init, ObservationColumns!E.init)))))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    static if (form == MahalanobisMatrix.covariance)
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
    double[6] rowData = [2, 3, 4, 0, 0, 6];
    double[6] columnData = [2, 4, 0, 3, 0, 6];
    double[2] center = [0, 0];
    double[4] covariance = [4, 0, double.nan, 9];
    auto rows = rowData[].sliced(3, 2);
    auto columns = columnData[].sliced(2, 3);
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
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
    import std.math: fabs;
    double[6] data = [2, 3, 4, 0, 0, 6], scratch, products;
    double[4] precision = [0.25, 0, double.nan, 1.0 / 9];
    double[2] center = [0, 0];
    double[3] result;
    mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovariance,
        TrianglePolicy.upper, Summation.fast)(data[].sliced(3, 2).byDim!0,
            center[], precision[].sliced(2, 2), result[],
            scratch[].sliced(3, 2).alongDim!1, products[].sliced(2, 3).alongDim!0);
    foreach (j; 0 .. 3) assert(fabs(result[j] - (j == 0 ? 2 : 4)) < 1e-12);
}

// Every metric form and summation policy forwards to the existing batch implementation.
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
    static foreach (form; [MahalanobisMatrix.covariance, MahalanobisMatrix.cholesky,
        MahalanobisMatrix.inverseCovariance, MahalanobisMatrix.inverseCovarianceCholesky])
    static foreach (summation; [Summation.kbn, Summation.fast])
    {{
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
        static if (form == MahalanobisMatrix.covariance)
        {
            mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
            assert(output == expected);
            mahalanobisWithWorkspace!(form, TrianglePolicy.upper, summation)(selected, center[], metric, output, work, factor[].sliced(2, 2));
        }
        else static if (form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast)
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
    static assert(!__traits(compiles, mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(
        selected, center[], identity[].sliced(2,2), result[], buffer[].sliced(3,2))));
}

/// Custom-allocation selected overloads retain caller ownership of the result.
version(mir_stat_test_distance)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim;
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
    static foreach (form; [MahalanobisMatrix.covariance, MahalanobisMatrix.cholesky,
        MahalanobisMatrix.inverseCovariance, MahalanobisMatrix.inverseCovarianceCholesky])
        static assert(!__traits(compiles, mahalanobisSquared!form(x[],center[],metric)));
}
