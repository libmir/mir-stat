/++
Probability density and natural log density of one observation or a batch from a
multivariate normal distribution. The observations share a mean vector and a positive-definite
covariance matrix. In zero dimensions the density is one and the log density is zero.

$(LREF multivariateNormalPDF) exponentiates each log density, including
the determinant contribution. The result can underflow to zero or overflow
to positive infinity. Batch PDF calls reuse one factorization and log determinant and write the densities
into the LPDF result storage. Use $(LREF multivariateNormalPDFBatch) for observations
in columns or $(LREF multivariateNormalPDF) for selected rows or columns.
For caller-owned buffers, use $(LREF multivariateNormalPDFBatchWithWorkspace) with
a prepared factor or $(LREF tryMultivariateNormalPDFBatchWithWorkspace) with an
unfactored matrix and factor scratch. A failed pivot leaves the result buffer untouched.

For repeated evaluations with the same parameters, $(LREF multivariateNormal)
captures the mean and factor and computes the normalization once. Use the
distribution's $(D lpdf) and $(D pdf) methods for individual observations or
$(D lpdfBatch) and $(D pdfBatch) for a matrix of observations in columns.
$(LREF rcMultivariateNormal) and $(LREF makeMultivariateNormal) select RC and
custom storage. For example, a custom model can evaluate each new observation
with $(D model.lpdfWithWorkspace(x, work)), reusing the same length-k buffer.
The four $(D WithWorkspace) methods are available for every storage policy.

$(H4 Matrix inputs)

Supply a dense square matrix, including a strided view. A view records the
representation and the selected triangle while referencing the existing entries.

$(TABLE
$(TR $(TH Matrix supplied) $(TH View for PDF and LPDF) $(TH PDF accepting the matrix directly) $(TH LPDF accepting the matrix directly))
$(TR $(TD Covariance matrix S) $(TD $(D_INLINECODE covarianceView(s))) $(TD $(D_INLINECODE multivariateNormalPDFFromCovariance)) $(TD $(D_INLINECODE multivariateNormalLPDFFromCovariance)))
$(TR $(TD Precision matrix P (the inverse of S)) $(TD $(D_INLINECODE precisionView(p))) $(TD $(D_INLINECODE multivariateNormalPDFFromPrecision)) $(TD $(D_INLINECODE multivariateNormalLPDFFromPrecision)))
$(TR $(TD Cholesky factor of S) $(TD $(D_INLINECODE covarianceCholeskyView(factor))) $(TD $(D_INLINECODE multivariateNormalPDFFromCovarianceCholesky)) $(TD $(D_INLINECODE multivariateNormalLPDFFromCovarianceCholesky)))
$(TR $(TD Cholesky factor of P) $(TD $(D_INLINECODE precisionCholeskyView(factor))) $(TD $(D_INLINECODE multivariateNormalPDFFromPrecisionCholesky)) $(TD $(D_INLINECODE multivariateNormalLPDFFromPrecisionCholesky)))
)

The upper triangle is the default. These equivalent calls read the lower triangle of p:
---
multivariateNormalLPDF(x, mean, precisionView!(TrianglePolicy.lower)(p));
multivariateNormalLPDFFromPrecision!(TrianglePolicy.lower)(x, mean, p);
---
For an upper precision factor U, $(D_INLINECODE P = U.transpose * U). For a lower precision
factor L, $(D_INLINECODE P = L * L.transpose).

The four view constructors are available from $(MREF mir,stat,covariance_views)
and this module.

$(H4 Factor reuse and failure)

A call with an unfactored covariance or precision matrix computes its Cholesky
factor once, even for a whole batch. Reusing the view in another call reuses
the matrix entries. To reuse the factorization as well, pass a view of the
prepared factor. Each batch computes the log determinant once.

The ordinary allocating functions throw if factorization fails. With an
unfactored matrix, the $(D try) functions report the Cholesky step instead.
For a single observation, $(LREF MultivariateNormalPDFResult) holds the density
and $(LREF MultivariateNormalLPDFResult) holds the log density. Both include
$(D failedPivot). Zero means success. On failure the pivot is one-based and
the density or log density is NaN. For an allocated batch,
$(LREF MultivariateNormalPDFBatchResult) and $(LREF MultivariateNormalLPDFBatchResult)
hold the same pivot and a result vector, empty on failure. An empty batch still checks the matrix.
With custom iterators or allocators, attributes such as $(D nothrow) depend
on the operations they supply.

$(H4 Batch evaluation)

Suppose a test rig records inlet, chamber, and outlet temperatures in degrees
Celsius on each of 100 runs. Store the observations in a 100-by-3 matrix,
one run per row. The following call returns 100 log densities in run order,
so $(D result[1]) corresponds to the second run.
---
auto result = multivariateNormalLPDF(rows.byDim!0, mean, covarianceView(covariance));
---
The default result uses GC storage, the $(D rc) form returns an owning
reference-counted vector, and the $(D make) form uses the supplied allocator
for result and scratch. For a 3-by-100 matrix with one run per column, pass
the matrix directly to the corresponding $(D Batch) function.

$(H4 Batch workspace and performance)

With caller-owned storage, supply a result vector of length 100 and select
the rows of a 100-by-3 scratch matrix with $(D workRows.byDim!0). If the
second run records $(D [12, 21, 35]) and the mean is $(D [10, 20, 30]),
its scratch row holds $(D [2, 1, 5]) after centering. The batch transforms
those centered values in place. A transpose view lets the batch routine
process observations as columns while the three readings of each run stay
together in row-major storage. Subtraction takes place in the input element
type and can overflow before the factor is applied.

The full batch uses 300 observation scratch elements (100 runs × 3 readings per run).
Processing ten runs per call uses 30. The workspace call for an unfactored
matrix also needs a 3-by-3 factor workspace. A prepared factor can be shared
across the smaller calls. With a compatible layout, BLAS can transform the
full batch in one triangular solve or multiply. The faster batch size
depends on the dimensions, storage layout, and backend.

$(H4 Probability ellipsoids)

For a normal observation in k dimensions with a specified mean and
positive-definite covariance, its squared Mahalanobis distance from the mean
has a chi-squared distribution with k degrees of freedom. The ellipsoid
containing 95% of observations consists of the points whose squared distance
is at most $(D chi2InvCDF(0.95, k)). The same cutoff applies to every
observation from that distribution.

Given a squared distance $(D d2), $(D chi2CDF(d2, k)) is the probability
inside the contour through that point. Use $(D chi2CCDF(d2, k)) for the
probability of a new observation lying farther from the mean.

For example, suppose two correlated measurements have mean $(D [0, 0]) and
a covariance with lower Cholesky factor $(D [[2, 0], [1, 1]]). For the point
$(D [4, 2]), the squared distance is 4.
---
double[2] point = [4, 2], mean = [0, 0];
double[4] lower = [2, 0, 1, 1];
auto metric = covarianceCholeskyView!(TrianglePolicy.lower)(lower[].sliced(2, 2));
const d2 = mahalanobisSquared(point[], mean[], metric);
assert(d2 <= chi2InvCDF(0.95, 2)); // Inside the 95% ellipse (cutoff about 5.99).
const outside = chi2CCDF(d2, 2);    // About 13.5% of draws lie farther out.
---

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.multivariate_normal;

version(mir_stat_test) version = mir_stat_test_multivariate_normal;
version(mir_stat_test_blas) version = mir_stat_test_multivariate_normal;

public import mir.stat.covariance_views: CovarianceMetric, CovarianceRepresentation, TrianglePolicy,
    covarianceView, precisionView, covarianceCholeskyView, precisionCholeskyView;
private import mir.stat.covariance_views: isCovarianceMetric, labelCovariance;
private import mir.stat.internal.linearAlgebra.types: isObservationSelection, ObservationColumns, observationColumns,
    asView, compatible, matrix, vector, Element;
private import mir.stat.distance: mahalanobisWithWorkspace;
private import mir.stat.internal.linearAlgebra.cholesky: tryCholeskyDecomposeInto, logDeterminantFromCholesky;
private import mir.stat.internal.linearAlgebra.allocation: Allocation, AllocatedSlice, allocate;

/// Whether the caller supplied a Cholesky factor.
private enum prepared(CovarianceRepresentation form) =
    form == CovarianceRepresentation.covarianceCholesky ||
    form == CovarianceRepresentation.inverseCovarianceCholesky;

/++
A multivariate normal distribution with its own mean and Cholesky factor.
Use $(LREF multivariateNormal) for GC storage or $(LREF rcMultivariateNormal)
for reference-counted storage, or $(LREF makeMultivariateNormal) for a custom allocator.
Copies share the captured parameters. GC and RC models retain their storage.
For a custom model, release the storage exactly once after the last use of
every copy. Workspace methods accept disjoint caller-owned buffers. Allocating
methods on a custom model take an explicit allocator for the evaluation. The default value is the zero-dimensional
distribution.

Params:
    T = Floating-point coordinate type.
    policy = Storage policy selected by the factory, GC by default.
+/
struct MultivariateNormal(T, Allocation policy = Allocation.gc)
    if (is(T == float) || is(T == double) || is(T == real))
{
    // One backing allocation: the first k values are the mean and the rest
    // store a k-by-k lower factor. Only its selected triangle is read.
    private AllocatedSlice!(policy, T, 1) parameters;
    private size_t k;
    private bool precision;
    private LogDensityTerms!T terms;

    private auto distances(alias evaluateDistance, X, W...)(scope auto ref X x, scope auto ref W buffers) const
    {
        import mir.ndslice.slice: sliced;
        auto data = parameters.lightScope.field;
        auto center = data[0 .. k];
        auto storage = data[k .. $].sliced(k, k);
        if (precision)
            return evaluateDistance(x, center, precisionCholeskyView!(TrianglePolicy.lower)(storage), buffers);
        return evaluateDistance(x, center, covarianceCholeskyView!(TrianglePolicy.lower)(storage), buffers);
    }

    /++
    Evaluate the log density with caller-owned scratch.
    Params:
        x = One finite observation with the same length and element type as the mean.
        work = Writable vector of the same shape and type as x, disjoint from the inputs.
    Returns:
        Natural log density at x, or zero in zero dimensions.
    +/
    T lpdfWithWorkspace(X, W)(scope auto ref X x, scope auto ref W work) const
        if (vector!X && vector!W && is(Element!X == T) && is(Element!W == T))
    {
        return terms.fromDistance(distances!mahalanobisWithWorkspace(x, work));
    }

    /++
    Evaluate the density with caller-owned scratch.
    Params:
        x = One finite observation with the same length and element type as the mean.
        work = Writable vector of the same shape and type as x, disjoint from the inputs.
    Returns:
        Density at x, or one in zero dimensions. Exponentiation can underflow or overflow.
    +/
    T pdfWithWorkspace(X, W)(scope auto ref X x, scope auto ref W work) const
        if (vector!X && vector!W && is(Element!X == T) && is(Element!W == T))
    {
        import mir.math.common: exp;
        return exp(lpdfWithWorkspace(x, work));
    }

    /++
    Write one log density per observation column using caller-owned storage.
    Params:
        columns = Matrix of finite observations with k rows and the model's element type.
        result = Writable vector of one log density per column, in column order.
        work = Writable matrix of the same shape as columns. All buffers must have
            the model's element type and be disjoint from each other and the inputs.
    +/
    void lpdfBatchWithWorkspace(X, R, W)(scope auto ref X columns, scope auto ref R result,
        scope auto ref W work) const
        if (matrix!X && vector!R && matrix!W && is(Element!X == T) && is(Element!R == T) && is(Element!W == T))
    {
        import mir.stat.distance: mahalanobisBatchWithWorkspace;
        distances!mahalanobisBatchWithWorkspace(columns, result, work);
        foreach (j; 0 .. result.length)
            result[j] = terms.fromDistance(result[j]);
    }

    /++
    Write one density per observation column using caller-owned storage.
    Params:
        columns = Matrix of finite observations with k rows and the model's element type.
        result = Writable vector of one density per column, in column order.
        work = Writable matrix of the same shape as columns. All buffers must have
            the model's element type and be disjoint from each other and the inputs.
    +/
    void pdfBatchWithWorkspace(X, R, W)(scope auto ref X columns, scope auto ref R result,
        scope auto ref W work) const
        if (matrix!X && vector!R && matrix!W && is(Element!X == T) && is(Element!R == T) && is(Element!W == T))
    {
        lpdfBatchWithWorkspace(columns, result, work);
        exponentiateLogDensities(result);
    }

    /++
    Evaluate the natural log density using the stored normalization.
    Params:
        x = One vector of finite coordinates matching the mean's length and element type.
    Returns:
        Log density at x, or zero in zero dimensions.
    +/
    T lpdf(X)(scope auto ref X x) const
        if (policy != Allocation.custom && vector!X && is(Element!X == T))
    {
        int allocator;
        return allocatedLogDensity!false(allocator, x);
    }

    /++
    Evaluate the density using the stored normalization.
    Params:
        x = One vector of finite coordinates matching the mean's length and element type.
    Returns:
        Density at x, or one in zero dimensions. Exponentiation can underflow or overflow.
    +/
    T pdf(X)(scope auto ref X x) const
        if (policy != Allocation.custom && vector!X && is(Element!X == T))
    {
        import mir.math.common: exp;
        return exp(lpdf(x));
    }

    /++
    Evaluate log densities for a batch using the stored normalization.
    Params:
        columns = Matrix with one observation per column and one row per coordinate,
            with the same floating-point element type as the mean.
    Returns:
        A vector of log densities in column order using the model's allocation policy.
    +/
    auto lpdfBatch(X)(scope auto ref X columns) const
        if (policy != Allocation.custom && matrix!X && is(Element!X == T))
    {
        int allocator;
        return allocatedLogDensity!true(allocator, columns);
    }

    /++
    Evaluate densities for a batch using the stored normalization.
    Params:
        columns = Matrix of finite observations in columns, with one row per coordinate
            and the same floating-point element type as the mean.
    Returns:
        A vector of densities in column order using the model's allocation policy. Exponentiation can underflow or overflow.
    +/
    auto pdfBatch(X)(scope auto ref X columns) const
        if (policy != Allocation.custom && matrix!X && is(Element!X == T))
    {
        auto values = lpdfBatch(columns);
        exponentiateLogDensities(values);
        return values;
    }

    /// Each result corresponds to one observation column.
    version(mir_stat_test_multivariate_normal)
    @safe pure nothrow unittest
    {
        import mir.ndslice.slice: sliced;
        double[1] mean = [1], factor = [2];
        double[2] data = [1, 3];
        auto model = multivariateNormal(mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
        auto batch = model.pdfBatch(data[].sliced(1, 2));
        assert(batch[0] == model.pdf(data[0 .. 1]));
        assert(batch[1] == model.pdf(data[1 .. 2]));
    }

    private auto allocatedLogDensity(bool batch, A, X)(ref A allocator, scope auto ref X x) const
    {
        import std.experimental.allocator: dispose;
        static if (batch)
        {
            // Keep the result below scratch so a region can release scratch on return.
            auto result = allocate!(policy, T)(allocator, [x.asView.length!1]);
            scope(failure) static if (policy == Allocation.custom) allocator.dispose(result.field);
            auto work = allocate!(policy, T)(allocator, x.asView.shape);
            scope(exit) static if (policy == Allocation.custom) allocator.dispose(work.field);
            lpdfBatchWithWorkspace(x, result, work);
            return result;
        }
        else
        {
            auto work = allocate!(policy, T)(allocator, [k]);
            scope(exit) static if (policy == Allocation.custom) allocator.dispose(work.field);
            return lpdfWithWorkspace(x, work);
        }
    }

    static if (policy == Allocation.custom)
    {
        /++
        Release the model's parameter storage and reset this handle. All other
        copies become invalid. Call once after the last use of every copy.
        Params:
            allocator = The allocator that supplied the parameter storage.
        +/
        void dispose(A)(auto ref A allocator) @system
        {
            import std.experimental.allocator : release = dispose;
            if (parameters.length) release(allocator, parameters.field);
            this = typeof(this).init;
        }

        /++
        Evaluate log density using the supplied allocator.
        Params:
            allocator = Allocator for scratch.
            x = Vector of k finite coordinates. Its element type must match the model's.
        Returns:
            The log density at x.
        +/
        T lpdf(A, X)(auto ref A allocator, scope auto ref X x) const
            if (vector!X && is(Element!X == T))
        {
            return allocatedLogDensity!false(allocator, x);
        }

        /++
        Evaluate density using the supplied allocator.
        Params:
            allocator = Allocator for scratch.
            x = Vector of k finite coordinates. Its element type must match the model's.
        Returns:
            The density at x.
        +/
        T pdf(A, X)(auto ref A allocator, scope auto ref X x) const
            if (vector!X && is(Element!X == T))
        {
            import mir.math.common: exp;
            return exp(lpdf(allocator, x));
        }

        /++
        Evaluate log densities using the supplied allocator.
        Params:
            allocator = Allocator for scratch and result storage.
            columns = Matrix of finite observation columns with k rows. Its element type must match the model's.
        Returns:
            A vector of log densities in column order. Release its field with the evaluation allocator.
        +/
        auto lpdfBatch(A, X)(auto ref A allocator, scope auto ref X columns) const
            if (matrix!X && is(Element!X == T))
        {
            return allocatedLogDensity!true(allocator, columns);
        }

        /++
        Evaluate densities using the supplied allocator.
        Params:
            allocator = Allocator for scratch and result storage.
            columns = Matrix of finite observation columns with k rows. Its element type must match the model's.
        Returns:
            A vector of densities in column order. Release its field with the evaluation allocator.
        +/
        auto pdfBatch(A, X)(auto ref A allocator, scope auto ref X columns) const
            if (matrix!X && is(Element!X == T))
        {
            auto result = lpdfBatch(allocator, columns);
            exponentiateLogDensities(result);
            return result;
        }
    }

}

/// Result of constructing a multivariate normal distribution.
struct MultivariateNormalResult(T, Allocation policy = Allocation.gc)
{
    import std.typecons: Nullable;
    /// Zero on success, or the one-based index of the first failed Cholesky pivot.
    size_t failedPivot;
    /// The distribution on success, including in zero dimensions. Null on factorization failure.
    Nullable!(MultivariateNormal!(T, policy)) distribution;
}

private enum modelInputs(C, M) = vector!C && isCovarianceMetric!M &&
    is(Element!C == Element!(typeof(M.storage)));

/++
Capture the parameters of a multivariate normal distribution in GC-managed storage.
Construction copies the selected matrix triangle. An unfactored input is
decomposed on the copy, and a supplied Cholesky factor is copied directly.

Params:
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as mean. The matrix must be positive definite.
        A supplied factor must have finite selected entries and a positive diagonal.
Returns:
    An owning $(LREF MultivariateNormal) that reuses its factor and log determinant across evaluations.
Throws:
    Exception if Cholesky factorization fails for an unfactored input.
+/
auto multivariateNormal(C, M)(scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    return checkedModel!M(tryMultivariateNormal(mean, metric));
}

/// Evaluate two observations against the same captured parameters.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[2] mean = [1, 2], x = [2, 3];
    double[4] covariance = [4, 1, 1, 2];
    auto metric = covarianceView(covariance[].sliced(2, 2));
    auto model = multivariateNormal(mean[], metric);
    assert(model.lpdf(x[]) == multivariateNormalLPDF(x[], mean[], metric));
    x = [3, 5];
    auto expected = multivariateNormalPDF(x[], mean[], metric);
    mean[] = 0;
    covariance[] = 0;
    assert(model.pdf(x[]) == expected);
}

/++
Construct a distribution and report a failed Cholesky pivot. Ordinary numeric
arrays and slices support nothrow construction. A supplied factor is trusted.

Params:
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or a prepared factor of either,
        with the same element type as mean. A prepared factor must have finite
        selected entries and a positive diagonal.
Returns:
    A $(LREF MultivariateNormalResult) with an owning distribution on success,
    or the one-based pivot index and a null distribution on factorization failure.
+/
auto tryMultivariateNormal(C, M)(scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    int allocator;
    return buildModel!(Allocation.gc)(allocator, mean, metric);
}

/// A singular covariance fails at its second pivot.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[2] mean = [0, 0];
    double[4] entries = [1, 1, 1, 1];
    auto outcome = tryMultivariateNormal(mean[], covarianceView(entries[].sliced(2, 2)));
    assert(outcome.failedPivot == 2 && outcome.distribution.isNull);
}

/++
Capture a multivariate normal distribution in reference-counted storage.
Params:
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as mean. Prepared factors require finite
        selected entries and a positive diagonal.
Returns:
    A distribution whose copies retain their parameters. Allocated batch results
    retain their own storage independently of the model.
Throws:
    Exception if Cholesky factorization fails for an unfactored input.
+/
auto rcMultivariateNormal(C, M)(scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    return checkedModel!M(tryRcMultivariateNormal(mean, metric));
}

/// A prepared covariance factor can be captured without the GC.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] mean = [2], factor = [3], point = [1];
    auto metric = covarianceCholeskyView(factor[].sliced(1, 1));
    auto model = rcMultivariateNormal(mean[], metric);
    assert(model.lpdf(point[]) == rcMultivariateNormalLPDF(point[], mean[], metric));
}

/++
Capture a distribution in reference-counted storage and report a failed pivot.
Params:
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either,
        matching the mean's element type. Prepared factors require finite
        selected entries and a positive diagonal.
Returns:
    The reference-counted model and pivot zero on success. Factorization
    failure gives a null model and the one-based failed pivot.
+/
auto tryRcMultivariateNormal(C, M)(scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    int allocator;
    return buildModel!(Allocation.rc)(allocator, mean, metric);
}

/// A batch result retains its storage after the model leaves scope.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    auto makeModel()
    {
        double[1] mean = [2], precision = [4];
        auto result = tryRcMultivariateNormal(mean[], precisionView(precision[].sliced(1, 1)));
        assert(result.failedPivot == 0);
        return result.distribution.get;
    }
    auto evaluateCopy()
    {
        auto model = makeModel();
        auto copy = model;
        model = typeof(model).init;
        double[2] data = [1, 2];
        auto batch = copy.pdfBatch(data[].sliced(1, 2));
        assert(batch[0] == copy.pdf(data[0 .. 1]));
        return batch;
    }
    auto values = evaluateCopy();
    assert(values[0] < values[1]);
    double[1] x = [1], negative = [-1];
    auto failure = tryRcMultivariateNormal(x[], covarianceView(negative[].sliced(1, 1)));
    assert(failure.failedPivot == 1 && failure.distribution.isNull);
    auto success = tryRcMultivariateNormal(x[], precisionView(x[].sliced(1, 1)));
    auto retained = success;
    success = failure;
    assert(success.failedPivot == 1 && success.distribution.isNull);
    assert(retained.failedPivot == 0 && retained.distribution.get.pdf(x[]) > 0);
}

/++
Capture a distribution in storage supplied by an allocator.
Params:
    allocator = Allocator for the captured parameters. Use it to dispose of the model.
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either,
        matching the element type of mean. Prepared factors require finite selected
        entries and a positive diagonal.
Returns:
    A custom-allocated distribution. Release it with $(D model.dispose(allocator))
    once after the last use of all copies.
Throws:
    Exception if Cholesky factorization fails for an unfactored input.
+/
auto makeMultivariateNormal(A, C, M)(auto ref A allocator, scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    return checkedModel!M(tryMakeMultivariateNormal(allocator, mean, metric));
}

/// Reuse a custom model with stack scratch to score a single observation.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    double[1] mean = [1], factor = [2], x = [3], work;
    auto model = makeMultivariateNormal(Mallocator.instance, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) model.dispose(Mallocator.instance);
    assert(model.lpdfWithWorkspace(x[], work[]) == model.lpdf(Mallocator.instance, x[]));
}

/++
Capture a distribution using an allocator and report factorization failure.
Params:
    allocator = Allocator for the captured parameters.
    mean = Vector of k finite means.
    metric = Labeled k-by-k covariance, precision, or Cholesky factor of either,
        matching the element type of mean. Prepared factors require finite selected
        entries and a positive diagonal.
Returns:
    On success, the custom model and pivot zero. Dispose of its storage once after
    the last use of all copies. On failure, the one-based pivot and a null model;
    the parameter storage has been released.
+/
auto tryMakeMultivariateNormal(A, C, M)(auto ref A allocator, scope auto ref C mean, scope auto ref M metric)
    if (modelInputs!(C, M))
{
    return buildModel!(Allocation.custom)(allocator, mean, metric);
}

version(mir_stat_test_multivariate_normal)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.exception: assertThrown;
    struct Counter
    {
        size_t live;
        void[] allocate(size_t n)
        {
            auto block = Mallocator.instance.allocate(n);
            if (block.length)
                ++live;
            return block;
        }
        bool deallocate(void[] block)
        {
            if (block.length)
                --live;
            return Mallocator.instance.deallocate(block);
        }
    }
    Counter owner, evaluation;
    double[2] mean = [1, 2], point = [2, 1], scratch;
    double[4] entries = [4, 1, 0, 2];
    auto matrix = entries[].sliced(2, 2);
    auto model = makeMultivariateNormal(owner, mean[], precisionView(matrix));
    assert(owner.live == 1);
    auto batch = model.pdfBatch(evaluation, point[].sliced(2, 1));
    scope(exit) evaluation.dispose(batch.field);
    assert(batch[0] == model.pdf(evaluation, point[]));
    assert(evaluation.live == 1);
    auto error = new Exception("input iterator failed");
    double fail(double x) { throw error; }
    assertThrown!Exception(model.lpdfBatch(evaluation, point[].sliced(2, 1).map!fail));
    assert(evaluation.live == 1);
    model.dispose(owner);
    assert(owner.live == 0 && batch[0] > 0);
    entries[] = [1, 1, 0, 1];
    auto failed = tryMakeMultivariateNormal(owner, mean[], covarianceView(matrix));
    assert(failed.failedPivot == 2 && failed.distribution.isNull && owner.live == 0);
    assertThrown!Exception(makeMultivariateNormal(owner, mean[], covarianceView(matrix)));
    assert(owner.live == 0);
    assertThrown!Exception(makeMultivariateNormal(owner, mean[].sliced.map!fail, covarianceCholeskyView(matrix)));
    assert(owner.live == 0);
    auto empty = tryMakeMultivariateNormal(owner, mean[0 .. 0], covarianceView(entries[0 .. 0].sliced(0, 0)));
    assert(empty.failedPivot == 0 && !empty.distribution.isNull);
    empty.distribution.get.dispose(owner);
    assert(owner.live == 0);
}

// Scratch must be released while the caller still owns the batch result.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.building_blocks.region: BorrowedRegion;

    double[1] mean = [0], factor = [1], scalarWork;
    double[2] observations = [0, 1], parameters;
    double[4] batchStorage;
    alias Region = BorrowedRegion!(double.alignof);
    auto owner = Region(cast(ubyte[]) parameters[]);
    auto region = Region(cast(ubyte[]) batchStorage[]);
    auto model = makeMultivariateNormal(owner, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) model.dispose(owner);
    auto columns = observations[].sliced(1, 2);
    foreach (attempt; 0 .. 2)
    {
        auto logs = model.lpdfBatch(region, columns);
        assert(region.available == batchStorage.sizeof - observations.sizeof);
        assert(logs[1] == model.lpdfWithWorkspace(observations[1 .. 2], scalarWork[]));
        region.dispose(logs.field);
        assert(region.available == batchStorage.sizeof);

        auto densities = model.pdfBatch(region, columns);
        assert(region.available == batchStorage.sizeof - observations.sizeof);
        assert(densities[1] == model.pdfWithWorkspace(observations[1 .. 2], scalarWork[]));
        region.dispose(densities.field);
        assert(region.available == batchStorage.sizeof);
    }
}

version(mir_stat_test_multivariate_normal)
@system unittest
{
    import core.exception: OutOfMemoryError;
    import mir.ndslice.topology: repeat;
    import std.exception: assertThrown;

    // Lazy inputs describe the shape without allocating any elements.
    enum n = size_t(1) << (size_t.sizeof * 4);
    auto mean = repeat(0.0, n);
    auto factor = covarianceCholeskyView(repeat(1.0, n, n));
    assertThrown!OutOfMemoryError(tryMultivariateNormal(mean, factor));
}

private auto checkedModel(M, R)(R outcome)
{
    static if (!prepared!(M.form))
    {
        import mir.exception: enforce;
        enforce!"Cholesky factorization failed"(outcome.failedPivot == 0);
    }
    return outcome.distribution.get;
}

private auto buildModel(Allocation policy, A, C, M)(ref A allocator, scope auto ref C mean, scope auto ref M metric)
{
    import core.exception: onOutOfMemoryError;
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.cholesky: tryCholeskyDecomposeInPlace;
    import std.typecons: nullable;
    alias T = Element!C;
    alias Result = MultivariateNormalResult!(T, policy);
    auto input = metric.storage.asView;
    const n = mean.length;
    assert(input.length!0 == n && input.length!1 == n);
    if (n && n > (size_t.max - n) / n)
        onOutOfMemoryError();
    MultivariateNormal!(T, policy) model;
    if (!n) return Result(0, nullable(model));
    model.k = n;
    model.parameters = allocate!(policy, T)(allocator, [n + n * n]);
    scope(failure) static if (policy == Allocation.custom) model.dispose(allocator);
    auto data = model.parameters.lightScope.field;
    auto storage = data[n .. $].sliced(n, n);
    foreach (i; 0 .. n)
        foreach (j; 0 .. i + 1)
            static if (M.triangle == TrianglePolicy.lower)
                storage[i, j] = input[i, j];
            else
                storage[i, j] = input[j, i];
    static if (!prepared!(M.form))
    {
        auto pivot = tryCholeskyDecomposeInPlace!(TrianglePolicy.lower)(storage);
        if (pivot)
        {
            static if (policy == Allocation.custom) model.dispose(allocator);
            return Result(pivot);
        }
    }

    foreach (i; 0 .. n)
        data[i] = mean[i];
    static if (M.form == CovarianceRepresentation.inverseCovariance ||
               M.form == CovarianceRepresentation.inverseCovarianceCholesky)
    {
        model.precision = true;
        model.terms = logDensityTerms(precisionCholeskyView!(TrianglePolicy.lower)(storage));
    }
    else
        model.terms = logDensityTerms(covarianceCholeskyView!(TrianglePolicy.lower)(storage));
    return Result(0, nullable(model));
}




version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.math.common: fabs;
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;

    foreach (T; AliasSeq!(float, double, real))
    foreach (triangle; AliasSeq!(TrianglePolicy.lower, TrianglePolicy.upper))
    foreach (label; AliasSeq!(covarianceView, precisionView,
        covarianceCholeskyView, precisionCholeskyView))
    {
        T[2] mean = [1, -2];
        T[4] entries = triangle == TrianglePolicy.upper
            ? [T(4), T(1), T.nan, T(2)] : [T(4), T.nan, T(1), T(2)];
        auto metric = label!triangle(entries[].sliced(2, 2));
        T[4] data = [3, 4, -1, 2];
        auto columns = data[].sliced(2, 2);
        auto expected = multivariateNormalLPDFBatch(columns, mean[], metric);
        auto densities = multivariateNormalPDFBatch(columns, mean[], metric);
        const model = tryMultivariateNormal(mean[], metric).distribution.get;
        entries[] = T.nan;
        mean[] = T.nan;
        auto logs = model.lpdfBatch(columns);
        auto probs = model.pdfBatch(columns);
        foreach (j; 0 .. columns.length!1)
        {
            auto point = columns[0 .. 2, j];
            assert(fabs(logs[j] - expected[j]) <= 64 * T.epsilon * fabs(expected[j]));
            assert(fabs(model.lpdf(point) - logs[j]) <= 64 * T.epsilon * fabs(logs[j]));
            assert(fabs(probs[j] - densities[j]) <= 128 * T.epsilon * densities[j]);
        }
    }

    double[1] mu = [1], factor = [2], tmp;
    double[2] points = [1, 3], output, scratch;
    const d = multivariateNormal(mu[], covarianceCholeskyView(factor[].sliced(1, 1)));
    void check() @safe pure nothrow @nogc
    {
        auto columns = points[].sliced(1, 2);
        d.pdfBatchWithWorkspace(columns, output[], scratch[].sliced(1, 2));
        assert(output[1] == d.pdfWithWorkspace(points[1 .. 2], tmp[]));
        d.lpdfBatchWithWorkspace(columns, output[], scratch[].sliced(1, 2));
        assert(output[0] == d.lpdfWithWorkspace(points[0 .. 1], tmp[]));
    }
    check();

    double[] empty;
    MultivariateNormal!double zero;
    assert(zero.lpdf(empty) == 0 && zero.pdf(empty) == 1);
    auto emptyModel = multivariateNormal(empty, covarianceView(empty.sliced(0, 0)));
    assert(emptyModel.pdfBatch(empty.sliced(0, 2)) == [1.0, 1.0]);
    double[2] origin = [0, 0];
    double[4] singular = [1, 1, 1, 1];
    assertThrown!Exception(multivariateNormal(origin[], covarianceView(singular[].sliced(2, 2))));
}

/++
Evaluate the multivariate normal log density using GC-managed scratch.
For covariance S and Mahalanobis distance d, the result is
$(D -0.5 * (d * d + k * log(2 * PI) + log(det(S)))).
A precision matrix contributes its log determinant with the opposite sign.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X multivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluate!(Allocation.gc)(allocator, x.asView, mean.asView, metric);
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
    assert(fabs(multivariateNormalLPDF(x[], mean[], covarianceView(cov[].sliced(2, 2))) - expected) < 1e-14);
    assert(fabs(rcMultivariateNormalLPDF(x[], mean[], covarianceCholeskyView(cf[].sliced(2, 2))) - expected) < 1e-14);
    assert(fabs(rcMultivariateNormalLPDF(x[], mean[], precisionView(prec[].sliced(2, 2))) - expected) < 1e-14);
    assert(fabs(multivariateNormalLPDF(x[], mean[], precisionCholeskyView(pf[].sliced(2, 2))) - expected) < 1e-14);
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
    const result = multivariateNormalLPDF(x[], mean[], precisionCholeskyView!(TrianglePolicy.lower)(lower[].sliced(2, 2)));
    assert(fabs(result - (log(6.0) - 2 * double(LOGSQRT2PI) - 26)) < 1e-14);
}

/++
Evaluate selected observation vectors with GC-managed result and scratch storage.
The selection forwards to $(LREF multivariateNormalLPDFBatch) in one batch.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A GC-managed vector of n natural log densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
multivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        multivariateNormalLPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return multivariateNormalLPDFBatch(columns, mean, metric);
}

/// Each row contains the inlet and chamber readings for one test run.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: PI, fabs, log;
    double[4] readings = [10, 20, 12, 21], covariance = [4, 2, 2, 5];
    double[2] mean = [10, 20];
    auto rows = readings[].sliced(2, 2);
    auto values = multivariateNormalLPDF(rows.byDim!0, mean[], covarianceView(covariance[].sliced(2, 2)));
    const atMean = -log(8 * double(PI));
    assert(values.length == 2);
    assert(fabs(values[0] - atMean) < 1e-14);
    assert(fabs(values[1] - (atMean - 0.5)) < 1e-14);
}

// Diagnose a missing label only for otherwise supported arguments.
auto multivariateNormalLPDF(X, C, M)(scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalLPDF(x, mean, covarianceView(metric))))
{
    static assert(0, "Specify the matrix representation. For covariance, pass covarianceView(metric) " ~
        "or call multivariateNormalLPDFFromCovariance(x, mean, metric).");
}

/++
Evaluate the multivariate normal log density from a covariance matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k covariance matrix or matching view from $(D covarianceView).
        The selected triangle must be finite and define a positive-definite matrix.

Returns:
    A natural log density in the input element type, or a GC-managed vector of n
    log densities in selection order. Zero-dimensional observations have log density zero.

Throws:
    Exception if Cholesky factorization fails, including for an empty batch.
+/
auto multivariateNormalLPDFFromCovariance(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalLPDF(x, mean, covarianceView!triangle(metric))))
{
    return multivariateNormalLPDF(x, mean, covarianceView!triangle(metric));
}

/// ditto
auto multivariateNormalLPDFFromCovariance(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalLPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.covariance,
        "multivariateNormalLPDFFromCovariance expects a covariance matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalLPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [4];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalLPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalLPDFFromCovariance(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalLPDFFromCovariance(x[], mean[], covarianceView(matrix)) - expected) < 1e-14);
}

/++
Evaluate the multivariate normal log density from a precision matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k precision matrix or matching view from $(D precisionView).
        The selected triangle must be finite and define a positive-definite matrix.

Returns:
    A natural log density in the input element type, or a GC-managed vector of n
    log densities in selection order. Zero-dimensional observations have log density zero.

Throws:
    Exception if Cholesky factorization fails, including for an empty batch.
+/
auto multivariateNormalLPDFFromPrecision(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalLPDF(x, mean, precisionView!triangle(metric))))
{
    return multivariateNormalLPDF(x, mean, precisionView!triangle(metric));
}

/// ditto
auto multivariateNormalLPDFFromPrecision(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalLPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovariance,
        "multivariateNormalLPDFFromPrecision expects a precision matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalLPDF(x, mean, metric);
}

/// Select two rows and read the lower triangle of their shared precision matrix.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: fabs, log, PI;

    double[4] readings = [0, 0, 1, 1], lower = [4, double.nan, 2, 2];
    double[2] mean = [0, 0];
    auto rows = readings[].sliced(2, 2);
    auto matrix = lower[].sliced(2, 2);
    const metric = precisionView!(TrianglePolicy.lower)(matrix);
    auto values = multivariateNormalLPDFFromPrecision(rows.byDim!0, mean[], metric);
    auto fromRaw = multivariateNormalLPDFFromPrecision!(TrianglePolicy.lower)(rows.byDim!0, mean[], matrix);
    const atMean = -log(double(PI));
    assert(values == fromRaw && values.length == 2);
    assert(fabs(values[0] - atMean) < 1e-14 && fabs(values[1] - (atMean - 5)) < 1e-14);
}

/++
Evaluate the multivariate normal log density from a Cholesky factor of the covariance matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k Cholesky factor of the covariance matrix or matching view from $(D covarianceCholeskyView).
        The selected triangle must be finite and the factor must have positive diagonal entries.

Returns:
    A natural log density in the input element type, or a GC-managed vector of n
    log densities in selection order. Zero-dimensional observations have log density zero.
+/
auto multivariateNormalLPDFFromCovarianceCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalLPDF(x, mean, covarianceCholeskyView!triangle(metric))))
{
    return multivariateNormalLPDF(x, mean, covarianceCholeskyView!triangle(metric));
}

/// ditto
auto multivariateNormalLPDFFromCovarianceCholesky(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalLPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.covarianceCholesky,
        "multivariateNormalLPDFFromCovarianceCholesky expects a Cholesky factor of the covariance matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalLPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [2];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalLPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalLPDFFromCovarianceCholesky(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalLPDFFromCovarianceCholesky(x[], mean[], covarianceCholeskyView(matrix)) - expected) < 1e-14);
}

/++
Evaluate the multivariate normal log density from a Cholesky factor of the precision matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k Cholesky factor of the precision matrix or matching view from $(D precisionCholeskyView).
        The selected triangle must be finite and the factor must have positive diagonal entries.

Returns:
    A natural log density in the input element type, or a GC-managed vector of n
    log densities in selection order. Zero-dimensional observations have log density zero.
+/
auto multivariateNormalLPDFFromPrecisionCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalLPDF(x, mean, precisionCholeskyView!triangle(metric))))
{
    return multivariateNormalLPDF(x, mean, precisionCholeskyView!triangle(metric));
}

/// ditto
auto multivariateNormalLPDFFromPrecisionCholesky(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalLPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovarianceCholesky,
        "multivariateNormalLPDFFromPrecisionCholesky expects a Cholesky factor of the precision matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalLPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [0.5];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalLPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalLPDFFromPrecisionCholesky(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalLPDFFromPrecisionCholesky(x[], mean[], precisionCholeskyView(matrix)) - expected) < 1e-14);
}

version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[1] x, mean, entry;
    auto matrix = entry[].sliced(1, 1);
    const metric = precisionView!(TrianglePolicy.lower)(matrix);
    static assert(!__traits(compiles, multivariateNormalLPDF(x[], mean[], matrix)));
    static assert(!__traits(compiles, multivariateNormalLPDF(matrix.byDim!0, mean[], matrix)));
    static assert(!__traits(compiles, multivariateNormalLPDFFromCovariance(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalLPDFFromPrecision(x[], mean[], covarianceView(matrix))));
    static assert(!__traits(compiles, multivariateNormalLPDFFromPrecisionCholesky(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalLPDFFromCovarianceCholesky(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalLPDFFromPrecision!(TrianglePolicy.upper)(x[], mean[], metric)));
}

/++
Evaluate the multivariate normal log density using reference-counted scratch.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X rcMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluate!(Allocation.rc)(allocator, x.asView, mean.asView, metric);
}

/// Evaluate a covariance factor using reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;

    double[1] x = [3], mean = [1], factor = [2];
    const result = rcMultivariateNormalLPDF(
        x[], mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    assert(fabs(result - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);
}

/++
Evaluate selected observation vectors with reference-counted result and scratch storage.
The selection forwards to $(LREF rcMultivariateNormalLPDFBatch) in one batch.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A reference-counted vector of n natural log densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        rcMultivariateNormalLPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return rcMultivariateNormalLPDFBatch(columns, mean, metric);
}

/// Use reference-counted storage for the selected rows.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto rows = readings[].sliced(2, 1);
    auto values = rcMultivariateNormalLPDF(
        rows.byDim!0, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate the multivariate normal log density using a caller-selected allocator.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.
    allocator = Allocator for scratch storage, which is released on return or exception.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X makeMultivariateNormalLPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    return evaluate!(Allocation.custom)(allocator, x.asView, mean.asView, metric);
}

/// Custom allocation also supports raw precision input and empty observations.
version(mir_stat_test_multivariate_normal)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    float[1] x = [3], mean = [1], precision = [0.25f];
    const value = makeMultivariateNormalLPDF(
        Mallocator.instance, x[], mean[], precisionView(precision[].sliced(1, 1)));
    static assert(is(typeof(value) == const(float)));
    assert(fabs(value - normalLPDF(3f, 1f, 2f)) < 4 * float.epsilon);
    assert(makeMultivariateNormalLPDF(Mallocator.instance, x[0 .. 0], mean[0 .. 0],
        covarianceView(precision[0 .. 0].sliced(0, 0))) == 0);
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
    const result = makeMultivariateNormalLPDF(
        Mallocator.instance, x[], mean[], precisionCholeskyView(factor[].sliced(1, 1)));
    assert(fabs(result - normalLPDF(3f, 1f, 2f)) < 4 * float.epsilon);
}

/++
Evaluate selected observation vectors with custom-allocated result and scratch storage.
The selection forwards to $(LREF makeMultivariateNormalLPDFBatch) in one batch.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A custom-allocated vector of n natural log densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero. Release the returned
    vector's field with the supplied allocator.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMultivariateNormalLPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        makeMultivariateNormalLPDFBatch(allocator, ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return makeMultivariateNormalLPDFBatch(allocator, columns, mean, metric);
}

/// Dispose of the custom result after evaluating the selected rows.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalLPDF;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto rows = readings[].sliced(2, 1);
    auto values = makeMultivariateNormalLPDF(
        Mallocator.instance, rows.byDim!0, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) Mallocator.instance.dispose(values.field);
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/// Allocate one work vector and, for a raw metric, one factor matrix.
private auto evaluate(Allocation policy, bool reportFailure = false, A, X, C, M)(
    ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
{
    import std.experimental.allocator: dispose;
    assert(x.length == mean.length && metric.storage.length!0 == x.length && metric.storage.length!1 == x.length);
    auto work = allocate!(policy, Element!X)(allocator, [x.length]);
    scope(exit) static if (policy == Allocation.custom) allocator.dispose(work.field);
    static if (prepared!(M.form))
        return multivariateNormalLPDFWithWorkspace(x, mean, metric, work);
    else
    {
        auto factor = allocate!(policy, Element!X)(allocator, metric.storage.shape);
        scope(exit) static if (policy == Allocation.custom) allocator.dispose(factor.field);
        static if (reportFailure)
            return tryMultivariateNormalLPDFWithWorkspace(x, mean, metric, work, factor);
        else
            return multivariateNormalLPDFWithWorkspace(x, mean, metric, work, factor);
    }
}

/// Factor-dependent terms shared by all observations in a batch.
private struct LogDensityTerms(T)
{
    T dimensionTerm = 0;
    T halfCovarianceLogDet = 0;

    T fromDistance(T distance) const @safe pure nothrow @nogc
    {
        // Halve before squaring so the negative half-square can remain finite.
        const base = (-T(0.5) * distance) * distance - dimensionTerm;
        return base - halfCovarianceLogDet;
    }
}

private auto logDensityTerms(M)(scope auto ref M factor)
    if (isCovarianceMetric!M && prepared!(M.form))
{
    import mir.stat.constant: LOGSQRT2PI;
    auto storage = factor.storage.asView;
    alias T = Element!(typeof(storage));
    const halfLogDet = T(0.5) * logDeterminantFromCholesky(storage);
    enum precision = M.form == CovarianceRepresentation.inverseCovarianceCholesky;
    return LogDensityTerms!T(T(storage.length!0) * T(LOGSQRT2PI), precision ? -halfLogDet : halfLogDet);
}

/++
Evaluate the log density from a prepared factor using caller-owned scratch.
The elements of work must be disjoint from each other and from all inputs.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    factor = Labeled dense k-by-k Cholesky factor of the covariance or precision matrix,
        with the same element type as x. The selected triangle must be finite
        and its diagonal entries positive.
    work = Writable vector of k elements of the same type as x for the centered
        and transformed observation.

Returns:
    The natural logarithm of the density at x, in the element type of x.
    A zero-dimensional observation has log density zero.
+/
Element!X multivariateNormalLPDFWithWorkspace(X, C, M, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor, scope auto ref W work)
    if (isCovarianceMetric!M && prepared!(M.form) && compatible!(X, C, typeof(factor.storage)) &&
        vector!W && is(Element!X == Element!W))
{
    auto observation = x.asView;
    const distance = mahalanobisWithWorkspace(observation, mean, factor, work);
    return logDensityTerms(factor).fromDistance(distance);
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
    auto factor = covarianceCholeskyView!(TrianglePolicy.lower)(storage[].sliced(2, 2));
    const atMean = -2 * double(LOGSQRT2PI) - log(6.0);

    // The second observation is mean + L * [1, 2], so its squared distance is 5.
    double[2] x = [3, 5];
    const first = multivariateNormalLPDFWithWorkspace(
        mean[], mean[], factor, scratch[]);
    const second = multivariateNormalLPDFWithWorkspace(
        x[], mean[], factor, scratch[]);
    assert(fabs(first - atMean) < 1e-14);
    assert(fabs(second - (atMean - 2.5)) < 1e-14);
}

/++
Factor a raw covariance or precision matrix and evaluate the log density.
Both scratch buffers must have nonoverlapping elements and be disjoint from
one another and from all inputs.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense, positive-definite k-by-k matrix of the same element type as x
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
Element!X multivariateNormalLPDFWithWorkspace(X, C, M, W, F)(scope auto ref X x, scope auto ref C mean,
    scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && matrix!F && is(Element!X == Element!W) && is(Element!X == Element!F))
{
    import mir.exception: enforce;
    const result = tryMultivariateNormalLPDFWithWorkspace(x, mean, metric, work, factorWork);
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
        x[], mean[], covarianceView(covariance[].sliced(1, 1)), work[], factorWork);
    assert(fabs(result - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);
}

// Halving the distance before squaring preserves a finite log-density in the far tail.
version(mir_stat_test_multivariate_normal)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [0x1p512], mean = [0], diagonal = [1], work;
    auto factor = diagonal[].sliced(1, 1);
    assert(multivariateNormalLPDFWithWorkspace(x[], mean[], covarianceCholeskyView(factor), work[]) == -0x1p1023);
}

/++
Evaluate selected rows or columns using a prepared Cholesky factor.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    factor = Labeled dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n log densities in selection order.
    work = Writable selection of n scratch vectors of length k.

Returns:
    void. Writes one log density per selected observation.
+/
void multivariateNormalLPDFWithWorkspace(X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (isCovarianceMetric!M && isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        multivariateNormalLPDFBatchWithWorkspace(
            ObservationColumns!X.init, mean, factor, result, ObservationColumns!W.init)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    multivariateNormalLPDFBatchWithWorkspace(columns, mean, factor, result, scratch);
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
    multivariateNormalLPDFWithWorkspace(
        rows.byDim!0, mean[], precisionCholeskyView(factor[].sliced(1, 1)), values[], workRows.byDim!0);
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
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x and finite
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
MultivariateNormalLPDFResult!(Element!X) tryMultivariateNormalLPDFWithWorkspace(X, C, M, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && matrix!F && is(Element!X == Element!W) && is(Element!X == Element!F))
{
    alias Result = MultivariateNormalLPDFResult!(Element!X);
    auto observation = x.asView;
    auto center = mean.asView;
    auto matrixView = metric.storage.asView;
    auto workView = work.asView;
    auto factor = factorWork.asView;
    assert(observation.length == center.length && observation.length == workView.length);
    assert(matrixView.length!0 == observation.length && matrixView.length!1 == observation.length);
    auto decomposition = tryCholeskyDecomposeInto(metric, factor);
    if (decomposition.failedPivot) return Result(Element!X.nan, decomposition.failedPivot);
    return Result(multivariateNormalLPDFWithWorkspace(
        observation, center, decomposition.factor.get, workView), 0);
}

/// A one-dimensional precision of one quarter gives a standard deviation of two.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    float[1] x = [3], mean = [1], precision = [0.25f], work, factor;
    const result = tryMultivariateNormalLPDFWithWorkspace(
        x[], mean[], precisionView(precision[].sliced(1, 1)), work[], factor[].sliced(1, 1));
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
        const result = tryMultivariateNormalLPDFWithWorkspace(
            x[], mean[], labelCovariance!(form, TrianglePolicy.lower)(metric), work[], factor[].sliced(2, 2));
        assert(result.failedPivot == 0);
        assert(fabs(result.logDensity - expected) < 1e-14);
    }}
    data[3] = 1;
    const singular = tryMultivariateNormalLPDFWithWorkspace(
        x[], mean[], covarianceView!(TrianglePolicy.lower)(metric), work[], factor[].sliced(2, 2));
    assert(singular.failedPivot == 2 && isNaN(singular.logDensity));
    assertThrown!Exception(multivariateNormalLPDF(
        x[], mean[], covarianceView!(TrianglePolicy.lower)(metric)));
    const empty = tryMultivariateNormalLPDFWithWorkspace(x[0 .. 0], mean[0 .. 0],
        covarianceView(data[0 .. 0].sliced(0, 0)), work[0 .. 0], factor[0 .. 0].sliced(0, 0));
    assert(empty.failedPivot == 0 && empty.logDensity == 0);
    x[0] = double.max;
    const tail = tryMultivariateNormalLPDFWithWorkspace(x[0 .. 1], mean[0 .. 1],
        covarianceView(data[0 .. 1].sliced(1, 1)), work[0 .. 1], factor[0 .. 1].sliced(1, 1));
    assert(tail.failedPivot == 0 && tail.logDensity == -double.infinity);
}

/++
Evaluate selected rows or columns using one shared covariance or precision matrix.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    metric = Raw covariance or precision view of a dense k-by-k matrix with finite selected triangle.
    result = Writable vector of n log densities in selection order.
    work = Writable selection of n scratch vectors of length k.
    factorWork = Writable k-by-k factor workspace. May contain a partial factor on failure.

Returns:
    Zero on success or the one-based failed pivot. Failure leaves result untouched.
+/
size_t tryMultivariateNormalLPDFWithWorkspace(X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        tryMultivariateNormalLPDFBatchWithWorkspace(
            ObservationColumns!X.init, mean, metric, result, ObservationColumns!W.init, factorWork)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    return tryMultivariateNormalLPDFBatchWithWorkspace(columns, mean, metric, result, scratch, factorWork);
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
        covarianceView(covariance[].sliced(2, 2)), logDensities[], workRows.byDim!0, factor[].sliced(2, 2));
    assert(pivot == 0);
    // The first run is at the mean. The covariance determinant is 16.
    const atMean = -log(8 * double(PI));
    assert(fabs(logDensities[0] - atMean) < 1e-14);
    // The second run's [12, 21] centers to [2, 1] and transforms to [1, 0].
    assert(fabs(logDensities[1] - (atMean - 0.5)) < 1e-14);
    // The third run's transformed row is [0, 2].
    assert(fabs(logDensities[2] - (atMean - 2)) < 1e-14);
}

/++
Outcome of batch evaluation with a raw covariance or precision matrix.

Params:
    V = Vector type for the chosen allocation strategy.
+/
struct MultivariateNormalLPDFBatchResult(V)
{
    /// One natural log density per observation on success. Empty on factorization failure.
    V logDensities;
    /// Zero on success, including an empty batch. A positive value is the one-based
    /// Cholesky step at which the selected matrix failed the positive-definiteness check.
    size_t failedPivot;
}

/++
Evaluate one observation per column with GC-managed result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A GC-managed vector of n natural log densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
multivariateNormalLPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluateBatch!(Allocation.gc)(allocator, x.asView, mean.asView, metric);
}

/// Evaluate two columns with a shared raw precision.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], precision = [0.25];
    auto columns = readings[].sliced(1, 2);
    auto values = multivariateNormalLPDFBatch(
        columns, mean[], precisionView(precision[].sliced(1, 1)));
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate one observation per column with reference-counted result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A reference-counted vector of n natural log densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMultivariateNormalLPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluateBatch!(Allocation.rc)(allocator, x.asView, mean.asView, metric);
}

/// Evaluate two observations using an upper Cholesky factor of the precision matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: PI, fabs, log;

    // The columns are [10, 20] (the mean) and [11, 21].
    double[4] readings = [10, 11, 20, 21];
    double[2] mean = [10, 20];
    auto columns = readings[].sliced(2, 2);

    // The precision is P = U.transpose * U, with U = [[2, 1], [0, 1]].
    double[4] upper = [2, 1, 0, 1], precision = [4, 2, 2, 2];
    auto values = rcMultivariateNormalLPDFBatch(
        columns, mean[], precisionCholeskyView(upper[].sliced(2, 2)));
    auto raw = tryRcMultivariateNormalLPDFBatch(
        columns, mean[], precisionView(precision[].sliced(2, 2)));
    assert(raw.failedPivot == 0);
    assert(values.length == 2 && raw.logDensities.length == 2);

    // det(P) = 4. For the second observation, U * [1, 1] = [3, 1],
    // so the log density drops by (3 * 3 + 1 * 1) / 2 = 5 from its value at the mean.
    const atMean = -log(double(PI));
    assert(fabs(values[0] - atMean) < 1e-14);
    assert(fabs(values[1] - (atMean - 5)) < 1e-14);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - raw.logDensities[j]) < 1e-14);
}

/++
Evaluate one observation per column with custom-allocated result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A custom-allocated vector of n natural log densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has log density zero. Release the returned
    vector's field with the supplied allocator.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMultivariateNormalLPDFBatch(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    return evaluateBatch!(Allocation.custom)(allocator, x.asView, mean.asView, metric);
}

/// Allocate the column results with Mallocator.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalLPDF;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto columns = readings[].sliced(1, 2);
    auto values = makeMultivariateNormalLPDFBatch(
        Mallocator.instance, columns, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) Mallocator.instance.dispose(values.field);
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.dynamic: reversed;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim;
    import std.exception: assertThrown;
    import std.math: fabs;
    alias R = CovarianceRepresentation;
    const float[6] data = [10, 20, 12, 21, 10, 24];
    const float[2] mean = [10, 20];
    const float[4] lower = [4, float.nan, 2, 5];
    auto rows = data[].sliced(3, 2).reversed!0;
    auto metric = lower[].sliced(2, 2);
    static foreach (form; [R.covariance, R.inverseCovariance])
    {{
        auto values = rcMultivariateNormalLPDF(rows.alongDim!1, mean[], labelCovariance!(form, TrianglePolicy.lower)(metric));
        static assert(is(typeof(values) == AllocatedSlice!(Allocation.rc, float, 1)));
        foreach (i; 0 .. values.length)
        {
            const expected = multivariateNormalLPDF(rows[i], mean[], labelCovariance!(form, TrianglePolicy.lower)(metric));
            assert(fabs(values[i] - expected) <= 8 * float.epsilon * (1 + fabs(expected)));
        }
    }}
    auto empty = data[0 .. 0];
    assert(multivariateNormalLPDFBatch(
        empty.sliced(2, 0), mean[], covarianceView!(TrianglePolicy.lower)(metric)).length == 0);
    assert(rcMultivariateNormalLPDFBatch(empty.sliced(0, 3), mean[0 .. 0], covarianceView(empty.sliced(0, 0))) == [0, 0, 0]);
    float[4] singular = [1, 1, 1, 1];
    assertThrown!Exception(multivariateNormalLPDFBatch(empty.sliced(2, 0), mean[], covarianceView(singular[].sliced(2, 2))));
}

/++
Evaluate one log density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a GC-managed vector of n natural log densities
    in column order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty.
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.gc, Element!X, 1))
tryMultivariateNormalLPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluateBatch!(Allocation.gc, true)(allocator, x.asView, mean.asView, metric);
}

/// A singular covariance has no batch log densities. The pivot identifies the failed step.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [0];
    auto columns = data[].sliced(1, 2);
    auto result = tryMultivariateNormalLPDFBatch(columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 1);
    assert(result.logDensities.length == 0);
}

/++
Evaluate one log density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a reference-counted vector of n natural log densities
    in column order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty.
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.rc, Element!X, 1))
tryRcMultivariateNormalLPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluateBatch!(Allocation.rc, true)(allocator, x.asView, mean.asView, metric);
}

/// Evaluate two columns and check whether factorization succeeded.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto columns = data[].sliced(1, 2);
    auto result = tryRcMultivariateNormalLPDFBatch(columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    assert(result.logDensities.length == 2);
    assert(result.logDensities[0] > result.logDensities[1]);
}

// A copied RC result keeps its log densities alive after the original leaves scope.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;

    static auto retainedResult()
    {
        double[2] data = [0, 1];
        double[1] mean = [0], covariance = [1];
        auto original = tryRcMultivariateNormalLPDFBatch(data[].sliced(1, 2), mean[], covarianceView(covariance[].sliced(1, 1)));
        auto copy = original;
        assert(copy.logDensities.length == original.logDensities.length);
        return copy;
    }

    auto retained = retainedResult();
    assert(retained.failedPivot == 0 && retained.logDensities.length == 2);
    assert(retained.logDensities[0] > retained.logDensities[1]);
}

/++
Evaluate one log density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a custom-allocated vector of n natural log densities
    in column order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty. Release the successful result with
    allocator.dispose(result.logDensities.field).
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.custom, Element!X, 1))
tryMakeMultivariateNormalLPDFBatch(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    return evaluateBatch!(Allocation.custom, true)(allocator, x.asView, mean.asView, metric);
}

/// Evaluate two columns and check whether factorization succeeded.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto columns = data[].sliced(1, 2);
    auto result = tryMakeMultivariateNormalLPDFBatch(Mallocator.instance, columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    scope(exit) Mallocator.instance.dispose(result.logDensities.field);
    assert(result.logDensities.length == 2);
    assert(result.logDensities[0] > result.logDensities[1]);
}

// On success only the result escapes. A failed factorization or scratch allocation releases it too.
version(mir_stat_test_multivariate_normal)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    struct CountingAllocator
    {
        size_t allocations, releases, failAfter = size_t.max;
        void[] allocate(size_t bytes)
        {
            if (allocations == failAfter) throw new Exception("test allocation failure");
            auto memory = Mallocator.instance.allocate(bytes);
            ++allocations;
            return memory;
        }
        bool deallocate(void[] memory)
        {
            ++releases;
            return Mallocator.instance.deallocate(memory);
        }
    }
    CountingAllocator allocator;
    double[2] data = [10, 14];
    double[1] mean = [10], diagonal = [4];
    auto columns = data[].sliced(1, 2);
    auto metric = diagonal[].sliced(1, 1);
    auto values = makeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric));
    assert(allocator.allocations == 3 && allocator.releases == 2);
    allocator.dispose(values.field);
    auto outcome = tryMakeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric));
    assert(outcome.failedPivot == 0 && outcome.logDensities.length == 2);
    allocator.dispose(outcome.logDensities.field);
    assert(allocator.allocations == allocator.releases);
    diagonal[0] = 0;
    outcome = tryMakeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric));
    assert(outcome.failedPivot == 1 && outcome.logDensities.length == 0);
    assert(allocator.allocations == allocator.releases);
    assertThrown!Exception(makeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric)));
    assert(allocator.allocations == allocator.releases);
    foreach (successfulAllocations; 0 .. 3)
    {
        allocator.failAfter = allocator.allocations + successfulAllocations;
        assertThrown!Exception(tryMakeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric)));
        assert(allocator.allocations == allocator.releases);
    }
    allocator.failAfter = allocator.allocations + 2;
    assertThrown!Exception(makeMultivariateNormalLPDFBatch(allocator, columns, mean[], covarianceView(metric)));
    assert(allocator.allocations == allocator.releases);
}

// A failed pivot must leave the entire region available for the next batch.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.building_blocks.region: BorrowedRegion;
    double[2] data = [10, 14];
    double[1] mean = [10], singular = [0];
    // Two results, two observation scratch elements, and one factor element.
    double[5] storage;
    auto region = BorrowedRegion!(double.alignof)(cast(ubyte[]) storage[]);
    foreach (attempt; 0 .. 2)
    {
        auto result = tryMakeMultivariateNormalLPDFBatch(region, data[].sliced(1, 2),
            mean[], covarianceView(singular[].sliced(1, 1)));
        assert(result.failedPivot == 1 && result.logDensities.length == 0);
        assert(region.available == storage.sizeof);
    }
}

private enum batchInputs(X, C, M) = matrix!X && vector!C && matrix!M &&
    is(Element!X == Element!C) && is(Element!X == Element!M);

/// Allocate result and scratch once per batch, transferring only the result to the caller.
private auto evaluateBatch(Allocation policy, bool reportFailure = false, A, X, C, M)(
    ref A allocator, X columns, C mean, scope auto ref M metric)
{
    import std.experimental.allocator: dispose;
    alias T = Element!X;
    assert(columns.length!0 == mean.length && metric.storage.length!0 == mean.length && metric.storage.length!1 == mean.length);
    auto result = allocate!(policy, T)(allocator, [columns.length!1]);
    static if (!prepared!(M.form)) size_t pivot;
    // Register result cleanup first so both scratch buffers are released before it.
    scope(success) static if (reportFailure && policy == Allocation.custom)
    {
        if (pivot) allocator.dispose(result.field);
    }
    scope(failure) static if (policy == Allocation.custom) allocator.dispose(result.field);
    auto work = allocate!(policy, T)(allocator, columns.shape);
    scope(exit) static if (policy == Allocation.custom) allocator.dispose(work.field);
    static if (prepared!(M.form))
        multivariateNormalLPDFBatchWithWorkspace(columns, mean, metric, result, work);
    else
    {
        import mir.exception: enforce;
        auto factor = allocate!(policy, T)(allocator, metric.storage.shape);
        scope(exit) static if (policy == Allocation.custom) allocator.dispose(factor.field);
        pivot = tryMultivariateNormalLPDFBatchWithWorkspace(
            columns, mean, metric, result, work, factor);
        static if (reportFailure)
        {
            alias Outcome = MultivariateNormalLPDFBatchResult!(typeof(result));
            if (pivot)
                return Outcome(typeof(result).init, pivot);
            return Outcome(result, 0);
        }
        else
            enforce!"Cholesky factorization failed"(pivot == 0);
    }
    static if (!reportFailure)
        return result;
}

private enum batchBuffers(X, C, M, R, W) = batchInputs!(X, C, M) && vector!R && matrix!W &&
    is(Element!X == Element!R) && is(Element!X == Element!W);

/++
Write one log density per observation column using a shared Cholesky factor.
All writable elements must be disjoint from one another and from the inputs.

Params:
    x = Dense k-by-n matrix of finite observations. Column j is observation j.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    factor = Labeled dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n log densities in column order.
    work = Writable k-by-n scratch for centering and transforming the observations.

Returns:
    void. Writes each log density to result. In zero dimensions each value is zero.
+/
void multivariateNormalLPDFBatchWithWorkspace(X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (isCovarianceMetric!M && prepared!(M.form) && batchBuffers!(X, C, typeof(factor.storage), R, W))
{
    import mir.stat.distance: mahalanobisBatchWithWorkspace;
    auto values = result.asView;
    auto columns = x.asView;
    mahalanobisBatchWithWorkspace(columns, mean, factor, values, work);
    const terms = logDensityTerms(factor);
    foreach (j; 0 .. values.length)
        values[j] = terms.fromDistance(values[j]);
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
    multivariateNormalLPDFBatchWithWorkspace(
        x[].sliced(1, 2), mean[], covarianceCholeskyView(factor[].sliced(1, 1)), values[], scratch[].sliced(1, 2));
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalLPDF(x[j], 1.0, 2.0)) < 1e-14);
}

/++
Factor a covariance or precision matrix once, then write one log density per column.
For a prepared factor, use $(LREF multivariateNormalLPDFBatchWithWorkspace) and omit factorWork.
All writable elements must be disjoint from one another and from the inputs.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with finite entries in the selected triangle.
    result = Writable vector of n log densities in column order. A failed factorization
        leaves every entry untouched.
    work = Writable k-by-n scratch for centering and transforming the observations.
    factorWork = Writable k-by-k scratch. Its selected triangle holds the factor on
        success and may contain a partial factor on failure.

Returns:
    Zero on success, or the one-based failed Cholesky pivot. The matrix is checked
    even for an empty batch. In zero dimensions each log density is zero.
+/
size_t tryMultivariateNormalLPDFBatchWithWorkspace(X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) &&
        batchBuffers!(X, C, typeof(metric.storage), R, W) && matrix!F && is(Element!X == Element!F))
{
    auto columns = x.asView;
    auto matrixView = metric.storage.asView;
    auto factor = factorWork.asView;
    assert(columns.length!0 == mean.asView.length && matrixView.length!0 == columns.length!0 &&
        matrixView.length!1 == columns.length!0);
    assert(columns.shape == work.asView.shape && columns.length!1 == result.asView.length);
    auto decomposition = tryCholeskyDecomposeInto(metric, factor);
    if (decomposition.failedPivot) return decomposition.failedPivot;
    multivariateNormalLPDFBatchWithWorkspace(
        columns, mean, decomposition.factor.get, result, work);
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
        covarianceView(metric[].sliced(2, 2)), values[], scratch[].sliced(2, 2), factor[].sliced(2, 2)) == 2);
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
        const pivot = tryMultivariateNormalLPDFWithWorkspace(
            rows.alongDim!1, mean[], labelCovariance!(form, TrianglePolicy.lower)(lower[].sliced(2, 2)), values[],
            workColumns.alongDim!0, factor[].sliced(2, 2));
        assert(pivot == 0);
        enum preparedForm = form == R.covariance ? R.covarianceCholesky : R.inverseCovarianceCholesky;
        foreach (j; 0 .. values.length)
        {
            auto expected = multivariateNormalLPDFWithWorkspace(
                rows[j], mean[], labelCovariance!(preparedForm, TrianglePolicy.lower)(factor[].sliced(2, 2)), vectorWork[]);
            assert(fabs(values[j] - expected) <= 1e-13 * (1 + fabs(expected)));
        }
    }}
    lower[3] = 1;
    values[] = 17;
    const failedPivot = tryMultivariateNormalLPDFWithWorkspace(rows.alongDim!1, mean[],
        precisionView!(TrianglePolicy.lower)(lower[].sliced(2, 2)), values[],
        workColumns.alongDim!0, factor[].sliced(2, 2));
    assert(failedPivot == 2 && values == [17, 17, 17]);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] x = [0x1p512], mean = [0], metric = [1], work, factor, result;
    auto unit = metric[].sliced(1, 1);
    auto f = factor[].sliced(1, 1);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(x[].sliced(1, 1), mean[], covarianceView(unit),
        result[], work[].sliced(1, 1), f) == 0);
    assert(result[0] == -0x1p1023);

    auto noCoordinates = x[0 .. 0].sliced(0, 1);
    auto emptyFactor = factor[0 .. 0].sliced(0, 0);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noCoordinates, mean[0 .. 0],
        covarianceView(metric[0 .. 0].sliced(0, 0)), result[], work[0 .. 0].sliced(0, 1), emptyFactor) == 0);
    assert(result[0] == 0);

    auto noObservations = x[0 .. 0].sliced(1, 0);
    auto emptyWork = work[0 .. 0].sliced(1, 0);
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noObservations, mean[], covarianceView(unit),
        result[0 .. 0], emptyWork, f) == 0);
    unit[0, 0] = 0;
    assert(tryMultivariateNormalLPDFBatchWithWorkspace(noObservations, mean[], covarianceView(unit),
        result[0 .. 0], emptyWork, f) == 1);
}

// Infer the return type so __traits(compiles) also rejects a prepared factor here.
auto tryMultivariateNormalLPDFBatchWithWorkspace(X, C, M, R, W, F...)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && prepared!(M.form) && batchBuffers!(X, C, typeof(factor.storage), R, W) &&
        (F.length == 0 || (F.length == 1 && matrix!(F[0]) && is(Element!X == Element!(F[0])))))
{
    static assert(0, "For a prepared Cholesky factor, use " ~
        "multivariateNormalLPDFBatchWithWorkspace(x, mean, factor, result, work). " ~
        "It writes the log densities and returns void.");
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[1] observations, center, metric, values, scratch, factorScratch;
    auto columns = observations[].sliced(1, 1), matrix = metric[].sliced(1, 1);
    auto work = scratch[].sliced(1, 1), factorWork = factorScratch[].sliced(1, 1);
    alias R = CovarianceRepresentation;
    static foreach (form; [R.covarianceCholesky, R.inverseCovarianceCholesky])
    {{
        auto factor = labelCovariance!(form, TrianglePolicy.upper)(matrix);
        static assert(!__traits(compiles, tryMultivariateNormalLPDFBatchWithWorkspace(
            columns, center[], factor, values[], work)));
        static assert(!__traits(compiles, tryMultivariateNormalLPDFBatchWithWorkspace(
            columns, center[], factor, values[], work, factorWork)));
        static assert(is(typeof(multivariateNormalLPDFBatchWithWorkspace(
            columns, center[], factor, values[], work)) == void));
    }}
}

/++
Evaluate the log density using GC-managed scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow`.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) tryMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluate!(Allocation.gc, true)(allocator, x.asView, mean.asView, metric);
}

/// The first diagonal gives a factor entry of 2. The second pivot is 1 - (2/2) * (2/2) = 0.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    double[2] x = [1, 2], mean = [0, 0];
    double[4] covariance = [4, 2, double.nan, 1];
    const result = tryMultivariateNormalLPDF(x[], mean[], covarianceView(covariance[].sliced(2, 2)));
    assert(result.failedPivot == 2 && isNaN(result.logDensity));
}

/++
Evaluate one log density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a GC-managed vector of n natural log densities
    in selection order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty.
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1))
tryMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryMultivariateNormalLPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryMultivariateNormalLPDFBatch(columns, mean, metric);
}

/// Check the pivot before using the log densities for the selected rows.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto rows = data[].sliced(2, 1);
    auto result = tryMultivariateNormalLPDF(rows.byDim!0, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    assert(result.logDensities.length == 2);
    assert(result.logDensities[0] > result.logDensities[1]);
}

/++
Evaluate the log density using reference-counted scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow @nogc`.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) tryRcMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    int allocator;
    return evaluate!(Allocation.rc, true)(allocator, x.asView, mean.asView, metric);
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
    const result = tryRcMultivariateNormalLPDF(x[], mean[], covarianceView(metric));
    assert(result.failedPivot == 0);
    assert(fabs(result.logDensity - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = tryRcMultivariateNormalLPDF(x[], mean[], covarianceView(metric));
    assert(singular.failedPivot == 1 && isNaN(singular.logDensity));
}

/++
Evaluate one log density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a reference-counted vector of n natural log densities
    in selection order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty.
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1))
tryRcMultivariateNormalLPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryRcMultivariateNormalLPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryRcMultivariateNormalLPDFBatch(columns, mean, metric);
}

/// Evaluate selected rows with raw precision. A precision of 0.25 gives a standard deviation of 2.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalLPDF;
    import std.math: fabs;

    double[2] readings = [10, 14];
    double[1] mean = [10], precision = [0.25];
    auto rows = readings[].sliced(2, 1);
    auto result = tryRcMultivariateNormalLPDF(
        rows.byDim!0, mean[], precisionView(precision[].sliced(1, 1)));
    assert(result.failedPivot == 0 && result.logDensities.length == 2);
    foreach (j; 0 .. readings.length)
        assert(fabs(result.logDensities[j] - normalLPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate the log density using a caller-selected scratch allocator and report Cholesky failure.
The call is `nothrow @nogc` with ordinary floating-point slices and
`Mallocator`. With other allocators or iterators, the inferred attributes depend
on their operations.

Params:
    allocator = Allocator used to obtain and release temporary storage.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFResult). On success, `failedPivot == 0`
    and `logDensity` is the natural log density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `logDensity` is NaN. In zero dimensions both fields are zero.
+/
MultivariateNormalLPDFResult!(Element!X) tryMakeMultivariateNormalLPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    return evaluate!(Allocation.custom, true)(allocator, x.asView, mean.asView, metric);
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
    const result = tryMakeMultivariateNormalLPDF(Mallocator.instance, x[], mean[], precisionView(metric));
    assert(result.failedPivot == 0);
    assert(fabs(result.logDensity - normalLPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = tryMakeMultivariateNormalLPDF(Mallocator.instance, x[], mean[], precisionView(metric));
    assert(singular.failedPivot == 1 && isNaN(singular.logDensity));
}

/++
Evaluate one log density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalLPDFBatchResult). On success, failedPivot is zero
    and logDensities is a custom-allocated vector of n natural log densities
    in selection order. Each zero-dimensional observation has log density zero.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and logDensities is empty. Release the successful result with
    allocator.dispose(result.logDensities.field).
+/
MultivariateNormalLPDFBatchResult!(AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1))
tryMakeMultivariateNormalLPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryMakeMultivariateNormalLPDFBatch(allocator, ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryMakeMultivariateNormalLPDFBatch(allocator, columns, mean, metric);
}

/// Check the pivot before using the log densities for the selected rows.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto rows = data[].sliced(2, 1);
    auto result = tryMakeMultivariateNormalLPDF(Mallocator.instance, rows.byDim!0, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    scope(exit) Mallocator.instance.dispose(result.logDensities.field);
    assert(result.logDensities.length == 2);
    assert(result.logDensities[0] > result.logDensities[1]);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.dynamic: reversed;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim;
    import std.math: fabs;
    alias R = CovarianceRepresentation;
    const float[6] data = [10, 20, 12, 21, 10, 24];
    const float[2] mean = [10, 20];
    float[4] lower = [4, float.nan, 2, 5];
    auto metric = lower[].sliced(2, 2);
    auto rows = data[].sliced(3, 2).reversed!0;
    auto empty = data[0 .. 0];
    static foreach (form; [R.covariance, R.inverseCovariance])
    {{
        auto outcome = tryRcMultivariateNormalLPDF(rows.alongDim!1, mean[], labelCovariance!(form, TrianglePolicy.lower)(metric));
        assert(outcome.failedPivot == 0 && outcome.logDensities.length == 3);
        foreach (j; 0 .. rows.length)
        {
            const scalar = tryRcMultivariateNormalLPDF(rows[j], mean[], labelCovariance!(form, TrianglePolicy.lower)(metric));
            assert(scalar.failedPivot == 0);
            assert(fabs(outcome.logDensities[j] - scalar.logDensity) <= 8 * float.epsilon * (1 + fabs(scalar.logDensity)));
        }
    }}
    auto noRuns = empty.sliced(2, 0);
    auto outcome = tryRcMultivariateNormalLPDFBatch(noRuns, mean[], covarianceView!(TrianglePolicy.lower)(metric));
    assert(outcome.failedPivot == 0 && outcome.logDensities.length == 0);
    metric[1, 1] = 1;
    outcome = tryRcMultivariateNormalLPDFBatch(noRuns, mean[], covarianceView!(TrianglePolicy.lower)(metric));
    assert(outcome.failedPivot == 2 && outcome.logDensities.length == 0);
    outcome = tryRcMultivariateNormalLPDFBatch(empty.sliced(0, 3), empty, covarianceView(empty.sliced(0, 0)));
    assert(outcome.failedPivot == 0 && outcome.logDensities == [0, 0, 0]);
}

/++
Evaluate the multivariate normal density using GC-managed scratch.
The result is computed by exponentiating the log density.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Labeled covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The probability density at x, in the element type of x.
    A zero-dimensional observation has density one.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X multivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    import mir.math.common: exp;
    return exp(multivariateNormalLPDF(x, mean, metric));
}

/// Check a correlated bivariate density at the mean.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs, PI;
    double[2] mean = [1, -2];
    double[4] covariance = [4, 2, double.nan, 5];
    const density = multivariateNormalPDF(mean[], mean[], covarianceView(covariance[].sliced(2, 2)));
    assert(fabs(density - 1 / (8 * double(PI))) < 1e-15);
}

/++
Evaluate selected observation vectors with GC-managed result and scratch storage.
The selection forwards to $(LREF multivariateNormalPDFBatch) in one batch.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A GC-managed vector of n densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1)
multivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        multivariateNormalPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return multivariateNormalPDFBatch(columns, mean, metric);
}

/// Each row contains the inlet and chamber readings for one test run.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: PI, fabs, exp;
    double[4] readings = [10, 20, 12, 21], covariance = [4, 2, 2, 5];
    double[2] mean = [10, 20];
    auto rows = readings[].sliced(2, 2);
    auto values = multivariateNormalPDF(rows.byDim!0, mean[], covarianceView(covariance[].sliced(2, 2)));
    const atMean = 1 / (8 * double(PI));
    assert(values.length == 2);
    assert(fabs(values[0] - atMean) < 1e-14);
    assert(fabs(values[1] - (atMean * exp(-0.5))) < 1e-14);
}

// Diagnose a missing label only for otherwise supported arguments.
auto multivariateNormalPDF(X, C, M)(scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDF(x, mean, covarianceView(metric))))
{
    static assert(0, "Specify the matrix representation. For covariance, pass covarianceView(metric) " ~
        "or call multivariateNormalPDFFromCovariance(x, mean, metric).");
}

/++
Evaluate the multivariate normal density from a covariance matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k covariance matrix or matching view from $(D covarianceView).
        The selected triangle must be finite and define a positive-definite matrix.

Returns:
    A density in the input element type, or a GC-managed vector of n
    densities in selection order. Zero-dimensional observations have density one.

Throws:
    Exception if Cholesky factorization fails, including for an empty batch.
+/
auto multivariateNormalPDFFromCovariance(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDF(x, mean, covarianceView!triangle(metric))))
{
    return multivariateNormalPDF(x, mean, covarianceView!triangle(metric));
}

/// ditto
auto multivariateNormalPDFFromCovariance(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.covariance,
        "multivariateNormalPDFFromCovariance expects a covariance matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [4];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalPDFFromCovariance(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalPDFFromCovariance(x[], mean[], covarianceView(matrix)) - expected) < 1e-14);
}

/++
Evaluate the multivariate normal density from a precision matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k precision matrix or matching view from $(D precisionView).
        The selected triangle must be finite and define a positive-definite matrix.

Returns:
    A density in the input element type, or a GC-managed vector of n
    densities in selection order. Zero-dimensional observations have density one.

Throws:
    Exception if Cholesky factorization fails, including for an empty batch.
+/
auto multivariateNormalPDFFromPrecision(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDF(x, mean, precisionView!triangle(metric))))
{
    return multivariateNormalPDF(x, mean, precisionView!triangle(metric));
}

/// ditto
auto multivariateNormalPDFFromPrecision(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovariance,
        "multivariateNormalPDFFromPrecision expects a precision matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalPDF(x, mean, metric);
}

/// Select two rows and read the lower triangle of their shared precision matrix.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: fabs, exp, PI;

    double[4] readings = [0, 0, 1, 1], lower = [4, double.nan, 2, 2];
    double[2] mean = [0, 0];
    auto rows = readings[].sliced(2, 2);
    auto matrix = lower[].sliced(2, 2);
    const metric = precisionView!(TrianglePolicy.lower)(matrix);
    auto values = multivariateNormalPDFFromPrecision(rows.byDim!0, mean[], metric);
    auto fromRaw = multivariateNormalPDFFromPrecision!(TrianglePolicy.lower)(rows.byDim!0, mean[], matrix);
    const atMean = 1 / double(PI);
    assert(values == fromRaw && values.length == 2);
    assert(fabs(values[0] - atMean) < 1e-14 && fabs(values[1] - (atMean * exp(-5.0))) < 1e-14);
}

/++
Evaluate the multivariate normal density from a Cholesky factor of the covariance matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k Cholesky factor of the covariance matrix or matching view from $(D covarianceCholeskyView).
        The selected triangle must be finite and the factor must have positive diagonal entries.

Returns:
    A density in the input element type, or a GC-managed vector of n
    densities in selection order. Zero-dimensional observations have density one.
+/
auto multivariateNormalPDFFromCovarianceCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDF(x, mean, covarianceCholeskyView!triangle(metric))))
{
    return multivariateNormalPDF(x, mean, covarianceCholeskyView!triangle(metric));
}

/// ditto
auto multivariateNormalPDFFromCovarianceCholesky(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.covarianceCholesky,
        "multivariateNormalPDFFromCovarianceCholesky expects a Cholesky factor of the covariance matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [2];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalPDFFromCovarianceCholesky(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalPDFFromCovarianceCholesky(x[], mean[], covarianceCholeskyView(matrix)) - expected) < 1e-14);
}

/++
Evaluate the multivariate normal density from a Cholesky factor of the precision matrix.

Params:
    triangle = Triangle of the supplied matrix to read, upper by default.
        A labeled input carries its own triangle.
    x = One vector of k finite coordinates or a rectangular selection of n such vectors.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Dense k-by-k Cholesky factor of the precision matrix or matching view from $(D precisionCholeskyView).
        The selected triangle must be finite and the factor must have positive diagonal entries.

Returns:
    A density in the input element type, or a GC-managed vector of n
    densities in selection order. Zero-dimensional observations have density one.
+/
auto multivariateNormalPDFFromPrecisionCholesky(TrianglePolicy triangle = TrianglePolicy.upper, X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDF(x, mean, precisionCholeskyView!triangle(metric))))
{
    return multivariateNormalPDF(x, mean, precisionCholeskyView!triangle(metric));
}

/// ditto
auto multivariateNormalPDFFromPrecisionCholesky(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && __traits(compiles, multivariateNormalPDF(x, mean, metric)))
{
    static assert(M.form == CovarianceRepresentation.inverseCovarianceCholesky,
        "multivariateNormalPDFFromPrecisionCholesky expects a Cholesky factor of the precision matrix. Received " ~ M.form.stringof ~ ".");
    return multivariateNormalPDF(x, mean, metric);
}

///
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[1] x = [3], mean = [1], entries = [0.5];
    auto matrix = entries[].sliced(1, 1);
    const expected = normalPDF(3.0, 1.0, 2.0);
    assert(fabs(multivariateNormalPDFFromPrecisionCholesky(x[], mean[], matrix) - expected) < 1e-14);
    assert(fabs(multivariateNormalPDFFromPrecisionCholesky(x[], mean[], precisionCholeskyView(matrix)) - expected) < 1e-14);
}

version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[1] x, mean, entry;
    auto matrix = entry[].sliced(1, 1);
    const metric = precisionView!(TrianglePolicy.lower)(matrix);
    static assert(!__traits(compiles, multivariateNormalPDF(x[], mean[], matrix)));
    static assert(!__traits(compiles, multivariateNormalPDF(matrix.byDim!0, mean[], matrix)));
    static assert(!__traits(compiles, multivariateNormalPDFFromCovariance(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalPDFFromPrecision(x[], mean[], covarianceView(matrix))));
    static assert(!__traits(compiles, multivariateNormalPDFFromPrecisionCholesky(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalPDFFromCovarianceCholesky(x[], mean[], metric)));
    static assert(!__traits(compiles, multivariateNormalPDFFromPrecision!(TrianglePolicy.upper)(x[], mean[], metric)));
}

/++
Evaluate the multivariate normal density using reference-counted scratch.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Labeled covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.

Returns:
    The probability density at x, in the element type of x.
    A zero-dimensional observation has density one.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X rcMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    import mir.math.common: exp;
    return exp(rcMultivariateNormalLPDF(x, mean, metric));
}

/// Use a prepared precision factor with reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;
    import mir.stat.distribution.normal: normalPDF;
    double[1] x = [3], mean = [1], factor = [0.5];
    const density = rcMultivariateNormalPDF(
        x[], mean[], precisionCholeskyView(factor[].sliced(1, 1)));
    assert(fabs(density - normalPDF(3.0, 1.0, 2.0)) < double.epsilon);
}

/++
Evaluate selected observation vectors with reference-counted result and scratch storage.
The selection forwards to $(LREF rcMultivariateNormalPDFBatch) in one batch.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A reference-counted vector of n densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1)
rcMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        rcMultivariateNormalPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return rcMultivariateNormalPDFBatch(columns, mean, metric);
}

/// Use reference-counted storage for the selected rows.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto rows = readings[].sliced(2, 1);
    auto values = rcMultivariateNormalPDF(
        rows.byDim!0, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate the multivariate normal density using a caller-selected allocator.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Labeled covariance, precision, or Cholesky factor view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must be finite. Raw matrices must be positive definite,
        and supplied factors must have positive diagonal entries.
    allocator = Allocator for scratch storage, which is released on return or exception.

Returns:
    The probability density at x, in the element type of x.
    A zero-dimensional observation has density one.

Throws:
    Exception if Cholesky factorization of a raw covariance or precision matrix fails.
+/
Element!X makeMultivariateNormalPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && compatible!(X, C, typeof(metric.storage)))
{
    import mir.math.common: exp;
    return exp(makeMultivariateNormalLPDF(allocator, x, mean, metric));
}

/// Use a custom allocator for a prepared covariance factor.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;
    import std.experimental.allocator.mallocator: Mallocator;
    import mir.stat.distribution.normal: normalPDF;
    float[1] x = [2], mean = [0], factor = [2];
    const density = makeMultivariateNormalPDF(
        Mallocator.instance, x[], mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    static assert(is(typeof(density) == const(float)));
    assert(fabs(density - normalPDF(2f, 0f, 2f)) < float.epsilon);
}

/++
Evaluate selected observation vectors with custom-allocated result and scratch storage.
The selection forwards to $(LREF makeMultivariateNormalPDFBatch) in one batch.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A custom-allocated vector of n densities in selection order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one. Release the returned
    vector's field with the supplied allocator.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1)
makeMultivariateNormalPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        makeMultivariateNormalPDFBatch(allocator, ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return makeMultivariateNormalPDFBatch(allocator, columns, mean, metric);
}

/// Dispose of the custom result after evaluating the selected rows.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalPDF;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto rows = readings[].sliced(2, 1);
    auto values = makeMultivariateNormalPDF(
        Mallocator.instance, rows.byDim!0, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) Mallocator.instance.dispose(values.field);
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate the density from a prepared factor using caller-owned scratch.
The elements of work must be disjoint from each other and from all inputs.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    factor = Labeled dense k-by-k Cholesky factor of the covariance or precision matrix,
        with the same element type as x. The selected triangle must be finite
        and its diagonal entries positive.
    work = Writable vector of k elements of the same type as x for the centered
        and transformed observation.

Returns:
    The probability density at x, in the element type of x.
    A zero-dimensional observation has density one.
+/
Element!X multivariateNormalPDFWithWorkspace(X, C, M, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor, scope auto ref W work)
    if (isCovarianceMetric!M && prepared!(M.form) && compatible!(X, C, typeof(factor.storage)) &&
        vector!W && is(Element!X == Element!W))
{
    import mir.math.common: exp;
    return exp(multivariateNormalLPDFWithWorkspace(x, mean, factor, work));
}

/// Supply scratch for a lower covariance factor.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;
    import mir.stat.distribution.normal: normalPDF;
    double[2] x = [2, 7], mean = [0, 0], work;
    double[4] factor = [2, double.nan, 1, 3];
    const density = multivariateNormalPDFWithWorkspace(x[], mean[],
        covarianceCholeskyView!(TrianglePolicy.lower)(factor[].sliced(2, 2)), work[]);
    assert(fabs(density - normalPDF(1.0) * normalPDF(2.0) / 6) < 1e-15);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] x = [0, 0], mean = [0, 0], work;
    double[4] data = [1e-200, 0, double.nan, 1e-200];
    assert(multivariateNormalPDFWithWorkspace(x[], mean[],
        covarianceCholeskyView(data[].sliced(2, 2)), work[]) == double.infinity);
    x[0] = 0x1p512;
    data[0] = 1;
    assert(multivariateNormalPDFWithWorkspace(x[0 .. 1], mean[0 .. 1],
        covarianceCholeskyView(data[0 .. 1].sliced(1, 1)), work[0 .. 1]) == 0);
    assert(multivariateNormalPDFWithWorkspace(x[0 .. 0], mean[0 .. 0],
        covarianceCholeskyView(data[0 .. 0].sliced(0, 0)), work[0 .. 0]) == 1);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;

    // A finite density even though the normalization alone exceeds double.max.
    enum scale = 0x1p-550;
    double[2] x = [39 * scale, 0], mean = [0, 0], work;
    double[4] factor = [scale, 0, double.nan, scale];
    const density = multivariateNormalPDFWithWorkspace(x[], mean[],
        covarianceCholeskyView(factor[].sliced(2, 2)), work[]);
    // 2^1100 * exp(-39^2 / 2) / (2 * pi), evaluated at 70 decimal digits.
    enum expected = 1.1320402656725539;
    assert(fabs(density / expected - 1) < 2e-13);
}

/++
Factor a raw covariance or precision matrix and evaluate the density.
Both scratch buffers must have nonoverlapping elements and be disjoint from
one another and from all inputs.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Labeled covariance or precision view of a dense, positive-definite k-by-k matrix of the same element type as x
        with finite entries in the selected triangle.
    work = Writable vector of k elements of the same type as x for the centered
        and transformed observation.
    factorWork = Writable matrix of the same shape and element type as metric.
        On success its selected triangle holds the Cholesky factor.

Returns:
    The probability density at x, in the element type of x.
    A zero-dimensional observation has density one.

Throws:
    Exception if Cholesky factorization fails. In that case factorWork may
    contain a partial factorization.
+/
Element!X multivariateNormalPDFWithWorkspace(X, C, M, W, F)(scope auto ref X x, scope auto ref C mean,
    scope auto ref M metric, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && matrix!F && is(Element!X == Element!W) && is(Element!X == Element!F))
{
    import mir.math.common: exp;
    return exp(multivariateNormalLPDFWithWorkspace(x, mean, metric, work, factorWork));
}

/// Evaluate a raw precision matrix using caller-owned scratch.
version(mir_stat_test_multivariate_normal)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;
    import mir.stat.distribution.normal: normalPDF;
    double[1] x = [3], mean = [1], precision = [0.25], work, factor;
    const density = multivariateNormalPDFWithWorkspace(
        x[], mean[], precisionView(precision[].sliced(1, 1)), work[], factor[].sliced(1, 1));
    assert(fabs(density - normalPDF(3.0, 1.0, 2.0)) < double.epsilon);
}

/++
Evaluate selected rows or columns using a prepared Cholesky factor.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    factor = Labeled dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n densities in selection order.
    work = Writable selection of n scratch vectors of length k.

Returns:
    void. Writes one density per selected observation.
+/
void multivariateNormalPDFWithWorkspace(X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (isCovarianceMetric!M && isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        multivariateNormalPDFBatchWithWorkspace(
            ObservationColumns!X.init, mean, factor, result, ObservationColumns!W.init)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    multivariateNormalPDFBatchWithWorkspace(columns, mean, factor, result, scratch);
}

/// Select two row observations and use one precision factor for both.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[2] data = [1, 3], values, scratch;
    double[1] mean = [1], factor = [0.5];
    auto rows = data[].sliced(2, 1);
    auto workRows = scratch[].sliced(2, 1);
    multivariateNormalPDFWithWorkspace(
        rows.byDim!0, mean[], precisionCholeskyView(factor[].sliced(1, 1)), values[], workRows.byDim!0);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(data[j], 1.0, 2.0)) < 1e-14);
}

/++
Result of evaluating a density from a raw covariance or precision matrix.
Check `failedPivot` first. If it is zero, use `density`.
For an empty observation the density is one and the pivot is zero.

Params:
    T = Floating-point element type of the observation.
+/
struct MultivariateNormalPDFResult(T)
{
    /// Density in the input element type, or NaN on factorization failure.
    T density;
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

private MultivariateNormalPDFResult!T densityResult(T)(MultivariateNormalLPDFResult!T outcome)
{
    import mir.math.common: exp;
    if (outcome.failedPivot)
        return MultivariateNormalPDFResult!T(T.nan, outcome.failedPivot);
    return MultivariateNormalPDFResult!T(exp(outcome.logDensity), 0);
}

/++
Evaluate the density using caller-owned scratch and report Cholesky failure.
With ordinary floating-point slices for inputs and scratch, the call is `nothrow @nogc`.
Each workspace must have nonoverlapping elements and be disjoint from the
other workspace and from all inputs.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x and finite
        entries in the selected triangle.
    work = Writable scratch vector of k elements of the same type as x.
    factorWork = Writable matrix of the same shape and element type as metric.
        On success its selected triangle holds the factor. On failure it may
        contain a partial factorization.

Returns:
    A $(LREF MultivariateNormalPDFResult). On success, `failedPivot == 0`
    and `density` is the density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `density` is NaN. In zero dimensions the density is one and the pivot is zero.
+/
MultivariateNormalPDFResult!(Element!X) tryMultivariateNormalPDFWithWorkspace(X, C, M, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)) &&
        vector!W && matrix!F && is(Element!X == Element!W) && is(Element!X == Element!F))
{
    return densityResult(tryMultivariateNormalLPDFWithWorkspace(x, mean, metric, work, factorWork));
}

/// A one-dimensional precision of one quarter gives a standard deviation of two.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    float[1] x = [3], mean = [1], precision = [0.25f], work, factor;
    const result = tryMultivariateNormalPDFWithWorkspace(
        x[], mean[], precisionView(precision[].sliced(1, 1)), work[], factor[].sliced(1, 1));
    static assert(is(typeof(result.density) == const(float)));
    assert(result.failedPivot == 0);
    assert(fabs(result.density - normalPDF(3f, 1f, 2f)) < 4 * float.epsilon);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    double[1] x = [0x1p512], mean = [0], covariance = [1], work, factor;
    auto metric = covarianceView(covariance[].sliced(1, 1));
    auto buffer = factor[].sliced(1, 1);
    const tail = tryMultivariateNormalPDFWithWorkspace(x[], mean[], metric, work[], buffer);
    assert(tail.failedPivot == 0 && tail.density == 0);
    covariance[0] = 0;
    const singular = tryMultivariateNormalPDFWithWorkspace(x[], mean[], metric, work[], buffer);
    assert(singular.failedPivot == 1 && isNaN(singular.density));
    double[] empty;
    const zeroDim = tryMultivariateNormalPDFWithWorkspace(empty, empty,
        covarianceView(empty.sliced(0, 0)), empty, empty.sliced(0, 0));
    assert(zeroDim.failedPivot == 0 && zeroDim.density == 1);
}

/++
Evaluate selected rows or columns using one shared covariance or precision matrix.
Select observation and scratch axes independently. Both selections are forwarded
as column views without copying. Writable elements must be disjoint from one
another and from all inputs.

Params:
    x = Rectangular selection of n observations of k finite coordinates, made with
        byDim or alongDim. For example, rows.byDim!0 selects rows.
    mean = Vector of k finite means. All buffers share the observation element type.
    metric = Raw covariance or precision view of a dense k-by-k matrix with finite selected triangle.
    result = Writable vector of n densities in selection order.
    work = Writable selection of n scratch vectors of length k.
    factorWork = Writable k-by-k factor workspace. May contain a partial factor on failure.

Returns:
    Zero on success or the one-based failed pivot. Failure leaves result untouched.
+/
size_t tryMultivariateNormalPDFWithWorkspace(X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && isObservationSelection!X && isObservationSelection!W && __traits(compiles,
        tryMultivariateNormalPDFBatchWithWorkspace(
            ObservationColumns!X.init, mean, metric, result, ObservationColumns!W.init, factorWork)))
{
    auto columns = observationColumns(x);
    auto scratch = observationColumns(work);
    return tryMultivariateNormalPDFBatchWithWorkspace(columns, mean, metric, result, scratch, factorWork);
}

/// Evaluate the inlet and chamber temperatures for three test runs.
/// Each row contains the first two readings in the module's temperature example.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.math: PI, fabs, exp;
    double[6] data = [10, 20, 12, 21, 10, 24], scratch;
    double[2] mean = [10, 20];
    double[4] covariance = [4, 2, 2, 5], factor;
    double[3] densities;
    auto rows = data[].sliced(3, 2);
    auto workRows = scratch[].sliced(3, 2);
    const pivot = tryMultivariateNormalPDFWithWorkspace(rows.byDim!0, mean[],
        covarianceView(covariance[].sliced(2, 2)), densities[], workRows.byDim!0, factor[].sliced(2, 2));
    assert(pivot == 0);
    // The first run is at the mean. The covariance determinant is 16.
    const atMean = 1 / (8 * double(PI));
    assert(fabs(densities[0] - atMean) < 1e-14);
    // The second run's [12, 21] centers to [2, 1] and transforms to [1, 0].
    assert(fabs(densities[1] - (atMean * exp(-0.5))) < 1e-14);
    // The third run's transformed row is [0, 2].
    assert(fabs(densities[2] - (atMean * exp(-2.0))) < 1e-14);
}

/++
Evaluate one observation per column with GC-managed result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A GC-managed vector of n densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.gc, Element!X, 1)
multivariateNormalPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    auto values = multivariateNormalLPDFBatch(x, mean, metric);
    exponentiateLogDensities(values);
    return values;
}

/// Evaluate two columns with a shared raw precision.
version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], precision = [0.25];
    auto columns = readings[].sliced(1, 2);
    auto values = multivariateNormalPDFBatch(
        columns, mean[], precisionView(precision[].sliced(1, 1)));
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

// Diagnose a missing label only for otherwise supported batch arguments.
auto multivariateNormalPDFBatch(X, C, M)(scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (matrix!M && __traits(compiles, multivariateNormalPDFBatch(x, mean, covarianceView(metric))))
{
    static assert(0, "Specify the matrix representation. For covariance, call " ~
        "multivariateNormalPDFBatch(x, mean, covarianceView(metric)).");
}

version(mir_stat_test_multivariate_normal)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;

    double[2] readings;
    double[1] mean, storage;
    auto columns = readings[].sliced(1, 2);
    auto metric = storage[].sliced(1, 1);
    static assert(!__traits(compiles, multivariateNormalPDFBatch(columns, mean[], metric)));
    static assert(__traits(compiles, multivariateNormalPDFBatch(columns, mean[], covarianceView(metric))));
}

/++
Evaluate one observation per column with reference-counted result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A reference-counted vector of n densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.rc, Element!X, 1)
rcMultivariateNormalPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    auto values = rcMultivariateNormalLPDFBatch(x, mean, metric);
    exponentiateLogDensities(values);
    return values;
}

/// Evaluate two observations using an upper Cholesky factor of the precision matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: PI, fabs, exp;

    // The columns are [10, 20] (the mean) and [11, 21].
    double[4] readings = [10, 11, 20, 21];
    double[2] mean = [10, 20];
    auto columns = readings[].sliced(2, 2);

    // The precision is P = U.transpose * U, with U = [[2, 1], [0, 1]].
    double[4] upper = [2, 1, 0, 1], precision = [4, 2, 2, 2];
    auto values = rcMultivariateNormalPDFBatch(
        columns, mean[], precisionCholeskyView(upper[].sliced(2, 2)));
    auto raw = tryRcMultivariateNormalPDFBatch(
        columns, mean[], precisionView(precision[].sliced(2, 2)));
    assert(raw.failedPivot == 0);
    assert(values.length == 2 && raw.densities.length == 2);

    // det(P) = 4. For the second observation, U * [1, 1] = [3, 1],
    // so the density is its value at the mean times exp(-5).
    const atMean = 1 / double(PI);
    assert(fabs(values[0] - atMean) < 1e-14);
    assert(fabs(values[1] - (atMean * exp(-5.0))) < 1e-14);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - raw.densities[j]) < 1e-14);
}

/++
Evaluate one observation per column with custom-allocated result and scratch storage.
A raw metric is factored once. All observations share the same mean and log determinant.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Labeled dense k-by-k covariance, precision, or Cholesky factor of either,
        with the same element type as x and finite selected triangle. Raw
        matrices must be positive definite and factors must have positive diagonals.

Returns:
    A custom-allocated vector of n densities in column order, in
    the element type of x. An empty batch gives an empty vector. Each
    zero-dimensional observation has density one. Release the returned
    vector's field with the supplied allocator.

Throws:
    Exception if factorization of a raw metric fails, including for an empty batch.
+/
AllocatedSlice!(Allocation.custom, Element!X, 1)
makeMultivariateNormalPDFBatch(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && batchInputs!(X, C, typeof(metric.storage)))
{
    auto values = makeMultivariateNormalLPDFBatch(allocator, x, mean, metric);
    exponentiateLogDensities(values);
    return values;
}

/// Allocate the column results with Mallocator.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs;
    double[2] readings = [10, 14];
    double[1] mean = [10], factor = [2];
    auto columns = readings[].sliced(1, 2);
    auto values = makeMultivariateNormalPDFBatch(
        Mallocator.instance, columns, mean[], covarianceCholeskyView(factor[].sliced(1, 1)));
    scope(exit) Mallocator.instance.dispose(values.field);
    assert(values.length == 2);
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Outcome of batch evaluation with a raw covariance or precision matrix.

Params:
    V = Vector type for the chosen allocation strategy.
+/
struct MultivariateNormalPDFBatchResult(V)
{
    /// One density per observation on success. Empty on factorization failure.
    V densities;
    /// Zero on success, including an empty batch. A positive value is the one-based
    /// Cholesky step at which the selected matrix failed the positive-definiteness check.
    size_t failedPivot;
}

/++
Evaluate one density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a GC-managed vector of n densities
    in column order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty.
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.gc, Element!X, 1))
tryMultivariateNormalPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    auto outcome = tryMultivariateNormalLPDFBatch(x, mean, metric);
    return densityBatchResult(outcome);
}

/// A singular covariance has no batch densities. The pivot identifies the failed step.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [0];
    auto columns = data[].sliced(1, 2);
    auto result = tryMultivariateNormalPDFBatch(columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 1);
    assert(result.densities.length == 0);
}

/++
Evaluate one density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a reference-counted vector of n densities
    in column order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty.
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.rc, Element!X, 1))
tryRcMultivariateNormalPDFBatch(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    auto outcome = tryRcMultivariateNormalLPDFBatch(x, mean, metric);
    return densityBatchResult(outcome);
}

/// Evaluate two columns and check whether factorization succeeded.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto columns = data[].sliced(1, 2);
    auto result = tryRcMultivariateNormalPDFBatch(columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    assert(result.densities.length == 2);
    assert(result.densities[0] > result.densities[1]);
}

// The copied RC result owns the densities after the original leaves scope.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;

    static auto retainedResult()
    {
        double[2] x = [1, 3];
        double[1] mean = [1], covariance = [4];
        auto original = tryRcMultivariateNormalPDFBatch(x[].sliced(1, 2), mean[],
            covarianceView(covariance[].sliced(1, 1)));
        auto copy = original;
        assert(copy.densities.length == original.densities.length);
        return copy;
    }

    auto retained = retainedResult();
    assert(retained.failedPivot == 0 && retained.densities.length == 2);
    assert(fabs(retained.densities[0] - normalPDF(1.0, 1.0, 2.0)) < double.epsilon);
    assert(fabs(retained.densities[1] - normalPDF(3.0, 1.0, 2.0)) < double.epsilon);
}

/++
Evaluate one density per observation column and report a failed Cholesky step.
The observations share a single factorization.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Dense k-by-n matrix of finite observations, one observation per column.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a custom-allocated vector of n densities
    in column order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty. Release the successful result with
    allocator.dispose(result.densities.field).
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.custom, Element!X, 1))
tryMakeMultivariateNormalPDFBatch(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && batchInputs!(X, C, typeof(metric.storage)))
{
    auto outcome = tryMakeMultivariateNormalLPDFBatch(allocator, x, mean, metric);
    return densityBatchResult(outcome);
}

/// Evaluate two columns and check whether factorization succeeded.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto columns = data[].sliced(1, 2);
    auto result = tryMakeMultivariateNormalPDFBatch(Mallocator.instance, columns, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    scope(exit) Mallocator.instance.dispose(result.densities.field);
    assert(result.densities.length == 2);
    assert(result.densities[0] > result.densities[1]);
}

/++
Evaluate the density using GC-managed scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow`.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFResult). On success, `failedPivot == 0`
    and `density` is the density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `density` is NaN. In zero dimensions the density is one and the pivot is zero.
+/
MultivariateNormalPDFResult!(Element!X) tryMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    return densityResult(tryMultivariateNormalLPDF(x, mean, metric));
}

/// The first diagonal gives a factor entry of 2. The second pivot is 1 - (2/2) * (2/2) = 0.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    double[2] x = [1, 2], mean = [0, 0];
    double[4] covariance = [4, 2, double.nan, 1];
    const result = tryMultivariateNormalPDF(x[], mean[], covarianceView(covariance[].sliced(2, 2)));
    assert(result.failedPivot == 2 && isNaN(result.density));
}

/++
Evaluate one density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a GC-managed vector of n densities
    in selection order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty.
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.gc, Element!(ObservationColumns!X), 1))
tryMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryMultivariateNormalPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryMultivariateNormalPDFBatch(columns, mean, metric);
}

/// Check the pivot before using the densities for the selected rows.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto rows = data[].sliced(2, 1);
    auto result = tryMultivariateNormalPDF(rows.byDim!0, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    assert(result.densities.length == 2);
    assert(result.densities[0] > result.densities[1]);
}

/++
Evaluate the density using reference-counted scratch and report Cholesky failure.
With ordinary floating-point slices, the call is `nothrow @nogc`.

Params:
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFResult). On success, `failedPivot == 0`
    and `density` is the density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `density` is NaN. In zero dimensions the density is one and the pivot is zero.
+/
MultivariateNormalPDFResult!(Element!X) tryRcMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    return densityResult(tryRcMultivariateNormalLPDF(x, mean, metric));
}

/// Evaluate a raw covariance using reference-counted scratch.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs, isNaN;

    double[1] x = [3], mean = [1], covariance = [4];
    auto metric = covariance[].sliced(1, 1);
    const result = tryRcMultivariateNormalPDF(x[], mean[], covarianceView(metric));
    assert(result.failedPivot == 0);
    assert(fabs(result.density - normalPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = tryRcMultivariateNormalPDF(x[], mean[], covarianceView(metric));
    assert(singular.failedPivot == 1 && isNaN(singular.density));
}

/++
Evaluate one density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a reference-counted vector of n densities
    in selection order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty.
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.rc, Element!(ObservationColumns!X), 1))
tryRcMultivariateNormalPDF(X, C, M)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryRcMultivariateNormalPDFBatch(ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryRcMultivariateNormalPDFBatch(columns, mean, metric);
}

/// Evaluate selected rows with raw precision. A precision of 0.25 gives a standard deviation of 2.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;

    double[2] readings = [10, 14];
    double[1] mean = [10], precision = [0.25];
    auto rows = readings[].sliced(2, 1);
    auto result = tryRcMultivariateNormalPDF(
        rows.byDim!0, mean[], precisionView(precision[].sliced(1, 1)));
    assert(result.failedPivot == 0 && result.densities.length == 2);
    foreach (j; 0 .. readings.length)
        assert(fabs(result.densities[j] - normalPDF(readings[j], 10.0, 2.0)) < 1e-14);
}

/++
Evaluate the density using a caller-selected scratch allocator and report Cholesky failure.
The call is `nothrow @nogc` with ordinary floating-point slices and
`Mallocator`. With other allocators or iterators, the inferred attributes depend
on their operations.

Params:
    allocator = Allocator used to obtain and release temporary storage.
    x = Observation vector of k finite floating-point coordinates.
    mean = Vector of k finite means with the same element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with the same element type as x. The selected
        triangle must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFResult). On success, `failedPivot == 0`
    and `density` is the density at `x`. On factorization
    failure, `failedPivot` identifies the failed step (counting from 1)
    and `density` is NaN. In zero dimensions the density is one and the pivot is zero.
+/
MultivariateNormalPDFResult!(Element!X) tryMakeMultivariateNormalPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && !prepared!(M.form) && compatible!(X, C, typeof(metric.storage)))
{
    return densityResult(tryMakeMultivariateNormalLPDF(allocator, x, mean, metric));
}

/// Evaluate a raw precision using Mallocator for scratch.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.math: fabs, isNaN;

    double[1] x = [3], mean = [1], precision = [0.25];
    auto metric = precision[].sliced(1, 1);
    const result = tryMakeMultivariateNormalPDF(Mallocator.instance, x[], mean[], precisionView(metric));
    assert(result.failedPivot == 0);
    assert(fabs(result.density - normalPDF(3.0, 1.0, 2.0)) < 1e-14);

    metric[0, 0] = 0;
    const singular = tryMakeMultivariateNormalPDF(Mallocator.instance, x[], mean[], precisionView(metric));
    assert(singular.failedPivot == 1 && isNaN(singular.density));
}

/++
Evaluate one density per selected observation and report a failed Cholesky step.
The observations share a single factorization.

Params:
    allocator = Allocator for the result and temporary storage.
    x = Rectangular selection of n observations with k finite coordinates each.
        Select rows with byDim!0 or alongDim!1, or columns with byDim!1 or alongDim!0.
    mean = Vector of k finite means with the same floating-point element type as x.
    metric = Raw covariance or precision view of a dense k-by-k matrix of the same element type. The selected triangle
        must contain finite entries and defines the symmetric matrix.

Returns:
    A $(LREF MultivariateNormalPDFBatchResult). On success, failedPivot is zero
    and densities is a custom-allocated vector of n densities
    in selection order. Each zero-dimensional observation has density one.
    On factorization failure, even for an empty batch, failedPivot is the one-based
    failed step and densities is empty. Release the successful result with
    allocator.dispose(result.densities.field).
+/
MultivariateNormalPDFBatchResult!(AllocatedSlice!(Allocation.custom, Element!(ObservationColumns!X), 1))
tryMakeMultivariateNormalPDF(A, X, C, M)(
    auto ref A allocator, scope auto ref X x, scope auto ref C mean, scope auto ref M metric)
    if (isCovarianceMetric!M && isObservationSelection!X && __traits(compiles,
        tryMakeMultivariateNormalPDFBatch(allocator, ObservationColumns!X.init, mean, metric)))
{
    auto columns = observationColumns(x);
    return tryMakeMultivariateNormalPDFBatch(allocator, columns, mean, metric);
}

/// Check the pivot before using the densities for the selected rows.
version(mir_stat_test_multivariate_normal)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim;
    import std.experimental.allocator: dispose;
    import std.experimental.allocator.mallocator: Mallocator;
    double[2] data = [10, 14];
    double[1] mean = [10], covariance = [4];
    auto rows = data[].sliced(2, 1);
    auto result = tryMakeMultivariateNormalPDF(Mallocator.instance, rows.byDim!0, mean[], covarianceView(covariance[].sliced(1, 1)));
    assert(result.failedPivot == 0);
    scope(exit) Mallocator.instance.dispose(result.densities.field);
    assert(result.densities.length == 2);
    assert(result.densities[0] > result.densities[1]);
}

/++
Write one density per observation column using a shared Cholesky factor.
All writable elements must be disjoint from one another and from the inputs.

Params:
    x = Dense k-by-n matrix of finite observations. Column j is observation j.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    factor = Labeled dense k-by-k factor with finite selected triangle and positive diagonal.
    result = Writable vector of n densities in column order.
    work = Writable k-by-n scratch for centering and transforming the observations.

Returns:
    void. Writes each density to result. In zero dimensions each value is zero.
+/
void multivariateNormalPDFBatchWithWorkspace(X, C, M, R, W)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work)
    if (isCovarianceMetric!M && prepared!(M.form) && batchBuffers!(X, C, typeof(factor.storage), R, W))
{
    auto values = result.asView;
    multivariateNormalLPDFBatchWithWorkspace(x, mean, factor, values, work);
    exponentiateLogDensities(values);
}

/// Two columns, each with one coordinate and standard deviation two.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;
    double[2] x = [1, 3], values, scratch;
    double[1] mean = [1], factor = [2];
    multivariateNormalPDFBatchWithWorkspace(
        x[].sliced(1, 2), mean[], covarianceCholeskyView(factor[].sliced(1, 1)), values[], scratch[].sliced(1, 2));
    foreach (j; 0 .. values.length)
        assert(fabs(values[j] - normalPDF(x[j], 1.0, 2.0)) < 1e-14);
}

/++
Factor a covariance or precision matrix once, then write one density per column.
For a prepared factor, use $(LREF multivariateNormalPDFBatchWithWorkspace) and omit factorWork.
All writable elements must be disjoint from one another and from the inputs.

Params:
    x = Dense k-by-n matrix of finite observations.
    mean = Vector of k finite means. All buffers share the floating-point element type of x.
    metric = Raw covariance or precision view of a dense k-by-k matrix with finite entries in the selected triangle.
    result = Writable vector of n densities in column order. A failed factorization
        leaves every entry untouched.
    work = Writable k-by-n scratch for centering and transforming the observations.
    factorWork = Writable k-by-k scratch. Its selected triangle holds the factor on
        success and may contain a partial factor on failure.

Returns:
    Zero on success, or the one-based failed Cholesky pivot. The matrix is checked
    even for an empty batch. In zero dimensions each density is one.
+/
size_t tryMultivariateNormalPDFBatchWithWorkspace(X, C, M, R, W, F)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M metric,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && !prepared!(M.form) &&
        batchBuffers!(X, C, typeof(metric.storage), R, W) && matrix!F && is(Element!X == Element!F))
{
    auto values = result.asView;
    const pivot = tryMultivariateNormalLPDFBatchWithWorkspace(x, mean, metric, values, work, factorWork);
    if (pivot == 0) exponentiateLogDensities(values);
    return pivot;
}

/// A failed second pivot preserves both entries of the density vector.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] x = [1, 2, 3, 4], metric = [1, 1, double.nan, 1], scratch, factor;
    double[2] mean = [0, 0], values = [7, 8];
    assert(tryMultivariateNormalPDFBatchWithWorkspace(x[].sliced(2, 2), mean[],
        covarianceView(metric[].sliced(2, 2)), values[], scratch[].sliced(2, 2), factor[].sliced(2, 2)) == 2);
    assert(values == [7, 8]);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: reversed;
    import mir.stat.distribution.normal: normalPDF;
    import std.math: fabs;

    double[3] x = [3, 1, 0x1p512], scratch;
    double[1] mean = [1], precision = [1], factor;
    double[6] buffer = [11, 12, 13, 14, 15, 16];
    auto values = buffer[].sliced(3, 2)[0 .. $, 1].reversed;
    auto metric = precisionView(precision[].sliced(1, 1));
    const pivot = tryMultivariateNormalPDFBatchWithWorkspace(x[].sliced(1, 3), mean[],
        metric, values, scratch[].sliced(1, 3), factor[].sliced(1, 1));
    assert(pivot == 0);
    assert(fabs(buffer[5] - normalPDF(3.0, 1.0, 1.0)) < double.epsilon);
    assert(fabs(buffer[3] - normalPDF(1.0, 1.0, 1.0)) < double.epsilon);
    assert(buffer[1] == 0 && buffer[0] == 11 && buffer[2] == 13 && buffer[4] == 15);

    double[] empty;
    auto emptyMetric = covarianceView(empty.sliced(0, 0));
    auto result = tryRcMultivariateNormalPDFBatch(empty.sliced(0, 3), empty, emptyMetric);
    assert(result.failedPivot == 0 && result.densities == [1, 1, 1]);
    auto noObservations = empty.sliced(1, 0);
    result = tryRcMultivariateNormalPDFBatch(noObservations, mean[], metric);
    assert(result.failedPivot == 0 && result.densities.length == 0);
    precision[0] = 0;
    result = tryRcMultivariateNormalPDFBatch(noObservations, mean[], metric);
    assert(result.failedPivot == 1 && result.densities.length == 0);
}

// Infer the return type so __traits(compiles) also rejects a prepared factor here.
auto tryMultivariateNormalPDFBatchWithWorkspace(X, C, M, R, W, F...)(
    scope auto ref X x, scope auto ref C mean, scope auto ref M factor,
    scope auto ref R result, scope auto ref W work, scope auto ref F factorWork)
    if (isCovarianceMetric!M && prepared!(M.form) && batchBuffers!(X, C, typeof(factor.storage), R, W) &&
        (F.length == 0 || (F.length == 1 && matrix!(F[0]) && is(Element!X == Element!(F[0])))))
{
    static assert(0, "For a prepared Cholesky factor, use " ~
        "multivariateNormalPDFBatchWithWorkspace(x, mean, factor, result, work). " ~
        "It writes the densities and returns void.");
}

private MultivariateNormalPDFBatchResult!V densityBatchResult(V)(MultivariateNormalLPDFBatchResult!V outcome)
{
    if (outcome.failedPivot == 0) exponentiateLogDensities(outcome.logDensities);
    return MultivariateNormalPDFBatchResult!V(outcome.logDensities, outcome.failedPivot);
}

/// Convert successful LPDF output in its existing storage, respecting the output stride.
private void exponentiateLogDensities(R)(scope auto ref R values)
{
    import mir.math.common: exp;
    foreach (j; 0 .. values.length)
        values[j] = exp(values[j]);
}
