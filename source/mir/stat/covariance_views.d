/++
Dense covariance and precision inputs for multivariate statistics.
Each view records which matrix is supplied and which triangle to read.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.covariance_views;

public import mir.stat.internal.linearAlgebra.types: CovarianceRepresentation, TrianglePolicy;
private import mir.stat.internal.linearAlgebra.types: matrix;
private import mir.ndslice.slice: isSlice, toSlice;

version(mir_stat_test) version = mir_stat_test_multivariate_normal;
version(mir_stat_test_blas) version = mir_stat_test_multivariate_normal;

/++
A square matrix descriptor with its statistical interpretation. Construct a view
with $(LREF covarianceView), $(LREF precisionView), $(LREF covarianceCholeskyView),
or $(LREF precisionCholeskyView). A view retains an owning slice. Borrowed
storage must remain alive while the view is used. Changes to the underlying
entries are visible to the view.

Params:
    representation = Covariance, precision, or a Cholesky factor of either.
    selectedTriangle = Triangle of the logical matrix containing the entries to read.
    A = Dense floating-point matrix descriptor.
+/
struct CovarianceMetric(CovarianceRepresentation representation, TrianglePolicy selectedTriangle, A)
    if (matrix!A &&
        (selectedTriangle == TrianglePolicy.upper || selectedTriangle == TrianglePolicy.lower) &&
        (representation == CovarianceRepresentation.covariance ||
         representation == CovarianceRepresentation.inverseCovariance ||
         representation == CovarianceRepresentation.covarianceCholesky ||
         representation == CovarianceRepresentation.inverseCovarianceCholesky))
{
    /// Statistical interpretation of the stored entries.
    enum form = representation;
    /// Logical triangle to read.
    enum triangle = selectedTriangle;
    package(mir.stat) A storage;
}

/// Whether a matrix descriptor carries a covariance or precision representation.
package(mir.stat) enum isCovarianceMetric(M) = is(M == CovarianceMetric!Args, Args...);

// Keep the descriptor itself so reference-counted iterators retain their owner.
package(mir.stat) auto labelCovariance(CovarianceRepresentation form, TrianglePolicy triangle, A)(
    return scope auto ref A input) if (matrix!A)
{
    static if (isSlice!A)
        alias view = input;
    else
        auto view = input.toSlice;
    assert(view.length!0 == view.length!1, "Covariance and precision inputs must be square");
    return CovarianceMetric!(form, triangle, typeof(view))(view);
}

/++
Label a dense covariance matrix for use in multivariate statistics.
The selected triangle defines the symmetric positive-definite covariance.

Params:
    triangle = Logical triangle to read, upper by default.
    input = Square floating-point matrix view containing the supplied entries.

Returns:
    A $(LREF CovarianceMetric) referencing the same entries. Construction
    copies the descriptor and performs no factorization or numerical validation.
+/
auto covarianceView(TrianglePolicy triangle = TrianglePolicy.upper, A)(return scope auto ref A input)
    if (matrix!A)
{
    return labelCovariance!(CovarianceRepresentation.covariance, triangle)(input);
}

/// Label the lower triangle of the matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] entries = [2, 0, 1, 2];
    auto metric = covarianceView!(TrianglePolicy.lower)(entries[].sliced(2, 2));
    static assert(metric.form == CovarianceRepresentation.covariance);
    static assert(metric.triangle == TrianglePolicy.lower);
}

/++
Label a dense precision matrix for use in multivariate statistics.
Precision is the inverse covariance. The selected triangle defines the symmetric positive-definite precision.

Params:
    triangle = Logical triangle to read, upper by default.
    input = Square floating-point matrix view containing the supplied entries.

Returns:
    A $(LREF CovarianceMetric) referencing the same entries. Construction
    copies the descriptor and performs no factorization or numerical validation.
+/
auto precisionView(TrianglePolicy triangle = TrianglePolicy.upper, A)(return scope auto ref A input)
    if (matrix!A)
{
    return labelCovariance!(CovarianceRepresentation.inverseCovariance, triangle)(input);
}

/// Label the lower triangle of the matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] entries = [2, 0, 1, 2];
    auto metric = precisionView!(TrianglePolicy.lower)(entries[].sliced(2, 2));
    static assert(metric.form == CovarianceRepresentation.inverseCovariance);
    static assert(metric.triangle == TrianglePolicy.lower);
}

/++
Label a dense Cholesky factor of a covariance matrix for use in multivariate statistics.
An upper factor represents covariance as U.transpose * U. A lower factor represents it as L * L.transpose.

Params:
    triangle = Logical triangle to read, upper by default.
    input = Square floating-point matrix view containing the supplied entries.

Returns:
    A $(LREF CovarianceMetric) referencing the same entries. Construction
    copies the descriptor and performs no factorization or numerical validation.
+/
auto covarianceCholeskyView(TrianglePolicy triangle = TrianglePolicy.upper, A)(return scope auto ref A input)
    if (matrix!A)
{
    return labelCovariance!(CovarianceRepresentation.covarianceCholesky, triangle)(input);
}

/// Label the lower triangle of the matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] entries = [2, 0, 1, 2];
    auto metric = covarianceCholeskyView!(TrianglePolicy.lower)(entries[].sliced(2, 2));
    static assert(metric.form == CovarianceRepresentation.covarianceCholesky);
    static assert(metric.triangle == TrianglePolicy.lower);
}

/++
Label a dense Cholesky factor of a precision matrix for use in multivariate statistics.
An upper factor represents precision as U.transpose * U. A lower factor represents it as L * L.transpose.

Params:
    triangle = Logical triangle to read, upper by default.
    input = Square floating-point matrix view containing the supplied entries.

Returns:
    A $(LREF CovarianceMetric) referencing the same entries. Construction
    copies the descriptor and performs no factorization or numerical validation.
+/
auto precisionCholeskyView(TrianglePolicy triangle = TrianglePolicy.upper, A)(return scope auto ref A input)
    if (matrix!A)
{
    return labelCovariance!(CovarianceRepresentation.inverseCovarianceCholesky, triangle)(input);
}

/// Label the lower triangle of the matrix.
version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] entries = [2, 0, 1, 2];
    auto metric = precisionCholeskyView!(TrianglePolicy.lower)(entries[].sliced(2, 2));
    static assert(metric.form == CovarianceRepresentation.inverseCovarianceCholesky);
    static assert(metric.triangle == TrianglePolicy.lower);
}

version(mir_stat_test_multivariate_normal)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;
    static auto retained()
    {
        auto owner = mininitRcslice!double(1, 1);
        owner[0, 0] = 3;
        auto original = precisionCholeskyView(owner);
        auto copy = original;
        assert(copy.storage[0, 0] == original.storage[0, 0]);
        return copy;
    }
    assert(retained().storage[0, 0] == 3);
    double[1] entries = [2];
    auto view = covarianceView(entries[].sliced(1, 1));
    entries[0] = 4;
    assert(view.storage[0, 0] == 4);

    version(mir_stat_test_lifetime) static assert(!__traits(compiles, {
        @safe auto escape()
        {
            double[4] local;
            return covarianceView(local[].sliced(2, 2));
        }
    }));
}
