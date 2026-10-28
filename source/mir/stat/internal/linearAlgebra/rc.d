/++
Convenience imports for reference-counted linear algebra results.

Import this module within `mir.stat` to select this result-storage strategy.
It selectively re-exports `rcMtimes`, `rcMtimesTriangular`, `rcMtimesTriangularRight`,
`rcCholeskyDecompose`, and `rcCholeskySolve` with their original names and overloads.
There are no wrappers or new implementations. Input ownership is independent
of result ownership. Vector-vector multiplication still returns a scalar
without allocating.

Views and structural policies are imported separately from `views` and `types`.
Batch quadratic forms and their selected-observation adapters are included
with the same ownership policy.
Caller-output and in-place operations remain in their operation modules. This module is package-restricted, not a public
application API.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.rc;

package(mir.stat) import mir.stat.internal.linearAlgebra.products: rcMtimes;
package(mir.stat) import mir.stat.internal.linearAlgebra.triangular: rcMtimesTriangular, rcMtimesTriangularRight;
package(mir.stat) import mir.stat.internal.linearAlgebra.cholesky: rcCholeskyDecompose, rcCholeskySolve;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

/// Select one allocation strategy while retaining the ordinary operation names.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.stat.internal.linearAlgebra.rc: rcMtimes, rcCholeskyDecompose, rcCholeskySolve;
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 0, 0, 16];
    double[2] values = [2, 4];
    auto a = data[].sliced(2, 2);
    auto x = values[].sliced;
    auto product = rcMtimes(a, x);
    auto factor = rcCholeskyDecompose(a);
    auto solution = rcCholeskySolve(factor, x);
    assert(product == [8, 64]);
    assert(solution == [0.5, 0.25]);
}

version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import products = mir.stat.internal.linearAlgebra.products;
    import triangular = mir.stat.internal.linearAlgebra.triangular;
    import cholesky = mir.stat.internal.linearAlgebra.cholesky;
    static assert(__traits(isSame, rcMtimes, products.rcMtimes));
    static assert(__traits(isSame, rcMtimesTriangular, triangular.rcMtimesTriangular));
    static assert(__traits(isSame, rcMtimesTriangularRight, triangular.rcMtimesTriangularRight));
    static assert(__traits(isSame, rcCholeskyDecompose, cholesky.rcCholeskyDecompose));
    static assert(__traits(isSame, rcCholeskySolve, cholesky.rcCholeskySolve));
}

package(mir.stat) import mir.stat.internal.linearAlgebra.quadratic: rcQuadraticFormBatch, rcQuadraticForm;
