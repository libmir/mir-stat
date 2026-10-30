/++
Convenience imports for custom-allocator linear algebra results.

Import this module within `mir.stat` to select this result-storage strategy.
It selectively re-exports `makeMtimes`, `makeMtimesTriangular`, `makeMtimesTriangularRight`,
`makeCholeskyDecompose`, and `makeCholeskySolve` with their original names and overloads.
There are no wrappers or new implementations. Input ownership is independent
of result ownership. Scalar dot products are allocation-free; import `dotProduct` separately
from `products`. Matrix multiplication requires at least one matrix operand.

Views and structural policies are imported separately from `views` and `types`.
Matrix cross-products are included with matrix-only inputs and results.
Batch quadratic forms and their selected-observation adapters are included
with the same ownership policy.
Caller-output and in-place operations remain in their operation modules. This module is package-restricted, not a public
application API.
One-input cross-products return symmetric views; release them with
`disposeSymmetric(allocator, result)`. For ordinary slices, callers dispose the field with the allocator that
created it, as described by the underlying operation.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.custom;

package(mir.stat) import mir.stat.internal.linearAlgebra.products: makeMtimes;
package(mir.stat) import mir.stat.internal.linearAlgebra.triangular: makeMtimesTriangular, makeMtimesTriangularRight;
package(mir.stat) import mir.stat.internal.linearAlgebra.cholesky: makeCholeskyDecompose, makeCholeskySolve;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

/// Select one allocation strategy while retaining the ordinary operation names.
version(mir_stat_test_linearAlgebra)
@system @nogc unittest
{
    import mir.stat.internal.linearAlgebra.custom: makeMtimes, makeCholeskyDecompose, makeCholeskySolve;
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [4, 0, 0, 16];
    double[2] values = [2, 4];
    auto a = data[].sliced(2, 2);
    auto x = values[].sliced;
    auto product = makeMtimes(Mallocator.instance, a, x);
    scope(exit) Mallocator.instance.dispose(product.field);
    auto factor = makeCholeskyDecompose(Mallocator.instance, a);
    scope(exit) Mallocator.instance.dispose(factor.field);
    auto solution = makeCholeskySolve(Mallocator.instance, factor, x);
    scope(exit) Mallocator.instance.dispose(solution.field);
    assert(product == [8, 64]);
    assert(solution == [0.5, 0.25]);
}

version(mir_stat_test_linearAlgebra)
@system @nogc unittest
{
    import products = mir.stat.internal.linearAlgebra.products;
    import triangular = mir.stat.internal.linearAlgebra.triangular;
    import cholesky = mir.stat.internal.linearAlgebra.cholesky;
    static assert(__traits(isSame, makeMtimes, products.makeMtimes));
    static assert(__traits(isSame, makeMtimesTriangular, triangular.makeMtimesTriangular));
    static assert(__traits(isSame, makeMtimesTriangularRight, triangular.makeMtimesTriangularRight));
    static assert(__traits(isSame, makeCholeskyDecompose, cholesky.makeCholeskyDecompose));
    static assert(__traits(isSame, makeCholeskySolve, cholesky.makeCholeskySolve));
}

package(mir.stat) import mir.stat.internal.linearAlgebra.quadratic: makeQuadraticFormBatch, makeQuadraticForm;

/// Matrix-only cross-products with the selected result ownership.
package(mir.stat) import mir.stat.internal.linearAlgebra.crossproducts: makeCrossprod, makeTcrossprod;

/// Release an original custom-allocated symmetric cross-product result.
package(mir.stat) import mir.stat.internal.linearAlgebra.views: disposeSymmetric;
