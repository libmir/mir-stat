/++
Internal linear algebra operations with explicit result ownership.

Declarations are restricted to `mir.stat` and its subpackages; this package
is not a public application API.

Import this package to use the matrix and vector multiplication APIs from
`mir.stat.internal.linearAlgebra.products` and triangular products from
`mir.stat.internal.linearAlgebra.triangular`.

Choose result ownership independently of input ownership:
$(UL
    $(LI `mtimes` returns GC-managed storage.)
    $(LI `rcMtimes` returns reference-counted storage.)
    $(LI `makeMtimes` uses a supplied allocator; the caller disposes the result.)
    $(LI `mtimesInto` overwrites caller-provided storage without allocating.)
)
Two vector operands produce a scalar without allocation in the first three
functions. Inputs can be borrowed or owning ndslices, including transposed,
strided, and computed views. Floating-point and complex element types are
supported; operands must have the same element type. Complex multiplication
does not conjugate either operand.

Use `quadraticForm(x, a)` for a general real quadratic form, or
`quadraticForm(x, symmetricView(a))` to read only one triangle of a symmetric matrix.
Both return scalars without allocating, regardless of input ownership.

`symmetricView` and `triangularView` attach structure to a matrix without
copying its elements. Symmetric views dispatch quadratic forms and Cholesky
decomposition; triangular views dispatch multiplication with one ordinary
operand. Views are read-only, retain RC ownership when supplied, and have no
implicit ndslice conversion. They do not assert positive definiteness or
represent Cholesky factors. Transpose updates the selected triangle.

BLAS is optional. The default configuration uses scalar kernels; compatible
native layouts can use BLAS when that dependency is available. Backend and
summation order can affect floating-point rounding.

For a square triangular operand, use the `mtimesTriangular` family on the left
or `mtimesTriangularRight` family on the right. They follow the same ownership
conventions, with `Triangle.upper`/`lower` and `Diagonal.nonUnit`/`unit` options.
Unused entries and implicit unit diagonals are not read. Triangle selection
always refers to the logical view, so a transposed upper triangle is lower.

For real symmetric positive-definite matrices, the `choleskyDecompose` family
produces upper or lower factors with the same ownership choices. `Into` uses
separate caller storage; `InPlace` reuses writable input. The `try` in-place
variant returns a failed pivot index without throwing. Factorization is
available without LAPACK, with optional acceleration for native layouts.
The `choleskySolve` family reuses a factor to solve vector or matrix right-hand
sides without inversion. It follows the same ownership conventions and offers
caller-output and in-place forms, with optional BLAS triangular solves.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

package(mir.stat) import mir.stat.internal.linearAlgebra.products;
package(mir.stat) import mir.stat.internal.linearAlgebra.views;
package(mir.stat) import mir.stat.internal.linearAlgebra.quadratic;
package(mir.stat) import mir.stat.internal.linearAlgebra.triangular;
package(mir.stat) import mir.stat.internal.linearAlgebra.cholesky;

/// Multiply borrowed matrix views and choose ownership only for the result.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.stat.internal.linearAlgebra: mtimes, rcMtimes, mtimesInto;
    import mir.ndslice.slice: sliced;

    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto gcResult = mtimes(a, a);
    auto rcResult = rcMtimes(a, a);
    double[4] buffer;
    auto output = buffer[].sliced(2, 2);
    mtimesInto(a, a, output);
    assert(gcResult == [[7, 10], [15, 22]]);
    assert(rcResult == gcResult && output == gcResult);
}
