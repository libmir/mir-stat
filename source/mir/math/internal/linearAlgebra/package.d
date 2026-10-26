/++
Internal linear algebra operations with explicit result ownership.

Import this package to use the matrix and vector multiplication APIs from
`mir.math.internal.linearAlgebra.products`.

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

BLAS is optional. The default configuration uses scalar kernels; compatible
native layouts can use BLAS when that dependency is available. Backend and
summation order can affect floating-point rounding. These internal APIs are
currently limited to dense matrix/vector products, not factorizations or
specialized triangular operations.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.math.internal.linearAlgebra;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

public import mir.math.internal.linearAlgebra.products;

/// Multiply borrowed matrix views and choose ownership only for the result.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.math.internal.linearAlgebra: mtimes, rcMtimes, mtimesInto;
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
