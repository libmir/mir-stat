/++
Products with a square triangular matrix on either side.

The selected triangle describes the logical ndslice view, including when that
view is transposed. Entries outside it are not read. A unit diagonal is treated
as one without reading its stored values. Complex inputs are not conjugated.
General strides and computed inputs use native D kernels. Compatible native
float, double and complex layouts use BLAS when available.

The `mtimesTriangular` family multiplies a triangular matrix on the left;
the `mtimesTriangularRight` family multiplies one on the right. Ordinary names
return GC storage, `rc` names return owning RC storage, `make` names use a
custom allocator, and `Into` names overwrite caller storage. Input ownership
is independent of output ownership. No scratch allocation is required.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.triangular;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.stat.internal.linearAlgebra.allocation;
import mir.stat.internal.linearAlgebra.packed: isPacked, PackedInfo, packedMatrix;
import mir.stat.internal.linearAlgebra.types: Scalar, numericSlice, TrianglePolicy, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.backend: hasBlas, nativeSlice, blasNumber, blasMatrix, blasVector;

/// Whether a square-matrix operand and numeric right-hand side have matching element types.
private enum compatible(A, B) = numericSlice!A && A.N == 2 &&
    numericSlice!B && is(Scalar!A == Scalar!B);
/// Triangle selected after transposing the logical matrix.
private enum opposite(TrianglePolicy triangle) =
    triangle == TrianglePolicy.upper ? TrianglePolicy.lower : TrianglePolicy.upper;

/++
Multiply a triangular matrix `a` by a matrix or column vector `b`.

Params:
    triangle = Triangle of the logical view of `a` to read.
    diagonal = Whether diagonal entries are stored or implicitly one.
    a = Square triangular matrix with floating-point or complex elements.
    b = Matrix or vector with the same element type as `a`.
Returns:
    A GC-backed slice with the same shape as `b`.
Preconditions:
    `a` must be square and its order must equal the first dimension of `b`.
    Zero-sized matrices are supported. Shapes are checked with assertions.
See_Also: $(LINK2 #rcMtimesTriangular, rcMtimesTriangular),
    $(LINK2 #makeMtimesTriangular, makeMtimesTriangular),
    $(LINK2 #mtimesTriangularInto, mtimesTriangularInto)
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, B.N)
mtimesTriangular(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    int unused;
    return multiplyTriangularWithAllocation!(Allocation.gc, triangle, diagonal, false)(unused, a, b);
}

/// Ignore the unused triangle, even when it contains NaNs.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    assert(mtimesTriangular(values[].sliced(2, 2), vector[].sliced) == [28, 24]);
}

/++
Reference-counted variant of $(LINK2 #mtimesTriangular, mtimesTriangular).
Parameters and shape requirements are the same. The returned owning slice
has the shape of `b`; storage is released with its last owning reference.
Native inputs support `@safe pure nothrow @nogc` calls.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, B.N)
rcMtimesTriangular(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    int unused;
    return multiplyTriangularWithAllocation!(Allocation.rc, triangle, diagonal, false)(unused, a, b);
}

/// Unit diagonals are implicit, so their stored NaNs are not read.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
    import mir.ndslice.slice: sliced;
    double[4] values = [double.nan, double.nan, 3, double.nan];
    double[2] vector = [5, 6];
    auto result = rcMtimesTriangular!(TrianglePolicy.lower, DiagonalPolicy.unit)(
        values[].sliced(2, 2), vector[].sliced);
    assert(result == [5, 21]);
}

/++
Custom-allocator variant of $(LINK2 #mtimesTriangular, mtimesTriangular).
`allocator` must support `makeUninitSlice` and `dispose`. Other parameters and
shape requirements are the same. The result has the shape of `b`; dispose
`result.field` with the same allocator after all result views are finished.
If computation throws, allocated result storage is disposed before propagating
the exception. Attributes depend on the allocator and input iterators.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, B.N)
makeMtimesTriangular(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, Allocator, A, B)(auto ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return multiplyTriangularWithAllocation!(Allocation.custom, triangle, diagonal, false)(allocator, a, b);
}

/// Dispose custom result storage with its allocator.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    auto result = makeMtimesTriangular(Mallocator.instance,
        values[].sliced(2, 2), vector[].sliced);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [28, 24]);
}

/++
Write `a * b` into caller storage without allocating.

Params:
    triangle = Triangle of the logical view of `a` to read.
    diagonal = Whether diagonal entries are stored or implicitly one.
    a = Square triangular matrix.
    b = Matrix or column vector of the same element type as `a`.
    c = Writable output slice of the same element type and shape as `b`.
Preconditions:
    `a` must be square and its order must equal the first dimension of `b`.
    Output must not overlap either input or contain overlapping elements.
    Shapes are checked with assertions; overlap is not checked.

Every output element is overwritten, regardless of its previous value. General
strides are supported. Computed input iterators can affect inferred attributes.
+/
package(mir.stat) void mtimesTriangularInto(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B, C)(scope auto ref A a,
    scope auto ref B b, scope auto ref C c)
    if (compatible!(A, B) && numericSlice!C && C.N == B.N && is(Scalar!A == Scalar!C))
{
    checkTriangularDimensions!false(a, b);
    assert(c.shape == b.shape, "Incorrect triangular product output shape");
    alias T = Scalar!A;
    if (!a.length!0)
        return;
    static if (B.N == 2)
        if (!b.length!1)
            return;

    import mir.stat.internal.linearAlgebra.packed: tryPackedTriangular;
    if (tryPackedTriangular!(triangle, diagonal)(a, b, c)) return;

    static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!C)
    {
        static if (C.N == 2)
            bool canUseBlas = blasMatrix(a) && blasMatrix(c);
        else
            bool canUseBlas = blasMatrix(a) && blasVector(c);
        if (canUseBlas)
        {
            // TRMM/TRMV overwrite the non-triangular operand. Copy it directly
            // into the final output; no temporary matrix or vector is needed.
            static if (B.N == 2)
            {
                // Follow contiguous output storage. In particular, right-side
                // products use transposed views with contiguous columns.
                if (c._stride!0 == 1 && c._stride!1 != 1)
                {
                    foreach (j; 0 .. b.length!1)
                        foreach (i; 0 .. b.length!0)
                            c[i, j] = b[i, j];
                }
                else
                {
                    foreach (i; 0 .. b.length!0)
                        foreach (j; 0 .. b.length!1)
                            c[i, j] = b[i, j];
                }
            }
            else
                foreach (i; 0 .. b.length!0)
                    c[i] = b[i];
            multiplyTriangularBlas!(triangle, diagonal)(a, c);
            return;
        }
    }

    import mir.stat.internal.linearAlgebra.packed: tryPackedTriangularNative;
    if (tryPackedTriangularNative!(triangle,diagonal)(a,b,c)) return;

    foreach (i; 0 .. a.length!0)
    {
        static if (triangle == TrianglePolicy.upper)
        {
            auto begin = i;
            auto end = a.length!0;
        }
        else
        {
            size_t begin = 0;
            auto end = i + 1;
        }
        static if (B.N == 2)
        {
            foreach (j; 0 .. b.length!1)
            {
                T sum = T(0);
                foreach (k; begin .. end)
                {
                    static if (diagonal == DiagonalPolicy.unit)
                        if (k == i) { sum += b[k, j]; continue; }
                    sum += a[i, k] * b[k, j];
                }
                c[i, j] = sum;
            }
        }
        else
        {
            T sum = T(0);
            foreach (k; begin .. end)
            {
                static if (diagonal == DiagonalPolicy.unit)
                    if (k == i) { sum += b[k]; continue; }
                sum += a[i, k] * b[k];
            }
            c[i] = sum;
        }
    }
}

/// Use a preallocated output vector.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    double[2] output;
    mtimesTriangularInto(values[].sliced(2, 2), vector[].sliced, output[].sliced);
    assert(output == [28, 24]);
}

/++
Multiply a matrix or row vector `a` by a triangular matrix `b`.

Params:
    triangle = Triangle of the logical view of `b` to read.
    diagonal = Whether diagonal entries are stored or implicitly one.
    a = Matrix or row vector with floating-point or complex elements.
    b = Square triangular matrix with the same element type as `a`.
Returns:
    A GC-backed slice with the same shape as `a`.
Preconditions:
    The last dimension of `a` must equal the order of square matrix `b`.

The unused triangle and implicit unit diagonal are never read. Transposed
views are supported without conjugation or materialization.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, A.N)
mtimesTriangularRight(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(B, A))
{
    int unused;
    return multiplyTriangularWithAllocation!(Allocation.gc, triangle, diagonal, true)(unused, b, a);
}

/// A vector on the left is a row vector.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    assert(mtimesTriangularRight(vector[].sliced, values[].sliced(2, 2)) == [10, 39]);
}

/++
RC variant of $(LINK2 #mtimesTriangularRight, mtimesTriangularRight).
Parameters and preconditions are the same; the result is an owning RC slice
with the shape of `a`, requiring no manual disposal.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, A.N)
rcMtimesTriangularRight(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(B, A))
{
    int unused;
    return multiplyTriangularWithAllocation!(Allocation.rc, triangle, diagonal, true)(unused, b, a);
}

/// A transposed upper-triangular view is logically lower triangular.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    double[4] values = [double.nan, 3, double.nan, double.nan];
    double[2] vector = [5, 6];
    auto lower = values[].sliced(2, 2).universal.transposed;
    auto result = rcMtimesTriangularRight!(TrianglePolicy.lower, DiagonalPolicy.unit)(vector[].sliced, lower);
    assert(result == [23, 6]);
}

/++
Custom-allocator variant of $(LINK2 #mtimesTriangularRight, mtimesTriangularRight).
`allocator` supplies result storage; other parameters and preconditions are the
same. Dispose `result.field` with that allocator once all views are finished.
The result has the shape of `a`. Computation failure releases result storage.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, A.N)
makeMtimesTriangularRight(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, Allocator, A, B)(auto ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
    if (compatible!(B, A))
{
    return multiplyTriangularWithAllocation!(Allocation.custom, triangle, diagonal, true)(allocator, b, a);
}

/// Right multiplication with custom result storage.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    auto result = makeMtimesTriangularRight(Mallocator.instance,
        vector[].sliced, values[].sliced(2, 2));
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [10, 39]);
}

/++
Write `a * b` into caller storage, where `b` is triangular.

`triangle`, `diagonal`, `a` and `b` follow
$(LINK2 #mtimesTriangularRight, mtimesTriangularRight). Output `c` must have
the shape and element type of `a`, be writable, and overlap neither input.
Its elements must not overlap each other. No allocation is performed; all
output elements are overwritten. Shape assertions do not check overlap.
+/
package(mir.stat) void mtimesTriangularRightInto(TrianglePolicy triangle = TrianglePolicy.upper,
    DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A, B, C)(scope auto ref A a,
    scope auto ref B b, scope auto ref C c)
    if (compatible!(B, A) && numericSlice!C && C.N == A.N && is(Scalar!B == Scalar!C))
{
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    checkTriangularDimensions!true(b, a);
    assert(c.shape == a.shape, "Incorrect triangular product output shape");
    auto bt = b.universal.transposed;
    static if (A.N == 2)
    {
        auto at = a.universal.transposed;
        auto ct = c.universal.transposed;
        mtimesTriangularInto!(opposite!triangle, diagonal)(bt, at, ct);
    }
    else
        mtimesTriangularInto!(opposite!triangle, diagonal)(bt, a, c);
}

/// Preserve input storage and overwrite a separate output buffer.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] values = [2, 3, double.nan, 4];
    double[2] vector = [5, 6];
    double[2] output;
    mtimesTriangularRightInto(vector[].sliced, values[].sliced(2, 2), output[].sliced);
    assert(output == [10, 39]);
    assert(vector == [5, 6]);
}

/// Assert a square triangular operand and compatible left- or right-product dimensions.
private void checkTriangularDimensions(bool right, A, B)(scope auto ref A a, scope auto ref B b)
{
    assert(a.length!0 == a.length!1, "Triangular matrix must be square");
    static if (right && B.N == 2)
        assert(a.length!0 == b.length!1, "Incompatible triangular product dimensions");
    else
        assert(a.length!0 == b.length!0, "Incompatible triangular product dimensions");
}

/// Allocate the final triangular-product result and release custom storage on failure.
private AllocatedSlice!(policy, Scalar!A, B.N)
multiplyTriangularWithAllocation(Allocation policy, TrianglePolicy triangle,
    DiagonalPolicy diagonal, bool right, Allocator, A, B)(ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
{
    checkTriangularDimensions!right(a, b);
    auto result = allocate!(policy, Scalar!A)(allocator, b.shape);
    scope(failure)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(result.field);
        }
    }
    static if (right)
        mtimesTriangularRightInto!(triangle, diagonal)(b, a, result);
    else
        mtimesTriangularInto!(triangle, diagonal)(a, b, result);
    return result;
}

// Layout and size checks are performed by the caller. mir.blas handles the
// conversion from logical triangles to the native BLAS storage convention.
/// Overwrite eligible native output with TRMV or TRMM after the caller has copied the other operand.
private void multiplyTriangularBlas(TrianglePolicy triangle, DiagonalPolicy diagonal, A, C)(
    scope auto ref A a, scope auto ref C c)
{
    import mir.blas: trmm, trmv;
    import cblas: Side, Uplo, Diag;
    alias T = Scalar!A;
    auto ap = a.lightScope.lightConst;
    auto cp = c.lightScope;
    enum uplo = triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
    enum diag = diagonal == DiagonalPolicy.unit ? Diag.Unit : Diag.NonUnit;
    static if (C.N == 1)
        trmv(uplo, diag, ap, cp);
    else
        trmm(Side.Left, uplo, diag, T(1), ap, cp);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.products: rcMtimes;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: rcslice;
    // Compare all policies with explicit dense matrices. Every ignored value
    // is NaN, including the stored diagonal for the unit-diagonal policy.
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    static foreach (diagonal; [DiagonalPolicy.nonUnit, DiagonalPolicy.unit])
    {{
        double[9] packed, denseStorage;
        denseStorage[] = 0;
        auto a = packed[].sliced(3, 3);
        auto dense = denseStorage[].sliced(3, 3);
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3)
            {
                if (i == j && diagonal == DiagonalPolicy.unit)
                    dense[i, j] = 1;
                else if (triangle == TrianglePolicy.upper ? j >= i : j <= i)
                    a[i, j] = dense[i, j] = 1.0 + 3 * i + j;
            }
        double[6] bData = [1, 2, 3, 4, 5, 6];
        auto b = bData[].sliced(3, 2);
        auto leftOperand = bData[].sliced(2, 3);
        auto expectedLeft = rcMtimes(dense, b);
        auto expectedRight = rcMtimes(leftOperand, dense);
        assert(rcMtimesTriangular!(triangle, diagonal)(a, b) == expectedLeft);
        assert(rcMtimesTriangularRight!(triangle, diagonal)(leftOperand, a) == expectedRight);
        // Const RC triangular input and a borrowed operand keep the same API.
        const owned = a.rcslice;
        assert(rcMtimesTriangular!(triangle, diagonal)(owned, b) == expectedLeft);

        auto at = a.universal.transposed;
        auto dt = dense.universal.transposed;
        assert(rcMtimesTriangular!(opposite!triangle, diagonal)(at, b) == rcMtimes(dt, b));
        assert(rcMtimesTriangularRight!(opposite!triangle, diagonal)(leftOperand, at) == rcMtimes(leftOperand, dt));
        double[6] outputStorage;
        auto output = outputStorage[].sliced(2, 3).universal.transposed;
        mtimesTriangularInto!(triangle, diagonal)(a, b, output);
        assert(output == expectedLeft);
        mtimesTriangularInto!(opposite!triangle, diagonal)(at, b, output);
        assert(output == rcMtimes(dt, b));
        auto rightOutput = outputStorage[].sliced(3, 2).universal.transposed;
        mtimesTriangularRightInto!(triangle, diagonal)(leftOperand, a, rightOutput);
        assert(rightOutput == expectedRight);
        mtimesTriangularRightInto!(opposite!triangle, diagonal)(leftOperand, at, rightOutput);
        assert(rightOutput == rcMtimes(leftOperand, dt));

        double[36] sparseStorage;
        auto sparse = sparseStorage[].sliced(6, 6).strided!(0, 1)(2, 2);
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3)
                sparse[i, j] = a[i, j];
        double[24] sparseOutputStorage;
        sparseOutputStorage[] = -99;
        auto sparseOutput = sparseOutputStorage[].sliced(6, 4).strided!(0, 1)(2, 2);
        mtimesTriangularInto!(triangle, diagonal)(sparse, b, sparseOutput);
        assert(sparseOutput == expectedLeft && sparseOutputStorage[1] == -99);
        assert(rcMtimesTriangular!(triangle, diagonal)(a.map!"a", b.map!"a") == expectedLeft);
        assert(rcMtimesTriangular!(triangle, diagonal)(a, b.map!"a") == expectedLeft);
        assert(rcMtimesTriangularRight!(triangle, diagonal)(leftOperand.map!"a", a.map!"a") == expectedRight);
        // Reversing both axes swaps the selected triangle; all strides are negative.
        auto reversedA = a.reversed!0.reversed!1;
        auto reversedDense = dense.reversed!0.reversed!1;
        assert(rcMtimesTriangular!(opposite!triangle, diagonal)(reversedA, b) == rcMtimes(reversedDense, b));
        auto reversedOutput = outputStorage[].sliced(3, 2).reversed!0.reversed!1;
        mtimesTriangularInto!(triangle, diagonal)(a, b, reversedOutput);
        assert(reversedOutput == expectedLeft);

        // Padded rows retain a unit inner stride and exercise BLAS lda/ldb.
        double[15] paddedStorage;
        auto paddedA = paddedStorage[].sliced(3, 5)[0 .. 3, 0 .. 3];
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3)
                paddedA[i, j] = a[i, j];
        double[12] paddedOutputStorage;
        paddedOutputStorage[] = -99;
        auto paddedOutput = paddedOutputStorage[].sliced(3, 4)[0 .. 3, 0 .. 2];
        mtimesTriangularInto!(triangle, diagonal)(paddedA, b, paddedOutput);
        assert(paddedOutput == expectedLeft && paddedOutputStorage[2] == -99);

        double[3] vectorData = [2, 3, 4];
        auto v = vectorData[].sliced;
        assert(rcMtimesTriangular!(triangle, diagonal)(a, v) == rcMtimes(dense, v));
        assert(rcMtimesTriangularRight!(triangle, diagonal)(v, a) == rcMtimes(v, dense));
        assert(rcMtimesTriangular!(opposite!triangle, diagonal)(at, v) == rcMtimes(dt, v));
        assert(rcMtimesTriangularRight!(opposite!triangle, diagonal)(v, at) == rcMtimes(v, dt));
        double[6] vectorOutput;
        vectorOutput[] = -99;
        auto stridedOutput = vectorOutput[].sliced.strided!0(2);
        mtimesTriangularInto!(triangle, diagonal)(a, v, stridedOutput);
        assert(stridedOutput == rcMtimes(dense, v) && vectorOutput[1] == -99);
        mtimesTriangularRightInto!(triangle, diagonal)(v, a, stridedOutput);
        assert(stridedOutput == rcMtimes(v, dense));
    }}
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.products: rcMtimes;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.complex: Complex;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Complex!float, Complex!double))
    {{
        T[4] values = [T(2), T(3), T.init, T(4)];
        T[4] denseValues = [T(2), T(3), T(0), T(4)];
        T[4] otherValues = [T(1), T(2), T(3), T(4)];
        static if (is(T == Complex!float) || is(T == Complex!double))
        {
            values[0] = denseValues[0] = T(2, 1);
            values[1] = denseValues[1] = T(3, -2);
            otherValues[2] = T(3, 1);
        }
        auto a = values[].sliced(2, 2);
        auto dense = denseValues[].sliced(2, 2);
        auto b = otherValues[].sliced(2, 2);
        assert(rcMtimesTriangular(a, b) == rcMtimes(dense, b));
        assert(rcMtimesTriangularRight(b, a) == rcMtimes(b, dense));
        assert(rcMtimesTriangular!(TrianglePolicy.lower)(a.universal.transposed, b) == rcMtimes(dense.universal.transposed, b));
        auto v = otherValues[0 .. 2].sliced;
        assert(rcMtimesTriangular(a, v) == rcMtimes(dense, v));
        assert(rcMtimesTriangularRight(v, a) == rcMtimes(v, dense));
        auto empty = mininitRcslice!T(0, 0);
        auto noRows = mininitRcslice!T(0, 3);
        auto noColumns = mininitRcslice!T(3, 0);
        assert(rcMtimesTriangular(empty, noRows).shape == [0, 3]);
        assert(rcMtimesTriangularRight(noColumns, empty).shape == [3, 0]);
        auto emptyVector = mininitRcslice!T(0);
        assert(rcMtimesTriangular(empty, emptyVector).length == 0);
        assert(rcMtimesTriangularRight(emptyVector, empty).length == 0);
        auto zeroColumns = mininitRcslice!T(2, 0);
        assert(rcMtimesTriangular(a, zeroColumns).shape == [2, 0]);
    }}
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import core.exception: AssertError;
    import std.exception: assertThrown;
    struct CountingAllocator
    {
        size_t allocations, releases;
        /// Allocate test storage while counting allocation attempts.
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        /// Release test storage while counting releases.
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[4] values = [2, 3, double.nan, 4];
    double[4] other = [1, 2, 3, 4];
    auto a = values[].sliced(2, 2);
    auto b = other[].sliced(2, 2);
    auto left = makeMtimesTriangular(allocator, a, b);
    assert(left == mtimesTriangular(a, b));
    allocator.dispose(left.field);
    auto right = makeMtimesTriangularRight(allocator, b, a);
    assert(right == mtimesTriangularRight(b, a));
    allocator.dispose(right.field);
    assert(allocator.allocations == 2 && allocator.releases == 2);
    auto invalid = other[].sliced(1, 4);
    assertThrown!AssertError(makeMtimesTriangular(allocator, invalid, b));
    assertThrown!AssertError(makeMtimesTriangularRight(allocator, invalid, a));
    assertThrown!AssertError(mtimesTriangularInto(a, b, invalid));
    assertThrown!AssertError(mtimesTriangularRightInto(b, a, invalid));
    assert(allocator.allocations == 2);

    auto failure = new Exception("input failed");
    /// Throw the test exception to verify cleanup after an input iterator fails.
    double fail(double value) { throw failure; }
    auto failing = a.map!fail;
    foreach (rightSide; [false, true])
    {
        bool caught;
        try
        {
            if (rightSide) makeMtimesTriangularRight(allocator, b, failing);
            else makeMtimesTriangular(allocator, failing, b);
        }
        catch (Exception error) { assert(error is failure); caught = true; }
        assert(caught);
    }
    assert(allocator.allocations == 4 && allocator.releases == 4);
    // With a native triangular input, BLAS first copies the other operand.
    // Failure during that copy must release the same final output allocation.
    auto failingOther = b.map!fail;
    assertThrown!Exception(makeMtimesTriangular(allocator, a, failingOther));
    assertThrown!Exception(makeMtimesTriangularRight(allocator, failingOther, a));
    assert(allocator.allocations == 6 && allocator.releases == 6);
    // A computed iterator must never be invoked on unused entries.
    /// Fail the test if an operation reads an ignored triangular entry.
    double rejectNaN(double value)
    {
        assert(value == value, "Read an ignored triangular entry");
        return value;
    }
    values[0] = values[3] = double.nan;
    auto checked = a.map!rejectNaN;
    auto unitResult = mtimesTriangular!(TrianglePolicy.upper, DiagonalPolicy.unit)(checked, b);
    assert(unitResult == [[10, 14], [3, 4]]);
    auto rightUnit = mtimesTriangularRight!(TrianglePolicy.upper, DiagonalPolicy.unit)(b, checked);
    assert(rightUnit == [[1, 5], [3, 13]]);
}

// Callers establish square shape, matching vector length, writable disjoint
// vector elements, and a nonoverlapping read-only triangular matrix.
/// Solve or multiply a triangular system in dependency-safe order, using BLAS when eligible; callers validate shapes and aliasing.
package(mir.stat) void triangularVectorInPlace(bool solve, TrianglePolicy triangle, A, X)(
    scope auto ref A a, scope auto ref X x)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    if (!x.length) return;
    static if (hasBlas && nativeSlice!A && nativeSlice!X &&
        (is(Scalar!A == float) || is(Scalar!A == double)))
    {
        if (blasMatrix(a) && blasVector(x))
        {
            import mir.blas: trsv, trmv, Uplo;
            import cblas: Diag;
            auto ap = a.lightScope.lightConst;
            auto xp = x.lightScope;
            enum uplo = triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
            static if (solve) trsv(uplo, Diag.NonUnit, ap, xp);
            else trmv(uplo, Diag.NonUnit, ap, xp);
            return;
        }
    }
    // Solve in dependency order; multiply in the opposite order to preserve
    // every source element until its last use. No temporary vector is needed.
    enum reverse = solve ? triangle == TrianglePolicy.upper : triangle == TrianglePolicy.lower;
    foreach (step; 0 .. x.length)
    {
        auto i = reverse ? x.length - 1 - step : step;
        auto value = solve ? x[i] : Scalar!A(0);
        auto first = triangle == TrianglePolicy.upper ? i + (solve ? 1 : 0) : 0;
        auto last = triangle == TrianglePolicy.upper ? x.length : i + (solve ? 0 : 1);
        foreach (k; first .. last)
            static if (solve) value -= a[i,k] * x[k];
            else value += a[i,k] * x[k];
        static if (solve) x[i] = value / a[i,i];
        else x[i] = value;
    }
}


/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, PackedInfo!S.Element, B.N)
mtimesTriangular(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B)(scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return mtimesTriangular!(PackedInfo!S.triangle, diagonal)(packedMatrix(a), b);
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, PackedInfo!S.Element, B.N)
rcMtimesTriangular(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B)(scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return rcMtimesTriangular!(PackedInfo!S.triangle, diagonal)(packedMatrix(a), b);
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, PackedInfo!S.Element, B.N)
makeMtimesTriangular(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B, Allocator)(auto ref Allocator allocator, scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return makeMtimesTriangular!(PackedInfo!S.triangle, diagonal)(allocator, packedMatrix(a), b);
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) void
mtimesTriangularInto(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B, C)(scope auto ref S a, scope auto ref B b, scope auto ref C c)
    if (isPacked!S && numericSlice!B)
{
    mtimesTriangularInto!(PackedInfo!S.triangle, diagonal)(packedMatrix(a), b, c);
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, PackedInfo!S.Element, B.N)
mtimesTriangularRight(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B)(scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return mtimesTriangularRight!(PackedInfo!S.triangle, diagonal)(b, packedMatrix(a));
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, PackedInfo!S.Element, B.N)
rcMtimesTriangularRight(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B)(scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return rcMtimesTriangularRight!(PackedInfo!S.triangle, diagonal)(b, packedMatrix(a));
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, PackedInfo!S.Element, B.N)
makeMtimesTriangularRight(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B, Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return makeMtimesTriangularRight!(PackedInfo!S.triangle, diagonal)(allocator, b, packedMatrix(a));
}

/++
Packed-stairs overload. Infer the triangle from the stairs direction and select
the diagonal policy independently. Ownership and output requirements match
the dense overload; packed storage is never expanded into a dense temporary.
+/
package(mir.stat) void
mtimesTriangularRightInto(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S, B, C)(scope auto ref B b, scope auto ref S a, scope auto ref C c)
    if (isPacked!S && numericSlice!B)
{
    mtimesTriangularRightInto!(PackedInfo!S.triangle, diagonal)(b, packedMatrix(a), c);
}

/// Apply one triangular solve or multiplication to all columns in place.
/// Callers establish square shape, matching rows, and disjoint writable storage.
package(mir.stat) void triangularMatrixInPlace(bool solve, TrianglePolicy triangle, A, X)(
    scope auto ref A a, scope auto ref X x)
{
    if (!x.length!0 || !x.length!1) return;
    static if (hasBlas && nativeSlice!A && nativeSlice!X &&
        (is(Scalar!A == float) || is(Scalar!A == double)))
    {
        if (blasMatrix(a) && blasMatrix(x))
        {
            import mir.blas: trsm, trmm;
            import cblas: Side, Uplo, Diag;
            enum uplo = triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
            static if (solve) trsm(Side.Left, uplo, Diag.NonUnit, Scalar!A(1), a.lightScope.lightConst, x.lightScope);
            else trmm(Side.Left, uplo, Diag.NonUnit, Scalar!A(1), a.lightScope.lightConst, x.lightScope);
            return;
        }
    }
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    auto columns = x.universal.transposed;
    foreach (j; 0 .. columns.length) triangularVectorInPlace!(solve, triangle)(a, columns[j]);
}
