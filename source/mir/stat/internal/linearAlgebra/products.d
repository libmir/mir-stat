/++
Matrix products with explicit result ownership. `mtimes` uses GC storage,
`rcMtimes` uses RC storage, and `makeMtimes` uses the supplied allocator.
`dotProduct` returns a scalar without allocation. `mtimes` requires at least
one matrix operand. Products do not conjugate complex inputs.
Inputs may be borrowed or RC slices, independently
of output ownership. General strides use an allocation-free native D kernel.
When BLAS is available, compatible single/double precision real and complex
layouts use BLAS.

Inputs must have the same floating-point or complex element type and conforming
dimensions. One-dimensional inputs represent a row vector on the left and a
column vector on the right. Invalid shapes violate the function preconditions.
Only result storage is allocated; input views are never materialized.

Pass `triangularView(a)` on either side of `mtimes` to select triangular
multiplication automatically. The same dispatch is available through `rcMtimes`,
`makeMtimes`, and `mtimesInto`. The view carries the triangle and diagonal policy;
the other operand must be an ordinary ndslice. Symmetric views work on either
side as well, selecting SYMM/SYMV/SPMV where eligible without materialization.

A `diagonalView(vector)` scales rows on the left or columns on the right.
It accepts ordinary, triangular, symmetric, or diagonal operands without
materializing an input. Results retain the ordinary dense slice contract.
All allocation families and caller-output variants share this dispatch.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.products;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;


import mir.stat.internal.linearAlgebra.views: diagonalView, DiagonalView, DiagonalInfo, logicalOperand;
import mir.stat.internal.linearAlgebra.types: Scalar, numericSlice, TrianglePolicy, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.backend: hasBlas, blasNumber, nativeSlice, blasMatrix, blasVector;
import mir.stat.internal.linearAlgebra.views: TriangularView, triangularView, SymmetricView, symmetricView;
import mir.stat.internal.linearAlgebra.packed: isPacked, PackedInfo;
import mir.stat.internal.linearAlgebra.allocation;

/// Whether operands are supported numeric slices with matching element types.
private enum compatible(A, B) = numericSlice!A && numericSlice!B &&
    is(Scalar!A == Scalar!B);

/// Matrix products have the contracted rank and explicit result ownership.
private alias ProductResult(Allocation policy, A, B) =
    AllocatedSlice!(policy, Scalar!A, A.N + B.N - 2);

/++
Return the scalar sum of corresponding vector products without allocating.
Both operands must be rank-one numeric ndslices with the same element type
and length. Empty vectors return zero. Complex inputs are not conjugated:
this is a bilinear dot product, not a Hermitian inner product.

Borrowed, owning, const, strided, reversed, and computed views are supported.
Compatible float/double storage uses BLAS when available; other inputs use
a native D fallback. Summation is not compensated, and rounding may depend
on the backend. Computed iterators determine the inferred function attributes.

For independent row/column products, select vectors with `byDim` or `alongDim`
and map this function. The map is lazy: choose `.slice`, `.rcslice`,
`makeSlice`, or assignment to an existing slice to control result storage.
There are no allocation-specific dot-product variants or implicit broadcasting.
+/
package(mir.stat) Scalar!A dotProduct(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B) && A.N == 1 && B.N == 1)
{
    assert(a.length == b.length, "Dot-product vector lengths must match");
    alias T = Scalar!A;
    static if (hasBlas && (is(T == float) || is(T == double)) && nativeSlice!A && nativeSlice!B)
    {
        if (a.length && blasVector(a) && blasVector(b))
        {
            import mir.blas: dot;
            return dot(a.lightScope.lightConst, b.lightScope.lightConst);
        }
    }
    T result = T(0);
    foreach (i; 0 .. a.length)
        result += a[i] * b[i];
    return result;
}

/++
Return the sum of vector element squares without allocating.
Equivalent to `dotProduct(x, x)`, including its backend and rounding behavior.
Complex elements are squared without conjugation; this is not a squared norm.
+/
package(mir.stat) Scalar!A dotProduct(A)(scope auto ref A a)
    if (numericSlice!A && A.N == 1)
{
    return dotProduct(a, a);
}

/// Scalar dot products and explicit matrix shapes distinguish inner and outer products.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[3] xs = [1, 2, 3], ys = [4, 5, 6];
    auto x = xs[].sliced;
    auto y = ys[].sliced;
    assert(dotProduct(x, y) == 32);
    assert(dotProduct(x) == 14);
    static assert(is(typeof(dotProduct(x, y)) == double));
    static assert(!__traits(compiles, mtimes(x, y)));
    static assert(!__traits(compiles, rcMtimes(x, y)));
    double[9] output;
    auto outer = output[].sliced(3, 3);
    mtimesInto(xs[].sliced(3, 1), ys[].sliced(1, 3), outer);
    static immutable expected = [[4.0, 5, 6], [8.0, 10, 12], [12.0, 15, 18]];
    assert(outer == expected);
}

/// Map corresponding rows or columns lazily, then choose GC or RC storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim, alongDim, zip, map;
    import mir.ndslice.allocation: slice, rcslice;
    double[6] xs = [1, 2, 3, 4, 5, 6], ys = [7, 8, 9, 10, 11, 12];
    auto a = xs[].sliced(2, 3);
    auto b = ys[].sliced(2, 3);
    // Require equal matrix shapes when pairing all corresponding vectors.
    assert(a.shape == b.shape);
    auto rows = zip(a.byDim!0, b.byDim!0).map!((x, y) => dotProduct(x, y));
    auto gcRows = rows.slice;
    assert(gcRows == [50, 167]);
    auto columns = zip(a.alongDim!0, b.alongDim!0).map!((x, y) => dotProduct(x, y));
    auto rcColumns = columns.rcslice;
    assert(rcColumns == [47, 71, 99]);
    // Self-products use the same selection convention, without zip.
    assert(a.alongDim!1.map!(row => dotProduct(row)).slice == [14, 77]);
    assert(a.byDim!1.map!(column => dotProduct(column)).slice == [17, 29, 45]);
    // Lazy values observe source changes; already materialized results do not.
    xs[0] = 2;
    assert(rows[0] == 57 && gcRows[0] == 50);
}

/// Materialize independent reductions in custom or caller-owned storage.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: byDim, map;
    import mir.ndslice.allocation: makeSlice;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[6] xs = [1, 2, 3, 4, 5, 6];
    auto values = xs[].sliced(2, 3).byDim!0.map!(row => dotProduct(row));
    auto owned = makeSlice(Mallocator.instance, values);
    scope(exit) Mallocator.instance.dispose(owned.field);
    assert(owned == [14, 77]);
    double[2] buffer;
    auto output = buffer[].sliced;
    output[] = values;
    assert(output == [14, 77]);
}

/++
Multiply two matrices or vectors, allocating a GC-managed result when needed.

This is matrix multiplication, not element-wise multiplication. For shapes
`(m, k)` and `(k, n)`, the result has shape `(m, n)`. A matrix and a vector
produce a vector of length `m`; a vector and a matrix produce a vector of
length `n`. Two vectors are rejected; use `dotProduct(a, b)` for a scalar
or reshape to `(m, 1)` and `(1, n)` for an outer-product matrix.
Complex inputs are multiplied without conjugation.

Use `mtimes(triangularView(a), b)` or `mtimes(b, triangularView(a))` for a
triangular operand. These overloads select the left or right triangular product
without requiring a different function name. Unused entries are not read.

Params:
    a = Left operand, a rank-one or rank-two ndslice.
    b = Right operand with the same floating-point or complex element type.
Returns:
    A GC-backed slice. The result element
    type is the unqualified input element type.
Preconditions:
    The inner dimensions must match. Empty inner dimensions yield zeros.
See_Also: $(LINK2 #rcMtimes, rcMtimes), $(LINK2 #makeMtimes, makeMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.gc, A, B)
mtimes(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B) && (A.N == 2 || B.N == 2))
{
    int unused;
    return multiplyWithAllocation!(Allocation.gc)(unused, a, b);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, B.N)
mtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B)(
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a,
    scope auto ref B b)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: mtimesTriangular;
    return mtimesTriangular!(triangle, diagonal)(a.storage, b);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, B.N)
mtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B)(
    scope auto ref B b,
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: mtimesTriangularRight;
    return mtimesTriangularRight!(triangle, diagonal)(b, a.storage);
}

/// Matrix multiplication and matrix-vector multiplication with GC results.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[6] left = [1, 2, 3, 4, 5, 6];
    double[6] right = [7, 8, 9, 10, 11, 12];
    auto a = left[].sliced(2, 3);
    auto b = right[].sliced(3, 2);
    assert(mtimes(a, b) == [[58, 64], [139, 154]]);
    double[3] data = [1, 2, 3];
    auto v = data[].sliced;
    assert(mtimes(a, v) == [14, 32]);
    assert(mtimes(v, b) == [58, 64]);
    assert(dotProduct(v, v) == 14);
}

/// Declare matrix structure once and use ordinary multiplication syntax on either side.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;

    double[4] data = [2, 3, double.nan, 4];
    double[4] values = [1, 2, 3, 4];
    auto upper = triangularView!(TrianglePolicy.upper)(data[].sliced(2, 2));
    auto b = values[].sliced(2, 2);
    // The NaN below the selected triangle is ignored.
    assert(mtimes(upper, b) == [[11, 16], [12, 16]]);
    assert(mtimes(b, upper) == [[2, 11], [6, 25]]);

    // Transposing the view also changes its triangle policy to lower.
    assert(mtimes(upper.transposed, b) == [[2, 4], [15, 22]]);
}

/++
Multiply two matrices or vectors with reference-counted result storage.

Shapes and arithmetic follow $(LINK2 #mtimes, mtimes). Inputs may be borrowed or owning
slices; their ownership does not determine result ownership. The result keeps
its storage alive until the last owning reference is released. No manual
disposal is needed. Native numeric slices support `@safe pure nothrow @nogc`
calls; computed iterators can affect the inferred attributes.

Params:
    a = Left matrix or vector operand.
    b = Right operand of the same floating-point or complex element type.
Returns:
    An RC-backed slice.
Preconditions:
    The inner dimensions must match.
See_Also: $(LINK2 #mtimes, mtimes), $(LINK2 #makeMtimes, makeMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.rc, A, B)
rcMtimes(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B) && (A.N == 2 || B.N == 2))
{
    int unused;
    return multiplyWithAllocation!(Allocation.rc)(unused, a, b);
}

/// Multiplication dispatches from triangular metadata without extra options.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;
    double[4] data = [2, 3, double.nan, 4];
    double[2] values = [5, 6];
    const a = triangularView(data[].sliced(2, 2));
    assert(rcMtimes(a, values[].sliced) == [28, 24]);
    assert(rcMtimes(values[].sliced, a) == [10, 39]);

    // Unit-diagonal metadata replaces stored diagonal entries with ones.
    auto unit = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(data[].sliced(2, 2));
    assert(rcMtimes(unit, values[].sliced) == [23, 6]);
    assert(rcMtimes(values[].sliced, unit) == [5, 21]);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, B.N)
rcMtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B)(
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a,
    scope auto ref B b)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: rcMtimesTriangular;
    return rcMtimesTriangular!(triangle, diagonal)(a.storage, b);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, B.N)
rcMtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B)(
    scope auto ref B b,
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: rcMtimesTriangularRight;
    return rcMtimesTriangularRight!(triangle, diagonal)(b, a.storage);
}

/// Borrowed inputs can produce an owning result without using the GC.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.rc.array: RCI;
    import mir.ndslice.slice: Slice;
    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto result = rcMtimes(a, a);
    static assert(is(typeof(result) == Slice!(RCI!double, 2)));
    static immutable expected = [[7.0, 10], [15.0, 22]];
    assert(result == expected);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;
    auto owner = mininitRcslice!double(2,2);
    owner[0,0] = 2; owner[0,1] = 3;
    owner[1,0] = double.nan; owner[1,1] = 4;
    const constOwner = owner;
    auto view = triangularView(constOwner);
    double[2] values = [5,6];
    assert(rcMtimes(view, values[].sliced) == [28,24]);
}

/++
Multiply two matrices or vectors using a caller-supplied result allocator.

Shapes and arithmetic follow $(LINK2 #mtimes, mtimes). Only result storage is allocated;
no scratch buffers are needed. The caller owns the returned slice and must
dispose `result.field` with the same allocator after all views are finished.
If multiplication throws after allocation, the result storage is disposed
before the exception propagates. Attributes depend on the allocator and inputs.

Params:
    allocator = Allocator compatible with `makeUninitSlice` and `dispose`.
    a = Left matrix or vector operand.
    b = Right operand of the same floating-point or complex element type.
Returns:
    A slice backed by the supplied allocator.
Preconditions:
    The inner dimensions must match; they are checked before allocation.
Throws:
    `OutOfMemoryError` if the result element or byte count overflows `size_t`.
See_Also: $(LINK2 #mtimes, mtimes), $(LINK2 #rcMtimes, rcMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.custom, A, B)
makeMtimes(Allocator, A, B)(auto ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B) && (A.N == 2 || B.N == 2))
{
    return multiplyWithAllocation!(Allocation.custom)(allocator, a, b);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, B.N)
makeMtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, Allocator)(
    auto ref Allocator allocator,
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a,
    scope auto ref B b)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: makeMtimesTriangular;
    return makeMtimesTriangular!(triangle, diagonal)(allocator, a.storage, b);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, B.N)
makeMtimes(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, Allocator)(
    auto ref Allocator allocator,
    scope auto ref B b,
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: makeMtimesTriangularRight;
    return makeMtimesTriangularRight!(triangle, diagonal)(allocator, b, a.storage);
}

/// Custom storage is explicitly released with the allocator that created it.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto result = makeMtimes(Mallocator.instance, a, a);
    scope(exit) Mallocator.instance.dispose(result.field);
    static immutable expected = [[7.0, 10], [15.0, 22]];
    assert(result == expected);
}

/// Triangular dispatch also preserves custom allocation and disposal conventions.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;

    double[4] data = [2, 3, double.nan, 4];
    double[2] values = [5, 6];
    auto upper = triangularView(data[].sliced(2, 2));
    auto v = values[].sliced;
    auto left = makeMtimes(Mallocator.instance, upper, v);
    scope(exit) Mallocator.instance.dispose(left.field);
    auto right = makeMtimes(Mallocator.instance, v, upper);
    scope(exit) Mallocator.instance.dispose(right.field);
    assert(left == [28, 24]);
    assert(right == [10, 39]);
}

// Validate the contracted dimension before allocating or writing output.
/// Assert that the contracted dimensions agree before allocation or output writes.
private void checkMultiplicationDimensions(A, B)(scope auto ref A a, scope auto ref B b)
{
    static if (A.N == 2)
        assert(a.length!1 == b.length!0, "Incompatible product dimensions");
    else
        assert(a.length == b.length!0, "Incompatible product dimensions");
}

/// Allocate the product and release custom storage if computation throws.
private ProductResult!(policy, A, B)
multiplyWithAllocation(Allocation policy, Allocator, A, B)(ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
{
    checkMultiplicationDimensions(a, b);
    alias T = Scalar!A;
    static if (A.N == 2 && B.N == 2)
        size_t[2] shape = [a.length!0, b.length!1];
    else static if (A.N == 2)
        size_t[1] shape = [a.length!0];
    else
        size_t[1] shape = [b.length!1];
    auto result = allocate!(policy, T)(allocator, shape);
    static if (policy == Allocation.custom)
    {
        scope(failure)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(result.field);
        }
        mtimesInto(a, b, result);
        return result;
    }
    else
    {
        mtimesInto(a, b, result);
        return result;
    }
}

/++
Write a matrix or vector product into existing storage without allocating.

Shapes and arithmetic follow $(LINK2 #mtimes, mtimes). Two vector operands
are not supported: use $(LINK2 #dotProduct, dotProduct) for a scalar result.
Every output element is overwritten; its prior value is ignored. General
strided output is supported. Empty inner dimensions fill the output with zeros.

Params:
    a = Left matrix or vector operand.
    b = Right operand of the same floating-point or complex element type.
    c = Writable output slice with the same element type and product shape.
Preconditions:
    Inner dimensions must match. Output must have rank two for two matrix
    operands, and rank one otherwise. Output must not overlap either input,
    and its elements must not overlap each other. Shapes are checked with
    assertions; overlap is the caller's responsibility and is not checked.
See_Also: $(LINK2 #mtimes, mtimes), $(LINK2 #rcMtimes, rcMtimes), $(LINK2 #makeMtimes, makeMtimes)
+/
package(mir.stat) void mtimesInto(A, B, C)(scope auto ref A a, scope auto ref B b, scope auto ref C c)
    if (compatible!(A, B) && numericSlice!C && is(Scalar!A == Scalar!C) &&
        (A.N != 1 || B.N != 1) && C.N == (A.N == 2 && B.N == 2 ? 2 : 1))
{
    checkMultiplicationDimensions(a, b);
    alias T = Scalar!A;
    static if (A.N == 2 && B.N == 2)
    {
        assert(c.length!0 == a.length!0 && c.length!1 == b.length!1,
            "Incorrect product output shape");
        // Avoid BLAS zero-size/leading-dimension corner cases.
        static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!B && nativeSlice!C)
        {
            if (a.length!0 && a.length!1 && b.length!1 &&
                blasMatrix(a) && blasMatrix(b) && blasMatrix(c))
            {
                import mir.blas: gemm;
                gemm(T(1), a.lightScope.lightConst, b.lightScope.lightConst,
                    T(0), c.lightScope);
                return;
            }
        }
        foreach (i; 0 .. c.length!0)
            foreach (j; 0 .. c.length!1)
            {
                T sum = T(0);
                foreach (k; 0 .. a.length!1)
                    sum += a[i, k] * b[k, j];
                c[i, j] = sum;
            }
    }
    else static if (A.N == 2)
    {
        assert(c.length == a.length!0, "Incorrect product output length");
        static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!B && nativeSlice!C)
        {
            if (a.length!0 && a.length!1 && blasMatrix(a) &&
                blasVector(b) && blasVector(c))
            {
                import mir.blas: gemv;
                gemv(T(1), a.lightScope.lightConst, b.lightScope.lightConst,
                    T(0), c.lightScope);
                return;
            }
        }
        foreach (i; 0 .. c.length)
        {
            T sum = T(0);
            foreach (k; 0 .. b.length)
                sum += a[i, k] * b[k];
            c[i] = sum;
        }
    }
    else
    {
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: universal;
        auto bt = b.universal.transposed;
        mtimesInto(bt, a, c);
    }
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) void mtimesInto(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, C)(
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a,
    scope auto ref B b,
    scope auto ref C c)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: mtimesTriangularInto;
    mtimesTriangularInto!(triangle, diagonal)(a.storage, b, c);
}

/++
Triangular-view overload. Uses the view's triangle and diagonal policy without
materializing implicit entries. The other operand must be an ordinary ndslice.
Ownership and aliasing requirements follow the corresponding plain-slice API.
+/
package(mir.stat) void mtimesInto(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, C)(
    scope auto ref B b,
    scope auto ref const(TriangularView!(triangle, diagonal, A)) a,
    scope auto ref C c)
    if (numericSlice!B)
{
    import mir.stat.internal.linearAlgebra.triangular: mtimesTriangularRightInto;
    mtimesTriangularRightInto!(triangle, diagonal)(b, a.storage, c);
}

/// Reuse caller storage while selecting triangular multiplication from a view.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: triangularView;

    double[4] data = [2, 3, double.nan, 4];
    double[2] values = [5, 6];
    double[2] buffer;
    auto upper = triangularView(data[].sliced(2, 2));
    auto v = values[].sliced;
    auto output = buffer[].sliced;
    mtimesInto(upper, v, output);
    assert(output == [28, 24]);
    mtimesInto(v, upper, output);
    assert(output == [10, 39]);
}

/// Reuse an output buffer, including a transposed view of its storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    double[4] input = [1, 2, 3, 4];
    double[4] storage;
    auto a = input[].sliced(2, 2);
    auto output = storage[].sliced(2, 2).universal.transposed;
    mtimesInto(a, a, output);
    static immutable expected = [[7.0, 10], [15.0, 22]];
    assert(output == expected);
    assert(storage == [7, 15, 10, 22]);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    double[6] av = [1, 2, 3, 4, 5, 6];
    double[6] bv = [7, 8, 9, 10, 11, 12];
    auto a = av[].sliced(2, 3);
    auto b = bv[].sliced(3, 2);
    static immutable expected = [[58.0, 64], [139.0, 154]];
    static immutable gram = [[17.0, 22, 27], [22.0, 29, 36], [27.0, 36, 45]];
    static immutable zeros = [[0.0, 0, 0], [0.0, 0, 0]];
    auto c = rcMtimes(a, b);
    assert(c == expected);
    const owner = rcslice!double([3, 2], 1.0);
    // Const RC and borrowed inputs are accepted without changing result ownership.
    auto constProduct = rcMtimes(a, owner);
    assert(constProduct[0, 0] == 6 && constProduct[1, 0] == 15);
    double[4] storage;
    auto output = storage[].sliced(2, 2).universal.transposed;
    mtimesInto(a, b, output);
    assert(output == expected);
    auto at = a.universal.transposed;
    assert(rcMtimes(at, a) == gram);
    double[3] vector = [1, 2, 3];
    auto v = vector[].sliced;
    assert(rcMtimes(a, v) == [14, 32]);
    assert(rcMtimes(v, at) == [14, 32]);
    assert(dotProduct(v, v) == 14);
    auto emptyA = mininitRcslice!double(2, 0);
    auto emptyB = mininitRcslice!double(0, 3);
    assert(rcMtimes(emptyA, emptyB) == zeros);
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto gc = mtimes(a, a);
    auto custom = makeMtimes(Mallocator.instance, a, a);
    scope(exit) Mallocator.instance.dispose(custom.field);
    assert(gc == [[7, 10], [15, 22]] && custom == gc);
}

version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto result = makeMtimes(Mallocator.instance, a, a);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result[1, 1] == 22);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [1, 2, 3, 4];
    auto result = mtimes(data[].sliced(2, 2), data[].sliced(2, 2));
    assert(result[1, 1] == 22);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: strided;
    import mir.complex: Complex;
    double[16] storage;
    storage[] = -99;
    storage[0] = 1; storage[2] = 2; storage[8] = 3; storage[10] = 4;
    auto a = storage[].sliced(4, 4).strided!(0, 1)(2, 2);
    double[16] outStorage;
    outStorage[] = -99;
    auto c = outStorage[].sliced(4, 4).strided!(0, 1)(2, 2);
    mtimesInto(a, a, c);
    assert(c[0, 0] == 7 && c[0, 1] == 10 && c[1, 0] == 15 && c[1, 1] == 22);
    assert(outStorage[1] == -99 && outStorage[9] == -99);
    float[4] floats = [1, 2, 3, 4];
    auto f = floats[].sliced(2, 2);
    auto ff = rcMtimes(f, f);
    static assert(is(Scalar!(typeof(ff)) == float));
    assert(ff[1, 1] == 22);
    Complex!double[1] z = [Complex!double(1, 1)];
    auto zs = z[].sliced;
    // A bilinear product, not a conjugating inner product.
    assert(dotProduct(zs, zs) == Complex!double(0, 2));
    auto zm = z[].sliced(1, 1);
    assert(rcMtimes(zm, zm)[0, 0] == Complex!double(0, 2));
    assert(rcMtimes(zm, zs)[0] == Complex!double(0, 2));
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, reversed, strided;
    import mir.ndslice.allocation: mininitRcslice;
    double[6] av = [1, 2, 3, 4, 5, 6];
    double[6] bv = [7, 8, 9, 10, 11, 12];
    auto a = av[].sliced(2, 3);
    auto b = bv[].sliced(3, 2);
    auto at = a.universal.transposed;
    auto bt = b.universal.transposed;
    auto product = rcMtimes(a, b);
    // Both transposed inputs and a transposed, nonsquare matrix-vector input.
    assert(rcMtimes(bt, at) == product.universal.transposed);
    double[2] vv = [2, 3];
    auto v = vv[].sliced;
    auto transposedVectorProduct = rcMtimes(at, v);
    assert(transposedVectorProduct == [14, 19, 24]);
    assert(rcMtimes(v, a) == transposedVectorProduct);
    assert(rcMtimes(a.reversed!1, b.reversed!0) == product);
    assert(dotProduct(v.reversed!0, v) == 12);
    auto mapped = a.map!"a * 2";
    auto doubled = rcMtimes(mapped, b);
    assert(doubled == product.map!"a * 2");
    assert(dotProduct(v.map!"a * 2", v) == 26);
    double[4] stridedVector = [2, -99, 3, -99];
    auto sv = stridedVector[].sliced.strided!0(2);
    assert(rcMtimes(at, sv) == transposedVectorProduct);
    auto emptyRows = mininitRcslice!double(0, 3);
    assert(rcMtimes(emptyRows, b).shape == [0, 2]);
    auto emptyColumns = mininitRcslice!double(3, 0);
    assert(rcMtimes(a, emptyColumns).shape == [2, 0]);
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import core.exception: AssertError, OutOfMemoryError;
    import std.exception: assertThrown;
    struct CountingAllocator
    {
        size_t allocations, releases;
        /// Allocate test storage while counting allocation attempts.
        void[] allocate(size_t bytes)
        {
            ++allocations;
            return Mallocator.instance.allocate(bytes);
        }
        /// Release test storage while counting releases.
        bool deallocate(void[] memory)
        {
            ++releases;
            return Mallocator.instance.deallocate(memory);
        }
    }
    CountingAllocator allocator;
    double[4] data = [1, 2, 3, 4];
    auto a = data[].sliced(2, 2);
    auto result = makeMtimes(allocator, a, a);
    assert(allocator.allocations == 1 && allocator.releases == 0);
    allocator.dispose(result.field);
    assert(allocator.releases == 1);
    assert(dotProduct(data[].sliced) == 30);
    assert(allocator.allocations == 1); // scalar products allocate nothing
    auto bad = data[].sliced(1, 4);
    assertThrown!AssertError(makeMtimes(allocator, a, bad));
    assertThrown!AssertError(mtimesInto(a, a, bad));
    assert(allocator.allocations == 1); // validate dimensions before allocating
    size_t[2] overflow = [size_t.max, 2];
    assertThrown!OutOfMemoryError(allocate!(Allocation.custom, double)(allocator, overflow));
    assert(allocator.allocations == 1);
    size_t[1] byteOverflow = [size_t.max / double.sizeof + 1];
    assertThrown!OutOfMemoryError(allocate!(Allocation.custom, double)(allocator, byteOverflow));
    assert(allocator.allocations == 1);

    // A computed input can fail after result allocation. Release that storage
    // exactly once and propagate the original exception.
    import mir.ndslice.topology: map;
    auto failure = new Exception("computed input failed");
    /// Throw the test exception to verify cleanup after an input iterator fails.
    double fail(double value) { throw failure; }
    auto failingInput = a.map!fail;
    bool caught;
    try
        makeMtimes(allocator, failingInput, a);
    catch (Exception error)
    {
        assert(error is failure);
        caught = true;
    }
    assert(caught && allocator.allocations == 2 && allocator.releases == 2);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: mininitRcslice;
    // Empty contractions must overwrite existing NaNs, including vector output.
    auto a = mininitRcslice!double(2, 0);
    auto b = mininitRcslice!double(0, 2);
    double[4] storage; // D initializes floating-point values to NaN.
    auto output = storage[].sliced(2, 2);
    mtimesInto(a, b, output);
    assert(storage == [0, 0, 0, 0]);
    auto empty = mininitRcslice!double(0);
    double[2] vectorStorage;
    auto vectorOutput = vectorStorage[].sliced;
    mtimesInto(a, empty, vectorOutput);
    assert(vectorStorage == [0, 0]);
    assert(dotProduct(empty, empty) == 0);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.complex: Complex;
    alias C = Complex!double;
    C[4] av = [C(1, 1), C(2, -1), C(3, 2), C(-1, 1)];
    C[4] bv = [C(0, 2), C(1, 0), C(2, 0), C(0, -1)];
    auto a = av[].sliced(2, 2);
    auto b = bv[].sliced(2, 2);
    auto c = rcMtimes(a, b);
    assert(c[0, 0] == C(2, 0) && c[0, 1] == C(0, -1));
    assert(c[1, 0] == C(-6, 8) && c[1, 1] == C(4, 3));
    assert(rcMtimes(b.universal.transposed, a.universal.transposed) == c.universal.transposed);
}

// Structural triangular operands dispatch to the existing specialized kernels.


/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, PackedInfo!S.Element, B.N)
mtimes(S, B)(scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return mtimes(triangularView(a), b);
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, PackedInfo!S.Element, B.N)
mtimes(S, B)(scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return mtimes(b, triangularView(a));
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, PackedInfo!S.Element, B.N)
rcMtimes(S, B)(scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return rcMtimes(triangularView(a), b);
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, PackedInfo!S.Element, B.N)
rcMtimes(S, B)(scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return rcMtimes(b, triangularView(a));
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, PackedInfo!S.Element, B.N)
makeMtimes(S, B, Allocator)(auto ref Allocator allocator, scope auto ref S a, scope auto ref B b)
    if (isPacked!S && numericSlice!B)
{
    return makeMtimes(allocator, triangularView(a), b);
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, PackedInfo!S.Element, B.N)
makeMtimes(S, B, Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref S a)
    if (isPacked!S && numericSlice!B)
{
    return makeMtimes(allocator, b, triangularView(a));
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) void
mtimesInto(S, B, C)(scope auto ref S a, scope auto ref B b, scope auto ref C c)
    if (isPacked!S && numericSlice!B)
{
    mtimesInto(triangularView(a), b, c);
}

/++
Packed-stairs overload. Bare stairs represent a triangular matrix with stored
diagonal values. The direction selects the triangle; the other operand is an
ordinary ndslice. Use `triangularView` to select a unit diagonal or
`symmetricView` to give the same storage symmetric meaning.
+/
package(mir.stat) void
mtimesInto(S, B, C)(scope auto ref B b, scope auto ref S a, scope auto ref C c)
    if (isPacked!S && numericSlice!B)
{
    mtimesInto(b, triangularView(a), c);
}

/// Packed stairs can be passed directly or wrapped to select structural policies.
version(mir_stat_test_linearAlgebra)
@system pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;
    import mir.stat.internal.linearAlgebra.triangular: mtimesTriangular, mtimesTriangularRight;
    import mir.stat.internal.linearAlgebra.quadratic: quadraticForm, quadraticFormWithWorkspace;
    import mir.stat.internal.linearAlgebra.views: symmetricView;
    import mir.math.sum: Summation;
    double[3] data = [2, 3, 4];
    double[2] values = [5, 6], buffer;
    auto upper = data[].stairs!"-"(2);
    auto lower = data[].stairs!"+"(2);
    auto x = values[].sliced;
    assert(mtimes(upper, x) == [28, 24]);
    assert(mtimes(x, upper) == [10, 39]);
    assert(mtimes(lower, x) == [10, 39]);
    assert(mtimes(x, lower) == [28, 24]);
    assert(mtimes(triangularView(upper).transposed, x) == [10, 39]);
    assert(mtimes(triangularView!(DiagonalPolicy.unit)(upper), x) == [23, 6]);
    assert(mtimesTriangular(upper, x) == [28, 24]);
    assert(mtimesTriangularRight(x, upper) == [10, 39]);
    assert(quadraticForm(x, symmetricView(upper)) == 374);
    assert(quadraticFormWithWorkspace!(Summation.fast)(x, symmetricView(lower), buffer[].sliced) == 374);
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs, universal, map;
    import mir.ndslice.dynamic: transposed, reversed;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.internal.linearAlgebra.triangular;
    import mir.stat.internal.linearAlgebra.quadratic: quadraticForm, quadraticFormWithWorkspace;
    import mir.stat.internal.linearAlgebra.views: symmetricView;
    import mir.stat.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
    import mir.stat.internal.linearAlgebra.packed: packedMatrix, tryPackedTriangular, tryPackedSymmetric;
    import mir.stat.internal.linearAlgebra.backend: hasBlas;
    import mir.math.sum: Summation;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (direction; ["+", "-"])
    static foreach (diagonal; [DiagonalPolicy.nonUnit, DiagonalPolicy.unit])
    {{
        enum triangle = direction == "+" ? TrianglePolicy.lower : TrianglePolicy.upper;
        T[6] data = [2,3,4,5,6,7];
        auto rows = data[].stairs!direction(3);
        T[9] denseData;
        auto dense = denseData[].sliced(3,3);
        foreach (i; 0..3) foreach (j; 0..3)
        {
            if (i == j && diagonal == DiagonalPolicy.unit)
            {
                rows[i][direction == "+" ? j : 0] = T.nan;
                dense[i,j] = 1;
            }
            else if (direction == "+" ? j <= i : j >= i)
                dense[i,j] = rows[i][direction == "+" ? j : j-i];
            else dense[i,j] = 0;
        }
        T[6] other = [1,2,3,4,5,6], buffer;
        auto rhs = other[].sliced(3,2);
        auto lhs = other[].sliced(2,3);
        auto expectedLeft = rcMtimes(dense,rhs);
        auto expectedRight = rcMtimes(lhs,dense);
        auto view = triangularView!diagonal(rows);
        assert(rcMtimes(view,rhs) == expectedLeft);
        assert(rcMtimes(lhs,view) == expectedRight);
        assert(rcMtimes(view.transposed,rhs) == rcMtimes(dense.universal.transposed,rhs));
        assert(rcMtimesTriangular!diagonal(rows,rhs) == expectedLeft);
        assert(rcMtimesTriangularRight!diagonal(lhs,rows) == expectedRight);
        auto custom = makeMtimes(Mallocator.instance,view,rhs);
        scope(exit) Mallocator.instance.dispose(custom.field);
        assert(custom == expectedLeft);
        auto customRight = makeMtimesTriangularRight!diagonal(Mallocator.instance,lhs,rows);
        scope(exit) Mallocator.instance.dispose(customRight.field);
        assert(customRight == expectedRight);
        auto outLeft = buffer[].sliced(2,3).universal.transposed;
        mtimesInto(view,rhs,outLeft);
        assert(outLeft == expectedLeft);
        mtimesTriangularInto!diagonal(rows,rhs,outLeft);
        assert(outLeft == expectedLeft);
        mtimesTriangularRightInto!diagonal(lhs,rows,buffer[].sliced(2,3));
        assert(buffer[].sliced(2,3) == expectedRight);
        const owner = data[].sliced.rcslice;
        auto ownedRows = owner.lightConst.stairs!direction(3);
        const frozenRows = ownedRows;
        auto ownedView = triangularView!diagonal(frozenRows);
        assert(rcMtimes(ownedView,rhs) == expectedLeft);
        static if (diagonal == DiagonalPolicy.nonUnit)
        {
            assert(rcMtimes(rows,rhs) == expectedLeft);
            assert(rcMtimes(lhs,rows) == expectedRight);
            auto allocated = makeMtimes(Mallocator.instance,rows,rhs);
            scope(exit) Mallocator.instance.dispose(allocated.field);
            assert(allocated == expectedLeft);
            auto allocatedRight = makeMtimes(Mallocator.instance,lhs,rows);
            scope(exit) Mallocator.instance.dispose(allocatedRight.field);
            assert(allocatedRight == expectedRight);
            auto explicitCustom = makeMtimesTriangular(Mallocator.instance,rows,rhs);
            scope(exit) Mallocator.instance.dispose(explicitCustom.field);
            assert(explicitCustom == expectedLeft);
            mtimesInto(rows,rhs,outLeft);
            assert(outLeft == expectedLeft);
            mtimesInto(lhs,rows,buffer[].sliced(2,3));
            assert(buffer[].sliced(2,3) == expectedRight);
        }
        static if (hasBlas && !is(T == real))
        {
            auto a = packedMatrix(rows);
            assert((tryPackedTriangular!(triangle,diagonal)(a,rhs,outLeft)));
            assert(outLeft == expectedLeft);
        }
    }}
    // Packed symmetric views retain owners across factory returns and transpose.
    /// Return an owning packed symmetric view after the local owner is destroyed.
    auto retained()
    {
        double[6] data = [4,2,0,5,0,9];
        auto owner = data[].sliced.rcslice;
        return symmetricView(owner.stairs!"-"(3)).transposed;
    }
    auto sym = retained();
    double[3] values = [1,2,3], work;
    assert(quadraticForm(values[].sliced,sym) == 113);
    assert(quadraticFormWithWorkspace!(Summation.fast)(values[].sliced,sym,work[].sliced) == 113);
    auto factor = rcCholeskyDecompose(sym);
    assert(factor[0,0] == 2 && factor[1,0] == 1 && factor[1,1] == 2 && factor[2,2] == 3);
    double[0] empty;
    auto zero = empty[].stairs!"-"(0);
    assert(rcMtimes(zero,empty[].sliced).length == 0);
    double[3] raw = [2,3,4];
    auto packed = raw[].stairs!"-"(2);
    assertThrown!AssertError(triangularView(packed[0..1]));
    auto computed = raw[].sliced.map!(v => v).stairs!"-"(2);
    double[2] v = [5,6];
    assert(rcMtimes(computed,v[].sliced.reversed) == [27,20]);
    mtimesInto(packed,v[].sliced,work[0..2].sliced.reversed);
    assert(work[0..2] == [24,28]);
    double m = double.max/3;
    double[3] cancellation = [m,-m,m];
    double[2] twos = [2,2];
    auto cancellationView = symmetricView(cancellation[].stairs!"-"(2));
    assert(quadraticForm(twos[].sliced,cancellationView) == 0);
    assert(quadraticFormWithWorkspace(twos[].sliced,cancellationView,work[0..2].sliced) == 0);
    static if (hasBlas)
    {
        auto a = packedMatrix(raw[].stairs!"-"(2));
        assert(tryPackedSymmetric!(TrianglePolicy.upper)(a,v[].sliced,work[0..2].sliced));
        assert(work[0..2] == [28,39]);
    }
}

version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.complex: Complex;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;
    import mir.stat.internal.linearAlgebra.triangular: rcMtimesTriangular;
    // Packed complex multiplication is nonconjugating, including transpose.
    Complex!double[3] data = [Complex!double(1,1), Complex!double(2,-1), Complex!double(3,2)];
    Complex!double[2] values = [Complex!double(2,1), Complex!double(1,-1)];
    auto packed = data[].stairs!"-"(2);
    auto x = values[].sliced;
    auto result = rcMtimes(packed,x);
    assert(result[0] == data[0]*values[0] + data[1]*values[1]);
    assert(result[1] == data[2]*values[1]);
    auto transposed = rcMtimes(triangularView(packed).transposed,x);
    assert(transposed[0] == data[0]*values[0]);
    assert(transposed[1] == data[1]*values[0] + data[2]*values[1]);
    assert(rcMtimesTriangular(packed,x) == result);
}

/++
Multiply using a symmetric view on the left. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, B.N)
mtimes(TrianglePolicy triangle, A, B)(scope auto ref const(SymmetricView!(triangle, A)) a, scope auto ref B b)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    int allocator;
    auto result = allocate!(Allocation.gc, Scalar!A)(allocator, b.shape);
    mtimesInto(a, b, result);
    return result;
}

/// Declare symmetry once and multiply on either side using the ordinary API.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[4] values = [4, 2, double.nan, 5];
    double[6] rhs = [1, 2, 3, 4, 5, 6];
    double[2] vector = [1, 2];
    auto a = symmetricView(values[].sliced(2, 2));
    assert(mtimes(a, vector[].sliced) == [8, 12]);
    assert(mtimes(vector[].sliced, a) == [8, 12]);
    assert(mtimes(a, rhs[].sliced(2, 3)) == [[12, 18, 24], [22, 29, 36]]);
    assert(mtimes(rhs[].sliced(3, 2), a) == [[8, 12], [20, 26], [32, 40]]);
}

/++
Multiply using a symmetric view on the right. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, B.N)
mtimes(TrianglePolicy triangle, A, B)(scope auto ref B b, scope auto ref const(SymmetricView!(triangle, A)) a)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    int allocator;
    auto result = allocate!(Allocation.gc, Scalar!A)(allocator, b.shape);
    mtimesInto(b, a, result);
    return result;
}

/++
Multiply using a symmetric view on the left. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, B.N)
rcMtimes(TrianglePolicy triangle, A, B)(scope auto ref const(SymmetricView!(triangle, A)) a, scope auto ref B b)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    int allocator;
    auto result = allocate!(Allocation.rc, Scalar!A)(allocator, b.shape);
    mtimesInto(a, b, result);
    return result;
}

/++
Multiply using a symmetric view on the right. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, B.N)
rcMtimes(TrianglePolicy triangle, A, B)(scope auto ref B b, scope auto ref const(SymmetricView!(triangle, A)) a)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    int allocator;
    auto result = allocate!(Allocation.rc, Scalar!A)(allocator, b.shape);
    mtimesInto(b, a, result);
    return result;
}

/++
Multiply using a symmetric view on the left. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, B.N)
makeMtimes(TrianglePolicy triangle, A, B, Allocator)(auto ref Allocator allocator, scope auto ref const(SymmetricView!(triangle, A)) a, scope auto ref B b)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    auto result = allocate!(Allocation.custom, Scalar!A)(allocator, b.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    mtimesInto(a, b, result);
    return result;
}

/++
Multiply using a symmetric view on the right. Only its selected triangle is
read, without conjugation or dense materialization. The other operand is an
ordinary ndslice. Ownership and dimensions follow the plain-slice overload.
Compatible dense matrices use SYMM, or blocked GEMM for mixed orientations.
Eligible real vectors use SYMV (SPMV for packed storage).
General layouts and types retain a native D fallback.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, B.N)
makeMtimes(TrianglePolicy triangle, A, B, Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref const(SymmetricView!(triangle, A)) a)
    if (numericSlice!B && is(Scalar!A == Scalar!B))
{
    auto result = allocate!(Allocation.custom, Scalar!A)(allocator, b.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    mtimesInto(b, a, result);
    return result;
}

/++
Overwrite caller storage with a symmetric product. Only the selected triangle
is read. Output has the ordinary operand's shape and must be disjoint from
both inputs, with nonoverlapping writable elements. No allocation occurs.
+/
package(mir.stat) void mtimesInto(TrianglePolicy triangle, A, B, C)(
    scope auto ref const(SymmetricView!(triangle, A)) a,
    scope auto ref B b, scope auto ref C c)
    if (numericSlice!B && numericSlice!C && B.N == C.N &&
        is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C))
{
    auto storage = a.storage;
    symmetricMultiply!triangle(storage, b, c);
}

/// Right symmetric product; output and aliasing requirements match the left overload.
package(mir.stat) void mtimesInto(TrianglePolicy triangle, A, B, C)(
    scope auto ref B b, scope auto ref const(SymmetricView!(triangle, A)) a,
    scope auto ref C c)
    if (numericSlice!B && numericSlice!C && B.N == C.N &&
        is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C))
{
    static if (B.N == 1) mtimesInto(a, b, c);
    else
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        mtimesInto(a, b.universal.transposed, c.universal.transposed);
    }
}

/// Dispatch a left symmetric product, preserving logical triangle and general strides.
private void symmetricMultiply(TrianglePolicy triangle, A, B, C)(
    scope auto ref A a, scope auto ref B b, scope auto ref C c)
{
    alias T = Scalar!A;
    assert(a.length!0 == a.length!1 && a.length!0 == b.length!0);
    assert(b.shape == c.shape);
    if (!b.length!0) return;
    static if (B.N == 2)
    {
        if (!b.length!1) return;
        // A rank-two single column is still a matrix result, but real
        // arithmetic can use the same kernel as the corresponding vector.
        static if (is(T == float) || is(T == double))
        {
            if (b.length!1 == 1)
            {
                import mir.ndslice.topology: universal;
                import mir.ndslice.dynamic: transposed;
                symmetricMultiply!triangle(a, b.universal.transposed[0], c.universal.transposed[0]);
                return;
            }
        }
        static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!B && nativeSlice!C)
        {
            import mir.ndslice.topology: universal;
            auto ap = a.lightScope.lightConst.universal;
            auto bp = b.lightScope.lightConst.universal;
            auto cp = c.lightScope.universal;
            if (blasMatrix(a) && blasMatrix(b) && blasMatrix(c) &&
                sameSymmetricOrientation(bp, cp))
            {
                import mir.blas: symm;
                import cblas: Side, Uplo;
                enum uplo = triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
                symm(Side.Left, uplo, T(1), ap, bp, T(0), cp);
                return;
            }
            if (blasMatrix(a) && blasMatrix(b) && blasMatrix(c))
            {
                symmetricMultiplyMixed!triangle(ap, bp, cp);
                return;
            }
        }
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        auto bt = b.universal.transposed;
        auto ct = c.universal.transposed;
        foreach (j; 0 .. b.length!1) symmetricMultiply!triangle(a, bt[j], ct[j]);
    }
    else
    {
        static if (is(T == float) || is(T == double))
        {
            import mir.stat.internal.linearAlgebra.packed: tryPackedSymmetric;
            if (tryPackedSymmetric!triangle(a, b, c)) return;
            static if (hasBlas && nativeSlice!A && nativeSlice!B && nativeSlice!C)
            {
                if (blasMatrix(a) && blasVector(b) && blasVector(c))
                {
                    import mir.blas: symv;
                    import cblas: Uplo;
                    enum uplo = triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
                    symv(uplo, T(1), a.lightScope.lightConst, b.lightScope.lightConst, T(0), c.lightScope);
                    return;
                }
            }
        }
        import mir.stat.internal.linearAlgebra.packed: tryPackedSymmetricNative;
        if (tryPackedSymmetricNative!triangle(a, b, c)) return;
        foreach (i; 0 .. b.length)
        {
            T sum = T(0);
            foreach (k; 0 .. i)
            {
                static if (triangle == TrianglePolicy.upper) sum += a[k, i] * b[k];
                else sum += a[i, k] * b[k];
            }
            foreach (k; i .. b.length)
            {
                static if (triangle == TrianglePolicy.upper) sum += a[i, k] * b[k];
                else sum += a[k, i] * b[k];
            }
            c[i] = sum;
        }
    }
}


// Exercise SYMM-compatible and mixed layouts, right products, packed storage,
// complex symmetric (not Hermitian) arithmetic, and explicit ownership.
version(mir_stat_test_linearAlgebra)
@system pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, stairs, map;
    import mir.ndslice.dynamic: transposed, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    import std.meta: AliasSeq;
    import std.math: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    foreach (ta; [false, true]) foreach (tb; [false, true]) foreach (tc; [false, true])
    {
        T[9] av; T[12] bv, cv;
        auto a = av[].sliced(3, 3).universal;
        auto b = bv[].sliced(tb ? 4 : 3, tb ? 3 : 4).universal;
        auto c = cv[].sliced(tc ? 4 : 3, tc ? 3 : 4).universal;
        if (ta) a = a.transposed;
        if (tb) b = b.transposed;
        if (tc) c = c.transposed;
        foreach (i;0 .. 3) foreach (j;0 .. 3) a[i, j] = j>=i ? T(i+j+1) : T.nan;
        foreach (i;0 .. 3) foreach (j;0 .. 4) b[i, j] = T(i+2*j+1);
        auto view = symmetricView(a);
        mtimesInto(view, b, c);
        foreach (i;0 .. 3) foreach (j;0 .. 4) {
            T expected = 0; foreach (k;0 .. 3) expected+=T(i+k+1)*b[k, j];
            assert(c[i, j] == expected);
        }
        mtimesInto(b.transposed, view, c.transposed);
        assert(rcMtimes(view, b) == c);
        assert(mtimes(view.transposed, b) == c);
        T[6] packed = [1, 2, 3, 3, 4, 5];
        auto pv = symmetricView(packed[].sliced.stairs!"-"(3));
        assert(mtimes(pv, b) == c);
        assert(mtimes(b.transposed, pv) == c.transposed);
        assert(mtimes(view, b.map!(v=>v)) == c);
    }
    import mir.complex: Complex;
    alias Z = Complex!double;
    Z[4] a = [Z(1, 1), Z(2, 3), Z(double.nan, 0), Z(4, -1)];
    Z[2] b = [Z(1, -1), Z(2, 1)];
    auto v = symmetricView(a[].sliced(2, 2));
    auto result = mtimes(v, b[].sliced);
    assert(result[0] == a[0]*b[0]+a[1]*b[1]);
    assert(result[1] == a[1]*b[0]+a[3]*b[1]);
    auto matrixResult = mtimes(v, b[].sliced(2, 1));
    assert(matrixResult[0, 0] == result[0] && matrixResult[1, 0] == result[1]);
    auto owner = mininitRcslice!double(2, 2); owner[] = 0; owner[0, 0] = owner[1, 1] = 1;
    const retained = symmetricView(owner);
    double[2] x = [2, 3];
    assert(rcMtimes(retained, x[].sliced) == x[]);
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] a = [1, 0, double.nan, 1]; double[2] x = [2, 3];
    auto result = makeMtimes(Mallocator.instance, symmetricView(a[].sliced(2, 2)), x[].sliced);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == x[]);
}

/// Multiply mixed-layout native matrices without allocating or reading an unused triangle.
/// Diagonal blocks initialize output; GEMM adds each stored off-diagonal block
/// and its reflection. Small leaves preserve ascending contraction order.
private void symmetricMultiplyMixed(TrianglePolicy triangle, A, B, C)(A a, B b, C c)
{
    import mir.blas: gemm;
    import mir.ndslice.dynamic: transposed;
    alias T = Scalar!A;
    const n = a.length!0;
    if (n <= 16)
    {
        foreach (i; 0 .. n) foreach (j; 0 .. b.length!1)
        {
            T sum = T(0);
            foreach (k; 0 .. i)
            {
                static if (triangle == TrianglePolicy.upper) sum += a[k, i] * b[k, j];
                else sum += a[i, k] * b[k, j];
            }
            foreach (k; i .. n)
            {
                static if (triangle == TrianglePolicy.upper) sum += a[i, k] * b[k, j];
                else sum += a[k, i] * b[k, j];
            }
            c[i, j] = sum;
        }
        return;
    }
    const split = n / 2;
    auto b0 = b[0 .. split, 0 .. $];
    auto b1 = b[split .. $, 0 .. $];
    auto c0 = c[0 .. split, 0 .. $];
    auto c1 = c[split .. $, 0 .. $];
    symmetricMultiplyMixed!triangle(a[0 .. split, 0 .. split], b0, c0);
    symmetricMultiplyMixed!triangle(a[split .. $, split .. $], b1, c1);
    static if (triangle == TrianglePolicy.upper) auto off = a[0 .. split, split .. $];
    else auto off = a[split .. $, 0 .. split].transposed;
    gemm(T(1), off, b1, T(1), c0);
    gemm(T(1), off.transposed, b0, T(1), c1);
}

// Cross block boundaries with unequal input/output orientations, padding,
// both stored triangles, and nonconjugating complex arithmetic.
version(mir_stat_test_linearAlgebra)
@safe pure unittest
{
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    import mir.complex: Complex;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, Complex!double))
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    foreach (n; [1, 17, 33, 65]) foreach (flip; [false, true])
    {
        auto a = uninitSlice!T(n, n + 2)[0 .. n, 0 .. n].universal;
        if (flip) a = a.transposed;
        auto b = uninitSlice!T(flip ? n : 5, flip ? 5 : n).universal;
        auto c = uninitSlice!T(flip ? 5 : n, flip ? n : 5).universal;
        if (!flip) b = b.transposed;
        if (flip) c = c.transposed;
        foreach (i; 0 .. n) foreach (j; 0 .. n)
        {
            auto value = T((i + j) % 7 + 1);
            static if (is(T == Complex!double)) value += T(0, 1);
            a[i, j] = (triangle == TrianglePolicy.upper ? j >= i : j <= i) ? value : T(double.nan);
        }
        foreach (i; 0 .. n) foreach (j; 0 .. 5) b[i, j] = T((i + j) % 5 + 1);
        mtimesInto(symmetricView!triangle(a), b, c);
        foreach (i; 0 .. n) foreach (j; 0 .. 5)
        {
            T expected = T(0);
            foreach (k; 0 .. n)
            {
                auto value = T((i + k) % 7 + 1);
                static if (is(T == Complex!double)) value += T(0, 1);
                expected += value * b[k, j];
            }
            assert(c[i, j] == expected);
        }
        auto left = mtimes(symmetricView!triangle(a), b);
        mtimesInto(b.transposed, symmetricView!triangle(a), c.transposed);
        assert(left == c);
        auto rowWork = uninitSlice!T(n, 5);
        auto columnWork = uninitSlice!T(5, n).universal.transposed;
        mtimesInto(symmetricView!triangle(a), b, c, rowWork);
        assert(left == c);
        mtimesInto(symmetricView!triangle(a), b, c, columnWork);
        assert(left == c);
        mtimesInto(b.transposed, symmetricView!triangle(a), c.transposed, rowWork.transposed);
        assert(left == c);
        mtimesInto(b.transposed, symmetricView!triangle(a), c.transposed, columnWork.transposed);
        assert(left == c);
    }
}

// Packed direct indexing supports lower storage, transpose, complex values,
// and reversed vectors; cropped adapters retain the general indexing fallback.
version(mir_stat_test_linearAlgebra)
@system pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs, universal;
    import mir.ndslice.dynamic: transposed, reversed;
    import mir.complex: Complex;
    import std.meta: AliasSeq;
    import mir.stat.internal.linearAlgebra.packed: packedMatrix;
    static foreach (T; AliasSeq!(float, double, real, Complex!double))
    {{
        T[6] data = [T(1), T(2), T(3), T(3), T(4), T(5)];
        static if (is(T == Complex!double)) foreach (ref v; data) v += T(0, 1);
        T[3] vector = [T(1), T(2), T(3)], output;
        auto x = vector[].sliced.universal.reversed;
        auto y = output[].sliced.universal.reversed;
        auto view = symmetricView(data[].sliced.stairs!"+"(3));
        mtimesInto(view.transposed, x, y);
        assert(y[0] == data[0]*x[0] + data[1]*x[1] + data[3]*x[2]);
        assert(y[1] == data[1]*x[0] + data[2]*x[1] + data[4]*x[2]);
        assert(y[2] == data[3]*x[0] + data[4]*x[1] + data[5]*x[2]);
        auto cropped = packedMatrix(data[].sliced.stairs!"+"(3))[1 .. 3, 1 .. 3];
        auto result = mtimes(symmetricView!(TrianglePolicy.lower)(cropped), x[0 .. 2]);
        assert(result[0] == data[2]*x[0] + data[4]*x[1]);
        assert(result[1] == data[4]*x[0] + data[5]*x[1]);
    }}
}

/// Mixed storage orientations do not require scratch allocation.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    double[17*17] matrix;
    double[17*3] rhs, output;
    auto a = matrix[].sliced(17,17);
    auto b = rhs[].sliced(3,17).universal.transposed;
    auto c = output[].sliced(17,3);
    foreach (i; 0 .. 17) foreach (j; 0 .. 17)
        a[i,j] = j < i ? double.nan : i == j ? 1 : 0;
    b[] = 2;
    mtimesInto(symmetricView(a),b,c);
    assert(c == b);
}

/++
Symmetric matrix multiplication with optional caller-provided layout workspace.
All operands are matrices. `work` has the same shape and element type as the
ordinary operand and output. Output and workspace must have nonoverlapping
writable elements, be mutually disjoint, and be disjoint from both inputs.
Shapes are asserted; overlap is not checked. No storage is allocated.

For eligible BLAS matrices with different input/output orientations, workspace
matching the output is used to copy the ordinary input before SYMM. Workspace
matching the ordinary input receives the product, which is then copied into
output. Thus callers choose the conversion through the workspace's strides.
Both alternatives read only the symmetric view's selected triangle.

Workspace contents are unspecified afterward. It may be unused: matching
input/output layouts, real single-column products, or unsupported backend,
type, or storage combinations retain the ordinary dispatch and native D fallback.
No timing-based threshold or dense expansion of the symmetric matrix is used.
+/
package(mir.stat) void mtimesInto(TrianglePolicy triangle, A, B, C, W)(
    scope auto ref const(SymmetricView!(triangle, A)) a,
    scope auto ref B b, scope auto ref C c, scope auto ref W work)
    if (numericSlice!B && numericSlice!C && numericSlice!W && B.N == 2 && C.N == 2 && W.N == 2 &&
        is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C) && is(Scalar!A == Scalar!W))
{
    assert(a.length!0 == b.length!0 && b.shape == c.shape && b.shape == work.shape,
        "Incompatible symmetric product workspace dimensions");
    alias T = Scalar!A;
    static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!B && nativeSlice!C && nativeSlice!W)
    {
        auto storage = a.storage;
        bool directVector = false;
        static if (is(T == float) || is(T == double)) directVector = b.length!1 == 1;
        if (b.length!0 && b.length!1 && !directVector &&
            blasMatrix(storage) && blasMatrix(b) && blasMatrix(c) && blasMatrix(work) &&
            !sameSymmetricOrientation(b, c))
        {
            if (sameSymmetricOrientation(work, c))
            {
                copySymmetricProduct(b, work);
                mtimesInto(a, work, c);
                return;
            }
            if (sameSymmetricOrientation(b, work))
            {
                mtimesInto(a, b, work);
                copySymmetricProduct(work, c);
                return;
            }
        }
    }
    mtimesInto(a, b, c);
}

/// Reuse ordinary workspace when the input is a transposed view.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    double[4] data = [4,2,double.nan,5], input = [1,2,3,4], output, buffer;
    auto a = symmetricView(data[].sliced(2,2));
    auto b = input[].sliced(2,2).transposed;
    auto c = output[].sliced(2,2);
    auto work = buffer[].sliced(2,2);
    mtimesInto(a,b,c,work); // Workspace matches output: copy input if useful.
    assert(c[0,0] == 8 && c[0,1] == 20 && c[1,0] == 12 && c[1,1] == 26);
    mtimesInto(a,b,c,work.transposed); // Workspace matches input: copy output.
    assert(c[0,0] == 8 && c[0,1] == 20 && c[1,0] == 12 && c[1,1] == 26);
}

/// Right symmetric product with layout workspace. Shape, disjointness, and
/// backend behavior follow the left-side workspace overload.
package(mir.stat) void mtimesInto(TrianglePolicy triangle, A, B, C, W)(
    scope auto ref B b, scope auto ref const(SymmetricView!(triangle, A)) a,
    scope auto ref C c, scope auto ref W work)
    if (numericSlice!B && numericSlice!C && numericSlice!W && B.N == 2 && C.N == 2 && W.N == 2 &&
        is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C) && is(Scalar!A == Scalar!W))
{
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;
    mtimesInto(a, b.universal.transposed, c.universal.transposed, work.universal.transposed);
}

/// Whether two BLAS-eligible descriptors satisfy mir.blas SYMM's shared orientation requirement.
private bool sameSymmetricOrientation(B, C)(scope auto ref B b, scope auto ref C c)
{
    return (b._stride!1 == 1 && c._stride!1 == 1) ||
        (b._stride!0 == 1 && c._stride!0 == 1 && b._stride!1 != 1 && c._stride!1 != 1);
}

/// Copy disjoint matrix descriptors with destination-contiguous traversal.
private void copySymmetricProduct(B, C)(scope auto ref B b, scope auto ref C c)
{
    if (c._stride!1 == 1)
        foreach (i; 0 .. c.length!0) foreach (j; 0 .. c.length!1) c[i,j] = b[i,j];
    else
        foreach (j; 0 .. c.length!1) foreach (i; 0 .. c.length!0) c[i,j] = b[i,j];
}


// Rank-two columns and rows preserve result shape while sharing real vector kernels.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed, strided;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float,double,real))
    {{
        T[9] matrix = [T(2),T(1),T(3),T.nan,T(4),T(2),T.nan,T.nan,T(5)];
        T[6] values = [T(1),T(99),T(2),T(99),T(3),T(99)], result, scratch;
        auto a = symmetricView(matrix[].sliced(3,3));
        auto b = values[].sliced(3,2)[0 .. 3,0 .. 1];
        auto c = result[].sliced(3,2)[0 .. 3,0 .. 1];
        auto work = scratch[].sliced(3,2)[0 .. 3,0 .. 1];
        mtimesInto(a,b,c,work);
        assert(c[0,0] == 13 && c[1,0] == 15 && c[2,0] == 22);
        mtimesInto(b.transposed,a,c.transposed,work.transposed);
        assert(c[0,0] == 13 && c[1,0] == 15 && c[2,0] == 22);
        T[0] empty;
        mtimesInto(symmetricView(empty[].sliced(0,0)),empty[].sliced(0,1),empty[].sliced(0,1),empty[].sliced(0,1));
    }}
}

// General strides and computed inputs retain the no-copy fallback.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided;
    double[4] data = [4,2,double.nan,5], values = [1,2,3,4], output;
    double[16] buffer;
    auto a = symmetricView(data[].sliced(2,2));
    auto b = values[].sliced(2,2).transposed;
    auto c = output[].sliced(2,2);
    auto work = buffer[].sliced(4,4).universal.strided!(0,1)(2,2);
    mtimesInto(a,b,c,work);
    assert(c[0,0] == 8 && c[1,1] == 26);
    mtimesInto(a,b.map!(v => v),c,work);
    assert(c[0,0] == 8 && c[1,1] == 26);
}

/++
Multiply two symmetric views without materializing either input. The result
is an ordinary dense matrix: a product of symmetric matrices need not be
symmetric. Each input's selected triangle is authoritative. This uses the
native D fallback for logical indexing, with the usual result ownership.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, 2)
mtimes(TrianglePolicy left, TrianglePolicy right, A, B)(
    scope auto ref const(SymmetricView!(left,A)) a,
    scope auto ref const(SymmetricView!(right,B)) b)
    if (is(Scalar!A == Scalar!B))
{
    return mtimes(a, b.asSlice);
}

/++
Multiply two symmetric views without materializing either input. The result
is an ordinary dense matrix: a product of symmetric matrices need not be
symmetric. Each input's selected triangle is authoritative. This uses the
native D fallback for logical indexing, with the usual result ownership.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, 2)
rcMtimes(TrianglePolicy left, TrianglePolicy right, A, B)(
    scope auto ref const(SymmetricView!(left,A)) a,
    scope auto ref const(SymmetricView!(right,B)) b)
    if (is(Scalar!A == Scalar!B))
{
    return rcMtimes(a, b.asSlice);
}

/++
Multiply two symmetric views without materializing either input. The result
is an ordinary dense matrix: a product of symmetric matrices need not be
symmetric. Each input's selected triangle is authoritative. This uses the
native D fallback for logical indexing, with the usual result ownership.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, 2)
makeMtimes(TrianglePolicy left, TrianglePolicy right, A, B, Allocator)(
    auto ref Allocator allocator, scope auto ref const(SymmetricView!(left,A)) a,
    scope auto ref const(SymmetricView!(right,B)) b)
    if (is(Scalar!A == Scalar!B))
{
    return makeMtimes(allocator, a, b.asSlice);
}

/// Multiply two symmetric views into disjoint caller storage without allocation.
package(mir.stat) void mtimesInto(TrianglePolicy left, TrianglePolicy right, A, B, C)(
    scope auto ref const(SymmetricView!(left,A)) a,
    scope auto ref const(SymmetricView!(right,B)) b, scope auto ref C output)
    if (numericSlice!C && C.N == 2 && is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C))
{
    mtimesInto(a, b.asSlice, output);
}

/// Symmetric operands need not commute; the product is an ordinary matrix.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[4] aa=[4,2,double.nan,5], bb=[3,double.nan,1,2], cc;
    auto a=symmetricView(aa[].sliced(2,2));
    auto b=symmetricView!(TrianglePolicy.lower)(bb[].sliced(2,2));
    assert(mtimes(a,b)==[[14,8],[11,12]]);
    assert(rcMtimes(a.transposed,b.transposed)==[[14,8],[11,12]]);
    mtimesInto(a,b,cc[].sliced(2,2));
    assert(cc==[14,8,11,12]);
}

// Both structural operands work with complex arithmetic and custom result ownership.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.complex: Complex;
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    alias T=Complex!double;
    T[4] aa=[T(1,1),T(2,0),T(double.nan),T(3,-1)];
    T[4] bb=[T(2,-1),T(double.nan),T(1,1),T(4,0)];
    auto a=symmetricView(aa[].sliced(2,2));
    auto b=symmetricView!(TrianglePolicy.lower)(bb[].sliced(2,2));
    auto result=makeMtimes(Mallocator.instance,a,b);
    scope(exit) Mallocator.instance.dispose(result.field);
    auto reference=rcMtimes(a.asSlice,b.asSlice);
    assert(result==reference);
    assert(rcMtimes(a,b)==reference);
}



/// Whether diagonal scaling accepts the other logical numeric operand.
private enum diagonalCompatible(D, B) = numericSlice!D && D.N == 1 &&
    __traits(compiles, logicalOperand(B.init)) &&
    numericSlice!(typeof(logicalOperand(B.init))) &&
    is(Scalar!D == Scalar!(typeof(logicalOperand(B.init))));

/++
Diagonal scaling on the left, returning gc storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!D, B.N)
mtimes(D, B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (diagonalCompatible!(D,B))
{
    int allocator;
    auto other = logicalOperand(b);
    assert(a.length!0 == other.length!0);
    auto result = allocate!(Allocation.gc, Scalar!D)(allocator, other.shape);

    mtimesInto(a, b, result);
    return result;
}

/++
Diagonal scaling on the right, returning gc storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!D, B.N)
mtimes(D, B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (diagonalCompatible!(D,B) && !DiagonalInfo!B.valid)
{
    int allocator;
    auto other = logicalOperand(b);
    static if (B.N == 1) assert(a.length!0 == other.length!0);
    else assert(a.length!0 == other.length!1);
    auto result = allocate!(Allocation.gc, Scalar!D)(allocator, other.shape);

    mtimesInto(other, a, result);
    return result;
}

/++
Diagonal scaling on the left, returning rc storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!D, B.N)
rcMtimes(D, B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (diagonalCompatible!(D,B))
{
    int allocator;
    auto other = logicalOperand(b);
    assert(a.length!0 == other.length!0);
    auto result = allocate!(Allocation.rc, Scalar!D)(allocator, other.shape);

    mtimesInto(a, b, result);
    return result;
}

/++
Diagonal scaling on the right, returning rc storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!D, B.N)
rcMtimes(D, B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (diagonalCompatible!(D,B) && !DiagonalInfo!B.valid)
{
    int allocator;
    auto other = logicalOperand(b);
    static if (B.N == 1) assert(a.length!0 == other.length!0);
    else assert(a.length!0 == other.length!1);
    auto result = allocate!(Allocation.rc, Scalar!D)(allocator, other.shape);

    mtimesInto(other, a, result);
    return result;
}

/++
Diagonal scaling on the left, returning custom storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!D, B.N)
makeMtimes(D, B, Allocator)(auto ref Allocator allocator, scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (diagonalCompatible!(D,B))
{

    auto other = logicalOperand(b);
    assert(a.length!0 == other.length!0);
    auto result = allocate!(Allocation.custom, Scalar!D)(allocator, other.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    mtimesInto(a, b, result);
    return result;
}

/++
Diagonal scaling on the right, returning custom storage.
Reads only the stored diagonal and the other operand's logical entries.
Results are ordinary dense slices, including for two diagonal operands.
Shapes, aliasing, and custom disposal follow mtimes. Structural operands may
be combined explicitly; complex arithmetic never conjugates either input.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!D, B.N)
makeMtimes(D, B, Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (diagonalCompatible!(D,B) && !DiagonalInfo!B.valid)
{

    auto other = logicalOperand(b);
    static if (B.N == 1) assert(a.length!0 == other.length!0);
    else assert(a.length!0 == other.length!1);
    auto result = allocate!(Allocation.custom, Scalar!D)(allocator, other.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    mtimesInto(other, a, result);
    return result;
}

/++
Write diagonal row scaling without allocating.
Output must have the other operand's shape, be disjoint from both inputs, and
have nonoverlapping elements. Computed and strided inputs are supported.
+/
package(mir.stat) void mtimesInto(D,B,C)(scope auto ref const(DiagonalView!D) a, scope auto ref B b, scope auto ref C output)
    if (diagonalCompatible!(D,B) && numericSlice!C && C.N == B.N && is(Scalar!D == Scalar!C))
{
    auto other = logicalOperand(b);
    assert(output.shape == other.shape);
    static if (DiagonalInfo!B.valid)
    {
        assert(a.shape == b.shape);
        foreach(i; 0 .. a.length!0) foreach(j; 0 .. a.length!1)
            output[i,j] = i == j ? a[i,i] * b[j,j] : Scalar!D(0);
    }
    else static if (B.N == 1)
    {
        assert(a.length!0 == other.length);
        foreach(i; 0 .. other.length) output[i] = a[i,i] * other[i];
    }
    else
    {
        assert(a.length!0 == other.length!0);
        foreach(i; 0 .. other.length!0) foreach(j; 0 .. other.length!1)
            output[i,j] = a[i,i] * other[i,j];
    }
}

/++
Write diagonal column scaling without allocating.
Output must have the other operand's shape, be disjoint from both inputs, and
have nonoverlapping elements. Computed and strided inputs are supported.
+/
package(mir.stat) void mtimesInto(D,B,C)(scope auto ref B b, scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (diagonalCompatible!(D,B) && numericSlice!C && C.N == B.N && is(Scalar!D == Scalar!C) && !DiagonalInfo!B.valid)
{
    auto other = logicalOperand(b);
    assert(output.shape == other.shape);
    static if (B.N == 1)
    {
        assert(a.length!0 == other.length);
        foreach(i; 0 .. other.length) output[i] = other[i] * a[i,i];
    }
    else
    {
        assert(a.length!0 == other.length!1);
        foreach(i; 0 .. other.length!0) foreach(j; 0 .. other.length!1)
            output[i,j] = other[i,j] * a[j,j];
    }
}


/// Diagonal multiplication scales rows on the left and columns on the right.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[2] ds = [2,3], xs = [5,7];
    double[4] bs = [1,2,3,4];
    auto d = diagonalView(ds[].sliced);
    auto b = bs[].sliced(2,2);
    assert(mtimes(d,xs[].sliced) == [10,21]);
    assert(mtimes(xs[].sliced,d) == [10,21]);
    assert(mtimes(d,b) == [[2,4],[9,12]]);
    assert(mtimes(b,d) == [[2,6],[6,12]]);
    assert(mtimes(d,d) == [[4,0],[0,9]]);
    assert(mtimes(d,triangularView(b)) == [[2,4],[0,12]]);
    assert(mtimes(symmetricView(b),d) == [[2,6],[4,12]]);
}

// Compare layouts and ownership families against explicit dense materialization.
version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: strided, reversed, transposed;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.allocation: rcslice;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.meta: AliasSeq;
    static foreach(T; AliasSeq!(float,double,real)) {{
        T[4] ds=[2,99,3,99], bs=[1,2,3,4], outputs;
        auto storage=ds[].sliced.strided!0(2).reversed!0;
        const d=diagonalView(storage.rcslice);
        auto b=bs[].sliced(2,2).universal.transposed;
        auto reference=rcMtimes(d.asSlice,b);
        assert(rcMtimes(d,b)==reference);
        auto result=makeMtimes(Mallocator.instance,d,b);
        scope(exit) Mallocator.instance.dispose(result.field);
        assert(result==reference);
        auto output=outputs[].sliced(2,2).universal.transposed.reversed!0;
        mtimesInto(d,b,output);
        assert(output==reference);
        assert(rcMtimes(b,d)==rcMtimes(b,d.asSlice));
        auto right=makeMtimes(Mallocator.instance,b,d);
        scope(exit) Mallocator.instance.dispose(right.field);
        assert(right==rcMtimes(b,d.asSlice));
        mtimesInto(b,d,output);
        assert(output==right);
        assert(rcMtimes(diagonalView(storage.map!(v=>v)),b)==reference);
        assertThrown!AssertError(rcMtimes(d,bs[].sliced));
        T[0] empty;
        auto e=diagonalView(empty[].sliced);
        assert(rcMtimes(e,empty[].sliced).length==0);
        assert(rcMtimes(e,empty[].sliced(0,3)).shape==[0,3]);
        assert(rcMtimes(empty[].sliced(3,0),e).shape==[3,0]);
    }}
    import mir.complex: Complex;
    alias Z=Complex!double;
    Z[2] ds=[Z(1,1),Z(2,-1)], xs=[Z(2,1),Z(1,-1)];
    auto d=diagonalView(ds[].sliced);
    assert(rcMtimes(d,xs[].sliced)==[ds[0]*xs[0],ds[1]*xs[1]]);
    assert(rcMtimes(xs[].sliced,d.transposed)==[xs[0]*ds[0],xs[1]*ds[1]]);
}
