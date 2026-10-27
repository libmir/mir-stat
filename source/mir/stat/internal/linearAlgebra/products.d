/++
Matrix products with explicit result ownership. `mtimes` uses GC storage,
`rcMtimes` uses RC storage, and `makeMtimes` uses the supplied allocator.
Vector-vector products return a scalar and do not allocate. Products do not
conjugate complex inputs. Inputs may be borrowed or RC slices, independently
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
the other operand must be an ordinary ndslice.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.products;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;


import mir.stat.internal.linearAlgebra.types: Scalar, numericSlice, TrianglePolicy, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.backend: hasBlas, blasNumber, nativeSlice, blasMatrix, blasVector;
import mir.stat.internal.linearAlgebra.views: TriangularView, triangularView;
import mir.stat.internal.linearAlgebra.packed: isPacked, PackedInfo;
import mir.stat.internal.linearAlgebra.allocation;

/// Whether operands are supported numeric slices with matching element types.
private enum compatible(A, B) = numericSlice!A && numericSlice!B &&
    is(Scalar!A == Scalar!B);

// A vector-vector product is scalar; other products have the contracted rank.
private template ProductResult(Allocation policy, A, B)
{
    static if (A.N == 1 && B.N == 1)
        alias ProductResult = Scalar!A;
    else
        alias ProductResult = AllocatedSlice!(policy, Scalar!A, A.N + B.N - 2);
}

/++
Multiply two matrices or vectors, allocating a GC-managed result when needed.

This is matrix multiplication, not element-wise multiplication. For shapes
`(m, k)` and `(k, n)`, the result has shape `(m, n)`. A matrix and a vector
produce a vector of length `m`; a vector and a matrix produce a vector of
length `n`. Two vectors of length `k` produce a scalar, without allocation.
Complex inputs are multiplied without conjugation.

Use `mtimes(triangularView(a), b)` or `mtimes(b, triangularView(a))` for a
triangular operand. These overloads select the left or right triangular product
without requiring a different function name. Unused entries are not read.

Params:
    a = Left operand, a rank-one or rank-two ndslice.
    b = Right operand with the same floating-point or complex element type.
Returns:
    A GC-backed slice, or a scalar for two vector operands. The result element
    type is the unqualified input element type.
Preconditions:
    The inner dimensions must match. Empty inner dimensions yield zeros.
See_Also: $(LINK2 #rcMtimes, rcMtimes), $(LINK2 #makeMtimes, makeMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.gc, A, B)
mtimes(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
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
    assert(mtimes(v, v) == 14);
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
    An RC-backed slice, or an allocation-free scalar for two vectors.
Preconditions:
    The inner dimensions must match.
See_Also: $(LINK2 #mtimes, mtimes), $(LINK2 #makeMtimes, makeMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.rc, A, B)
rcMtimes(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
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
    A slice backed by the supplied allocator, or a scalar for two vectors.
    Scalar products do not call the allocator.
Preconditions:
    The inner dimensions must match; they are checked before allocation.
Throws:
    `OutOfMemoryError` if the result element or byte count overflows `size_t`.
See_Also: $(LINK2 #mtimes, mtimes), $(LINK2 #rcMtimes, rcMtimes), $(LINK2 #mtimesInto, mtimesInto)
+/
package(mir.stat) ProductResult!(Allocation.custom, A, B)
makeMtimes(Allocator, A, B)(auto ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
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

// Shared factory: scalar products allocate nothing; other products allocate
// according to policy and delegate computation to mtimesInto.
/// Compute a dot product directly, or allocate one result and release custom storage if computation throws.
private ProductResult!(policy, A, B)
multiplyWithAllocation(Allocation policy, Allocator, A, B)(ref Allocator allocator,
    scope auto ref A a, scope auto ref B b)
{
    checkMultiplicationDimensions(a, b);
    alias T = Scalar!A;
    static if (A.N == 1 && B.N == 1)
    {
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
    else
    {
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
}

/++
Write a matrix or vector product into existing storage without allocating.

Shapes and arithmetic follow $(LINK2 #mtimes, mtimes), except that two vector operands
are not supported: use $(LINK2 #mtimes, mtimes) for an allocation-free scalar result.
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
    assert(rcMtimes(v, v) == 14);
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
    assert(rcMtimes(zs, zs) == Complex!double(0, 2));
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
    assert(rcMtimes(v.reversed!0, v) == 12);
    auto mapped = a.map!"a * 2";
    auto doubled = rcMtimes(mapped, b);
    assert(doubled == product.map!"a * 2");
    assert(rcMtimes(v.map!"a * 2", v) == 26);
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
    assert(makeMtimes(allocator, data[].sliced, data[].sliced) == 30);
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
    assert(rcMtimes(empty, empty) == 0);
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
