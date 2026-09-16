/++
This module contains code copied from the lubeck2 module of the lubeck project (with minor modifications).

License: $(HTTP boost.org/LICENSE_1_0.txt, Boost License 1.0)

Authors: Ilya Yaroshenko, Thomas Webster, Lars Tandle Kyllingstad (SciD author), John Michael Hall

Copyright: 2023 Mir Stat Authors; 2017-2020 Symmetry Investments & Kaleidic Associates; 2009, Lars T. Kyllingstad (SciD)

+/

module mir.math.internal.lubeck2;

static if (is(typeof({ import mir.blas; import mir.lapack; }))) {

import mir.blas: Side, Uplo;
import mir.internal.utility: isComplex;
import mir.ndslice.slice: Slice, SliceKind;
import mir.rc.array: RCI;
import std.traits: isFloatingPoint, Unqual;

/++
Identity matrix.

Params:
    n = number of columns
    m = optional number of rows, default n
Results:
    Matrix which is 1 on the diagonal and 0 elsewhere
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 2) eye(T = double)(
    size_t n,
    size_t m = 0
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(n > 0);
    assert(m >= 0);
}
out (i)
{
    assert(i.length!0 == n);
    assert(i.length!1 == (m == 0 ? n : m));
}
do
{
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.topology: diagonal;

    auto c = rcslice!T([n, (m == 0 ? n : m)], cast(T)0);
    c.diagonal[] = cast(T)1;
    return c;
}

/// Real numbers
@safe pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    assert(eye(1)== [
        [1]]);
    assert(eye(2)== [
        [1, 0],
        [0, 1]]);
    assert(eye(3)== [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]);
    assert(eye(1,2) == [
        [1,0]]);
    assert(eye(2,1) == [
        [1],
        [0]]);
}

/++
Matrix multiplication. Allocates result to using Mir refcounted arrays.

This function has multiple overloads that include the following functionality:
- a.mtimes(b) where `a` and `b` are both two-dimensional slices. The result is a
two-dimensional slice.
- a.mtimes(b) where `a` is a two-dimensional slice and `b` is a one-dimensional
slice. The result is a one-dimensional slice. In this case, `b` can be thought
of as a column vector.
- b.mtimes(a) where `a` is a two-dimensional slice and `b` is a one-dimensional
slice. The result is a one-dimensional slice. In this case, `b` can be thought
of as a row vector.

Params:
    a = m(rows) x k(cols) matrix
    b = k(rows) x n(cols) matrix
Result:
    m(rows) x n(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 2) mtimes(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
}
out (c)
{
    assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
    assert(c.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
}
do
{
    import mir.blas: gemm;
    import mir.ndslice.allocation: mininitRcslice;

    auto c = mininitRcslice!T(a.length!0, b.length!1);
    gemm(cast(T)1, a, b, cast(T)0, c.lightScope);
    return c;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .mtimes(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(a, scopeB);
}

/++
Params:
    a = m(rows) x n(cols) matrix
    b = n(rows) x 1(cols) vector
Result:
    m(rows) x 1(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 1) mtimes(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindB) b
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
}
out (c)
{
    assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
}
do
{
    import mir.blas: gemv;
    import mir.ndslice.allocation: mininitRcslice;

    auto c = mininitRcslice!T(a.length!0);
    gemv(cast(T)1, a, b, cast(T)0, c.lightScope);
    return c;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .mtimes(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(a, scopeB);
}

/++
Params:
    b = 1(rows) x n(cols) vector
    a = n(rows) x m(cols) matrix
Result:
    1(rows) x m(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 1) mtimes(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 1, kindB) b,
    Slice!(const(T)*, 2, kindA) a
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(b.length!0 == a.length!0, "The length of `b` must match the first dimension of `a`");
}
out (c)
{
    assert(c.length!0 == a.length!1, "The first dimension of the result must match the second dimension of `a`");
}
do
{
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: universal;

    return .mtimes(a.universal.transposed, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!B, 1, kindB) b,
    auto ref const Slice!(RCI!A, 2, kindA) a
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(b.length!0 == a.length!0, "The length of `b` must match the first dimension of `a`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(scopeB, scopeA);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(B)*, 1, kindB) b,
    auto ref const Slice!(RCI!A, 2, kindA) a
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(b.length!0 == a.length!0, "The length of `b` must match the first dimension of `a`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .mtimes(b, scopeA);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!B, 1, kindB) b,
    Slice!(const(A)*, 2, kindA) a
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(b.length!0 == a.length!0, "The length of `b` must match the first dimension of `a`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .mtimes(scopeB, a);
}

/++
Params:
    a = 1(rows) x n(cols) vector
    b = n(rows) x 1(cols) vector
Result:
    dot product
+/
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The length of `a` must match the length of `b`");
}
do
{
    import mir.blas: dot;

    return dot(a, b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The length of `a` must match the length of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return mtimes(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The length of `a` must match the length of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return mtimes(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The length of `a` must match the length of `b`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return mtimes(a, scopeB);
}

/// Matrix-matrix multiplication (real)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.allocation: mininitRcslice;

    auto a = mininitRcslice!double(3, 5);
    auto b = mininitRcslice!double(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9,  8,  3, -2]];

    assert(mtimes(a, b) ==
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]]);

    assert(mtimes(b.transposed, a.transposed) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);
}

// test mixed strides
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.allocation: mininitRcslice;

    auto a = mininitRcslice!double(3, 5);
    auto b = mininitRcslice!double(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9,  8,  3, -2]];
         
    auto at = mininitRcslice!double(5, 3);
    at[] =
    [[-5, -1, -5],
     [ 1, -5, -2],
     [ 7,  6, -3],
     [ 7,  3,  6],
     [-4, -3,  0]];
     assert(mtimes(b.transposed, at) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);

    auto bt = mininitRcslice!double(4, 5);
    bt[] =
    [[-5, 4, -4, -1,  9],
     [-3, 3, -2,  9,  8],
     [ 3, 6, -2,  4,  3],
     [ 1, 4,  2,  8, -2]];
     assert(mtimes(bt, a.transposed) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);
}

/// Matrix-matrix multiplication (complex)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.complex: Complex;

    auto a = mininitRcslice!(Complex!double)(3, 5);
    auto b = mininitRcslice!(Complex!double)(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9, 8,  3, -2]];

    assert(mtimes(a, b) ==
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]]);
}

/// Matrix-matrix multiplication, specialization for MxN times Nx1
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[3.0, 5, 2, -3], [-2.0, 2, 3, 10], [0.0, 2, 1, 1]];
    static immutable b = [2.0, 3, 4, 5];
    static immutable c = [14.0, 64, 15];

    auto X = mininitRcslice!double(3, 4);
    auto y = mininitRcslice!double(4);
    auto result = mininitRcslice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimes(y);
    assert(Xy.equal(result));
    auto yXT = y.mtimes(X.transposed);
    assert(yXT.equal(result));
}

/// Reference-counted dot product
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [-5.0,  1,  7,  7, -4];
    static immutable b = [ 4.0, -4, -2, 10,  4];

    auto x = mininitRcslice!double(5);
    auto y = mininitRcslice!double(5);

    x[] = a;
    y[] = b;

    assert(x.mtimes(y) == 16);
}

/// Mix slice & RC dot product
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;

    static immutable a = [-5.0,  1,  7,  7, -4];
    static immutable b = [ 4.0, -4, -2, 10,  4];

    auto x = mininitRcslice!double(5);
    auto y = b.sliced;

    x[] = a;

    assert(x.mtimes(y) == 16);
    assert(y.mtimes(x) == 16);
}

/++
Symmetric matrix multiplication. Allocates result to using Mir refcounted arrays.

Similar to `mtimes`, but allows for the `a` parameter to be symmetric.

Params:
    uplo = controls whether `a` is upper symmetric or lower symmetric
+/
template mtimesSymmetric(Uplo uplo = Uplo.Upper)
{
    /+
    Params:
        a = m(rows) x m(cols) symmetric matrix
        b = m(rows) x n(cols) matrix
    Result:
        m(rows) x n(cols)
    +/
    Slice!(RCI!T, 2) mtimesSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` assumed to be a square matrix");
    }
    out (c)
    {
        assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
        assert(c.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
    }
    do
    {
        import mir.blas: symm;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(a.length!0, b.length!1);
        symm(Side.Left, uplo, cast(T)1, a, b, cast(T)0, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` assumed to be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(scopeA, scopeB);
    }

    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` assumed to be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` assumed to be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(a, scopeB);
    }

    /++
    Params:
        a = m(rows) x m(cols) symmetric matrix
        b = n(rows) x 1(cols) vector
    Result:
        m(rows) x 1(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimesSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    out (c)
    {
        assert(c.length == a.length);
    }
    do
    {
        import mir.blas: symv;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(a.length!0);
        symv(uplo, cast(T)1, a, b, cast(T)0, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetric!(uplo)(a, scopeB);
    }
}

/// ditto
template mtimesSymmetric(string uplo)
{
    mixin("alias mtimesSymmetric = .mtimesSymmetric!(Uplo." ~ uplo ~ ");");
}

/// Symmetric Matrix-Matrix multiplication
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable b = [[2.0, 3], [4.0, 3], [0.0, -5]];
    static immutable c = [[26.0, 14], [18.0, 6], [16.0, 10]];

    auto X = mininitRcslice!double(3, 3);
    auto Y = mininitRcslice!double(3, 2);
    auto result = mininitRcslice!double(3, 2);

    X[] = a;
    Y[] = b;
    result[] = c;

    auto XY = X.mtimesSymmetric(Y);
    assert(XY.equal(result));
}

/// Symmetric Matrix, specialization for MxN times Nx1
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable b = [2.0, 3, 4];
    static immutable c = [29, 28, 17];

    auto X = mininitRcslice!double(3, 3);
    auto y = mininitRcslice!double(3);
    auto result = mininitRcslice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimesSymmetric(y);
    assert(Xy.equal(result));
}

/// Symmetric Matrix, specialization for MxN times Nx1 (GC version)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;

    static immutable a = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable b = [2.0, 3, 4];
    static immutable c = [29, 28, 17];

    auto X = uninitSlice!double(3, 3);
    auto y = uninitSlice!double(3);
    auto result = uninitSlice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimesSymmetric(y);
    assert(Xy.equal(result));
}

/++
Symmetric matrix multiplication. Allocates result to using Mir refcounted arrays.

Similar to `mtimes`, but allows for the `a` parameter to be symmetric.

Params:
    uplo = controls whether `a` is upper symmetric or lower symmetric
+/
template mtimesSymmetricRight(Uplo uplo = Uplo.Upper)
{
    /+
    Params:
        a = m(rows) x m(cols) matrix
        b = m(rows) x n(cols) matrix
    Result:
        m(rows) x n(cols)
    +/
    Slice!(RCI!T, 2) mtimesSymmetricRight(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` assumed to be a square matrix");
    }
    out (c)
    {
        assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
        assert(c.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
    }
    do
    {
        import mir.blas: symm;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(a.length!0, b.length!1);
        symm(Side.Right, uplo, cast(T)1, b, a, cast(T)0, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` assumed to be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(scopeA, scopeB);
    }

    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` assumed to be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` assumed to be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(a, scopeB);
    }

    /++
    Params:
        a = 1(rows) x m(cols) vector
        b = m(rows) x m(cols) symmetric matrix
    Result:
        m(rows) x 1(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimesSymmetricRight(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 1, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length == b.length!0, "The length of `a` must match the second dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    out (c)
    {
        assert(c.length == a.length);
    }
    do
    {
        return .mtimesSymmetric!(uplo)(b, a);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 1, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length == b.length!0, "The length of `a` must match the second dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 1, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length == b.length!0, "The length of `a` must match the second dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesSymmetricRight(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 1, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length == b.length!0, "The length of `a` must match the second dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesSymmetricRight!(uplo)(a, scopeB);
    }
}

/// ditto
template mtimesSymmetricRight(string uplo)
{
    mixin("alias mtimesSymmetricRight = .mtimesSymmetricRight!(Uplo." ~ uplo ~ ");");
}

/// Symmetric Matrix-Matrix multiplication
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[2.0, 4, 0], [3.0, 3, -5]];
    static immutable b = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable c = [[26.0, 18, 16], [14.0, 6, 10]];

    auto X = mininitRcslice!double(2, 3);
    auto Y = mininitRcslice!double(3, 3);
    auto result = mininitRcslice!double(2, 3);

    X[] = a;
    Y[] = b;
    result[] = c;

    auto XY = X.mtimesSymmetricRight(Y);
    assert(XY.equal(result));
}

/// Symmetric Matrix, specialization for MxN times Nx1
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [2.0, 3, 4];
    static immutable b = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable c = [29, 28, 17];

    auto x = mininitRcslice!double(3);
    auto Y = mininitRcslice!double(3, 3);
    auto result = mininitRcslice!double(3);

    x[] = a;
    Y[] = b;
    result[] = c;

    auto xY = x.mtimesSymmetricRight(Y);
    assert(xY.equal(result));
}

/// Symmetric Matrix, specialization for MxN times Nx1 (GC version)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;

    static immutable a = [2.0, 3, 4];
    static immutable b = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable c = [29, 28, 17];

    auto x = uninitSlice!double(3);
    auto Y = uninitSlice!double(3, 3);
    auto result = uninitSlice!double(3);

    x[] = a;
    Y[] = b;
    result[] = c;

    auto xY = x.mtimesSymmetricRight(Y);
    assert(xY.equal(result));
}

/++
Solve systems of linear equations AX = B for X.

Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
@safe pure @nogc
Slice!(RCI!T, 2) mldivide(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b,
)
    if (isFloatingPoint!T || isComplex!T)
{
    import mir.exception: enforce;
    import mir.internal.utility: realType;
    import mir.lapack: gelsd, gelsd_wq, gesv, lapackint;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: as, canonical;
    import mir.utility: min;

    enforce!"mldivide: parameter shapes mismatch"(a.length!0 == b.length!0);

    auto rcat = a.transposed.as!T.rcslice;
    auto at = rcat.lightScope.canonical;
    auto rcbt = b.transposed.as!T.rcslice;
    auto bt = rcbt.lightScope.canonical;
    size_t info;
    if (a.length!0 == a.length!1)
    {
        auto rcipiv = at.length.mininitRcslice!lapackint;
        auto ipiv = rcipiv.lightScope;
        foreach(i; 0 .. ipiv.length)
            ipiv[i] = 0;
        info = gesv!T(at, ipiv, bt);
        //info > 0 means some diagonla elem of U is 0 so no solution
    }
    else
    {
        static if(!isComplex!T)
        {
            size_t liwork = void;
            auto lwork = gelsd_wq(at, bt, liwork);
            auto rcs = min(at.length!0, at.length!1).mininitRcslice!T;
            auto s = rcs.lightScope;
            auto rcwork = lwork.rcslice!T;
            auto work = rcwork.lightScope;
            auto rciwork = liwork.rcslice!lapackint;
            auto iwork = rciwork.lightScope;
            size_t rank = void;
            T rcond = -1;

            info = gelsd!T(at, bt, s, rcond, rank, work, iwork);
            //info > 0 means that many components failed to converge
        }
        else
        {
            size_t liwork = void;
            size_t lrwork = void;
            auto lwork = gelsd_wq(at, bt, lrwork, liwork);
            auto rcs = min(at.length!0, at.length!1).mininitRcslice!(realType!T);
            auto s = rcs.lightScope;
            auto rcwork = lwork.rcslice!T;
            auto work = rcwork.lightScope;
            auto rciwork = liwork.rcslice!lapackint;
            auto iwork = rciwork.lightScope;
            auto rcrwork = lrwork.rcslice!(realType!T);
            auto rwork = rcrwork.lightScope;
            size_t rank = void;
            realType!T rcond = -1;

            info = gelsd!T(at, bt, s, rcond, rank, work, rwork, iwork);
            //info > 0 means that many components failed to converge
        }
        bt = bt[0 .. $, 0 .. at.length!0];
    }
    enforce!"mldivide: some off-diagonal elements of an intermediate bidiagonal form did not converge to zero."(!info);
    return bt.transposed.as!T.rcslice;
}

/// ditto
@safe pure @nogc
Slice!(RCI!(Unqual!A), 2) mldivide(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
{
    auto al = a.lightScope.lightConst;
    auto bl = b.lightScope.lightConst;
    return mldivide(al, bl);
}

/// ditto
@safe pure @nogc
Slice!(RCI!(Unqual!A), 1) mldivide(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
{
    auto al = a.lightScope.lightConst;
    auto bl = b.lightScope.lightConst;
    return mldivide(al, bl);
}

/// ditto
@safe pure @nogc
Slice!(RCI!T, 1) mldivide(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindB) b,
)
    if (isFloatingPoint!T || isComplex!T)
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: flattened;

    return mldivide(a, b.sliced(b.length, 1)).flattened;
}

version(mir_stat_test_blas)
pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.complex: Complex;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;

    auto a = mininitRcslice!double(2, 2);
    a[] = [[2,3],
           [1, 4]];
    auto res = mldivide(eye(2), a);
    assert(equal!approxEqual(res, a));
    auto b = mininitRcslice!(Complex!double)(2, 2);
    b[] = [[Complex!double(2, 1),Complex!double(3, 2)],
           [Complex!double(1, 3), Complex!double(4, 4)]];
    auto cres = mldivide(eye!(Complex!double)(2), b);
    assert(cres == b);
    auto c = mininitRcslice!double(2, 2);
    c[] = [[5,3],
           [2,6]];
    auto d = mininitRcslice!double(2,1);
    d[] = [[4],
           [1]];
    auto e = mininitRcslice!double(2,1);
    e[] = [[23],
           [14]];
    res = mldivide(c, e);
    assert(equal!approxEqual(res, d));
}

version(mir_stat_test_blas)
pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.complex: Complex;
    import mir.math: fabs;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;

    auto a =  mininitRcslice!double(6, 4);
    a[] = [
        -0.57,  -1.28,  -0.39,   0.25,
        -1.93,   1.08,  -0.31,  -2.14,
        2.30,   0.24,   0.40,  -0.35,
        -1.93,   0.64,  -0.66,   0.08,
        0.15,   0.30,   0.15,  -2.13,
        -0.02,   1.03,  -1.43,   0.50,
    ].sliced(6, 4);

    auto b = mininitRcslice!double(6,1);
    b[] = [
        -2.67,
        -0.55,
        3.34,
        -0.77,
        0.48,
        4.10,
    ].sliced(6,1);

    auto x = mininitRcslice!double(4,1);
    x[] = [
        1.5339,
        1.8707,
        -1.5241,
        0.0392
    ].sliced(4,1);

    auto res = mldivide(a, b);
    assert(equal!((a, b) => fabs(a - b) < 5e-5)(res, x));

    auto ca =  mininitRcslice!(Complex!double)(6, 4);
    ca[] = [
        -0.57,  -1.28,  -0.39,   0.25,
        -1.93,   1.08,  -0.31,  -2.14,
        2.30,   0.24,   0.40,  -0.35,
        -1.93,   0.64,  -0.66,   0.08,
        0.15,   0.30,   0.15,  -2.13,
        -0.02,   1.03,  -1.43,   0.50,
    ].sliced(6, 4);

    auto cb = mininitRcslice!(Complex!double)(6,1);
    cb[] = [
        -2.67,
        -0.55,
        3.34,
        -0.77,
        0.48,
        4.10,
    ].sliced(6,1);

    auto cx = mininitRcslice!(Complex!double)(4,1);
    cx[] = [
        1.5339,
        1.8707,
        -1.5241,
        0.0392
    ].sliced(4,1);

    auto cres = mldivide(ca, cb);
    assert(equal!((a, b) => fabs(a - b) < 5e-5)(cres, x));
}


/++
Solve systems of linear equations AX = I for X, where I is the identity.
X is the right inverse of A if it exists, it's also a (Moore-Penrose) Pseudoinverse if A is invertible then X is the inverse.
Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
@safe pure @nogc
Slice!(RCI!A, 2) mlinverse(A, SliceKind kindA)(
    auto ref Slice!(RCI!A, 2, kindA) a
)
{
    auto aScope = a.lightScope.lightConst;
    return mlinverse!A(aScope);
}

@safe pure @nogc
Slice!(RCI!A, 2) mlinverse(A, SliceKind kindA)(
    Slice!(const(A)*, 2, kindA) a
)
{
    import mir.exception: enforce;
    import mir.lapack: getrf, getri, getri_wq, lapackint;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.topology: as, canonical;
    import mir.utility: min;

    auto a_i = a.as!A.rcslice;
    auto a_i_light = a_i.lightScope.canonical;
    auto rcipiv = min(a_i.length!0, a_i.length!1).mininitRcslice!lapackint;
    auto ipiv = rcipiv.lightScope;
    auto info = getrf!A(a_i_light, ipiv);
    if (info == 0)
    {
        auto rcwork = getri_wq!A(a_i_light).mininitRcslice!A;
        auto work = rcwork.lightScope;
        info = getri!A(a_i_light, ipiv, work);
    }
    enforce!"Matrix is not invertible as has zero determinant"(!info);
    return a_i;
}

@safe pure
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;

    auto a = mininitRcslice!double(2, 2);
    a[] = [[1,0],
           [0,-1]];
    auto ans = mlinverse!double(a);
    assert(equal!approxEqual(ans, a));
}

@safe pure
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;

    auto a = mininitRcslice!double(2, 2);
    a[] = [[ 0, 1],
           [-1, 0]];
    auto aInv = mininitRcslice!double(2, 2);
    aInv[] = [[0, -1],
              [1,  0]];
    auto ans = a.mlinverse;
    assert(equal!approxEqual(ans, aInv));
}

}
