/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.math.internal.linearAlgebra;

static if (is(typeof({ import mir.blas; import mir.lapack; }))) {

static import cblas;
import mir.blas: Side, Uplo;
import mir.ndslice.slice: Slice, SliceKind;
import mir.rc.array: RCI;
import std.traits: isFloatingPoint, Unqual;

alias Diag = cblas.Diag;

version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.lapack;
}

/++
Given a matrix `a`, computes the matrix cross-product `a' * a`. Alternately,
given matrices `a` and `b`, computes the matrix cross-product `a' * b`.

This function uses more specialized algorithms than those used in `mtimes` where
possible.

In the vector case, the input is treated as a matrix whose first dimension has a
length of 1. Since the input is represented in memory as a C-style row vector,
that is the natural choice in this case.

Params:
    a = m(rows) x n(cols) matrix

Result:
    n(rows) x n(cols)

See_also:
    $(LINK2 https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/crossprod, R's crossprod function)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(
    Slice!(const(T)*, 2, sliceKind) a
)
    if (isFloatingPoint!T)
out (result)
{
    assert(result.length!0 == a.length!1, "The first dimension of the result must match the second dimension of the input");
    assert(result.length!0 == result.length!1, "The result must be a square matrix");
}
do
{
    import mir.algorithm.iteration: eachUploPair;
    import mir.blas: syrk;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;

    auto result = mininitRcslice!T(a.length!1, a.length!1);
    syrk(Uplo.Upper, cast(T)1, a.transposed, cast(T)0, result.lightScope);
    result.eachUploPair!((u, ref l) { l = u; });
    return result;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(
    auto ref const Slice!(RCI!T, 2, sliceKind) a
)
{
    auto scopeA = a.lightScope.lightConst;
    return .crossprod(scopeA);
}

/++
Params:
    a = m(rows) x n(cols) matrix
    b = m(rows) x p(cols) matrix

Result:
    n(rows) x p(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(
    Slice!(const(T)*, 2, sliceKind) a,
    Slice!(const(T)*, 2, sliceKind) b
)
    if (isFloatingPoint!T)
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
}
out (result)
{
    assert(result.length!0 == a.length!1, "The first dimension of the result must match the second dimension of `a`");
    assert(result.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
}
do
{
    import mir.math.internal.lubeck2: mtimes;
    import mir.ndslice.dynamic: transposed;

    return a.transposed.mtimes(b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .crossprod(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 2, sliceKind) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .crossprod(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .crossprod(a, scopeB);
}

/++
Params:
    a = m(rows) x 1(cols) vector

Result:
    m(rows) x m(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(Slice!(const(T)*, 1, sliceKind) a)
    if (isFloatingPoint!T)
out (result)
{
    assert(result.length!1 == a.length, "The second dimension of the result must match the length of the input");
    assert(result.length!0 == result.length!1, "The result must be a square matrix");
}
do
{
    import mir.algorithm.iteration: eachUploPair;
    import mir.blas: syr;
    import mir.ndslice.allocation: rcslice;

    auto result = rcslice!T([a.length, a.length], 0);
    syr(Uplo.Upper, cast(T)1, a, result.lightScope);
    result.eachUploPair!((u, ref l) { l = u; });
    return result;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(auto ref const Slice!(RCI!T, 1, sliceKind) a)
{
    auto scopeA = a.lightScope.lightConst;
    return .crossprod(scopeA);
}

/++
Params:
    a = m(rows) x 1(cols) vector
    b = m(rows) x 1(cols) vector

Result:
    m(rows) x m(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) crossprod(T, SliceKind sliceKind)(
    Slice!(const(T)*, 1, sliceKind) a,
    Slice!(const(T)*, 1, sliceKind) b
)
    if (isFloatingPoint!T)
out (result)
{
    assert(result.length!0 == a.length, "The first dimension of the result must match the length of `a`");
    assert(result.length!1 == b.length, "The second dimension of the result must match the length of `b`");
}
do
{
    import mir.algorithm.iteration: eachUploPair;
    import mir.blas: ger;
    import mir.ndslice.allocation: rcslice;

    auto result = rcslice!T([a.length, b.length], 0);
    ger(cast(T)1, a, b, result.lightScope);
    return result;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .crossprod(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    Slice!(const(B)*, 1, sliceKind) b
)
    if (is(Unqual!A == Unqual!B))
do
{
    auto scopeA = a.lightScope.lightConst;
    return .crossprod(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) crossprod(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
do
{
    auto scopeB = b.lightScope.lightConst;
    return .crossprod(a, scopeB);
}

/// crossprod
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[3.0, 5, 2, -3],
                          [-2.0, 2, 3, 10],
                          [0.0, 2, 1, 1]];
    static immutable b = [[ 1.0,  8],
                          [ 7.0, -3],
                          [-5.0,  2]];
    static immutable c = [[13.0, 11, 0, -29],
                          [11.0, 33, 18, 7],
                          [0.0, 18, 14, 25],
                          [-29.0, 7, 25, 110]];
    static immutable d = [[-11.0,  30],
                          [  9.0,  38],
                          [ 18.0,   9],
                          [ 62.0, -52]];

    auto X = mininitRcslice!double(3, 4);
    auto Y = mininitRcslice!double(3, 2);
    auto result1 = mininitRcslice!double(4, 4);
    auto result2 = mininitRcslice!double(4, 2);

    X[] = a;
    Y[] = b;
    result1[] = c;
    result2[] = d;

    auto Xcross = X.crossprod;
    assert(Xcross == result1);
    auto XYcross = X.crossprod(Y);
    assert(XYcross == result2);
}

/// crossprod (vector)
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [3.0, 5, 2, -3];
    static immutable b = [-2.0, 2, 3];
    static immutable c = [[ 9.0,  15,  6, -9],
                          [15.0,  25, 10, -15],
                          [ 6.0,  10,  4, -6],
                          [-9.0, -15, -6,  9]];
    static immutable d = [[ -6,  6,  9],
                          [-10, 10, 15],
                          [ -4,  4,  6],
                          [ 6,  -6, -9]];

    auto x = mininitRcslice!double(4);
    auto y = mininitRcslice!double(3);
    auto result1 = mininitRcslice!double(4, 4);
    auto result2 = mininitRcslice!double(4, 3);

    x[] = a;
    y[] = b;
    result1[] = c;
    result2[] = d;

    auto xcross = x.crossprod;
    assert(xcross == result1);
    auto xycross = x.crossprod(y);
    assert(xycross == result2);
}

/++
Given a matrix `a`, computes the matrix cross-product `a * a'`. Alternately,
given matrices `a` and `b`, computes the matrix cross-product `a * b'`.

This function uses more specialized algorithms than those used in `mtimes` where
possible.

In the vector case, the input is treated as a matrix whose first dimension has a
length of 1. Since the input is represented in memory as a C-style row vector,
that is the natural choice in this case.

Params:
    a = m(rows) x n(cols) matrix

Result:
    m(rows) x m(cols)

See_also:
    $(LINK2 https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/crossprod, R's crossprod function)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) tcrossprod(T, SliceKind sliceKind)(
    Slice!(const(T)*, 2, sliceKind) a
)
    if (isFloatingPoint!T)
out (result)
{
    assert(result.length!0 == a.length!0, "The first dimension of the result must match the first dimension of the input");
    assert(result.length!0 == result.length!1, "The result must be a square matrix");
}
do
{
    import mir.algorithm.iteration: eachUploPair;
    import mir.blas: syrk;
    import mir.ndslice.allocation: mininitRcslice;

    auto result = mininitRcslice!T(a.length!0, a.length!0);
    syrk(Uplo.Upper, cast(T)1, a, cast(T)0, result.lightScope);
    result.eachUploPair!((u, ref l) { l = u; });
    return result;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) tcrossprod(T, SliceKind sliceKind)(
    auto ref const Slice!(RCI!T, 2, sliceKind) a
)
{
    auto scopeA = a.lightScope.lightConst;
    return .tcrossprod(scopeA);
}

/++
Params:
    a = m(rows) x n(cols) matrix
    b = p(rows) x n(cols) matrix

Result:
    m(rows) x p(cols)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) tcrossprod(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b
)
    if (isFloatingPoint!T)
in
{
    assert(a.length!1 == b.length!1);
}
out (result)
{
    assert(result.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
    assert(result.length!1 == b.length!0, "The second dimension of the result must match the first dimension of `b`");
}
do
{
    import mir.ndslice.dynamic: transposed;
    import mir.math.internal.lubeck2: mtimes;

    return a.mtimes(b.transposed);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!1);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .tcrossprod(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!1);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .tcrossprod(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!1);
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .tcrossprod(a, scopeB);
}

/++
Params:
    a = m(rows) x 1(cols) vector

Result:
    scaler
+/
@safe pure nothrow @nogc
T tcrossprod(T, SliceKind sliceKind)(Slice!(const(T)*, 1, sliceKind) a)
    if (isFloatingPoint!T)
{
    import mir.math.internal.lubeck2: mtimes;

    return a.mtimes(a);
}

/// ditto
@safe pure nothrow @nogc
T tcrossprod(T, SliceKind sliceKind)(auto ref const Slice!(RCI!T, 1, sliceKind) a)
{
    auto scopeA = a.lightScope.lightConst;
    return .tcrossprod(scopeA);
}

/++
Params:
    a = m(rows) x 1(cols) vector
    b = p(rows) x 1(cols) vector

Result:
    scaler
+/
@safe pure nothrow @nogc
T tcrossprod(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 1, kindA) a,
    Slice!(const(T)*, 1, kindB) b
)
    if (isFloatingPoint!T)
in
{
    assert(a.length == b.length);
}
do
{
    import mir.math.internal.lubeck2: mtimes;

    return a.mtimes(b);
}

/// ditto
@safe pure nothrow @nogc
A tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .tcrossprod(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
A tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .tcrossprod(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
A tcrossprod(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .tcrossprod(a, scopeB);
}

/// tcrossprod
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[ 3.0,  5,  2, -3],
                          [-2.0,  2,  3, 10],
                          [ 0.0,  2,  1,  1]];
    static immutable b = [[-3.0, -5, 10, -3],
                          [ 4.0,  7,  6,  9]];
    static immutable c = [[ 47.0, -20, 9],
                          [-20.0, 117, 17],
                          [  9.0,  17, 6]];
    static immutable d = [[ -5.0,  32],
                          [ -4.0, 114],
                          [ -3.0,  29]];

    auto X = mininitRcslice!double(3, 4);
    auto Y = mininitRcslice!double(2, 4);
    auto result1 = mininitRcslice!double(3, 3);
    auto result2 = mininitRcslice!double(3, 2);

    X[] = a;
    Y[] = b;
    result1[] = c;
    result2[] = d;

    auto Xtcross = X.tcrossprod;
    assert(Xtcross == result1);
    auto XYtcross = X.tcrossprod(Y);
    assert(XYtcross == result2);
}

/// tcrossprod (vector)
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [3.0, 5, 2, -3];
    static immutable b = [-2.0, 2, 3, 10];

    auto x = mininitRcslice!double(4);
    auto y = mininitRcslice!double(4);

    x[] = a;
    y[] = b;

    auto xtcross = x.tcrossprod;
    assert(xtcross == 47);
    auto xytcross = x.tcrossprod(y);
    assert(xytcross == -20);
}

/++
Given a square matrix `a` and another matrix `b`, computes the quadratic form
`b' * a * b`.

Params:
    a = input `M x M` matrix
    b = input `M x N` matrix

Returns:
    `N x N` matrix

See_also:
    $(WEB en.wikipedia.org/wiki/Quadratic_form, Quadratic Form)
+/
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!T), 2) quadraticForm(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b
)
    if (isFloatingPoint!T)
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
out (result)
{
    assert(result.length!0 == b.length!1, "The first dimension of the result must match the second dimension of `b`");
    assert(result.length!0 == result.length!1, "`result` must be a square matrix");
}
do
{
    import mir.math.internal.lubeck2: mtimes;
    import mir.ndslice.dynamic: transposed;

    auto result = a.mtimes(b);
    return b.transposed.mtimes(result);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .quadraticForm(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .quadraticForm(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 2) quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .quadraticForm(a, scopeB);
}

/// ditto
@safe pure nothrow @nogc
T quadraticForm(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindB) b
)
    if (isFloatingPoint!T)
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    import mir.math.internal.lubeck2: mtimes;

    auto result = a.mtimes(b);
    return result.mtimes(b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .quadraticForm(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .quadraticForm(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A quadraticForm(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
    assert(a.length!0 == a.length!1, "`a` must be a square matrix");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .quadraticForm(a, scopeB);
}

///
version(mir_stat_test_blas)
@safe pure
unittest {
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto sigma = [[1.0,  2, 3],
                  [10.0, 9, 8],
                  [5.0,  6, 7]].fuse;
    auto w = [[1.0, 2],
              [2.0, 6],
              [4.0, 1]].fuse;
    auto result = [[317.0, 393],
                   [439.0, 579]].fuse;

    auto val = sigma.quadraticForm(w);
    assert(val == result);
}

/// Ditto, but RC
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[1.0,  2, 3],
                          [10.0, 9, 8],
                          [5.0,  6, 7]];
    static immutable b = [[1.0, 2],
                          [2.0, 6],
                          [4.0, 1]];
    static immutable c = [[317.0, 393],
                          [439.0, 579]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(3, 2);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticForm(w);
    assert(val == result);
}

// make sure it works with a transpose
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[1.0,  2, 3],
                          [10.0, 9, 8],
                          [5.0,  6, 7]];
    static immutable b = [[1.0, 2, 4],
                          [2.0, 6, 1]];
    static immutable c = [[317.0, 393],
                          [439.0, 579]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(2, 3);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticForm(w.transposed);
    assert(val == result);
}

/// quadraticForm (vector)
version(mir_stat_test_blas)
@safe pure
unittest {
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    auto sigma = [[1.0,  2, 3],
                  [10.0, 9, 8],
                  [5.0,  6, 7]].fuse;
    auto w = [1.0, 2, 4].sliced;
    double val = sigma.quadraticForm(w);
    val.should == 317;
}

/// Ditto, but RC
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: should;

    static immutable a = [[1.0,  2, 3],
                          [10.0, 9, 8],
                          [5.0,  6, 7]];
    static immutable b = [1.0, 2, 4];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(3);

    sigma[] = a;
    w[] = b;

    double val = sigma.quadraticForm(w);
    val.should == 317;
}

/++
Given a symmetric square matrix `a` and another matrix `b`, computes the
quadratic form `b' * a * b`.

Params:
    a = input `M x M` matrix
    b = input `M x N` matrix

Returns:
    `N x N` matrix

See_also:
    $(WEB en.wikipedia.org/wiki/Quadratic_form, Quadratic Form)
+/
@safe pure nothrow @nogc
template quadraticFormSymmetric(Uplo uplo = Uplo.Upper)
{
    Slice!(RCI!(Unqual!T), 2) quadraticFormSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    out (result)
    {
        assert(result.length!0 == b.length!1, "The first dimension of the result must match the second dimension of `b`");
        assert(result.length!0 == result.length!1, "`result` must be a square matrix");
    }
    do
    {
        import mir.math.internal.lubeck2: mtimes, mtimesSymmetric, mtimesSymmetricRight;
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.slice: Universal;

        static if (kindB == Universal) {
            if (b._stride!1 != 1)
            {
                auto result = b.transposed.mtimesSymmetricRight!(uplo)(a);
                return result.mtimes(b);
            }
        }
        auto result = a.mtimesSymmetric!(uplo)(b);
        return b.transposed.mtimes(result);
    }

    /// ditto
    Slice!(RCI!(Unqual!A), 2) quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(scopeA, scopeB);
    }

    /// ditto
    Slice!(RCI!(Unqual!A), 2) quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(a, scopeB);
    }

    /// ditto
    T quadraticFormSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        import mir.math.internal.lubeck2: mtimes, mtimesSymmetric;

        auto result = a.mtimesSymmetric!uplo(b);
        return result.mtimes(b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Unqual!A quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Unqual!A quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Unqual!A quadraticFormSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must be equal to the length of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .quadraticFormSymmetric!uplo(a, scopeB);
    }
}

/// ditto
template quadraticFormSymmetric(string uplo)
{
    mixin("alias quadraticFormSymmetric = .quadraticFormSymmetric!(Uplo." ~ uplo ~ ");");
}

///
version(mir_stat_test_blas)
@safe pure
unittest {
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto sigma = [[0.010, 0.0030, 0.006],
                  [0,     0.0225, 0.012],
                  [0,     0,      0.040]].fuse;
    auto w = [[0.25, 0.3],
              [0.30, 0.4],
              [0.45, 0.3]].fuse;
    auto result = [[0.01579, 0.01392],
                   [0.01392, 0.01278]].fuse;

    auto val = sigma.quadraticFormSymmetric(w);
    assert(val.equal!approxEqual(result));
}

/// Ditto, but RC
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[0.010, 0.0030, 0.006],
                          [0,     0.0225, 0.012],
                          [0,     0,      0.040]];
    static immutable b = [[0.25, 0.3],
                          [0.30, 0.4],
                          [0.45, 0.3]];
    static immutable c = [[0.01579, 0.01392],
                          [0.01392, 0.01278]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(3, 2);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticFormSymmetric(w);
    assert(val.equal!approxEqual(result));
}

// make sure it works with a transpose
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[0.010, 0.0030, 0.006],
                          [0,     0.0225, 0.012],
                          [0,     0,      0.040]];
    static immutable b = [[0.25, 0.3, 0.45],
                          [0.30, 0.4, 0.30]];
    static immutable c = [[0.01579, 0.01392],
                          [0.01392, 0.01278]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(2, 3);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticFormSymmetric(w.transposed);
    assert(val.equal!approxEqual(result));
}

/// quadraticFormSymmetric (vector)
version(mir_stat_test_blas)
@safe pure
unittest {
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto sigma = [[0.010, 0.0030, 0.006],
                  [0,     0.0225, 0.012],
                  [0,     0,      0.040]].fuse;
    auto w = [0.25, 0.3, 0.45].sliced;
    double val = sigma.quadraticFormSymmetric(w);
    val.shouldApprox == 0.01579;
}

/// Ditto, but RC
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [[0.010, 0.0030, 0.006],
                          [0,     0.0225, 0.012],
                          [0,     0,      0.040]];
    static immutable b = [0.25, 0.3, 0.45];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(3);

    sigma[] = a;
    w[] = b;

    double val = sigma.quadraticFormSymmetric(w);
    val.shouldApprox == 0.01579;
}

/// Use template parameter if using lower triangular symmetric matrix
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[0.0100, 0,      0],
                          [0.0030, 0.0225, 0],
                          [0.0060, 0.012,  0.040]];
    static immutable b = [[0.25, 0.3],
                          [0.30, 0.4],
                          [0.45, 0.3]];
    static immutable c = [[0.01579, 0.01392],
                          [0.01392, 0.01278]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(3, 2);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticFormSymmetric!"Lower"(w);
    assert(val.equal!approxEqual(result));
}

// transposed version with lower triangular symmetric matrix
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest {
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[0.0100, 0,      0],
                          [0.0030, 0.0225, 0],
                          [0.0060, 0.012,  0.040]];
    static immutable b = [[0.25, 0.3, 0.45],
                          [0.30, 0.4, 0.30]];
    static immutable c = [[0.01579, 0.01392],
                          [0.01392, 0.01278]];

    auto sigma = mininitRcslice!double(3, 3);
    auto w = mininitRcslice!double(2, 3);
    auto result = mininitRcslice!double(2, 2);

    sigma[] = a;
    w[] = b;
    result[] = c;

    auto val = sigma.quadraticFormSymmetric!"Lower"(w.transposed);
    assert(val.equal!approxEqual(result));
}

/++
Solve systems of linear equations AX = B for X where A is assumed to be a 
symmetric matrix.
+/
@safe pure @nogc
template mldivideSymmetric(Uplo uplo = Uplo.Upper)
{
    Slice!(RCI!T, 2) mldivideSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b,
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
        assert(a.length!0 == b.length!0, "`a` must conform with `b`");
    }
    do
    {
        import mir.exception: enforce;
        import mir.lapack: sysv_rook_wk, sysv_rook, lapackint;
        import mir.ndslice.allocation: mininitRcslice, rcslice;
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: as, canonical;

        auto rcat = a.transposed.as!T.rcslice;
        auto at = rcat.lightScope.canonical;
        auto rcbt = b.transposed.as!T.rcslice;
        auto bt = rcbt.lightScope.canonical;

        auto rcipiv = at.length.mininitRcslice!lapackint;
        auto ipiv = rcipiv.lightScope;
        foreach(i; 0 .. ipiv.length)
            ipiv[i] = 0;

        char uplo_ = void;
        static if (uplo == Uplo.Upper) {
            uplo_ = 'U';
        } else static if (uplo == Uplo.Lower) {
            uplo_ = 'L';
        } else {
            static assert(0);
        }

        size_t lwork = sysv_rook_wk(uplo_, at, bt);
        auto rcwork = lwork.rcslice!T;
        auto work = rcwork.lightScope;

        size_t info = sysv_rook!T(uplo_, at, ipiv, bt, work);
        //info > 0 means that many components failed to converge
        bt = bt[0 .. $, 0 .. at.length!0];

        enforce!"mldivideSymmetric: some off-diagonal elements of an intermediate bidiagonal form did not converge to zero."(!info);
        return bt.transposed.as!T.rcslice;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldivideSymmetric!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldivideSymmetric!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldivideSymmetric!uplo(al, b);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!T, 1) mldivideSymmetric(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b,
    )
        if (isFloatingPoint!T)
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: flattened;
        return .mldivideSymmetric!uplo(a, b.sliced(b.length, 1)).flattened;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldivideSymmetric!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldivideSymmetric!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideSymmetric(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldivideSymmetric!uplo(al, b);
    }
}

///
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [1.0, -1];
    static immutable b = [[2.0, 1],
                          [1.0, 3]];
    static immutable c = [0.8, -0.6];

    auto x = mininitRcslice!double(2);
    auto sigma = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2);

    x[] = a;
    sigma[] = b;
    result[] = c;

    auto y = sigma.mldivideSymmetric(x);
    assert(y.equal!approxEqual(result));
}

///
version(mir_stat_test_blas)
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [1.0, -1].sliced;
    auto sigma = [[2.0, 1],
                  [1.0, 3]].fuse;
    auto result = [0.8, -0.6].sliced;

    auto xCopy = x.rcslice;
    auto scopeXCopy = xCopy.lightScope;
    auto y = sigma.mldivideSymmetric(scopeXCopy);
    assert(y.equal!approxEqual(result));
}

/++
Calculates dot product of two similarly matrices by row.

Allocates result to using Mir refcounted arrays.

Params:
    a = m(rows) x n(cols) matrix
    b = m(rows) x n(cols) matrix
Result:
    m(rows) vector
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 1) rowsDotProduct(T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
    assert(a.length!1 == b.length!1, "The second dimension of `a` must match the second dimension of `b`");
}
out (c)
{
    assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
}
do
{
    import mir.math.internal.lubeck2: mtimes;
    import mir.ndslice.allocation: mininitRcslice;

    size_t m = a.length!0;
    auto c = mininitRcslice!T(m);
    foreach (i; 0..m) {
        c[i] = a[i].mtimes(b[i]);
    }
    return c;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) rowsDotProduct(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
    assert(a.length!1 == b.length!1, "The second dimension of `a` must match the second dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return .rowsDotProduct(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) rowsDotProduct(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    Slice!(const(B)*, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
    assert(a.length!1 == b.length!1, "The second dimension of `a` must match the second dimension of `b`");
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return .rowsDotProduct(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!(Unqual!A), 1) rowsDotProduct(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length!0 == b.length!0, "The first dimension of `a` must match the first dimension of `b`");
    assert(a.length!1 == b.length!1, "The second dimension of `a` must match the second dimension of `b`");
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return .rowsDotProduct(a, scopeB);
}

///
version(mir_stat_test_blas)
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto X = [[-5.0,  1],
              [ 0.0,  7],
              [ 7.0, -4]].fuse;
    auto Y = [[ 4.0, -4],
              [-2.0, 10],
              [ 4.0,  1]].fuse;
    auto z = [-24.0, 70, 24].sliced;

    assert(X.rowsDotProduct(Y).equal(z));
}

/// Reference-counted row dot product
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[-5.0,  1],
                          [ 0.0,  7],
                          [ 7.0, -4]];
    static immutable b = [[ 4.0, -4],
                          [-2.0, 10],
                          [ 4.0,  1]];
    static immutable c = [-24.0, 70, 24];

    auto X = mininitRcslice!double(3, 2);
    auto Y = mininitRcslice!double(3, 2);
    auto z = mininitRcslice!double(3);

    X[] = a;
    Y[] = b;
    z[] = c;

    assert(X.rowsDotProduct(Y).equal(z));
}

/// Mix slice & RC row dot product
version(mir_stat_test_blas)
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.fuse: fuse;

    static immutable a = [[-5.0,  1],
                          [ 0.0,  7],
                          [ 7.0, -4]];
    static immutable b = [[ 4.0, -4],
                          [-2.0, 10],
                          [ 4.0,  1]];
    static immutable c = [-24.0, 70, 24];

    auto X = mininitRcslice!double(3, 2);
    auto Y = b.fuse;
    auto z = mininitRcslice!double(3);

    X[] = a;
    z[] = c;

    assert(X.rowsDotProduct(Y).equal(z));
    assert(Y.rowsDotProduct(X).equal(z));
}

/++
Solve systems of linear equations AX = B for X where A is assumed to be a 
positive definite matrix.
+/
@safe pure @nogc
template mldividePositiveDefinite(Uplo uplo = Uplo.Upper)
{
    Slice!(RCI!T, 2) mldividePositiveDefinite(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b,
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
        assert(a.length!0 == b.length!0, "`a` must conform with `b`");
    }
    do
    {
        import mir.exception: enforce;
        import mir.lapack: posv;
        import mir.ndslice.allocation: mininitRcslice, rcslice;
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: as, canonical;

        auto rcat = a.transposed.as!T.rcslice;
        auto at = rcat.lightScope.canonical;

        // Need to make a copy of b since posv overwrites it
        size_t m = b.length!0;
        size_t n = b.length!1;
        auto bCopy = mininitRcslice!T(m, n);
        foreach (size_t i; 0 .. m) {
            foreach (size_t j; 0 .. n) {
                bCopy[i, j] = b[i, j];
            }
        }
        auto rcbt = bCopy.transposed.as!T.rcslice;
        auto bt = rcbt.lightScope.canonical;

        char uplo_ = void;
        static if (uplo == Uplo.Upper) {
            uplo_ = 'U';
        } else static if (uplo == Uplo.Lower) {
            uplo_ = 'L';
        } else {
            static assert(0);
        }

        size_t info = posv!T(uplo_, at, bt);
        //info > 0 means that many components failed to converge
        bt = bt[0 .. $, 0 .. at.length!0];

        enforce!"mldividePositiveDefinite: some off-diagonal elements of an intermediate bidiagonal form did not converge to zero."(!info);
        return bt.transposed.as!T.rcslice;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(al, b);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!T, 1) mldividePositiveDefinite(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b,
    )
        if (isFloatingPoint!T)
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: flattened;
        return .mldividePositiveDefinite!uplo(a, b.sliced(b.length, 1)).flattened;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldividePositiveDefinite(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldividePositiveDefinite!uplo(al, b);
    }
}

///
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [1.0, -1];
    static immutable b = [[2.0, 1],
                          [1.0, 3]];
    static immutable c = [0.8, -0.6];

    auto x = mininitRcslice!double(2);
    auto sigma = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2);

    x[] = a;
    sigma[] = b;
    result[] = c;

    auto y = sigma.mldividePositiveDefinite(x);
    assert(y.equal!approxEqual(result));
}

///
version(mir_stat_test_blas)
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [1.0, -1].sliced;
    auto sigma = [[2.0, 1],
                  [1.0, 3]].fuse;
    auto result = [0.8, -0.6].sliced;

    auto xCopy = x.rcslice;
    auto scopeXCopy = xCopy.lightScope;
    auto y = sigma.mldividePositiveDefinite(scopeXCopy);
    assert(y.equal!approxEqual(result));
}

/++
Solve systems of linear equations AX = B for X where A is assumed to be the
Cholesky decomposition of a positive definite matrix.
+/
@safe pure nothrow @nogc
template mldivideCholesky(Uplo uplo = Uplo.Upper)
{
    Slice!(RCI!T, 2) mldivideCholesky(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b,
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
        assert(a.length!0 == b.length!0, "`a` must conform with `b`");
    }
    do
    {
        import mir.exception: enforce;
        import mir.lapack: potrs;
        import mir.ndslice.allocation: mininitRcslice, rcslice;
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: as, canonical;

        auto rcat = a.transposed.as!T.rcslice;
        auto at = rcat.lightScope.canonical;

        // Need to make a copy of b since potrs overwrites it
        size_t m = b.length!0;
        size_t n = b.length!1;
        auto bCopy = mininitRcslice!T(m, n);
        foreach (size_t i; 0 .. m) {
            foreach (size_t j; 0 .. n) {
                bCopy[i, j] = b[i, j];
            }
        }
        auto rcbt = bCopy.transposed.as!T.rcslice;
        auto bt = rcbt.lightScope.canonical;

        char uplo_ = void;
        static if (uplo == Uplo.Upper) {
            uplo_ = 'U';
        } else static if (uplo == Uplo.Lower) {
            uplo_ = 'L';
        } else {
            static assert(0);
        }

        size_t info = potrs!T(uplo_, at, bt);
        //info > 0 means that many components failed to converge
        bt = bt[0 .. $, 0 .. at.length!0];
        // NOTE: originally had enforce message on info, but removed after trying various options and not being able to trigger it.
        return bt.transposed.as!T.rcslice;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldivideCholesky!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldivideCholesky!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 2) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldivideCholesky!uplo(al, b);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!T, 1) mldivideCholesky(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b,
    )
        if (isFloatingPoint!T)
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: flattened;
        return .mldivideCholesky!uplo(a, b.sliced(b.length, 1)).flattened;
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto al = a.lightScope.lightConst;
        auto bl = b.lightScope.lightConst;
        return .mldivideCholesky!uplo(al, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
    {
        auto bl = b.lightScope.lightConst;
        return .mldivideCholesky!uplo(a, bl);
    }

    /// ditto
    @safe pure @nogc
    Slice!(RCI!(Unqual!A), 1) mldivideCholesky(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b,
    )
    {
        auto al = a.lightScope.lightConst;
        return .mldivideCholesky!uplo(al, b);
    }
}

///
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [1.0, -1];
    static immutable b = [[sqrt(2.0), sqrt(2.0) / 2],
                          [      0.0,     sqrt(2.5)]];
    static immutable c = [0.8, -0.6];

    auto x = mininitRcslice!double(2);
    auto sigmaCholesky = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2);

    x[] = a;
    sigmaCholesky[] = b;
    result[] = c;

    auto y = sigmaCholesky.mldivideCholesky(x);
    assert(y.equal!approxEqual(result));
}

///
version(mir_stat_test_blas)
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [1.0, -1].sliced;
    auto sigmaCholesky = [[sqrt(2.0), sqrt(2.0) / 2],
                          [      0.0,     sqrt(2.5)]].fuse;
    auto result = [0.8, -0.6].sliced;

    auto xCopy = x.rcslice;
    auto scopeXCopy = xCopy.lightScope;
    auto y = sigmaCholesky.mldivideCholesky(scopeXCopy);
    assert(y.equal!approxEqual(result));
}

/++
Triangular matrix multiplication. Allocates result to using Mir refcounted arrays.

Similar to `mtimes`, but allows for the `a` parameter to be triangular.

Params:
    uplo = controls whether `a` is upper triangular or lower triangular
    diag = controls whether `a` is is non-unit triangular or unit triangular
+/
template mtimesTriangular(Uplo uplo = Uplo.Upper, Diag diag = Diag.NonUnit)
{
    /+
    Params:
        a = m(rows) x m(cols) triangular matrix
        b = m(rows) x n(cols) matrix
    Result:
        m(rows) x n(cols)
    +/
    Slice!(RCI!T, 2) mtimesTriangular(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    out (c)
    {
        assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
        assert(c.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
    }
    do
    {
        import mir.blas: trmm;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(b.length!0, b.length!1);
        foreach (size_t i; 0 .. b.length!0) {
            foreach (size_t j; 0 .. b.length!1) {
                c[i, j] = b[i, j]; //note: purposefully filling with `b` here since trmm overwrites it
            }
        }
        trmm(Side.Left, uplo, diag, cast(T)1, a, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangular!(uplo, diag)(scopeA, scopeB);
    }

    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesTriangular!(uplo, diag)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangular!(uplo, diag)(a, scopeB);
    }

    /++
    Params:
        a = m(rows) x n(cols) matrix
        b = n(rows) x 1(cols) vector
    Result:
        m(rows) x 1(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimesTriangular(T, SliceKind kindA, SliceKind kindB)(
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
        import mir.blas: trmv;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(b.length); //need to make a copy of b in order since trmv overwrites it
        foreach (size_t i; 0 .. b.length) {
            c[i] = b[i];
        }
        trmv(uplo, diag, a, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
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
        return .mtimesTriangular!(uplo, diag)(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
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
        return .mtimesTriangular!(uplo, diag)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangular(A, B, SliceKind kindA, SliceKind kindB)(
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
        return .mtimesTriangular!(uplo, diag)(a, scopeB);
    }
}

/// ditto
template mtimesTriangular(string uplo, string diag = "NonUnit")
{
    mixin("alias mtimesTriangular = .mtimesTriangular!(Uplo." ~ uplo ~ ", Diag." ~ diag ~ ");");
}

/// Triangular Matrix-Matrix multiplication
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable b = [[2.0, 3], [4.0, 3], [0.0, -5]];
    static immutable c = [[26.0, 14], [8.0, -9], [0.0, -5]];

    auto X = mininitRcslice!double(3, 3);
    auto Y = mininitRcslice!double(3, 2);
    auto result = mininitRcslice!double(3, 2);

    X[] = a;
    Y[] = b;
    result[] = c;

    auto XY = X.mtimesTriangular(Y);
    assert(XY.equal(result));
}

/// Triangular Matrix, specialization for MxN times Nx1
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable b = [2.0, 3, 4];
    static immutable c = [29, 18, 4];

    auto X = mininitRcslice!double(3, 3);
    auto y = mininitRcslice!double(3);
    auto result = mininitRcslice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimesTriangular(y);
    assert(Xy.equal(result));
}

/// Triangular Matrix, specialization for MxN times Nx1 (GC version)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;

    static immutable a = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable b = [2.0, 3, 4];
    static immutable c = [29, 18, 4];

    auto X = uninitSlice!double(3, 3);
    auto y = uninitSlice!double(3);
    auto result = uninitSlice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimesTriangular(y);
    assert(Xy.equal(result));
}

/++
Triangular matrix multiplication. Allocates result to using Mir refcounted arrays.

Similar to `mtimesTriangular`, but allows for the `b` parameter to be triangular instead of `a`.

Params:
    uplo = controls whether `b` is upper triangular or lower triangular
    diag = controls whether `b` is is non-unit triangular or unit triangular
+/
template mtimesTriangularRight(Uplo uplo = Uplo.Upper, Diag diag = Diag.NonUnit)
{
    /+
    Params:
        a = m(rows) x n(cols) matrix
        b = n(rows) x n(cols) triangular matrix
    Result:
        m(rows) x n(cols)
    +/
    Slice!(RCI!T, 2) mtimesTriangularRight(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    out (c)
    {
        assert(c.length!0 == a.length!0, "The first dimension of the result must match the first dimension of `a`");
        assert(c.length!1 == b.length!1, "The second dimension of the result must match the second dimension of `b`");
    }
    do
    {
        import mir.blas: trmm;
        import mir.ndslice.allocation: mininitRcslice;

        auto c = mininitRcslice!T(a.length!0, a.length!1);
        foreach (size_t i; 0 .. a.length!0) {
            foreach (size_t j; 0 .. a.length!1) {
                c[i, j] = a[i, j]; //note: purposefully filling with `a` here since trmm overwrites it
            }
        }
        trmm(Side.Right, uplo, diag, cast(T)1, b, c.lightScope);
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(scopeA, scopeB);
    }

    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0, "The second dimension of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(a, scopeB);
    }

    /++
    Params:
        a = 1(rows) x n(cols) vector
        b = n(rows) x n(cols) triangular matrix
    Result:
        1(rows) x m(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimesTriangularRight(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 1, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T || isComplex!T)
    in
    {
        assert(a.length!0 == b.length!0, "The length of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    out (result)
    {
        assert(result.length!0 == a.length!0, "The length the result must match the length of `a`");
    }
    do
    {
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: universal;
        static if (uplo == Uplo.Upper) {
            return mtimesTriangular!(Uplo.Lower)(b.universal.transposed, a); //NOTE: switching Uplo.Lower to Uplo.Upper because of the transpose
        } else {
            return mtimesTriangular!(Uplo.Upper)(b.universal.transposed, a); //reverse
        }
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 1, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length!0, "The length of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 1, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length!0, "The length of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimesTriangularRight(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 1, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length!0, "The length of `a` must match the first dimension of `b`");
        assert(b.length!0 == b.length!1, "`b` must be a square matrix");
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimesTriangularRight!(uplo, diag)(a, scopeB);
    }
}

/// ditto
template mtimesTriangularRight(string uplo, string diag = "NonUnit")
{
    mixin("alias mtimesTriangularRight = .mtimesTriangularRight!(SUplo." ~ uplo ~ ", Diag." ~ diag ~ ");");
}

/// Triangular Matrix-Matrix multiplication
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice, rcslice;
    import mir.ndslice.dynamic: transposed;

    static immutable a = [[2.0, 4, 0], [3.0, 3, -5]];
    static immutable b = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable c = [[6.0, 18, 16], [9.0, 21, 10]];

    auto X = mininitRcslice!double(2, 3);
    auto Y = mininitRcslice!double(3, 3);
    auto result = mininitRcslice!double(2, 3);

    X[] = a;
    Y[] = b;
    result[] = c;

    auto XY = X.mtimesTriangularRight(Y);
    assert(XY.equal(result));
}

/// Triangular Matrix, specialization for MxN times Nx1
version(mir_stat_test_blas)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [2.0, 3, 4];
    static immutable b = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable c = [6, 16, 17];

    auto x = mininitRcslice!double(3);
    auto Y = mininitRcslice!double(3, 3);
    auto result = mininitRcslice!double(3);

    x[] = a;
    Y[] = b;
    result[] = c;

    auto xY = x.mtimesTriangularRight(Y);
    assert(xY.equal(result));
}

/// Triangular Matrix, specialization for MxN times Nx1 (GC version)
version(mir_stat_test_blas)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;

    static immutable a = [2.0, 3, 4];
    static immutable b = [[3.0, 5, 2], [0.0, 2, 3], [0.0, 0, 1]];
    static immutable c = [6, 16, 17];

    auto x = uninitSlice!double(3);
    auto Y = uninitSlice!double(3, 3);
    auto result = uninitSlice!double(3);

    x[] = a;
    Y[] = b;
    result[] = c;

    auto xY = x.mtimesTriangularRight(Y);
    assert(xY.equal(result));
}

/++
Computes the Cholesky decomposition of symmetric positive definite matrix 'A' in place.

The factorization has the form:
    \A = U**T * U, if UPLO = Uplo.Upper, or
    \A = L * L**T, if UPLO = Uplo.Lower

Where U is an upper triangular matrix and L is lower triangular.

Params:
    uplo = if uplo is Upper, then upper triangle of A is stored, else
    lower.
+/
@safe pure @nogc
template choleskyDecomposeInPlace(Uplo uplo = Uplo.Upper)
{
    /++
    Params:
        a = symmetric, positive definite 'N x N' matrix.
    +/
    void choleskyDecomposeInPlace(T)(
        Slice!(T*, 2, SliceKind.canonical) a
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        import mir.algorithm.iteration: eachUploPair;
        import mir.exception: enforce;
        import mir.lapack: potrf;
        import mir.ndslice.dynamic: transposed;
        import mir.ndslice.topology: universal;

        auto info = potrf!T(uplo == Uplo.Upper ? 'L' : 'U', a);
        enforce!"choleskyDecomposeInPlace: one of the leading minors is not positive definite and the factorization could not be completed."(!info);
        auto d = a.universal;
        static if (uplo == Uplo.Upper) {
            d = d.transposed;
        }
        d.eachUploPair!("a = 0");
    }
    
    void choleskyDecomposeInPlace(T)(
        Slice!(RCI!(T), 2, SliceKind.canonical) a
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope;
        scopeA.choleskyDecomposeInPlace;
    }
}

/// Upper cholesky decomposition (in place)
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.topology: assumeCanonical;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 1],
                          [0.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2).assumeCanonical;
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    auto y = x.lightScope;
    y.choleskyDecomposeInPlace;
    assert(y.all!approxEqual(result));
}

/// Lower cholesky decomposition (in place)
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.blas: Uplo;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.topology: assumeCanonical;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 0],
                          [1.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2).assumeCanonical;
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    auto y = x.lightScope;
    y.choleskyDecomposeInPlace!(Uplo.Lower);
    assert(y.all!approxEqual(result));
}

/++
Computes the Cholesky decomposition of symmetric positive definite matrix 'A'.

The factorization has the form:
    \A = U**T * U, if UPLO = Uplo.Upper, or
    \A = L * L**T, if UPLO = Uplo.Lower

Where U is an upper triangular matrix and L is lower triangular.

Params:
    uplo = if uplo is Upper, then upper triangle of A is stored, else
    lower.
+/
@safe pure @nogc
template choleskyDecompose(Uplo uplo = Uplo.Upper)
{
    /++
    Params:
        a = symmetric, positive definite 'N x N' matrix.
    Returns:
        the Cholesky decomposition of `a`
    +/
    Slice!(RCI!(Unqual!T), 2) choleskyDecompose(T, SliceKind kind)(
        Slice!(const(T)*, 2, kind) a
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        import mir.ndslice.allocation: mininitRcslice;
        import mir.ndslice.topology: assumeCanonical;

        auto result = mininitRcslice!T(a.length!0, a.length!1);
        /++ Only need to fill upper or lower part of matrix
            Note: When calling `void choleskyDecompose!Uplo(a)`,
            it impliclty assumes all the data is there and then tries to avoid
            unncessary tranposing to work with lapack. Thus, here we are filling
            the lower and upper instead of upper and lower.
            Iterates through the matrix differently if the stride of the second
            dimension is not equal to 1 (implicitly assuming it has been transposed)
        +/
        if (a._stride!1 == 1) {
            foreach (size_t i; 0 .. a.length!0) {
                static if (uplo == Uplo.Lower) {
                    foreach (size_t j; 0 .. a.length!1) {
                        result[i, j] = a[i, j];
                    }
                } else {
                    foreach (size_t j; i .. a.length!1) {
                        result[i, j] = a[i, j];
                    }
                }
            }
        } else {
            foreach (size_t j; 0 .. a.length!1) {
                static if (uplo == Uplo.Lower) {
                    foreach (size_t i; j .. a.length!0) {
                        result[i, j] = a[i, j];
                    }
                } else {
                    foreach (size_t i; 0 .. a.length!0) {
                        result[i, j] = a[i, j];
                    }
                }
            }
        }
        choleskyDecomposeInPlace!uplo(result.assumeCanonical);
        return result;
    }

    /// ditto
    Slice!(RCI!(Unqual!T), 2) choleskyDecompose(T, SliceKind kind)(
        auto ref const Slice!(RCI!T, 2, kind) a
    )
        if (isFloatingPoint!T)
    in
    {
        assert(a.length!0 == a.length!1, "`a` must be a square matrix");
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .choleskyDecompose!uplo(scopeA);
    }
}

/// Upper cholesky decomposition
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 1],
                          [0.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    import std.stdio: writeln;
    auto y = x.choleskyDecompose;
    assert(y.all!approxEqual(result));
}

/// Lower cholesky decomposition
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.blas: Uplo;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 0],
                          [1.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    auto y = x.choleskyDecompose!(Uplo.Lower);
    assert(y.all!approxEqual(result));
}

/// Upper cholesky decomposition (universal, transposed)
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: universal;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 1],
                          [0.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    auto y = x.universal.transposed;
    auto z = y.choleskyDecompose;
    assert(z.all!approxEqual(result));
}

/// Lower cholesky decomposition (universal, transposed)
version(mir_stat_test_blas)
@safe pure @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.blas: Uplo;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: universal;

    static immutable a = [[4.0, 2],
                          [2.0, 4]];
    static immutable b = [[2.0, 0],
                          [1.0, sqrt(3.0)]];

    auto x = mininitRcslice!double(2, 2);
    auto result = mininitRcslice!double(2, 2);

    x[] = a;
    result[] = b;

    auto y = x.universal.transposed;
    auto z = y.choleskyDecompose!(Uplo.Lower);
    assert(z.all!approxEqual(result));
}

}