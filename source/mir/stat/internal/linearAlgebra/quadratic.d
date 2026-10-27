/++
Allocation-free quadratic forms for real vectors and matrices.

`quadraticForm(x, a)` evaluates `x.transpose * a * x` using every matrix entry.
`quadraticForm(x, symmetricView(a))` uses one logical triangle and never reads
the other. Neither overload requires positive definiteness, and negative
results are valid. Both return scalars without modifying inputs.

Inputs must be same-type float, double or real ndslices with matching shapes.
Borrowed, owning, const, computed and strided views are supported. Rows and
the final reduction use compensated summation by default. Multiplication can
overflow, and cancellation can still lose accuracy; results are not clamped.
No scratch allocation or BLAS dependency is required.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.quadratic;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.stat.internal.linearAlgebra.types: Scalar, numericSlice, TrianglePolicy;
import mir.stat.internal.linearAlgebra.views: SymmetricView, symmetricView;
import mir.math.sum: Summation, Summator;
import std.traits: isFloatingPoint;

/// Whether a real vector and real matrix share an element type.
private enum compatible(X, A) = numericSlice!X && X.N == 1 &&
    numericSlice!A && A.N == 2 && is(Scalar!X == Scalar!A) && isFloatingPoint!(Scalar!X);

/++
Compute `x.transpose * a * x` without allocating.
Params:
    summation = Summation method for each matrix-vector row and the final dot product.
    x = Real vector.
    a = Square real matrix with order `x.length` and the same element type.
Returns:
    A scalar of the input element type; zero for an empty vector and matrix.

All matrix entries are read. Symmetry and positive definiteness are not assumed.
Shapes are checked with assertions. Native inputs support
`@safe pure nothrow @nogc`; computed iterators can affect these attributes.
+/
package(mir.stat) Scalar!X quadraticForm(Summation summation = Summation.kbn, X, A)(
    scope auto ref X x, scope auto ref A a)
    if (compatible!(X, A))
{
    return quadratic!(false, TrianglePolicy.upper, summation)(x, a);
}

/// A nonsymmetric matrix is used exactly as supplied.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [1, 2];
    double[4] matrix = [2, 3, 5, 7];
    assert(quadraticForm(values[].sliced, matrix[].sliced(2, 2)) == 46);
}

/++
Compute a quadratic form using the triangle carried by a symmetric view.
Params:
    summation = Summation method for each row and the final reduction.
    x = Real vector.
    a = Symmetric view with order `x.length` and the same real element type.
Returns:
    `x.transpose * a * x`, treating the unselected triangle as the reflection
    of the selected one. An empty vector and matrix produce zero.

The unselected triangle is never read. No symmetry or positive-definiteness
check is performed. There is no allocation, and inputs are unchanged.
Shapes are checked with assertions. Strided and computed inputs are supported.
+/
package(mir.stat) Scalar!X quadraticForm(Summation summation = Summation.kbn, X,
    TrianglePolicy triangle, A)(scope auto ref X x,
    scope auto ref const(SymmetricView!(triangle, A)) a)
    if (compatible!(X, A))
{
    return quadratic!(true, triangle, summation)(x, a.storage);
}

/// Ignore unused entries in a symmetric precision matrix.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [1, 2];
    double[4] precision = [0.3125, -0.125, double.nan, 0.25];
    assert(quadraticForm(values[].sliced, symmetricView(precision[].sliced(2, 2))) == 0.8125);
}

/// Carry the selected triangle in a read-only symmetric view.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.views: symmetricView;
    double[2] values = [1, 2];
    double[4] data = [4, double.nan, 2, 5];
    const a = symmetricView!(TrianglePolicy.lower)(data[].sliced(2, 2));
    assert(quadraticForm(values[].sliced, a) == 32);
    assert(quadraticForm(values[].sliced, a.transposed) == 32);
}

/// Evaluate rows in ascending column order and then reduce their products with x; symmetry reflects only stored entries.
private Scalar!X quadratic(bool symmetric, TrianglePolicy triangle, Summation summation, X, A)(
    scope auto ref X x, scope auto ref A a)
{
    assert(a.length!0 == x.length && a.length!1 == x.length,
        "Incompatible quadratic form dimensions");
    alias T = Scalar!X;
    Summator!(T, summation) total;
    foreach (i; 0 .. x.length)
    {
        Summator!(T, summation) row;
        static if (symmetric)
        {
            // Keep ascending column order and row-wise compensation, but
            // choose the stored triangle once per loop rather than per entry.
            foreach (j; 0 .. i)
            {
                static if (triangle == TrianglePolicy.upper) auto value = a[j, i];
                else auto value = a[i, j];
                row.put(value * x[j]);
            }
            foreach (j; i .. x.length)
            {
                static if (triangle == TrianglePolicy.upper) auto value = a[i, j];
                else auto value = a[j, i];
                row.put(value * x[j]);
            }
        }
        else
            foreach (j; 0 .. x.length)
                row.put(a[i, j] * x[j]);
        total.put(x[i] * row.sum);
    }
    return total.sum;
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[4] values = [1, 99, 2, 99];
        auto x = values[].sliced.universal.strided(2);
        T[6] data = [2, 3, 99, 5, 7, 99];
        auto a = data[].sliced(2, 3)[0..2,0..2].universal;
        assert(quadraticForm(x, a) == 46);
        assert(quadraticForm(x, a.transposed) == 46);
        assert(quadraticForm(x.reversed, a) == 31);
        auto owner = mininitRcslice!T(2, 2);
        owner[] = a;
        const constOwner = owner;
        assert(quadraticForm(x, constOwner) == 46);
        auto empty = mininitRcslice!T(0);
        auto emptyMatrix = mininitRcslice!T(0, 0);
        assert(quadraticForm(empty, emptyMatrix) == 0);
        assert(quadraticForm(empty, symmetricView(emptyMatrix)) == 0);
        a[1,0] = T.nan;
        assert(quadraticForm(x, symmetricView(a)) == 42);
        assert(quadraticForm(x, symmetricView!(TrianglePolicy.lower)(a.transposed)) == 42);
        auto computed = a.map!((T value) { assert(value == value); return value; });
        assert(quadraticForm(x, symmetricView(computed)) == 42);
        a[0,0] = -2; a[0,1] = 0; a[1,1] = -7;
        assert(quadraticForm(x, symmetricView(a)) == -30);
        assert(values[1] == 99 && values[3] == 99 && data[2] == 99 && data[5] == 99);
        // Compensation recovers a small term between large cancelling terms.
        T[3] ones = [1, 1, 1];
        T large = 1 / T.epsilon * 4;
        T[9] cancellation = [large, 1, -large, 0, 0, 0, 0, 0, 0];
        assert(quadraticForm(ones[].sliced, cancellation[].sliced(3,3)) == 1);
    }}
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[2] x = [1,2];
    double[6] data;
    auto wrong = data[].sliced(2,3);
    assertThrown!AssertError(quadraticForm(x[].sliced, wrong));
    assertThrown!AssertError(quadraticForm(x[].sliced, symmetricView(data[0..1].sliced(1,1))));
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.internal.linearAlgebra.views: symmetricView;
    import mir.complex: Complex;
    double[3] values = [1, 1, 1];
    double large = 4 / double.epsilon;
    // Each row is exact; only the outer reduction needs compensation.
    double[9] data = [large, 0, 0, 0, 1, 0, 0, 0, -large];
    auto x = values[].sliced;
    auto a = data[].sliced(3, 3);
    assert(quadraticForm(x, a) == 1);
    assert(quadraticForm(x, symmetricView(a)) == 1);
    assert(quadraticForm(x.map!((double v) => v), symmetricView(a)) == 1);
    assert(quadraticForm!(Summation.naive)(x, a) == 0);
    assert(quadraticForm!(Summation.naive)(x, symmetricView!(TrianglePolicy.lower)(a)) == 0);
    assert(quadraticForm!(Summation.naive)(x, symmetricView(a)) == 0);
    float[3] otherType;
    int[3] integers;
    Complex!double[3] complexValues;
    static assert(!__traits(compiles, quadraticForm(otherType[].sliced, a)));
    static assert(!__traits(compiles, quadraticForm(integers[].sliced, a)));
    static assert(!__traits(compiles, quadraticForm(complexValues[].sliced, a)));
    static assert(!__traits(compiles, quadraticForm(a, a)));
    static assert(!__traits(compiles, quadraticForm(x, x)));
}

// Preserve cancellation within each row before multiplying by the outer x.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[2] values = [2, 2];
        T large = T.max / 3;
        T[4] data = [large, -large, -large, large];
        auto x = values[].sliced;
        auto a = data[].sliced(2, 2);
        assert(quadraticForm(x, a) == 0);
        assert(quadraticForm(x, symmetricView(a)) == 0);
        assert(quadraticForm(x, symmetricView!(TrianglePolicy.lower)(a)) == 0);
    }}
}
