/++
Internal linear algebra operations with explicit result ownership.

Operations and structural views are restricted to `mir.stat` and its subpackages.
The shared `TrianglePolicy` and `DiagonalPolicy` enums are public; this package
is not otherwise a public application API.

`types` defines triangle/diagonal policies and shared numeric constraints;
`views` describes matrix structure without depending on arithmetic operations.
`backend` centralizes optional BLAS/LAPACK availability and storage checks,
while `allocation` handles result storage. Operation modules build on these
helpers: `products`, `triangular`, `quadratic`, and `cholesky`.

Import this package to use the matrix and vector multiplication APIs from
`mir.stat.internal.linearAlgebra.products` and triangular products from
`mir.stat.internal.linearAlgebra.triangular`.

For allocation-specific imports, use `mir.stat.internal.linearAlgebra.gc`,
`mir.stat.internal.linearAlgebra.rc`, or `mir.stat.internal.linearAlgebra.custom`.
These convenience modules selectively re-export existing operation names;
they do not introduce wrappers or change allocation behavior. Import views,
policies, and allocation-free operations separately as needed.

Choose result ownership independently of input ownership:
$(UL
    $(LI `mtimes` returns GC-managed storage.)
    $(LI `rcMtimes` returns reference-counted storage.)
    $(LI `makeMtimes` uses a supplied allocator; the caller disposes the result.)
    $(LI `mtimesInto` overwrites caller-provided storage without allocating.)
)
Allocated results are contiguous `Slice!(T*, N)` values for GC/custom storage
and `Slice!(RCI!T, N)` values for RC storage, where `T` is the unqualified input
element type and `N` is the output rank. GC and custom results share a type,
but custom results still require disposal with their allocator.
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

BLAS is optional. The default configuration uses built-in D kernels; compatible
native layouts can use BLAS when that dependency is available. Backend and
summation order can affect floating-point rounding.

For a square triangular operand, use the `mtimesTriangular` family on the left
or `mtimesTriangularRight` family on the right. They follow the same ownership
conventions, with `TrianglePolicy.upper`/`lower` and `DiagonalPolicy.nonUnit`/`unit` options.
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

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

/// Public structural policies shared with the distance API.
public import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
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

// Symmetric-view integration with quadratic forms and Cholesky.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.quadratic: quadraticForm;
    import mir.stat.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
    double[4] data = [4, 2, double.nan, 5];
    double[2] values = [1, 2];
    const a = symmetricView(data[].sliced(2, 2));
    assert(quadraticForm(values[].sliced, a) == 32);
    auto factor = rcCholeskyDecompose(a.transposed);
    static immutable expected = [[2.0, 0], [1.0, 2]];
    assert(factor == expected);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced, isSlice;
    import mir.ndslice.topology: map;
    import mir.ndslice.dynamic: strided;
    import mir.stat.internal.linearAlgebra.products: rcMtimes, mtimesInto;
    import mir.stat.internal.linearAlgebra.quadratic: quadraticForm;
    import mir.complex: Complex;
    double[16] data = double.nan;
    auto raw = data[].sliced(4, 4).strided!(0,1)(2,2);
    raw[0,0] = 2; raw[0,1] = 3; raw[1,1] = 4;
    auto a = triangularView(raw);
    auto s = symmetricView(raw);
    static assert(!isSlice!(typeof(a)) && !isSlice!(typeof(s)));
    static assert(!__traits(compiles, a[0,0] = 0));
    static assert(!__traits(compiles, s[0,0] = 0));
    static assert(!__traits(compiles, a.storage[0,0] = 0));
    assert(a.shape == [2,2] && a.length!0 == 2 && a.length!1 == 2);
    double[2] xs = [5,6], output;
    auto x = xs[].sliced;
    auto y = output[].sliced;
    mtimesInto(a, x, y);
    assert(output == [28,24]);
    mtimesInto(x, a, y);
    assert(output == [10,39]);
    auto twice = a.transposed.transposed;
    assert(rcMtimes(twice, x) == [28,24]);
    raw[0,1] = 7;
    assert(a[0,1] == 7 && s[1,0] == 7);
    assert(rcMtimes(a, x) == [52,24]);
    auto computed = raw.map!((double value) { assert(value == value); return value; });
    assert(rcMtimes(triangularView(computed), x) == [52,24]);
    assert(quadraticForm(x, symmetricView(computed)) == 614);
    raw[0,0] = raw[1,1] = double.nan;
    auto unit = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(computed);
    assert(unit[0,0] == 1 && unit[1,0] == 0);
    assert(rcMtimes(unit, x) == [47,6]);
    assert(rcMtimes(x, unit.transposed) == [47,6]);
    alias T = Complex!double;
    T[4] complexData = [T(2,1),T(3,-2),T.init,T(4,1)];
    auto complexView = triangularView(complexData[].sliced(2,2));
    assert(complexView.transposed[1,0] == T(3,-2));
    assert(complexView[1,0] == T(0));
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.products: mtimes, makeMtimes;
    import mir.stat.internal.linearAlgebra.cholesky: choleskyDecompose, makeCholeskyDecompose, choleskyDecomposeInto;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[4] data = [4,2,double.nan,5];
    auto raw = data[].sliced(2,2);
    auto a = triangularView(raw);
    auto s = symmetricView(raw);
    double[2] xs = [1,2];
    auto x = xs[].sliced;
    assert(mtimes(a,x) == [8,10]);
    assert(mtimes(x,a) == [4,12]);
    auto left = makeMtimes(Mallocator.instance,a,x);
    auto right = makeMtimes(Mallocator.instance,x,a);
    assert(left == [8,10] && right == [4,12]);
    Mallocator.instance.dispose(left.field);
    Mallocator.instance.dispose(right.field);
    auto factor = makeCholeskyDecompose(Mallocator.instance,s);
    assert(factor == choleskyDecompose(s));
    double[4] buffer;
    auto output = buffer[].sliced(2,2);
    choleskyDecomposeInto(s,output);
    assert(output == factor);
    Mallocator.instance.dispose(factor.field);
    auto invalid = data[].sliced(1,4);
    assertThrown!AssertError(symmetricView(invalid));
    assertThrown!AssertError(triangularView(invalid));
    assertThrown!AssertError(a[2,0]);
    assertThrown!AssertError(s[0,2]);
}

// Result types depend on operation rank and chosen ownership, not input storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.traits: isFloatingPoint;
    import mir.ndslice.slice: Slice;
    import mir.ndslice.topology: map, universal;
    import mir.ndslice.dynamic: transposed;
    import mir.rc.array: RCI;
    import mir.complex: Complex;
    import std.experimental.allocator.mallocator: Mallocator;
    static foreach (T; AliasSeq!(float, double, real, Complex!double))
    {{
        alias Vector = Slice!(T*, 1);
        alias Matrix = Slice!(T*, 2);
        alias RcVector = Slice!(RCI!T, 1);
        alias RcMatrix = Slice!(RCI!T, 2);
        Vector x;
        Matrix a;
        const RcMatrix owned;
        auto computed = a.map!((T value) => value);
        static assert(is(typeof(mtimes(x, x)) == T));
        static assert(is(typeof(rcMtimes(x, x)) == T));
        static assert(is(typeof(makeMtimes(Mallocator.instance, x, x)) == T));
        static assert(is(typeof(mtimes(a, a)) == Matrix));
        static assert(is(typeof(rcMtimes(a, a)) == RcMatrix));
        static assert(is(typeof(makeMtimes(Mallocator.instance, a, a)) == Matrix));
        static assert(is(typeof(mtimes(a, x)) == Vector));
        static assert(is(typeof(mtimes(x, a)) == Vector));
        static assert(is(typeof(rcMtimes(a, x)) == RcVector));
        static assert(is(typeof(rcMtimes(x, a)) == RcVector));
        static assert(is(typeof(makeMtimes(Mallocator.instance, a, x)) == Vector));
        static assert(is(typeof(makeMtimes(Mallocator.instance, x, a)) == Vector));
        static assert(is(typeof(mtimes(owned, a)) == Matrix));
        static assert(is(typeof(rcMtimes(computed, a.universal.transposed)) == RcMatrix));
        static assert(is(typeof(symmetricView(a)) == SymmetricView!(TrianglePolicy.upper, Matrix)));
        static assert(is(typeof(triangularView(a)) == TriangularView!(TrianglePolicy.upper, DiagonalPolicy.nonUnit, Matrix)));
        auto triangular = triangularView(owned);
        static assert(is(typeof(mtimes(triangular, x)) == Vector));
        static assert(is(typeof(rcMtimes(x, triangular)) == RcVector));
        static assert(is(typeof(makeMtimes(Mallocator.instance, triangular, a)) == Matrix));
        static assert(is(typeof(mtimesTriangular(a, x)) == Vector));
        static assert(is(typeof(rcMtimesTriangular(a, a)) == RcMatrix));
        static assert(is(typeof(makeMtimesTriangular(Mallocator.instance, a, x)) == Vector));
        static assert(is(typeof(mtimesTriangularRight(x, a)) == Vector));
        static assert(is(typeof(rcMtimesTriangularRight(a, a)) == RcMatrix));
        static assert(is(typeof(makeMtimesTriangularRight(Mallocator.instance, x, a)) == Vector));
        static if (isFloatingPoint!T)
        {
            auto symmetric = symmetricView(owned);
            static assert(is(typeof(quadraticForm(x, computed)) == T));
            static assert(is(typeof(quadraticForm(x, symmetric)) == T));
            static assert(is(typeof(choleskyDecompose(owned)) == Matrix));
            static assert(is(typeof(rcCholeskyDecompose(computed)) == RcMatrix));
            static assert(is(typeof(makeCholeskyDecompose(Mallocator.instance, a)) == Matrix));
            static assert(is(typeof(choleskyDecompose(symmetric)) == Matrix));
            static assert(is(typeof(rcCholeskyDecompose(symmetric)) == RcMatrix));
            static assert(is(typeof(makeCholeskyDecompose(Mallocator.instance, symmetric)) == Matrix));
            static assert(is(typeof(choleskySolve(owned, x)) == Vector));
            static assert(is(typeof(rcCholeskySolve(owned, x)) == RcVector));
            static assert(is(typeof(makeCholeskySolve(Mallocator.instance, owned, x)) == Vector));
            static assert(is(typeof(choleskySolve(owned, a)) == Matrix));
            static assert(is(typeof(rcCholeskySolve(owned, a)) == RcMatrix));
            static assert(is(typeof(makeCholeskySolve(Mallocator.instance, owned, a)) == Matrix));
        }
    }}
    // Explicit result aliases must not break rejection of unrelated arguments.
    static assert(!__traits(compiles, mtimes(1, 2)));
    static assert(!__traits(compiles, choleskyDecompose(1)));
    static assert(!__traits(compiles, quadraticForm(1, 2)));
}
