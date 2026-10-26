/++
Read-only structural matrix views without copying matrix elements.

`symmetricView` reflects a selected triangle across the diagonal.
`triangularView` supplies zeros outside the selected triangle and, optionally,
ones on an implicit diagonal. Neither view validates positive definiteness or
identifies a Cholesky factor. Complex symmetry does not imply conjugation.

A view stores the supplied ndslice descriptor by value. Borrowed slices still
borrow their storage, while RC slices retain their existing owner. No matrix
allocation is performed. Changes through the original storage remain visible.
The caller must preserve borrowed storage and computed-iterator dependencies.

Views deliberately have no implicit ndslice conversion. Structure-aware
algorithms dispatch explicitly; `storage` exposes the raw underlying matrix
read-only, including entries outside the selected triangle. Transpose keeps
storage ownership and updates triangle metadata without copying elements.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.math.internal.linearAlgebra.views;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.ndslice.slice: isSlice;
import mir.math.internal.linearAlgebra.triangular: Triangle, Diagonal;
import std.traits: Unqual;

private enum matrix(A) = isSlice!A && A.N == 2;
private enum opposite(Triangle triangle) = triangle == Triangle.upper ? Triangle.lower : Triangle.upper;

/++
Create a symmetric view using one logical triangle of a square ndslice.
Unused entries are not read. The descriptor is copied, retaining RC ownership
when present. The original storage is not modified. This expresses symmetry,
not positive definiteness; no element-wise validation is performed.
+/
auto symmetricView(Triangle triangle = Triangle.upper, A)(A a)
    if (matrix!A)
{
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    assert(a.length!0 == a.length!1, "Symmetric view requires a square matrix");
    return SymmetricView!(triangle, A)(a);
}

/// Read either half without accessing the unused stored triangle.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 2, double.nan, 5];
    auto a = symmetricView(data[].sliced(2, 2));
    assert(a[0,1] == 2 && a[1,0] == 2);
    assert(a.transposed[1,0] == 2);
    static assert(typeof(a.transposed).triangle == Triangle.lower);
}

/// Structural symmetric matrix descriptor. Construct with `symmetricView`.
struct SymmetricView(Triangle selectedTriangle, A)
    if (matrix!A)
{
    private A _storage;
    /// Triangle supplying the logical matrix entries.
    enum triangle = selectedTriangle;
    /// Matrix rank.
    enum N = 2;
    /// Logical matrix shape.
    @property auto shape() const { return _storage.shape; }
    /// Length of a logical matrix dimension.
    size_t length(size_t dimension = 0)() const { return _storage.length!dimension; }
    /// Read an element, reflecting indices into the selected triangle.
    auto opIndex(size_t i, size_t j) const
    {
        assert(i < length!0 && j < length!1, "Symmetric view index out of bounds");
        return (triangle == Triangle.upper ? j >= i : j <= i)
            ? _storage[i,j] : _storage[j,i];
    }
    /// Raw read-only storage; unused entries retain their original values.
    @property auto storage() const { return _storage.lightConst; }
    /// Transpose both the descriptor and selected triangle, preserving ownership.
    @property auto transposed() const
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        return symmetricView!(opposite!triangle)(_storage.lightConst.universal.transposed);
    }
}

/++
Create a triangular view of a square ndslice. Entries outside `triangle`
read as zero. `Diagonal.unit` reads diagonal entries as one without accessing
stored diagonal values. The descriptor retains the input's ownership policy.
This is a structural view, not a claim of invertibility or a Cholesky factor.
+/
auto triangularView(Triangle triangle = Triangle.upper, Diagonal diagonal = Diagonal.nonUnit, A)(A a)
    if (matrix!A)
{
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    static assert(diagonal == Diagonal.nonUnit || diagonal == Diagonal.unit);
    assert(a.length!0 == a.length!1, "Triangular view requires a square matrix");
    return TriangularView!(triangle, diagonal, A)(a);
}

/// Implicit zeros and unit diagonals ignore poisoned storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [double.nan, 3, double.nan, double.nan];
    auto a = triangularView!(Triangle.upper, Diagonal.unit)(data[].sliced(2, 2));
    assert(a[0,0] == 1 && a[0,1] == 3 && a[1,0] == 0 && a[1,1] == 1);
    assert(a.transposed[1,0] == 3);
    static assert(typeof(a.transposed).triangle == Triangle.lower);
    static assert(typeof(a.transposed).diagonal == Diagonal.unit);
}

/// Structural triangular descriptor. Construct with `triangularView`.
struct TriangularView(Triangle selectedTriangle, Diagonal selectedDiagonal, A)
    if (matrix!A)
{
    private A _storage;
    /// Logical selected triangle.
    enum triangle = selectedTriangle;
    /// Stored or implicit-unit diagonal policy.
    enum diagonal = selectedDiagonal;
    /// Matrix rank.
    enum N = 2;
    /// Logical matrix shape.
    @property auto shape() const { return _storage.shape; }
    /// Length of a logical dimension.
    size_t length(size_t dimension = 0)() const { return _storage.length!dimension; }
    /// Read a logical value without touching implicit entries.
    auto opIndex(size_t i, size_t j) const
    {
        alias T = Unqual!(typeof(_storage[i,j]));
        assert(i < length!0 && j < length!1, "Triangular view index out of bounds");
        static if (diagonal == Diagonal.unit)
            if (i == j) return T(1);
        if (triangle == Triangle.upper ? j < i : j > i) return T(0);
        T value = _storage[i,j];
        return value;
    }
    /// Raw read-only storage; it does not materialize implicit entries.
    @property auto storage() const { return _storage.lightConst; }
    /// Transpose storage and triangle metadata, preserving the diagonal policy.
    @property auto transposed() const
    {
        import mir.ndslice.topology: universal;
        import mir.ndslice.dynamic: transposed;
        return triangularView!(opposite!triangle, diagonal)(_storage.lightConst.universal.transposed);
    }
}


/// Multiplication dispatches from triangular metadata without extra options.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.internal.linearAlgebra.products: rcMtimes;
    double[4] data = [2, 3, double.nan, 4];
    double[2] values = [5, 6];
    const a = triangularView(data[].sliced(2, 2));
    assert(rcMtimes(a, values[].sliced) == [28, 24]);
    assert(rcMtimes(values[].sliced, a) == [10, 39]);
}

/// A symmetric view forwards its selected triangle to Cholesky decomposition.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.internal.linearAlgebra.cholesky: rcCholeskyDecompose;
    double[4] data = [4, 2, double.nan, 5];
    double[2] values = [1, 2];
    const a = symmetricView(data[].sliced(2, 2));
    auto factor = rcCholeskyDecompose(a.transposed);
    static immutable expected = [[2.0, 0], [1.0, 2]];
    assert(factor == expected);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.ndslice.dynamic: strided;
    import mir.math.internal.linearAlgebra.products: rcMtimes, mtimesInto;
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
    raw[0,0] = raw[1,1] = double.nan;
    auto unit = triangularView!(Triangle.upper, Diagonal.unit)(computed);
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
@system @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.math.internal.linearAlgebra.products: rcMtimes;
    import mir.ndslice.slice: sliced;
    auto owner = mininitRcslice!double(2,2);
    owner[0,0] = 2; owner[0,1] = 3; owner[1,0] = double.nan; owner[1,1] = 4;
    const initialCount = owner._iterator._array._counter;
    {
        auto view = triangularView(owner);
        assert(owner._iterator._array._counter == initialCount + 1);
        auto transpose = view.transposed;
        assert(owner._iterator._array._counter == initialCount + 2);
        assert(transpose[1,0] == 3);
    }
    assert(owner._iterator._array._counter == initialCount);
    auto retained()
    {
        auto local = mininitRcslice!double(2,2);
        local[0,0] = 4; local[0,1] = 2; local[1,0] = double.nan; local[1,1] = 5;
        return symmetricView(local).transposed;
    }
    auto alive = retained();
    assert(alive[0,1] == 2 && alive[1,1] == 5);
    const constOwner = owner;
    auto constView = triangularView(constOwner);
    double[2] xs = [5,6];
    assert(rcMtimes(constView, xs[].sliced) == [28,24]);
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.internal.linearAlgebra.products: mtimes, makeMtimes;
    import mir.math.internal.linearAlgebra.cholesky: choleskyDecompose, makeCholeskyDecompose, choleskyDecomposeInto;
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
