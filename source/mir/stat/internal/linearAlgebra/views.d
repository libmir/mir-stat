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
module mir.stat.internal.linearAlgebra.views;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.ndslice.slice: isSlice;
import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
import std.traits: Unqual;

private enum matrix(A) = isSlice!A && A.N == 2;
private enum opposite(TrianglePolicy triangle) = triangle == TrianglePolicy.upper ? TrianglePolicy.lower : TrianglePolicy.upper;

/++
Create a symmetric view using one logical triangle of a square ndslice.
Unused entries are not read. The descriptor is copied, retaining RC ownership
when present. The original storage is not modified. This expresses symmetry,
not positive definiteness; no element-wise validation is performed.
+/
package(mir.stat) auto symmetricView(TrianglePolicy triangle = TrianglePolicy.upper, A)(A a)
    if (matrix!A)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
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
    static assert(typeof(a.transposed).triangle == TrianglePolicy.lower);
}

/// Structural symmetric matrix descriptor. Construct with `symmetricView`.
package(mir.stat) struct SymmetricView(TrianglePolicy selectedTriangle, A)
    if (matrix!A)
{
    private A _storage;
    /// TrianglePolicy supplying the logical matrix entries.
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
        return (triangle == TrianglePolicy.upper ? j >= i : j <= i)
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
read as zero. `DiagonalPolicy.unit` reads diagonal entries as one without accessing
stored diagonal values. The descriptor retains the input's ownership policy.
This is a structural view, not a claim of invertibility or a Cholesky factor.
+/
package(mir.stat) auto triangularView(TrianglePolicy triangle = TrianglePolicy.upper, DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A)(A a)
    if (matrix!A)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static assert(diagonal == DiagonalPolicy.nonUnit || diagonal == DiagonalPolicy.unit);
    assert(a.length!0 == a.length!1, "Triangular view requires a square matrix");
    return TriangularView!(triangle, diagonal, A)(a);
}

/// Implicit zeros and unit diagonals ignore poisoned storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [double.nan, 3, double.nan, double.nan];
    auto a = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(data[].sliced(2, 2));
    assert(a[0,0] == 1 && a[0,1] == 3 && a[1,0] == 0 && a[1,1] == 1);
    assert(a.transposed[1,0] == 3);
    static assert(typeof(a.transposed).triangle == TrianglePolicy.lower);
    static assert(typeof(a.transposed).diagonal == DiagonalPolicy.unit);
}

/// Structural triangular descriptor. Construct with `triangularView`.
package(mir.stat) struct TriangularView(TrianglePolicy selectedTriangle, DiagonalPolicy selectedDiagonal, A)
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
        static if (diagonal == DiagonalPolicy.unit)
            if (i == j) return T(1);
        if (triangle == TrianglePolicy.upper ? j < i : j > i) return T(0);
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


version(mir_stat_test_linearAlgebra)
@system @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
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
    assert(constView[0,0] == 2 && constView[0,1] == 3 && constView[1,1] == 4);
}
