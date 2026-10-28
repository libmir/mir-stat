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
read-only. Dense storage includes unused entries; packed storage provides
square indexing with zeros outside the stored triangle. Transpose keeps
storage ownership and updates triangle metadata without copying elements.
Both views expose a read-only logical asSlice adapter compatible with Mir's
hasAsSlice/toSlice convention. Generic forwarding reads logical values without
copying; it does not preserve structural dispatch. Use the original view for
structure-aware algorithms, or explicitly allocate a writable dense copy.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.views;

// Include this module in both the full suite and the focused BLAS suite.
version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.ndslice.slice: isSlice, Slice;
import mir.ndslice.iterator: FieldIterator;
import mir.stat.internal.linearAlgebra.types: TrianglePolicy, DiagonalPolicy;
import std.traits: Unqual, isMutable;
import mir.stat.internal.linearAlgebra.packed: isPacked, PackedInfo, PackedMatrix, packedMatrix;

/// Whether a descriptor is a rank-two ndslice.
private enum matrix(A) = isSlice!A && A.N == 2;
/// Triangle selected by a transposed structural view.
private enum opposite(TrianglePolicy triangle) = triangle == TrianglePolicy.upper ? TrianglePolicy.lower : TrianglePolicy.upper;

/++
Create a symmetric view using one logical triangle of a square ndslice.
Unused entries are not read. The descriptor is copied, retaining RC ownership
when present. The original storage is not modified. This expresses symmetry,
not positive definiteness; no element-wise validation is performed.
+/
package(mir.stat) SymmetricView!(triangle, A)
symmetricView(TrianglePolicy triangle = TrianglePolicy.upper, A)(A a)
    if (matrix!A)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    assert(a.length!0 == a.length!1, "Symmetric view requires a square matrix");
    return SymmetricView!(triangle, A)(a);
}

/++
Interpret packed stairs as symmetric storage. The stairs direction selects the
stored triangle; no matrix elements are copied. RC ownership is retained.
+/
package(mir.stat) SymmetricView!(PackedInfo!S.triangle, PackedMatrix!S)
symmetricView(S)(S rows) if (isPacked!S)
{
    return symmetricView!(PackedInfo!S.triangle)(packedMatrix(rows));
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
    @property size_t[2] shape() const { return _storage.shape; }
    /// Length of a logical matrix dimension.
    size_t length(size_t dimension = 0)() const { return _storage.length!dimension; }
    private alias Element = typeof(const(A).init[0,0]);
    /// Read an element, reflecting indices into the selected triangle.
    Element opIndex(size_t i, size_t j) const
    {
        assert(i < length!0 && j < length!1, "Symmetric view index out of bounds");
        return (triangle == TrianglePolicy.upper ? j >= i : j <= i)
            ? _storage[i,j] : _storage[j,i];
    }
    /// Read-only square storage; unused dense entries retain their values, while absent packed entries read as zero.
    @property auto storage() const { return _storage.lightConst; }
    /++
    Read-only logical ndslice adapter, reflecting the selected triangle.
    No elements are copied. RC ownership is retained; borrowed storage must
    remain alive. Use `view.asSlice.slice` or `view.asSlice.rcslice` for an
    independent writable dense copy. `makeSlice(allocator, view.asSlice)`
    provides the corresponding custom allocation. Raw `storage` is not a
    logical dense conversion: its unused triangle may be uninitialized.
    +/
    @property auto asSlice() return scope const
    {
        import mir.ndslice.slice: sliced;
        const n = length!0;
        assert(!n || n <= cast(size_t)ptrdiff_t.max / n,
            "Symmetric logical shape exceeds iterator indexing range");
        auto view = symmetricView!triangle(_storage.lightConst);
        return FieldIterator!(LogicalMatrixField!(typeof(view)))(0,
            LogicalMatrixField!(typeof(view))(view)).sliced(n, n);
    }
    /// Borrow read-only raw storage while tying its lifetime to this descriptor.
    package(mir.stat) @property auto borrowStorage() return scope const
    {
        return _storage.lightScope.lightConst;
    }
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
package(mir.stat) TriangularView!(triangle, diagonal, A)
triangularView(TrianglePolicy triangle = TrianglePolicy.upper, DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, A)(A a)
    if (matrix!A)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static assert(diagonal == DiagonalPolicy.nonUnit || diagonal == DiagonalPolicy.unit);
    assert(a.length!0 == a.length!1, "Triangular view requires a square matrix");
    return TriangularView!(triangle, diagonal, A)(a);
}

/++
Interpret packed stairs as triangular storage, inferring the triangle from the
stairs direction. `DiagonalPolicy.unit` ignores stored diagonal entries.
No elements are copied; borrowed storage stays borrowed and RC owners are retained.
+/
package(mir.stat) TriangularView!(PackedInfo!S.triangle, diagonal, PackedMatrix!S)
triangularView(DiagonalPolicy diagonal = DiagonalPolicy.nonUnit, S)(S rows) if (isPacked!S)
{
    return triangularView!(PackedInfo!S.triangle, diagonal)(packedMatrix(rows));
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
    @property size_t[2] shape() const { return _storage.shape; }
    /// Length of a logical dimension.
    size_t length(size_t dimension = 0)() const { return _storage.length!dimension; }
    private alias Element = Unqual!(typeof(const(A).init[0,0]));
    /// Read a logical value without touching implicit entries.
    Element opIndex(size_t i, size_t j) const
    {
        alias T = Element;
        assert(i < length!0 && j < length!1, "Triangular view index out of bounds");
        static if (diagonal == DiagonalPolicy.unit)
            if (i == j) return T(1);
        if (triangle == TrianglePolicy.upper ? j < i : j > i) return T(0);
        T value = _storage[i,j];
        return value;
    }
    /// Read-only square storage without materialization; packed storage supplies zeros for absent entries.
    @property auto storage() const { return _storage.lightConst; }
    /++
    Read-only logical ndslice adapter with implicit zeros and unit diagonals.
    No elements are copied. RC ownership is retained; borrowed storage and
    computed dependencies must remain alive. Use .asSlice.slice, .asSlice.rcslice,
    or makeSlice(allocator, view.asSlice) for independent writable dense copies.
    Pass the structural view directly to linear algebra operations to retain
    triangular dispatch; this adapter exposes ordinary logical matrix indexing.
    +/
    @property auto asSlice() return scope const
    {
        import mir.ndslice.slice: sliced;
        const n = length!0;
        assert(!n || n <= cast(size_t)ptrdiff_t.max / n,
            "Triangular logical shape exceeds iterator indexing range");
        auto view = triangularView!(triangle, diagonal)(_storage.lightConst);
        return FieldIterator!(LogicalMatrixField!(typeof(view)))(0,
            LogicalMatrixField!(typeof(view))(view)).sliced(n, n);
    }
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
    /// Return a view whose RC storage must survive the local owner.
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

// Symmetric indexing preserves stored qualifiers; triangular values also
// represent implicit zero/one entries and remain unqualified values.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: Slice;
    import mir.ndslice.topology: map;
    import mir.rc.array: RCI;
    import mir.complex: Complex;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Complex!double))
    {{
        static foreach (Matrix; AliasSeq!(Slice!(T*, 2), Slice!(const(T)*, 2), Slice!(RCI!T, 2)))
        {{
            Matrix matrix;
            static assert(is(typeof(symmetricView(matrix)[0,0]) == const(T)));
            static assert(is(typeof(triangularView(matrix)[0,0]) == T));
        }}
        Slice!(T*, 2) matrix;
        auto computed = matrix.map!((T value) => value);
        static assert(is(typeof(symmetricView(computed)[0,0]) == T));
        static assert(is(typeof(triangularView(computed)[0,0]) == T));
    }}
}
/// Recognize structural symmetry without interpreting ordinary slices as symmetric.
package(mir.stat) template SymmetricInfo(V)
{
    static if (is(Unqual!V : SymmetricView!(triangle, A), TrianglePolicy triangle, A))
    {
        enum valid = true;
        enum selectedTriangle = triangle;
        alias Storage = A;
    }
    else enum valid = false;
}

/// Structural descriptors must be dispatched before generic asSlice conversion.
package(mir.stat) enum isStructuralView(V) = SymmetricInfo!V.valid ||
    is(Unqual!V : TriangularView!(triangle, diagonal, A),
        TrianglePolicy triangle, DiagonalPolicy diagonal, A);

/// Logical flat indexing retains the descriptor and any RC owner it carries.
private struct LogicalMatrixField(V)
{
    V view;
    /// Read through the structural descriptor, including reflected or implicit entries.
    Unqual!(typeof(V.init[0,0])) opIndex(ptrdiff_t index) const
    {
        const n = view.length!0;
        assert(index >= 0 && n && cast(size_t)index / n < n);
        return view[cast(size_t)index / n, cast(size_t)index % n];
    }
    /// Preserve read-only access and ownership when ndslice copies the adapter.
    auto lightConst()() const @property { return this; }
}

/++
Release the original custom-allocated symmetric result using its allocator.
Call exactly once on the original contiguous result, after all derived views
are finished. The allocator must be the one that created its entire storage.
This is manual disposal, not reference counting: copied descriptors and other
views become invalid. The supplied result is cleared after successful disposal.
Borrowed/GC storage and cropped views must never be passed. Transposed,
const-element, and RC descriptors are rejected by the type constraint.
Like `allocator.dispose(result.field)` for ordinary custom results, provenance
is a caller precondition; it is not encoded by a pointer-backed slice type.
+/
package(mir.stat) void disposeSymmetric(Allocator, TrianglePolicy triangle, T)(
    auto ref Allocator allocator, ref SymmetricView!(triangle, Slice!(T*, 2)) result)
    if (isMutable!T)
{
    import std.experimental.allocator: dispose;
    allocator.dispose(result._storage.field);
    result = typeof(result).init;
}

/// Logical copies initialize both triangles; the adapter itself does not copy data.
version(mir_stat_test_linearAlgebra)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: slice, rcslice;
    double[4] data = [4,2,double.nan,5];
    auto g = symmetricView(data[].sliced(2,2));
    auto dense = g.asSlice.slice;
    auto owned = g.transposed.asSlice.rcslice;
    assert(dense == [[4,2],[2,5]] && owned == dense);
    dense[0,0] = 100;
    assert(g[0,0] == 4);
    assert(g.asSlice[0..1,0..2] == [[4,2]]);
    static assert(!__traits(compiles, g.asSlice[0,0] = 1));
}

/// Dispose original custom storage, not a read-only or transposed view of it.
version(mir_stat_test_linearAlgebra)
@system @nogc unittest
{
    import mir.ndslice.allocation: makeUninitSlice, makeSlice;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    auto storage = makeUninitSlice!double(Mallocator.instance,2,2);
    storage[0,0]=4; storage[0,1]=2; storage[1,0]=double.nan; storage[1,1]=5;
    auto original = symmetricView(storage);
    auto dense = makeSlice(Mallocator.instance, original.asSlice);
    assert(dense[1,0] == 2);
    Mallocator.instance.dispose(dense.field);
    static assert(!__traits(compiles, disposeSymmetric(Mallocator.instance,original.transposed)));
    disposeSymmetric(Mallocator.instance, original);
    assert(original.shape == [0,0]);
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;
    /// Logical adapters keep the RC owner after the factory's local reference dies.
    auto retained() {
        auto a=mininitRcslice!double(2,2);
        a[0,0]=4; a[0,1]=2; a[1,0]=double.nan; a[1,1]=5;
        return symmetricView(a).transposed.asSlice;
    }
    auto alive=retained();
    assert(alive[1,0]==2);
}

version(mir_stat_test_linearAlgebra)
@system pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;
    double[3] packed=[4,2,5];
    assert(symmetricView(packed[].stairs!"-"(2)).asSlice[1,0]==2);
    double[0] empty;
    assert(symmetricView(empty[].sliced(0,0)).asSlice.shape == [0,0]);
}

/// Mir's ordinary conversion and forwarding conventions see the logical matrix.
version(mir_stat_test_linearAlgebra)
@safe pure unittest
{
    import mir.ndslice.slice: sliced, hasAsSlice, toSlice;
    import mir.ndslice.topology: map;
    import mir.ndslice.allocation: slice, rcslice;
    double[4] data = [double.nan, 3, double.nan, double.nan];
    auto triangle = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(
        data[].sliced(2, 2));
    static assert(hasAsSlice!(typeof(triangle)) == 2);
    static assert(isStructuralView!(typeof(triangle)));
    auto logical = triangle.toSlice;
    auto forwarded = triangle.map!"a * 2";
    assert(logical == [[1, 3], [0, 1]]);
    assert(forwarded == [[2, 6], [0, 2]]);
    data[1] = 4;
    assert(logical[0,1] == 4 && forwarded[0,1] == 8);
    auto dense = logical.slice;
    auto owned = triangle.transposed.asSlice.rcslice;
    assert(owned == [[1, 0], [4, 1]]);
    dense[0,0] = 10;
    assert(logical[0,0] == 1);
    static assert(!__traits(compiles, logical[0,0] = 2));

    double[4] values = [4, 2, double.nan, 5];
    auto symmetric = symmetricView(values[].sliced(2, 2));
    static assert(hasAsSlice!(typeof(symmetric)) == 2);
    assert(symmetric.toSlice == [[4, 2], [2, 5]]);
    assert(symmetric.map!"a * 2" == [[8, 4], [4, 10]]);
}

/// A returned logical adapter preserves its RC owner through Mir forwarding.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced, toSlice;
    import mir.ndslice.topology: map;
    /// Return a forwarded view after the local owner and structural view die.
    auto retained()
    {
        auto owner = mininitRcslice!double(2, 2);
        owner[0,0] = 2; owner[0,1] = 3;
        owner[1,0] = double.nan; owner[1,1] = 4;
        const view = triangularView(owner);
        return view.transposed.map!"a * 2";
    }
    auto logical = retained();
    assert(logical[0,0] == 4 && logical[1,0] == 6 && logical[0,1] == 0);
    double[0] empty;
    assert(triangularView(empty[].sliced(0,0)).asSlice.shape == [0,0]);
    version(mir_stat_test_lifetime) static assert(!__traits(compiles, {
        @safe auto escape()
        {
            double[4] local;
            return triangularView(local[].sliced(2,2)).asSlice;
        }
    }));
}

/// Packed storage, lower triangles, and custom copies obey the same logical contract.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced, toSlice;
    import mir.ndslice.topology: stairs, map;
    import mir.ndslice.allocation: makeSlice;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[3] packed = [double.nan, 3, double.nan];
    auto triangle = triangularView!(DiagonalPolicy.unit)(packed[].stairs!"+"(2));
    auto copy = makeSlice(Mallocator.instance, triangle.asSlice);
    scope(exit) Mallocator.instance.dispose(copy.field);
    assert(copy[0,0] == 1 && copy[0,1] == 0 && copy[1,0] == 3 && copy[1,1] == 1);
    assert(triangle.toSlice[1,0] == 3);
    assert(triangle.map!"a * 2"[1,0] == 6);
}

/// Direct linear algebra calls retain structure; conversion is explicit.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.internal.linearAlgebra.products: mtimes;
    import mir.stat.internal.linearAlgebra.types: numericSlice;
    double[4] data = [double.nan, 3, double.nan, double.nan];
    double[2] values = [1,2];
    auto upper = triangularView!(TrianglePolicy.upper, DiagonalPolicy.unit)(
        data[].sliced(2,2));
    static assert(!numericSlice!(typeof(upper)));
    assert(mtimes(upper, values[].sliced) == [7,2]);
    assert(mtimes(upper.asSlice, values[].sliced) == [7,2]);
    static assert(!__traits(compiles, mtimes(upper, upper)));
}

version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.topology: stairs, map;
    /// Retain the packed RC owner through structural and generic adapters.
    auto retained()
    {
        auto owner = mininitRcslice!double(3);
        owner[0] = double.nan; owner[1] = 3; owner[2] = double.nan;
        return triangularView!(DiagonalPolicy.unit)(owner.stairs!"+"(2)).map!"a * 2";
    }
    auto logical = retained();
    assert(logical[0,0] == 2 && logical[1,0] == 6 && logical[0,1] == 0);
}
