/++
Internal zero-copy square indexing for packed stairs storage.
The adapter retains the original iterator, including any RC owner. Its unused
triangle reads as zero; structural views supply symmetry or unit diagonals.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.packed;

import mir.ndslice.slice: Slice, sliced;
import mir.ndslice.iterator: StairsIterator, FieldIterator;
import mir.stat.internal.linearAlgebra.types: TrianglePolicy;
import std.traits: Unqual;
import mir.stat.internal.linearAlgebra.types: Scalar, DiagonalPolicy;
import mir.stat.internal.linearAlgebra.backend: hasBlas, blasNumber, nativeSlice, blasVector, blasMatrix;
import mir.ndslice.slice: isSlice;
import mir.qualifier: lightScope;
import mir.math.sum: Summation, Summator;

/// Recognize complete, contiguous sequences of packed stairs rows.
package template PackedInfo(A)
{
    static if (is(Unqual!A : Slice!(StairsIterator!(I, direction)), I, string direction))
    {
        enum valid = true;
        enum triangle = direction == "+" ? TrianglePolicy.lower : TrianglePolicy.upper;
        alias Element = Unqual!(typeof(I.init[0]));
    }
    else enum valid = false;
}

/// Whether an argument is packed stairs storage rather than a dense numeric slice.
package enum isPacked(A) = PackedInfo!A.valid;

/// Numeric square view retaining packed storage without materializing elements.
package alias PackedMatrix(S) = Slice!(FieldIterator!(PackedField!S), 2);

/// Construct square indexing, rejecting partial stairs that do not start at a complete triangle.
package PackedMatrix!S packedMatrix(S)(S rows) if (isPacked!S)
{
    const n = rows.length;
    assert(!n || rows._iterator._length == (PackedInfo!S.triangle == TrianglePolicy.lower ? 1 : n),
        "Packed matrix requires a complete stairs triangle");
    assert(!n || n <= cast(size_t)ptrdiff_t.max / n, "Packed matrix logical shape exceeds iterator indexing range");
    return FieldIterator!(PackedField!S)(0, PackedField!S(rows)).sliced(n, n);
}

/// Map a logical square offset to its stored stairs row, retaining input ownership.
package struct PackedField(S) if (isPacked!S)
{
    S rows;
    /// Preserve the owner while exposing read-only element access.
    auto lightConst()() const @property
    {
        return PackedField!(typeof(rows.lightConst))(rows.lightConst);
    }
    /// Read an entry known to belong to the stored triangle without square-index division.
    PackedInfo!S.Element storedValue(size_t i, size_t j)
    {
        const n = rows.length;
        assert(i < n && j < n);
        static if (PackedInfo!S.triangle == TrianglePolicy.upper)
        {
            assert(j >= i);
            return rows._iterator._iterator[i*n - i*(i+1)/2 + j];
        }
        else
        {
            assert(j <= i);
            return rows._iterator._iterator[i*(i+1)/2 + j];
        }
    }
    /// Read a stored element or return zero outside the selected triangle.
    PackedInfo!S.Element opIndex(ptrdiff_t index)
    {
        const n = rows.length;
        assert(index >= 0 && n && cast(size_t) index / n < n);
        const i = cast(size_t) index / n;
        const j = cast(size_t) index % n;
        static if (PackedInfo!S.triangle == TrianglePolicy.upper)
            return j < i ? PackedInfo!S.Element(0) : rows._iterator._iterator[i*n - i*(i+1)/2 + j];
        else
            return j > i ? PackedInfo!S.Element(0) : rows._iterator._iterator[i*(i+1)/2 + j];
    }
}


/// Recognize our zero-copy adapter, including transposed universal slices.
package template PackedAdapter(A)
{
    static if (isSlice!A && is(A.Iterator : FieldIterator!F, F) && is(F : PackedField!S, S))
    {
        enum valid = true;
        alias Storage = S;
    }
    else enum valid = false;
}

/// Whether packed storage has a BLAS-compatible native element iterator.
private template nativePacked(A)
{
    static if (PackedAdapter!A.valid)
        enum nativePacked = is(typeof(lightScope(A.init._iterator._field.rows._iterator._iterator)) : const(Scalar!A)*);
    else enum nativePacked = false;
}

/// Accept only complete square adapters and their ordinary transposes, never submatrices.
private bool completePacked(TrianglePolicy triangle, A)(scope auto ref A a)
{
    const n = a._iterator._field.rows.length;
    const transposed = n > 1 && a._stride!0 == 1;
    enum stored = PackedInfo!(PackedAdapter!A.Storage).triangle;
    return a._iterator._index == 0 && a.length!0 == n && a.length!1 == n &&
        ((a._stride!0 == n && a._stride!1 == 1) || (a._stride!0 == 1 && a._stride!1 == n)) &&
        triangle == (transposed ? (stored == TrianglePolicy.upper ? TrianglePolicy.lower : TrianglePolicy.upper) : stored);
}

/// Try packed triangular BLAS after shape checks; each output column is a separate TPMV.
package bool tryPackedTriangular(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, C)(
    scope auto ref A a, scope auto ref B b, scope auto ref C c)
{
    static if (hasBlas && nativePacked!A && blasNumber!(Scalar!A) && nativeSlice!C)
    {
        import cblas: blasint, Order, Uplo, Transpose, Diag, tpmv;
        if (!completePacked!triangle(a) || a.length!0 > blasint.max) return false;
        static if (C.N == 1) { if (!blasVector(c)) return false; }
        else { if (!blasMatrix(c)) return false; }
        alias T = Scalar!A;
        enum stored = PackedInfo!(PackedAdapter!A.Storage).triangle;
        enum uplo = stored == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
        enum diag = diagonal == DiagonalPolicy.unit ? Diag.Unit : Diag.NonUnit;
        const trans = a.length!0 > 1 && a._stride!0 == 1 ? Transpose.Trans : Transpose.NoTrans;
        const ap = lightScope(a._iterator._field.rows._iterator._iterator);
        // Bounds, positive BLAS-sized strides, full packed extent, and nonoverlap
        // follow checked descriptors and the caller's documented input contract.
        static if (C.N == 1)
        {
            foreach (i; 0 .. b.length) c[i] = b[i];
            () @trusted { tpmv(Order.RowMajor, uplo, trans, diag, cast(blasint)c.length,
                ap, c.lightScope._iterator, cast(blasint)c._stride!0); }();
        }
        else
        {
            foreach (j; 0 .. b.length!1)
            {
                foreach (i; 0 .. b.length!0) c[i,j] = b[i,j];
                () @trusted { tpmv(Order.RowMajor, uplo, trans, diag, cast(blasint)c.length!0,
                    ap, c.lightScope._iterator + j * c._stride!1, cast(blasint)c._stride!0); }();
            }
        }
        return true;
    }
    else return false;
}

/// Try packed symmetric BLAS into separate workspace; transpose preserves symmetry.
package bool tryPackedSymmetric(TrianglePolicy triangle, A, X, W)(
    scope auto ref A a, scope auto ref X x, scope auto ref W work)
{
    static if (hasBlas && nativePacked!A && blasNumber!(Scalar!A) && nativeSlice!X && nativeSlice!W)
    {
        import cblas: blasint, Order, Uplo, spmv;
        if (!x.length || x.length > blasint.max || !completePacked!triangle(a) ||
            !blasVector(x) || !blasVector(work)) return false;
        alias T = Scalar!A;
        enum stored = PackedInfo!(PackedAdapter!A.Storage).triangle;
        enum uplo = stored == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower;
        const ap = lightScope(a._iterator._field.rows._iterator._iterator);
        // Complete packed extent and vector bounds are established above;
        // disjoint workspace is a precondition of the public operation.
        () @trusted { spmv(Order.RowMajor, uplo, cast(blasint)x.length, T(1), ap,
            x.lightScope._iterator, cast(blasint)x._stride!0, T(0),
            work.lightScope._iterator, cast(blasint)work._stride!0); }();
        return true;
    }
    else return false;
}

/// Dispatch complete adapters to native packed indexing, avoiding square-offset division in inner loops.
package bool tryPackedTriangularNative(TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, C)(
    scope auto ref A a, scope auto ref B b, scope auto ref C c)
{
    static if (PackedAdapter!A.valid)
    {
        if (!completePacked!triangle(a)) return false;
        if (a.length!0 > 1 && a._stride!0 == 1)
            packedTriangularNative!(true, triangle, diagonal)(a,b,c);
        else packedTriangularNative!(false, triangle, diagonal)(a,b,c);
        return true;
    }
    else return false;
}

/// Multiply stored triangular rows with a compile-time transpose and diagonal policy.
private void packedTriangularNative(bool transpose, TrianglePolicy triangle, DiagonalPolicy diagonal, A, B, C)(
    scope auto ref A a, scope auto ref B b, scope auto ref C c)
{
    alias T = Scalar!A;
    foreach (i; 0 .. a.length!0)
    {
        const begin = triangle == TrianglePolicy.upper ? i : 0;
        const end = triangle == TrianglePolicy.upper ? a.length!0 : i+1;
        static if (C.N == 2) const columns = c.length!1;
        else enum columns = 1;
        foreach (j; 0 .. columns)
        {
            T sum = T(0);
            foreach (k; begin .. end)
            {
                static if (B.N == 2) auto rhs = b[k,j];
                else auto rhs = b[k];
                static if (diagonal == DiagonalPolicy.unit)
                    if (i == k) { sum += rhs; continue; }
                static if (transpose) auto value = a._iterator._field.storedValue(k,i);
                else auto value = a._iterator._field.storedValue(i,k);
                sum += value * rhs;
            }
            static if (C.N == 2) c[i,j] = sum;
            else c[i] = sum;
        }
    }
}


/// Evaluate complete symmetric packed adapters using direct stored coordinates.
package bool tryPackedQuadratic(bool workspace, TrianglePolicy triangle, Summation summation, A, X, W, R)(
    scope auto ref A a, scope auto ref X x, scope auto ref W work, ref R result)
{
    static if (PackedAdapter!A.valid)
    {
        if (!completePacked!triangle(a)) return false;
        // A symmetric matrix equals its transpose. Use the original stored
        // triangle even when the descriptor has been transposed.
        enum stored = PackedInfo!(PackedAdapter!A.Storage).triangle;
        Summator!(Scalar!X, summation) total;
        foreach (i; 0 .. x.length)
        {
            Summator!(Scalar!X, summation) row;
            foreach (j; 0 .. i)
            {
                static if (stored == TrianglePolicy.upper) auto value = a._iterator._field.storedValue(j,i);
                else auto value = a._iterator._field.storedValue(i,j);
                row.put(value * x[j]);
            }
            foreach (j; i .. x.length)
            {
                static if (stored == TrianglePolicy.upper) auto value = a._iterator._field.storedValue(i,j);
                else auto value = a._iterator._field.storedValue(j,i);
                row.put(value * x[j]);
            }
            auto value = row.sum;
            static if (workspace) work[i] = value;
            total.put(x[i] * value);
        }
        result = total.sum;
        return true;
    }
    else return false;
}
