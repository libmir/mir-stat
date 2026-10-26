/++
Real positive-definite Cholesky factorization with explicit result ownership.

Only the selected triangle of the logical input view is read. Upper factors
satisfy `A = U.transpose * U`; lower factors satisfy `A = L * L.transpose`.
Successful results have zeros in the unused triangle. The selected input
entries must be finite; symmetry is assumed rather than checked. Singular and
indefinite matrices fail, without regularization or a pseudoinverse.

Float, double and real inputs, including computed and strided views, are
supported. LAPACK is optional and accelerates compatible native float/double
output layouts. Other layouts use an allocation-free scalar kernel. Backend
and summation order can affect rounding and failure near singularity.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.math.internal.linearAlgebra.cholesky;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.math.internal.linearAlgebra.allocation;
import mir.math.internal.linearAlgebra.products: Scalar, numericSlice, nativeSlice;
import mir.math.internal.linearAlgebra.triangular: Triangle;
import std.traits: isFloatingPoint;

private enum matrix(A) = numericSlice!A && A.N == 2 && isFloatingPoint!(Scalar!A);
private enum hasLapack = is(typeof({ import mir.lapack; }));

/++
Factor a real symmetric positive-definite matrix into GC-managed storage.
Params:
    triangle = Logical input triangle to read and output factor to produce.
    a = Square real matrix; its selected entries must be finite.
Returns:
    A factor of the same shape and element type as `a`. The unused triangle
    is zero. A zero-by-zero input produces an empty factor.
Throws:
    Exception if factorization fails. Shape and finite-input preconditions
    are checked with assertions. Input is never modified.
+/
auto choleskyDecompose(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
    if (matrix!A)
{
    int unused;
    return factor!(Allocation.gc, triangle)(unused, a);
}

/// The unselected input triangle may contain NaNs.
version(mir_stat_test_linearAlgebra)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 2, double.nan, 5];
    assert(choleskyDecompose(data[].sliced(2, 2)) == [[2, 1], [0, 2]]);
}

/++
Reference-counted variant of $(LINK2 #choleskyDecompose, choleskyDecompose).
The parameters and failure behavior are the same. Storage is released with
its last owning reference. Native numeric inputs support `@safe pure @nogc`.
+/
auto rcCholeskyDecompose(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
    if (matrix!A)
{
    int unused;
    return factor!(Allocation.rc, triangle)(unused, a);
}

/// A lower factor reads only the lower triangle.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, double.nan, 2, 5];
    static immutable expected = [[2.0, 0], [1.0, 2]];
    assert(rcCholeskyDecompose!(Triangle.lower)(data[].sliced(2, 2)) == expected);
}

/++
Custom-allocator variant of $(LINK2 #choleskyDecompose, choleskyDecompose).
`allocator` must support `makeUninitSlice` and `dispose`. Dispose `result.field`
with the same allocator after all views are finished. Failure releases result
storage before propagating the exception. No separate scratch buffer is used.
+/
auto makeCholeskyDecompose(Triangle triangle = Triangle.upper, Allocator, A)(
    auto ref Allocator allocator, scope auto ref A a)
    if (matrix!A)
{
    return factor!(Allocation.custom, triangle)(allocator, a);
}

/// Dispose the custom factor with its allocator.
version(mir_stat_test_linearAlgebra)
@system @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [4, 2, double.nan, 5];
    auto result = makeCholeskyDecompose(Mallocator.instance, data[].sliced(2, 2));
    scope(exit) Mallocator.instance.dispose(result.field);
    static immutable expected = [[2.0, 1], [0.0, 2]];
    assert(result == expected);
}

private auto factor(Allocation policy, Triangle triangle, Allocator, A)(
    ref Allocator allocator, scope auto ref A a)
{
    assert(a.length!0 == a.length!1, "Cholesky requires a square matrix");
    auto result = allocate!(policy, Scalar!A)(allocator, a.shape);
    scope(failure)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(result.field);
        }
    }
    choleskyDecomposeInto!triangle(a, result);
    return result;
}

/++
Copy the selected triangle of `a` to caller-owned `output` and factor it.
Both matrices must be square and have the same shape and real element type.
Input and output must not overlap; output elements must not overlap each other.
Arbitrary strides are supported. No allocation is performed on success.
On numerical failure, throws an Exception and leaves a partial factorization
in output. Input iterator exceptions propagate. Overlap is not checked.
+/
void choleskyDecomposeInto(Triangle triangle = Triangle.upper, A, C)(
    scope auto ref A a, scope auto ref C output)
    if (matrix!A && matrix!C && is(Scalar!A == Scalar!C))
{
    assert(a.length!0 == a.length!1, "Cholesky requires a square matrix");
    assert(a.shape == output.shape, "Incorrect Cholesky output shape");
    foreach (i; 0 .. a.length!0)
        foreach (j; 0 .. a.length!1)
            if (triangle == Triangle.upper ? j >= i : j <= i)
                output[i, j] = a[i, j];
    choleskyDecomposeInPlace!triangle(output);
}

/// Preserve the input and overwrite a separate output buffer.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 2, double.nan, 5], buffer;
    auto output = buffer[].sliced(2, 2);
    choleskyDecomposeInto(data[].sliced(2, 2), output);
    static immutable expected = [[2.0, 1], [0.0, 2]];
    assert(output == expected && data[0] == 4);
}

/++
Factor writable storage in place without allocating. Returns zero on success,
or the one-based failed pivot index on numerical failure (including a nonfinite
computed factor). The matrix must be square, its selected entries finite, and
its elements nonoverlapping. General strides are supported. Preconditions use
assertions. Native inputs support `@safe pure nothrow @nogc`.

Success zeros the unused triangle. Failure leaves a partial factorization;
its precise contents depend on the backend and must not be used as a factor.
Only the selected triangle is read; no symmetry check is performed.
+/
size_t tryCholeskyDecomposeInPlace(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
    if (matrix!A)
{
    import std.math: isFinite, sqrt;
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    assert(a.length!0 == a.length!1, "Cholesky requires a square matrix");
    const n = a.length!0;
    if (!n) return 0;
    foreach (i; 0 .. n)
        foreach (j; 0 .. n)
            if (triangle == Triangle.upper ? j >= i : j <= i)
                assert(isFinite(a[i, j]), "Cholesky input must be finite");
    bool nativeDone;
    static if (hasLapack && nativeSlice!A &&
        (is(Scalar!A == float) || is(Scalar!A == double)))
    {
        import mir.lapack: potrf, lapackint;
        import mir.ndslice.topology: universal, assumeCanonical;
        import mir.ndslice.dynamic: transposed;
        if (n <= lapackint.max && a._stride!0 > 0 && a._stride!1 > 0 &&
            a._stride!0 <= lapackint.max && a._stride!1 <= lapackint.max &&
            ((a._stride!1 == 1 && a._stride!0 >= n) ||
             (a._stride!0 == 1 && a._stride!1 >= n)))
        {
            auto storage = a.lightScope.universal;
            bool rowMajor = a._stride!1 == 1;
            if (!rowMajor) storage = storage.transposed;
            // mir.lapack exposes Fortran column-major storage, so a row-major
            // logical view requires the opposite triangle.
            char uplo = (triangle == Triangle.upper) != rowMajor ? 'U' : 'L';
            auto info = potrf(uplo, storage.assumeCanonical);
            if (info) return info;
            nativeDone = true;
        }
    }
    if (!nativeDone)
    {
        foreach (j; 0 .. n)
        {
            auto pivot = a[j, j];
            foreach (k; 0 .. j)
            {
                auto value = triangle == Triangle.upper ? a[k, j] : a[j, k];
                pivot -= value * value;
            }
            if (!(pivot > 0) || !isFinite(pivot)) return j + 1;
            a[j, j] = sqrt(pivot);
            foreach (i; j + 1 .. n)
            {
                auto value = triangle == Triangle.upper ? a[j, i] : a[i, j];
                foreach (k; 0 .. j)
                    static if (triangle == Triangle.upper)
                        value -= a[k, j] * a[k, i];
                    else
                        value -= a[j, k] * a[i, k];
                static if (triangle == Triangle.upper) a[j, i] = value / a[j, j];
                else a[i, j] = value / a[j, j];
            }
        }
    }
    // Never report success for a nonfinite computed factor, even if the native
    // backend returned success after intermediate overflow.
    foreach (i; 0 .. n)
        foreach (j; 0 .. n)
            if (triangle == Triangle.upper ? j >= i : j <= i)
            {
                if (!isFinite(a[i, j])) return (i > j ? i : j) + 1;
            }
    foreach (i; 0 .. n)
        foreach (j; 0 .. n)
            if (triangle == Triangle.upper ? j < i : j > i)
                a[i, j] = 0;
    return 0;
}

/// A singular matrix reports the first failed leading minor.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [1, 1, double.nan, 1];
    auto a = data[].sliced(2, 2);
    assert(tryCholeskyDecomposeInPlace(a) == 2);
}

/++
Throwing variant of $(LINK2 #tryCholeskyDecomposeInPlace, tryCholeskyDecomposeInPlace).
The same preconditions and partial-output rules apply. No allocation is
performed on success; an Exception reports numerical failure.
+/
void choleskyDecomposeInPlace(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
    if (matrix!A)
{
    import mir.exception: enforce;
    enforce!"Cholesky factorization failed"(tryCholeskyDecomposeInPlace!triangle(a) == 0);
}

/// Reuse writable storage for the factor.
version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 2, double.nan, 5];
    auto a = data[].sliced(2, 2);
    choleskyDecomposeInPlace(a);
    static immutable expected = [[2.0, 1], [0.0, 2]];
    assert(a == expected);
}

version(mir_stat_test_linearAlgebra)
@safe pure @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.math.internal.linearAlgebra.products: rcMtimes;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (triangle; [Triangle.upper, Triangle.lower])
    {{
        T[9] data = [4, 2, 2, 2, 10, 4, 2, 4, 6];
        auto a = data[].sliced(3, 3);
        foreach (i; 0 .. 3) foreach (j; 0 .. 3)
            if (triangle == Triangle.upper ? j < i : j > i) a[i,j] = T.nan;
        auto factor = rcCholeskyDecompose!triangle(a);
        auto ft = factor.universal.transposed;
        static if (triangle == Triangle.upper) auto reconstructed = rcMtimes(ft, factor);
        else auto reconstructed = rcMtimes(factor, ft);
        static immutable expected = [[4,2,2],[2,10,4],[2,4,6]];
        foreach (i; 0 .. 3) foreach (j; 0 .. 3)
            assert(fabs(reconstructed[i,j] - expected[i][j]) <= 32 * T.epsilon);
        // General-stride output forces the scalar kernel; transposed/padded
        // output exercises the optional native storage conversion separately.
        T[36] buffer;
        auto general = buffer[].sliced(6, 6).strided!(0,1)(2,2);
        choleskyDecomposeInto!triangle(a, general);
        assert(general == factor);
        auto backward = general.reversed!(0,1);
        choleskyDecomposeInto!triangle(a, backward);
        assert(backward == factor);
        T[15] padded;
        auto output = padded[].sliced(3,5)[0..3,0..3].universal.transposed;
        choleskyDecomposeInto!triangle(a, output);
        assert(output == factor);
        auto computed = a.map!((T value) { assert(value == value); return value; });
        assert(rcCholeskyDecompose!triangle(computed) == factor);
        auto inputOwner = mininitRcslice!T(3, 3);
        inputOwner[] = a;
        const ownedInput = inputOwner;
        assert(rcCholeskyDecompose!triangle(ownedInput) == factor);
        auto empty = mininitRcslice!T(0, 0);
        assert(tryCholeskyDecomposeInPlace!triangle(empty) == 0);
        assert(rcCholeskyDecompose!triangle(empty).shape == [0,0]);
        T[1] singleton = [9];
        auto one = singleton[].sliced(1,1);
        assert(tryCholeskyDecomposeInPlace!triangle(one) == 0 && singleton[0] == 3);
        T[4] indefinite = [1,2,2,1];
        auto bad = indefinite[].sliced(2,2);
        assert(tryCholeskyDecomposeInPlace!triangle(bad) == 2);
        T[1] zero = [0];
        auto z = zero[].sliced(1,1);
        assert(tryCholeskyDecomposeInPlace!triangle(z) == 1);
        zero[0] = -1;
        assert(tryCholeskyDecomposeInPlace!triangle(z) == 1);
        T[4] overflow = [T.min_normal, T.max / 2, T.max / 2, 1];
        auto overflowInput = overflow[].sliced(2, 2);
        assert(tryCholeskyDecomposeInPlace!triangle(overflowInput) != 0);
        T[4] fractional = [T(2), T(0.3), T(0.3), T(1.7)];
        auto fractionalInput = fractional[].sliced(2, 2);
        auto f = rcCholeskyDecompose!triangle(fractionalInput);
        static if (triangle == Triangle.upper)
            auto reconstruction = rcMtimes(f.universal.transposed, f);
        else
            auto reconstruction = rcMtimes(f, f.universal.transposed);
        foreach (i; 0 .. 2) foreach (j; 0 .. 2)
            assert(fabs(reconstruction[i, j] - fractionalInput[i, j]) < 64 * T.epsilon);
        // Small and large positive scales remain representable.
        foreach (scale; [T.min_normal, T.max / 4])
        {
            singleton[0] = scale;
            assert(tryCholeskyDecomposeInPlace!triangle(one) == 0);
            assert(fabs(singleton[0] * singleton[0] / scale - 1) < 16 * T.epsilon);
        }
    }}
}

version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    struct CountingAllocator
    {
        size_t allocations, releases;
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[4] data = [4,2,double.nan,5];
    auto a = data[].sliced(2,2);
    auto result = makeCholeskyDecompose(allocator, a);
    assert(result == [[2,1],[0,2]]);
    allocator.dispose(result.field);
    data[] = [1,2,double.nan,1];
    assertThrown!Exception(makeCholeskyDecompose(allocator, a));
    assert(allocator.allocations == 2 && allocator.releases == 2);
    auto failure = new Exception("input failed");
    double fail(double value) { throw failure; }
    assertThrown!Exception(makeCholeskyDecompose(allocator, a.map!fail));
    assert(allocator.allocations == 3 && allocator.releases == 3);
    auto nonsquare = data[].sliced(1,4);
    assertThrown!AssertError(makeCholeskyDecompose(allocator, nonsquare));
    assert(allocator.allocations == 3);
    double[1] invalid = [double.infinity];
    auto nonfinite = invalid[].sliced(1,1);
    assertThrown!AssertError(tryCholeskyDecomposeInPlace(nonfinite));
    invalid[0] = double.nan;
    assertThrown!AssertError(tryCholeskyDecomposeInPlace(nonfinite));
    double[4] outData;
    auto output = outData[].sliced(1,4);
    assertThrown!AssertError(choleskyDecomposeInto(a, output));
}
