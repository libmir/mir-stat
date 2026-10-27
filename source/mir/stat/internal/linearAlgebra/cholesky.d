/++
Real positive-definite Cholesky factorization and solves with explicit result ownership.

Only the selected triangle of the logical input view is read. Upper factors
satisfy `A = U.transpose * U`; lower factors satisfy `A = L * L.transpose`.
Successful results have zeros in the unused triangle. The selected input
entries must be finite; symmetry is assumed rather than checked. Singular and
indefinite matrices fail, without regularization or a pseudoinverse.

Float, double and real inputs, including computed and strided views, are
supported. LAPACK is optional and accelerates compatible native float/double
output layouts. Other layouts use an allocation-free scalar kernel. Backend
and summation order can affect rounding and failure near singularity.

The `choleskySolve` family reuses a factor for vector or matrix right-hand
sides, without forming an inverse. Choose GC, RC, custom, caller-output, or
in-place solution storage. Optional BLAS accelerates compatible native solves.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.cholesky;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.stat.internal.linearAlgebra.allocation;
import mir.stat.internal.linearAlgebra.products: Scalar, numericSlice, nativeSlice;
import mir.stat.internal.linearAlgebra.triangular: Triangle;
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
package(mir.stat) auto choleskyDecompose(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
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
package(mir.stat) auto rcCholeskyDecompose(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
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
package(mir.stat) auto makeCholeskyDecompose(Triangle triangle = Triangle.upper, Allocator, A)(
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
package(mir.stat) void choleskyDecomposeInto(Triangle triangle = Triangle.upper, A, C)(
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
package(mir.stat) size_t tryCholeskyDecomposeInPlace(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
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
package(mir.stat) void choleskyDecomposeInPlace(Triangle triangle = Triangle.upper, A)(scope auto ref A a)
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
    import mir.stat.internal.linearAlgebra.products: rcMtimes;
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


private enum rightHandSide(B) = numericSlice!B && isFloatingPoint!(Scalar!B);
private enum hasBlas = is(typeof({ import mir.blas; }));

/++
Solve `A * X = B` using an existing Cholesky factor, returning GC storage.
Params:
    triangle = Triangle containing the factor, interpreted in its logical view.
    factor = Square real factor from a successful Cholesky decomposition with
        the same triangle. Upper factors represent `A = U.transpose * U`;
        lower factors represent `A = L * L.transpose`.
    b = Vector or matrix with the same element type and row count as the factor.
        Each matrix column is a separate right-hand side.
Returns:
    A solution with the shape and element type of `b`. Neither input is modified.

The factor is reused without copying, inversion, or rechecking positive
definiteness. Its unused triangle is never read. Invalid factors violate the
preconditions; nonfinite right-hand sides propagate through the arithmetic.
General strides and computed inputs are supported. Empty systems and matrices
with zero right-hand sides return empty results. Shape checks use assertions.
+/
package(mir.stat) auto choleskySolve(Triangle triangle = Triangle.upper, A, B)(
    scope auto ref A factor, scope auto ref B b)
    if (matrix!A && rightHandSide!B && is(Scalar!A == Scalar!B))
{
    int unused;
    return solve!(Allocation.gc, triangle)(unused, factor, b);
}

/// Factor once and solve without forming an inverse.
version(mir_stat_test_linearAlgebra)
@safe pure unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [4, 2, double.nan, 5];
    double[2] rhs = [8, 12];
    auto factor = choleskyDecompose(data[].sliced(2, 2));
    assert(choleskySolve(factor, rhs[].sliced) == [1, 2]);
}

/++
Reference-counted variant of $(LINK2 #choleskySolve, choleskySolve).
Parameters and preconditions are the same. The owning solution releases its
storage with the last owning reference. Native inputs support
`@safe pure nothrow @nogc` calls.
+/
package(mir.stat) auto rcCholeskySolve(Triangle triangle = Triangle.upper, A, B)(
    scope auto ref A factor, scope auto ref B b)
    if (matrix!A && rightHandSide!B && is(Scalar!A == Scalar!B))
{
    int unused;
    return solve!(Allocation.rc, triangle)(unused, factor, b);
}

/// Multiple right-hand sides occupy columns.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [2, 1, double.nan, 2], rhs = [8, 4, 12, 2];
    auto result = rcCholeskySolve(data[].sliced(2, 2), rhs[].sliced(2, 2));
    static immutable expected = [[1.0, 1], [2.0, 0]];
    assert(result == expected);
}

/++
Custom-allocator variant of $(LINK2 #choleskySolve, choleskySolve).
`allocator` must support `makeUninitSlice` and `dispose`. Dispose `result.field`
with the same allocator after all solution views are finished. Exceptions from
input iterators release the allocated solution before propagating. No factor
copy or separate scratch allocation is needed.
+/
package(mir.stat) auto makeCholeskySolve(Triangle triangle = Triangle.upper, Allocator, A, B)(
    auto ref Allocator allocator, scope auto ref A factor, scope auto ref B b)
    if (matrix!A && rightHandSide!B && is(Scalar!A == Scalar!B))
{
    return solve!(Allocation.custom, triangle)(allocator, factor, b);
}

/// Choose ownership of the solution independently of the factor.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [2, 1, double.nan, 2];
    double[2] rhs = [8, 12];
    auto result = makeCholeskySolve(Mallocator.instance, data[].sliced(2, 2), rhs[].sliced);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [1, 2]);
}

private void checkSolve(A, B)(scope auto ref A factor, scope auto ref B b)
{
    assert(factor.length!0 == factor.length!1, "Cholesky factor must be square");
    assert(factor.length!0 == b.length!0, "Incompatible Cholesky solve dimensions");
}

private auto solve(Allocation policy, Triangle triangle, Allocator, A, B)(
    ref Allocator allocator, scope auto ref A factor, scope auto ref B b)
{
    checkSolve(factor, b);
    auto result = allocate!(policy, Scalar!A)(allocator, b.shape);
    scope(failure)
    {
        static if (policy == Allocation.custom)
        {
            import std.experimental.allocator: dispose;
            allocator.dispose(result.field);
        }
    }
    choleskySolveInto!triangle(factor, b, result);
    return result;
}

/++
Write the solution of $(LINK2 #choleskySolve, choleskySolve) into caller storage.
`output` must have the same shape and element type as `b`. Its elements must
not overlap each other or the factor. Output may be exactly the same logical
view as `b`; all other overlap with `b` is forbidden. Overlap is not checked.

Every output element is overwritten. No allocation is performed. Arbitrary
strides are supported. An input iterator exception may leave partial output.
+/
package(mir.stat) void choleskySolveInto(Triangle triangle = Triangle.upper, A, B, C)(
    scope auto ref A factor, scope auto ref B b, scope auto ref C output)
    if (matrix!A && rightHandSide!B && rightHandSide!C && B.N == C.N &&
        is(Scalar!A == Scalar!B) && is(Scalar!A == Scalar!C))
{
    checkSolve(factor, b);
    assert(b.shape == output.shape, "Incorrect Cholesky solution shape");
    static if (B.N == 1)
        foreach (i; 0 .. b.length) output[i] = b[i];
    else
    {
        if (output._stride!0 == 1 && output._stride!1 != 1)
            foreach (j; 0 .. b.length!1) foreach (i; 0 .. b.length!0)
                output[i, j] = b[i, j];
        else
            foreach (i; 0 .. b.length!0) foreach (j; 0 .. b.length!1)
                output[i, j] = b[i, j];
    }
    choleskySolveInPlace!triangle(factor, output);
}

/// Exact aliasing of the right-hand side and output is supported.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [2, 1, double.nan, 2];
    double[2] rhs = [8, 12];
    auto b = rhs[].sliced;
    choleskySolveInto(data[].sliced(2, 2), b, b);
    assert(rhs == [1, 2]);
}

/++
Overwrite vector or matrix `x` with the solution using a read-only Cholesky
factor. Factor requirements follow $(LINK2 #choleskySolve, choleskySolve).
The row count and element type must match the factor. Output elements must
not overlap one another or the factor. No allocation is performed.

Compatible native layouts use two BLAS triangular solves when available;
other layouts and computed factors use scalar forward/back substitution.
Native inputs support `@safe pure nothrow @nogc` calls. Computed iterators may
throw and leave partial output. Backend and summation order affect rounding.
+/
package(mir.stat) void choleskySolveInPlace(Triangle triangle = Triangle.upper, A, X)(
    scope auto ref A factor, scope auto ref X x)
    if (matrix!A && rightHandSide!X && is(Scalar!A == Scalar!X))
{
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    checkSolve(factor, x);
    if (!x.length!0) return;
    static if (X.N == 2)
        if (!x.length!1) return;
    static if (hasBlas && nativeSlice!A && nativeSlice!X &&
        (is(Scalar!A == float) || is(Scalar!A == double)))
    {
        import mir.stat.internal.linearAlgebra.products: blasMatrix, blasVector;
        static if (X.N == 1) bool nativeOutput = blasVector(x);
        else bool nativeOutput = blasMatrix(x);
        if (blasMatrix(factor) && nativeOutput)
        {
            import mir.blas: trsm, trsv, Side, Uplo;
            import cblas: Diag;
            import mir.ndslice.topology: universal;
            import mir.ndslice.dynamic: transposed;
            auto a = factor.lightScope.lightConst.universal;
            auto xt = x.lightScope;
            static if (triangle == Triangle.upper)
            {
                auto lower = a.transposed;
                auto upper = a;
            }
            else
            {
                auto lower = a;
                auto upper = a.transposed;
            }
            static if (X.N == 1)
            {
                trsv(Uplo.Lower, Diag.NonUnit, lower, xt);
                trsv(Uplo.Upper, Diag.NonUnit, upper, xt);
            }
            else
            {
                trsm(Side.Left, Uplo.Lower, Diag.NonUnit, Scalar!A(1), lower, xt);
                trsm(Side.Left, Uplo.Upper, Diag.NonUnit, Scalar!A(1), upper, xt);
            }
            return;
        }
    }
    static if (X.N == 1)
        solveVector!triangle(factor, x);
    else
        foreach (j; 0 .. x.length!1)
        {
            auto column = x[0 .. $, j];
            solveVector!triangle(factor, column);
        }
}

/// Lower factors use the same solve API with an explicit triangle.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [2, double.nan, 1, 2];
    double[2] rhs = [8, 12];
    auto b = rhs[].sliced;
    choleskySolveInPlace!(Triangle.lower)(data[].sliced(2, 2), b);
    assert(rhs == [1, 2]);
}

private void solveVector(Triangle triangle, A, X)(scope auto ref A factor, scope auto ref X x)
{
    foreach (i; 0 .. x.length)
    {
        auto value = x[i];
        foreach (k; 0 .. i)
            static if (triangle == Triangle.upper) value -= factor[k, i] * x[k];
            else value -= factor[i, k] * x[k];
        x[i] = value / factor[i, i];
    }
    foreach_reverse (i; 0 .. x.length)
    {
        auto value = x[i];
        foreach (k; i + 1 .. x.length)
            static if (triangle == Triangle.upper) value -= factor[i, k] * x[k];
            else value -= factor[k, i] * x[k];
        x[i] = value / factor[i, i];
    }
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed, strided, reversed;
    import mir.ndslice.allocation: mininitRcslice;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (triangle; [Triangle.upper, Triangle.lower])
    {{
        foreach (transposedFactor; [false, true])
        foreach (transposedOutput; [false, true])
        {
            T[15] factorData = T(-99);
            auto factor = factorData[].sliced(3,5)[0..3,0..3].universal;
            if (transposedFactor) factor = factor.transposed;
            static immutable upper = [[2,1,1],[0,3,1],[0,0,2]];
            foreach (i; 0 .. 3) foreach (j; 0 .. 3)
                factor[i,j] = (triangle == Triangle.upper ? j >= i : j <= i)
                    ? T(triangle == Triangle.upper ? upper[i][j] : upper[j][i]) : T.nan;
            // Independent dense matrix and known solutions determine the RHS.
            static immutable dense = [[4,2,2],[2,10,4],[2,4,6]];
            T[6] answers = [T(0.3), T(-2), T(0.5), T(3), T(-1), T(0.7)];
            T[6] rhsData;
            auto rhs = rhsData[].sliced(3,2);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
            {
                T sum = 0;
                foreach (k; 0 .. 3) sum += dense[i][k] * answers[k*2+j];
                rhs[i,j] = sum;
            }
            T[15] storage = T(-99);
            auto output = storage[].sliced(3,5)
                [0..(transposedOutput ? 2 : 3),0..(transposedOutput ? 3 : 2)].universal;
            if (transposedOutput) output = output.transposed;
            choleskySolveInto!triangle(factor.lightConst, rhs, output);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
                assert(fabs(output[i,j] - answers[i*2+j]) < 64 * T.epsilon);
            // Check A*X-B separately, using the original RHS.
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
            {
                T residual = -rhs[i,j];
                foreach (k; 0 .. 3) residual += dense[i][k] * output[k,j];
                assert(fabs(residual) < 256 * T.epsilon);
            }
            foreach (i; 0 .. 3) foreach (j; 0 .. 5)
                if (i >= (transposedOutput ? 2 : 3) || j >= (transposedOutput ? 3 : 2))
                    assert(storage[i*5+j] == T(-99));
            auto rc = rcCholeskySolve!triangle(factor.lightConst, rhs);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
                assert(fabs(rc[i,j] - output[i,j]) < 64 * T.epsilon);
            // General output strides and reversed vectors select the scalar path.
            T[24] generalStorage = T(-99);
            auto general = generalStorage[].sliced(6,4).strided!(0,1)(2,2);
            choleskySolveInto!triangle(factor, rhs, general);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
                assert(fabs(general[i,j] - answers[i*2+j]) < 64 * T.epsilon);
            auto vector = rhs[0..$,0];
            T[6] vectorStorage;
            auto reversedOutput = vectorStorage[].sliced.universal.strided(2).reversed;
            choleskySolveInto!triangle(factor, vector, reversedOutput);
            foreach (i; 0 .. 3)
                assert(fabs(reversedOutput[i] - answers[i*2]) < 64 * T.epsilon);
            auto vectorResult = rcCholeskySolve!triangle(factor, vector);
            foreach (i; 0 .. 3)
                assert(fabs(vectorResult[i] - answers[i*2]) < 64 * T.epsilon);
            // Computed factors must never evaluate the unselected NaNs.
            auto computed = factor.map!((T value) { assert(value == value); return value; });
            auto mapped = rcCholeskySolve!triangle(computed, rhs);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
                assert(fabs(mapped[i,j] - answers[i*2+j]) < 64 * T.epsilon);
            auto owner = mininitRcslice!T(3,3);
            owner[] = factor;
            const constOwner = owner;
            auto ownedResult = rcCholeskySolve!triangle(constOwner, rhs);
            foreach (i; 0 .. 3) foreach (j; 0 .. 2)
                assert(fabs(ownedResult[i,j] - answers[i*2+j]) < 64 * T.epsilon);
            auto noColumns = mininitRcslice!T(3,0);
            assert(rcCholeskySolve!triangle(factor, noColumns).shape == [3,0]);
            foreach (i; 0 .. 3) foreach (j; 0 .. 3)
                if (triangle == Triangle.upper ? j >= i : j <= i)
                    assert(factor[i,j] == T(triangle == Triangle.upper ? upper[i][j] : upper[j][i]));
        }
        auto empty = mininitRcslice!T(0,0);
        auto noRows = mininitRcslice!T(0,2);
        auto emptyVector = mininitRcslice!T(0);
        assert(rcCholeskySolve!triangle(empty, noRows).shape == [0,2]);
        assert(rcCholeskySolve!triangle(empty, emptyVector).length == 0);
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
    double[4] factorData = [2,1,double.nan,2];
    double[2] rhsData = [8,12];
    auto factor = factorData[].sliced(2,2);
    auto rhs = rhsData[].sliced;
    auto custom = makeCholeskySolve(allocator, factor, rhs);
    assert(custom == [1,2]);
    allocator.dispose(custom.field);
    auto failure = new Exception("input failed");
    double fail(double value) { throw failure; }
    assertThrown!Exception(makeCholeskySolve(allocator, factor.map!fail, rhs));
    assertThrown!Exception(makeCholeskySolve(allocator, factor, rhs.map!fail));
    assert(allocator.allocations == 3 && allocator.releases == 3);
    auto wrongFactor = factorData[].sliced(1,4);
    assertThrown!AssertError(makeCholeskySolve(allocator, wrongFactor, rhs));
    auto wrongRhs = rhs[0..1];
    assertThrown!AssertError(makeCholeskySolve(allocator, factor, wrongRhs));
    assert(allocator.allocations == 3);
    assertThrown!AssertError(choleskySolveInto(factor, rhs, wrongRhs));
}

import mir.stat.internal.linearAlgebra.views: SymmetricView;

/++
Symmetric-view overload. Uses the view's selected triangle and produces the
factor in that triangle. Positive definiteness is checked by factorization,
not implied by the view. Other requirements follow the plain-slice overload.
+/
package(mir.stat) auto choleskyDecompose(Triangle triangle, A)(scope auto ref const(SymmetricView!(triangle, A)) a)
    if (matrix!A)
{
    return choleskyDecompose!triangle(a.storage);
}

/++
Symmetric-view overload. Uses the view's selected triangle and produces the
factor in that triangle. Positive definiteness is checked by factorization,
not implied by the view. Other requirements follow the plain-slice overload.
+/
package(mir.stat) auto rcCholeskyDecompose(Triangle triangle, A)(scope auto ref const(SymmetricView!(triangle, A)) a)
    if (matrix!A)
{
    return rcCholeskyDecompose!triangle(a.storage);
}

/++
Symmetric-view overload. Uses the view's selected triangle and produces the
factor in that triangle. Positive definiteness is checked by factorization,
not implied by the view. Other requirements follow the plain-slice overload.
+/
package(mir.stat) auto makeCholeskyDecompose(Triangle triangle, A, Allocator)(
    auto ref Allocator allocator,
    scope auto ref const(SymmetricView!(triangle, A)) a)
    if (matrix!A)
{
    return makeCholeskyDecompose!triangle(allocator, a.storage);
}

/++
Symmetric-view overload. Uses the view's selected triangle and produces the
factor in that triangle. Positive definiteness is checked by factorization,
not implied by the view. Other requirements follow the plain-slice overload.
+/
package(mir.stat) void choleskyDecomposeInto(Triangle triangle, A, C)(
    scope auto ref const(SymmetricView!(triangle, A)) a,
    scope auto ref C output)
    if (matrix!A)
{
    choleskyDecomposeInto!triangle(a.storage, output);
}
