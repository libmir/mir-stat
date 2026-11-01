/++
Matrix cross-products with explicit orientation and result ownership.

`crossprod(a)` computes `a.transpose * a`; `tcrossprod(a)` computes
`a * a.transpose`. Two-input overloads compute `a.transpose * b` and
`a * b.transpose`, respectively. All inputs and results are rank-two matrices.
Represent vectors explicitly as `(n, 1)` or `(1, n)` matrices; physical storage
order never determines mathematical orientation. Transposes do not conjugate
complex elements, matching `mtimes`.

Allocated self-products return a symmetric view and initialize only its selected
triangle, using SYRK when BLAS types/layouts permit. Other types/layouts
use a native D triangle kernel. Two-input products use ordinary multiplication;
they do not assume symmetry even when the two arguments refer to the same data.
No input is materialized. Computed iterators must give stable values during a
call. Backend and reduction order can change floating-point rounding.

Diagonal views use coordinate products rather than general matrix kernels.
Self-products retain the symmetric-view result and selected-triangle writes;
two-input products return ordinary dense matrices. Complex entries are squared
without conjugation. All result ownership and caller-output forms are supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.crossproducts;

version(mir_stat_test) version = mir_stat_test_linearAlgebra;
version(mir_stat_test_blas) version = mir_stat_test_linearAlgebra;

import mir.stat.internal.linearAlgebra.views: diagonalView, DiagonalView, DiagonalInfo, logicalOperand;
import mir.stat.internal.linearAlgebra.allocation;
import mir.stat.internal.linearAlgebra.types: Scalar, numericSlice, TrianglePolicy;
import mir.stat.internal.linearAlgebra.views: SymmetricView, symmetricView, disposeSymmetric;
import mir.stat.internal.linearAlgebra.backend: hasBlas, blasNumber, nativeSlice, blasMatrix;
import mir.stat.internal.linearAlgebra.products: mtimes, rcMtimes, makeMtimes, mtimesInto;
import mir.ndslice.topology: universal;
import mir.ndslice.dynamic: transposed;

/// Whether an operand is a floating-point or complex rank-two slice.
private enum matrix(A) = numericSlice!A && A.N == 2;
/// Matrix operands must share an unqualified element type.
private enum compatible(A, B) = matrix!A && matrix!B && is(Scalar!A == Scalar!B);

/++
Compute `a.transpose * a` into GC-managed storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!1` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.gc, Scalar!A, 2))
crossprod(TrianglePolicy triangle = TrianglePolicy.upper, A)(scope auto ref A a) if (matrix!A)
{
    int allocator;
    return selfAllocated!(Allocation.gc, true, triangle)(allocator, a);
}

/++
Compute `a.transpose * b` into GC-managed storage.
Both inputs must be rank-two matrices with matching element types and equal
row counts. Returns shape `(a.length!1, b.length!1)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `mtimes`; incompatible
dimensions violate assertion-checked preconditions.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, 2)
crossprod(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return mtimes(a.lightConst.universal.transposed, b);
}

/++
Compute `a.transpose * a` into reference-counted storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!1` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.rc, Scalar!A, 2))
rcCrossprod(TrianglePolicy triangle = TrianglePolicy.upper, A)(scope auto ref A a) if (matrix!A)
{
    int allocator;
    return selfAllocated!(Allocation.rc, true, triangle)(allocator, a);
}

/++
Compute `a.transpose * b` into reference-counted storage.
Both inputs must be rank-two matrices with matching element types and equal
row counts. Returns shape `(a.length!1, b.length!1)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `rcMtimes`; incompatible
dimensions violate assertion-checked preconditions.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, 2)
rcCrossprod(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return rcMtimes(a.lightConst.universal.transposed, b);
}

/++
Compute `a.transpose * a` into custom-allocated storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!1` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
Use `disposeSymmetric(allocator, result)` after all derived views are finished.
Custom result storage is released if an input iterator throws.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.custom, Scalar!A, 2))
makeCrossprod(TrianglePolicy triangle = TrianglePolicy.upper, Allocator, A)(auto ref Allocator allocator, scope auto ref A a) if (matrix!A)
{
    return selfAllocated!(Allocation.custom, true, triangle)(allocator, a);
}

/++
Compute `a.transpose * b` into custom-allocated storage.
Both inputs must be rank-two matrices with matching element types and equal
row counts. Returns shape `(a.length!1, b.length!1)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `makeMtimes`; incompatible
dimensions violate assertion-checked preconditions.
Dispose `result.field` with the same allocator after all views are finished.
Custom result storage is released if an input iterator throws.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, 2)
makeCrossprod(Allocator, A, B)(auto ref Allocator allocator, scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return makeMtimes(allocator, a.lightConst.universal.transposed, b);
}

/++
Overwrite caller storage with `a.transpose * a` without allocating.
Output must be a square rank-two matrix of order `a.length!1`, with the same
element type as the input. Every entry is written, including both triangles.
Output elements must not overlap one another or the input. Shapes are checked
with assertions; overlap is not checked. General strides are supported.
Iterator exceptions may leave partial output. Empty contractions write zeros.
Returns a symmetric descriptor retaining output ownership. Borrowed output
must remain alive while using that descriptor; ignoring it still leaves both
triangles fully initialized.
+/
package(mir.stat) SymmetricView!(TrianglePolicy.upper, C)
crossprodInto(A, C)(scope auto ref A a, return scope auto ref C output)
    if (compatible!(A, C))
{
    selfInto!(true, TrianglePolicy.upper, true)(a, output);
    return symmetricView(output);
}

/++
Write only the explicitly selected triangle and return its symmetric view.
The other triangle is untouched and must not be read through raw storage.
Shape, disjointness, and exception contracts match the full-output overload.
The returned descriptor retains output ownership; borrowed output must remain
alive. Without an explicit triangle, both triangles are always initialized.
+/
package(mir.stat) SymmetricView!(triangle, C)
crossprodInto(TrianglePolicy triangle, A, C)(scope auto ref A a, return scope auto ref C output)
    if (compatible!(A, C))
{
    selfInto!(true, triangle, false)(a, output);
    return symmetricView!triangle(output);
}

/++
Overwrite caller storage with `a.transpose * b` without allocating.
Input matrices must have equal row counts. Output has shape `(a.length!1, b.length!1)`;
all element types must match. Output must have nonoverlapping elements and be
disjoint from both inputs; the inputs may overlap each other. Shape checks use
assertions; overlap is not checked. Arithmetic, general-stride support, and
partial output on iterator exceptions follow `mtimesInto`.
+/
package(mir.stat) void crossprodInto(A, B, C)(scope auto ref A a, scope auto ref B b,
    scope auto ref C output) if (compatible!(A, B) && compatible!(A, C))
{
    mtimesInto(a.lightConst.universal.transposed, b, output);
}

/++
Compute `a * a.transpose` into GC-managed storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!0` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.gc, Scalar!A, 2))
tcrossprod(TrianglePolicy triangle = TrianglePolicy.upper, A)(scope auto ref A a) if (matrix!A)
{
    int allocator;
    return selfAllocated!(Allocation.gc, false, triangle)(allocator, a);
}

/++
Compute `a * b.transpose` into GC-managed storage.
Both inputs must be rank-two matrices with matching element types and equal
column counts. Returns shape `(a.length!0, b.length!0)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `mtimes`; incompatible
dimensions violate assertion-checked preconditions.
+/
package(mir.stat) AllocatedSlice!(Allocation.gc, Scalar!A, 2)
tcrossprod(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return mtimes(a, b.lightConst.universal.transposed);
}

/++
Compute `a * a.transpose` into reference-counted storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!0` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.rc, Scalar!A, 2))
rcTcrossprod(TrianglePolicy triangle = TrianglePolicy.upper, A)(scope auto ref A a) if (matrix!A)
{
    int allocator;
    return selfAllocated!(Allocation.rc, false, triangle)(allocator, a);
}

/++
Compute `a * b.transpose` into reference-counted storage.
Both inputs must be rank-two matrices with matching element types and equal
column counts. Returns shape `(a.length!0, b.length!0)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `rcMtimes`; incompatible
dimensions violate assertion-checked preconditions.
+/
package(mir.stat) AllocatedSlice!(Allocation.rc, Scalar!A, 2)
rcTcrossprod(A, B)(scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return rcMtimes(a, b.lightConst.universal.transposed);
}

/++
Compute `a * a.transpose` into custom-allocated storage.
Input must be a rank-two floating-point or complex ndslice. Returns a square
matrix with order `a.length!0` and the unqualified input element type.
The returned symmetric view initializes only `triangle` (upper by default).
Its logical indexing reflects that triangle; transpose does not conjugate complex values.
Use `result.asSlice.slice` for a full writable dense copy.
An empty contraction produces zeros. Input ownership is independent of result
ownership; const, RC, computed, transposed and strided inputs are supported.
Only the result is allocated.
Use `disposeSymmetric(allocator, result)` after all derived views are finished.
Custom result storage is released if an input iterator throws.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.custom, Scalar!A, 2))
makeTcrossprod(TrianglePolicy triangle = TrianglePolicy.upper, Allocator, A)(auto ref Allocator allocator, scope auto ref A a) if (matrix!A)
{
    return selfAllocated!(Allocation.custom, false, triangle)(allocator, a);
}

/++
Compute `a * b.transpose` into custom-allocated storage.
Both inputs must be rank-two matrices with matching element types and equal
column counts. Returns shape `(a.length!0, b.length!0)`. No symmetry is assumed,
and complex values are not conjugated. Strides and input ownership are
independent. Allocation and arithmetic follow `makeMtimes`; incompatible
dimensions violate assertion-checked preconditions.
Dispose `result.field` with the same allocator after all views are finished.
Custom result storage is released if an input iterator throws.
+/
package(mir.stat) AllocatedSlice!(Allocation.custom, Scalar!A, 2)
makeTcrossprod(Allocator, A, B)(auto ref Allocator allocator, scope auto ref A a, scope auto ref B b)
    if (compatible!(A, B))
{
    return makeMtimes(allocator, a, b.lightConst.universal.transposed);
}

/++
Overwrite caller storage with `a * a.transpose` without allocating.
Output must be a square rank-two matrix of order `a.length!0`, with the same
element type as the input. Every entry is written, including both triangles.
Output elements must not overlap one another or the input. Shapes are checked
with assertions; overlap is not checked. General strides are supported.
Iterator exceptions may leave partial output. Empty contractions write zeros.
Returns a symmetric descriptor retaining output ownership. Borrowed output
must remain alive while using that descriptor; ignoring it still leaves both
triangles fully initialized.
+/
package(mir.stat) SymmetricView!(TrianglePolicy.upper, C)
tcrossprodInto(A, C)(scope auto ref A a, return scope auto ref C output)
    if (compatible!(A, C))
{
    selfInto!(false, TrianglePolicy.upper, true)(a, output);
    return symmetricView(output);
}

/++
Write only the explicitly selected triangle and return its symmetric view.
The other triangle is untouched and must not be read through raw storage.
Shape, disjointness, and exception contracts match the full-output overload.
The returned descriptor retains output ownership; borrowed output must remain
alive. Without an explicit triangle, both triangles are always initialized.
+/
package(mir.stat) SymmetricView!(triangle, C)
tcrossprodInto(TrianglePolicy triangle, A, C)(scope auto ref A a, return scope auto ref C output)
    if (compatible!(A, C))
{
    selfInto!(false, triangle, false)(a, output);
    return symmetricView!triangle(output);
}

/++
Overwrite caller storage with `a * b.transpose` without allocating.
Input matrices must have equal column counts. Output has shape `(a.length!0, b.length!0)`;
all element types must match. Output must have nonoverlapping elements and be
disjoint from both inputs; the inputs may overlap each other. Shape checks use
assertions; overlap is not checked. Arithmetic, general-stride support, and
partial output on iterator exceptions follow `mtimesInto`.
+/
package(mir.stat) void tcrossprodInto(A, B, C)(scope auto ref A a, scope auto ref B b,
    scope auto ref C output) if (compatible!(A, B) && compatible!(A, C))
{
    mtimesInto(a, b.lightConst.universal.transposed, output);
}

/// Column and row Gram matrices have explicit, independent shapes.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    double[6] data = [1, 2, 3, 4, 5, 6];
    auto a = data[].sliced(2, 3);
    assert(crossprod(a).asSlice == [[17,22,27], [22,29,36], [27,36,45]]);
    assert(tcrossprod(a).asSlice == [[14,32], [32,77]]);
    double[4] rhs = [1,2,3,4];
    assert(crossprod(a, rhs[].sliced(2,2)) == [[13,18], [17,24], [21,30]]);
    double[12] other = [1,0,0, 0,1,0, 0,0,1, 1,1,1];
    assert(tcrossprod(a, other[].sliced(4,3)) == [[1,2,3,6], [4,5,6,15]]);
    assert(tcrossprod(a, a) == tcrossprod(a).asSlice);
    // Explicit shape determines whether a vector produces an inner or outer product.
    double[2] values = [2,3];
    assert(crossprod(values[].sliced(2,1)).asSlice == [[13]]);
    assert(tcrossprod(values[].sliced(2,1)).asSlice == [[4,6], [6,9]]);
    assert(crossprod(values[].sliced(1,2)).asSlice == [[4,6], [6,9]]);
    assert(tcrossprod(values[].sliced(1,2)).asSlice == [[13]]);
    static assert(!__traits(compiles, crossprod(values[].sliced)));
    static assert(!__traits(compiles, tcrossprod(values[].sliced)));
    static assert(!__traits(compiles, crossprod(a, values[].sliced)));
}

/// Reference-counted inputs and outputs have independent lifetimes.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    auto input = mininitRcslice!double(2,2);
    input[0,0]=1; input[0,1]=2; input[1,0]=3; input[1,1]=4;
    const source = input;
    auto columns = rcCrossprod(source);
    auto rows = rcTcrossprod(source, source);
    input[0,0]=100;
    static immutable expectedColumns = [[10.0,14], [14.0,20]];
    static immutable expectedRows = [[5.0,11], [11.0,25]];
    assert(columns.asSlice == expectedColumns && rows == expectedRows);
}

/// Caller-output variants overwrite the entire result without allocating.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[6] data = [1,2,3,4,5,6];
    double[9] columns = double.nan;
    double[4] rows = double.nan;
    auto a = data[].sliced(2,3);
    crossprodInto(a, columns[].sliced(3,3));
    tcrossprodInto(a, a, rows[].sliced(2,2));
    assert(columns[1] == 22 && columns[3] == 22 && columns[8] == 45);
    assert(rows[0] == 14 && rows[1] == 32 && rows[2] == 32 && rows[3] == 77);
}

/// Custom results are disposed through the allocator that created them.
version(mir_stat_test_linearAlgebra)
@system nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.views: disposeSymmetric;
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[4] data = [1,2,3,4];
    auto a = data[].sliced(2,2);
    auto columns = makeCrossprod(Mallocator.instance, a);
    scope(exit) disposeSymmetric(Mallocator.instance, columns);
    auto rows = makeTcrossprod(Mallocator.instance, a, a);
    scope(exit) Mallocator.instance.dispose(rows.field);
    assert(columns[0,1] == 14 && rows[0,1] == 11);
}

// Exercise every input/output orientation, padding, fallback strides, and numeric type.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: reversed;
    import mir.ndslice.topology: map;
    import mir.complex: Complex;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Complex!double))
    foreach (ta; [false,true]) foreach (tc; [false,true]) foreach (reverse; [false,true])
    {
        T[25] aa, bb, cc, rr;
        auto a=aa[].sliced(5,5)[0 .. (ta ? 3 : 2),0 .. (ta ? 2 : 3)].universal;
        if(ta) a=a.transposed;
        if(reverse) a=a.reversed!0;
        foreach(i;0..2) foreach(j;0..3) a[i,j]=T(i*3+j+1);
        auto b=bb[].sliced(5,5)[0..2,0..2].universal;
        b[0,0]=T(2); b[0,1]=T(1); b[1,0]=T(0); b[1,1]=T(3);
        auto c=cc[].sliced(5,5)[0..3,0..3].universal;
        auto r=rr[].sliced(5,5)[0..3,0..3].universal;
        if(tc) c=c.transposed;
        if(reverse) c=c.reversed!1;
        crossprodInto(a,c);
        mtimesInto(a.transposed,a,r);
        assert(c==r);
        static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
        {{
            foreach(i;0..3) foreach(j;0..3) c[i,j]=T(-99);
            auto logical=crossprodInto!triangle(a,c);
            assert(logical.asSlice==r);
            foreach(i;0..3) foreach(j;0..3)
                if (triangle == TrianglePolicy.upper ? j<i : j>i)
                    assert(c[i,j]==T(-99));
            auto computed=crossprodInto!triangle(a.map!"a",c);
            assert(computed.asSlice==r);
        }}

        crossprodInto(a.map!"a",c);
        assert(c==r);
        crossprodInto(a,b,c[0..3,0..2]);
        mtimesInto(a.transposed,b,r[0..3,0..2]);
        assert(c[0..3,0..2]==r[0..3,0..2]);
        tcrossprodInto(a,c[0..2,0..2]);
        mtimesInto(a,a.transposed,r[0..2,0..2]);
        assert(c[0..2,0..2]==r[0..2,0..2]);
        tcrossprodInto(a,a.map!"a",c[0..2,0..2]);
        assert(c[0..2,0..2]==r[0..2,0..2]);
    }
}

// Complex products use transpose rather than conjugate transpose.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.complex: Complex;
    import mir.ndslice.slice: sliced;
    alias C=Complex!double;
    C[2] values=[C(1,1),C(2,-1)]; C[4] result;
    crossprodInto(values[].sliced(1,2), result[].sliced(2,2));
    assert(result[0]==C(0,2) && result[1]==C(3,1));
    assert(result[2]==C(3,1) && result[3]==C(3,-4));
}

// Empty contraction dimensions write zeros; zero-sized result dimensions stay empty.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[0] empty; double[9] result=double.nan;
    crossprodInto(empty[].sliced(0,3),result[].sliced(3,3));
    foreach(value;result) assert(value==0);
    result[]=double.nan;
    tcrossprodInto(empty[].sliced(3,0),result[].sliced(3,3).transposed);
    foreach(value;result) assert(value==0);
    crossprodInto(empty[].sliced(3,0),empty[].sliced(0,0));
    tcrossprodInto(empty[].sliced(0,3),empty[].sliced(0,0));
    assert(rcCrossprod(empty[].sliced(3,0)).asSlice.shape==[0,0]);
    assert(rcTcrossprod(empty[].sliced(0,3)).asSlice.shape==[0,0]);
}

// A computed input exception must release each custom result allocation.
version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import std.exception: assertThrown;
    struct CountingAllocator {
        size_t allocations, releases;
        /// Count successful storage requests.
        void[] allocate(size_t bytes) { ++allocations; return Mallocator.instance.allocate(bytes); }
        /// Count released result buffers.
        bool deallocate(void[] memory) { ++releases; return Mallocator.instance.deallocate(memory); }
    }
    CountingAllocator allocator;
    double[4] values=[1,2,3,4];
    auto a=values[].sliced(2,2);
    auto failure=new Exception("computed input failed");
    /// Throw after result allocation to exercise failure cleanup.
    double fail(double x) { throw failure; }
    assertThrown!Exception(makeCrossprod(allocator,a.map!fail));
    assertThrown!Exception(makeTcrossprod(allocator,a.map!fail));
    assertThrown!Exception(makeCrossprod(allocator,a,a.map!fail));
    assertThrown!Exception(makeTcrossprod(allocator,a,a.map!fail));
    assert(allocator.allocations==4 && allocator.releases==4);
    auto x=makeCrossprod(allocator,a,a);
    allocator.dispose(x.field);
    auto y=makeTcrossprod(allocator,a);
    disposeSymmetric(allocator, y);
    assert(allocator.allocations==6 && allocator.releases==6);
    assert(rcCrossprod(a,a)==rcCrossprod(a).asSlice);
    assert(rcTcrossprod(a).asSlice==rcTcrossprod(a,a));
}

/// Allocate one self-product result and release custom storage if computation throws.
private SymmetricView!(triangle, AllocatedSlice!(policy, Scalar!A, 2)) selfAllocated(Allocation policy,
    bool transpose, TrianglePolicy triangle, Allocator, A)(ref Allocator allocator, scope auto ref A a)
{
    const n = a.length!(transpose ? 1 : 0);
    auto result = allocate!(policy, Scalar!A)(allocator, [n, n]);
    scope(failure) { static if (policy == Allocation.custom) {
        import std.experimental.allocator: dispose;
        allocator.dispose(result.field);
    } }
    selfInto!(transpose, triangle, false)(a, result);
    return symmetricView!triangle(result);
}

/// Compute the selected triangle, optionally mirroring it without conjugation.
private void selfInto(bool transpose, TrianglePolicy triangle, bool mirror, A, C)(scope auto ref A a, scope auto ref C output)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    static if (transpose) auto rows = a.lightConst.universal.transposed;
    else auto rows = a.lightConst.universal;
    const n = rows.length!0;
    const k = rows.length!1;
    assert(output.length!0 == n && output.length!1 == n,
        "Incorrect cross-product output shape");
    alias T = Scalar!A;
    bool accelerated;
    static if (hasBlas && blasNumber!T && nativeSlice!A && nativeSlice!C)
    {
        if (n && k && blasMatrix(rows) && blasMatrix(output))
        {
            import mir.blas: syrk;
            import cblas: Uplo;
            syrk(triangle == TrianglePolicy.upper ? Uplo.Upper : Uplo.Lower, T(1), rows.lightScope.lightConst, T(0), output.lightScope);
            accelerated = true;
        }
    }
    if (!accelerated)
        foreach (i; 0 .. n) foreach (j; (triangle == TrianglePolicy.upper ? i : 0) .. (triangle == TrianglePolicy.upper ? n : i + 1))
        {
            T sum = T(0);
            foreach (p; 0 .. k) sum += rows[i,p] * rows[j,p];
            output[i,j] = sum;
        }
    static if (mirror)
    {
        static if (triangle == TrianglePolicy.upper) mirrorUpper(output);
        else mirrorUpper(output.universal.transposed);
    }
}

/// Mirror square upper storage in small tiles, keeping both triangles' cache lines nearby.
/// Output is disjoint from inputs; diagonal entries are already initialized.
private void mirrorUpper(C)(scope auto ref C output)
{
    import std.algorithm: min;
    enum tile = 32;
    const n = output.length!0;
    for (size_t row = 0; row < n; row += tile)
        for (size_t column = 0; column <= row; column += tile)
        {
            const rowEnd = min(n, row + tile);
            const columnEnd = min(n, column + tile);
            if (output._stride!0 == 1 && output._stride!1 != 1)
                foreach (j; column .. columnEnd)
                    foreach (i; (row > j ? row : j + 1) .. rowEnd)
                        output[i,j] = output[j,i];
            else
                foreach (i; row .. rowEnd)
                    foreach (j; column .. min(i, columnEnd))
                        output[i,j] = output[j,i];
        }
}

version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    foreach (n; [31,32,33,65]) foreach (transpose; [false,true])
    {
        double[65*3] input;
        double[65*65] buffer;
        auto a=input[0..n*3].sliced(n,3);
        foreach (i;0..n) foreach (k;0..3) a[i,k]=cast(double)(i+k+1);
        auto c=buffer[0..n*n].sliced(n,n).universal;
        if (transpose) c=c.transposed;
        tcrossprodInto(a,c);
        foreach (i;0..n) foreach (j;0..n)
        {
            double expected=0;
            foreach(k;0..3) expected+=(i+k+1)*(j+k+1);
            assert(c[i,j]==expected);
        }
    }
}

/// Opt in to one-triangle caller output; ordinary Into still fills both halves.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.stat.internal.linearAlgebra.types: TrianglePolicy;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: slice;
    import mir.stat.internal.linearAlgebra.quadratic: quadraticForm;
    double[4] data=[1,2,3,4], target=double.nan;
    auto a=data[].sliced(2,2);
    auto result=crossprodInto!(TrianglePolicy.lower)(a,target[].sliced(2,2));
    assert(target[1] != target[1]);
    assert(result[0,1]==14 && result[1,0]==14);
    auto dense=result.asSlice.slice;
    assert(dense==[[10,14],[14,20]]);
    auto owned=rcCrossprod!(TrianglePolicy.lower)(a);
    assert(owned.asSlice==dense);
    double[2] x=[1,2];
    assert(quadraticForm(x[].sliced,owned)==146);
    crossprodInto(a,target[].sliced(2,2));
    assert(target[1]==14);
    version (mir_stat_test_lifetime) static assert(!__traits(compiles, {
        @safe auto escape() {
            double[4] local;
            auto matrix=local[].sliced(2,2);
            return crossprodInto(matrix,matrix);
        }
    }));
}

/// Returning an Into descriptor retains reference-counted caller storage.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow @nogc unittest
{
    import mir.stat.internal.linearAlgebra.types: TrianglePolicy;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: mininitRcslice;
    /// Local output ownership is transferred through a copied descriptor.
    auto retained() {
        double[4] input=[1,2,3,4];
        auto output=mininitRcslice!double(2,2);
        return crossprodInto!(TrianglePolicy.lower)(input[].sliced(2,2),output);
    }
    auto result=retained();
    assert(result[0,1]==14 && result[1,1]==20);
}


/++
Diagonal self-product with gc storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.gc,Scalar!D,2))
crossprod(TrianglePolicy triangle = TrianglePolicy.upper,D)(
    scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{
    int allocator;
    auto result = allocate!(Allocation.gc,Scalar!D)(allocator,a.shape);

    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.gc,Scalar!D,2)
crossprod(D,B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimes(a,crossOperand!false(b))))
{
    return mtimes(a,crossOperand!false(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.gc,Scalar!D,2)
crossprod(D,B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimes(crossOperand!true(b),a)) && !DiagonalInfo!B.valid)
{
    return mtimes(crossOperand!true(b),a);
}

/++
Diagonal self-product with rc storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.rc,Scalar!D,2))
rcCrossprod(TrianglePolicy triangle = TrianglePolicy.upper,D)(
    scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{
    int allocator;
    auto result = allocate!(Allocation.rc,Scalar!D)(allocator,a.shape);

    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.rc,Scalar!D,2)
rcCrossprod(D,B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, rcMtimes(a,crossOperand!false(b))))
{
    return rcMtimes(a,crossOperand!false(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.rc,Scalar!D,2)
rcCrossprod(D,B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, rcMtimes(crossOperand!true(b),a)) && !DiagonalInfo!B.valid)
{
    return rcMtimes(crossOperand!true(b),a);
}

/++
Diagonal self-product with custom storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.custom,Scalar!D,2))
makeCrossprod(TrianglePolicy triangle = TrianglePolicy.upper,D,Allocator)(
    auto ref Allocator allocator, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{

    auto result = allocate!(Allocation.custom,Scalar!D)(allocator,a.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.custom,Scalar!D,2)
makeCrossprod(D,B,Allocator)(auto ref Allocator allocator, scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, makeMtimes(allocator,a,crossOperand!false(b))))
{
    return makeMtimes(allocator,a,crossOperand!false(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.custom,Scalar!D,2)
makeCrossprod(D,B,Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, makeMtimes(allocator,crossOperand!true(b),a)) && !DiagonalInfo!B.valid)
{
    return makeMtimes(allocator,crossOperand!true(b),a);
}

/// Diagonal self-product into caller storage; fills both triangles. Output must be disjoint from the input.
package(mir.stat) SymmetricView!(TrianglePolicy.upper,C)
crossprodInto(D,C)(scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && matrix!C && is(Scalar!D == Scalar!C))
{
    diagonalSelfInto!(true,TrianglePolicy.upper)(a,output);
    return symmetricView!(TrianglePolicy.upper)(output);
}

/// Diagonal self-product into caller storage; writes only the explicitly selected triangle. Output must be disjoint from the input.
package(mir.stat) SymmetricView!(triangle,C)
crossprodInto(TrianglePolicy triangle,D,C)(scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && matrix!C && is(Scalar!D == Scalar!C))
{
    diagonalSelfInto!(false,triangle)(a,output);
    return symmetricView!(triangle)(output);
}

/// Write a diagonal two-input cross-product. Output shape and disjointness follow mtimesInto.
package(mir.stat) void crossprodInto(D,B,C)(scope auto ref const(DiagonalView!D) a, scope auto ref B b, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimesInto(a,crossOperand!false(b),output)))
{
    mtimesInto(a,crossOperand!false(b),output);
}

/// Write a diagonal two-input cross-product. Output shape and disjointness follow mtimesInto.
package(mir.stat) void crossprodInto(D,B,C)(scope auto ref B b, scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimesInto(crossOperand!true(b),a,output)) && !DiagonalInfo!B.valid)
{
    mtimesInto(crossOperand!true(b),a,output);
}

/++
Diagonal self-product with gc storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.gc,Scalar!D,2))
tcrossprod(TrianglePolicy triangle = TrianglePolicy.upper,D)(
    scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{
    int allocator;
    auto result = allocate!(Allocation.gc,Scalar!D)(allocator,a.shape);

    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.gc,Scalar!D,2)
tcrossprod(D,B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimes(a,crossOperand!true(b))))
{
    return mtimes(a,crossOperand!true(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.gc,Scalar!D,2)
tcrossprod(D,B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimes(crossOperand!false(b),a)) && !DiagonalInfo!B.valid)
{
    return mtimes(crossOperand!false(b),a);
}

/++
Diagonal self-product with rc storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.rc,Scalar!D,2))
rcTcrossprod(TrianglePolicy triangle = TrianglePolicy.upper,D)(
    scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{
    int allocator;
    auto result = allocate!(Allocation.rc,Scalar!D)(allocator,a.shape);

    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.rc,Scalar!D,2)
rcTcrossprod(D,B)(scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, rcMtimes(a,crossOperand!true(b))))
{
    return rcMtimes(a,crossOperand!true(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.rc,Scalar!D,2)
rcTcrossprod(D,B)(scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, rcMtimes(crossOperand!false(b),a)) && !DiagonalInfo!B.valid)
{
    return rcMtimes(crossOperand!false(b),a);
}

/++
Diagonal self-product with custom storage. Returns the same symmetric-view
result as an ordinary matrix self-product, without a cubic matrix calculation.
Only the selected triangle is initialized; custom results use disposeSymmetric.
Complex diagonal entries are squared without conjugation.
+/
package(mir.stat) SymmetricView!(triangle, AllocatedSlice!(Allocation.custom,Scalar!D,2))
makeTcrossprod(TrianglePolicy triangle = TrianglePolicy.upper,D,Allocator)(
    auto ref Allocator allocator, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1)
{

    auto result = allocate!(Allocation.custom,Scalar!D)(allocator,a.shape);
    scope(failure) { import std.experimental.allocator: dispose; allocator.dispose(result.field); }
    diagonalSelfInto!(false,triangle)(a,result);
    return symmetricView!triangle(result);
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.custom,Scalar!D,2)
makeTcrossprod(D,B,Allocator)(auto ref Allocator allocator, scope auto ref const(DiagonalView!D) a, scope auto ref B b)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, makeMtimes(allocator,a,crossOperand!true(b))))
{
    return makeMtimes(allocator,a,crossOperand!true(b));
}

/// Diagonal two-input cross-product; other inputs are logical matrices and no symmetry is assumed.
package(mir.stat) AllocatedSlice!(Allocation.custom,Scalar!D,2)
makeTcrossprod(D,B,Allocator)(auto ref Allocator allocator, scope auto ref B b, scope auto ref const(DiagonalView!D) a)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, makeMtimes(allocator,crossOperand!false(b),a)) && !DiagonalInfo!B.valid)
{
    return makeMtimes(allocator,crossOperand!false(b),a);
}

/// Diagonal self-product into caller storage; fills both triangles. Output must be disjoint from the input.
package(mir.stat) SymmetricView!(TrianglePolicy.upper,C)
tcrossprodInto(D,C)(scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && matrix!C && is(Scalar!D == Scalar!C))
{
    diagonalSelfInto!(true,TrianglePolicy.upper)(a,output);
    return symmetricView!(TrianglePolicy.upper)(output);
}

/// Diagonal self-product into caller storage; writes only the explicitly selected triangle. Output must be disjoint from the input.
package(mir.stat) SymmetricView!(triangle,C)
tcrossprodInto(TrianglePolicy triangle,D,C)(scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && matrix!C && is(Scalar!D == Scalar!C))
{
    diagonalSelfInto!(false,triangle)(a,output);
    return symmetricView!(triangle)(output);
}

/// Write a diagonal two-input cross-product. Output shape and disjointness follow mtimesInto.
package(mir.stat) void tcrossprodInto(D,B,C)(scope auto ref const(DiagonalView!D) a, scope auto ref B b, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimesInto(a,crossOperand!true(b),output)))
{
    mtimesInto(a,crossOperand!true(b),output);
}

/// Write a diagonal two-input cross-product. Output shape and disjointness follow mtimesInto.
package(mir.stat) void tcrossprodInto(D,B,C)(scope auto ref B b, scope auto ref const(DiagonalView!D) a, scope auto ref C output)
    if (numericSlice!D && D.N == 1 && B.N == 2 &&
        __traits(compiles, mtimesInto(crossOperand!false(b),a,output)) && !DiagonalInfo!B.valid)
{
    mtimesInto(crossOperand!false(b),a,output);
}

/// Fill the requested portion of a dense result without reading implicit zeros.
private void diagonalSelfInto(bool full,TrianglePolicy triangle,D,C)(
    scope auto ref const(DiagonalView!D) a, scope auto ref C output)
{
    static assert(triangle == TrianglePolicy.upper || triangle == TrianglePolicy.lower);
    assert(a.shape == output.shape);
    foreach(i; 0 .. a.length!0)
    {
        auto d = a[i,i];
        foreach(j; 0 .. a.length!1)
            if (full || (triangle == TrianglePolicy.upper ? j >= i : j <= i))
                output[i,j] = i == j ? d*d : Scalar!D(0);
    }
}


/// Diagonal cross-products preserve symmetric self-results and explicit triangle writes.
version(mir_stat_test_linearAlgebra)
@system unittest
{
    import mir.stat.internal.linearAlgebra.types: TrianglePolicy;
    import mir.stat.internal.linearAlgebra.views: diagonalView, disposeSymmetric;
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[2] ds=[2,3]; double[4] values=[1,2,3,4], buffer=[-99,-99,-99,-99];
    auto a=diagonalView(ds[].sliced); auto b=values[].sliced(2,2);
    assert(crossprod(a).asSlice==[[4,0],[0,9]]);
    assert(tcrossprod(a).asSlice==[[4,0],[0,9]]);
    assert(rcCrossprod(a).asSlice==rcTcrossprod(a).asSlice);
    auto custom=makeCrossprod(Mallocator.instance,a);
    scope(exit) disposeSymmetric(Mallocator.instance,custom);
    auto customT=makeTcrossprod(Mallocator.instance,a);
    scope(exit) disposeSymmetric(Mallocator.instance,customT);
    assert(custom.asSlice==customT.asSlice);
    auto outView=buffer[].sliced(2,2);
    crossprodInto!(TrianglePolicy.upper)(a,outView);
    assert(buffer==[4,0,-99,9]);
    tcrossprodInto(a,outView);
    assert(buffer==[4,0,0,9]);
    assert(crossprod(a,b)==crossprod(a.asSlice,b));
    assert(crossprod(b,a)==crossprod(b,a.asSlice));
    assert(tcrossprod(a,b)==tcrossprod(a.asSlice,b));
    assert(tcrossprod(b,a)==tcrossprod(b,a.asSlice));
    crossprodInto(a,b,outView); assert(outView==rcCrossprod(a,b));
    tcrossprodInto(b,a,outView); assert(outView==rcTcrossprod(b,a));
    auto ab=makeCrossprod(Mallocator.instance,a,b);
    scope(exit) Mallocator.instance.dispose(ab.field);
    auto ba=makeTcrossprod(Mallocator.instance,b,a);
    scope(exit) Mallocator.instance.dispose(ba.field);
    assert(ab==rcCrossprod(a,b) && ba==rcTcrossprod(b,a));
}

/// Preserve diagonal structure through an ordinary transpose; adapt other logical matrices.
private auto crossOperand(bool transpose, B)(return scope B b)
{
    static if (DiagonalInfo!B.valid) return b;
    else static if (transpose) return logicalOperand(b).universal.transposed;
    else return logicalOperand(b);
}


// Two diagonal operands preserve implicit zeros, including with nonfinite entries.
version(mir_stat_test_linearAlgebra)
@safe pure nothrow unittest
{
    import mir.ndslice.slice: sliced;
    import mir.complex: Complex;
    double[2] values = [double.nan, 3];
    auto d = diagonalView(values[].sliced);
    auto c = crossprod(d, d);
    auto t = tcrossprod(d, d);
    assert(c[0,1] == 0 && c[1,0] == 0 && c[1,1] == 9);
    assert(t[0,1] == 0 && t[1,0] == 0 && t[1,1] == 9);
    alias Z = Complex!double;
    Z[2] complexValues = [Z(1,1), Z(2,-1)];
    auto z = diagonalView(complexValues[].sliced);
    auto product = crossprod(z);
    assert(product[0,0] == Z(0,2) && product[1,1] == Z(3,-4));
    assert(product[0,1] == Z(0));
}
