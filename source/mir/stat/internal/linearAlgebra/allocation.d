/++
Allocation policies shared by linear algebra result factories.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.allocation;

// Result ownership policy; independent of the ownership of input slices.
package(mir.stat) enum Allocation { gc, rc, custom }

import mir.ndslice.slice: Slice;
import mir.rc.array: RCI;

// Allocated results are contiguous slices of the requested rank. GC and
// custom allocation share a pointer-backed type but retain distinct lifetimes.
package(mir.stat) template AllocatedSlice(Allocation policy, T, size_t rank)
    if (rank > 0)
{
    static if (policy == Allocation.rc)
        alias AllocatedSlice = Slice!(RCI!T, rank);
    else
        alias AllocatedSlice = Slice!(T*, rank);
}

// Allocate result storage after checking element and byte counts. Callers must
// initialize every element before reading it. Custom storage belongs to the
// caller and is released by disposing the returned slice's field.
/// Allocate checked contiguous storage; custom allocations transfer ownership to the caller.
package(mir.stat) AllocatedSlice!(policy, T, N)
allocate(Allocation policy, T, size_t N, Allocator)(
    ref Allocator allocator, size_t[N] shape)
{
    import mir.ndslice.allocation: uninitSlice, mininitRcslice, makeUninitSlice;
    import core.exception: onOutOfMemoryError;
    // Check both the element count and byte count before entering an allocator.
    size_t count = 1;
    foreach (length; shape)
    {
        if (length && count > size_t.max / length)
            onOutOfMemoryError();
        count *= length;
    }
    if (count > size_t.max / T.sizeof)
        onOutOfMemoryError();
    static if (policy == Allocation.gc)
        return uninitSlice!T(shape);
    else static if (policy == Allocation.rc)
        return mininitRcslice!T(shape);
    else
        return makeUninitSlice!T(allocator, shape);
}
