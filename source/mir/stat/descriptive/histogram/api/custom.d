/++
Construct histograms using a caller-selected allocator.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.api.custom;

import mir.ndslice.slice: Slice, SliceKind;
import mir.stat.descriptive.histogram.traits: isAxis;

/++
Allocate counts with a caller-selected allocator and a preconstructed axis.
Counts start at zero, including enabled underflow/overflow bins. All elements
of the observation slice are inserted into the one-dimensional histogram.

The allocator follows $(REF makeSlice, mir, ndslice, allocation) conventions.
It is passed by reference and is not stored in the result. The returned count
slice does not own its allocation: the caller must keep the storage valid and
release it through the same allocator when no histogram or view uses it.
Axis boundary ownership is unchanged.

If initialization or insertion throws, the factory disposes of the allocated
counts. Attributes are inferred from allocation, cleanup, and insertion;
allocators with `@system` deallocation make this factory `@system` as well.

Params:
    allocator = allocator instance providing allocation and deallocation
    observations = observations to count
    axis = axis defining the bins and counter type
+/
auto makeHistogram(Allocator, Iterator, size_t N, SliceKind kind, Axis)(
    ref Allocator allocator, Slice!(Iterator, N, kind) observations, Axis axis)
    if (isAxis!Axis)
{
    import mir.ndslice.allocation: makeSlice;
    import mir.stat.descriptive.histogram.traits: storageExtent;
    import mir.stat.descriptive.histogram.api.factory: initializeHistogram;
    import std.experimental.allocator: dispose;

    auto counts = makeSlice!(Axis.CountType)(allocator, storageExtent(axis));
    scope(failure) allocator.dispose(counts.field);
    return initializeHistogram(counts, axis, observations);
}

/// Allocate and count without using the GC, then release the count storage.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;

    double[4] data = [0, 1, 2, 3];
    auto axis = RegularAxis!(uint, double, AxisOptions())(2u, 0.0, 4.0);
    auto h = makeHistogram(Mallocator.instance, data[].sliced, axis);
    scope(exit) Mallocator.instance.dispose(h.counts.field);
    assert(h.counts == [2u, 2]);
    h.put(0.5);
    assert(h.counts == [3u, 2]);
}

version(mir_stat_test)
private struct CountingAllocator
{
    import std.experimental.allocator.mallocator: Mallocator;
    enum alignment = Mallocator.alignment;
    size_t allocations;
    size_t releases;
    size_t lastBytes;

    void[] allocate(size_t bytes) @safe pure nothrow @nogc
    {
        ++allocations;
        lastBytes = bytes;
        return Mallocator.instance.allocate(bytes);
    }

    bool deallocate(void[] memory) @system pure nothrow @nogc
    {
        ++releases;
        return Mallocator.instance.deallocate(memory);
    }
}

// A GC-backed allocator whose release operation leaves reclamation to the GC.
// This permits testing genuine @safe allocation and cleanup without @trusted
// wrappers around manual frees, which could invalidate aliases.
version(mir_stat_test)
private struct SafeAllocator
{
    import std.experimental.allocator.gc_allocator: GCAllocator;
    enum alignment = GCAllocator.alignment;
    size_t allocations;
    size_t releases;

    void[] allocate(size_t bytes) @safe pure nothrow
    {
        ++allocations;
        return GCAllocator.instance.allocate(bytes);
    }

    bool deallocate(void[] memory) @safe pure nothrow @nogc
    {
        ++releases;
        return true;
    }
}

// Safety depends on the allocator's contract, including its release operation.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.experimental.allocator: dispose;

    SafeAllocator allocator;
    double[4] values = [0, 1, 2, 3];
    auto h = makeHistogram(allocator, values[].sliced,
        RegularAxis!(uint, double, AxisOptions())(2u, 0.0, 4.0));
    assert(allocator.allocations == 1 && allocator.releases == 0);
    assert(h.counts == [2, 2]);
    allocator.dispose(h.counts.field);
    assert(allocator.releases == 1);
}

// A stateful allocator is not copied; every float counter starts at zero.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.experimental.allocator: dispose;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        CountingAllocator allocator;
        double[0] empty;
        alias Axis = RegularAxis!(T, double, AxisOptions(false, true, true));
        auto h = makeHistogram(allocator, empty[].sliced, Axis(3, 0.0, 3.0));
        assert(allocator.allocations == 1 && allocator.releases == 0);
        assert(allocator.lastBytes == 5 * T.sizeof);
        assert(h.counts == [0, 0, 0, 0, 0]);
        h.put(-1.0, 0.5, 2.5, 3.0);
        assert(h.counts == [1, 1, 0, 1, 1]);
        allocator.dispose(h.counts.field);
        assert(allocator.releases == 1);
    }}
}

// Release exactly once if insertion fails after allocation.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.exception: assertThrown;

    static double failOnTwo(double x) @safe pure
    {
        if (x == 2) throw new Exception("test insertion failure");
        return x;
    }
    double[3] values = [0, 1, 2];
    CountingAllocator allocator;
    assertThrown!Exception(makeHistogram(allocator,
        values[].sliced.map!failOnTwo, RegularAxis!(uint, double, AxisOptions())(3u, 0.0, 3.0)));
    assert(allocator.allocations == 1 && allocator.releases == 1);
}

// Observation layout does not change the allocation shape of a one-axis histogram.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.experimental.allocator: dispose;

    double[6] values = [0.5, 0.25, 1.5, 0.25, 2.5, 0.25];
    auto matrix = values[].sliced(3, 2);
    auto axis = RegularAxis!(uint, double, AxisOptions())(3u, 0.0, 3.0);
    CountingAllocator allocator;
    auto all = makeHistogram(allocator, matrix.transposed, axis);
    scope(exit) allocator.dispose(all.counts.field);
    auto column = makeHistogram(allocator, matrix.transposed[0], axis);
    scope(exit) allocator.dispose(column.counts.field);
    assert(all.counts == [4, 1, 1]);
    assert(column.counts == [1, 1, 1]);
    assert(allocator.allocations == 2);
}

// Borrowed boundaries cannot escape; a local allocator handle is not retained.
version(mir_stat_test)
version(mir_stat_test_lifetime)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: VariableAxis, RegularAxis, AxisOptions;
    import mir.rc.array: RCI;

    static assert(!__traits(compiles, () @safe {
        SafeAllocator allocator;
        double[3] edges = [0, 1, 3];
        double[1] values = [0.5];
        auto axis = VariableAxis!(uint, double*, AxisOptions())(edges[].sliced);
        auto h = makeHistogram(allocator, values[].sliced, axis);
        return h;
    }));

    static auto owned() @safe pure nothrow
    {
        SafeAllocator allocator;
        double[3] edges = [0, 1, 3];
        double[1] values = [0.5];
        auto axis = VariableAxis!(uint, RCI!double, AxisOptions())(rcslice!double(edges[]));
        // This allocator uses GC-backed memory, which survives the local handle.
        return makeHistogram(allocator, values[].sliced, axis);
    }
    auto h = owned();
    assert(h.counts == [1, 0]);
    h.put(2.0);
    assert(h.counts == [1, 1]);
    assert(h.axis[0].bin(1).high == 3.0);
}
