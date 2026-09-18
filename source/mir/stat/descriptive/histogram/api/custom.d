/++
Construct histograms using a caller-selected allocator.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.api.custom;

private import mir.stat.descriptive.histogram.api.factory: HistogramFactory;
private mixin HistogramFactory!(allocateCounts, releaseCounts) implementation;

private auto allocateCounts(T, Allocator)(ref Allocator allocator, size_t extent)
{
    import mir.ndslice.allocation: makeSlice;
    return makeSlice!T(allocator, extent);
}

private void releaseCounts(Allocator, Storage)(ref Allocator allocator, Storage counts)
{
    import std.experimental.allocator: dispose;
    allocator.dispose(counts.field);
}

/++
Allocate counts with a caller-selected allocator.
Accepts the same axis instances, axis templates, counter/coordinate overrides,
transforms, rules, and axis options as
$(REF rchistogram, mir, stat, descriptive, histogram, api, rc), with the allocator
as the first function argument.
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
    Options = axis template/type, counter/coordinate types, transforms, rules, and options
    args = observation slice followed by an axis instance or axis construction arguments
+/
template makeHistogram(Options...)
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.traits: isAxis;

    // Preserve the direct construction path for an explicitly supplied axis.
    // Shared overload dispatch is only needed when the axis must be deduced.
    auto makeHistogram(Allocator, Iterator, size_t N, SliceKind kind, Axis)(
        ref Allocator allocator, Slice!(Iterator, N, kind) observations, Axis axis)
        if (!Options.length && isAxis!Axis)
    {
        import mir.ndslice.allocation: makeSlice;
        import mir.stat.descriptive.histogram.traits: storageExtent;
        import mir.stat.descriptive.histogram.api.factory: initializeHistogram;
        import std.experimental.allocator: dispose;

        auto counts = makeSlice!(Axis.CountType)(allocator, storageExtent(axis));
        scope(failure) allocator.dispose(counts.field);
        return initializeHistogram(counts, axis, observations);
    }

    // Borrow lvalue handles without an extra RC copy. Pass arguments directly:
    // core.lifetime.forward can hide borrowed-memory escapes from DIP1000.
    // Value arguments match the shared overloads and preserve escape inference.
    auto makeHistogram(Allocator, Args...)(ref Allocator allocator, auto ref Args args)
        if (Options.length || Args.length != 2 || !isAxis!(Args[1]))
    {
        static if (Options.length)
            return implementation.factory!Options(allocator, args);
        else
            return implementation.factory(allocator, args);
    }
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

/// Select an axis template while retaining caller-controlled allocation.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;

    double[5] values = [-1, 0, 1, 2, 4];
    enum options = AxisOptions(false, true, true);
    auto h = makeHistogram!(ulong, double, RegularAxis, options)(
        Mallocator.instance, values[].sliced, 2u, 0.0, 4.0);
    scope(exit) Mallocator.instance.dispose(h.counts.field);
    assert(h.counts == [1UL, 2, 1, 1]);
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

// Adapt the public API to the common behavioral suite. GC-backed test storage
// survives the local allocator handle and needs no explicit manual free.
version(mir_stat_test)
package template customHistogramForTests(Options...)
{
    auto customHistogramForTests(Args...)(auto ref Args args)
    {
        SafeAllocator allocator;
        return makeHistogram!Options(allocator, args);
    }
}

// Rules run before allocation; forwarding preserves the caller's allocator.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, TransformAxis;
    import mir.math.common: log2, exp2;
    import std.experimental.allocator: dispose;

    CountingAllocator allocator;
    double[4] values = [1, 2, 4, 8];
    static uint two(S)(S data) { return 2; }
    auto regular = makeHistogram!(RegularAxis, two)(allocator, values[].sliced, 0.0, 10.0);
    scope(exit) allocator.dispose(regular.counts.field);
    assert(allocator.allocations == 1 && allocator.releases == 0);
    assert(regular.counts == [3, 1]);
    auto transformed = makeHistogram!(TransformAxis, log2, exp2, two)(
        allocator, values[].sliced, 1.0, 16.0);
    scope(exit) allocator.dispose(transformed.counts.field);
    assert(allocator.allocations == 2 && allocator.releases == 0);
    assert(transformed.counts == [2, 2]);
}

// Failure before allocation does not release storage; failure after it does.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.exception: assertThrown;

    static uint failRule(S)(S data) { throw new Exception("test rule failure"); }
    static double failValue(double x) @safe pure
    {
        throw new Exception("test insertion failure");
    }
    double[2] values = [0, 1];
    CountingAllocator allocator;
    assertThrown!Exception(makeHistogram!(RegularAxis, failRule)(
        allocator, values[].sliced, 0.0, 2.0));
    assert(allocator.allocations == 0 && allocator.releases == 0);
    assertThrown!Exception(makeHistogram!RegularAxis(
        allocator, values[].sliced.map!failValue, 2u, 0.0, 2.0));
    assert(allocator.allocations == 1 && allocator.releases == 1);
}

// A local allocator's type and local transform aliases do not require copies.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: TransformAxis;
    import mir.math.common: log2, exp2;

    struct LocalAllocator
    {
        SafeAllocator backing;
        @disable this(this);
        void[] allocate(size_t bytes) @safe pure nothrow { return backing.allocate(bytes); }
        bool deallocate(void[] memory) @safe pure nothrow @nogc { return backing.deallocate(memory); }
    }
    LocalAllocator allocator;
    double[4] values = [1, 2, 4, 8];
    auto h = makeHistogram!(TransformAxis, x => log2(x), x => exp2(x))(
        allocator, values[].sliced, 2u, 1.0, 16.0);
    assert(allocator.backing.allocations == 1);
    assert(h.counts == [2, 2]);
}

// Convenience construction borrows stack boundaries for the result's lifetime.
version(mir_stat_test)
version(mir_stat_test_lifetime)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis;

    SafeAllocator allocator;
    double[3] edges = [0, 1, 3];
    double[2] values = [0.5, 2.0];
    auto h = makeHistogram!(uint, VariableAxis)(allocator,
        values[].sliced, edges[].sliced);
    assert(h.counts == [1u, 1]);
    h.put(2.5);
    assert(h.counts == [1u, 2]);

    static assert(!__traits(compiles, () @safe {
        SafeAllocator allocator;
        double[3] edges = [0, 1, 3];
        double[1] values = [0.5];
        return makeHistogram!(uint, VariableAxis)(allocator,
            values[].sliced, edges[].sliced);
    }));
}

// An owning boundary handle can outlive the caller's copy and allocator handle.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: VariableAxis;

    static auto fromLvalue() @safe pure nothrow
    {
        SafeAllocator allocator;
        double[1] values = [0.5];
        auto edges = rcslice!double([0.0, 1.0, 3.0]);
        auto h = makeHistogram!(uint, VariableAxis)(allocator, values[].sliced, edges);
        assert(edges.length == 3 && edges[2] == 3.0);
        // The histogram must retain its own owner when this handle is released.
        edges = typeof(edges).init;
        return h;
    }
    static auto fromTemporary() @safe pure nothrow
    {
        SafeAllocator allocator;
        double[1] values = [0.5];
        return makeHistogram!(uint, VariableAxis)(allocator, values[].sliced,
            rcslice!double([0.0, 1.0, 3.0]));
    }
    auto h = fromLvalue();
    auto temporary = fromTemporary();
    h.put(2.0);
    temporary.put(2.0);
    assert(h.counts == [1u, 1]);
    assert(temporary.counts == [1u, 1]);
}
