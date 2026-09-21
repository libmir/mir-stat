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
    Options = axis template/type, counter/coordinate types, transforms, rules, and options
+/
template makeHistogram(Options...)
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.traits: isAxis;

    // Preserve the direct construction path for an explicitly supplied axis.
    // Shared overload dispatch is only needed when the axis must be deduced.
    /++
    Params:
        allocator = allocator instance providing allocation and deallocation
        observations = observations to count
        axis = axis defining the bins and counter type
    +/
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
    /++
    Params:
        allocator = allocator instance providing allocation and deallocation
        args = observation slice followed by axis construction arguments
    +/
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
    // Cleanup must also use the original noncopyable allocator instance.
    import std.experimental.allocator: dispose;
    assert(allocator.backing.releases == 0);
    allocator.dispose(h.counts.field);
    assert(allocator.backing.releases == 1);
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

/++
Construct a relative-frequency accumulator with caller-allocated count storage.
Accepts the same arguments and axis options as $(LREF makeHistogram).
The total is calculated from the stored counts, including enabled underflow
and overflow bins. Out-of-range observations follow the underlying histogram
factory's axis rules. This scans the bins once without allocating another count
buffer. Axis ownership is unchanged.
Counter types must accommodate both each bin and the total.

The caller owns the count allocation and must release it through the same
allocator after all uses of the accumulator and its views. The allocator is
not retained. Construction failure releases allocated counts.
+/
template makeRelativeFrequencyHistogram(Options...)
{
    auto makeRelativeFrequencyHistogram(Allocator, Args...)(ref Allocator allocator, auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = makeHistogram!Options(allocator, args);
        scope(failure) releaseCounts(allocator, h.counts);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct relative frequencies directly and keep the total updated.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.experimental.allocator.mallocator: Mallocator;
    double[4] values = [0, 1, 1, 3];
    auto f = makeRelativeFrequencyHistogram!RegularAxis(Mallocator.instance, values[].sliced, 2u, 0.0, 4.0);
    // Counts are read-only; the cast is solely for final manual deallocation.
    scope(exit) Mallocator.instance.deallocate(cast(void[]) f.counts.field);
    assert(f.total == 4);
    assert(f.relativeFrequency(0) == 0.75);
    f.put(3.5);
    assert(f.total == 5);
    assert(f.relativeFrequency(1) == 0.4);
}

version(mir_stat_test)
package template customRelativeFrequencyForTests(Options...)
{
    auto customRelativeFrequencyForTests(Args...)(auto ref Args args)
    {
        SafeAllocator allocator;
        return makeRelativeFrequencyHistogram!Options(allocator, args);
    }
}

// Custom construction retains allocator identity and releases on insertion failure.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.exception: assertThrown;

    static double fail(double value) @safe pure { throw new Exception("insertion"); }
    CountingAllocator allocator;
    double[2] values = [0.5, 1.5];
    assertThrown!Exception(makeRelativeFrequencyHistogram!RegularAxis(
        allocator, values[].sliced.map!fail, 2u, 0.0, 2.0));
    assert(allocator.allocations == 1 && allocator.releases == 1);
    auto f = makeRelativeFrequencyHistogram!RegularAxis(
        allocator, values[].sliced, 2u, 0.0, 2.0);
    assert(allocator.allocations == 2 && allocator.releases == 1);
    assert(f.total == 2 && f.relativeFrequency(0) == 0.5);
    allocator.deallocate(cast(void[]) f.counts.field);
    assert(allocator.releases == 2);
}

// Conversion failure after insertion must release the completed count allocation.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;

    struct ThrowingCopyAxis
    {
        alias CountType = uint;
        alias BinType = double;
        enum N_bin = 1;
        bool populated;

        size_t index(double value) @safe pure nothrow @nogc
        {
            populated = true;
            return 0;
        }

        this(this) @safe pure
        {
            if (populated)
                throw new Exception("axis copy after insertion");
        }
    }

    CountingAllocator allocator;
    double[1] values = [0.5];
    // Plain construction succeeds: the exception is specific to copying the
    // populated axis when the relative-frequency wrapper is constructed.
    auto h = makeHistogram(allocator, values[].sliced, ThrowingCopyAxis());
    assert(h.counts[0] == 1);
    allocator.deallocate(cast(void[]) h.counts.field);
    assert(allocator.allocations == 1 && allocator.releases == 1);

    assertThrown!Exception(makeRelativeFrequencyHistogram(
        allocator, values[].sliced, ThrowingCopyAxis()));
    assert(allocator.allocations == 2 && allocator.releases == 2);

    uint[1] weights = [2];
    auto weighted = makeWeightedHistogram(allocator, values, weights, ThrowingCopyAxis());
    assert(weighted.counts[0] == 2);
    allocator.deallocate(cast(void[]) weighted.counts.field);
    assert(allocator.allocations == 3 && allocator.releases == 3);
    assertThrown!Exception(makeWeightedRelativeFrequencyHistogram(
        allocator, values, weights, ThrowingCopyAxis()));
    assert(allocator.allocations == 4 && allocator.releases == 4);
}

/++
A manually managed percentogram and the original allocations backing it.
Use `histogram` for counts, relative frequencies, densities, and traversal.
Call `dispose` exactly once across all copies, through the original allocator,
after all aliases and borrowed views have finished using either allocation.
The allocator is not retained and there is no automatic destructor cleanup.
Copies share storage; do not update independent copies of the accumulator.
+/
struct AllocatedPercentogram(Histogram, BoundaryStorage, CountStorage)
{
    /// Existing relative-frequency accumulator; do not replace it while storage is in use.
    Histogram histogram;
    private BoundaryStorage boundaries;
    private CountStorage counts;
    private bool active;

    private this(Histogram value, BoundaryStorage edges, CountStorage storage)
    {
        histogram = value;
        boundaries = edges;
        counts = storage;
        active = true;
    }

    /++
    Release counts and the original boundary allocation through the same allocator.
    Duplicate compaction does not shorten the allocation passed to cleanup.
    Repeated disposal of this instance is harmless; other copies become invalid.
    Deallocation must not throw. Attributes follow the allocator's operations.
    +/
    void dispose(Allocator)(ref Allocator allocator)
    {
        import std.experimental.allocator: dispose;
        if (!active) return;
        allocator.dispose(counts.field);
        allocator.dispose(boundaries.field);
        counts = CountStorage.init;
        boundaries = BoundaryStorage.init;
        histogram = Histogram.init;
        active = false;
    }
}

/++
Construct a percentogram with caller-selected allocation for scratch, boundaries,
and counts. Accepts the same observations and bin count or probabilities as
$(REF percentogram, mir, stat, descriptive, histogram, api, gc).
The result exposes a `histogram` and must be disposed through the same allocator.
Generated probabilities and quantile scratch are released before return.
Exceptions during construction release completed allocations. Invalid-input
assertions are contract violations; recovery through nothrow code is not supported.
Deallocation must not throw.
Only the active boundary slice is compacted: the full original allocation is
retained for cleanup. Attributes depend on the allocator.

Omitting probabilities requests `ceil(cuberoot(n))` ordinary bins for `n` observations,
with equally spaced probabilities from zero to one. This is a sample-size heuristic.
Tied boundaries can reduce the number of ordinary bins.

Params:
    allocator = allocator providing allocation and nonthrowing deallocation
    data = one-dimensional observations, as an array or Mir slice
    probabilities = positive bin count or probability array/slice within zero to one
+/
auto makePercentogram(Allocator, Data, P)(ref Allocator allocator,
    scope auto ref Data data, scope auto ref P probabilities)
{
    import std.traits: isIntegral;
    import std.experimental.allocator: dispose;
    import mir.ndslice.slice: isSlice, sliced;
    import mir.ndslice.topology: as;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.univariate: makeQuantile;
    import mir.stat.descriptive.histogram.axis: variableAxis, AxisOptions;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
    import mir.stat.descriptive.histogram.api.factory: validatePercentogramInputs, preparePercentogramEdges;
    static if (isIntegral!P)
    {
        assert(probabilities > 0 && probabilities < size_t.max,
            "percentogram: bin count must be positive and leave room for an extra boundary");
        auto levels = allocateCounts!double(allocator, cast(size_t) probabilities + 1);
        scope(exit) allocator.dispose(levels.field);
        foreach (i; 0 .. levels.length)
            levels[i] = cast(double) i / probabilities;
        return makePercentogram(allocator, data, levels);
    }
    else
    {
        static if (isSlice!Data) scope auto observations = data;
        else scope auto observations = data[].sliced;
        static if (isSlice!P) scope auto levels = probabilities;
        else scope auto levels = probabilities[].sliced;
        validatePercentogramInputs(observations, levels);
        auto edges = makeQuantile(allocator, observations, levels);
        scope(failure) allocator.dispose(edges.field);
        const distinct = preparePercentogramEdges(edges);
        auto axis = variableAxis!(AxisOptions(false, true, true))(edges[0 .. distinct]);
        auto h = makeHistogram(allocator,
            observations.as!(DeepElementType!(typeof(edges))), axis);
        static if (is(typeof(h) == HistogramAccumulator!Args, Args...))
        {
            auto f = RelativeFrequencyAccumulator!Args(h.counts, h.axis);
            return AllocatedPercentogram!(typeof(f), typeof(edges), typeof(h.counts))(f, edges, h.counts);
        }
    }
}

/// ditto
auto makePercentogram(Allocator, Data)(ref Allocator allocator, scope auto ref Data data)
{
    import mir.stat.descriptive.histogram.api.factory: defaultPercentogramBinCount;
    return makePercentogram(allocator, data, defaultPercentogramBinCount(data.length));
}

/// Choose the bin count from the sample size without the GC; explicitly release the result.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    double[8] data = [0, 1, 2, 3, 4, 8, 12, 16];
    // Eight observations request two bins, with probabilities [0, 0.5, 1].
    auto p = makePercentogram(Mallocator.instance, data);
    scope(exit) p.dispose(Mallocator.instance);
    assert(p.histogram.total == 8 && p.histogram.counts == [0, 4, 4, 0]);
    assert(p.histogram.density(0) == 0.5 / 3.5);

    // Override the default with four ordinary bins: probabilities [0, 0.25, 0.5, 0.75, 1].
    auto quartiles = makePercentogram(Mallocator.instance, data, 4);
    scope(exit) quartiles.dispose(Mallocator.instance);
    // The first and last counters are underflow and overflow, both zero here.
    assert(quartiles.histogram.counts == [0, 2, 2, 2, 2, 0]);
    assert(quartiles.histogram.relativeFrequency(0) == 0.25);
    assert(quartiles.histogram.density(0) == 0.25 / 1.75);
}

/// Mir slices and built-in dynamic arrays support explicit probability intervals.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import mir.ndslice.slice: sliced;
    double[5] data = [0, 0, 1, 2, 2];
    const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
    auto p = makePercentogram(Mallocator.instance, data[].sliced, levels[]);
    scope(exit) p.dispose(Mallocator.instance);
    // Five quantiles become three boundaries. Disposal still releases all five slots.
    assert(p.histogram.axis.N_bin == 2 && p.histogram.counts == [0, 2, 3, 0]);
}

/// Restricted quantile intervals keep tail counts available for normalization.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    double[9] data = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const double[3] levels = [0.25, 0.5, 0.75];
    auto p = makePercentogram(Mallocator.instance, data, levels);
    scope(exit) p.dispose(Mallocator.instance);
    assert(p.histogram.underflow == 2 && p.histogram.overflow == 2);
    assert(p.histogram.relativeFrequency(0) == 2.0 / 9);
    assert(p.histogram.relativeFrequency!(double, Normalization.ordinary)(0) == 2.0 / 5);
}

// Manual disposal is required, and copies do not represent independent ownership.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    // This allocator records release requests but leaves reclamation to the GC.
    // Misuse can therefore be demonstrated without an unmanaged leak or double free.
    double[3] data = [0, 1, 2];
    const double[3] levels = [0, 0.5, 1];
    SafeAllocator omitted;
    {
        auto p = makePercentogram(omitted, data, levels);
        assert(p.histogram.total == 3);
        // Scratch was released automatically; boundaries and counts remain allocated.
        assert(omitted.allocations == 3 && omitted.releases == 1);
    }
    // Going out of scope does not release the result's two allocations.
    // With a manually reclaimed allocator, omitting dispose would leak them.
    assert(omitted.allocations - omitted.releases == 2);

    SafeAllocator copied;
    auto original = makePercentogram(copied, data, levels);
    auto aliasCopy = original; // Shares both allocations; does not allocate new storage.
    assert(copied.allocations == 3 && copied.releases == 1);
    original.dispose(copied);
    assert(copied.releases == copied.allocations);
    original.dispose(copied); // The same instance remembers it was disposed.
    assert(copied.releases == 3);

    // The copy still has its own active flag. Never do this with a real freeing
    // allocator: it would attempt to free the same boundaries and counts twice.
    // Do not read aliasCopy.histogram after the first disposal either.
    aliasCopy.dispose(copied);
    assert(copied.releases == 5); // Two duplicate release requests, not two new allocations.
}

// A safe allocator supports construction and cleanup in @safe code.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    SafeAllocator allocator;
    int[3] data = [0, 1, 2];
    auto p = makePercentogram(allocator, data, 2);
    assert(p.histogram.total == 3);
    p.dispose(allocator);
    assert(allocator.allocations == allocator.releases);
    p.dispose(allocator);
    assert(allocator.allocations == allocator.releases);
}

version(mir_stat_test)
private struct PercentogramAllocator(bool throwing = false)
{
    import std.experimental.allocator.mallocator: Mallocator;
    enum alignment = Mallocator.alignment;
    size_t allocations, releases, failAt;
    private void*[8] addresses;
    private size_t[8] lengths;
    private bool[8] live;
    auto allocate(size_t bytes)
    {
        ++allocations;
        static if (throwing)
            if (allocations == failAt)
                throw new Exception("percentogram allocation failure");
        auto memory = Mallocator.instance.allocate(bytes);
        const i = allocations - 1;
        assert(i < addresses.length);
        addresses[i] = memory.ptr;
        lengths[i] = memory.length;
        live[i] = true;
        return memory;
    }
    bool deallocate(void[] memory) @system pure nothrow @nogc
    {
        foreach (i; 0 .. addresses.length)
            if (live[i] && addresses[i] == memory.ptr)
            {
                // In particular, disposal must not use the compacted boundary length.
                assert(lengths[i] == memory.length);
                live[i] = false;
                ++releases;
                return Mallocator.instance.deallocate(memory);
            }
        assert(false, "unknown allocation or double release");
    }
}

// Compacted extrema and interior duplicates retain full allocation extents for disposal.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    const double[5][4] samples = [[0, 1, 1, 1, 2], [0, 0, 0, 1, 2],
        [0, 1, 2, 2, 2], [0, 0, 1, 2, 2]];
    const uint[4] firstCounts = [1, 3, 1, 2];
    const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
    foreach (i; 0 .. samples.length)
    {
        PercentogramAllocator!() allocator;
        auto p = makePercentogram(allocator, samples[i], levels);
        assert(p.histogram.counts == [0, firstCounts[i], 5 - firstCounts[i], 0]);
        assert(allocator.allocations == 3 && allocator.releases == 1); // scratch only
        p.dispose(allocator);
        assert(allocator.releases == 3);
        p.dispose(allocator);
        assert(allocator.releases == 3);
    }
    PercentogramAllocator!() allocator;
    auto p = makePercentogram(allocator, samples[3], 4);
    assert(allocator.allocations == 4 && allocator.releases == 2); // levels and scratch
    assert(p.histogram.total == 5);
    p.dispose(allocator);
    assert(allocator.releases == 4);
}

// Every allocation failure releases the allocations that preceded it.
version(mir_stat_test)
@system pure
unittest
{
    import std.exception: assertThrown;
    double[3] data = [0, 1, 2];
    foreach (failure; 1 .. 5)
    {
        PercentogramAllocator!true allocator;
        allocator.failAt = failure;
        assertThrown!Exception(makePercentogram(allocator, data, 2));
        assert(allocator.allocations == failure && allocator.releases == failure - 1);
    }
}

// With exception unwinding enabled, failed boundary validation also cleans up.
version(mir_stat_test)
@system pure
unittest
{
    import core.exception: AssertError;
    PercentogramAllocator!true allocator;
    double[3] data = [1, 1, 1];
    bool rejected;
    try { makePercentogram(allocator, data, 2); }
    catch (AssertError) { rejected = true; }
    assert(rejected && allocator.allocations == 3 && allocator.releases == 3);
}

// The result owns allocations independently of local observations and probabilities.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    static auto fromLocal()
    {
        double[3] data = [2, 0, 1];
        const double[3] levels = [0, 0.5, 1];
        return makePercentogram(Mallocator.instance, data, levels);
    }
    auto p = fromLocal();
    scope(exit) p.dispose(Mallocator.instance);
    assert(p.histogram.total == 3 && p.histogram.counts == [0, 1, 2, 0]);
}

// User code can throw while validating, copying, or inserting observations.
version(mir_stat_test)
@system pure
unittest
{
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    double[3] data = [0, 1, 2];
    const double[3] levels = [0, 0.5, 1];
    foreach (failure; [1, 4, 7])
    {
        PercentogramAllocator!() allocator;
        size_t calls;
        double read(double value)
        {
            if (++calls == failure)
                throw new Exception("percentogram observation failure");
            return value;
        }
        auto mapped = data[].sliced.map!read;
        assertThrown!Exception(makePercentogram(allocator, mapped, levels));
        assert(calls == failure);
        assert(allocator.allocations == (failure == 1 ? 0 : failure == 4 ? 1 : 3));
        assert(allocator.releases == allocator.allocations);
    }
}

version(mir_stat_test)
private auto customPercentogramIntervalFactory(Data, P)(scope auto ref Data data, scope auto ref P probabilities)
{
    import std.experimental.allocator.mallocator: Mallocator;
    return makePercentogram(Mallocator.instance, data, probabilities);
}

version(mir_stat_test)
private void releasePercentogramForTests(T)(ref T result)
{
    import std.experimental.allocator.mallocator: Mallocator;
    result.dispose(Mallocator.instance);
}

// Restricted intervals retain tail counts and the usual explicit disposal contract.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramIntervals;
    testPercentogramIntervals!(customPercentogramIntervalFactory, releasePercentogramForTests)();
}

// Sample-size defaults match explicit probabilities, including tied boundaries.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.experimental.allocator.mallocator: Mallocator;
    const double[8] data = [0, 0, 0, 0, 0, 2, 3, 4];
    const double[3] levels = [0, 0.5, 1];
    auto automatic = makePercentogram(Mallocator.instance, data[].sliced);
    scope(exit) automatic.dispose(Mallocator.instance);
    auto explicit = makePercentogram(Mallocator.instance, data, levels);
    scope(exit) explicit.dispose(Mallocator.instance);
    assert(automatic.histogram.counts == explicit.histogram.counts);
    assert(automatic.histogram.axis.N_bin == 1);
    foreach (i; 0 .. automatic.histogram.axis.N_bin)
        assert(automatic.histogram.density(i) == explicit.histogram.density(i));
    // Use the logical length of a strided view: nine observations request three bins.
    import mir.ndslice.topology: stride;
    double[18] backing;
    foreach (i, ref value; backing)
        value = i;
    auto strided = makePercentogram(Mallocator.instance, backing[].sliced.stride(2));
    scope(exit) strided.dispose(Mallocator.instance);
    assert(strided.histogram.total == 9);
    assert(strided.histogram.axis.N_bin == 3);
    assert(strided.histogram.counts == [0, 3, 3, 3, 0]);
}

private import mir.stat.descriptive.histogram.api.factory: WeightedHistogramFactory;
private mixin WeightedHistogramFactory!(allocateCounts, releaseCounts) weightedImplementation;

/++
Construct a weighted histogram with caller-allocated counts.
Supply observations, weights, and the usual histogram axis arguments after the allocator.
Built-in arrays and Mir slices are accepted. Their shapes must match; matching
multidimensional slices are traversed elementwise into a one-axis histogram.
Weights must be finite, nonnegative, and implicitly convertible to the counter
type. Axis templates default to `double` counters, independently of the bin-count
argument. An explicit counter override or concrete axis retains its counter type.
Integral counters require integral weights. Counts must accommodate their sums.
Bin-count rules operate on observations, without weighting the rule itself.
Axis ownership and explicit count disposal follow $(LREF makeHistogram).
+/
template makeWeightedHistogram(Options...)
{
    auto makeWeightedHistogram(Allocator, Data, Weights, Args...)(ref Allocator allocator,
        scope auto ref Data data, scope auto ref Weights weights, auto ref Args args)
    {
        return weightedImplementation.weightedFactory!Options(allocator, data, weights, args);
    }
}

/++
Construct relative frequencies from weighted counts. Accepts the arguments and
counter-type choices of $(LREF makeWeightedHistogram). The total is the sum of
stored weights, including enabled underflow/overflow bins. Normalization and
subsequent weighted insertion use the existing relative-frequency accumulator.
+/
template makeWeightedRelativeFrequencyHistogram(Options...)
{
    auto makeWeightedRelativeFrequencyHistogram(Allocator, Args...)(ref Allocator allocator, auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = makeWeightedHistogram!Options(allocator, args);
        scope(failure) releaseCounts(allocator, h.counts);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct weighted counts and relative frequencies with explicit disposal.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto h = makeWeightedHistogram!RegularAxis(Mallocator.instance, observations, weights, 2u, 0.0, 2.0);
    scope(exit) Mallocator.instance.dispose(h.counts.field);
    assert(h.counts == [2.0, 2.0]);
    auto f = makeWeightedRelativeFrequencyHistogram!RegularAxis(Mallocator.instance, observations, weights, 2u, 0.0, 2.0);
    // Counts are read-only; cast only for final manual deallocation.
    scope(exit) Mallocator.instance.deallocate(cast(void[]) f.counts.field);
    assert(f.total == 4.0);
    assert(f.relativeFrequency(0) == 0.5);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.experimental.allocator: dispose;
    SafeAllocator allocator;
    double[2] data = [0.5, 1.5];
    uint[2] weights = [1, 2];
    auto h = makeWeightedHistogram!RegularAxis(allocator, data, weights, 2u, 0.0, 2.0);
    assert(h.counts == [1, 2]);
    allocator.dispose(h.counts.field);
    assert(allocator.allocations == 1 && allocator.releases == 1);
}

// Reject mismatched shapes before allocation, and release counts on insertion failure.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    CountingAllocator allocator;
    double[4] data = [0, 1, 2, 3];
    double[4] weights = [1, 2, 3, 4];
    assertThrown!AssertError(makeWeightedHistogram!RegularAxis(
        allocator, data, weights[0 .. 3], 4u, 0.0, 4.0));
    assertThrown!AssertError(makeWeightedHistogram!RegularAxis(
        allocator, data[].sliced(2, 2), weights[].sliced(1, 4), 4u, 0.0, 4.0));
    assert(allocator.allocations == 0 && allocator.releases == 0);
    static double failOnTwo(double x) @safe pure
    {
        if (x == 2) throw new Exception("weighted insertion failure");
        return x;
    }
    assertThrown!Exception(makeWeightedHistogram!RegularAxis(
        allocator, data[].sliced.map!failOnTwo, weights, 4u, 0.0, 4.0));
    assert(allocator.allocations == 1 && allocator.releases == 1);
    assertThrown!Exception(makeWeightedRelativeFrequencyHistogram!RegularAxis(
        allocator, data, weights[].sliced.map!failOnTwo, 4u, 0.0, 4.0));
    assert(allocator.allocations == 2 && allocator.releases == 2);
}
