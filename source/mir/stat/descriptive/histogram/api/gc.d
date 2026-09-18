/++
Factories for histograms with garbage-collected count storage.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.api.gc;

import mir.stat.descriptive.histogram.api.factory: HistogramFactory, NoAllocationContext;

private auto allocateCounts(T)(ref NoAllocationContext context, size_t length) @safe pure nothrow
{
    import mir.ndslice.slice: sliced;
    return (new T[length]).sliced;
}

private mixin HistogramFactory!allocateCounts implementation;

/++
Construct a histogram with garbage-collected count storage.
Accepts the same axes, bin-count rules, type overrides, and options as
$(REF rchistogram, mir, stat, descriptive, histogram, api, rc).
Counts, including enabled underflow/overflow bins, start at zero.

Count allocation does not change axis boundary ownership: borrowed variable-axis
boundaries must still outlive the histogram. Construction allocates GC memory;
subsequent counting can be `@nogc`.
+/
template histogram(Options...)
{
    // Borrow lvalue handles without an extra RC copy. Pass arguments directly:
    // core.lifetime.forward can hide borrowed-memory escapes from DIP1000.
    auto histogram(Args...)(auto ref Args args)
    {
        NoAllocationContext context;
        static if (Options.length)
            return implementation.factory!Options(context, args);
        else
            return implementation.factory(context, args);
    }
}

/// Construct two equal-width bins from observations.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    auto data = [0.0, 1, 2, 3].sliced;
    auto h = data.histogram!RegularAxis(2u, 0.0, 4.0);
    assert(h.counts == [2u, 2]);
    static assert(is(typeof(h.counts.iterator) == uint*));
}

/// Override the counter type and include underflow and overflow bins.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    alias Axis = RegularAxis!(ulong, double, AxisOptions(false, true, true));
    auto h = [-1.0, 0, 1, 2, 3, 4].sliced.histogram!Axis(2, 0.0, 4.0);
    assert(h.counts == [1UL, 2, 2, 1]);
    assert(h.underflow == 1 && h.overflow == 1);
    static assert(is(h.CountType == ulong));
}

/// Build a frequency accumulator sharing the histogram's GC-backed counts.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import mir.stat.descriptive.histogram.frequency: FrequencyAccumulator;

    auto h = [0.0, 1, 2, 3].sliced.histogram!RegularAxis(2u, 0.0, 4.0);
    auto f = FrequencyAccumulator!(typeof(h.counts), typeof(h.axis[0]))(
        h.counts, h.axis[0]);
    assert(f.count == 4);
    // Make subsequent updates through f so its total stays synchronized.
    f.put(1.0);
    assert(f.count == 5);
    assert(h.counts == [3u, 2]); // The count storage is shared.
}


// The result owns counts independently of the factory's local observations.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    static auto makeOwned() @safe pure nothrow
    {
        double[4] values = [0, 1, 2, 3];
        return values[].sliced.histogram!RegularAxis(2u, 0.0, 4.0);
    }
    auto h = makeOwned();
    assert(h.counts == [2, 2]);
    static void update(H)(ref H value) @safe pure nothrow @nogc
    {
        value.put(0.5);
    }
    update(h);
    assert(h.counts == [3, 2]);
}
