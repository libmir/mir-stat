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
    static assert(is(typeof(h.counts.iterator) == size_t*));
}

/// Override the counter type and include underflow and overflow bins.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    alias Axis = RegularAxis!(double, AxisOptions(false, true, true));
    auto h = [-1.0, 0, 1, 2, 3, 4].sliced.histogram!(ulong, Axis)(2, 0.0, 4.0);
    assert(h.counts == [1UL, 2, 2, 1]);
    assert(h.underflow == 1 && h.overflow == 1);
    static assert(is(h.CountType == ulong));
}

/// Build a relative frequency accumulator sharing the histogram's GC-backed counts.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;

    auto h = [0.0, 1, 2, 3].sliced.histogram!RegularAxis(2u, 0.0, 4.0);
    auto f = RelativeFrequencyAccumulator!(typeof(h.counts), typeof(h.axis[0]))(
        h.counts, h.axis[0]);
    assert(f.total == 4);
    // Make subsequent updates through f so its total stays synchronized.
    f.put(1.0);
    assert(f.total == 5);
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

/++
Construct a relative-frequency accumulator with garbage-collected count storage.
Accepts the same arguments and axis options as $(LREF histogram).
The total is calculated from the stored counts, including enabled underflow
and overflow bins. Out-of-range observations follow the underlying histogram
factory's axis rules. This scans the bins once without allocating another count
buffer. Axis ownership is unchanged.
Counter types must accommodate both each bin and the total.
+/
template relativeFrequencyHistogram(Options...)
{
    auto relativeFrequencyHistogram(Args...)(auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = histogram!Options(args);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct relative frequencies directly and keep the total updated.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[4] values = [0, 1, 1, 3];
    auto f = relativeFrequencyHistogram!RegularAxis(values[].sliced, 2u, 0.0, 4.0);
    assert(f.total == 4);
    assert(f.relativeFrequency(0) == 0.75);
    f.put(3.5);
    assert(f.total == 5);
    assert(f.relativeFrequency(1) == 0.4);
}

/++
Construct a percentogram using quantile boundaries and observed relative frequencies.
Returns a relative-frequency accumulator with GC-owned boundaries and counts.
Use `density` or `densityBins` for bar heights: area represents observed probability.

Observations must be nonempty and finite and are not modified. Supply a positive
bin count or strictly increasing probabilities within zero to one. The default
quantile algorithm is type7. Equal boundaries are combined, so fewer bins may be
returned; constant observations are rejected. Counts need not be equal, especially
with ties. Observations are converted to the quantile boundary type for counting;
integral inputs use double boundaries, so sufficiently large integers may lose
precision or yield coincident boundaries. Later updates retain the original boundaries.

Both underflow and overflow counters are always enabled, including for full-range
probabilities. Raw counts contain underflow first and overflow last; ordinary-bin
indices used by relativeFrequency and density still start at zero.
Bins are left-closed and right-open. The final boundary is increased by one
representable step to include observations equal to the upper quantile cutoff;
this slightly increases its width. A cutoff without a finite successor is rejected.
Observations equal to either outer quantile cutoff are included in ordinary bins.
Values below or above the selected interval are counted in underflow/overflow.
By default they remain in the normalization total. Use `Normalization.ordinary`
on relative-frequency, density, or cumulative accessors to exclude them from the
probability distribution. With ties, actual retained counts can differ from the
requested probability span.

Omitting probabilities requests `ceil(cuberoot(n))` ordinary bins for `n` observations,
with equally spaced probabilities from zero to one. This is a sample-size heuristic.
Tied boundaries can reduce the number of ordinary bins.

Params:
    data = one-dimensional observations, as an array or slice
    probabilities = positive bin count or probability array/slice
+/
auto percentogram(Data, P)(scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.stat.descriptive.univariate: quantile;
    import mir.stat.descriptive.histogram.api.factory: buildPercentogram;
    return buildPercentogram!(allocateCounts, quantile, relativeFrequencyHistogram)(data, probabilities);
}

/// ditto
auto percentogram(Data)(scope auto ref Data data)
{
    import mir.stat.descriptive.histogram.api.factory: defaultPercentogramBinCount;
    return percentogram(data, defaultPercentogramBinCount(data.length));
}

/// Choose the bin count from the sample size and use density as bar height.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    double[8] data = [0, 1, 2, 3, 4, 8, 12, 16];
    // Eight observations request two bins, with probabilities [0, 0.5, 1].
    auto p = percentogram(data);
    assert(p.total == 8 && p.counts == [0, 4, 4, 0]);
    assert(p.relativeFrequency(0) == 0.5);
    // The first bin spans [0, 3.5); height times width equals its probability.
    assert(p.density(0) == 0.5 / 3.5);
    // Construction preserves the observations; later updates keep the same bins.
    assert(data[] == [0.0, 1, 2, 3, 4, 8, 12, 16]);
    p.put(1.0);
    assert(p.total == 9 && p.counts[1] == 5);

    // Override the default with four ordinary bins: probabilities [0, 0.25, 0.5, 0.75, 1].
    auto quartiles = percentogram(data, 4);
    // The first and last counters are underflow and overflow, both zero here.
    assert(quartiles.counts == [0, 2, 2, 2, 2, 0]);
    assert(quartiles.relativeFrequency(0) == 0.25);
    assert(quartiles.density(0) == 0.25 / 1.75);
}

/// Select probability intervals explicitly using Mir slices.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    double[8] observations = [0, 1, 2, 3, 4, 8, 12, 16];
    const double[3] levels = [0, 0.25, 1];
    // These Mir slices borrow the input arrays; the result owns its storage.
    auto p = percentogram(observations[].sliced, levels[].sliced);
    assert(p.total == 8 && p.counts == [0, 2, 6, 0]);
    assert(p.relativeFrequency(0) == 0.25);
    assert(p.relativeFrequency(1) == 0.75);
}

/// Built-in dynamic arrays can be passed directly, without conversion to Mir slices.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    double[4] observations = [0, 1, 2, 3];
    const double[3] levels = [0, 0.5, 1];
    // A dynamic array is a length/pointer view; it need not use GC storage.
    double[] data = observations[];
    const(double)[] probabilities = levels[];
    auto p = percentogram(data, probabilities);
    assert(p.total == 4 && p.counts == [0, 2, 2, 0]);
    // Mutating the original data does not change the stored boundaries or counts.
    data[] = -1;
    assert(p.bins()[0].bin.low == 0 && p.counts == [0, 2, 2, 0]);
}

// Boundaries and counts survive local inputs; tied boundaries are combined.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    static auto fromLocal()
    {
        const double[5] data = [0, 0, 0, 1, 2];
        const double[5] probabilities = [0, 0.25, 0.5, 0.75, 1];
        return percentogram(data, probabilities);
    }
    auto p = fromLocal();
    assert(p.total == 5 && p.counts == [0, 3, 2, 0]);
    double area = 0;
    foreach (i; 0 .. p.axis.N_bin)
    {
        auto bin = p.bins()[i].bin;
        area += p.density(i) * (bin.high - bin.low);
    }
    assert(area > 0.999999 && area < 1.000001);
}

// Boundary precision follows observations; both endpoints are counted.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[6] values = [0, 99, 1, 99, 2, 99];
        const double[3] probabilities = [0, 0.5, 1];
        auto p = percentogram(values[].sliced.stride(2), probabilities);
        assert(p.total == 3 && p.counts == [0, 1, 2, 0]);
        static assert(is(typeof(p.bins()[0].bin.low) == T));
        auto one = percentogram(values[].sliced.stride(2), 1);
        assert(one.total == 3 && one.counts == [0, 3, 0]);
    }}
}

// Invalid inputs are rejected rather than producing degenerate density bins.
version(mir_stat_test)
@system pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramRejections;
    testPercentogramRejections!percentogram();
}

// Combine duplicate quantiles before extending the maximum and constructing the axis.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramDuplicates;
    testPercentogramDuplicates!percentogram();
}

/// Keep excluded tails in underflow/overflow and choose the normalization explicitly.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    double[9] data = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const double[3] probabilities = [0.25, 0.5, 0.75];
    auto p = percentogram(data, probabilities);
    // Cutoffs are 2 and 6, both included. Raw counts also contain the two tails.
    assert(p.counts == [2, 2, 3, 2]);
    assert(p.underflow == 2 && p.overflow == 2 && p.total == 9);
    assert(p.relativeFrequency(0) == 2.0 / 9);
    // Condition on the five retained observations without changing any counts.
    assert(p.relativeFrequency!(double, Normalization.ordinary)(0) == 2.0 / 5);
    assert(p.density!(double, Normalization.ordinary)(0) == 0.2);
    assert(p.cumulativeRelativeFrequency!(double, Normalization.ordinary)(1) == 1);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramIntervals;
    testPercentogramIntervals!percentogram();
}

// Sample-size defaults match explicit probabilities, including tied boundaries.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    const double[8] data = [0, 0, 0, 0, 0, 2, 3, 4];
    const double[3] levels = [0, 0.5, 1];
    auto automatic = percentogram(data[].sliced);
    auto explicit = percentogram(data, levels);
    assert(automatic.counts == explicit.counts);
    assert(automatic.axis.N_bin == 1);
    foreach (i; 0 .. automatic.axis.N_bin)
        assert(automatic.density(i) == explicit.density(i));
    // Use the logical length of a strided view: nine observations request three bins.
    import mir.ndslice.topology: stride;
    double[18] backing;
    foreach (i, ref value; backing)
        value = i;
    auto strided = percentogram(backing[].sliced.stride(2));
    assert(strided.total == 9);
    assert(strided.axis.N_bin == 3);
    assert(strided.counts == [0, 3, 3, 3, 0]);
}

private import mir.stat.descriptive.histogram.api.factory: WeightedHistogramFactory;
private mixin WeightedHistogramFactory!(allocateCounts) weightedImplementation;

/++
Construct a weighted histogram with garbage-collected counts.
Supply observations, weights, and the usual histogram axis arguments.
Built-in arrays and Mir slices are accepted. Their shapes must match; matching
multidimensional slices are traversed elementwise into a one-axis histogram.
Weights must be finite, nonnegative, and implicitly convertible to the counter
type. Axis templates default to `double` counters, independently of the bin-count
argument. An explicit leading counter type overrides this default, including with a supplied
axis instance or concrete axis type. Axes never select counter storage.
Integral counters require integral weights. Counts must accommodate their sums.
Bin-count rules operate on observations, without weighting the rule itself.
Axis ownership and count ownership follow $(LREF histogram).
+/
template weightedHistogram(Options...)
{
    auto weightedHistogram(Data, Weights, Args...)(
        scope auto ref Data data, scope auto ref Weights weights, auto ref Args args)
    {
        NoAllocationContext context;
        return weightedImplementation.weightedFactory!Options(context, data, weights, args);
    }
}

/++
Construct relative frequencies from weighted counts. Accepts the arguments and
counter-type choices of $(LREF weightedHistogram). The total is the sum of
stored weights, including enabled underflow/overflow bins. Normalization and
subsequent weighted insertion use the existing relative-frequency accumulator.
+/
template weightedRelativeFrequencyHistogram(Options...)
{
    auto weightedRelativeFrequencyHistogram(Args...)(auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = weightedHistogram!Options(args);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct weighted counts and relative frequencies from built-in arrays.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto h = weightedHistogram!RegularAxis(observations, weights, 2u, 0.0, 2.0);
    assert(h.counts == [2.0, 2.0]);
    auto f = weightedRelativeFrequencyHistogram!RegularAxis(observations, weights, 2u, 0.0, 2.0);
    assert(f.total == 4.0);
    assert(f.relativeFrequency(0) == 0.5);
}

version(mir_stat_test)
@system pure nothrow
unittest
{
    import core.exception: AssertError;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[1] data = [0.5];
    double[1] weights;
    foreach (weight; [-1.0, double.nan, double.infinity, -double.infinity])
    {
        weights[0] = weight;
        bool rejected;
        try { auto h = weightedHistogram!RegularAxis(data, weights, 2u, 0.0, 2.0); }
        catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    weights[0] = 0;
    auto f = weightedRelativeFrequencyHistogram!RegularAxis(data, weights, 2u, 0.0, 2.0);
    assert(f.total == 0 && f.counts == [0, 0]);
}
