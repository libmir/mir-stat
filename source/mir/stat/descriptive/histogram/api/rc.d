/++
This module contains an API for creating reference-counted histograms.

Bin-count rules are supplied as template aliases, such as functions or function
templates. Runtime callbacks that capture local variables are not supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(REF_ALTTEXT $(TT $2), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.histogram.api.rc;

import mir.ndslice.slice: Slice, SliceKind;
import mir.rc.array: RCI;
import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
import mir.stat.descriptive.histogram.axis: AxisOptions,
    inverseTransformMapping, hasInverseTransformMapping, isTransformFunction,
    TransformAxis, acceptsTransformedBreakFunction;
import mir.stat.descriptive.histogram.traits: isAxis, storageExtent, acceptsBreakFunction;

import mir.ndslice.allocation: mininitRcslice;
import mir.stat.descriptive.histogram.api.factory: HistogramFactory, NoAllocationContext;

private auto allocateRC(T)(ref NoAllocationContext context, size_t length)
{
    return mininitRcslice!T(length);
}

private mixin HistogramFactory!allocateRC implementation;
private import mir.stat.descriptive.histogram.api.factory: AxisHistogramFactory, areHistogramAxes;
private auto allocateCells(T)(ref NoAllocationContext context, size_t length)
{
    import mir.ndslice.allocation: rcslice;
    return rcslice!T(length);
}
private mixin AxisHistogramFactory!allocateCells axisImplementation;

// Retain the existing construction entry point.
auto rchistogramImplBasic(Data, Axis)(Data data, Axis axis)
{
    NoAllocationContext context;
    return implementation.factoryImplBasic(context, data, axis);
}

// Check rchistogramImplBasic
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.rc.array: RCI;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis = IntegralAxis!(double, AxisOptions())(5, 2.0);
    auto x = [2.0, 2.5, 3.0, 3.5].sliced;

    auto h = rchistogramImplBasic(x, integralAxis);
    assert(h.counts == [2, 2, 0, 0, 0]);
    static assert(is(typeof(h.counts) == Slice!(RCI!size_t)));
}

/++
Construct a histogram with reference-counted count storage.
Counts, including enabled underflow/overflow bins, start at zero.
Axis boundary storage retains the ownership supplied by the caller.

Supply an axis instance as `data.rchistogram(axis)`, or select an axis template:
$(UL
$(LI `data.rchistogram!IntegralAxis(n, low)`)
$(LI `data.rchistogram!RegularAxis(n, low, high)`)
$(LI `data.rchistogram!(TransformAxis, transform, inverse)(n, low, high)`)
$(LI `data.rchistogram!VariableAxis(boundaries)`)
$(LI `data.rchistogram!EnumAxis()` or `data.rchistogram!CategoryAxis()`))

Counters default to size_t, independently of the bin-count argument and axis.
A concrete axis type can replace the axis template. Explicit template arguments
select the counter type first and optionally the coordinate type second, for example
`rchistogram!(uint, double, RegularAxis)` or `rchistogram!uint(data, axis)`; axis options follow the axis (and transforms,
when supplied). A supported transform may omit its inverse. A bin-count rule
can replace the explicit bin count; see the examples below. Data may be an
ndslice of any rank; its elements are counted as one-dimensional observations.

Supply only axis instances to allocate an empty one-dimensional or joint
histogram: rchistogram!Cell(axis, ...). Cell defaults to size_t. Numeric cells
start at zero; accumulator structs retain their default initialization.
No observations are inserted. Use putSample or putWeightedSample to accumulate
measurements in nonnumeric cells. Cell storage is reference-counted; borrowed
axis boundaries must still outlive the histogram and its views.
+/
template rchistogram(Options...)
{
    // Borrow lvalue handles without an extra RC copy. Pass arguments directly:
    // core.lifetime.forward can hide borrowed-memory escapes from DIP1000.
    auto rchistogram(Args...)(auto ref Args args)
    {
        NoAllocationContext context;
        static if (areHistogramAxes!Args)
        {
            static assert(Options.length <= 1, "Axis-only construction accepts one cell type");
            return axisImplementation.axisFactory!Options(context, args);
        }
        else static if (Options.length)
            return implementation.factory!Options(context, args);
        else
            return implementation.factory(context, args);
    }
}

/++
Allocate mean-latency bins before requests arrive. Temperature selects a bin;
response time updates its mean. Reference-counted storage owns the cells and
keeps them alive while histogram copies or bin views still reference them.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.univariate: MeanAccumulator;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    alias Cell = MeanAccumulator!(double, Summation.pairwise);
    auto temperature = RegularAxis!(double, AxisOptions())(2, 20.0, 60.0);
    auto timings = rchistogram!Cell(temperature);
    assert(timings.counts[0].count == 0);
    timings.putSample(100.0, 25.0);
    timings.putSample(200.0, 35.0);
    assert(timings.bins.front.value.mean == 150.0);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testAxisOnlyFactory;
    testAxisOnlyFactory!rchistogram();
}

// Cell destructors run only when the last owning histogram/view is released.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static struct Cell
    {
        size_t[] destroyed;
        ~this() @safe pure nothrow @nogc
        {
            if (destroyed.length) ++destroyed[0];
        }
    }
    auto destructionCount = rcslice!size_t(1);
    {
        auto h = rchistogram!Cell(IntegralAxis!(int, AxisOptions())(2, 0));
        foreach (ref cell; h.counts.field)
            cell.destroyed = destructionCount.lightScope.field;
        auto view = h.bins;
        h = typeof(h).init;
        assert(destructionCount[0] == 0);
        // The view retains the cells even after the histogram releases them.
        assert(view.length == 2);
    }
    assert(destructionCount[0] == 2);
}

/// Construct two equal-width bins from observations.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    auto data = [0.0, 1, 2, 3].sliced;
    auto h = data.rchistogram!RegularAxis(2u, 0.0, 4.0);
    assert(h.counts == [2u, 2]);
    static assert(is(typeof(h.counts.iterator) == RCI!size_t));
}

/// Override the counter type and include underflow and overflow bins.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    alias Axis = RegularAxis!(double, AxisOptions(false, true, true));
    auto h = [-1.0, 0, 1, 2, 3, 4].sliced.rchistogram!(ulong, Axis)(2, 0.0, 4.0);
    assert(h.counts == [1UL, 2, 2, 1]);
    assert(h.underflow == 1 && h.overflow == 1);
    static assert(is(h.CountType == ulong));
}

/// Build a relative frequency accumulator sharing the histogram's reference-counted counts.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;

    auto h = [0.0, 1, 2, 3].sliced.rchistogram!RegularAxis(2u, 0.0, 4.0);
    auto f = RelativeFrequencyAccumulator!(typeof(h.counts), typeof(h.axis[0]))(
        h.counts, h.axis[0]);
    assert(f.total == 4);
    // Make subsequent updates through f so its total stays synchronized.
    f.put(1.0);
    assert(f.total == 5);
    assert(h.counts == [3u, 2]); // The count storage is shared.
}

/// Choose logarithmic bins using a rule evaluated in logarithmic coordinates.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.common: log2, exp2;
    import mir.stat.descriptive.histogram.axis: TransformAxis;
    import mir.stat.descriptive.histogram.breaks: freedmanDiaconis;
    auto data = [1.0, 2, 4, 8, 16, 32, 64, 128, 256].sliced;

    // The rule sees [0, 1, ..., 8], choosing three bins in log2 space.
    // Bounds and inserted values are still in the original units.
    auto h = data.rchistogram!(TransformAxis, log2, freedmanDiaconis)(1.0, 512.0);
    assert(h.counts == [3, 3, 3]);
    assert(h.axis[0].bin(0).low == 1 && h.axis[0].bin(0).high == 8);

    // An explicit inverse produces the same histogram.
    auto explicitInverse = data.rchistogram!(TransformAxis, log2, exp2,
        freedmanDiaconis)(1.0, 512.0);
    assert(explicitInverse.counts == h.counts);
}

/// Choose a regular-bin count using Sturges, retaining explicit bounds.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import mir.stat.descriptive.histogram.breaks: sturges;
    static immutable values = [0.0, 1, 4, 5, 6, 9, 10, 13, 14];
    auto data = values[].sliced;
    auto h = data.rchistogram!(RegularAxis, sturges)(0.0, 15.0);
    // Sturges selects five bins, each of width three.
    assert(h.axis[0].N_bin == 5);
    assert(h.counts == [2, 2, 1, 2, 2]);
}

/// Supply a custom rule and override count types and axis options.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    static size_t threePerBin(S)(S data)
    {
        import mir.primitives: elementCount;
        const n = data.elementCount;
        return n / 3 + (n % 3 != 0);
    }
    static immutable values = [0.0, 1, 4, 5, 6, 9, 10, 13, 14];
    auto data = values[].sliced;
    enum options = AxisOptions(false, true, true);
    auto h = data.rchistogram!(ulong, double, RegularAxis, threePerBin, options)(0.0, 15.0);
    // Nine observations give three ordinary bins. End bins are stored too.
    static assert(is(h.CountType == ulong));
    assert(h.counts == [0, 3, 3, 3, 0]);
    assert(h.underflow == 0 && h.overflow == 0);
}

/// Evaluate a rule using runtime settings before constructing the histogram.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    auto data = [1.0, 2, 3, 4, 5, 6, 7, 8, 9].sliced;
    size_t observationsPerBin = 3; // A positive runtime setting.
    auto rule = (typeof(data) values) => values.length / observationsPerBin +
        (values.length % observationsPerBin != 0);

    // Evaluate the capturing rule ourselves, then pass its result as a count.
    const n = rule(data);
    auto h = data.rchistogram!RegularAxis(n, 0.0, 12.0);
    assert(h.axis[0].N_bin == 3);
    // The rule selects the number of equal-width bins, not their occupancy.
    assert(h.counts == [3, 4, 2]);
}

// A locally evaluated capturing rule need not allocate a GC closure. Keep
// observations in static storage to test the factory rather than array setup.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    static immutable double[8] values = [0, 1, 2, 3, 4, 5, 6, 7];
    auto data = values[].sliced;
    size_t observationsPerBin = 2;
    // scope prevents the captured setting from requiring a GC closure.
    scope auto rule = (typeof(data) observations) => observations.length / observationsPerBin;
    auto first = data.rchistogram!RegularAxis(rule(data), 0.0, 8.0);
    assert(first.axis[0].N_bin == 4);
    foreach (i; 0 .. 4)
        assert(first.counts[i] == 2);

    // Changing the captured setting affects the next evaluation, not the
    // histogram already built from the previous result.
    observationsPerBin = 4;
    auto second = data.rchistogram!RegularAxis(rule(data), 0.0, 8.0);
    assert(second.axis[0].N_bin == 2);
    assert(second.counts[0] == 4 && second.counts[1] == 4);
    assert(first.axis[0].N_bin == 4 && first.counts[0] == 2);
}

/// Integral Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, integralAxis;
    import mir.stat.descriptive.histogram.breaks: sturges;

    static immutable a = [0.0, 0.5, 1, 1.5, 2];
    static immutable b = [2, 2, 1];
    static immutable c = [2, 2, 1, 0];

    auto x = rcslice!double(a);
    auto result1 = rcslice!size_t(b);
    auto result2 = rcslice!size_t(c);

    auto h1 = x.rchistogram!IntegralAxis(3u, 0.0);
    assert(h1.counts == result1);
    static assert(is(h1.CountType == size_t));

    // Pass axis directly
    auto iAxis = integralAxis(3u, 0.0);
    auto h2 = x.rchistogram(iAxis);
    assert(h2.counts == result1);

    // Use function to calculate N_bin
    auto iAxis2 = x.integralAxis!sturges(0.0);
    auto h3 = x.rchistogram(iAxis2);
    assert(h3.counts == result2);
}

/// Regular Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: RegularAxis, regularAxis;
    import mir.stat.descriptive.histogram.breaks: sturges;

    static immutable a = [0.0, 1, 4, 5, 6, 9, 10, 13, 14];
    static immutable b = [3, 3, 3];
    static immutable c = [2, 2, 1, 2, 2];

    auto x = rcslice!double(a);
    auto result1 = rcslice!size_t(b);
    auto result2 = rcslice!size_t(c);

    auto h1 = x.rchistogram!RegularAxis(3u, 0.0, 15.0);
    assert(h1.counts == result1);
    static assert(is(h1.CountType == size_t));

    // Pass axis directly
    auto regularAxis2 = regularAxis(3u, 0.0, 15.0);
    auto h2 = x.rchistogram(regularAxis2);
    assert(h2.counts == result1);

    // Use function to calculate N_bin
    auto regularAxis3 = x.regularAxis!sturges(0.0, 15.0);
    auto h3 = rchistogram(x, regularAxis3);
    assert(h3.counts == result2);
}

/// Transform Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: TransformAxis, transformAxis, inverseTransformMapping;
    import mir.stat.descriptive.histogram.breaks: sturges;


    static immutable a = [10.0 ^^ 2.0, 10.0 ^^ 2.5, 10.0 ^^ 5.0, 10.0 ^^ 11.5];
    static immutable b = [2, 1, 0, 1];
    static immutable c = [3, 0, 1];

    auto x = rcslice!double(a);
    auto result1 = rcslice!size_t(b);
    auto result2 = rcslice!size_t(c);

    auto h1 = x.rchistogram!(TransformAxis, log10, inverseTransformMapping!log10)(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(h1.counts == result1);
    static assert(is(h1.CountType == size_t));

    // Pass axis directly
    auto regularAxis2 = transformAxis!(log10, inverseTransformMapping!log10)(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    auto h2 = x.rchistogram(regularAxis2);
    assert(h2.counts == result1);

    // Use function to calculate N_bin
    auto regularAxis3 = x.transformAxis!(log10, inverseTransformMapping!log10, sturges)(10.0 ^^ 2.0, 10.0 ^^ 12.0);
    auto h3 = x.rchistogram(regularAxis3);
    assert(h3.counts == result2);

    // Can also supply lambda
    auto h4 = x.rchistogram!(TransformAxis, a => log10(a), a => (10.0 ^^ a))(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(h4.counts == result1);

    // Or string lambda
    auto h5 = x.rchistogram!(TransformAxis, "log10(a)", "10.0 ^^ a")(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(h5.counts == result1);

    // For some functions, inverseTransform is not needed
    auto h6 = x.rchistogram!(TransformAxis, log10)(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(h6.counts == result1);
    static assert(is(h6.CountType == size_t));

    // Pass axis directly without inverseTransform
    auto regularAxis4 = transformAxis!log10(4u, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    auto h7 = x.rchistogram(regularAxis4);
    assert(h7.counts == result1);

    // Same, but use function to calculate N_bin
    auto regularAxis5 = x.transformAxis!(log10, sturges)(10.0 ^^ 2.0, 10.0 ^^ 12.0);
    auto h8 = x.rchistogram(regularAxis5);
    assert(h8.counts == result2);

}

/// Enum Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: EnumAxis, enumAxis;

    enum Foo {
        A,
        B
    }

    static immutable a = [Foo.A, Foo.B, Foo.A, Foo.A, Foo.B];
    static immutable b = [3, 2];

    auto x = rcslice!Foo(a);
    auto result = rcslice!size_t(b);

    auto h1 = x.rchistogram!EnumAxis;
    assert(h1.counts == result);
    static assert(is(h1.CountType == size_t));

    // Pass axis directly
    auto eAxis = enumAxis!Foo;
    auto h2 = x.rchistogram(eAxis);
    assert(h2.counts == result);
}

/// Category Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: CategoryAxis, categoryAxis;

    enum Foo {
        A,
        B
    }

    static immutable a = [Foo.A, Foo.B, Foo.A, Foo.A, Foo.B];
    static immutable b = ["A", "B", "A", "A", "B"];
    static immutable c = [3, 2];

    auto x = rcslice!Foo(a);
    auto y = rcslice!string(b);
    auto result = rcslice!size_t(c);

    auto h1 = x.rchistogram!CategoryAxis;
    assert(h1.counts == result);
    static assert(is(h1.CountType == size_t));

    // Can handle string inputs
    auto h2 = y.rchistogram!(size_t, Foo, CategoryAxis);
    assert(h2.counts == result);

    // Pass axis directly
    auto cAxis = categoryAxis!Foo();
    auto h3 = x.rchistogram(cAxis);
    assert(h3.counts == result);
}

/// Variable Axis example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: VariableAxis, variableAxis;

    static immutable a = [0.0, 0.5, 1, 1.5, 2];
    static immutable b = [0.0, 1, 2, 3];
    static immutable c = [2, 2, 1];

    auto x = rcslice!double(a);
    auto breaks = rcslice!double(b);
    auto result = rcslice!size_t(c);

    auto h1 = x.rchistogram!VariableAxis(breaks);
    assert(h1.counts == result);
    static assert(is(h1.CountType == size_t));

    // Pass axis directly
    auto vAxis = variableAxis(breaks);
    auto h2 = x.rchistogram(vAxis);
    assert(h2.counts == result);
}

/// Compute quantile boundaries first to construct a percentogram's counts.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: rcquantile;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import std.math: nextUp;

    auto data = [0.0, 1, 2, 3, 4, 8, 12, 16].sliced;
    // Probabilities can be selected at runtime; here each interval spans 25%.
    auto probabilities = [0.0, 0.25, 0.5, 0.75, 1.0].sliced;
    // Rcquantile returns owning boundaries; the histogram retains that ownership.
    auto boundaries = data.rcquantile(probabilities);
    assert(boundaries == [0.0, 1.75, 3.5, 9.0, 16.0]);

    // VariableAxis uses [low, high) bins by default. Extend the last boundary
    // by one representable step so the sample maximum belongs to the last bin.
    boundaries[$ - 1] = nextUp(boundaries[$ - 1]);
    auto h = data.rchistogram!VariableAxis(boundaries);
    assert(h.counts == [2, 2, 2, 2]);

    // These bins have unequal widths. For a percentogram, plot probability
    // divided by width as height, so each bar's AREA represents probability.
    // Ties can produce repeated quantiles: combine those boundaries before
    // constructing VariableAxis, which requires strictly increasing edges.
    // With ties or other sample sizes, equal probabilities need not yield
    // exactly equal observed counts.
}


// The result owns counts independently of the factory's local observations.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    static auto makeOwned() @safe pure nothrow @nogc
    {
        double[4] values = [0, 1, 2, 3];
        return values[].sliced.rchistogram!RegularAxis(2u, 0.0, 4.0);
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
Construct a relative-frequency accumulator with reference-counted count storage.
Accepts the same arguments and axis options as $(LREF rchistogram).
The total is calculated from the stored counts, including enabled underflow
and overflow bins. Out-of-range observations follow the underlying histogram
factory's axis rules. This scans the bins once without allocating another count
buffer. Axis ownership is unchanged.
Counter types must accommodate both each bin and the total.
+/
template rcRelativeFrequencyHistogram(Options...)
{
    auto rcRelativeFrequencyHistogram(Args...)(auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = rchistogram!Options(args);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct relative frequencies directly and keep the total updated.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[4] values = [0, 1, 1, 3];
    auto f = rcRelativeFrequencyHistogram!RegularAxis(values[].sliced, 2u, 0.0, 4.0);
    assert(f.total == 4);
    assert(f.relativeFrequency(0) == 0.75);
    f.put(3.5);
    assert(f.total == 5);
    assert(f.relativeFrequency(1) == 0.4);
}

/++
Construct a percentogram using quantile boundaries and observed relative frequencies.
Returns a relative-frequency accumulator with RC-owned boundaries and counts.
Construction supports `@nogc` for ordinary numeric inputs.
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
auto rcpercentogram(Data, P)(scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.stat.descriptive.univariate: rcquantile;
    import mir.stat.descriptive.histogram.api.factory: buildPercentogram;
    return buildPercentogram!(allocateRC, rcquantile, rcRelativeFrequencyHistogram)(data, probabilities);
}

/// ditto
auto rcpercentogram(Data)(scope auto ref Data data)
{
    import mir.stat.descriptive.histogram.api.factory: defaultPercentogramBinCount;
    return rcpercentogram(data, defaultPercentogramBinCount(data.length));
}

/// Choose the bin count from the sample size and use density as bar height.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double[8] data = [0, 1, 2, 3, 4, 8, 12, 16];
    // Eight observations request two bins, with probabilities [0, 0.5, 1].
    auto p = rcpercentogram(data);
    assert(p.total == 8 && p.counts == [0, 4, 4, 0]);
    assert(p.relativeFrequency(0) == 0.5);
    // The first bin spans [0, 3.5); height times width equals its probability.
    assert(p.density(0) == 0.5 / 3.5);
    // Construction preserves the observations; later updates keep the same bins.
    assert(data[] == [0.0, 1, 2, 3, 4, 8, 12, 16]);
    p.put(1.0);
    assert(p.total == 9 && p.counts[1] == 5);

    // Override the default with four ordinary bins: probabilities [0, 0.25, 0.5, 0.75, 1].
    auto quartiles = rcpercentogram(data, 4);
    // The first and last counters are underflow and overflow, both zero here.
    assert(quartiles.counts == [0, 2, 2, 2, 2, 0]);
    assert(quartiles.relativeFrequency(0) == 0.25);
    assert(quartiles.density(0) == 0.25 / 1.75);
}

/// Select probability intervals explicitly using Mir slices.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    double[8] observations = [0, 1, 2, 3, 4, 8, 12, 16];
    const double[3] levels = [0, 0.25, 1];
    // These Mir slices borrow the input arrays; the result owns its storage.
    auto p = rcpercentogram(observations[].sliced, levels[].sliced);
    assert(p.total == 8 && p.counts == [0, 2, 6, 0]);
    assert(p.relativeFrequency(0) == 0.25);
    assert(p.relativeFrequency(1) == 0.75);
}

/// Built-in dynamic arrays can be passed directly, without conversion to Mir slices.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double[4] observations = [0, 1, 2, 3];
    const double[3] levels = [0, 0.5, 1];
    // A dynamic array is a length/pointer view; it need not use GC storage.
    double[] data = observations[];
    const(double)[] probabilities = levels[];
    auto p = rcpercentogram(data, probabilities);
    assert(p.total == 4 && p.counts == [0, 2, 2, 0]);
    // Mutating the original data does not change the stored boundaries or counts.
    data[] = -1;
    assert(p.bins()[0].bin.low == 0 && p.counts == [0, 2, 2, 0]);
}

// Boundaries and counts survive local inputs; tied boundaries are combined.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static auto fromLocal()
    {
        const double[5] data = [0, 0, 0, 1, 2];
        const double[5] probabilities = [0, 0.25, 0.5, 0.75, 1];
        return rcpercentogram(data, probabilities);
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
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[6] values = [0, 99, 1, 99, 2, 99];
        const double[3] probabilities = [0, 0.5, 1];
        auto p = rcpercentogram(values[].sliced.stride(2), probabilities);
        assert(p.total == 3 && p.counts == [0, 1, 2, 0]);
        static assert(is(typeof(p.bins()[0].bin.low) == T));
        auto one = rcpercentogram(values[].sliced.stride(2), 1);
        assert(one.total == 3 && one.counts == [0, 3, 0]);
    }}
}

// Invalid inputs are rejected rather than producing degenerate density bins.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramRejections;
    testPercentogramRejections!rcpercentogram();
}

// Combine duplicate quantiles before extending the maximum and constructing the axis.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramDuplicates;
    testPercentogramDuplicates!rcpercentogram();
}

/// Keep excluded tails in underflow/overflow and choose the normalization explicitly.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    double[9] data = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const double[3] probabilities = [0.25, 0.5, 0.75];
    auto p = rcpercentogram(data, probabilities);
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
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testPercentogramIntervals;
    testPercentogramIntervals!rcpercentogram();
}

// Sample-size defaults match explicit probabilities, including tied boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    const double[8] data = [0, 0, 0, 0, 0, 2, 3, 4];
    const double[3] levels = [0, 0.5, 1];
    auto automatic = rcpercentogram(data[].sliced);
    auto explicit = rcpercentogram(data, levels);
    assert(automatic.counts == explicit.counts);
    assert(automatic.axis.N_bin == 1);
    foreach (i; 0 .. automatic.axis.N_bin)
        assert(automatic.density(i) == explicit.density(i));
    // Use the logical length of a strided view: nine observations request three bins.
    import mir.ndslice.topology: stride;
    double[18] backing;
    foreach (i, ref value; backing)
        value = i;
    auto strided = rcpercentogram(backing[].sliced.stride(2));
    assert(strided.total == 9);
    assert(strided.axis.N_bin == 3);
    assert(strided.counts == [0, 3, 3, 3, 0]);
}

private import mir.stat.descriptive.histogram.api.factory: WeightedHistogramFactory;
private mixin WeightedHistogramFactory!(allocateRC) weightedImplementation;

/++
Construct a weighted histogram with reference-counted counts.
Supply observations, weights, and the usual histogram axis arguments.
Built-in arrays and Mir slices are accepted. Their shapes must match; matching
multidimensional slices are traversed elementwise into a one-axis histogram.
Weights must be finite, nonnegative, and implicitly convertible to the counter
type. Axis templates default to `double` counters, independently of the bin-count
argument. An explicit leading counter type overrides this default, including with a supplied
axis instance or concrete axis type. Axes never select counter storage.
Integral counters require integral weights. Counts must accommodate their sums.
Bin-count rules operate on observations, without weighting the rule itself.
Axis ownership and count ownership follow $(LREF rchistogram).
+/
template rcWeightedHistogram(Options...)
{
    auto rcWeightedHistogram(Data, Weights, Args...)(
        scope auto ref Data data, scope auto ref Weights weights, auto ref Args args)
    {
        NoAllocationContext context;
        return weightedImplementation.weightedFactory!Options(context, data, weights, args);
    }
}

/++
Construct relative frequencies from weighted counts. Accepts the arguments and
counter-type choices of $(LREF rcWeightedHistogram). The total is the sum of
stored weights, including enabled underflow/overflow bins. Normalization and
subsequent weighted insertion use the existing relative-frequency accumulator.
+/
template rcWeightedRelativeFrequencyHistogram(Options...)
{
    auto rcWeightedRelativeFrequencyHistogram(Args...)(auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = rcWeightedHistogram!Options(args);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Integral weights still default to double counters, allowing fractional updates later.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    double[3] observations = [0.25, 0.75, 1.25];
    uint[3] weights = [1, 3, 2];
    auto h = rcWeightedHistogram!RegularAxis(observations, weights, 2u, 0.0, 2.0);
    static assert(is(h.CountType == double)); // 2u selects the number of bins only.
    assert(h.counts == [4.0, 2.0]);
    h.putWeighted(0.5, 1.25);
    assert(h.counts == [4.0, 2.5]);
}

/// Override the counter type when building an axis from a template.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto h = rcWeightedHistogram!(real, RegularAxis)(
        observations[].sliced, weights[].sliced, 2u, 0.0, 2.0);
    static assert(is(h.CountType == real));
    assert(h.counts == [2.0L, 2.0L]);
}

/// Select integral counters independently of a supplied axis for integral weights.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    double[3] observations = [0.25, 0.75, 1.25];
    uint[3] weights = [1, 3, 2];
    auto axis = RegularAxis!(double, AxisOptions())(2, 0.0, 2.0);
    auto h = rcWeightedHistogram!uint(observations[], weights[], axis);
    static assert(is(h.CountType == uint));
    assert(h.counts == [4u, 2]);
    // Fractional weights require floating-point counters, as in the first example.
    static assert(!__traits(compiles, h.putWeighted(0.5, 0.25)));
}

/// Relative frequencies divide bin weights by their total, not by the number of observations.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto f = rcWeightedRelativeFrequencyHistogram!RegularAxis(
        observations, weights, 2u, 0.0, 2.0);
    assert(f.total == 4.0);
    assert(f.relativeFrequency(0) == 0.5);
    assert(f.relativeFrequency(1) == 0.5);
}

// Weighted construction preserves logical pairing, qualifiers, and axis options.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: RegularAxis, VariableAxis, TransformAxis, AxisOptions;
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    double[4] values = [-1, 0.5, 1.5, 2];
    double[4] weights = [1, 2, 3, 4];
    const data = values[].sliced(2, 2).transposed;
    const masses = weights[].sliced(2, 2).transposed;
    enum options = AxisOptions(false, true, true);
    auto f = rcWeightedRelativeFrequencyHistogram!(RegularAxis, options)(data, masses, 2u, 0.0, 2.0);
    assert(f.counts == [1, 2, 3, 4]);
    assert(f.total == 10);
    assert(f.relativeFrequency(0) == 0.2);
    assert(f.relativeFrequency!(double, Normalization.ordinary)(0) == 0.4);
    double[2] cumulative;
    f.cumulativeRelativeFrequencies(cumulative[]);
    assert(cumulative == [0.3, 0.6]);
    auto column = rcWeightedHistogram!(RegularAxis, options)(data[0], masses[0], 2u, 0.0, 2.0);
    assert(column.counts == [1, 0, 3, 0]);

    const(double)[3] edges = [0, 1, 2];
    auto variable = rcWeightedHistogram!VariableAxis(values[1 .. 3], weights[1 .. 3], edges[].sliced);
    assert(variable.counts == [2, 3]);
    static assert(is(variable.CountType == double));

    double[2] powers = [1, 4];
    auto transformed = rcWeightedHistogram!(TransformAxis, "log2(a)", "exp2(a)")(
        powers, weights[1 .. 3], 2u, 1.0, 16.0);
    assert(transformed.counts == [2, 3]);

    double[0] empty;
    auto zero = rcWeightedRelativeFrequencyHistogram!RegularAxis(empty, empty, 2u, 0.0, 2.0);
    assert(zero.counts == [0, 0] && zero.total == 0);
    import std.math: isNaN;
    assert(isNaN(zero.relativeFrequency(0)));
    const(uint)[2] integralWeights = [1, 2];
    auto integral = rcWeightedHistogram!(uint, RegularAxis)(powers, integralWeights, 2u, 0.0, 8.0);
    static assert(is(integral.CountType == uint));
    assert(integral.counts == [1, 2]);
}

// Temporary inputs may disappear; the result owns its counts, including under DIP1000.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    static auto build() @safe pure nothrow @nogc
    {
        const(double)[2] data = [0.5, 1.5];
        const(double)[2] weights = [0.5, 1.5];
        return rcWeightedRelativeFrequencyHistogram!RegularAxis(data, weights, 2u, 0.0, 2.0);
    }
    auto f = build();
    assert(f.counts == [0.5, 1.5] && f.total == 2);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis, IntegralAxis, CategoryAxis, EnumAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    double[2] data = [0.5, 1.5];
    float[2] weights = [0.5f, 1.5f];
    alias Axis = RegularAxis!(double, AxisOptions());
    auto concrete = rcWeightedHistogram!(float, Axis)(data, weights, 2u, 0.0, 2.0);
    auto instance = rcWeightedHistogram!float(data, weights, Axis(2, 0, 2));
    static assert(is(concrete.CountType == float));
    assert(concrete.counts == instance.counts && instance.counts == [0.5f, 1.5f]);
    double[2] wideWeights = [0.5, 1.5];
    static assert(!__traits(compiles, rcWeightedHistogram!uint(data, wideWeights, RegularAxis!(double, AxisOptions())(2, 0, 2))));
    static assert(!__traits(compiles, rcWeightedHistogram!RegularAxis(
        data[].sliced(1, 2), weights, 2u, 0.0, 2.0)));
    auto integral = rcWeightedHistogram!IntegralAxis(data, weights, 2u, 0.0);
    assert(integral.counts == [0.5, 1.5]);
    static uint two(S)(S samples) @safe pure nothrow @nogc { return 2; }
    auto rule = rcWeightedHistogram!(RegularAxis, two)(data, weights, 0.0, 2.0);
    assert(rule.counts == [0.5, 1.5]);
    enum Label { first, second }
    Label[2] labels = [Label.first, Label.second];
    auto enumerated = rcWeightedHistogram!EnumAxis(labels, weights);
    auto categorized = rcWeightedHistogram!CategoryAxis(labels, weights);
    assert(enumerated.counts == [0.5, 1.5]);
    assert(categorized.counts == [0.5, 1.5]);
}

version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    static assert(!__traits(compiles, () @safe {
        double[3] edges = [0, 1, 2];
        double[1] data = [0.5];
        double[1] weights = [1];
        return rcWeightedHistogram!VariableAxis(data, weights, edges[].sliced);
    }));
    static assert(!__traits(compiles, () @safe {
        double[3] edges = [0, 1, 2];
        double[1] data = [0.5];
        double[1] weights = [1];
        return rcWeightedRelativeFrequencyHistogram!VariableAxis(data, weights, edges[].sliced);
    }));
}

// Floating-point counters must work with every view and both snapshot APIs.
version(mir_stat_test)
private void testWeightedFactoryViews()()
{
    import std.meta: AliasSeq;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import mir.stat.descriptive.histogram.accumulator: BinCoverage;
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        double[4] data = [-1, 0.5, 1.5, 2];
        T[4] weights = [1, 2, 3, 4];
        enum options = AxisOptions(false, true, true);
        static if (is(T == double))
            auto h = rcWeightedHistogram!(RegularAxis, options)(data, weights, 2u, 0.0, 2.0);
        else
            auto h = rcWeightedHistogram!(T, RegularAxis, options)(data, weights, 2u, 0.0, 2.0);
        auto hb = h.bins();
        assert(hb.length == 2 && hb.front.count == 2 && hb.back.count == 3);
        auto all = h.bins!(BinCoverage.all)();
        assert(all.length == 4 && all.front.count == 1 && all.back.count == 4);
        static if (is(T == double))
            auto f = rcWeightedRelativeFrequencyHistogram!(RegularAxis, options)(data, weights, 2u, 0.0, 2.0);
        else
            auto f = rcWeightedRelativeFrequencyHistogram!(T, RegularAxis, options)(data, weights, 2u, 0.0, 2.0);
        auto counts = f.bins();
        auto frequencies = f.relativeFrequencyBins();
        auto densities = f.densityBins();
        auto cumulative = f.cumulativeRelativeFrequencyBins();
        assert(counts.front.count == 2 && frequencies.length == 2);
        assert(frequencies.front.relativeFrequency == 0.2);
        assert(densities.front.density == 0.2);
        assert(cumulative.front.cumulativeRelativeFrequency == 0.3);
        cumulative.popFront();
        assert(cumulative.front.cumulativeRelativeFrequency == 0.6);
        auto snapshot = f.cumulativeRelativeFrequencies();
        assert(snapshot == [0.3, 0.6]);
        double[2] output;
        f.cumulativeRelativeFrequencies!(Normalization.ordinary)(output[]);
        assert(output == [0.4, 1.0]);
    }}
}

version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    testWeightedFactoryViews();
}
else version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    testWeightedFactoryViews();
}

/// Reuse one axis with different counter storage; bin counts and indices stay integral.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    double[3] data = [0.25, 0.75, 1.25];
    float[3] weights = [0.5f, 1.5f, 2.0f];
    auto axis = RegularAxis!(double, AxisOptions())(2, 0.0, 2.0);
    auto counts = rchistogram!uint(data[].sliced, axis);
    auto weighted = rcWeightedHistogram!float(data, weights, axis);
    auto relative = rcWeightedRelativeFrequencyHistogram(data, weights, axis);
    static assert(is(typeof(axis.N_bin()) == size_t));
    static assert(is(typeof(axis.index(0.5)) == size_t));
    static assert(is(counts.CountType == uint));
    static assert(is(weighted.CountType == float));
    static assert(is(relative.CountType == double));
    assert(counts.counts == [2u, 1]);
    assert(weighted.counts == [2.0f, 2.0f]);
    assert(relative.total == 4 && relative.relativeFrequency(0) == 0.5);
}

// Weighted factories share mixed-bound inference without changing counter defaults.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis, TransformAxis;
    double[3] values = [0.25, 1.25, 2.25];
    uint[3] weights = [1, 2, 3];
    double low = 0;
    float high = 4;
    auto h = rcWeightedHistogram!RegularAxis(values, weights, 2u, low, high);
    auto f = rcWeightedRelativeFrequencyHistogram!(TransformAxis, "a", "a")(
        values, weights, 2u, float(0), double(4));
    static assert(is(h.axis[0].BinType == double));
    static assert(is(f.CountType == double));
    static assert(is(h.CountType == double));
    assert(h.counts == [3.0, 3.0]);
    assert(f.counts == h.counts && f.total == 6);
}

// Custom axes supply geometry and integral indices, without counter metadata.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    struct TwoBins
    {
        alias BinType = double;
        enum uint N_bin = 2;
        uint index(double value) const @safe pure nothrow @nogc
        {
            assert(value >= 0 && value < 2);
            return value < 1 ? 0u : 1u;
        }
    }
    double[2] data = [0.5, 1.5];
    uint[2] weights = [2, 3];
    auto h = rchistogram!ubyte(data[].sliced, TwoBins());
    auto f = rcWeightedRelativeFrequencyHistogram!float(data, weights, TwoBins());
    static assert(is(h.CountType == ubyte));
    static assert(is(f.CountType == float));
    assert(h.counts == [1, 1]);
    assert(f.total == 5 && f.counts == [2, 3]);
}
