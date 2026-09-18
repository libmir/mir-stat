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

    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions())(5, 2.0);
    auto x = [2.0, 2.5, 3.0, 3.5].sliced;

    auto h = rchistogramImplBasic(x, integralAxis);
    assert(h.counts == [2, 2, 0, 0, 0]);
    static assert(is(typeof(h.counts) == Slice!(RCI!(integralAxis.CountType))));
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

A concrete axis type can replace the axis template. Explicit template arguments
can override count and bin types; axis options follow the axis (and transforms,
when supplied). A supported transform may omit its inverse. A bin-count rule
can replace the explicit bin count; see the examples below. Data may be an
ndslice of any rank; its elements are counted as one-dimensional observations.
+/
template rchistogram(Options...)
{
    // Borrow lvalue handles without an extra RC copy. Pass arguments directly:
    // core.lifetime.forward can hide borrowed-memory escapes from DIP1000.
    auto rchistogram(Args...)(auto ref Args args)
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
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.axis: RegularAxis;

    auto data = [0.0, 1, 2, 3].sliced;
    auto h = data.rchistogram!RegularAxis(2u, 0.0, 4.0);
    assert(h.counts == [2u, 2]);
    static assert(is(typeof(h.counts.iterator) == RCI!uint));
}

/// Override the counter type and include underflow and overflow bins.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    alias Axis = RegularAxis!(ulong, double, AxisOptions(false, true, true));
    auto h = [-1.0, 0, 1, 2, 3, 4].sliced.rchistogram!Axis(2, 0.0, 4.0);
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
    assert(f.count == 4);
    // Make subsequent updates through f so its total stays synchronized.
    f.put(1.0);
    assert(f.count == 5);
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
    static assert(is(h1.CountType == uint));

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
    static assert(is(h1.CountType == uint));

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
    static assert(is(h1.CountType == uint));

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
    static assert(is(h6.CountType == uint));

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
    auto h2 = y.rchistogram!(Foo, CategoryAxis);
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
    import mir.stat.descriptive.univariate: quantile;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import std.math: nextUp;

    auto data = [0.0, 1, 2, 3, 4, 8, 12, 16].sliced;
    // Probabilities can be selected at runtime; here each interval spans 25%.
    auto probabilities = [0.0, 0.25, 0.5, 0.75, 1.0].sliced;
    // Quantile returns owning boundaries; the histogram retains that ownership.
    auto boundaries = data.quantile(probabilities);
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
    assert(f.count == 4);
    assert(f.relativeFrequency(0) == 0.75);
    f.put(3.5);
    assert(f.count == 5);
    assert(f.relativeFrequency(1) == 0.4);
}
