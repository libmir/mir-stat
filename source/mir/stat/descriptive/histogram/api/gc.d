/++
Factories for histograms with garbage-collected count storage.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.api.gc;

private import mir.stat.descriptive.univariate: WeightedQuantileAlgo, QuantileAlgo;
private import mir.stat.descriptive.histogram.axis: isQuantileAxis;

// Joint numeric batches preserve pairing, numeric types, and axis lifetimes.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testJointNumericFactories;
    testJointNumericFactories!(histogram, weightedHistogram)();
    testJointNumericFactories!(relativeFrequencyHistogram, weightedRelativeFrequencyHistogram)();
}

// Arrays preserve counting, counter selection, and observation/axis lifetimes.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testArrayHistogramFactories;
    testArrayHistogramFactories!(histogram, relativeFrequencyHistogram)();
}

private import mir.stat.descriptive.histogram.api.factory: HistogramBatchFactory, HistogramBatchKind, isSampleCellSelection, isAdaptiveCountSelection;
private mixin HistogramBatchFactory!(allocateCounts) batchImplementation;

// Exercise shared batch insertion checks with GC-owned cells.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testSampleFactories;
    testSampleFactories!(histogram, weightedHistogram)();
}

import mir.stat.descriptive.histogram.api.factory: HistogramFactory, NoAllocationContext;

private auto allocateCounts(T)(ref NoAllocationContext context, size_t length) @safe pure nothrow
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    static if (is(T == AdaptiveCounts!Initial, Initial))
    {
        import mir.stat.descriptive.histogram.internal.shared_counts: gcSharedCountSlice;
        return gcSharedCountSlice!Initial(length);
    }
    else
        return (new T[length]).sliced;
}

private mixin HistogramFactory!allocateCounts implementation;
private import mir.stat.descriptive.histogram.api.factory: AxisHistogramFactory, areHistogramAxes;
private mixin AxisHistogramFactory!allocateCounts axisImplementation;

/++
Construct a histogram with garbage-collected count storage.
For equal-width bins, use data.histogram!RegularAxis(n, low, high), where data
is a built-in array or Mir slice. Each observation increments its selected bin.
Counters default
to size_t. Supply an existing axis with data.histogram(axis).
Accepts the same axes, bin-count rules, type overrides, and options as
$(REF rchistogram, mir, stat, descriptive, histogram, api, rc).
Use AdaptiveCounts!() in place of an explicit counter type for unweighted counters
that widen together from the selected unsigned type to ulong.
Use AdaptiveCounts!ushort to start wider when larger counts are expected. Read numeric values through bins or
counts[index].count(); counts are not assignable numeric cells. Promotion
preserves saved proxies and views. Both state and buffers use GC allocation;
insertion may allocate and is not @nogc.

Counts start at zero before insertion, including enabled underflow/overflow bins.
All elements of a multidimensional observation slice contribute to the same
one-axis histogram; they are not interpreted as joint coordinates.

To populate numeric joint counts, use histogram(x, y, xAxis, yAxis),
with one coordinate collection per explicit axis. Arrays and Mir slices must
have matching ranks and shapes; strided slices are paired by logical position.
An optional leading numeric template argument selects the counter type.

Supply only axis instances to allocate an empty one-dimensional or joint
histogram: histogram!Cell(axis, ...). Cell defaults to size_t. Numeric cells
start at zero; accumulator structs retain their default initialization.
No observations are inserted. Use putSample or putWeightedSample to accumulate
measurements in nonnumeric cells.

To summarize recorded measurements, supply an accumulator Cell type followed by
samples, one coordinate collection per axis, then explicit axis instances:
histogram!Cell(samples, coordinates, axis). For example, use Summator to total
purchase amounts by customer age, or MeanAccumulator to average request latency
by temperature. Joint histograms accept additional coordinate collections and axes.
Built-in arrays and Mir slices are accepted. All input shapes must match;
multidimensional slices are paired elementwise, including strided views.
Empty inputs leave cells in their default state. Cells receive put(sample),
which determines sample validity and inferred attributes. Samples are passed
by reference where supported; cells retaining sample references require those
samples to outlive the result.

Count allocation does not change axis boundary ownership: borrowed variable-axis
boundaries must still outlive the histogram. Construction allocates GC memory;
subsequent counting with fixed-width counters can be `@nogc`.
+/
template histogram(Options...)
{
    // Borrow lvalue handles without an extra RC copy. Pass arguments directly:
    // core.lifetime.forward can hide borrowed-memory escapes from DIP1000.
    auto histogram(Args...)(auto ref Args args)
    {
        NoAllocationContext context;
        static if (areHistogramAxes!Args)
        {
            static assert(Options.length <= 1, "Axis-only construction accepts one cell type");
            return axisImplementation.axisFactory!Options(context, args);
        }
        else static if (isSampleCellSelection!Options)
            return batchImplementation.batchFactory!(Options[0], HistogramBatchKind.samples)(context, args);
        else static if (Options.length)
            return implementation.factory!Options(context, args);
        else
            return implementation.factory(context, args);
    }
}

/// Construct two equal-width bins, then count additional observations.
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
    h.put(0.5);
    assert(h.counts == [3u, 2]);
    h.put(3.5);
    assert(h.counts == [3u, 3]);
}

/++
Pass built-in static or dynamic arrays directly, including const observations.
Construction reads the observations and owns fresh counts; no conversion to a
Mir slice is needed at the call site.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    const double[4] values = [0, 1, 1, 3];
    auto fromStatic = values.histogram!RegularAxis(2u, 0.0, 4.0);
    auto fromDynamic = values[].histogram!RegularAxis(2u, 0.0, 4.0);
    assert(fromStatic.counts == [3, 1]);
    assert(fromDynamic.counts == fromStatic.counts);
}

/++
Pair coordinate collections elementwise to populate a joint histogram. Each
pair selects one bin; the first two pairs below both select (0, 1).
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    int[3] x = [0, 0, 1], y = [1, 1, 0];
    auto h = histogram(x, y, A(2, 0), A(2, 0));
    assert(h.counts[0, 1] == 2);
    assert(h.counts[1, 0] == 1);
    assert(h.counts[0, 0] == 0);
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


/// Compute quantile boundaries separately to count observations in equal-probability intervals.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: quantile;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import std.math: nextUp;

    double[8] values = [0, 1, 2, 3, 4, 8, 12, 16];
    double[5] levels = [0, 0.25, 0.5, 0.75, 1];
    // Replace this calculation with your preferred quantiles or supplied edges.
    auto boundaries = values[].sliced.quantile(levels[].sliced);
    assert(boundaries == [0.0, 1.75, 3.5, 9.0, 16.0]);
    // Combine duplicate quantiles first if the data contain ties.
    // Include the maximum in the final left-closed, right-open bin.
    boundaries[$ - 1] = nextUp(boundaries[$ - 1]);
    auto h = values[].sliced.histogram!VariableAxis(boundaries);
    assert(h.counts == [2, 2, 2, 2]);
    h.put(1.0);
    assert(h.counts == [3, 2, 2, 2]);
}

/++
Adaptive counts avoid allocating wide counters for every bin when most bins
receive few observations. Select AdaptiveCounts!() to start small and widen when
needed; saved proxies and bin views continue to refer to the same counts.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    alias A = IntegralAxis!(int, AxisOptions());
    int[255] values;
    auto h = values[].sliced.histogram!(AdaptiveCounts!())(A(2, 0));
    auto saved = h.counts[0];
    auto bins = h.bins;
    assert(saved.count() == 255);
    h.put(0); // Widen the count buffer from ubyte to ushort.
    assert(saved.count() == 256 && bins[0].count == 256);
    assert(bins[1].count == 0);
}

/++
Start with ushort when bins are expected to receive hundreds of observations.
This avoids the first promotion and copy, while still allowing counts to grow
beyond ushort.max. The initial type changes storage, not the ulong read type.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    import mir.ndslice.slice: sliced;
    double[300] values;
    values[] = 0.5;
    auto h = values[].sliced.histogram!(AdaptiveCounts!ushort, RegularAxis)(2u, 0.0, 2.0);
    static assert(is(h.CountType == ulong));
    assert(h.bins[0].count == 300);
    h.put(1.5);
    assert(h.bins[1].count == 1);
}

// Shared factory coverage also checks the ownership-specific attributes.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testAdaptiveFactory;
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    testAdaptiveFactory!histogram();
    int[1] data;
    static assert(!__traits(compiles, weightedHistogram!(AdaptiveCounts!())(data, data, A(2, 0))));
}

// Exercise shared empty-axis construction checks with GC-owned cells.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testAxisOnlyFactory;
    testAxisOnlyFactory!histogram();
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

/++
Track sales revenue by customer age as purchases arrive. GC storage initializes
each Summator before insertion, and each purchase updates its age group's total.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summator, Summation;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    alias Cell = Summator!(double, Summation.pairwise);
    auto ages = RegularAxis!(double, AxisOptions())(2, 20.0, 60.0);
    auto sales = histogram!Cell(ages);
    sales.putSample(30.0, 25.0);
    sales.putSample(50.0, 35.0);
    assert(sales.bins.front.value.sum == 80.0);
    assert(sales.bins.back.value.sum == 0.0);
}

/++
Populate accumulator cells from existing arrays with
`histogram!Cell(samples, coordinates, axis)`. Corresponding elements form one
observation: the coordinate selects the bin, and the sample updates its cell.

For example, total purchase amounts by customer age. Ages select the intervals
[20, 40) and [40, 60), while a Summator in each bin adds the purchase amounts
using pairwise summation. The first two purchases contribute 30 + 50 to the
first bin; the remaining purchase contributes 120 to the second.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summator, Summation;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    alias Cell = Summator!(double, Summation.pairwise);
    double[3] purchases = [30, 50, 120];
    double[] ages = [25, 35, 45];
    auto sales = histogram!Cell(purchases, ages,
        RegularAxis!(double, AxisOptions())(2, 20, 60));
    assert(sales.bins[0].value.sum == 80);
    assert(sales.bins[1].value.sum == 120);
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
Accepts the numeric-count forms and axis options of $(LREF histogram).
Pass observations as a built-in array or Mir slice to populate a one-axis
histogram. Supply one coordinate collection per explicit axis to populate joint
counts, or only axis instances to allocate empty counts. The argument order
and shape requirements follow the underlying histogram factory.
Accumulator-valued cells, such as MeanAccumulator, are not supported: relative
frequencies require numeric counts that can be summed and normalized.
The total is calculated from the stored counts, including enabled underflow
and overflow bins. Out-of-range observations follow the underlying histogram
factory's axis rules. This scans the bins once without allocating another count
buffer. Axis ownership is unchanged.
Counter types must accommodate both each bin and the total.
Use relativeFrequency!(double, Normalization.ordinary) to exclude underflow and
overflow counts from the denominator; total continues to include those counts.
The result provides relativeFrequency and relativeFrequencyBins, plus cumulative
relative-frequency accessors for one-dimensional histograms. Numeric axes with
supported bin geometry also provide density and densityBins. Updates through
put and putWeighted keep the total synchronized.
A zero normalization total produces NaN relative frequencies.

Select AdaptiveCounts!() (or AdaptiveCounts!ushort, !uint, or !ulong) in place
of a fixed counter type for automatically widening unweighted counts. Counts use
GC ownership; the total remains ulong and must fit in that type. Weighted
insertion and merging are unavailable for adaptive counts.
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
    auto f = values[].sliced.relativeFrequencyHistogram!RegularAxis(2u, 0.0, 4.0);
    assert(f.total == 4);
    assert(f.relativeFrequency(0) == 0.75);
    f.put(3.5);
    assert(f.total == 5);
    assert(f.relativeFrequency(1) == 0.4);
}


/// Use your own quantile boundaries for percentogram densities.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: quantile;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import std.math: nextUp;

    double[8] values = [0, 1, 2, 3, 4, 8, 12, 16];
    double[5] levels = [0, 0.25, 0.5, 0.75, 1];
    // Replace this calculation with your preferred quantiles or supplied edges.
    auto boundaries = values[].sliced.quantile(levels[].sliced);
    assert(boundaries == [0.0, 1.75, 3.5, 9.0, 16.0]);
    // Combine duplicate quantiles first if the data contain ties.
    // Include the maximum in the final left-closed, right-open bin.
    boundaries[$ - 1] = nextUp(boundaries[$ - 1]);
    auto f = values[].sliced.relativeFrequencyHistogram!VariableAxis(boundaries);
    assert(f.counts == [2, 2, 2, 2]);
    assert(f.total == 8);
    assert(f.relativeFrequency(0) == 0.25);
    assert(f.density(0) == 0.25 / 1.75);
    // Density is the bar height: width times height is probability.
    f.put(1.0);
    assert(f.total == 9 && f.counts[0] == 3);
    // Updates change counts and normalization, but retain the original edges.
}

/// Use adaptive counts when collecting an unknown number of observations.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[4] values = [0, 1, 2, 3];
    auto f = values[].sliced.relativeFrequencyHistogram!(AdaptiveCounts!(), RegularAxis)(2u, 0.0, 4.0);
    foreach (i; 0 .. 254) f.put(1.0);
    assert(f.bins()[0].count == 256 && f.total == 258);
    assert(f.relativeFrequency(0) == 256.0 / 258);
}

/++
Supply two axis instances to start with empty joint counts, then insert coordinate
pairs. Each insertion updates one bin and the total used for relative frequencies.
An explicit double counter type also permits later fractional-weight updates.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    auto f = relativeFrequencyHistogram!double(A(2, 0), A(2, 0));
    assert(f.total == 0);
    f.put(0, 1);
    f.put(1, 0);
    assert(f.total == 2);
    assert(f.relativeFrequency(0, 1) == 0.5);
    f.putWeighted(0.5, 0, 1);
    assert(f.total == 2.5);
    assert(f.relativeFrequency(0, 1) == 0.6);
}

private import mir.stat.descriptive.histogram.api.factory: WeightedHistogramFactory;
private mixin WeightedHistogramFactory!(allocateCounts) weightedImplementation;

/++
Construct a weighted histogram with garbage-collected counts.
Supply weights, observations, and the usual histogram axis arguments.
Built-in arrays and Mir slices are accepted. Their shapes must match; matching
multidimensional slices are traversed elementwise into a one-axis histogram.
For joint counts, use weightedHistogram(weights, x, y, xAxis, yAxis).
Supply one coordinate collection per explicit axis. All coordinate collections
and weights must have matching ranks and shapes. The default counter type
remains double; an explicit leading numeric type overrides it.
Weights must be finite, nonnegative, and implicitly convertible to the counter
type. Axis templates default to `double` counters, independently of the bin-count
argument. An explicit leading counter type overrides this default, including with a supplied
axis instance or concrete axis type. Axes never select counter storage.
Integral counters require integral weights. Counts must accommodate their sums.
Bin-count rules operate on observations, without weighting the rule itself.
Axis ownership and count ownership follow $(LREF histogram).

An explicit accumulator Cell type summarizes weighted samples:
weightedHistogram!Cell(weights, samples, coordinates, axis). Supply one coordinate
collection per axis, followed by explicit axis instances. Weights precede samples,
matching putWeightedSample and the numeric weighted factories. For example,
WMeanAccumulator computes weighted mean measurements by location. Input shapes
must match as described for $(LREF histogram). Insertion calls
putWeightedSample(weight, sample, coordinates...), so cells receive put(sample, weight).
The cell determines weight validity, attributes, and reference lifetimes;
no separate numeric count or total is maintained.
+/
template weightedHistogram(Options...)
{
    static assert(!isAdaptiveCountSelection!Options,
        "AdaptiveCounts supports only unweighted histogram construction");
    auto weightedHistogram(Weights, Data, Args...)(auto ref Weights weights,
        auto ref Data data, auto ref Args args)
        if (isSampleCellSelection!Options)
    {
        NoAllocationContext context;
        return batchImplementation.batchFactory!(Options[0], HistogramBatchKind.weightedSamples)(context, weights, data, args);
    }

    auto weightedHistogram(Weights, Data, Args...)(
        scope auto ref Weights weights, scope auto ref Data data, auto ref Args args)
        if (!isSampleCellSelection!Options)
    {
        NoAllocationContext context;
        return weightedImplementation.weightedFactory!Options(context, weights, data, args);
    }
}

/// Total observation weights in two equal-width bins.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto h = weightedHistogram!RegularAxis(
        weights, observations, 2u, 0.0, 2.0);
    assert(h.counts == [2.0, 2.0]);
}

/++
For weighted joint counts, pass weights first, then all coordinate collections
and axis instances. Each bin stores
the sum of weights for its coordinate pairs.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    int[3] x = [0, 0, 1], y = [1, 1, 0];
    double[3] weights = [0.5, 1.5, 3.0];
    auto h = weightedHistogram(weights, x, y, A(2, 0), A(2, 0));
    assert(h.counts[0, 1] == 2.0);
    assert(h.counts[1, 0] == 3.0);
    assert(h.counts[0, 0] == 0);
}

/++
Construct relative frequencies from weighted counts. Accepts the numeric-count
arguments and counter-type choices of $(LREF weightedHistogram). The total is the sum of
stored weights, including enabled underflow/overflow bins. Normalization and
subsequent weighted insertion use the existing relative-frequency accumulator.
Built-in arrays and Mir slices are accepted. Accumulator-valued cells are not
supported. A single observation collection populates one axis. For joint counts,
use the weighted histogram factory's coordinate/weight ordering and explicit axes.
Use relativeFrequency!(double, Normalization.ordinary) to normalize by ordinary
bin weights only, without discarding the underflow/overflow counts.
The result supports the same relative-frequency, cumulative, and density accessors
as the unweighted relative-frequency factory. Use putWeighted(weight, coordinates...)
for subsequent weighted observations; put adds unit weight. Both update the total.
If the selected normalization total is zero, relative frequencies are NaN.
+/
template weightedRelativeFrequencyHistogram(Options...)
{
    static assert(!isAdaptiveCountSelection!Options,
        "AdaptiveCounts supports only unweighted histogram construction");
    auto weightedRelativeFrequencyHistogram(Args...)(auto ref Args args)
    {
        import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
        import mir.stat.descriptive.histogram.relative_frequency: RelativeFrequencyAccumulator;
        auto h = weightedHistogram!Options(args);
        static if (is(typeof(h) == HistogramAccumulator!Types, Types...))
            return RelativeFrequencyAccumulator!Types(h.counts, h.axis);
    }
}

/// Construct relative frequencies from built-in arrays of observations and weights.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: RegularAxis;
    double[3] observations = [0.25, 0.75, 1.25];
    double[3] weights = [0.5, 1.5, 2.0];
    auto f = weightedRelativeFrequencyHistogram!RegularAxis(weights, observations, 2u, 0.0, 2.0);
    assert(f.total == 4.0);
    assert(f.relativeFrequency(0) == 0.5);
    // Later weighted observations update both the bin and the denominator.
    f.putWeighted(2.0, 0.25);
    assert(f.total == 6.0);
    assert(f.relativeFrequency(0) == 2.0 / 3);
}

// Reject invalid numeric weights while allowing zero-weight observations.
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
        try { auto h = weightedHistogram!RegularAxis(weights, data, 2u, 0.0, 2.0); }
        catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    weights[0] = 0;
    auto f = weightedRelativeFrequencyHistogram!RegularAxis(weights, data, 2u, 0.0, 2.0);
    assert(f.total == 0 && f.counts == [0, 0]);
}

/++
Construct a percentogram using quantile boundaries and observed relative frequencies.
Returns a relative-frequency accumulator with GC-owned boundaries and counts.
Use `density` or `densityBins` for bar heights: area represents observed probability.

The probabilities argument supplies probability levels, not precomputed quantile
boundaries. Alternatively, pass an axis from $(LREF quantileAxis) or
$(LREF quantileAxisFromBoundaries) to reuse prepared boundaries. For direct control
of variable-axis boundaries, the $(LREF relativeFrequencyHistogram) examples show
the individual preparation steps. Use $(LREF histogram) for raw counts.

Omitting probabilities requests `ceil(cuberoot(n))` ordinary bins for `n` observations,
with equally spaced probabilities from zero to one. This is a sample-size heuristic.
Tied boundaries can reduce the number of ordinary bins.

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

Use percentogram!(AdaptiveCounts!()) to widen counts automatically, or select
a larger initial width such as AdaptiveCounts!ushort. Boundaries retain their
usual allocation policy; only the count storage selection changes. The default
remains fixed size_t counts.

Params:
    Counts = fixed counter type (size_t by default), or an AdaptiveCounts selection
    data = one-dimensional observations, as an array or slice
    probabilities = positive bin count or probability array/slice
+/
auto percentogram(Counts = size_t, Data, P)(scope auto ref Data data, scope auto ref P probabilities)
    if (!isQuantileAxis!P)
{
    import mir.stat.descriptive.histogram.api.factory: buildPercentogram;
    return buildPercentogram!(allocateCounts, computeAxisQuantiles!(QuantileAlgo.type7),
        relativeFrequencyHistogram, Counts)(data, probabilities);
}

/// ditto
auto percentogram(Counts = size_t, Data)(scope auto ref Data data)
{
    import mir.stat.descriptive.histogram.api.factory: defaultPercentogramBinCount;
    return percentogram!Counts(data, defaultPercentogramBinCount(data.length));
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

/// Keep quantile boundaries fixed while adaptive counts grow with later observations.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: AdaptiveCounts;
    import mir.ndslice.slice: sliced;
    double[5] values = [0, 0, 1, 2, 2];
    auto p = percentogram!(AdaptiveCounts!ushort)(values[].sliced, 2);
    p.put(0.0);
    assert(p.total == 6 && p.bins()[0].count == 3);
    assert(p.relativeFrequency(0) == 0.5);
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

// Check restricted probability intervals and tail normalization.
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

/++
Project a histogram onto selected axes using fresh garbage-collected storage.
Retain at least one axis and fewer than the source rank, without duplicates.
The template argument order becomes the result's axis order. All stored counts
on discarded axes contribute, including underflow/overflow bins; retained axes
keep their definitions and enabled end bins. Cell types are preserved.

Numeric cells are added. Accumulator cells merge their full state through
put(sourceCell), or through += when that put operation is unavailable. For
example, MeanAccumulator combines counts and sums, so a marginal mean weights
each contributing bin by its observation count rather than averaging bin means.
WMeanAccumulator similarly combines weighted sums and total weights.

Accumulator cells start in their normal default state, which must represent
an empty accumulator. Custom merge operations must combine contributions without
modifying the source. The result has fresh cell storage; references retained by
custom cells follow their merge semantics and are not automatically deep-copied.
Empty bins retain the accumulator's usual empty-state behavior.

Works with HistogramAccumulator and RelativeFrequencyAccumulator. A relative
frequency result recomputes its total from the projected counts. Source and
result numeric counts are independent. Axes must support mir.qualifier.lightConst.
Owning axis boundaries remain owned; borrowed
boundaries must outlive the result and its views. Counts must accommodate the
resulting sums and total.

Use h.marginal!dimension() through UFCS. For reference-counted or custom storage,
use rcMarginal or makeMarginal. This replaces the former RC-only marginal member.
Params:
    dimensions = source axes to retain, in result order
    source = histogram with mergeable cells, or relative frequency accumulator
+/
template marginal(dimensions...)
{
    import mir.stat.descriptive.histogram.api.factory: acceptsMarginal;
    auto marginal(H)(auto ref const H source)
        if (acceptsMarginal!(H, dimensions))
    {
        NoAllocationContext context;
        return source.projectMarginal!(axisImplementation.axisFactory, null,
            NoAllocationContext, dimensions)(context);
    }
}

/++
Summarize request counts by temperature after recording temperature and server
jointly. Keep using GC storage for the summary; its counts are independent of
later requests recorded in the original histogram.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    auto requests = histogram(A(2, 0), A(2, 0));
    requests.put(0, 0);
    requests.put(0, 1);
    requests.put(1, 1);
    auto byTemperature = requests.marginal!0();
    static assert(is(typeof(byTemperature.counts.iterator) == size_t*));
    assert(byTemperature.counts == [2, 1]);
    requests.put(0, 0);
    assert(byTemperature.counts == [2, 1]);
}

/++
Combine sensor-report summaries across longitude to compare latitude bands.
Each report represents a different number of readings. Marginalization preserves
those weights when combining the regional weighted means.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.weighted: WMeanAccumulator, AssumeWeights;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    alias Cell = WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary);
    auto coordinate = RegularAxis!(double, AxisOptions())(2, 0.0, 2.0);
    auto reports = histogram!Cell(coordinate, coordinate);
    reports.putWeightedSample(2.0, 10.0, 0.5, 0.5);
    reports.putWeightedSample(6.0, 30.0, 0.5, 1.5);

    auto byLatitude = reports.marginal!0();
    assert(byLatitude.counts[0].weight == 8.0);
    assert(byLatitude.counts[0].wmean == 25.0);
    // Averaging the two regional means would incorrectly give 20.
    assert(byLatitude.counts[1].weight == 0.0);
    // WMeanAccumulator requires a nonzero weight before reading wmean.
}

// Exercise shared marginalization checks with GC-owned result storage.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testMarginalFactory;
    testMarginalFactory!marginal();
}

// Adaptive selection preserves relative-frequency and percentogram behavior.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testAdaptiveRelativeFactory, testAdaptivePercentogram;
    testAdaptiveRelativeFactory!relativeFrequencyHistogram();
    testAdaptivePercentogram!percentogram();
}

/++
Construct a percentogram whose boundaries and bin totals both account for weights.
For example, reweight simulated outcomes to represent a different scenario:
the weights move the quantile boundaries as well as changing the bar areas.
Returns a relative-frequency accumulator with GC-owned boundaries and counts.

The default inverseCDF algorithm selects observed quantiles. Select another
WeightedQuantileAlgo explicitly; frequencyType7 and frequencyType8 interpret
weights as occurrence counts. All algorithms use the supplied weights for bin
totals. Counts default to double; integral counters require integral weights.
Counts and their total must accommodate the sums. AdaptiveCounts is unsupported.

Supply a positive bin count or strictly increasing probabilities in [0,1].
Omitting probabilities requests ceil(cuberoot(n)) bins, where n is the number
of positive-weight rows, including for frequency algorithms. This heuristic
is invariant to weight rescaling; it does not use the sum of the weights.

Weights and observations must be one-dimensional arrays or Mir slices with
matching lengths. Weighted-quantile input rules apply: finite nonnegative
weights, at least one positive weight, and finite positive-weight observations.
Zero-weight observations are ignored, including nonfinite values. Inputs are
not modified. Inverse-CDF boundaries retain the observation type; interpolating
algorithms use floating-point boundaries and may round large integers.
Floating observations retain their boundary type.

Equal boundaries are combined; at least two distinct finite boundaries are
required. Ties and discrete weights can prevent equal bin weights. Bins are
left-closed. Floating-point final edges are extended by one representable
step to include their cutoff; integral final bins include the unchanged upper
endpoint. Both underflow and overflow are enabled. Tail
weights remain in the default normalization; use Normalization.ordinary to
exclude them. These boundary and density rules follow $(LREF percentogram).
Use density or densityBins for bar heights whose areas represent probability.
Later putWeighted calls update bin weights and the total without moving edges.

Params:
    Counts = numeric counter type, double by default
    algorithm = weighted quantile definition, inverseCDF by default
    weights = probability masses, or occurrence counts for frequency algorithms
    data = one-dimensional observations, as an array or Mir slice
    probabilities = positive bin count or probability array/slice
+/
auto weightedPercentogram(Counts = double, WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF,
    Weights, Data, P)(scope auto ref Weights weights, scope auto ref Data data,
    scope auto ref P probabilities)
    if (!isQuantileAxis!P)
{
    import mir.stat.descriptive.histogram.api.factory: buildWeightedPercentogram;
    return buildWeightedPercentogram!(allocateCounts, computeWeightedAxisQuantiles!algorithm,
        relativeFrequencyHistogram, Counts)(weights, data, probabilities);
}

/// ditto
auto weightedPercentogram(Counts = double, WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF,
    Weights, Data)(scope auto ref Weights weights, scope auto ref Data data)
{
    import mir.stat.descriptive.histogram.api.factory: weightedPercentogramBinCount;
    return weightedPercentogram!(Counts, algorithm)(weights, data, weightedPercentogramBinCount(weights));
}

/// Reweighting outcomes changes the median boundary and the observed bin masses.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    const double[4] values = [0, 10, 20, 30];
    const double[4] equal = [1, 1, 1, 1], weights = [1, 1, 6, 2];
    auto original = weightedPercentogram(equal, values, 2);
    auto reweighted = weightedPercentogram(weights[].sliced, values[].sliced, 2);
    assert(original.bins()[0].bin.high == 10);
    assert(reweighted.bins()[0].bin.high == 20);
    assert(reweighted.counts == [0.0, 2, 8, 0]);
    assert(reweighted.relativeFrequency(1) == 0.8);
    reweighted.putWeighted(2.0, 15.0);
    assert(reweighted.total == 12 && reweighted.counts[1] == 4);
}

// Weighted boundaries, ties, tails, zero weights, and strided inputs.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testWeightedPercentograms;
    testWeightedPercentograms!weightedPercentogram();
}


/++
Choose quantile boundaries once, then reuse them to compare samples against the
same reference distribution. Returns GC-owned boundaries in a prepared quantile axis.
Supply probability levels or a positive bin count for equally spaced levels.
Inputs are unchanged; the selected quantile algorithm defaults to type7.
Equal boundaries are combined. Type1 and type3 retain integral observation types;
interpolating algorithms promote integral observations to double. Floating upper
endpoints are extended once; integral final bins include their unchanged upper
endpoint. Both underflow and overflow are enabled.
+/
auto quantileAxis(QuantileAlgo algorithm = QuantileAlgo.type7, Data, P)(
    scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.stat.descriptive.histogram.api.factory: computeQuantileAxisEdges;
    import mir.stat.descriptive.histogram.axis: preparedQuantileAxis;
    NoAllocationContext context;
    auto edges = computeQuantileAxisEdges!(allocateCounts, computeAxisQuantiles!algorithm)(
        context, data, probabilities);
    return preparedQuantileAxis(edges);
}

/// Compare a later sample with fixed reference quantile intervals.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    double[5] reference = [0, 1, 2, 3, 4];
    double[3] levels = [0, 0.5, 1];
    double[4] later = [-1, 1, 3, 5];
    auto prepared = quantileAxis(reference[].sliced, levels);
    auto p = percentogram(later, prepared);
    assert(p.counts == [1, 1, 1, 1]);
    assert(p.relativeFrequency(0) == 0.25);
    assert(p.density(0) == 0.125);
    p.put(2.5);
    assert(p.total == 5 && p.counts[2] == 2);
}

/++
Copy precomputed quantile boundaries into GC-owned storage.
Use this when another calculation supplies the quantiles. Input boundaries must
be finite and nondecreasing, with at least two distinct values. Duplicates are
combined. Floating-point maxima are extended by one representable step and must
have a finite successor. Integral boundaries retain their exact type and values;
the last ordinary bin includes its upper endpoint without extending it.
The input is unchanged.
+/
auto quantileAxisFromBoundaries(Data)(scope auto ref Data boundaries)
{
    import mir.stat.descriptive.histogram.api.factory: copyQuantileAxisEdges;
    import mir.stat.descriptive.histogram.axis: preparedQuantileAxis;
    NoAllocationContext context;
    auto edges = copyQuantileAxisEdges!(allocateCounts)(context, boundaries);
    return preparedQuantileAxis(edges);
}

/// Use quantiles supplied by another calculation without modifying its boundary array.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: nextUp;
    const double[5] edges = [0, 0, 1, 2, 2];
    auto prepared = quantileAxisFromBoundaries(edges[].sliced);
    assert(prepared.N_bin == 2);
    assert(prepared.high == nextUp(2.0));
    assert(edges == [0, 0, 1, 2, 2]); // Compaction and nextUp affect only the copy.
    double[5] values = [0, 0, 1, 2, 2];
    auto p = percentogram(values, prepared);
    assert(p.counts == [0, 2, 3, 0]);
    assert(p.axis.high == nextUp(2.0)); // Reusing the axis does not extend it again.
}

private template computeAxisQuantiles(QuantileAlgo algorithm)
{
    auto computeAxisQuantiles(Data, P)(
        ref NoAllocationContext context, scope auto ref Data data, scope auto ref P probabilities)
    {
        import mir.stat.descriptive.univariate: quantile;
        return quantile!algorithm(data, probabilities);
    }
}

/++
Count observations against an already-prepared quantile axis. Boundaries are
reused without recalculating quantiles, compacting them, or extending the endpoint.
The result exposes relative frequencies and densities. Empty counting samples
are allowed; subsequent put calls retain the same axis.
+/
auto percentogram(Counts = size_t, Data, Axis)(
    scope auto ref Data data, Axis axis)
    if (isQuantileAxis!Axis)
{
    import mir.stat.descriptive.histogram.api.factory: percentogramOnAxis;
    return percentogramOnAxis!(relativeFrequencyHistogram, Counts)(data, axis);
}

/++
Choose fixed quantile boundaries from a weighted reference sample. Weights come
before observations. Algorithm semantics are those of WeightedQuantileAlgo;
the default is inverseCDF. Supply probability levels or a positive bin count.
Zero-weight observations are ignored. Inputs are unchanged.
The axis can subsequently count a different sample with different weights.
+/
auto weightedQuantileAxis(WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF,
    Weights, Data, P)(scope auto ref Weights weights,
    scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.stat.descriptive.histogram.api.factory: computeQuantileAxisEdges;
    import mir.stat.descriptive.histogram.axis: preparedQuantileAxis;
    NoAllocationContext context;
    auto edges = computeQuantileAxisEdges!(allocateCounts, computeWeightedAxisQuantiles!algorithm)(
        context, weights, data, probabilities);
    return preparedQuantileAxis(edges);
}

/// Reuse a weighted reference median when comparing a later sample.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    double[4] reference = [0, 10, 20, 30], weights = [1, 1, 6, 2];
    auto prepared = weightedQuantileAxis(weights, reference, 2);
    assert(prepared.bin(0).high == 20);
    double[3] later = [double.nan, 10, 25], laterWeights = [0, 3, 1];
    auto p = weightedPercentogram(laterWeights, later, prepared);
    assert(p.counts == [0, 3, 1, 0] && p.total == 4);
    assert(p.relativeFrequency(0) == 0.75);
}

private template computeWeightedAxisQuantiles(WeightedQuantileAlgo algorithm)
{
    auto computeWeightedAxisQuantiles(Weights, Data, P)(
        ref NoAllocationContext context, scope auto ref Weights weights,
        scope auto ref Data data, scope auto ref P probabilities)
    {
        import mir.stat.descriptive.univariate: weightedQuantile;
        return weightedQuantile!algorithm(
            weights, data, probabilities);
    }
}

/++
Accumulate weights against a prepared quantile axis without recalculating boundaries.
The counting weights need not be those used to construct the axis. They must be
finite and nonnegative; zero-weight observations are ignored, including nonfinite
values. Positive-weight observations must be finite. Shapes must match.
No quantile algorithm is selected here because the boundaries are already fixed.
+/
auto weightedPercentogram(Counts = double, Weights, Data, Axis)(
    scope auto ref Weights weights, scope auto ref Data data, Axis axis)
    if (isQuantileAxis!Axis)
{
    import mir.stat.descriptive.histogram.api.factory: weightedPercentogramOnAxis;
    return weightedPercentogramOnAxis!(histogram, Counts)(
        weights, data, axis);
}

// Reuse prepared axes across counts, densities, weights, and read-only views.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testQuantileAxes;
    testQuantileAxes!(quantileAxis, quantileAxisFromBoundaries, percentogram, histogram, weightedQuantileAxis, weightedPercentogram)();
}

// Invalid input is rejected before counting; catches require @system.
version(mir_stat_test)
@system pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.factory: testQuantileAxisRejections;
    testQuantileAxisRejections!(quantileAxis, quantileAxisFromBoundaries, percentogram, weightedPercentogram)();
}
