/++
This module contains algorithms for frequency statistics.

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


module mir.stat.descriptive.histogram.frequency;

import mir.internal.utility: isFloatingPoint;
import std.meta: allSatisfy;
import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
public import mir.stat.descriptive.histogram.accumulator: BinCoverage;
import mir.stat.descriptive.histogram.traits: isAxis, includeUnderflow;
import mir.stat.descriptive.histogram.internal.view: supportsBinView;
import mir.stat.descriptive.histogram.internal.projection: validMarginalAxes;
import mir.stat.internal.borrow: hasBorrowEscapeChecking, uncheckedBorrow;

// Limit destinations to writable floating-point arrays and one-dimensional slices.
private template isFrequencyDestination(Destination)
{
    import mir.ndslice.slice: isSlice;
    import mir.primitives: DeepElementType;
    import std.traits: isDynamicArray, Unqual;

    static if (isDynamicArray!Destination)
        private enum supportedShape = true;
    else static if (isSlice!Destination)
        private enum supportedShape = Destination.N == 1;
    else
        private enum supportedShape = false;

    static if (supportedShape)
        enum isFrequencyDestination =
            isFloatingPoint!(Unqual!(DeepElementType!Destination)) &&
            __traits(compiles, {
                Destination destination;
                destination[0] = Unqual!(DeepElementType!Destination).init;
            });
    else
        enum isFrequencyDestination = false;
}

/++
Histogram wrapper that also maintains the total recorded count.
The total includes ordinary bins and enabled underflow and overflow counters.
Storage supplied to the constructor may already contain counts, including
enabled underflow/overflow bins on every dimension, as in
$(LREF HistogramAccumulator). The total is calculated once from all storage at
construction.

Counts are exposed read-only. Arrays and slices retain their usual aliasing
semantics: callers must not modify backing storage through external aliases or
independently mutate copies of this wrapper that share storage. Counter types
must be large enough for both bin counts and the total.

Relative frequencies divide by the total, including flow counts. The output
defaults to double and can be selected independently on each accessor. An empty
accumulator returns NaNs; an unoccupied bin in a nonempty accumulator returns zero.

Params:
    Storage = count storage, as in $(LREF HistogramAccumulator)
    Axis = types of the histogram axes

See_also:
    $(LREF HistogramAccumulator),
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF TransformAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
struct FrequencyAccumulator(Storage, Axis...)
    if (Axis.length > 0 && allSatisfy!(isAxis, Axis))
{
    import std.traits: isIterable, isSomeString, isStaticArray;
    import mir.stat.descriptive.histogram.traits: includeOverflow, includeUnderflow,
        BinTypeOf, isCategoryAxis, acceptsAxisValue;

    private enum N = Axis.length;
    private alias AxisType = Axis[0];
    private alias Histogram = HistogramAccumulator!(Storage, Axis);
    private Histogram histogramAccumulator;
    private CountType total;

    /// Type used for bin counts and the running total.
    alias CountType = Histogram.CountType;

    /++
    Params:
        counts = counts with the shape required by the histogram axes
        axis = axes used to classify observations
    +/
    this(Storage counts, Axis axis)
    {
        histogramAccumulator = Histogram(counts, axis);
        total = storageTotal(counts);
    }

    // Recurse over dimensions so nested arrays and strided slices agree.
    private static CountType storageTotal(size_t depth = 0, S)(auto ref const S storage)
    {
        CountType result = 0;
        foreach (i; 0 .. storage.length)
        {
            static if (depth + 1 == N)
                result += storage[i];
            else
                result += storageTotal!(depth + 1)(storage[i]);
        }
        return result;
    }

    private static CountType storageCount(size_t depth = 0, S)(
        auto ref const S storage, const ref size_t[N] indices)
    {
        static if (depth + 1 == N)
            return storage[indices[depth]];
        else
            return storageCount!(depth + 1)(storage[indices[depth]], indices);
    }

    /++
    Sum over discarded axes to create a marginal frequency accumulator.

    Uses HistogramAccumulator.marginal's axis selection and ownership rules.
    All stored counts contribute, including underflow/overflow bins. Fresh
    reference-counted storage holds the resulting counts, and the result's
    maintained total is calculated from those counts. Subsequent source and
    result updates are independent. Borrowed axis boundaries remain borrowed;
    scope-bound sources with borrowed axis data are rejected in @safe code.
    The counter type is preserved and must accommodate the sums and total.

    Params:
        dimensions = zero-based source axes to retain, in result order;
            select at least one and fewer than N axes, without duplicates
    +/
    auto marginal(dimensions...)() const
        if (validMarginalAxes!(N, dimensions))
    {
        auto projected = histogramAccumulator.marginal!dimensions();
        static if (is(typeof(projected) == HistogramAccumulator!Args, Args...))
            return FrequencyAccumulator!Args(projected.counts, projected.axis);
    }

    /++
    Read-only bin descriptions and counts, without per-entry frequencies.
    Uses the same coverage options and element type as HistogramAccumulator.bins.
    Owning storage handles keep counts alive independently of this accumulator;
    borrowed storage and axis boundaries must remain alive. Keep shared storage
    shape and axis boundaries unchanged. Use frequencyBins to also read frequencies.

    Params:
        coverage = ordinary bins by default, or all enabled stored bins
    +/
    auto bins(BinCoverage coverage = BinCoverage.ordinary)() const
        if (supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        return histogramAccumulator.bins!coverage();
    }

    /++
    Borrow descriptions and counts from static-array storage without copying.
    The accumulator must stay alive and in place. Safety is inferred as @safe
    when borrow escape checking is enabled, otherwise @system.
    Moving or replacing the source while borrowed is prohibited by contract.

    Params:
        coverage = ordinary bins by default, or all enabled stored bins
    +/
    auto bins(BinCoverage coverage = BinCoverage.ordinary)() return const
        if (isStaticArray!Storage &&
            supportsBinView!(typeof(histogramAccumulator.counts[]), Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        return histogramAccumulator.bins!coverage();
    }

    /++
    Borrow a read-only random-access view of bins and frequencies.
    Coverage defaults to ordinary bins; BinCoverage.all includes enabled
    underflow/overflow bins. The last axis advances fastest.

    Each element reads its count and frequency from this accumulator. The
    denominator includes enabled flow bins. FrequencyType defaults to double;
    frequencies are NaN when the total is zero.

    The accumulator must outlive the view and every saved or sliced cursor.
    Do not move, replace, or destroy it while borrowed. Modify counts only
    through this original accumulator. Returned bin descriptions may themselves
    borrow axis storage. This view does not retain ownership of the accumulator.

    Safety is inferred as @safe when the required escape checking is available
    (currently -preview=dip1000), otherwise @system. Compiler checks do not enforce
    the restriction against moving or replacing the source. In @safe code,
    scope-bound accumulators (for example, with stack-array storage) cannot be
    borrowed by this accessor.

    Params:
        FrequencyType = floating-point output type
        coverage = ordinary bins or all enabled stored bins
    +/
    auto frequencyBins(FrequencyType = double, BinCoverage coverage = BinCoverage.ordinary)() return const
        if (isFloatingPoint!FrequencyType && supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        static if (!hasBorrowEscapeChecking)
            uncheckedBorrow();
        return FrequencyBinView!(Storage, FrequencyType, coverage, Axis)(&this);
    }

    /++
    Borrow a one-dimensional forward range of cumulative frequencies.

    Entries expose count, cumulativeCount, cumulativeFrequency, and the usual
    bin description and classification. Coverage defaults to ordinary bins;
    enabled underflow always contributes to cumulative counts. BinCoverage.all
    also emits enabled underflow/overflow entries. The denominator includes all
    recorded counts; a zero total produces NaNs. Categorical axes follow bin order.

    Traversal takes linear time and constant auxiliary storage, without allocating.
    Reading front does not advance accumulation. Saved cursors move independently.
    The accumulator must remain alive, in place, and unchanged until all cursors
    are finished. Returned descriptions may borrow axis storage. The borrowing
    and @safe restrictions of $(LREF frequencyBins) also apply here.

    Params:
        FrequencyType = floating-point output type; defaults to double
        coverage = ordinary bins or all enabled stored bins
    +/
    auto cumulativeFrequencyBins(FrequencyType = double,
        BinCoverage coverage = BinCoverage.ordinary)() return const
        if (N == 1 && isFloatingPoint!FrequencyType && supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        static if (!hasBorrowEscapeChecking)
            uncheckedBorrow();
        return CumulativeFrequencyBinView!(Storage, FrequencyType, coverage, Axis)(&this);
    }

    /// Total recorded observations, including flow bins.
    CountType count() const @property
    {
        return total;
    }

    /// Read-only access to count storage, including enabled underflow/overflow bins.
    ref const(Storage) counts() const @property
    {
        return histogramAccumulator.counts;
    }

    /++
    Relative frequency of an ordinary bin.

    Supply one ordinary bin index per axis. Storage offsets for enabled
    underflow bins are applied automatically. The denominator includes all
    enabled underflow and overflow counts.
    Returns `FrequencyType.nan` when the total count is zero.

    Params:
        FrequencyType = floating-point output type; defaults to double
        index = ordinary bin indices, one per axis, each less than its N_bin
    +/
    FrequencyType frequency(FrequencyType = double, Indices...)(Indices index) const
        if (isFloatingPoint!FrequencyType && Indices.length == N &&
            allSatisfy!(isIndex, Indices))
    {
        size_t[N] indices;
        static foreach (i; 0 .. N)
        {
            assert(index[i] >= 0 && index[i] < histogramAccumulator.axis[i].N_bin,
                "FrequencyAccumulator.frequency: index is out of range");
            indices[i] = cast(size_t) index[i];
            indices[i] += includeUnderflow!(Axis[i]);
        }
        return relativeFrequency!FrequencyType(storageCount(counts, indices));
    }

    private template isIndex(T)
    {
        import std.traits: isIntegral;
        enum isIndex = isIntegral!T;
    }

    /++
    One-dimensional cumulative relative frequency through an ordinary bin, inclusive.

    Sums current counts from bin zero through index, plus enabled underflow.
    The denominator includes overflow, so the final ordinary bin can have a
    cumulative frequency below one. Returns `FrequencyType.nan` when the total
    is zero. Each call takes time proportional to index + 1 and stores no
    additional cumulative state. For categorical axes, accumulation follows
    the axis's bin order.

    Params:
        FrequencyType = floating-point output type; defaults to double
        index = ordinary bin index, less than axis.N_bin
    +/
    FrequencyType cumulativeFrequency(FrequencyType = double)(size_t index) const
        if (N == 1 && isFloatingPoint!FrequencyType)
    {
        assert(index < axis.N_bin,
            "FrequencyAccumulator.cumulativeFrequency: index is out of range");
        CountType cumulative = 0;
        static if (includeUnderflow!AxisType)
            cumulative = histogramAccumulator.underflow;
        foreach (i; 0 .. index + 1)
            cumulative += histogramAccumulator.counts[i + includeUnderflow!AxisType];
        return relativeFrequency!FrequencyType(cumulative);
    }

    /++
    One-dimensional snapshot of cumulative relative frequencies for all ordinary bins.

    Returns a newly allocated reference-counted Mir slice with one value per
    ordinary bin, in axis order. Each value has the same meaning as
    $(LREF cumulativeFrequency): enabled underflow contributes to the numerator,
    and overflow contributes only to the denominator. No flow entries are
    appended. All values are `FrequencyType.nan` when the total is zero.

    Computes the result in one pass, using linear time and output storage.
    The accumulator must remain unchanged during the call. The result owns
    separate storage, can outlive the accumulator, and is unaffected by later
    insertions or merges. Changing result values does not change the accumulator.

    Params:
        FrequencyType = floating-point output type; defaults to double
    +/
    auto cumulativeFrequencies(FrequencyType = double)() const
        if (N == 1 && isFloatingPoint!FrequencyType)
    {
        import mir.ndslice.allocation: mininitRcslice;

        auto result = mininitRcslice!FrequencyType(axis.N_bin);
        cumulativeFrequencies(result);
        return result;
    }

    /++
    Write cumulative relative frequencies into caller-supplied storage.

    The destination must be a writable floating-point array or one-dimensional
    Mir slice with exactly one element per ordinary bin. Its element type
    determines the output precision. Existing values are overwritten in one
    pass without allocating output storage. Values have the same meaning as
    $(LREF cumulativeFrequency), including NaNs when the total is zero.

    The destination must not overlap the accumulator's count storage. The
    accumulator must remain unchanged during the call. No reference to the
    destination is retained; its values are independent of later source updates.

    Params:
        destination = output storage, with length equal to axis.N_bin
    +/
    void cumulativeFrequencies(Destination)(scope Destination destination) const
        if (N == 1 && isFrequencyDestination!Destination)
    {
        import mir.primitives: DeepElementType;
        import std.traits: Unqual;

        alias FrequencyType = Unqual!(DeepElementType!Destination);
        assert(destination.length == axis.N_bin,
            "FrequencyAccumulator.cumulativeFrequencies: destination length must match ordinary bin count");
        CountType cumulative = 0;
        static if (includeUnderflow!AxisType)
            cumulative = histogramAccumulator.underflow;
        foreach (i; 0 .. axis.N_bin)
        {
            cumulative += histogramAccumulator.counts[i + includeUnderflow!AxisType];
            destination[i] = relativeFrequency!FrequencyType(cumulative);
        }
    }

    private FrequencyType relativeFrequency(FrequencyType)(CountType value) const
        if (isFloatingPoint!FrequencyType)
    {
        if (total == 0)
            return FrequencyType.nan;
        return cast(FrequencyType) value / cast(FrequencyType) total;
    }

    /// Required storage length on an axis, including enabled underflow/overflow.
    size_t storageExtent(size_t dimension = 0)() const
        if (dimension < N)
    {
        return histogramAccumulator.storageExtent!dimension();
    }

    /// Read-only access to an axis; defaults to dimension zero.
    ref auto axis(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return histogramAccumulator.axis[dimension];
    }

    /// Recorded overflow observations, summed over the other dimensions.
    CountType overflow(size_t dimension = 0)() const
        if (dimension < N && includeOverflow!(Axis[dimension]))
    {
        static if (N == 1)
            return histogramAccumulator.overflow;
        else
            return histogramAccumulator.overflow!dimension();
    }

    /++
    Relative frequency of overflow observations on a selected dimension.
    Returns NaN when the total is zero. Totals on different dimensions can overlap.

    Params:
        FrequencyType = floating-point output type; defaults to double
        dimension = axis dimension; defaults to zero
    +/
    FrequencyType overflowFrequency(FrequencyType = double, size_t dimension = 0)() const
        if (isFloatingPoint!FrequencyType && dimension < N && includeOverflow!(Axis[dimension]))
    {
        return relativeFrequency!FrequencyType(overflow!dimension());
    }

    /// Recorded underflow observations, summed over the other dimensions.
    CountType underflow(size_t dimension = 0)() const
        if (dimension < N && includeUnderflow!(Axis[dimension]))
    {
        static if (N == 1)
            return histogramAccumulator.underflow;
        else
            return histogramAccumulator.underflow!dimension();
    }

    /++
    Relative frequency of underflow observations on a selected dimension.
    Returns NaN when the total is zero. Totals on different dimensions can overlap.

    Params:
        FrequencyType = floating-point output type; defaults to double
        dimension = axis dimension; defaults to zero
    +/
    FrequencyType underflowFrequency(FrequencyType = double, size_t dimension = 0)() const
        if (isFloatingPoint!FrequencyType && dimension < N && includeUnderflow!(Axis[dimension]))
    {
        return relativeFrequency!FrequencyType(underflow!dimension());
    }

    /// Record an iterable of observations, preserving the total after each put.
    void put(Range)(Range r)
        if (N == 1 && isIterable!Range &&
            !(isCategoryAxis!AxisType && isSomeString!Range))
    {
        foreach (x; r)
            put(x);
    }

    private template acceptsArguments(T...)
    {
        enum acceptsArguments = () {
            static if (T.length == 0 || (N != 1 && T.length != N))
                return false;
            else
            {
                bool accepts = true;
                static foreach (i; 0 .. T.length)
                    accepts = accepts && acceptsAxisValue!(Axis[N == 1 ? 0 : i], T[i]);
                return accepts;
            }
        }();
    }

    /++
    Record observations, updating the total after each successful insertion.
    With one axis each argument is an observation; with multiple axes supply
    exactly one coordinate per axis for one joint observation.
    +/
    void put(T...)(T x)
        if (acceptsArguments!T)
    {
        static if (N == 1)
        {
            static foreach (i; 0 .. T.length)
            {
                histogramAccumulator.put(x[i]);
                total++;
            }
        }
        else
        {
            histogramAccumulator.put(x);
            total++;
        }
    }

    private template acceptsMerge(F)
    {
        import std.traits: Unqual;
        static if (is(Unqual!F == FrequencyAccumulator!Args, Args...))
            enum acceptsMerge =
                is(Unqual!F == FrequencyAccumulator!(Args[0], Axis)) &&
                is(Unqual!(F.CountType) == Unqual!CountType);
        else
            enum acceptsMerge = false;
    }

    /++
    Merge an accumulator with matching axes and counter type.
    Storage layouts may differ. Self-merging doubles counts and the total.
    Other wrappers must not share destination storage, since their totals would
    become stale.
    +/
    void put(F)(auto ref const F f) if (acceptsMerge!F)
    {
        auto addedCount = f.count;
        histogramAccumulator.put(f.histogramAccumulator);
        total += addedCount;
    }

}

/// Collect observations, inspect counts, and read relative frequencies.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u], Axis(3, 0.0));
    assert(f.count == 0);

    // Bins are [0, 1), [1, 2), and [2, 3).
    f.put(0.5);
    f.put([1.0, 1.5, 2.5]);
    assert(f.counts == [1u, 2u, 1u]);
    assert(f.count == 4);

    // Frequencies default to double and divide each bin count by the total.
    assert(f.frequency(0) == 0.25);
    assert(f.frequency(1) == 0.5);
    assert(f.frequency(2) == 0.25);
}

/// Choose the frequency output type without changing the accumulator.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    f.put([0.5, 1.5]);

    double frequency = f.frequency(0);
    float floatFrequency = f.frequency!float(0);
    assert(frequency == 0.5);
    assert(floatFrequency == 0.5f);
}

/// Evaluate a runtime rule before constructing a frequency accumulator.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;

    auto data = [1.0, 2, 3, 4, 5, 6, 7, 8, 9].sliced;
    size_t observationsPerBin = 3; // A positive runtime setting.
    auto rule = (typeof(data) values) => values.length / observationsPerBin +
        (values.length % observationsPerBin != 0);
    auto n = rule(data);

    // Only the resulting count is needed to construct the axis and storage.
    alias Axis = RegularAxis!(size_t, double, AxisOptions());
    auto f = FrequencyAccumulator!(size_t[], Axis)(new size_t[n], Axis(n, 0.0, 12.0));
    f.put(data);
    assert(f.counts == [3, 4, 2]);
    assert(f.frequency(0) == 3.0 / 9);
    // The callback is never passed to or retained by the accumulator.
}

/// Use quantile boundaries and relative frequencies to prepare a percentogram.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: quantile;
    import mir.stat.descriptive.histogram.axis: variableAxis;
    import std.math: nextUp, fabs;

    auto data = [0.0, 1, 2, 3, 4, 8, 12, 16].sliced;
    auto probabilities = [0.0, 0.25, 0.5, 0.75, 1.0].sliced;
    // First compute the data-dependent boundaries; then build the accumulator.
    // Keep the boundaries in caller-owned storage.
    auto boundaries = (new double[probabilities.length]).sliced;
    foreach (i; 0 .. probabilities.length)
        boundaries[i] = data.quantile(probabilities[i]);
    // Include the sample maximum in the final left-closed, right-open bin.
    boundaries[$ - 1] = nextUp(boundaries[$ - 1]);
    auto axis = variableAxis(boundaries);
    auto f = FrequencyAccumulator!(size_t[], typeof(axis))(
        new size_t[boundaries.length - 1], axis);
    f.put(data);
    assert(f.count == 8 && f.counts == [2, 2, 2, 2]);

    foreach (i; 0 .. boundaries.length - 1)
    {
        const probability = f.frequency(i);
        const width = boundaries[i + 1] - boundaries[i];
        const height = probability / width;
        assert(probability == 0.25);
        // Plot this density as bar height: area, not height, represents 25%.
        assert(fabs(height * width - 0.25) < 1e-14);
    }
    // Repeated quantiles from tied data must be combined before constructing
    // the axis. Equal observed counts are not guaranteed for arbitrary data.
    // Additional observations update frequencies but do not recompute edges.
}

/// Read counts without borrowing the running total.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = FrequencyAccumulator!(uint[], A)([1u, 2u, 3u, 0u], A(2, 0.0));

    // bins has the same meaning as on HistogramAccumulator: descriptions
    // and counts, with ordinary-only coverage by default.
    const counts = f.bins();
    assert(counts.length == 2 && counts.front.count == 2);
    static assert(!__traits(hasMember, typeof(counts.front), "frequency"));

    // Request end bins explicitly. All cursors see shared count updates.
    auto all = f.bins!(BinCoverage.all)();
    f.put(2.5);
    assert(all.back.isOverflow && all.back.count == 1);
    assert(counts.front.index == 0 && counts.front.bin.low == 0);

    // frequencyBins is the separate accessor for per-entry relative frequencies.
    // Its views also borrow f's running total; bins does not.
}

/// Read cumulative frequencies in bin order and choose the output type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([2u, 3u, 5u], Axis(3, 0.0));

    // Include the selected bin and all earlier bins in the numerator.
    assert(f.cumulativeFrequency(0) == 0.2);
    assert(f.cumulativeFrequency(1) == 0.5);
    assert(f.cumulativeFrequency(2) == 1.0);

    // Select an output type per call, just as with frequency.
    float cumulative = f.cumulativeFrequency!float(1);
    assert(cumulative == 0.5f);

    // Recompute from current counts and total after recording another value.
    f.put(0.5);
    assert(f.cumulativeFrequency(1) == 6.0 / 11.0);
}

/// Traverse cumulative frequencies without allocating a snapshot.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], A)([1u, 2u, 1u], A(3, 0));
    auto bins = f.cumulativeFrequencyBins();
    assert(bins.front.count == 1 && bins.front.cumulativeFrequency == 0.25);

    // Reading front twice leaves the running count and position unchanged.
    assert(bins.front.cumulativeCount == 1);
    bins.popFront();
    assert(bins.front.index == 1 && bins.front.bin.low == 1);
    assert(bins.front.count == 2 && bins.front.cumulativeCount == 3);

    // A saved cursor starts here, then advances independently of bins.
    auto saved = bins.save;
    saved.popFront();
    assert(saved.front.cumulativeFrequency == 1.0);
    assert(bins.front.cumulativeFrequency == 0.75);
    // Keep f alive and unchanged until both cursors are finished.
}

/// Include underflow/overflow entries and select the output precision.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto f = FrequencyAccumulator!(uint[], A)([1u, 2u, 3u, 4u], A(2, 0));
    auto all = f.cumulativeFrequencyBins!(float, BinCoverage.all)();
    static assert(is(typeof(all.front.cumulativeFrequency) == float));
    assert(all.front.isUnderflow && all.front.cumulativeFrequency == 0.1f);

    // Underflow contributes even when only ordinary entries are requested.
    auto ordinary = f.cumulativeFrequencyBins();
    assert(ordinary.front.cumulativeCount == 3);
    assert(ordinary.front.cumulativeFrequency == 0.3);

    // Advance past underflow and both ordinary bins to reach overflow.
    all.popFront(); all.popFront(); all.popFront();
    assert(all.front.isOverflow && all.front.cumulativeCount == 10);
    assert(all.front.cumulativeFrequency == 1.0f);
    all.popFront();
    assert(all.empty);
}

/// Collect all cumulative frequencies in an independent snapshot.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([2u, 3u, 5u], Axis(3, 0.0));

    // Allocate one double value per ordinary bin, computing all prefixes once.
    auto cumulative = f.cumulativeFrequencies();
    assert(cumulative == [0.2, 0.5, 1.0]);

    // The snapshot retains its values after the source changes.
    f.put(0.5);
    assert(cumulative == [0.2, 0.5, 1.0]);
    // A new call captures the updated counts and total.
    assert(f.cumulativeFrequencies() == [3.0 / 11, 6.0 / 11, 1.0]);
}

/// Reuse output storage and infer precision from its element type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([1u, 3u], Axis(2, 0.0));
    auto output = new double[2];
    f.cumulativeFrequencies(output);
    assert(output == [0.25, 1.0]);

    // Updating the accumulator leaves existing output values unchanged.
    f.put(0.5);
    assert(output == [0.25, 1.0]);
    // Reuse the same buffer to replace them with the current cumulative values.
    f.cumulativeFrequencies(output);
    assert(output == [0.4, 1.0]);

    // A Mir slice is also accepted; float elements select float precision.
    auto floats = rcslice!float([0.0f, 0.0f]);
    f.cumulativeFrequencies(floats);
    assert(floats == [0.4f, 1.0f]);
}

/// Select a snapshot output type and include flow counts in the calculation.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    alias Axis = IntegralAxis!(uint, double,
        AxisOptions(EnableOverflow(true), EnableUnderflow(true)));
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);

    auto cumulative = f.cumulativeFrequencies!float();
    // Underflow contributes to both prefixes. Overflow stays in the total,
    // so the last ordinary bin ends at 3/4. There are no extra flow entries.
    assert(cumulative == [0.5f, 0.75f]);
    assert(cumulative.length == f.axis.N_bin);

    // Result storage is independent: editing it leaves source counts intact.
    cumulative[0] = 0;
    assert(f.cumulativeFrequency(0) == 0.5);
}

/// Cumulative frequencies include underflow but leave overflow beyond the last bin.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    alias Axis = IntegralAxis!(uint, double,
        AxisOptions(EnableOverflow(true), EnableUnderflow(true)));
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);

    // The first numerator includes one underflow and one ordinary observation.
    assert(f.cumulativeFrequency(0) == 0.5);
    // Overflow contributes to the total of four, but neither numerator.
    // There is no extra ordinary bin for overflow.
    assert(f.cumulativeFrequency(1) == 0.75);
}

/// Enabled flow bins contribute to the total used by all frequencies.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    alias Axis = IntegralAxis!(uint, double,
        AxisOptions(EnableOverflow(true), EnableUnderflow(true)));
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);
    assert(f.counts == [1u, 1u, 1u, 1u]);
    assert(f.underflow == 1);
    assert(f.overflow == 1);
    assert(f.count == 4);

    assert(f.frequency(0) == 0.25);
    assert(f.frequency(1) == 0.25);
    assert(f.underflowFrequency == 0.25);
    assert(f.overflowFrequency == 0.25);
}

/// Empty accumulators return NaN; unoccupied bins in nonempty ones return zero.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.math: isNaN;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    assert(isNaN(f.frequency(0)));
    assert(isNaN(f.frequency(1)));

    f.put(0.5);
    assert(f.frequency(0) == 1.0);
    assert(f.frequency(1) == 0.0);
}

/// Initialize from existing reference-counted storage and merge another total.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto counts = rcslice!uint([2u, 1u]);
    alias F = FrequencyAccumulator!(typeof(counts), Axis);
    auto f = F(counts, Axis(2, 0.0));
    assert(f.count == 3);

    // Each accumulator has separate backing storage.
    auto other = F(rcslice!uint([0u, 0u]), Axis(2, 0.0));
    other.put([0.5, 1.5]);
    f.put(other);
    assert(f.counts == [3u, 2u]);
    assert(f.count == 5);
    assert(other.count == 2);

    // Frequencies use the combined counts and total.
    assert(f.frequency(0) == 0.6);
    assert(f.frequency(1) == 0.4);
}

/// Record joint observations and select an axis when inspecting underflow.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    // Each axis has two ordinary bins plus underflow and overflow.
    uint[4][4] storage;
    auto f = FrequencyAccumulator!(typeof(storage), A, A)(
        storage, A(2, 0.0), A(2, 0.0));

    // Two coordinates describe one observation, so the total increases once.
    f.put(0.5, 1.5);
    f.put(-1.0, 1.5);
    assert(f.count == 2);

    // Frequency indices refer to ordinary bins, without the storage offset.
    assert(f.frequency(0, 1) == 0.5);
    assert(f.frequency!real(0, 1) == 0.5L);

    // Axis 0 has one underflow; axis 1 has none.
    // The denominator includes both observations.
    assert(f.underflow() == 1);
    assert(f.underflow!1() == 0);
    assert(f.underflowFrequency() == 0.5);
    assert(f.underflowFrequency!(float, 1)() == 0.0f);
}

/// Marginal frequencies use all recorded counts, including underflow/overflow.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias X = IntegralAxis!(uint, int, AxisOptions());
    alias Y = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto joint = FrequencyAccumulator!(uint[][], X, Y)(
        [[1u, 4u, 2u], [0u, 3u, 1u]], X(2, 0), Y(1, 0));
    const source = joint;
    auto marginal = source.marginal!0();

    // Sum over every position of axis one. The two retained bins represent
    // all eleven observations, including those outside axis one's interval.
    assert(marginal.counts == [7u, 4u]);
    assert(marginal.count == 11 && marginal.count == joint.count);
    assert(marginal.frequency(0) == 7.0 / 11);
    assert(marginal.frequency!float(1) == 4.0f / 11);

    // The marginal maintains its own counts and total after construction.
    marginal.put(1);
    assert(marginal.count == 12 && joint.count == 11);
    joint.put(0, -1);
    assert(marginal.counts == [7u, 5u]);
}

// Construction, insertion, and merging with array and reference-counted storage.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions(false, true, true));

    void check(Storage)(Storage counts, Storage otherCounts)
    {
        alias F = FrequencyAccumulator!(Storage, Axis);
        auto axis = Axis(3, 0.0);
        auto f = F(counts, axis);
        static assert(is(f.CountType == uint));
        static assert(!__traits(compiles, {
            f.counts[0] = 9;
        }));
        static assert(!__traits(compiles, {
            f.count = 9;
        }));
        assert(f.count == 3);
        assert(f.counts == [0u, 1u, 2u, 0u, 0u]);
        assert(f.axis.N_bin == 3);
        assert(f.underflow == 0 && f.overflow == 0);

        void checkTotal()
        {
            uint sum = 0;
            foreach (value; f.counts)
                sum += value;
            assert(f.count == sum);
        }
        checkTotal();
        f.put(0.5);
        assert(f.count == 4);
        checkTotal();
        f.put([-1.0, 1.5, 3.0]);
        assert(f.count == 7);
        checkTotal();
        f.put([2.0, 2.5].sliced);
        assert(f.counts == [1u, 2u, 3u, 2u, 1u]);
        assert(f.underflow == 1 && f.overflow == 1);
        checkTotal();
        f.put((double[]).init);
        assert(f.count == 9);
        checkTotal();

        auto other = F(otherCounts, axis);
        assert(other.count == 0);
        other.put([-2.0, 0.5, 4.0, 5.0]);
        f.put(other);
        assert(f.counts == [2u, 3u, 3u, 2u, 3u]);
        assert(f.underflow == 2 && f.overflow == 3);
        assert(f.count == 13);
        assert(other.count == 4);
        checkTotal();
    }

    check([0u, 1u, 2u, 0u, 0u], [0u, 0u, 0u, 0u, 0u]);
    check(rcslice!uint([0u, 1u, 2u, 0u, 0u]), rcslice!uint([0u, 0u, 0u, 0u, 0u]));
}

// No-flow axes and category strings use the same counting paths.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        CategoryAxis, EnableOverflow;
    import std.range: only;

    alias Axis = IntegralAxis!(size_t, double, AxisOptions());
    auto f = FrequencyAccumulator!(size_t[], Axis)([0UL, 0], Axis(2, 0.0));
    static assert(!__traits(compiles, f.overflow()));
    static assert(!__traits(compiles, f.underflow()));
    f.put(only(0.5, 1.5, 1.75));
    assert(f.count == 3 && f.counts == [1, 2]);

    enum Label { first, second }
    alias Categories = CategoryAxis!(uint, Label, AxisOptions(EnableOverflow(true)));
    auto c = FrequencyAccumulator!(uint[], Categories)([0u, 0u, 0u], Categories());
    c.put("first");
    c.put(["second", "unknown"]);
    c.put(Label.second);
    assert(c.counts == [1u, 2u, 1u]);
    assert(c.overflow == 1 && c.count == 4);
}

// Rejected input must not inflate the total; a partially accepted range stays consistent.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    alias F = FrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u], Axis(2, 0.0));
    assertThrown!AssertError(f.put(3.0));
    assert(f.count == 0 && f.counts == [0u, 0u]);
    assertThrown!AssertError(f.put([0.5, 3.0]));
    assert(f.count == 1 && f.counts == [1u, 0u]);

    auto incompatible = F([1u, 0u], Axis(2, 1.0));
    assertThrown!AssertError(f.put(incompatible));
    assert(f.count == 1 && f.counts == [1u, 0u]);
    assertThrown!AssertError(F([0u], Axis(2, 0.0)));
}

// Custom axes can provide flow predicates without an AxisOptions member.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    static struct Axis
    {
        alias CountType = uint;
        alias BinType = int;
        enum N_bin = 1;
        uint index(int x) const { assert(x == 0); return 0; }
        bool isUnderflow(int x) const { return x < 0; }
        bool isOverflow(int x) const { return x > 0; }
    }
    alias F = FrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u, 0u], Axis());
    auto other = F([0u, 0u, 0u], Axis());
    f.put([0, -1]);
    other.put([0, 1, 2]);
    f.put(other);
    assert(f.counts == [1u, 2u, 2u]);
    assert(f.underflow == 1 && f.overflow == 2);
    assert(f.count == 5);

    f.put(f);
    assert(f.counts == [2u, 4u, 4u]);
    assert(f.underflow == 2 && f.overflow == 4);
    assert(f.count == 10);
}

// Frequencies use current totals, including flows, for all supported output types.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.math.common: approxEqual;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    void check(Storage)(Storage emptyCounts, Storage populatedCounts)
    {
        alias F = FrequencyAccumulator!(Storage, Axis);
        auto f = F(emptyCounts, Axis(3, 0.0));
        static assert(is(typeof(f.overflowFrequency()) == double));
        static assert(is(typeof(f.underflowFrequency()) == double));
        static assert(!__traits(compiles, f.frequency!uint(0)));
        static assert(!__traits(compiles, f.overflowFrequency!uint()));
        static assert(!__traits(compiles, f.underflowFrequency!uint()));

        static foreach (T; AliasSeq!(float, double, real))
        {
            static assert(is(typeof(f.frequency!T(0)) == T));
            static assert(is(typeof(f.overflowFrequency!T()) == T));
            static assert(is(typeof(f.underflowFrequency!T()) == T));
            foreach (i; 0 .. 3)
                assert(isNaN(f.frequency!T(i)));
            assert(isNaN(f.overflowFrequency!T()));
            assert(isNaN(f.underflowFrequency!T()));
        }

        f.put(0.5);
        assert(f.frequency(0) == 1.0);
        assert(f.frequency(1) == 0.0);
        assert(f.overflowFrequency == 0.0 && f.underflowFrequency == 0.0);
        f.put([-1.0, 0.75, 4.0]);
        static foreach (T; AliasSeq!(float, double, real))
        {
            assert(f.frequency!T(0) == 0.5);
            assert(f.frequency!T(1) == 0.0);
            assert(f.overflowFrequency!T() == 0.25);
            assert(f.underflowFrequency!T() == 0.25);
        }

        auto other = F(populatedCounts, Axis(3, 0.0));
        assert(other.frequency(1) == 1.0);
        assert(other.overflowFrequency == 0.0);
        f.put(other);
        static foreach (T; AliasSeq!(float, double, real))
        {
            assert(f.frequency!T(0).approxEqual(cast(T) 1 / 3));
            assert(f.frequency!T(1).approxEqual(cast(T) 1 / 3));
            assert(f.overflowFrequency!T().approxEqual(cast(T) 1 / 6));
            assert(f.underflowFrequency!T().approxEqual(cast(T) 1 / 6));
        }
        // Reading frequencies does not mutate either counts or the total.
        assert(f.count == 6 && f.counts == [1u, 2u, 2u, 0u, 1u]);
        assert(f.underflow == 1 && f.overflow == 1);
    }
    check([0u, 0u, 0u, 0u, 0u], [0u, 0u, 2u, 0u, 0u]);
    check(rcslice!uint([0u, 0u, 0u, 0u, 0u]), rcslice!uint([0u, 0u, 2u, 0u, 0u]));
}

// Cumulative reads cover all flow options, storage types, and floating outputs.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    static foreach (hasUnderflow; AliasSeq!(false, true))
    static foreach (hasOverflow; AliasSeq!(false, true))
    {{
        alias Axis = IntegralAxis!(uint, double,
            AxisOptions(EnableOverflow(hasOverflow), EnableUnderflow(hasUnderflow)));
        void check(Storage)(Storage storage, Storage otherStorage)
        {
            alias F = FrequencyAccumulator!(Storage, Axis);
            auto f = F(storage, Axis(3, 0.0));
            auto read(return ref const F source) @safe pure nothrow
            {
                return source.cumulativeFrequency(1);
            }
            static assert(is(typeof(f.cumulativeFrequency(0)) == double));
            static assert(!__traits(compiles, f.cumulativeFrequency!uint(0)));
            static foreach (T; AliasSeq!(float, double, real))
            {
                static assert(is(typeof(f.cumulativeFrequency!T(0)) == T));
                assert(isNaN(f.cumulativeFrequency!T(0)));
                assert(isNaN(f.cumulativeFrequency!T(2)));
            }
            // An occupied later bin leaves the preceding cumulative values zero.
            f.put(2.5);
            assert(read(f) == 0);
            assert(f.cumulativeFrequency(2) == 1);
            f.put(0.5);
            static if (hasUnderflow) f.put(-1.0);
            static if (hasOverflow) f.put(3.0);
            const uint low = hasUnderflow ? 1 : 0;
            const uint high = hasOverflow ? 1 : 0;
            static foreach (T; AliasSeq!(float, double, real))
            {
                assert(f.cumulativeFrequency!T(0) == cast(T)(1 + low) / (2 + low + high));
                assert(f.cumulativeFrequency!T(2) == cast(T)(2 + low) / (2 + low + high));
            }
            // Merging must update both the prefix counts and its denominator.
            auto other = F(otherStorage, Axis(3, 0.0));
            other.put(1.5);
            static if (hasUnderflow) other.put(-1.0);
            static if (hasOverflow) other.put(3.0);
            f.put(other);
            assert(read(f) == cast(double)(2 + 2 * low) / (3 + 2 * low + 2 * high));
            assert(f.counts[hasUnderflow .. $ - hasOverflow] == [1u, 1u, 1u]);
            assert(f.count == 3 + 2 * low + 2 * high);
        }
        check(new uint[3 + hasUnderflow + hasOverflow], new uint[3 + hasUnderflow + hasOverflow]);
        check(rcslice!uint(3 + hasUnderflow + hasOverflow), rcslice!uint(3 + hasUnderflow + hasOverflow));
    }}
}

// Snapshot values match scalar access for every flow option and output type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.slice: Slice;
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    static foreach (hasUnderflow; AliasSeq!(false, true))
    static foreach (hasOverflow; AliasSeq!(false, true))
    {{
        alias Axis = IntegralAxis!(uint, double,
            AxisOptions(EnableOverflow(hasOverflow), EnableUnderflow(hasUnderflow)));
        void check(Storage)(Storage storage, Storage otherStorage)
        {
            alias F = FrequencyAccumulator!(Storage, Axis);
            auto f = F(storage, Axis(3, 0.0));
            static assert(is(typeof(f.cumulativeFrequencies()) == Slice!(RCI!double)));
            static assert(!__traits(compiles, f.cumulativeFrequencies!uint()));

            void checkValues(T)(ref const F source)
            {
                auto values = source.cumulativeFrequencies!T();
                static assert(is(typeof(values) == Slice!(RCI!T)));
                assert(values.length == source.axis.N_bin);
                foreach (i; 0 .. values.length)
                {
                    if (source.count == 0)
                        assert(isNaN(values[i]));
                    else
                        assert(values[i] == source.cumulativeFrequency!T(i));
                }
            }
            static foreach (T; AliasSeq!(float, double, real))
                checkValues!T(f);

            // Also cover histograms with only flow observations, where enabled.
            static if (hasUnderflow) f.put(-1.0);
            static if (hasOverflow) f.put(3.0);
            static foreach (T; AliasSeq!(float, double, real))
                checkValues!T(f);

            f.put([0.5, 2.5, 2.5]);
            auto snapshot = f.cumulativeFrequencies();
            const first = snapshot[0];
            const last = snapshot[2];
            f.put(1.5);
            auto other = F(otherStorage, Axis(3, 0.0));
            other.put(0.5);
            static if (hasUnderflow) other.put(-1.0);
            static if (hasOverflow) other.put(3.0);
            f.put(other);
            assert(snapshot[0] == first && snapshot[2] == last);
            static foreach (T; AliasSeq!(float, double, real))
                checkValues!T(f);

            auto fresh = f.cumulativeFrequencies();
            fresh[0] = -1;
            assert(snapshot[0] == first);
            assert(f.counts[hasUnderflow .. $ - hasOverflow] == [2u, 1u, 2u]);
            assert(f.cumulativeFrequency(0) >= 0);
        }
        check(new uint[3 + hasUnderflow + hasOverflow], new uint[3 + hasUnderflow + hasOverflow]);
        check(rcslice!uint(3 + hasUnderflow + hasOverflow), rcslice!uint(3 + hasUnderflow + hasOverflow));
    }}
}

// Destination precision, strided storage, const sources, and allocation-free writes.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    alias F = FrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u], Axis(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Stack output verifies that the overload neither allocates nor escapes.
        void write(ref const F source, scope T[] output) @safe pure nothrow @nogc
        {
            source.cumulativeFrequencies(output);
        }
        T[2] output;
        write(f, output[]);
        assert(isNaN(output[0]) && isNaN(output[1]));
        auto populated = F([1u, 3u], Axis(2, 0.0));
        write(populated, output[]);
        assert(output[] == [T(0.25), T(1)]);

        auto rcOutput = rcslice!T([T(-1), T(-1)]);
        populated.cumulativeFrequencies(rcOutput);
        assert(rcOutput == populated.cumulativeFrequencies!T());

        // Only selected elements are overwritten; the intervening values survive.
        auto backing = [T(-1), T(-1), T(-1), T(-1)];
        auto everyOther = backing.sliced.stride(2);
        populated.cumulativeFrequencies(everyOther);
        assert(backing == [T(0.25), T(-1), T(1), T(-1)]);
    }}
}

// Invalid destinations are rejected, with length checked before any writes.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    const(double)[] readOnly = [0.0, 0.0];
    immutable(double)[] fixedValues = [0.0, 0.0];
    uint[] integers = [0u, 0u];
    auto matrix = [0.0, 0.0].sliced(1, 2);
    static assert(!__traits(compiles, f.cumulativeFrequencies(readOnly)));
    static assert(!__traits(compiles, f.cumulativeFrequencies(readOnly.sliced)));
    static assert(!__traits(compiles, f.cumulativeFrequencies(fixedValues)));
    static assert(!__traits(compiles, f.cumulativeFrequencies(integers)));
    static assert(!__traits(compiles, f.cumulativeFrequencies(matrix)));
    static assert(!__traits(compiles, f.cumulativeFrequencies(0.0)));
    foreach (length; [0, 1, 3])
    {
        auto output = new double[length];
        output[] = -1;
        assertThrown!AssertError(f.cumulativeFrequencies(output));
        foreach (value; output) assert(value == -1);
    }
    f.put(0.5);
    auto tooLong = [-1.0, -1.0, -1.0];
    assertThrown!AssertError(f.cumulativeFrequencies(tooLong.sliced));
    assert(tooLong == [-1.0, -1.0, -1.0]);
}

// Owning snapshots can escape a local accumulator even with DIP1000 enabled.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto snapshot()
    {
        alias Axis = IntegralAxis!(uint, double, AxisOptions());
        auto counts = rcslice!uint([1u, 3u]);
        auto f = FrequencyAccumulator!(typeof(counts), Axis)(counts, Axis(2, 0.0));
        return f.cumulativeFrequencies();
    }
    auto values = snapshot();
    assert(values == [0.25, 1.0]);
    values[0] = 0.5;
    assert(values == [0.5, 1.0]);
}

// Invalid bin indices remain errors even when the total is zero.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    static assert(!__traits(compiles, f.overflowFrequency()));
    static assert(!__traits(compiles, f.underflowFrequency()));
    assertThrown!AssertError(f.frequency(2));
    assertThrown!AssertError(f.cumulativeFrequency(2));
    assertThrown!AssertError(f.frequency(size_t.max));
    assertThrown!AssertError(f.cumulativeFrequency(size_t.max));
    f.put(0.5);
    assertThrown!AssertError(f.frequency(2));
    assertThrown!AssertError(f.cumulativeFrequency(2));
    assert(f.count == 1);
}


/++
A bin's count and cumulative statistics, returned by value.
Check classification before accessing index or bin for underflow/overflow entries.
Changing returned values does not update the accumulator.

Params:
    HistogramElement = underlying one-dimensional histogram bin element
    FrequencyType = floating-point output type
+/
struct CumulativeFrequencyBin(HistogramElement, FrequencyType)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBin;
    static if (is(HistogramElement == HistogramBin!Args, Args...))
        private enum N = Args.length - 1;
    else
        static assert(false, "CumulativeFrequencyBin requires a HistogramBin element");
    static assert(N == 1, "CumulativeFrequencyBin requires one axis");
    private HistogramElement _entry;

    /++
    Write coordinates and the recorded statistics to an output range.
    Uses the same coordinate notation as HistogramBin.
    +/
    void toString(Writer)(ref Writer writer) const
    {
        import mir.format: print;
        import mir.appender: scopedBuffer;
        import std.range.primitives: put;
        // Mir printers need both character and string put overloads. Buffering
        // also supports the character-only writer used by std.format.
        auto buffer = scopedBuffer!(char, 256);
        _entry.formatCoordinates(buffer);
        print(buffer, ": count=", count, ", cumulativeCount=", cumulativeCount,
            ", cumulativeFrequency=", cumulativeFrequency);
        put(writer, buffer.data);
    }

    /// Whether this coordinate is ordinary; dimension defaults to zero.
    bool isOrdinary(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isOrdinary!dimension;
    }

    /// Whether this coordinate is underflow; dimension defaults to zero.
    bool isUnderflow(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isUnderflow!dimension;
    }

    /// Whether this coordinate is overflow; dimension defaults to zero.
    bool isOverflow(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isOverflow!dimension;
    }

    /// Original ordinary-bin index along an axis; defaults to axis zero.
    auto index(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.index!dimension;
    }

    /// Bin description along an axis; defaults to axis zero.
    auto bin(size_t dimension = 0)() @property
        if (dimension < N)
    {
        return _entry.bin!dimension;
    }

    /// ditto
    auto bin(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.bin!dimension;
    }

    /// Count in this bin.
    typeof(HistogramElement.init.count) count;
    /// Count through this bin, including enabled underflow.
    typeof(HistogramElement.init.count) cumulativeCount;
    /// Cumulative count divided by the total, or NaN for a zero total.
    FrequencyType cumulativeFrequency;
}

/// Format cumulative entries with both the bin count and running statistics.
version(mir_stat_test)
pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));
    auto entries = f.cumulativeFrequencyBins;
    entries.popFront();
    assert(format("%s", entries.front) ==
        "bin(low=1.0, high=2.0): count=3, cumulativeCount=4, cumulativeFrequency=1.0");
}

/++
Borrowed forward range created by FrequencyAccumulator.cumulativeFrequencyBins.
The source must stay alive, in place, and unchanged throughout traversal.
Saved cursors share the source but keep independent running counts and positions.
A const cursor supports front and save; save returns a mutable cursor.

Params:
    Storage = source count storage
    FrequencyType = floating-point output type
    coverage = ordinary bins or all enabled stored bins
    Axis = the single source axis type
+/
struct CumulativeFrequencyBinView(Storage, FrequencyType, BinCoverage coverage, Axis)
    if (isFloatingPoint!FrequencyType && supportsBinView!(Storage, Axis) &&
        (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
{
    private alias Accumulator = FrequencyAccumulator!(Storage, Axis);
    private alias Cursor = FrequencyBinView!(Storage, FrequencyType, coverage, Axis);
    private Cursor _bins;
    private Accumulator.CountType _preceding = 0;

    /// Type returned by front.
    alias Element = CumulativeFrequencyBin!(Cursor.BinView.Element, FrequencyType);

    private this(const(Accumulator)* source)
    {
        _bins = Cursor(source);
        static if (coverage == BinCoverage.ordinary && includeUnderflow!Axis)
            _preceding = source.histogramAccumulator.underflow;
    }

    private this(Cursor bins, Accumulator.CountType preceding)
    {
        _bins = bins;
        _preceding = preceding;
    }

    /// Number of remaining entries.
    size_t length() const @property { return _bins.length; }

    /// Whether traversal is exhausted.
    bool empty() const @property { return _bins.empty; }

    /// Current entry; repeated reads do not advance accumulation.
    Element front() const @property
    {
        auto entry = _bins.front;
        Accumulator.CountType cumulative = _preceding;
        cumulative += entry.count;
        return Element(entry._entry, entry.count, cumulative,
            _bins._source.relativeFrequency!FrequencyType(cumulative));
    }

    /// Advance once, retaining the count of the bin just visited.
    void popFront()
    {
        _preceding += _bins.front.count;
        _bins.popFront();
    }

    /// Copy the position and running count, borrowing the same accumulator.
    auto save() const @property
    {
        return CumulativeFrequencyBinView(_bins.save, _preceding);
    }
}

/++
A bin description, count, and relative frequency read from an accumulator.
Values are returned by value; changing an entry does not update the accumulator.
Bin descriptions follow the axis's existing bin API. Check the per-axis
classification before accessing index or bin: both assert for an
underflow/overflow coordinate.

Params:
    HistogramElement = element type of the underlying histogram bin view
    FrequencyType = floating-point output type
+/
struct FrequencyBin(HistogramElement, FrequencyType)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBin;
    static if (is(HistogramElement == HistogramBin!Args, Args...))
        private enum N = Args.length - 1;
    else
        static assert(false, "FrequencyBin requires a HistogramBin element");
    private HistogramElement _entry;

    /++
    Write coordinates and the recorded statistics to an output range.
    Uses the same coordinate notation as HistogramBin.
    +/
    void toString(Writer)(ref Writer writer) const
    {
        import mir.format: print;
        import mir.appender: scopedBuffer;
        import std.range.primitives: put;
        // Mir printers need both character and string put overloads. Buffering
        // also supports the character-only writer used by std.format.
        auto buffer = scopedBuffer!(char, 256);
        _entry.formatCoordinates(buffer);
        print(buffer, ": count=", count, ", frequency=", frequency);
        put(writer, buffer.data);
    }

    /// Whether this coordinate is ordinary; dimension defaults to zero.
    bool isOrdinary(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isOrdinary!dimension;
    }

    /// Whether this coordinate is underflow; dimension defaults to zero.
    bool isUnderflow(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isUnderflow!dimension;
    }

    /// Whether this coordinate is overflow; dimension defaults to zero.
    bool isOverflow(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.isOverflow!dimension;
    }

    /// Original ordinary-bin index along an axis; defaults to axis zero.
    auto index(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.index!dimension;
    }

    /// Bin description along an axis; defaults to axis zero.
    auto bin(size_t dimension = 0)() @property
        if (dimension < N)
    {
        return _entry.bin!dimension;
    }

    /// ditto
    auto bin(size_t dimension = 0)() const @property
        if (dimension < N)
    {
        return _entry.bin!dimension;
    }

    /// Count when the entry was read.
    typeof(HistogramElement.init.count) count;
    /// Relative frequency when the entry was read.
    FrequencyType frequency;
}

/// Format frequencies with the count and bin coordinates.
version(mir_stat_test)
pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));
    const entry = f.frequencyBins.front;
    assert(format("%s", entry) == "bin(low=0.0, high=1.0): count=1, frequency=0.25");
    // Formatting a saved entry reads its recorded values, not live counts.
    f.put(0.5);
    assert(format("%s", entry) == "bin(low=0.0, high=1.0): count=1, frequency=0.25");
}

/// Print frequency entries with writeln or writefln, or choose precision per field.
version(mir_stat_test)
unittest
{
    import std.stdio: writeln, writefln;
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));

    // Call printFrequencies(f) to write to stdout. The helper is compiled but
    // deliberately not called here, keeping documentation tests silent.
    void printFrequencies(typeof(f) frequencies)
    {
        // bins() exposes the underlying counts without frequency statistics.
        writeln(frequencies.bins());
        writefln("Histogram: %s", frequencies.bins());
        writeln(frequencies.frequencyBins());
        writefln("Frequencies: %s", frequencies.frequencyBins());
        foreach (entry; frequencies.frequencyBins())
        {
            writeln(entry);
            // Format fields individually to control their numeric precision.
            writefln("low=%.2f, high=%.2f: count=%s, frequency=%.2f",
                entry.bin.low, entry.bin.high, entry.count, entry.frequency);
        }
        // Cumulative entries also include running counts and frequencies.
        writefln("Cumulative frequencies: %s", frequencies.cumulativeFrequencyBins());
    }

    // Check the corresponding text without performing console I/O.
    enum expectedHistogram = "[bin(low=0.0, high=1.0): count=1, " ~
        "bin(low=1.0, high=2.0): count=3]";
    assert(format("%s", f.bins()) == expectedHistogram);
    assert(format("Histogram: %s", f.bins()) == "Histogram: " ~ expectedHistogram);
    enum expected = "[bin(low=0.0, high=1.0): count=1, frequency=0.25, " ~
        "bin(low=1.0, high=2.0): count=3, frequency=0.75]";
    assert(format("%s", f.frequencyBins()) == expected);
    assert(format("Frequencies: %s", f.frequencyBins()) == "Frequencies: " ~ expected);
    auto entry = f.frequencyBins().front;
    assert(format("low=%.2f, high=%.2f: count=%s, frequency=%.2f",
        entry.bin.low, entry.bin.high, entry.count, entry.frequency) ==
        "low=0.00, high=1.00: count=1, frequency=0.25");
    assert(format("Cumulative frequencies: %s", f.cumulativeFrequencyBins()) ==
        "Cumulative frequencies: [bin(low=0.0, high=1.0): count=1, cumulativeCount=1, cumulativeFrequency=0.25, " ~
        "bin(low=1.0, high=2.0): count=3, cumulativeCount=4, cumulativeFrequency=1.0]");
}

// Mir formatting preserves GC-free output for all frequency precisions.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import mir.appender: scopedBuffer;
    import std.meta: AliasSeq;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    static void checkEntry(E)(E entry, string expected) @safe pure nothrow @nogc
    {
        import mir.format: print;
        auto writer = scopedBuffer!(char, 256);
        print(writer, entry);
        assert(writer.data == expected);
    }

    alias A = IntegralAxis!(uint, double, AxisOptions());
    uint[2] storage = [1, 3];
    auto f = FrequencyAccumulator!(uint[], A)(storage[], A(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {
        checkEntry(f.frequencyBins!T.front,
            "bin(low=0.0, high=1.0): count=1, frequency=0.25");
        checkEntry(f.cumulativeFrequencyBins!T.front,
            "bin(low=0.0, high=1.0): count=1, cumulativeCount=1, cumulativeFrequency=0.25");
    }
}

/++
Borrowed random-access range created by FrequencyAccumulator.frequencyBins.

The range refers to the original accumulator, including its running total.
It does not copy the total or retain separate count storage. Updates through
that accumulator are visible on later reads. Each returned count and frequency
is a value; a bin description may still borrow axis storage.

Zero-count bins are included. Coverage selects ordinary bins or all enabled
underflow/overflow combinations, each visited once. The last axis advances
fastest, independent of storage strides.
The denominator includes underflow/overflow counts. Slicing preserves original
per-axis bin indices.
A const view supports indexing, save, and slicing; derived cursors are mutable.

Keep the source alive and in place until all views and borrowed descriptions
are finished. Do not move or replace it, or mutate shared storage through other
aliases. Assertions detect invalid indices, uninitialized views, and changes to
the bin count; they cannot detect a destroyed source or every replacement.

Params:
    Storage = source count storage
    FrequencyType = floating-point output type
    coverage = ordinary bins or all enabled stored bins
    Axis = source axis types
+/
struct FrequencyBinView(Storage, FrequencyType, BinCoverage coverage, Axis...)
    if (isFloatingPoint!FrequencyType && supportsBinView!(Storage, Axis) &&
        (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBinView;
    private alias Accumulator = FrequencyAccumulator!(Storage, Axis);
    private alias BinView = HistogramBinView!(Storage, coverage, Axis);
    private const(Accumulator)* _source;
    private size_t[Axis.length] _shape;
    private size_t _begin, _end, _outerLength;

    /// Type returned by element access.
    alias Element = FrequencyBin!(BinView.Element, FrequencyType);

    private this(const(Accumulator)* source)
    {
        _source = source;
        auto bins = source.histogramAccumulator.bins!coverage();
        static foreach (i; 0 .. Axis.length)
            _shape[i] = source.axis!i.N_bin;
        _outerLength = source.counts.length;
        _end = bins.length;
    }

    private this(const(Accumulator)* source, size_t begin, size_t end,
        size_t[Axis.length] shape, size_t outerLength)
    {
        _source = source;
        _shape = shape;
        _outerLength = outerLength;
        _begin = begin;
        _end = end;
    }

    private void checkSource() const
    {
        assert(_source !is null, "FrequencyBinView: uninitialized view");
        assert(_source.counts.length == _outerLength,
            "FrequencyBinView: source storage shape changed while borrowed");
        static foreach (i; 0 .. Axis.length)
            assert(_source.axis!i.N_bin == _shape[i],
                "FrequencyBinView: source bin count changed while borrowed");
    }

    /// Number of remaining bins in the selected coverage.
    size_t length() const @property { return _end - _begin; }

    /// Whether the cursor is exhausted.
    bool empty() const @property { return _begin == _end; }

    /// First remaining element, returned by value.
    Element front() const @property
    {
        assert(!empty, "FrequencyBinView.front: empty range");
        return this[0];
    }

    /// Last remaining element, returned by value.
    Element back() const @property
    {
        assert(!empty, "FrequencyBinView.back: empty range");
        return this[length - 1];
    }

    /// Advance the cursor without changing the accumulator.
    void popFront()
    {
        assert(!empty, "FrequencyBinView.popFront: empty range");
        ++_begin;
    }

    /// Remove the last bin from this cursor's range.
    void popBack()
    {
        assert(!empty, "FrequencyBinView.popBack: empty range");
        --_end;
    }

    /// Copy the cursor, borrowing the same source.
    auto save() const @property
    {
        return FrequencyBinView(_source, _begin, _end, _shape, _outerLength);
    }

    /// Read a bin, count, and frequency from the same accumulator.
    Element opIndex(size_t index) const
    {
        checkSource();
        assert(index < length, "FrequencyBinView: index is out of range");
        auto entry = BinView.readElement(_source.counts, _begin + index,
            _shape, _source.histogramAccumulator.axis);
        return Element(entry, entry.count,
            _source.relativeFrequency!FrequencyType(entry.count));
    }

    /// Slice relative to the cursor; entry indices remain original bin indices.
    auto opSlice(size_t begin, size_t end) const
    {
        assert(begin <= end && end <= length,
            "FrequencyBinView: slice is out of range");
        return FrequencyBinView(_source, _begin + begin, _begin + end, _shape, _outerLength);
    }

    /// Copy the full remaining range.
    auto opSlice() const { return save; }

    /// Support $ in indexing and slicing.
    size_t opDollar() const { return length; }
}

/// Iterate over ordinary bins with their counts and relative frequencies.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    f.put([0.5, 1.25, 1.5, 1.75]);

    // f stays alive and in place for the entire traversal.
    foreach (entry; f.frequencyBins())
    {
        assert(entry.bin.low == entry.index);
        assert(entry.frequency == cast(double) entry.count / 4);
    }
}

/// Select an output type independently of the accumulator.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([1u, 1u], Axis(2, 0.0));
    auto bins = f.frequencyBins!float();
    static assert(is(typeof(bins.front.frequency) == float));
    assert(bins.front.frequency == 0.5f);
}

/// Const views share live counts and totals while cursors move independently.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], Axis)([1u, 1u], Axis(2, 0.0));

    // const fixes this view's traversal position; it does not freeze f's data.
    // f must stay alive and in place while fixed and its derived cursors exist.
    const fixed = f.frequencyBins();

    // An entry captures count and frequency values at the time it is read.
    // Initially, the first bin contains one of the two recorded observations.
    auto previous = fixed.front;

    // Adding 0.5 through f increments the first bin to 2 and the total to 3.
    // fixed reads both updated values from f. previous keeps its earlier values.
    f.put(0.5);
    assert(fixed.front.count == 2);
    assert(fixed.front.frequency == 2.0 / 3);
    assert(previous.frequency == 0.5);

    // save creates a mutable cursor at fixed's current position.
    // Advancing cursor skips the first bin without moving fixed.
    auto cursor = fixed.save;
    cursor.popFront();
    assert(cursor.front.frequency == 1.0 / 3);

    // Slicing creates another independent cursor over just the second bin.
    // Its denominator is still f's full total of 3, not the sliced bin's count.
    // Original bin indices are preserved, and fixed remains at bin 0.
    auto tail = fixed[1 .. $];
    assert(tail.front.index == 1 && fixed.front.index == 0);
}

/// Traverse joint bins, keeping per-axis coordinates and a live denominator.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    // Two ordinary bins per axis, plus underflow and overflow on each axis.
    auto f = FrequencyAccumulator!(uint[][], A, A)(
        [[0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u],
         [0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));
    f.put(0.5, 1.5);
    f.put(-1.0, 1.5);

    // Only the four ordinary joint bins appear. Axis 1 advances fastest.
    // Underflow still contributes to the total used for every frequency.
    const bins = f.frequencyBins!real();
    assert(bins.length == 4);
    auto entry = bins[1];
    assert(entry.index == 0 && entry.index!1 == 1);
    assert(entry.bin.low == 0.0 && entry.bin!1.low == 1.0);
    assert(entry.count == 1 && entry.frequency == 0.5L);

    // Reading the same position again sees the new count and total.
    // The earlier entry remains a snapshot of the values it read.
    f.put(0.5, 1.5);
    assert(bins[1].count == 2 && bins[1].frequency == 2.0L / 3);
    assert(entry.count == 1 && entry.frequency == 0.5L);

    // Slicing preserves original coordinates and uses the full total.
    // Its cursor moves independently of the const view.
    auto tail = bins[1 .. $];
    assert(tail.front.index!1 == 1);
    tail.popFront();
    assert(tail.front.index == 1 && tail.front.index!1 == 0);
    assert(bins.front.index == 0 && bins.front.index!1 == 0);
}

/// Count out-of-range observations once, including joint corner bins.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = FrequencyAccumulator!(uint[][], A, A)(
        [[0u, 3u, 0u, 2u], [0u, 0u, 0u, 0u],
         [0u, 5u, 0u, 4u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));

    // Axis totals overlap: the two observations at (underflow, overflow)
    // contribute to both totals.
    assert(f.underflow == 5 && f.overflow!1 == 6);
    uint outside;
    double sum = 0;
    foreach (entry; f.frequencyBins!(double, BinCoverage.all)())
    {
        if (!entry.isOrdinary || !entry.isOrdinary!1)
            outside += entry.count;
        sum += entry.frequency;
    }
    // Every stored joint bin appears once, so the corner is counted once.
    assert(outside == 9 && f.count == 14);
    import mir.math.common: approxEqual;
    assert(sum.approxEqual(1.0));
}

// Runtime behavior is identical with and without escape checking.
version(mir_stat_test)
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.math.common: approxEqual;
    import std.math: isNaN;
    import std.algorithm: equal, map;
    import std.range: retro, take;
    import std.range.primitives: isRandomAccessRange, hasSlicing, hasAssignableElements;
    import std.meta: AliasSeq;

    alias Axis = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    void check(T, Storage)(Storage counts, Storage otherCounts)
    {
        alias F = FrequencyAccumulator!(Storage, Axis);
        auto f = F(counts, Axis(3, 0.0));
        auto bins = f.frequencyBins!T();
        alias View = typeof(bins);
        static assert(isRandomAccessRange!View && hasSlicing!View);
        static assert(!hasAssignableElements!View);
        static assert(is(typeof(bins.front.frequency) == T));
        static assert(!__traits(compiles, {
            bins._source.counts[0] = 10u;
        }));
        assert(isNaN(bins[0].frequency));
        f.put([-1.0, 0.5, 0.75, 2.5, 4.0]);
        assert(bins.length == 3 && f.count == 5);
        assert(bins[0].count == 2 && bins[0].frequency == cast(T) 2 / 5);
        assert(bins[1].count == 0 && bins[1].frequency == 0);
        assert(bins[2].frequency == cast(T) 1 / 5);
        auto previous = bins.front;
        auto other = F(otherCounts, Axis(3, 0.0));
        other.put([-1.0, 1.5]);
        f.put(other);
        assert(f.count == 7 && bins.front.frequency.approxEqual(cast(T) 2 / 7));
        assert(previous.frequency == cast(T) 2 / 5);
        previous.count = 100;
        assert(f.counts[1] == 2);
        assert(bins.map!(e => e.count).equal([2u, 1u, 1u]));
        assert(bins.retro.map!(e => e.index).equal([2UL, 1UL, 0UL]));
        assert(bins.take(2).length == 2);

        const fixed = bins;
        auto cursor = fixed.save;
        cursor.popFront();
        cursor.popBack();
        assert(cursor.front.index == 1 && cursor.length == 1);
        assert(fixed.length == 3);
        assert(fixed[1 .. $][1 .. $].front.index == 2);
        assert(fixed[].length == 3);
        assert(fixed[$ .. $].empty);
        cursor.popFront();
        assert(cursor.empty);

        // Further insertion changes the denominator even for a sliced view.
        auto tail = fixed[2 .. $];
        f.put(0.5);
        assert(tail.front.frequency == cast(T) 1 / 8);
    }
    static foreach (T; AliasSeq!(float, double, real))
    {
        check!T([0u, 0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u, 0u]);
        check!T(rcslice!uint([0u, 0u, 0u, 0u, 0u]), rcslice!uint([0u, 0u, 0u, 0u, 0u]));
    }
}

// Bin descriptions work across all built-in axis types.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis,
        TransformAxis, VariableAxis, EnumAxis, CategoryAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: rcslice;
    import mir.rc.array: RCI;
    import mir.math.common: approxEqual;

    void check(Axis)(Axis axis)
    {
        auto f = FrequencyAccumulator!(uint[], Axis)([1u, 3u], axis);
        const reader = f;
        auto bins = reader.frequencyBins();
        foreach (i; 0 .. bins.length)
        {
            const expected = axis;
            auto description = expected.bin(i);
            static if (__traits(hasMember, typeof(description), "slot"))
                assert(bins[i].bin.slot == description.slot);
            else
            {
                assert(bins[i].bin.low.approxEqual(description.low));
                assert(bins[i].bin.high.approxEqual(description.high));
            }
            assert(bins[i].frequency == (i == 0 ? 0.25 : 0.75));
        }
    }
    check(IntegralAxis!(uint, double, AxisOptions())(2, 0.0));
    check(RegularAxis!(uint, double, AxisOptions(true))(2, 0.0, 4.0));
    check(TransformAxis!(uint, double, "a * 2", "a / 2", AxisOptions())(2, 0.0, 4.0));
    check(VariableAxis!(uint, double*, AxisOptions())([0.0, 1.0, 4.0].sliced));
    check(VariableAxis!(uint, RCI!double, AxisOptions())(rcslice!double([0.0, 1.0, 4.0])));
    enum Label { first, second }
    check(EnumAxis!(uint, Label)());
    check(CategoryAxis!(uint, Label, AxisOptions())());
}

// Assertions catch malformed use, but do not claim to detect dangling pointers.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    alias F = FrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u], Axis(2, 0.0));
    auto bins = f.frequencyBins();
    alias View = typeof(bins);
    static assert(!__traits(compiles, f.frequencyBins!uint()));
    View uninitialized;
    assert(uninitialized.empty);
    assertThrown!AssertError(uninitialized[0]);
    assertThrown!AssertError(bins[2]);
    assertThrown!AssertError(bins[size_t.max]);
    assertThrown!AssertError(bins[0 .. 3]);
    assertThrown!AssertError(bins[1 .. 0]);
    auto empty = bins[0 .. 0];
    assertThrown!AssertError(empty.front);
    assertThrown!AssertError(empty.back);
    assertThrown!AssertError(empty.popFront());
    assertThrown!AssertError(empty.popBack());

    // Replacement is forbidden by contract. Detect a changed shape when possible.
    f = F([0u], Axis(1, 0.0));
    assertThrown!AssertError(bins.front);
}

// The API cannot be called from @safe code without the required escape checking.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    alias F = FrequencyAccumulator!(uint[], Axis);
    enum safeBorrowCompiles = __traits(compiles, () @safe {
        auto f = F([1u, 1u], Axis(2, 0.0));
        auto bins = f.frequencyBins();
        assert(bins.front.frequency == 0.5);
    });
    static assert(safeBorrowCompiles == hasBorrowEscapeChecking);

    // Safe code must never return a view of a local accumulator. Without escape
    // checking, borrowing itself is @system; with it, the return is rejected.
    // Older compilers may permit this escape in @system code.
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 1u], Axis(2, 0.0));
        return f.frequencyBins();
    }));
}

// This version selects tests, never the production safety annotation.
version(mir_stat_test_lifetime)
{
    static assert(hasBorrowEscapeChecking,
        "Histogram lifetime tests require -preview=dip1000 escape checking");

    @safe pure nothrow
    unittest
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
        import mir.ndslice.allocation: rcslice;
        import std.algorithm: map, equal;
        alias Axis = IntegralAxis!(uint, double, AxisOptions());

        void check(Storage)(Storage counts)
        {
            alias F = FrequencyAccumulator!(Storage, Axis);
            auto borrow(return ref const F f) { return f.frequencyBins(); }
            auto saved(return ref const F f)
            {
                const bins = f.frequencyBins();
                return bins.save;
            }
            auto sliced(return ref const F f)
            {
                const bins = f.frequencyBins();
                return bins[0 .. 1];
            }
            auto f = F(counts, Axis(2, 0.0));
            auto bins = borrow(f);
            auto copy = saved(f);
            auto part = sliced(f);
            f.put(0.5);
            assert(bins.front.frequency == 2.0 / 3);
            assert(copy.front.frequency == 2.0 / 3);
            assert(part.front.frequency == 2.0 / 3);
            assert(bins.map!(e => e.count).equal([2u, 1u]));

            // Numeric entries contain values and may outlive the borrowed source.
            auto entry()
            {
                auto local = F(counts, Axis(2, 0.0));
                return local.frequencyBins().front;
            }
            assert(entry().count == 2);
        }
        check([1u, 1u]);
        check(rcslice!uint([1u, 1u]));

        // A borrowed variable axis is usable while its backing arrays live.
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: VariableAxis;
        alias Variable = VariableAxis!(uint, double*, AxisOptions());
        uint[] counts = [1, 3];
        double[] breaks = [0, 1, 4];
        auto f = FrequencyAccumulator!(uint[], Variable)(
            counts, Variable(breaks.sliced));
        auto bins = f.frequencyBins();
        assert(bins[1].bin.low == 1 && bins[1].bin.high == 4);
        assert(bins[1].frequency == 0.75);
    }

    unittest
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
        alias Axis = IntegralAxis!(uint, double, AxisOptions());
        alias F = FrequencyAccumulator!(uint[], Axis);
        alias View = FrequencyBinView!(uint[], double, BinCoverage.ordinary, Axis);
        static assert(!__traits(compiles, () @safe {
            auto bins = F([1u, 1u], Axis(2, 0.0)).frequencyBins();
            auto value = bins.front;
        }));
        static assert(!__traits(compiles, () @safe {
            auto f = F([1u, 1u], Axis(2, 0.0));
            auto bins = f.frequencyBins();
            return bins.save;
        }));
        static assert(!__traits(compiles, () @safe {
            auto f = F([1u, 1u], Axis(2, 0.0));
            const bins = f.frequencyBins();
            return bins[0 .. 1];
        }));
        static assert(!__traits(compiles, () @safe {
            View bins;
            {
                auto f = F([1u, 1u], Axis(2, 0.0));
                bins = f.frequencyBins();
            }
            auto value = bins.front;
        }));
        static assert(!__traits(compiles, () @safe {
            static View escaped;
            auto f = F([1u, 1u], Axis(2, 0.0));
            escaped = f.frequencyBins();
        }));
        static assert(!__traits(compiles, () @safe {
            auto forward(return ref const F f) { return f.frequencyBins().save; }
            auto f = F([1u, 1u], Axis(2, 0.0));
            return forward(f);
        }));
        // DIP1000 also rejects borrowing an already scope-bound accumulator.
        static assert(!__traits(compiles, () @safe {
            import mir.ndslice.slice: sliced;
            import mir.stat.descriptive.histogram.axis: VariableAxis;
            alias Variable = VariableAxis!(uint, double*, AxisOptions());
            uint[2] counts = [1, 3];
            double[3] breaks = [0, 1, 4];
            auto f = FrequencyAccumulator!(uint[], Variable)(
                counts[], Variable(breaks[].sliced));
            auto bins = f.frequencyBins();
            auto entry = bins.front;
        }));
    }
}

// Frequency reads, merges, and owning cumulative snapshots remain @nogc.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static immutable uint[4] zero = [0, 0, 0, 0];
    static immutable double[4] samples = [-1.0, 0.5, 1.5, 3.0];
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto counts = rcslice!uint(zero[]);
    alias F = FrequencyAccumulator!(typeof(counts), A);
    auto f = F(counts, A(2, 0.0));
    f.put(samples[]);
    auto other = F(rcslice!uint(zero[]), A(2, 0.0));
    other.put(samples[]);
    f.put(other);
    void read(ref const F source) @safe pure nothrow @nogc
    {
        assert(source.count == 8);
        assert(source.underflow == 2 && source.overflow == 2);
        assert(source.underflowFrequency!float() == 0.25f);
        assert(source.overflowFrequency!real() == 0.25L);
        assert(source.frequency(0) == 0.25);
        assert(source.cumulativeFrequency(1) == 0.75);
        auto snapshot = source.cumulativeFrequencies();
        assert(snapshot[0] == 0.5 && snapshot[1] == 0.75);
        double[2] output;
        source.cumulativeFrequencies(output[]);
        assert(output[0] == snapshot[0] && output[1] == snapshot[1]);
    }
    read(f);
}

// Borrowed-view traversal is @nogc with either compiler escape-checking mode.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static immutable uint[2] initial = [1, 1];
    // Infer safety so the same test is @safe with escape checking and @system
    // without it; neither version may introduce GC allocation.
    auto exercise = () @nogc {
        alias A = IntegralAxis!(uint, double, AxisOptions());
        auto counts = rcslice!uint(initial[]);
        auto f = FrequencyAccumulator!(typeof(counts), A)(counts, A(2, 0.0));
        auto view = f.frequencyBins();
        auto saved = view.save;
        auto tail = view[1 .. 2];
        view.popFront(); saved.popBack();
        assert(view.front.frequency == 0.5 && saved.back.frequency == 0.5);
        f.put(1.5);
        assert(tail.front.frequency == 2.0 / 3.0);
        tail.popFront();
        assert(tail.empty);
    };
    static if (hasBorrowEscapeChecking)
    {
        scope auto runSafe = () @safe @nogc { exercise(); };
        runSafe();
    }
    else
        exercise();
}

// Variadic frequency insertion validates every type and counts every observation.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = FrequencyAccumulator!(uint[], A)([0u, 0u, 0u, 0u], A(2, 0.0));
    static assert(__traits(compiles, f.put(0.5, 1.5)));
    static assert(!__traits(compiles, f.put()));
    static assert(!__traits(compiles, f.put(0.5, "invalid")));
    static assert(!__traits(compiles, f.put("invalid", 0.5)));
    f.put(-1.0, 0.5, 1.5, 3.0);
    assert(f.count == 4 && f.counts == [1u, 1u, 1u, 1u]);
    assert(f.underflow == 1 && f.overflow == 1);
    const double first = 0.5;
    immutable double second = 1.5;
    static assert(__traits(compiles, f.put(first, second)));
    f.put(first, second);
    const double[] readOnly = [0.5, 1.5];
    immutable double[] frozen = [0.5, 1.5];
    f.put(readOnly);
    f.put(frozen);
    assert(f.count == 10 && f.counts == [1u, 4u, 4u, 1u]);
    enum Label { first, second }
    alias C = CategoryAxis!(uint, Label, AxisOptions(false, true));
    auto categories = FrequencyAccumulator!(uint[], C)([0u, 0u, 0u], C());
    static assert(__traits(compiles, categories.put(Label.first, "second")));
    static assert(!__traits(compiles, categories.put(Label.first, 0.5)));
    categories.put(Label.first, "second", "unknown");
    assert(categories.count == 3 && categories.overflow == 1);
    assert(categories.counts == [1u, 1u, 1u]);
}

// A rejected batch element preserves the counts and total of earlier insertions.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = FrequencyAccumulator!(uint[], A)([0u, 0u], A(2, 0.0));
    assertThrown!AssertError(f.put(0.5, double.nan, 1.5));
    assert(f.count == 1 && f.counts == [1u, 0u]);
}

// Joint totals and access agree for nested arrays and noncontiguous slices.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    uint[4][4] initial;
    initial[0][3] = 2; // Underflow on axis 0, overflow on axis 1.
    initial[1][2] = 3; // Ordinary bin (0, 1).
    auto f = FrequencyAccumulator!(typeof(initial), A, A)(
        initial, A(2, 0.0), A(2, 0.0));
    assert(f.count == 5);
    assert(f.underflow == 2 && f.overflow!1 == 2);
    assert(f.frequency(0, 1) == 0.6);
    assert(f.axis!1.N_bin == 2);
    static assert(is(typeof(f.axis!1()) == const(A)));
    static assert(!__traits(compiles, f.frequency(0)));
    static assert(!__traits(compiles, f.frequency(0, 0, 0)));
    static assert(!__traits(compiles, f.frequency(0.5, 0)));
    static assert(!__traits(compiles, f.frequency!uint(0, 0)));
    static assert(!__traits(compiles, f.put(0.5)));
    static assert(!__traits(compiles, f.put(0.5, 0.5, 0.5)));
    static assert(!__traits(compiles, f.cumulativeFrequency(0)));
    static assert(!__traits(compiles, f.cumulativeFrequencies()));
    static assert(!__traits(compiles, f.frequencyBins()));
    static assert(!__traits(compiles, f.underflow!2()));
    static assert(!__traits(compiles, { f.counts[0][0] = 1; }));

    uint[16] buffer;
    auto storage = buffer[].sliced(4, 4).transposed;
    storage[0, 3] = 2;
    storage[1, 2] = 3;
    auto g = FrequencyAccumulator!(typeof(storage), A, A)(
        storage, A(2, 0.0), A(2, 0.0));
    assert(g.count == f.count);
    assert(g.frequency(0, 1) == f.frequency(0, 1));

    // Merge across storage representations, then verify self-merge totals.
    f.put(g);
    assert(f.count == 10 && f.counts[1][2] == 6);
    assert(f.underflow == 4 && f.overflow!1 == 4);
    g.put(f);
    assert(g.count == 15 && storage[1, 2] == 9);
    f.put(f);
    assert(f.count == 20 && f.frequency(0, 1) == 0.6);
    const snapshot = f; // Static storage is an independent copy.
    g.put(snapshot);
    assert(g.count == 35 && g.frequency(0, 1) == 0.6);

    static foreach (T; AliasSeq!(float, double, real))
    {
        static assert(is(typeof(f.frequency!T(0, 1)) == T));
        static assert(is(typeof(f.overflowFrequency!(T, 1)()) == T));
        assert(f.underflowFrequency!T() == cast(T) 0.4);
        assert(f.overflowFrequency!(T, 1)() == cast(T) 0.4);
    }
}

// Three dimensions, mixed axis options, empty totals, and rejected observations.
// Catching assertion failures requires @system.
version(mir_stat_test)
@system pure
unittest
{
    import std.meta: AliasSeq;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.math: isNaN;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    alias B = IntegralAxis!(uint, double, AxisOptions(false, true, false));
    alias C = IntegralAxis!(uint, double, AxisOptions(false, false, true));
    uint[3][3][2] storage;
    auto f = FrequencyAccumulator!(typeof(storage), A, B, C)(
        storage, A(2, 0.0), B(2, 0.0), C(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {
        assert(isNaN(f.frequency!T(0, 0, 0)));
        assert(isNaN(f.overflowFrequency!(T, 1)()));
        assert(isNaN(f.underflowFrequency!(T, 2)()));
    }
    f.put(0.5, 3.0, -1.0);
    assert(f.count == 1 && f.counts[0][2][0] == 1);
    assert(f.overflow!1 == 1 && f.underflow!2 == 1);
    assert(f.frequency(0, 0, 0) == 0);
    f.put(1.5, 0.5, 1.5);
    assert(f.count == 2 && f.frequency(1, 0, 1) == 0.5);
    assertThrown!AssertError(f.put(0.5, 0.5, 9.0));
    assert(f.count == 2 && f.counts[0][0][0] == 0);
    assertThrown!AssertError(f.frequency(-1, 0, 0));
    assertThrown!AssertError(f.frequency(0, 2, 0));

    auto incompatible = FrequencyAccumulator!(typeof(storage), A, B, C)(
        storage, A(2, 1.0), B(2, 0.0), C(2, 0.0));
    assertThrown!AssertError(f.put(incompatible));
    assert(f.count == 2 && f.frequency(1, 0, 1) == 0.5);
}

// Dynamic nested arrays validate every row and support independent merge sources.
version(mir_stat_test)
@system
unittest
{
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    alias F = FrequencyAccumulator!(uint[][], A, A);
    auto f = F([[1u, 2u], [3u, 4u]], A(2, 0.0), A(2, 0.0));
    assert(f.count == 10 && f.frequency(1, 0) == 0.3);
    f.put(0.5, 1.5);
    assert(f.count == 11 && f.counts[0][1] == 3);
    f.put(F([[0u, 1u], [0u, 0u]], A(2, 0.0), A(2, 0.0)));
    assert(f.count == 12 && f.counts[0][1] == 4);

    assertThrown!AssertError(F([[0u, 0u], [0u]], A(2, 0.0), A(2, 0.0)));
    alias OtherCount = FrequencyAccumulator!(ulong[][], A, A);
    OtherCount other;
    static assert(!__traits(compiles, f.put(other)));
    static assert(!__traits(compiles, f.put()));
    static assert(!__traits(compiles, f.put("x", 1)));
}


// Joint traversal works with nested arrays and strided storage in three dimensions.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import std.range.primitives: isRandomAccessRange, hasSlicing, hasLength;
    import std.meta: AliasSeq;
    import std.math: isNaN;

    alias X = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    alias Y = IntegralAxis!(uint, double, AxisOptions());
    alias Z = IntegralAxis!(uint, int, AxisOptions(false, true, false));
    uint[36] buffer;
    auto storage = buffer[].sliced(3, 3, 4).transposed!(2, 1, 0);
    auto f = FrequencyAccumulator!(typeof(storage), X, Y, Z)(
        storage, X(2, 0), Y(3, 0.0), Z(2, 0));
    auto bins = f.frequencyBins();
    static assert(isRandomAccessRange!(typeof(bins)));
    static assert(hasSlicing!(typeof(bins)) && hasLength!(typeof(bins)));
    assert(bins.length == 12);
    assert(isNaN(bins.front.frequency));

    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 2)
                f.put(cast(int) i, cast(double) j, cast(int) k);
    f.put(-1, 0.0, 3); // Excluded entry, included in the total.
    assert(f.count == 13);
    foreach (flat; 0 .. bins.length)
    {
        auto entry = bins[flat];
        assert(entry.index == flat / 6);
        assert(entry.index!1 == flat / 2 % 3);
        assert(entry.index!2 == flat % 2);
        assert(entry.bin.low == entry.index);
        assert(entry.bin!1.low == entry.index!1);
        assert(entry.bin!2.low == entry.index!2);
        assert(entry.count == 1 && entry.frequency == 1.0 / 13);
    }
    static assert(!__traits(compiles, bins.front.index!3));
    static assert(!__traits(compiles, bins.front.bin!3));
    const fixed = bins;
    auto saved = fixed.save;
    auto middle = fixed[3 .. 9];
    saved.popBack();
    assert(saved.length == 11 && fixed.length == 12);
    assert(middle.front.index!1 == 1 && middle.front.index!2 == 1);
    assert(middle.back.index == 1 && middle.back.index!1 == 1);

    // A merge changes the live denominator and the selected bin.
    uint[3][3][4] otherStorage;
    auto other = FrequencyAccumulator!(typeof(otherStorage), X, Y, Z)(
        otherStorage, X(2, 0), Y(3, 0.0), Z(2, 0));
    other.put(0, 0.0, 0);
    f.put(other);
    assert(fixed.front.count == 2 && fixed.front.frequency == 2.0 / 14);
    static foreach (T; AliasSeq!(float, double, real))
    {{
        auto typed = f.frequencyBins!T();
        static assert(is(typeof(typed.front.frequency) == T));
        assert(typed.front.frequency == cast(T) 2 / 14);
    }}
}


// Owning storage still yields borrowed frequency views: only the source owns the total.
version(mir_stat_test_lifetime)
@safe @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto storage = rcslice!uint(2, 2);
    alias F = FrequencyAccumulator!(typeof(storage), A, A);
    auto f = F(storage, A(2, 0), A(2, 0));
    auto borrow(return ref const F source) { return source.frequencyBins(); }
    auto bins = borrow(f);
    const fixed = bins;
    auto saved = fixed.save;
    auto part = fixed[0 .. 2];
    f.put(0, 1);
    assert(saved[1].frequency == 1 && part[1].count == 1);
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.frequencyBins();
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.frequencyBins().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.frequencyBins()[0 .. 2];
    }));
}


// Heterogeneous axes preserve the numeric and categorical description APIs.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, EnumAxis, AxisOptions;
    enum Label { first, second }
    alias X = IntegralAxis!(uint, double, AxisOptions());
    alias Y = EnumAxis!(uint, Label);
    alias F = FrequencyAccumulator!(uint[][], X, Y);
    auto f = F([[0u, 0u], [0u, 0u]], X(2, 0.0), Y());
    f.put(1.5, Label.second);
    const bins = f.frequencyBins();
    assert(bins.back.index == 1 && bins.back.index!1 == 1);
    assert(bins.back.bin.low == 1.0);
    assert(bins.back.bin!1.slot == Label.second);
    assert(bins.back.count == 1 && bins.back.frequency == 1);

    // The public borrowing accessor is safe only with escape checking.
    static assert(__traits(compiles, () @safe {
        auto local = F([[0u, 0u], [0u, 0u]], X(2, 0.0), Y());
        auto view = local.frequencyBins();
        auto entry = view.front;
    }) == hasBorrowEscapeChecking);
}


// Prepopulated end bins contribute once to totals and cumulative frequencies.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static foreach (u; [false, true])
    static foreach (o; [false, true])
    {{
        alias A = IntegralAxis!(uint, double, AxisOptions(false, o, u));
        uint[2 + u + o] initial;
        initial[u] = 3;
        initial[u + 1] = 5;
        static if (u) initial[0] = 7;
        static if (o) initial[$ - 1] = 11;
        alias F = FrequencyAccumulator!(typeof(initial), A);
        auto f = F(initial, A(2, 0.0));
        enum total = 8 + 7 * u + 11 * o;
        assert(f.count == total && f.storageExtent() == initial.length);
        assert(f.frequency(0) == 3.0 / total);
        assert(f.frequency(1) == 5.0 / total);
        double[2] result;
        f.cumulativeFrequencies(result[]);
        assert(result[0] == (3.0 + 7 * u) / total);
        assert(result[1] == (8.0 + 7 * u) / total);
        assert(f.cumulativeFrequency(1) == result[1]);
        auto snapshot = f.cumulativeFrequencies();
        assert(snapshot.length == 2 && snapshot == result[]);
        const source = f;
        f.put(source);
        assert(f.count == 2 * total);
        assert(f.frequency(0) == 3.0 / total);
        static if (u) assert(f.underflow == 14);
        static if (o) assert(f.overflow == 22);
    }}
}


// All-bin frequency views retain live totals, precision selection, and independent cursors.
version(mir_stat_test)
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    alias F = FrequencyAccumulator!(uint[], A);
    auto f = F([0u, 0u, 0u, 0u], A(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const empty = f.frequencyBins!(T, BinCoverage.all)();
        foreach (entry; empty.save) assert(isNaN(entry.frequency));
    }}
    f.put(-1.0, 0.5, 1.5, 2.0);
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const all = f.frequencyBins!(T, BinCoverage.all)();
        static assert(is(typeof(all.front.frequency) == T));
        auto before = all.back;
        auto tail = all[1 .. $];
        auto saved = all.save;
        saved.popFront();
        saved.popBack();
        assert(all.front.isUnderflow && all.back.isOverflow);
        assert(saved.front.isOrdinary && saved.front.index == 0);
        assert(saved.back.index == 1);
        assertThrown!AssertError(all.front.index);
        assertThrown!AssertError(all.back.bin);
        auto other = F([0u, 0u, 0u, 1u], A(2, 0.0));
        const oldTotal = f.count;
        f.put(other);
        assert(all.back.count == before.count + 1);
        assert(all.back.frequency == cast(T)(before.count + 1) / (oldTotal + 1));
        assert(tail.back.frequency == all.back.frequency);
        assert(all.front.frequency == cast(T) 1 / f.count);
    }}
    static assert(!__traits(compiles, f.frequencyBins!(double, cast(BinCoverage) 99)()));
}

// Enabling all-bin coverage does not weaken borrowing or introduce GC allocation.
version(mir_stat_test_lifetime)
@safe @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto counts = rcslice!uint(4, 4);
    alias F = FrequencyAccumulator!(typeof(counts), A, A);
    auto f = F(counts, A(2, 0), A(2, 0));
    auto borrow(return ref const F source) { return source.frequencyBins!(real, BinCoverage.all)(); }
    const bins = borrow(f);
    auto part = bins[0 .. 4];
    f.put(-1, 2);
    assert(part.back.isUnderflow && part.back.isOverflow!1);
    assert(part.back.count == 1 && part.back.frequency == 1);
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.frequencyBins!(real, BinCoverage.all)();
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.frequencyBins!(real, BinCoverage.all)().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.frequencyBins!(real, BinCoverage.all)()[0 .. 2];
    }));
}


// Both accessors use the same coordinates and count values in every coverage.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = FrequencyAccumulator!(uint[][], A, A)(
        [[0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u],
         [0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));
    static foreach (coverage; [BinCoverage.ordinary, BinCoverage.all])
    {{
        const counts = f.bins!coverage();
        const frequencies = f.frequencyBins!(real, coverage)();
        alias H = HistogramAccumulator!(uint[][], A, A);
        static assert(is(typeof(counts.save) == typeof(H.init.bins!coverage())));
        auto saved = counts.save;
        auto slice = counts[0 .. $];
        f.put(-1.0, 2.0);
        f.put(0.5, 1.5);
        foreach (i; 0 .. counts.length)
        {
            auto c = counts[i];
            auto v = frequencies[i];
            assert(c.count == v.count && saved[i].count == c.count && slice[i].count == c.count);
            static foreach (dimension; 0 .. 2)
            {
                assert(c.isOrdinary!dimension == v.isOrdinary!dimension);
                assert(c.isUnderflow!dimension == v.isUnderflow!dimension);
                assert(c.isOverflow!dimension == v.isOverflow!dimension);
                if (c.isOrdinary!dimension)
                    assert(c.index!dimension == v.index!dimension);
            }
        }
    }}
    static assert(!__traits(compiles, f.bins!double()));
    static assert(!__traits(compiles, f.bins!(cast(BinCoverage) 99)()));
}

// Count-only views can retain owning storage; static-array storage is still borrowed.
version(mir_stat_test_lifetime)
@safe @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto owning()
    {
        auto storage = rcslice!uint(4);
        auto f = FrequencyAccumulator!(typeof(storage), A)(storage, A(2, 0));
        f.put(-1, 0, 2);
        return f.bins!(BinCoverage.all)();
    }
    auto retained = owning();
    assert(retained.front.isUnderflow && retained.front.count == 1);
    assert(retained.back.isOverflow && retained.back.count == 1);

    uint[4] storage;
    alias F = FrequencyAccumulator!(typeof(storage), A);
    auto f = F(storage, A(2, 0));
    const counts = f.bins!(BinCoverage.all)();
    auto saved = counts.save;
    f.put(-1, 1);
    assert(counts.front.count == 1 && saved[2].count == 1);
    static assert(!__traits(compiles, () @safe {
        uint[4] buffer;
        auto local = F(buffer, A(2, 0));
        return local.bins!(BinCoverage.all)();
    }));
    static assert(!__traits(compiles, () @safe {
        uint[4] buffer;
        auto local = F(buffer, A(2, 0));
        return local.bins!(BinCoverage.all)().save;
    }));
    static assert(!__traits(compiles, () @safe {
        uint[4] buffer;
        auto local = F(buffer, A(2, 0));
        return local.bins!(BinCoverage.all)()[0 .. 2];
    }));
}

// Fractional counters start at zero, preserve precision, and keep empty frequencies NaN.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[3][3] data;
        foreach (ref row; data) row[] = 0;
        auto f = FrequencyAccumulator!(typeof(data), A, A)(data, A(1, 0), A(1, 0));
        auto empty = f.marginal!1();
        static assert(is(typeof(empty).CountType == T));
        assert(empty.count == 0 && empty.counts == [T(0), T(0), T(0)]);
        assert(isNaN(empty.frequency!T(0)));
        data[0][2] = T(0.25);
        data[1][1] = T(0.5);
        data[2][0] = T(0.25);
        auto filled = FrequencyAccumulator!(typeof(data), A, A)(data, A(1, 0), A(1, 0));
        auto m = filled.marginal!1();
        assert(m.count == 1 && m.count == filled.count);
        assert(m.underflow == T(0.25) && m.overflow == T(0.25));
        assert(m.frequency!T(0) == T(0.5));
        static assert(!__traits(compiles, filled.marginal!(0, 0)()));
        static assert(!__traits(compiles, filled.marginal!2()));
    }}
}

// A marginal frequency accumulator owns its counts and maintained total.
version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto makeMarginal()
    {
        uint[2][2] data = [[1u, 2u], [3u, 4u]];
        auto f = FrequencyAccumulator!(typeof(data), A, A)(data, A(2, 0), A(2, 0));
        return f.marginal!0();
    }
    auto m = makeMarginal();
    assert(m.count == 10 && m.counts == [3u, 7u]);
    m.put(1);
    assert(m.count == 11 && m.frequency(1) == 8.0 / 11);
}


// Multiple retained axes preserve their requested order and the maintained total.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    uint[24] data;
    auto storage = data[].sliced(4, 3, 2).transposed!(2, 1, 0);
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 4)
                storage[i,j,k] = cast(uint)(1 + i*12 + j*4 + k);
    const f = FrequencyAccumulator!(typeof(storage), A, A, A)(
        storage, A(2, 0), A(3, 10), A(4, 20));
    auto m = f.marginal!(2, 0)();
    assert(m.count == 300 && m.count == f.count);
    assert(m.counts.shape == [4, 2]);
    assert(m.axis!0.bin(0).low == 20 && m.axis!1.bin(0).low == 0);
    assert(m.frequency(0, 0) == 15.0 / 300);
    m.put(20, 0);
    assert(m.count == 301 && f.count == 300);
}


// Read-only storage permits frequency reductions without permitting count mutation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    void check(S)(S storage)
    {
        auto f = FrequencyAccumulator!(S, A, A)(storage, A(1, 0), A(1, 0));
        static assert(is(typeof(f).CountType == uint));
        assert(f.count == 45 && f.frequency(0, 0) == 5.0 / 45);
        assert(f.underflow == 6 && f.overflow == 24);
        assert(f.underflow!1 == 12 && f.overflow!1 == 18);
        assert(f.underflowFrequency!(double, 1) == 12.0 / 45);
        const frozen = f;
        assert(frozen.count == 45 && frozen.frequency(0, 0) == 5.0 / 45);
        auto marginal = f.marginal!0();
        assert(marginal.count == 45 && marginal.counts == [6u, 15u, 24u]);
        marginal.put(0);
        assert(marginal.count == 46 && f.count == 45);
        static assert(!__traits(compiles, f.put(0, 0)));
        static assert(!__traits(compiles, f.put(f)));
        static assert(!__traits(compiles, { f.counts[0][0] = 0; }));
    }
    static immutable uint[3][3] data = [[1u, 2u, 3u], [4u, 5u, 6u], [7u, 8u, 9u]];
    check(cast(const) data);
    check(data);
    // Static backing avoids borrowing an array of stack-bound row pointers.
    static immutable rows = [data[0][], data[1][], data[2][]];
    check(cast(const) rows);
    check(rows);
    const uint[9] backing = [1u, 4u, 7u, 2u, 5u, 8u, 3u, 6u, 9u];
    check(backing[].sliced(3, 3).transposed);
}

// The same value-type rule applies to one-dimensional cumulative sums.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    void check(S)(S storage)
    {
        auto f = FrequencyAccumulator!(S, A)(storage, A(2, 0));
        static assert(is(typeof(f).CountType == uint));
        assert(f.count == 10 && f.underflow == 1 && f.overflow == 4);
        assert(f.frequency(0) == 0.2 && f.cumulativeFrequency(1) == 0.6);
        double[2] output;
        f.cumulativeFrequencies(output[]);
        assert(output == [0.3, 0.6]);
        assert(f.cumulativeFrequencies!double() == output[]);
        static assert(!__traits(compiles, f.put(0)));
    }
    const uint[4] data = [1u, 2u, 3u, 4u];
    immutable uint[4] fixedData = data;
    check(data[]);
    check(fixedData[]);
    check(data[].sliced);
}

// Coverage, output types, const cursors, and empty-total behavior.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import std.range.primitives: isForwardRange, isRandomAccessRange;
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    static foreach (under; [false, true])
    static foreach (over; [false, true])
    {{
        alias A = IntegralAxis!(uint, int, AxisOptions(false, over, under));
        uint[] counts = new uint[2 + under + over];
        counts[] = 1;
        const f = FrequencyAccumulator!(uint[], A)(counts, A(2, 0));
        static foreach (F; AliasSeq!(float, double, real))
        {{
            const fixed = f.cumulativeFrequencyBins!F();
            auto cursor = fixed.save;
            static assert(isForwardRange!(typeof(cursor)));
            static assert(!isRandomAccessRange!(typeof(cursor)));
            assert(cursor.length == 2);
            assert(cursor.front.cumulativeCount == 1 + under);
            cursor.popFront();
            assert(cursor.front.cumulativeFrequency == f.cumulativeFrequency!F(1));
            assert(fixed.front.index == 0);
            cursor.popFront();
            assert(cursor.empty);
            assertThrown!AssertError(cursor.front);
            assertThrown!AssertError(cursor.popFront());
        }}
        auto all = f.cumulativeFrequencyBins!(double, BinCoverage.all)();
        uint sum;
        foreach (entry; all)
        {
            ++sum;
            assert(entry.cumulativeCount == sum);
            assert(entry.cumulativeFrequency == cast(double) sum / counts.length);
        }
        counts[] = 0;
        auto zero = FrequencyAccumulator!(uint[], A)(counts, A(2, 0));
        foreach (entry; zero.cumulativeFrequencyBins!(double, BinCoverage.all)())
            assert(entry.cumulativeCount == 0 && isNaN(entry.cumulativeFrequency));
    }}
    alias A = IntegralAxis!(uint, int, AxisOptions());
    alias Joint = FrequencyAccumulator!(uint[][], A, A);
    static assert(!__traits(compiles, Joint.init.cumulativeFrequencyBins()));
    alias F = FrequencyAccumulator!(uint[], A);
    static assert(!__traits(compiles, F.init.cumulativeFrequencyBins!int()));
}

// No GC allocations during traversal, including owning and strided storage.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static immutable uint[3] initial = [1, 2, 1];
    auto exercise = () @nogc {
        alias A = IntegralAxis!(uint, int, AxisOptions());
        auto counts = rcslice!uint(initial[]);
        auto f = FrequencyAccumulator!(typeof(counts), A)(counts, A(3, 0));
        auto cursor = f.cumulativeFrequencyBins();
        auto saved = cursor.save;
        cursor.popFront();
        assert(cursor.front.cumulativeCount == 3 && saved.front.cumulativeCount == 1);
        cursor.popFront();
        assert(cursor.front.cumulativeFrequency == 1);
    };
    static if (hasBorrowEscapeChecking)
    {
        scope auto safeExercise = () @safe @nogc { exercise(); };
        safeExercise();
    }
    else
        exercise();

    uint[5] data = [1, 99, 2, 99, 1];
    auto strided = Slice!(uint*, 1, SliceKind.universal)([3], [2], data.ptr);
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto f = FrequencyAccumulator!(typeof(strided), A)(strided, A(3, 0));
    auto cursor = f.cumulativeFrequencyBins();
    cursor.popFront(); cursor.popFront();
    assert(cursor.front.cumulativeCount == 4);
}

version(mir_stat_test_lifetime)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    alias F = FrequencyAccumulator!(uint[], A);
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 2u], A(2, 0));
        return f.cumulativeFrequencyBins();
    }));
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 2u], A(2, 0));
        return f.cumulativeFrequencyBins().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto cursor = F([1u, 2u], A(2, 0)).cumulativeFrequencyBins();
        auto entry = cursor.front;
    }));
    static assert(!__traits(compiles, () @safe {
        uint[2] data = [1, 2];
        auto f = F(data[], A(2, 0));
        auto cursor = f.cumulativeFrequencyBins();
        auto entry = cursor.front;
    }));
}

// Borrowed frequency reads, snapshots, and formatting preserve all four
// attributes when escape checking is enabled. Owning count storage keeps
// construction independent of the lifetime of a caller's stack buffer.
version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.appender: scopedBuffer;
    import mir.format: print;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    static immutable uint[4] initial = [1, 2, 3, 2];
    auto counts = rcslice!uint(initial[]);
    auto f = FrequencyAccumulator!(typeof(counts), A)(counts, A(2, 0));

    assert(f.count == 8 && f.frequency!float(0) == 0.25f);
    assert(f.underflowFrequency() == 0.125 && f.overflowFrequency() == 0.25);

    // Snapshot allocation uses reference-counted storage, not the GC.
    auto snapshot = f.cumulativeFrequencies!float();
    double[2] destination;
    f.cumulativeFrequencies(destination[]);
    assert(snapshot[0] == 0.375f && snapshot[1] == 0.75f);
    assert(destination[0] == snapshot[0] && destination[1] == snapshot[1]);

    {
        auto bins = f.frequencyBins!float();
        auto writer = scopedBuffer!(char, 256);
        print(writer, bins.front);
        assert(writer.data == "bin(low=0, high=1): count=2, frequency=0.25");

        auto cumulative = f.cumulativeFrequencyBins!(double, BinCoverage.all)();
        assert(cumulative.front.isUnderflow);
        cumulative.popFront();
        auto cumulativeWriter = scopedBuffer!(char, 256);
        print(cumulativeWriter, cumulative.front);
        assert(cumulativeWriter.data ==
            "bin(low=0, high=1): count=2, cumulativeCount=3, cumulativeFrequency=0.375");
        cumulative.popFront();
        cumulative.popFront();
        assert(cumulative.front.isOverflow && cumulative.front.cumulativeFrequency == 1);
    }

    // Finish cumulative traversal before changing the source. Saved snapshots
    // remain independent, while a newly borrowed range sees the updated counts.
    f.put(0);
    assert(snapshot[0] == 0.375f && snapshot[1] == 0.75f);
    assert(f.frequencyBins().front.count == 3 && f.count == 9);
}

// Joint traversal and marginalization also preserve the complete attribute set;
// the marginal owns independent counts, while views read the original source.
version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto counts = rcslice!uint(4, 4);
    auto f = FrequencyAccumulator!(typeof(counts), A, A)(counts, A(2, 0), A(2, 0));
    f.put(-1, -1);
    f.put(0, 1);
    f.put(1, 0);
    f.put(2, 2);
    assert(f.count == 4 && f.frequency!float(0, 1) == 0.25f);
    assert(f.underflow!0() == 1 && f.overflow!1() == 1);

    auto bins = f.frequencyBins!(float, BinCoverage.all)();
    auto saved = bins.save;
    auto middle = bins[5 .. 11];
    assert(bins.length == 16 && bins.front.frequency == 0.25f);
    assert(bins.front.isUnderflow!0 && bins.front.isUnderflow!1);
    assert(bins.back.isOverflow!0 && bins.back.isOverflow!1);
    bins.popFront();
    saved.popBack();
    assert(bins.length == 15 && saved.length == 15 && middle.length == 6);
    assert(f.bins!(BinCoverage.all)()[6].count == 1);

    auto marginal = f.marginal!0();
    assert(marginal.count == 4 && marginal.frequency(0) == 0.25);
    marginal.put(0);
    assert(marginal.count == 5 && f.count == 4);
    assert(f.frequency(0, 1) == 0.25);
}

// Small count types retain their type instead of exposing integer promotion.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(ubyte, int, AxisOptions());
    auto f = FrequencyAccumulator!(ubyte[], A)([cast(ubyte) 1, 2], A(2, 0));
    auto cursor = f.cumulativeFrequencyBins();
    static assert(is(typeof(cursor.front.cumulativeCount) == ubyte));
    cursor.popFront();
    assert(cursor.front.cumulativeCount == 3 && cursor.front.cumulativeFrequency == 1);
    typeof(cursor) empty;
    assert(empty.empty && empty.length == 0);
}

// Floating-point count storage starts accumulation at zero, not its NaN init.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto f = FrequencyAccumulator!(double[], A)([0.5, 1.5], A(2, 0));
    auto cursor = f.cumulativeFrequencyBins();
    assert(cursor.front.cumulativeCount == 0.5);
    assert(cursor.front.cumulativeFrequency == 0.25);
    cursor.popFront();
    assert(cursor.front.cumulativeCount == 2);
    assert(cursor.front.cumulativeFrequency == 1);
}
