/++
This module contains algorithms for relative frequency statistics.

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


module mir.stat.descriptive.histogram.relative_frequency;

import mir.internal.utility: isFloatingPoint;
import std.meta: allSatisfy;
import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
public import mir.stat.descriptive.histogram.accumulator: BinCoverage;
import mir.stat.descriptive.histogram.traits: isAxis, includeUnderflow;
import mir.stat.descriptive.histogram.internal.view: supportsBinView;
private import mir.stat.descriptive.histogram.internal.density: supportsDensityAxis;
import mir.stat.descriptive.histogram.internal.projection: validMarginalAxes;
import mir.stat.internal.borrow: hasBorrowEscapeChecking, uncheckedBorrow;

// Limit destinations to writable floating-point arrays and one-dimensional slices.
private template isRelativeFrequencyDestination(Destination)
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
        enum isRelativeFrequencyDestination =
            isFloatingPoint!(Unqual!(DeepElementType!Destination)) &&
            __traits(compiles, {
                Destination destination;
                destination[0] = Unqual!(DeepElementType!Destination).init;
            });
    else
        enum isRelativeFrequencyDestination = false;
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
struct RelativeFrequencyAccumulator(Storage, Axis...)
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
    Sum over discarded axes to create a marginal relative frequency accumulator.

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
            return RelativeFrequencyAccumulator!Args(projected.counts, projected.axis);
    }

    /++
    Read-only bin descriptions and counts, without per-entry relative frequencies.
    Uses the same coverage options and element type as HistogramAccumulator.bins.
    Owning storage handles keep counts alive independently of this accumulator;
    borrowed storage and axis boundaries must remain alive. Keep shared storage
    shape and axis boundaries unchanged. Use relativeFrequencyBins to also read relative frequencies.

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
    Borrow a read-only random-access view of bins and relative frequencies.
    Coverage defaults to ordinary bins; BinCoverage.all includes enabled
    underflow/overflow bins. The last axis advances fastest.

    Each element reads its count and relative frequency from this accumulator. The
    denominator includes enabled flow bins. RelativeFrequencyType defaults to double;
    relative frequencies are NaN when the total is zero.

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
        RelativeFrequencyType = floating-point output type
        coverage = ordinary bins or all enabled stored bins
    +/
    auto relativeFrequencyBins(RelativeFrequencyType = double, BinCoverage coverage = BinCoverage.ordinary)() return const
        if (isFloatingPoint!RelativeFrequencyType && supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        static if (!hasBorrowEscapeChecking)
            uncheckedBorrow();
        return RelativeFrequencyBinView!(Storage, RelativeFrequencyType, coverage, Axis)(&this);
    }

    /++
    Borrow a one-dimensional forward range of cumulative relative frequencies.

    Entries expose count, cumulativeCount, cumulativeRelativeFrequency, and the usual
    bin description and classification. Coverage defaults to ordinary bins;
    enabled underflow always contributes to cumulative counts. BinCoverage.all
    also emits enabled underflow/overflow entries. The denominator includes all
    recorded counts; a zero total produces NaNs. Categorical axes follow bin order.

    Traversal takes linear time and constant auxiliary storage, without allocating.
    Reading front does not advance accumulation. Saved cursors move independently.
    The accumulator must remain alive, in place, and unchanged until all cursors
    are finished. Returned descriptions may borrow axis storage. The borrowing
    and @safe restrictions of $(LREF relativeFrequencyBins) also apply here.

    Params:
        RelativeFrequencyType = floating-point output type; defaults to double
        coverage = ordinary bins or all enabled stored bins
    +/
    auto cumulativeRelativeFrequencyBins(RelativeFrequencyType = double,
        BinCoverage coverage = BinCoverage.ordinary)() return const
        if (N == 1 && isFloatingPoint!RelativeFrequencyType && supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        static if (!hasBorrowEscapeChecking)
            uncheckedBorrow();
        return CumulativeRelativeFrequencyBinView!(Storage, RelativeFrequencyType, coverage, Axis)(&this);
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
    Returns `RelativeFrequencyType.nan` when the total count is zero.

    Params:
        RelativeFrequencyType = floating-point output type; defaults to double
        index = ordinary bin indices, one per axis, each less than its N_bin
    +/
    RelativeFrequencyType relativeFrequency(RelativeFrequencyType = double, Indices...)(Indices index) const
        if (isFloatingPoint!RelativeFrequencyType && Indices.length == N &&
            allSatisfy!(isIndex, Indices))
    {
        size_t[N] indices;
        static foreach (i; 0 .. N)
        {
            assert(index[i] >= 0 && index[i] < histogramAccumulator.axis[i].N_bin,
                "RelativeFrequencyAccumulator.relativeFrequency: index is out of range");
            indices[i] = cast(size_t) index[i];
            indices[i] += includeUnderflow!(Axis[i]);
        }
        return normalizeCount!RelativeFrequencyType(storageCount(counts, indices));
    }

    /++
    Probability density in an ordinary numeric bin.
    Divides relative frequency by the product of actual bin widths, measured
    in the original input coordinates, including for transformed axes.
    Enabled underflow/overflow counts remain in the total, so integrating over
    ordinary bins can give less than one. A zero total produces NaN.

    All axes must expose numeric interval boundaries. Boundaries and widths
    must be finite, with positive widths; assertions check this on access.
    Categorical bins and underflow/overflow densities are not supported.
    Intermediate geometry uses real precision and a scaled volume; the result
    is rounded to DensityType and may underflow or overflow in that type.

    Params:
        DensityType = floating-point output type; defaults to double
        index = ordinary bin indices, one per axis
    +/
    DensityType density(DensityType = double, Indices...)(Indices index) const
        if (isFloatingPoint!DensityType && Indices.length == N &&
            allSatisfy!(isIndex, Indices) && allSatisfy!(supportsDensityAxis, Axis))
    {
        import mir.stat.descriptive.histogram.internal.density: ScaledBinVolume;
        // Validate indices and obtain normalization through the existing API.
        const frequency = relativeFrequency!real(index);
        ScaledBinVolume volume;
        static foreach (i; 0 .. N)
            volume.include(histogramAccumulator.axis[i].bin(cast(size_t) index[i]));
        return volume.normalize!DensityType(frequency);
    }

    /++
    Borrow a random-access view of ordinary bins with densities.
    Entries expose count, relativeFrequency, density, and bin coordinates.
    Uses the same numeric geometry as $(LREF density). Saved and sliced cursors
    share live counts and totals while retaining independent positions.
    All lifetime, const, and @safe restrictions of $(LREF relativeFrequencyBins)
    apply. Only ordinary bins are exposed; no coverage option is provided.

    Params:
        DensityType = floating-point output type; defaults to double
    +/
    auto densityBins(DensityType = double)() return const
        if (isFloatingPoint!DensityType && supportsBinView!(Storage, Axis) &&
            allSatisfy!(supportsDensityAxis, Axis))
    {
        // Keep relative frequencies in real precision until density is computed.
        auto bins = relativeFrequencyBins!real();
        return DensityBinView!(Storage, DensityType, Axis)(bins);
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
    cumulative relative frequency below one. Returns `RelativeFrequencyType.nan` when the total
    is zero. Each call takes time proportional to index + 1 and stores no
    additional cumulative state. For categorical axes, accumulation follows
    the axis's bin order.

    Params:
        RelativeFrequencyType = floating-point output type; defaults to double
        index = ordinary bin index, less than axis.N_bin
    +/
    RelativeFrequencyType cumulativeRelativeFrequency(RelativeFrequencyType = double)(size_t index) const
        if (N == 1 && isFloatingPoint!RelativeFrequencyType)
    {
        assert(index < axis.N_bin,
            "RelativeFrequencyAccumulator.cumulativeRelativeFrequency: index is out of range");
        CountType cumulative = 0;
        static if (includeUnderflow!AxisType)
            cumulative = histogramAccumulator.underflow;
        foreach (i; 0 .. index + 1)
            cumulative += histogramAccumulator.counts[i + includeUnderflow!AxisType];
        return normalizeCount!RelativeFrequencyType(cumulative);
    }

    /++
    One-dimensional snapshot of cumulative relative frequencies for all ordinary bins.

    Returns a newly allocated reference-counted Mir slice with one value per
    ordinary bin, in axis order. Each value has the same meaning as
    $(LREF cumulativeRelativeFrequency): enabled underflow contributes to the numerator,
    and overflow contributes only to the denominator. No flow entries are
    appended. All values are `RelativeFrequencyType.nan` when the total is zero.

    Computes the result in one pass, using linear time and output storage.
    The accumulator must remain unchanged during the call. The result owns
    separate storage, can outlive the accumulator, and is unaffected by later
    insertions or merges. Changing result values does not change the accumulator.

    Params:
        RelativeFrequencyType = floating-point output type; defaults to double
    +/
    auto cumulativeRelativeFrequencies(RelativeFrequencyType = double)() const
        if (N == 1 && isFloatingPoint!RelativeFrequencyType)
    {
        import mir.ndslice.allocation: mininitRcslice;

        auto result = mininitRcslice!RelativeFrequencyType(axis.N_bin);
        cumulativeRelativeFrequencies(result);
        return result;
    }

    /++
    Write cumulative relative frequencies into caller-supplied storage.

    The destination must be a writable floating-point array or one-dimensional
    Mir slice with exactly one element per ordinary bin. Its element type
    determines the output precision. Existing values are overwritten in one
    pass without allocating output storage. Values have the same meaning as
    $(LREF cumulativeRelativeFrequency), including NaNs when the total is zero.

    The destination must not overlap the accumulator's count storage. The
    accumulator must remain unchanged during the call. No reference to the
    destination is retained; its values are independent of later source updates.

    Params:
        destination = output storage, with length equal to axis.N_bin
    +/
    void cumulativeRelativeFrequencies(Destination)(scope Destination destination) const
        if (N == 1 && isRelativeFrequencyDestination!Destination)
    {
        import mir.primitives: DeepElementType;
        import std.traits: Unqual;

        alias RelativeFrequencyType = Unqual!(DeepElementType!Destination);
        assert(destination.length == axis.N_bin,
            "RelativeFrequencyAccumulator.cumulativeRelativeFrequencies: destination length must match ordinary bin count");
        CountType cumulative = 0;
        static if (includeUnderflow!AxisType)
            cumulative = histogramAccumulator.underflow;
        foreach (i; 0 .. axis.N_bin)
        {
            cumulative += histogramAccumulator.counts[i + includeUnderflow!AxisType];
            destination[i] = normalizeCount!RelativeFrequencyType(cumulative);
        }
    }

    private RelativeFrequencyType normalizeCount(RelativeFrequencyType)(CountType value) const
        if (isFloatingPoint!RelativeFrequencyType)
    {
        if (total == 0)
            return RelativeFrequencyType.nan;
        return cast(RelativeFrequencyType) value / cast(RelativeFrequencyType) total;
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
        RelativeFrequencyType = floating-point output type; defaults to double
        dimension = axis dimension; defaults to zero
    +/
    RelativeFrequencyType overflowRelativeFrequency(RelativeFrequencyType = double, size_t dimension = 0)() const
        if (isFloatingPoint!RelativeFrequencyType && dimension < N && includeOverflow!(Axis[dimension]))
    {
        return normalizeCount!RelativeFrequencyType(overflow!dimension());
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
        RelativeFrequencyType = floating-point output type; defaults to double
        dimension = axis dimension; defaults to zero
    +/
    RelativeFrequencyType underflowRelativeFrequency(RelativeFrequencyType = double, size_t dimension = 0)() const
        if (isFloatingPoint!RelativeFrequencyType && dimension < N && includeUnderflow!(Axis[dimension]))
    {
        return normalizeCount!RelativeFrequencyType(underflow!dimension());
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
        static if (is(Unqual!F == RelativeFrequencyAccumulator!Args, Args...))
            enum acceptsMerge =
                is(Unqual!F == RelativeFrequencyAccumulator!(Args[0], Axis)) &&
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
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u], Axis(3, 0.0));
    assert(f.count == 0);

    // Bins are [0, 1), [1, 2), and [2, 3).
    f.put(0.5);
    f.put([1.0, 1.5, 2.5]);
    assert(f.counts == [1u, 2u, 1u]);
    assert(f.count == 4);

    // Frequencies default to double and divide each bin count by the total.
    assert(f.relativeFrequency(0) == 0.25);
    assert(f.relativeFrequency(1) == 0.5);
    assert(f.relativeFrequency(2) == 0.25);
}

/// Choose the relative frequency output type without changing the accumulator.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    f.put([0.5, 1.5]);

    double relativeFrequency = f.relativeFrequency(0);
    float floatRelativeFrequency = f.relativeFrequency!float(0);
    assert(relativeFrequency == 0.5);
    assert(floatRelativeFrequency == 0.5f);
}

/// Evaluate a runtime rule before constructing a relative frequency accumulator.
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
    auto f = RelativeFrequencyAccumulator!(size_t[], Axis)(new size_t[n], Axis(n, 0.0, 12.0));
    f.put(data);
    assert(f.counts == [3, 4, 2]);
    assert(f.relativeFrequency(0) == 3.0 / 9);
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
    // Compute all boundaries together in independent GC-owned storage.
    auto boundaries = data.quantile(probabilities);
    // Include the sample maximum in the final left-closed, right-open bin.
    boundaries[$ - 1] = nextUp(boundaries[$ - 1]);
    auto axis = variableAxis(boundaries);
    auto f = RelativeFrequencyAccumulator!(size_t[], typeof(axis))(
        new size_t[boundaries.length - 1], axis);
    f.put(data);
    assert(f.count == 8 && f.counts == [2, 2, 2, 2]);

    foreach (i; 0 .. boundaries.length - 1)
    {
        const probability = f.relativeFrequency(i);
        const width = boundaries[i + 1] - boundaries[i];
        const height = probability / width;
        assert(probability == 0.25);
        // Plot this density as bar height: area, not height, represents 25%.
        assert(fabs(height * width - 0.25) < 1e-14);
    }
    // Repeated quantiles from tied data must be combined before constructing
    // the axis. Equal observed counts are not guaranteed for arbitrary data.
    // Additional observations update relative frequencies but do not recompute edges.
}

/// Read counts without borrowing the running total.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 2u, 3u, 0u], A(2, 0.0));

    // bins has the same meaning as on HistogramAccumulator: descriptions
    // and counts, with ordinary-only coverage by default.
    const counts = f.bins();
    assert(counts.length == 2 && counts.front.count == 2);
    static assert(!__traits(hasMember, typeof(counts.front), "relativeFrequency"));

    // Request end bins explicitly. All cursors see shared count updates.
    auto all = f.bins!(BinCoverage.all)();
    f.put(2.5);
    assert(all.back.isOverflow && all.back.count == 1);
    assert(counts.front.index == 0 && counts.front.bin.low == 0);

    // relativeFrequencyBins is the separate accessor for per-entry relative frequencies.
    // Its views also borrow f's running total; bins does not.
}

/// Read cumulative relative frequencies in bin order and choose the output type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([2u, 3u, 5u], Axis(3, 0.0));

    // Include the selected bin and all earlier bins in the numerator.
    assert(f.cumulativeRelativeFrequency(0) == 0.2);
    assert(f.cumulativeRelativeFrequency(1) == 0.5);
    assert(f.cumulativeRelativeFrequency(2) == 1.0);

    // Select an output type per call, just as with relative frequency.
    float cumulative = f.cumulativeRelativeFrequency!float(1);
    assert(cumulative == 0.5f);

    // Recompute from current counts and total after recording another value.
    f.put(0.5);
    assert(f.cumulativeRelativeFrequency(1) == 6.0 / 11.0);
}

/// Traverse cumulative relative frequencies without allocating a snapshot.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 2u, 1u], A(3, 0));
    auto bins = f.cumulativeRelativeFrequencyBins();
    assert(bins.front.count == 1 && bins.front.cumulativeRelativeFrequency == 0.25);

    // Reading front twice leaves the running count and position unchanged.
    assert(bins.front.cumulativeCount == 1);
    bins.popFront();
    assert(bins.front.index == 1 && bins.front.bin.low == 1);
    assert(bins.front.count == 2 && bins.front.cumulativeCount == 3);

    // A saved cursor starts here, then advances independently of bins.
    auto saved = bins.save;
    saved.popFront();
    assert(saved.front.cumulativeRelativeFrequency == 1.0);
    assert(bins.front.cumulativeRelativeFrequency == 0.75);
    // Keep f alive and unchanged until both cursors are finished.
}

/// Include underflow/overflow entries and select the output precision.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 2u, 3u, 4u], A(2, 0));
    auto all = f.cumulativeRelativeFrequencyBins!(float, BinCoverage.all)();
    static assert(is(typeof(all.front.cumulativeRelativeFrequency) == float));
    assert(all.front.isUnderflow && all.front.cumulativeRelativeFrequency == 0.1f);

    // Underflow contributes even when only ordinary entries are requested.
    auto ordinary = f.cumulativeRelativeFrequencyBins();
    assert(ordinary.front.cumulativeCount == 3);
    assert(ordinary.front.cumulativeRelativeFrequency == 0.3);

    // Advance past underflow and both ordinary bins to reach overflow.
    all.popFront(); all.popFront(); all.popFront();
    assert(all.front.isOverflow && all.front.cumulativeCount == 10);
    assert(all.front.cumulativeRelativeFrequency == 1.0f);
    all.popFront();
    assert(all.empty);
}

/// Collect all cumulative relative frequencies in an independent snapshot.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([2u, 3u, 5u], Axis(3, 0.0));

    // Allocate one double value per ordinary bin, computing all prefixes once.
    auto cumulative = f.cumulativeRelativeFrequencies();
    assert(cumulative == [0.2, 0.5, 1.0]);

    // The snapshot retains its values after the source changes.
    f.put(0.5);
    assert(cumulative == [0.2, 0.5, 1.0]);
    // A new call captures the updated counts and total.
    assert(f.cumulativeRelativeFrequencies() == [3.0 / 11, 6.0 / 11, 1.0]);
}

/// Reuse output storage and infer precision from its element type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([1u, 3u], Axis(2, 0.0));
    auto output = new double[2];
    f.cumulativeRelativeFrequencies(output);
    assert(output == [0.25, 1.0]);

    // Updating the accumulator leaves existing output values unchanged.
    f.put(0.5);
    assert(output == [0.25, 1.0]);
    // Reuse the same buffer to replace them with the current cumulative values.
    f.cumulativeRelativeFrequencies(output);
    assert(output == [0.4, 1.0]);

    // A Mir slice is also accepted; float elements select float precision.
    auto floats = rcslice!float([0.0f, 0.0f]);
    f.cumulativeRelativeFrequencies(floats);
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
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);

    auto cumulative = f.cumulativeRelativeFrequencies!float();
    // Underflow contributes to both prefixes. Overflow stays in the total,
    // so the last ordinary bin ends at 3/4. There are no extra flow entries.
    assert(cumulative == [0.5f, 0.75f]);
    assert(cumulative.length == f.axis.N_bin);

    // Result storage is independent: editing it leaves source counts intact.
    cumulative[0] = 0;
    assert(f.cumulativeRelativeFrequency(0) == 0.5);
}

/// Cumulative relative frequencies include underflow but leave overflow beyond the last bin.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    alias Axis = IntegralAxis!(uint, double,
        AxisOptions(EnableOverflow(true), EnableUnderflow(true)));
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);

    // The first numerator includes one underflow and one ordinary observation.
    assert(f.cumulativeRelativeFrequency(0) == 0.5);
    // Overflow contributes to the total of four, but neither numerator.
    // There is no extra ordinary bin for overflow.
    assert(f.cumulativeRelativeFrequency(1) == 0.75);
}

/// Enabled flow bins contribute to the total used by all relative frequencies.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    alias Axis = IntegralAxis!(uint, double,
        AxisOptions(EnableOverflow(true), EnableUnderflow(true)));
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u, 0u, 0u], Axis(2, 0.0));
    f.put([-1.0, 0.5, 1.5, 3.0]);
    assert(f.counts == [1u, 1u, 1u, 1u]);
    assert(f.underflow == 1);
    assert(f.overflow == 1);
    assert(f.count == 4);

    assert(f.relativeFrequency(0) == 0.25);
    assert(f.relativeFrequency(1) == 0.25);
    assert(f.underflowRelativeFrequency == 0.25);
    assert(f.overflowRelativeFrequency == 0.25);
}

/// Empty accumulators return NaN; unoccupied bins in nonempty ones return zero.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.math: isNaN;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    assert(isNaN(f.relativeFrequency(0)));
    assert(isNaN(f.relativeFrequency(1)));

    f.put(0.5);
    assert(f.relativeFrequency(0) == 1.0);
    assert(f.relativeFrequency(1) == 0.0);
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
    alias F = RelativeFrequencyAccumulator!(typeof(counts), Axis);
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
    assert(f.relativeFrequency(0) == 0.6);
    assert(f.relativeFrequency(1) == 0.4);
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
    auto f = RelativeFrequencyAccumulator!(typeof(storage), A, A)(
        storage, A(2, 0.0), A(2, 0.0));

    // Two coordinates describe one observation, so the total increases once.
    f.put(0.5, 1.5);
    f.put(-1.0, 1.5);
    assert(f.count == 2);

    // RelativeFrequency indices refer to ordinary bins, without the storage offset.
    assert(f.relativeFrequency(0, 1) == 0.5);
    assert(f.relativeFrequency!real(0, 1) == 0.5L);

    // Axis 0 has one underflow; axis 1 has none.
    // The denominator includes both observations.
    assert(f.underflow() == 1);
    assert(f.underflow!1() == 0);
    assert(f.underflowRelativeFrequency() == 0.5);
    assert(f.underflowRelativeFrequency!(float, 1)() == 0.0f);
}

/// Marginal relative frequencies use all recorded counts, including underflow/overflow.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias X = IntegralAxis!(uint, int, AxisOptions());
    alias Y = IntegralAxis!(uint, int, AxisOptions(false, true, true));
    auto joint = RelativeFrequencyAccumulator!(uint[][], X, Y)(
        [[1u, 4u, 2u], [0u, 3u, 1u]], X(2, 0), Y(1, 0));
    const source = joint;
    auto marginal = source.marginal!0();

    // Sum over every position of axis one. The two retained bins represent
    // all eleven observations, including those outside axis one's interval.
    assert(marginal.counts == [7u, 4u]);
    assert(marginal.count == 11 && marginal.count == joint.count);
    assert(marginal.relativeFrequency(0) == 7.0 / 11);
    assert(marginal.relativeFrequency!float(1) == 4.0f / 11);

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
        alias F = RelativeFrequencyAccumulator!(Storage, Axis);
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
    auto f = RelativeFrequencyAccumulator!(size_t[], Axis)([0UL, 0], Axis(2, 0.0));
    static assert(!__traits(compiles, f.overflow()));
    static assert(!__traits(compiles, f.underflow()));
    f.put(only(0.5, 1.5, 1.75));
    assert(f.count == 3 && f.counts == [1, 2]);

    enum Label { first, second }
    alias Categories = CategoryAxis!(uint, Label, AxisOptions(EnableOverflow(true)));
    auto c = RelativeFrequencyAccumulator!(uint[], Categories)([0u, 0u, 0u], Categories());
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
    alias F = RelativeFrequencyAccumulator!(uint[], Axis);
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
    alias F = RelativeFrequencyAccumulator!(uint[], Axis);
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
        alias F = RelativeFrequencyAccumulator!(Storage, Axis);
        auto f = F(emptyCounts, Axis(3, 0.0));
        static assert(is(typeof(f.overflowRelativeFrequency()) == double));
        static assert(is(typeof(f.underflowRelativeFrequency()) == double));
        static assert(!__traits(compiles, f.relativeFrequency!uint(0)));
        static assert(!__traits(compiles, f.overflowRelativeFrequency!uint()));
        static assert(!__traits(compiles, f.underflowRelativeFrequency!uint()));

        static foreach (T; AliasSeq!(float, double, real))
        {
            static assert(is(typeof(f.relativeFrequency!T(0)) == T));
            static assert(is(typeof(f.overflowRelativeFrequency!T()) == T));
            static assert(is(typeof(f.underflowRelativeFrequency!T()) == T));
            foreach (i; 0 .. 3)
                assert(isNaN(f.relativeFrequency!T(i)));
            assert(isNaN(f.overflowRelativeFrequency!T()));
            assert(isNaN(f.underflowRelativeFrequency!T()));
        }

        f.put(0.5);
        assert(f.relativeFrequency(0) == 1.0);
        assert(f.relativeFrequency(1) == 0.0);
        assert(f.overflowRelativeFrequency == 0.0 && f.underflowRelativeFrequency == 0.0);
        f.put([-1.0, 0.75, 4.0]);
        static foreach (T; AliasSeq!(float, double, real))
        {
            assert(f.relativeFrequency!T(0) == 0.5);
            assert(f.relativeFrequency!T(1) == 0.0);
            assert(f.overflowRelativeFrequency!T() == 0.25);
            assert(f.underflowRelativeFrequency!T() == 0.25);
        }

        auto other = F(populatedCounts, Axis(3, 0.0));
        assert(other.relativeFrequency(1) == 1.0);
        assert(other.overflowRelativeFrequency == 0.0);
        f.put(other);
        static foreach (T; AliasSeq!(float, double, real))
        {
            assert(f.relativeFrequency!T(0).approxEqual(cast(T) 1 / 3));
            assert(f.relativeFrequency!T(1).approxEqual(cast(T) 1 / 3));
            assert(f.overflowRelativeFrequency!T().approxEqual(cast(T) 1 / 6));
            assert(f.underflowRelativeFrequency!T().approxEqual(cast(T) 1 / 6));
        }
        // Reading relative frequencies does not mutate either counts or the total.
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
            alias F = RelativeFrequencyAccumulator!(Storage, Axis);
            auto f = F(storage, Axis(3, 0.0));
            auto read(return ref const F source) @safe pure nothrow
            {
                return source.cumulativeRelativeFrequency(1);
            }
            static assert(is(typeof(f.cumulativeRelativeFrequency(0)) == double));
            static assert(!__traits(compiles, f.cumulativeRelativeFrequency!uint(0)));
            static foreach (T; AliasSeq!(float, double, real))
            {
                static assert(is(typeof(f.cumulativeRelativeFrequency!T(0)) == T));
                assert(isNaN(f.cumulativeRelativeFrequency!T(0)));
                assert(isNaN(f.cumulativeRelativeFrequency!T(2)));
            }
            // An occupied later bin leaves the preceding cumulative values zero.
            f.put(2.5);
            assert(read(f) == 0);
            assert(f.cumulativeRelativeFrequency(2) == 1);
            f.put(0.5);
            static if (hasUnderflow) f.put(-1.0);
            static if (hasOverflow) f.put(3.0);
            const uint low = hasUnderflow ? 1 : 0;
            const uint high = hasOverflow ? 1 : 0;
            static foreach (T; AliasSeq!(float, double, real))
            {
                assert(f.cumulativeRelativeFrequency!T(0) == cast(T)(1 + low) / (2 + low + high));
                assert(f.cumulativeRelativeFrequency!T(2) == cast(T)(2 + low) / (2 + low + high));
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
            alias F = RelativeFrequencyAccumulator!(Storage, Axis);
            auto f = F(storage, Axis(3, 0.0));
            static assert(is(typeof(f.cumulativeRelativeFrequencies()) == Slice!(RCI!double)));
            static assert(!__traits(compiles, f.cumulativeRelativeFrequencies!uint()));

            void checkValues(T)(ref const F source)
            {
                auto values = source.cumulativeRelativeFrequencies!T();
                static assert(is(typeof(values) == Slice!(RCI!T)));
                assert(values.length == source.axis.N_bin);
                foreach (i; 0 .. values.length)
                {
                    if (source.count == 0)
                        assert(isNaN(values[i]));
                    else
                        assert(values[i] == source.cumulativeRelativeFrequency!T(i));
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
            auto snapshot = f.cumulativeRelativeFrequencies();
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

            auto fresh = f.cumulativeRelativeFrequencies();
            fresh[0] = -1;
            assert(snapshot[0] == first);
            assert(f.counts[hasUnderflow .. $ - hasOverflow] == [2u, 1u, 2u]);
            assert(f.cumulativeRelativeFrequency(0) >= 0);
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
    alias F = RelativeFrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u], Axis(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Stack output verifies that the overload neither allocates nor escapes.
        void write(ref const F source, scope T[] output) @safe pure nothrow @nogc
        {
            source.cumulativeRelativeFrequencies(output);
        }
        T[2] output;
        write(f, output[]);
        assert(isNaN(output[0]) && isNaN(output[1]));
        auto populated = F([1u, 3u], Axis(2, 0.0));
        write(populated, output[]);
        assert(output[] == [T(0.25), T(1)]);

        auto rcOutput = rcslice!T([T(-1), T(-1)]);
        populated.cumulativeRelativeFrequencies(rcOutput);
        assert(rcOutput == populated.cumulativeRelativeFrequencies!T());

        // Only selected elements are overwritten; the intervening values survive.
        auto backing = [T(-1), T(-1), T(-1), T(-1)];
        auto everyOther = backing.sliced.stride(2);
        populated.cumulativeRelativeFrequencies(everyOther);
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
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    const(double)[] readOnly = [0.0, 0.0];
    immutable(double)[] fixedValues = [0.0, 0.0];
    uint[] integers = [0u, 0u];
    auto matrix = [0.0, 0.0].sliced(1, 2);
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(readOnly)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(readOnly.sliced)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(fixedValues)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(integers)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(matrix)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies(0.0)));
    foreach (length; [0, 1, 3])
    {
        auto output = new double[length];
        output[] = -1;
        assertThrown!AssertError(f.cumulativeRelativeFrequencies(output));
        foreach (value; output) assert(value == -1);
    }
    f.put(0.5);
    auto tooLong = [-1.0, -1.0, -1.0];
    assertThrown!AssertError(f.cumulativeRelativeFrequencies(tooLong.sliced));
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
        auto f = RelativeFrequencyAccumulator!(typeof(counts), Axis)(counts, Axis(2, 0.0));
        return f.cumulativeRelativeFrequencies();
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
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    static assert(!__traits(compiles, f.overflowRelativeFrequency()));
    static assert(!__traits(compiles, f.underflowRelativeFrequency()));
    assertThrown!AssertError(f.relativeFrequency(2));
    assertThrown!AssertError(f.cumulativeRelativeFrequency(2));
    assertThrown!AssertError(f.relativeFrequency(size_t.max));
    assertThrown!AssertError(f.cumulativeRelativeFrequency(size_t.max));
    f.put(0.5);
    assertThrown!AssertError(f.relativeFrequency(2));
    assertThrown!AssertError(f.cumulativeRelativeFrequency(2));
    assert(f.count == 1);
}


/++
A bin's count and cumulative statistics, returned by value.
Check classification before accessing index or bin for underflow/overflow entries.
Changing returned values does not update the accumulator.

Params:
    HistogramElement = underlying one-dimensional histogram bin element
    RelativeFrequencyType = floating-point output type
+/
struct CumulativeRelativeFrequencyBin(HistogramElement, RelativeFrequencyType)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBin;
    static if (is(HistogramElement == HistogramBin!Args, Args...))
        private enum N = Args.length - 1;
    else
        static assert(false, "CumulativeRelativeFrequencyBin requires a HistogramBin element");
    static assert(N == 1, "CumulativeRelativeFrequencyBin requires one axis");
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
            ", cumulativeRelativeFrequency=", cumulativeRelativeFrequency);
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
    RelativeFrequencyType cumulativeRelativeFrequency;
}

/// Format cumulative entries with both the bin count and running statistics.
version(mir_stat_test)
pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));
    auto entries = f.cumulativeRelativeFrequencyBins;
    entries.popFront();
    assert(format("%s", entries.front) ==
        "bin(low=1.0, high=2.0): count=3, cumulativeCount=4, cumulativeRelativeFrequency=1.0");
}

/++
Borrowed forward range created by RelativeFrequencyAccumulator.cumulativeRelativeFrequencyBins.
The source must stay alive, in place, and unchanged throughout traversal.
Saved cursors share the source but keep independent running counts and positions.
A const cursor supports front and save; save returns a mutable cursor.

Params:
    Storage = source count storage
    RelativeFrequencyType = floating-point output type
    coverage = ordinary bins or all enabled stored bins
    Axis = the single source axis type
+/
struct CumulativeRelativeFrequencyBinView(Storage, RelativeFrequencyType, BinCoverage coverage, Axis)
    if (isFloatingPoint!RelativeFrequencyType && supportsBinView!(Storage, Axis) &&
        (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
{
    private alias Accumulator = RelativeFrequencyAccumulator!(Storage, Axis);
    private alias Cursor = RelativeFrequencyBinView!(Storage, RelativeFrequencyType, coverage, Axis);
    private Cursor _bins;
    private Accumulator.CountType _preceding = 0;

    /// Type returned by front.
    alias Element = CumulativeRelativeFrequencyBin!(Cursor.BinView.Element, RelativeFrequencyType);

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
            _bins._source.normalizeCount!RelativeFrequencyType(cumulative));
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
        return CumulativeRelativeFrequencyBinView(_bins.save, _preceding);
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
    RelativeFrequencyType = floating-point output type
+/
struct RelativeFrequencyBin(HistogramElement, RelativeFrequencyType)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBin;
    static if (is(HistogramElement == HistogramBin!Args, Args...))
        private enum N = Args.length - 1;
    else
        static assert(false, "RelativeFrequencyBin requires a HistogramBin element");
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
        print(buffer, ": count=", count, ", relativeFrequency=", relativeFrequency);
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
    RelativeFrequencyType relativeFrequency;
}

/// Format relative frequencies with the count and bin coordinates.
version(mir_stat_test)
pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));
    const entry = f.relativeFrequencyBins.front;
    assert(format("%s", entry) == "bin(low=0.0, high=1.0): count=1, relativeFrequency=0.25");
    // Formatting a saved entry reads its recorded values, not live counts.
    f.put(0.5);
    assert(format("%s", entry) == "bin(low=0.0, high=1.0): count=1, relativeFrequency=0.25");
}

/// Print relative frequency entries with writeln or writefln, or choose precision per field.
version(mir_stat_test)
unittest
{
    import std.stdio: writeln, writefln;
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], A)([1u, 3u], A(2, 0.0));

    // Call printFrequencies(f) to write to stdout. The helper is compiled but
    // deliberately not called here, keeping documentation tests silent.
    void printFrequencies(typeof(f) frequencies)
    {
        // bins() exposes the underlying counts without relative frequency statistics.
        writeln(frequencies.bins());
        writefln("Histogram: %s", frequencies.bins());
        writeln(frequencies.relativeFrequencyBins());
        writefln("Frequencies: %s", frequencies.relativeFrequencyBins());
        foreach (entry; frequencies.relativeFrequencyBins())
        {
            writeln(entry);
            // Format fields individually to control their numeric precision.
            writefln("low=%.2f, high=%.2f: count=%s, relativeFrequency=%.2f",
                entry.bin.low, entry.bin.high, entry.count, entry.relativeFrequency);
        }
        // Cumulative entries also include running counts and relative frequencies.
        writefln("Cumulative frequencies: %s", frequencies.cumulativeRelativeFrequencyBins());
    }

    // Check the corresponding text without performing console I/O.
    enum expectedHistogram = "[bin(low=0.0, high=1.0): count=1, " ~
        "bin(low=1.0, high=2.0): count=3]";
    assert(format("%s", f.bins()) == expectedHistogram);
    assert(format("Histogram: %s", f.bins()) == "Histogram: " ~ expectedHistogram);
    enum expected = "[bin(low=0.0, high=1.0): count=1, relativeFrequency=0.25, " ~
        "bin(low=1.0, high=2.0): count=3, relativeFrequency=0.75]";
    assert(format("%s", f.relativeFrequencyBins()) == expected);
    assert(format("Frequencies: %s", f.relativeFrequencyBins()) == "Frequencies: " ~ expected);
    auto entry = f.relativeFrequencyBins().front;
    assert(format("low=%.2f, high=%.2f: count=%s, relativeFrequency=%.2f",
        entry.bin.low, entry.bin.high, entry.count, entry.relativeFrequency) ==
        "low=0.00, high=1.00: count=1, relativeFrequency=0.25");
    assert(format("Cumulative frequencies: %s", f.cumulativeRelativeFrequencyBins()) ==
        "Cumulative frequencies: [bin(low=0.0, high=1.0): count=1, cumulativeCount=1, cumulativeRelativeFrequency=0.25, " ~
        "bin(low=1.0, high=2.0): count=3, cumulativeCount=4, cumulativeRelativeFrequency=1.0]");
}

// Mir formatting preserves GC-free output for all relative frequency precisions.
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
    auto f = RelativeFrequencyAccumulator!(uint[], A)(storage[], A(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {
        checkEntry(f.relativeFrequencyBins!T.front,
            "bin(low=0.0, high=1.0): count=1, relativeFrequency=0.25");
        checkEntry(f.cumulativeRelativeFrequencyBins!T.front,
            "bin(low=0.0, high=1.0): count=1, cumulativeCount=1, cumulativeRelativeFrequency=0.25");
    }
}

/++
Borrowed random-access range created by RelativeFrequencyAccumulator.relativeFrequencyBins.

The range refers to the original accumulator, including its running total.
It does not copy the total or retain separate count storage. Updates through
that accumulator are visible on later reads. Each returned count and relative frequency
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
    RelativeFrequencyType = floating-point output type
    coverage = ordinary bins or all enabled stored bins
    Axis = source axis types
+/
struct RelativeFrequencyBinView(Storage, RelativeFrequencyType, BinCoverage coverage, Axis...)
    if (isFloatingPoint!RelativeFrequencyType && supportsBinView!(Storage, Axis) &&
        (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
{
    import mir.stat.descriptive.histogram.accumulator: HistogramBinView;
    private alias Accumulator = RelativeFrequencyAccumulator!(Storage, Axis);
    private alias BinView = HistogramBinView!(Storage, coverage, Axis);
    private const(Accumulator)* _source;
    private size_t[Axis.length] _shape;
    private size_t _begin, _end, _outerLength;

    /// Type returned by element access.
    alias Element = RelativeFrequencyBin!(BinView.Element, RelativeFrequencyType);

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
        assert(_source !is null, "RelativeFrequencyBinView: uninitialized view");
        assert(_source.counts.length == _outerLength,
            "RelativeFrequencyBinView: source storage shape changed while borrowed");
        static foreach (i; 0 .. Axis.length)
            assert(_source.axis!i.N_bin == _shape[i],
                "RelativeFrequencyBinView: source bin count changed while borrowed");
    }

    /// Number of remaining bins in the selected coverage.
    size_t length() const @property { return _end - _begin; }

    /// Whether the cursor is exhausted.
    bool empty() const @property { return _begin == _end; }

    /// First remaining element, returned by value.
    Element front() const @property
    {
        assert(!empty, "RelativeFrequencyBinView.front: empty range");
        return this[0];
    }

    /// Last remaining element, returned by value.
    Element back() const @property
    {
        assert(!empty, "RelativeFrequencyBinView.back: empty range");
        return this[length - 1];
    }

    /// Advance the cursor without changing the accumulator.
    void popFront()
    {
        assert(!empty, "RelativeFrequencyBinView.popFront: empty range");
        ++_begin;
    }

    /// Remove the last bin from this cursor's range.
    void popBack()
    {
        assert(!empty, "RelativeFrequencyBinView.popBack: empty range");
        --_end;
    }

    /// Copy the cursor, borrowing the same source.
    auto save() const @property
    {
        return RelativeFrequencyBinView(_source, _begin, _end, _shape, _outerLength);
    }

    /// Read a bin, count, and relative frequency from the same accumulator.
    Element opIndex(size_t index) const
    {
        checkSource();
        assert(index < length, "RelativeFrequencyBinView: index is out of range");
        auto entry = BinView.readElement(_source.counts, _begin + index,
            _shape, _source.histogramAccumulator.axis);
        return Element(entry, entry.count,
            _source.normalizeCount!RelativeFrequencyType(entry.count));
    }

    /// Slice relative to the cursor; entry indices remain original bin indices.
    auto opSlice(size_t begin, size_t end) const
    {
        assert(begin <= end && end <= length,
            "RelativeFrequencyBinView: slice is out of range");
        return RelativeFrequencyBinView(_source, _begin + begin, _begin + end, _shape, _outerLength);
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
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([0u, 0u], Axis(2, 0.0));
    f.put([0.5, 1.25, 1.5, 1.75]);

    // f stays alive and in place for the entire traversal.
    foreach (entry; f.relativeFrequencyBins())
    {
        assert(entry.bin.low == entry.index);
        assert(entry.relativeFrequency == cast(double) entry.count / 4);
    }
}

/// Select an output type independently of the accumulator.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([1u, 1u], Axis(2, 0.0));
    auto bins = f.relativeFrequencyBins!float();
    static assert(is(typeof(bins.front.relativeFrequency) == float));
    assert(bins.front.relativeFrequency == 0.5f);
}

/// Const views share live counts and totals while cursors move independently.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], Axis)([1u, 1u], Axis(2, 0.0));

    // const fixes this view's traversal position; it does not freeze f's data.
    // f must stay alive and in place while fixed and its derived cursors exist.
    const fixed = f.relativeFrequencyBins();

    // An entry captures count and relative frequency values at the time it is read.
    // Initially, the first bin contains one of the two recorded observations.
    auto previous = fixed.front;

    // Adding 0.5 through f increments the first bin to 2 and the total to 3.
    // fixed reads both updated values from f. previous keeps its earlier values.
    f.put(0.5);
    assert(fixed.front.count == 2);
    assert(fixed.front.relativeFrequency == 2.0 / 3);
    assert(previous.relativeFrequency == 0.5);

    // save creates a mutable cursor at fixed's current position.
    // Advancing cursor skips the first bin without moving fixed.
    auto cursor = fixed.save;
    cursor.popFront();
    assert(cursor.front.relativeFrequency == 1.0 / 3);

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
    auto f = RelativeFrequencyAccumulator!(uint[][], A, A)(
        [[0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u],
         [0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));
    f.put(0.5, 1.5);
    f.put(-1.0, 1.5);

    // Only the four ordinary joint bins appear. Axis 1 advances fastest.
    // Underflow still contributes to the total used for every relative frequency.
    const bins = f.relativeFrequencyBins!real();
    assert(bins.length == 4);
    auto entry = bins[1];
    assert(entry.index == 0 && entry.index!1 == 1);
    assert(entry.bin.low == 0.0 && entry.bin!1.low == 1.0);
    assert(entry.count == 1 && entry.relativeFrequency == 0.5L);

    // Reading the same position again sees the new count and total.
    // The earlier entry remains a snapshot of the values it read.
    f.put(0.5, 1.5);
    assert(bins[1].count == 2 && bins[1].relativeFrequency == 2.0L / 3);
    assert(entry.count == 1 && entry.relativeFrequency == 0.5L);

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
    auto f = RelativeFrequencyAccumulator!(uint[][], A, A)(
        [[0u, 3u, 0u, 2u], [0u, 0u, 0u, 0u],
         [0u, 5u, 0u, 4u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));

    // Axis totals overlap: the two observations at (underflow, overflow)
    // contribute to both totals.
    assert(f.underflow == 5 && f.overflow!1 == 6);
    uint outside;
    double sum = 0;
    foreach (entry; f.relativeFrequencyBins!(double, BinCoverage.all)())
    {
        if (!entry.isOrdinary || !entry.isOrdinary!1)
            outside += entry.count;
        sum += entry.relativeFrequency;
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
        alias F = RelativeFrequencyAccumulator!(Storage, Axis);
        auto f = F(counts, Axis(3, 0.0));
        auto bins = f.relativeFrequencyBins!T();
        alias View = typeof(bins);
        static assert(isRandomAccessRange!View && hasSlicing!View);
        static assert(!hasAssignableElements!View);
        static assert(is(typeof(bins.front.relativeFrequency) == T));
        static assert(!__traits(compiles, {
            bins._source.counts[0] = 10u;
        }));
        assert(isNaN(bins[0].relativeFrequency));
        f.put([-1.0, 0.5, 0.75, 2.5, 4.0]);
        assert(bins.length == 3 && f.count == 5);
        assert(bins[0].count == 2 && bins[0].relativeFrequency == cast(T) 2 / 5);
        assert(bins[1].count == 0 && bins[1].relativeFrequency == 0);
        assert(bins[2].relativeFrequency == cast(T) 1 / 5);
        auto previous = bins.front;
        auto other = F(otherCounts, Axis(3, 0.0));
        other.put([-1.0, 1.5]);
        f.put(other);
        assert(f.count == 7 && bins.front.relativeFrequency.approxEqual(cast(T) 2 / 7));
        assert(previous.relativeFrequency == cast(T) 2 / 5);
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
        assert(tail.front.relativeFrequency == cast(T) 1 / 8);
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
        auto f = RelativeFrequencyAccumulator!(uint[], Axis)([1u, 3u], axis);
        const reader = f;
        auto bins = reader.relativeFrequencyBins();
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
            assert(bins[i].relativeFrequency == (i == 0 ? 0.25 : 0.75));
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
    alias F = RelativeFrequencyAccumulator!(uint[], Axis);
    auto f = F([0u, 0u], Axis(2, 0.0));
    auto bins = f.relativeFrequencyBins();
    alias View = typeof(bins);
    static assert(!__traits(compiles, f.relativeFrequencyBins!uint()));
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
    alias F = RelativeFrequencyAccumulator!(uint[], Axis);
    enum safeBorrowCompiles = __traits(compiles, () @safe {
        auto f = F([1u, 1u], Axis(2, 0.0));
        auto bins = f.relativeFrequencyBins();
        assert(bins.front.relativeFrequency == 0.5);
    });
    static assert(safeBorrowCompiles == hasBorrowEscapeChecking);

    // Safe code must never return a view of a local accumulator. Without escape
    // checking, borrowing itself is @system; with it, the return is rejected.
    // Older compilers may permit this escape in @system code.
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 1u], Axis(2, 0.0));
        return f.relativeFrequencyBins();
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
            alias F = RelativeFrequencyAccumulator!(Storage, Axis);
            auto borrow(return ref const F f) { return f.relativeFrequencyBins(); }
            auto saved(return ref const F f)
            {
                const bins = f.relativeFrequencyBins();
                return bins.save;
            }
            auto sliced(return ref const F f)
            {
                const bins = f.relativeFrequencyBins();
                return bins[0 .. 1];
            }
            auto f = F(counts, Axis(2, 0.0));
            auto bins = borrow(f);
            auto copy = saved(f);
            auto part = sliced(f);
            f.put(0.5);
            assert(bins.front.relativeFrequency == 2.0 / 3);
            assert(copy.front.relativeFrequency == 2.0 / 3);
            assert(part.front.relativeFrequency == 2.0 / 3);
            assert(bins.map!(e => e.count).equal([2u, 1u]));

            // Numeric entries contain values and may outlive the borrowed source.
            auto entry()
            {
                auto local = F(counts, Axis(2, 0.0));
                return local.relativeFrequencyBins().front;
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
        auto f = RelativeFrequencyAccumulator!(uint[], Variable)(
            counts, Variable(breaks.sliced));
        auto bins = f.relativeFrequencyBins();
        assert(bins[1].bin.low == 1 && bins[1].bin.high == 4);
        assert(bins[1].relativeFrequency == 0.75);
    }

    unittest
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
        alias Axis = IntegralAxis!(uint, double, AxisOptions());
        alias F = RelativeFrequencyAccumulator!(uint[], Axis);
        alias View = RelativeFrequencyBinView!(uint[], double, BinCoverage.ordinary, Axis);
        static assert(!__traits(compiles, () @safe {
            auto bins = F([1u, 1u], Axis(2, 0.0)).relativeFrequencyBins();
            auto value = bins.front;
        }));
        static assert(!__traits(compiles, () @safe {
            auto f = F([1u, 1u], Axis(2, 0.0));
            auto bins = f.relativeFrequencyBins();
            return bins.save;
        }));
        static assert(!__traits(compiles, () @safe {
            auto f = F([1u, 1u], Axis(2, 0.0));
            const bins = f.relativeFrequencyBins();
            return bins[0 .. 1];
        }));
        static assert(!__traits(compiles, () @safe {
            View bins;
            {
                auto f = F([1u, 1u], Axis(2, 0.0));
                bins = f.relativeFrequencyBins();
            }
            auto value = bins.front;
        }));
        static assert(!__traits(compiles, () @safe {
            static View escaped;
            auto f = F([1u, 1u], Axis(2, 0.0));
            escaped = f.relativeFrequencyBins();
        }));
        static assert(!__traits(compiles, () @safe {
            auto forward(return ref const F f) { return f.relativeFrequencyBins().save; }
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
            auto f = RelativeFrequencyAccumulator!(uint[], Variable)(
                counts[], Variable(breaks[].sliced));
            auto bins = f.relativeFrequencyBins();
            auto entry = bins.front;
        }));
    }
}

// RelativeFrequency reads, merges, and owning cumulative snapshots remain @nogc.
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
    alias F = RelativeFrequencyAccumulator!(typeof(counts), A);
    auto f = F(counts, A(2, 0.0));
    f.put(samples[]);
    auto other = F(rcslice!uint(zero[]), A(2, 0.0));
    other.put(samples[]);
    f.put(other);
    void read(ref const F source) @safe pure nothrow @nogc
    {
        assert(source.count == 8);
        assert(source.underflow == 2 && source.overflow == 2);
        assert(source.underflowRelativeFrequency!float() == 0.25f);
        assert(source.overflowRelativeFrequency!real() == 0.25L);
        assert(source.relativeFrequency(0) == 0.25);
        assert(source.cumulativeRelativeFrequency(1) == 0.75);
        auto snapshot = source.cumulativeRelativeFrequencies();
        assert(snapshot[0] == 0.5 && snapshot[1] == 0.75);
        double[2] output;
        source.cumulativeRelativeFrequencies(output[]);
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
        auto f = RelativeFrequencyAccumulator!(typeof(counts), A)(counts, A(2, 0.0));
        auto view = f.relativeFrequencyBins();
        auto saved = view.save;
        auto tail = view[1 .. 2];
        view.popFront(); saved.popBack();
        assert(view.front.relativeFrequency == 0.5 && saved.back.relativeFrequency == 0.5);
        f.put(1.5);
        assert(tail.front.relativeFrequency == 2.0 / 3.0);
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

// Variadic relative frequency insertion validates every type and counts every observation.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = RelativeFrequencyAccumulator!(uint[], A)([0u, 0u, 0u, 0u], A(2, 0.0));
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
    auto categories = RelativeFrequencyAccumulator!(uint[], C)([0u, 0u, 0u], C());
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
    auto f = RelativeFrequencyAccumulator!(uint[], A)([0u, 0u], A(2, 0.0));
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
    auto f = RelativeFrequencyAccumulator!(typeof(initial), A, A)(
        initial, A(2, 0.0), A(2, 0.0));
    assert(f.count == 5);
    assert(f.underflow == 2 && f.overflow!1 == 2);
    assert(f.relativeFrequency(0, 1) == 0.6);
    assert(f.axis!1.N_bin == 2);
    static assert(is(typeof(f.axis!1()) == const(A)));
    static assert(!__traits(compiles, f.relativeFrequency(0)));
    static assert(!__traits(compiles, f.relativeFrequency(0, 0, 0)));
    static assert(!__traits(compiles, f.relativeFrequency(0.5, 0)));
    static assert(!__traits(compiles, f.relativeFrequency!uint(0, 0)));
    static assert(!__traits(compiles, f.put(0.5)));
    static assert(!__traits(compiles, f.put(0.5, 0.5, 0.5)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequency(0)));
    static assert(!__traits(compiles, f.cumulativeRelativeFrequencies()));
    static assert(!__traits(compiles, f.relativeFrequencyBins()));
    static assert(!__traits(compiles, f.underflow!2()));
    static assert(!__traits(compiles, { f.counts[0][0] = 1; }));

    uint[16] buffer;
    auto storage = buffer[].sliced(4, 4).transposed;
    storage[0, 3] = 2;
    storage[1, 2] = 3;
    auto g = RelativeFrequencyAccumulator!(typeof(storage), A, A)(
        storage, A(2, 0.0), A(2, 0.0));
    assert(g.count == f.count);
    assert(g.relativeFrequency(0, 1) == f.relativeFrequency(0, 1));

    // Merge across storage representations, then verify self-merge totals.
    f.put(g);
    assert(f.count == 10 && f.counts[1][2] == 6);
    assert(f.underflow == 4 && f.overflow!1 == 4);
    g.put(f);
    assert(g.count == 15 && storage[1, 2] == 9);
    f.put(f);
    assert(f.count == 20 && f.relativeFrequency(0, 1) == 0.6);
    const snapshot = f; // Static storage is an independent copy.
    g.put(snapshot);
    assert(g.count == 35 && g.relativeFrequency(0, 1) == 0.6);

    static foreach (T; AliasSeq!(float, double, real))
    {
        static assert(is(typeof(f.relativeFrequency!T(0, 1)) == T));
        static assert(is(typeof(f.overflowRelativeFrequency!(T, 1)()) == T));
        assert(f.underflowRelativeFrequency!T() == cast(T) 0.4);
        assert(f.overflowRelativeFrequency!(T, 1)() == cast(T) 0.4);
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
    auto f = RelativeFrequencyAccumulator!(typeof(storage), A, B, C)(
        storage, A(2, 0.0), B(2, 0.0), C(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {
        assert(isNaN(f.relativeFrequency!T(0, 0, 0)));
        assert(isNaN(f.overflowRelativeFrequency!(T, 1)()));
        assert(isNaN(f.underflowRelativeFrequency!(T, 2)()));
    }
    f.put(0.5, 3.0, -1.0);
    assert(f.count == 1 && f.counts[0][2][0] == 1);
    assert(f.overflow!1 == 1 && f.underflow!2 == 1);
    assert(f.relativeFrequency(0, 0, 0) == 0);
    f.put(1.5, 0.5, 1.5);
    assert(f.count == 2 && f.relativeFrequency(1, 0, 1) == 0.5);
    assertThrown!AssertError(f.put(0.5, 0.5, 9.0));
    assert(f.count == 2 && f.counts[0][0][0] == 0);
    assertThrown!AssertError(f.relativeFrequency(-1, 0, 0));
    assertThrown!AssertError(f.relativeFrequency(0, 2, 0));

    auto incompatible = RelativeFrequencyAccumulator!(typeof(storage), A, B, C)(
        storage, A(2, 1.0), B(2, 0.0), C(2, 0.0));
    assertThrown!AssertError(f.put(incompatible));
    assert(f.count == 2 && f.relativeFrequency(1, 0, 1) == 0.5);
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
    alias F = RelativeFrequencyAccumulator!(uint[][], A, A);
    auto f = F([[1u, 2u], [3u, 4u]], A(2, 0.0), A(2, 0.0));
    assert(f.count == 10 && f.relativeFrequency(1, 0) == 0.3);
    f.put(0.5, 1.5);
    assert(f.count == 11 && f.counts[0][1] == 3);
    f.put(F([[0u, 1u], [0u, 0u]], A(2, 0.0), A(2, 0.0)));
    assert(f.count == 12 && f.counts[0][1] == 4);

    assertThrown!AssertError(F([[0u, 0u], [0u]], A(2, 0.0), A(2, 0.0)));
    alias OtherCount = RelativeFrequencyAccumulator!(ulong[][], A, A);
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
    auto f = RelativeFrequencyAccumulator!(typeof(storage), X, Y, Z)(
        storage, X(2, 0), Y(3, 0.0), Z(2, 0));
    auto bins = f.relativeFrequencyBins();
    static assert(isRandomAccessRange!(typeof(bins)));
    static assert(hasSlicing!(typeof(bins)) && hasLength!(typeof(bins)));
    assert(bins.length == 12);
    assert(isNaN(bins.front.relativeFrequency));

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
        assert(entry.count == 1 && entry.relativeFrequency == 1.0 / 13);
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
    auto other = RelativeFrequencyAccumulator!(typeof(otherStorage), X, Y, Z)(
        otherStorage, X(2, 0), Y(3, 0.0), Z(2, 0));
    other.put(0, 0.0, 0);
    f.put(other);
    assert(fixed.front.count == 2 && fixed.front.relativeFrequency == 2.0 / 14);
    static foreach (T; AliasSeq!(float, double, real))
    {{
        auto typed = f.relativeFrequencyBins!T();
        static assert(is(typeof(typed.front.relativeFrequency) == T));
        assert(typed.front.relativeFrequency == cast(T) 2 / 14);
    }}
}


// Owning storage still yields borrowed relative frequency views: only the source owns the total.
version(mir_stat_test_lifetime)
@safe @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto storage = rcslice!uint(2, 2);
    alias F = RelativeFrequencyAccumulator!(typeof(storage), A, A);
    auto f = F(storage, A(2, 0), A(2, 0));
    auto borrow(return ref const F source) { return source.relativeFrequencyBins(); }
    auto bins = borrow(f);
    const fixed = bins;
    auto saved = fixed.save;
    auto part = fixed[0 .. 2];
    f.put(0, 1);
    assert(saved[1].relativeFrequency == 1 && part[1].count == 1);
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins();
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(storage, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins()[0 .. 2];
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
    alias F = RelativeFrequencyAccumulator!(uint[][], X, Y);
    auto f = F([[0u, 0u], [0u, 0u]], X(2, 0.0), Y());
    f.put(1.5, Label.second);
    const bins = f.relativeFrequencyBins();
    assert(bins.back.index == 1 && bins.back.index!1 == 1);
    assert(bins.back.bin.low == 1.0);
    assert(bins.back.bin!1.slot == Label.second);
    assert(bins.back.count == 1 && bins.back.relativeFrequency == 1);

    // The public borrowing accessor is safe only with escape checking.
    static assert(__traits(compiles, () @safe {
        auto local = F([[0u, 0u], [0u, 0u]], X(2, 0.0), Y());
        auto view = local.relativeFrequencyBins();
        auto entry = view.front;
    }) == hasBorrowEscapeChecking);
}


// Prepopulated end bins contribute once to totals and cumulative relative frequencies.
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
        alias F = RelativeFrequencyAccumulator!(typeof(initial), A);
        auto f = F(initial, A(2, 0.0));
        enum total = 8 + 7 * u + 11 * o;
        assert(f.count == total && f.storageExtent() == initial.length);
        assert(f.relativeFrequency(0) == 3.0 / total);
        assert(f.relativeFrequency(1) == 5.0 / total);
        double[2] result;
        f.cumulativeRelativeFrequencies(result[]);
        assert(result[0] == (3.0 + 7 * u) / total);
        assert(result[1] == (8.0 + 7 * u) / total);
        assert(f.cumulativeRelativeFrequency(1) == result[1]);
        auto snapshot = f.cumulativeRelativeFrequencies();
        assert(snapshot.length == 2 && snapshot == result[]);
        const source = f;
        f.put(source);
        assert(f.count == 2 * total);
        assert(f.relativeFrequency(0) == 3.0 / total);
        static if (u) assert(f.underflow == 14);
        static if (o) assert(f.overflow == 22);
    }}
}


// All-bin relative frequency views retain live totals, precision selection, and independent cursors.
version(mir_stat_test)
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    alias F = RelativeFrequencyAccumulator!(uint[], A);
    auto f = F([0u, 0u, 0u, 0u], A(2, 0.0));
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const empty = f.relativeFrequencyBins!(T, BinCoverage.all)();
        foreach (entry; empty.save) assert(isNaN(entry.relativeFrequency));
    }}
    f.put(-1.0, 0.5, 1.5, 2.0);
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const all = f.relativeFrequencyBins!(T, BinCoverage.all)();
        static assert(is(typeof(all.front.relativeFrequency) == T));
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
        assert(all.back.relativeFrequency == cast(T)(before.count + 1) / (oldTotal + 1));
        assert(tail.back.relativeFrequency == all.back.relativeFrequency);
        assert(all.front.relativeFrequency == cast(T) 1 / f.count);
    }}
    static assert(!__traits(compiles, f.relativeFrequencyBins!(double, cast(BinCoverage) 99)()));
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
    alias F = RelativeFrequencyAccumulator!(typeof(counts), A, A);
    auto f = F(counts, A(2, 0), A(2, 0));
    auto borrow(return ref const F source) { return source.relativeFrequencyBins!(real, BinCoverage.all)(); }
    const bins = borrow(f);
    auto part = bins[0 .. 4];
    f.put(-1, 2);
    assert(part.back.isUnderflow && part.back.isOverflow!1);
    assert(part.back.count == 1 && part.back.relativeFrequency == 1);
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins!(real, BinCoverage.all)();
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins!(real, BinCoverage.all)().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto local = F(counts, A(2, 0), A(2, 0));
        return local.relativeFrequencyBins!(real, BinCoverage.all)()[0 .. 2];
    }));
}


// Both accessors use the same coordinates and count values in every coverage.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto f = RelativeFrequencyAccumulator!(uint[][], A, A)(
        [[0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u],
         [0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u]],
        A(2, 0.0), A(2, 0.0));
    static foreach (coverage; [BinCoverage.ordinary, BinCoverage.all])
    {{
        const counts = f.bins!coverage();
        const frequencies = f.relativeFrequencyBins!(real, coverage)();
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
        auto f = RelativeFrequencyAccumulator!(typeof(storage), A)(storage, A(2, 0));
        f.put(-1, 0, 2);
        return f.bins!(BinCoverage.all)();
    }
    auto retained = owning();
    assert(retained.front.isUnderflow && retained.front.count == 1);
    assert(retained.back.isOverflow && retained.back.count == 1);

    uint[4] storage;
    alias F = RelativeFrequencyAccumulator!(typeof(storage), A);
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

// Fractional counters start at zero, preserve precision, and keep empty relative frequencies NaN.
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
        auto f = RelativeFrequencyAccumulator!(typeof(data), A, A)(data, A(1, 0), A(1, 0));
        auto empty = f.marginal!1();
        static assert(is(typeof(empty).CountType == T));
        assert(empty.count == 0 && empty.counts == [T(0), T(0), T(0)]);
        assert(isNaN(empty.relativeFrequency!T(0)));
        data[0][2] = T(0.25);
        data[1][1] = T(0.5);
        data[2][0] = T(0.25);
        auto filled = RelativeFrequencyAccumulator!(typeof(data), A, A)(data, A(1, 0), A(1, 0));
        auto m = filled.marginal!1();
        assert(m.count == 1 && m.count == filled.count);
        assert(m.underflow == T(0.25) && m.overflow == T(0.25));
        assert(m.relativeFrequency!T(0) == T(0.5));
        static assert(!__traits(compiles, filled.marginal!(0, 0)()));
        static assert(!__traits(compiles, filled.marginal!2()));
    }}
}

// A marginal relative frequency accumulator owns its counts and maintained total.
version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    auto makeMarginal()
    {
        uint[2][2] data = [[1u, 2u], [3u, 4u]];
        auto f = RelativeFrequencyAccumulator!(typeof(data), A, A)(data, A(2, 0), A(2, 0));
        return f.marginal!0();
    }
    auto m = makeMarginal();
    assert(m.count == 10 && m.counts == [3u, 7u]);
    m.put(1);
    assert(m.count == 11 && m.relativeFrequency(1) == 8.0 / 11);
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
    const f = RelativeFrequencyAccumulator!(typeof(storage), A, A, A)(
        storage, A(2, 0), A(3, 10), A(4, 20));
    auto m = f.marginal!(2, 0)();
    assert(m.count == 300 && m.count == f.count);
    assert(m.counts.shape == [4, 2]);
    assert(m.axis!0.bin(0).low == 20 && m.axis!1.bin(0).low == 0);
    assert(m.relativeFrequency(0, 0) == 15.0 / 300);
    m.put(20, 0);
    assert(m.count == 301 && f.count == 300);
}


// Read-only storage permits relative frequency reductions without permitting count mutation.
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
        auto f = RelativeFrequencyAccumulator!(S, A, A)(storage, A(1, 0), A(1, 0));
        static assert(is(typeof(f).CountType == uint));
        assert(f.count == 45 && f.relativeFrequency(0, 0) == 5.0 / 45);
        assert(f.underflow == 6 && f.overflow == 24);
        assert(f.underflow!1 == 12 && f.overflow!1 == 18);
        assert(f.underflowRelativeFrequency!(double, 1) == 12.0 / 45);
        const frozen = f;
        assert(frozen.count == 45 && frozen.relativeFrequency(0, 0) == 5.0 / 45);
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
        auto f = RelativeFrequencyAccumulator!(S, A)(storage, A(2, 0));
        static assert(is(typeof(f).CountType == uint));
        assert(f.count == 10 && f.underflow == 1 && f.overflow == 4);
        assert(f.relativeFrequency(0) == 0.2 && f.cumulativeRelativeFrequency(1) == 0.6);
        double[2] output;
        f.cumulativeRelativeFrequencies(output[]);
        assert(output == [0.3, 0.6]);
        assert(f.cumulativeRelativeFrequencies!double() == output[]);
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
        const f = RelativeFrequencyAccumulator!(uint[], A)(counts, A(2, 0));
        static foreach (F; AliasSeq!(float, double, real))
        {{
            const fixed = f.cumulativeRelativeFrequencyBins!F();
            auto cursor = fixed.save;
            static assert(isForwardRange!(typeof(cursor)));
            static assert(!isRandomAccessRange!(typeof(cursor)));
            assert(cursor.length == 2);
            assert(cursor.front.cumulativeCount == 1 + under);
            cursor.popFront();
            assert(cursor.front.cumulativeRelativeFrequency == f.cumulativeRelativeFrequency!F(1));
            assert(fixed.front.index == 0);
            cursor.popFront();
            assert(cursor.empty);
            assertThrown!AssertError(cursor.front);
            assertThrown!AssertError(cursor.popFront());
        }}
        auto all = f.cumulativeRelativeFrequencyBins!(double, BinCoverage.all)();
        uint sum;
        foreach (entry; all)
        {
            ++sum;
            assert(entry.cumulativeCount == sum);
            assert(entry.cumulativeRelativeFrequency == cast(double) sum / counts.length);
        }
        counts[] = 0;
        auto zero = RelativeFrequencyAccumulator!(uint[], A)(counts, A(2, 0));
        foreach (entry; zero.cumulativeRelativeFrequencyBins!(double, BinCoverage.all)())
            assert(entry.cumulativeCount == 0 && isNaN(entry.cumulativeRelativeFrequency));
    }}
    alias A = IntegralAxis!(uint, int, AxisOptions());
    alias Joint = RelativeFrequencyAccumulator!(uint[][], A, A);
    static assert(!__traits(compiles, Joint.init.cumulativeRelativeFrequencyBins()));
    alias F = RelativeFrequencyAccumulator!(uint[], A);
    static assert(!__traits(compiles, F.init.cumulativeRelativeFrequencyBins!int()));
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
        auto f = RelativeFrequencyAccumulator!(typeof(counts), A)(counts, A(3, 0));
        auto cursor = f.cumulativeRelativeFrequencyBins();
        auto saved = cursor.save;
        cursor.popFront();
        assert(cursor.front.cumulativeCount == 3 && saved.front.cumulativeCount == 1);
        cursor.popFront();
        assert(cursor.front.cumulativeRelativeFrequency == 1);
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
    auto f = RelativeFrequencyAccumulator!(typeof(strided), A)(strided, A(3, 0));
    auto cursor = f.cumulativeRelativeFrequencyBins();
    cursor.popFront(); cursor.popFront();
    assert(cursor.front.cumulativeCount == 4);
}

version(mir_stat_test_lifetime)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, int, AxisOptions());
    alias F = RelativeFrequencyAccumulator!(uint[], A);
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 2u], A(2, 0));
        return f.cumulativeRelativeFrequencyBins();
    }));
    static assert(!__traits(compiles, () @safe {
        auto f = F([1u, 2u], A(2, 0));
        return f.cumulativeRelativeFrequencyBins().save;
    }));
    static assert(!__traits(compiles, () @safe {
        auto cursor = F([1u, 2u], A(2, 0)).cumulativeRelativeFrequencyBins();
        auto entry = cursor.front;
    }));
    static assert(!__traits(compiles, () @safe {
        uint[2] data = [1, 2];
        auto f = F(data[], A(2, 0));
        auto cursor = f.cumulativeRelativeFrequencyBins();
        auto entry = cursor.front;
    }));
}

// Borrowed relative frequency reads, snapshots, and formatting preserve all four
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
    auto f = RelativeFrequencyAccumulator!(typeof(counts), A)(counts, A(2, 0));

    assert(f.count == 8 && f.relativeFrequency!float(0) == 0.25f);
    assert(f.underflowRelativeFrequency() == 0.125 && f.overflowRelativeFrequency() == 0.25);

    // Snapshot allocation uses reference-counted storage, not the GC.
    auto snapshot = f.cumulativeRelativeFrequencies!float();
    double[2] destination;
    f.cumulativeRelativeFrequencies(destination[]);
    assert(snapshot[0] == 0.375f && snapshot[1] == 0.75f);
    assert(destination[0] == snapshot[0] && destination[1] == snapshot[1]);

    {
        auto bins = f.relativeFrequencyBins!float();
        auto writer = scopedBuffer!(char, 256);
        print(writer, bins.front);
        assert(writer.data == "bin(low=0, high=1): count=2, relativeFrequency=0.25");

        auto cumulative = f.cumulativeRelativeFrequencyBins!(double, BinCoverage.all)();
        assert(cumulative.front.isUnderflow);
        cumulative.popFront();
        auto cumulativeWriter = scopedBuffer!(char, 256);
        print(cumulativeWriter, cumulative.front);
        assert(cumulativeWriter.data ==
            "bin(low=0, high=1): count=2, cumulativeCount=3, cumulativeRelativeFrequency=0.375");
        cumulative.popFront();
        cumulative.popFront();
        assert(cumulative.front.isOverflow && cumulative.front.cumulativeRelativeFrequency == 1);
    }

    // Finish cumulative traversal before changing the source. Saved snapshots
    // remain independent, while a newly borrowed range sees the updated counts.
    f.put(0);
    assert(snapshot[0] == 0.375f && snapshot[1] == 0.75f);
    assert(f.relativeFrequencyBins().front.count == 3 && f.count == 9);
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
    auto f = RelativeFrequencyAccumulator!(typeof(counts), A, A)(counts, A(2, 0), A(2, 0));
    f.put(-1, -1);
    f.put(0, 1);
    f.put(1, 0);
    f.put(2, 2);
    assert(f.count == 4 && f.relativeFrequency!float(0, 1) == 0.25f);
    assert(f.underflow!0() == 1 && f.overflow!1() == 1);

    auto bins = f.relativeFrequencyBins!(float, BinCoverage.all)();
    auto saved = bins.save;
    auto middle = bins[5 .. 11];
    assert(bins.length == 16 && bins.front.relativeFrequency == 0.25f);
    assert(bins.front.isUnderflow!0 && bins.front.isUnderflow!1);
    assert(bins.back.isOverflow!0 && bins.back.isOverflow!1);
    bins.popFront();
    saved.popBack();
    assert(bins.length == 15 && saved.length == 15 && middle.length == 6);
    assert(f.bins!(BinCoverage.all)()[6].count == 1);

    auto marginal = f.marginal!0();
    assert(marginal.count == 4 && marginal.relativeFrequency(0) == 0.25);
    marginal.put(0);
    assert(marginal.count == 5 && f.count == 4);
    assert(f.relativeFrequency(0, 1) == 0.25);
}

// Small count types retain their type instead of exposing integer promotion.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(ubyte, int, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(ubyte[], A)([cast(ubyte) 1, 2], A(2, 0));
    auto cursor = f.cumulativeRelativeFrequencyBins();
    static assert(is(typeof(cursor.front.cumulativeCount) == ubyte));
    cursor.popFront();
    assert(cursor.front.cumulativeCount == 3 && cursor.front.cumulativeRelativeFrequency == 1);
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
    auto f = RelativeFrequencyAccumulator!(double[], A)([0.5, 1.5], A(2, 0));
    auto cursor = f.cumulativeRelativeFrequencyBins();
    assert(cursor.front.cumulativeCount == 0.5);
    assert(cursor.front.cumulativeRelativeFrequency == 0.25);
    cursor.popFront();
    assert(cursor.front.cumulativeCount == 2);
    assert(cursor.front.cumulativeRelativeFrequency == 1);
}

/++
A snapshot of a numeric bin's count, relative frequency, and density.
Coordinates and count access are forwarded to the relative-frequency entry.
Changing this value does not update the source accumulator.
+/
struct DensityBin(RelativeFrequencyElement, DensityType)
{
    private RelativeFrequencyElement _entry;
    alias _entry this;
    /// Probability per unit bin volume when this entry was read.
    DensityType density;

    /// Format bin coordinates, count, relative frequency, and density.
    void toString(Writer)(ref Writer writer) const
    {
        import mir.format: print;
        import mir.appender: scopedBuffer;
        import std.range.primitives: put;
        auto buffer = scopedBuffer!(char, 256);
        _entry.toString(buffer);
        print(buffer, ", density=", density);
        put(writer, buffer.data);
    }
}

/++
Borrowed density view created by RelativeFrequencyAccumulator.densityBins.
Delegates traversal and source checks to the relative-frequency view.
The source must remain alive and in place; returned descriptions can borrow
axis storage. Const cursors support reads and save returns a mutable cursor.
+/
struct DensityBinView(Storage, DensityType, Axis...)
    if (isFloatingPoint!DensityType && supportsBinView!(Storage, Axis) &&
        allSatisfy!(supportsDensityAxis, Axis))
{
    private alias Base = RelativeFrequencyBinView!(Storage, real, BinCoverage.ordinary, Axis);
    private Base _bins;
    private alias FrequencyElement = RelativeFrequencyBin!(Base.BinView.Element, DensityType);
    /// Value returned by element access; both normalized fields use DensityType.
    alias Element = DensityBin!(FrequencyElement, DensityType);
    private this(Base bins) { _bins = bins; }
    /// Number of remaining bins.
    size_t length() const @property { return _bins.length; }
    /// Whether traversal is exhausted.
    bool empty() const @property { return _bins.empty; }
    /// Current first entry.
    Element front() const @property { return this[0]; }
    /// Current last entry.
    Element back() const @property { return this[length - 1]; }
    /// Advance by one bin.
    void popFront() { _bins.popFront(); }
    /// Remove the last bin from this cursor.
    void popBack() { _bins.popBack(); }
    /// Independent cursor at the same position.
    auto save() const @property { return DensityBinView(_bins.save); }
    /// Read a density at an offset from the cursor.
    Element opIndex(size_t index) const
    {
        import mir.stat.descriptive.histogram.internal.density: ScaledBinVolume;
        auto entry = _bins[index];
        ScaledBinVolume volume;
        static foreach (i; 0 .. Axis.length)
            volume.include(entry.bin!i);
        return Element(FrequencyElement(entry._entry, entry.count,
            cast(DensityType) entry.relativeFrequency),
            volume.normalize!DensityType(entry.relativeFrequency));
    }
    /// Slice relative to this cursor; original bin indices are retained.
    auto opSlice(size_t begin, size_t end) const
    {
        return DensityBinView(_bins[begin .. end]);
    }
    /// End index for slicing with $.
    size_t opDollar() const { return length; }
}

/// Equal bin probabilities can have different densities when widths differ.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis, AxisOptions;
    uint[2] counts = [1, 1];
    double[3] boundaries = [0, 1, 3];
    alias A = VariableAxis!(uint, double*, AxisOptions());
    auto f = RelativeFrequencyAccumulator!(uint[], A)(counts[], A(boundaries[].sliced));
    assert(f.relativeFrequency(0) == 0.5 && f.relativeFrequency(1) == 0.5);
    assert(f.density(0) == 0.5 && f.density(1) == 0.25);
    // Integrating each constant bin density recovers the total probability.
    assert(f.density(0) * 1 + f.density(1) * 2 == 1);
    assert(f.density!float(1) == 0.25f);
}

// Joint density uses volume, and normalization includes underflow/overflow counts.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.math: isNaN;
    alias X = RegularAxis!(uint, double, AxisOptions());
    uint[4] data = [1, 1, 1, 1];
    auto f = RelativeFrequencyAccumulator!(typeof(data[].sliced(2, 2)), X, X)(
        data[].sliced(2, 2), X(2, 0, 2), X(2, 0, 4));
    assert(f.density(0, 1) == 0.125); // Probability 1/4, area 1 * 2.
    alias A = RegularAxis!(uint, double, AxisOptions(false, true, true));
    uint[4] flowCounts = [1, 2, 1, 4];
    const all = RelativeFrequencyAccumulator!(uint[], A)(flowCounts[], A(2, 0, 4));
    assert(all.density(0) == 0.125 && all.density(1) == 0.0625);
    assert((all.density(0) + all.density(1)) * 2 == 3.0 / 8);
    uint[2] zeros;
    auto empty = RelativeFrequencyAccumulator!(uint[], X)(zeros[], X(2, 0, 2));
    assert(isNaN(empty.density(0)));
    empty.put(0.5);
    assert(empty.density(1) == 0);
    static assert(!__traits(compiles, f.density!int(0, 0)));
    static assert(!__traits(compiles, f.density(0)));
}

// Transformed-axis density is measured in original coordinates, not log units.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: TransformAxis, IntegralAxis, CategoryAxis, AxisOptions;
    import mir.math.common: log2, exp2;
    alias A = TransformAxis!(uint, double, log2, exp2, AxisOptions());
    uint[2] counts = [1, 1];
    auto f = RelativeFrequencyAccumulator!(uint[], A)(counts[], A(2, 1, 4));
    assert(f.density(0) == 0.5 && f.density(1) == 0.25);
    alias Unit = IntegralAxis!(uint, long, AxisOptions());
    auto integers = RelativeFrequencyAccumulator!(uint[], Unit)(counts[], Unit(2, long.max - 2));
    assert(integers.density!float(0) == 0.5f);
    enum Label { a, b }
    alias Categories = RelativeFrequencyAccumulator!(uint[], CategoryAxis!(uint, Label, AxisOptions()));
    static assert(!__traits(compiles, Categories.init.density(0)));
    static assert(!__traits(compiles, Categories.init.densityBins()));
}

// The density adapter preserves const random access, live updates, and attributes.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    static void check()()
    {
        import mir.ndslice.allocation: rcslice;
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: VariableAxis, AxisOptions;
        import std.range.primitives: isRandomAccessRange;
        import mir.appender: scopedBuffer;
        static immutable uint[2] initial = [1, 1];
        static immutable double[3] boundaries = [0, 1, 3];
        auto counts = rcslice!uint(initial[]);
        alias A = VariableAxis!(uint, immutable(double)*, AxisOptions());
        auto f = RelativeFrequencyAccumulator!(typeof(counts), A)(counts, A(boundaries[].sliced));
        auto view = f.densityBins!float();
        static assert(isRandomAccessRange!(typeof(view)));
        static assert(is(typeof(view.front.density) == float));
        assert(view[0].density == 0.5f && view.back.density == 0.25f);
        const fixed = view;
        assert(fixed.front.bin.low == 0 && fixed.front.count == 1);
        auto cursor = fixed.save;
        cursor.popFront();
        assert(cursor.front.index == 1 && fixed.length == 2);
        auto tail = fixed[1 .. 2];
        assert(tail.front.density == 0.25f);
        auto writer = scopedBuffer!(char, 256);
        view.front.toString(writer);
        assert(writer.data == "bin(low=0.0, high=1.0): count=1, relativeFrequency=0.5, density=0.5");
        f.put(2.0);
        assert(view.front.density == cast(float)(1.0 / 3));
        assert(tail.front.density == cast(float)(1.0 / 3));
        cursor.popBack();
        assert(cursor.empty);
        static assert(!__traits(compiles, () @safe {
            auto local = typeof(f)(counts, A(boundaries[].sliced));
            return local.densityBins();
        }));
        static assert(!__traits(compiles, () @safe {
            auto local = typeof(f)(counts, A(boundaries[].sliced));
            return local.densityBins()[0 .. 1].save;
        }));
    }
    version(mir_stat_test_lifetime)
        () @safe { check!()(); }();
    else
        check!()();
}

// Transformed density traversal uses original-coordinate bounds and widths.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    static void check()()
    {
        import mir.ndslice.allocation: rcslice;
        import mir.stat.descriptive.histogram.axis: TransformAxis, AxisOptions;
        import mir.math.common: log2, exp2;
        static immutable uint[2] initial = [1, 1];
        auto counts = rcslice!uint(initial[]);
        alias A = TransformAxis!(uint, double, log2, exp2, AxisOptions());
        const f = RelativeFrequencyAccumulator!(typeof(counts), A)(counts, A(2, 1, 4));
        const view = f.densityBins();
        assert(view.length == 2);
        assert(view[0].bin.low == 1 && view[0].bin.high == 2);
        assert(view[1].bin.low == 2 && view[1].bin.high == 4);
        assert(view[0].density == 0.5 && view[1].density == 0.25);
        double mass = 0;
        foreach (i; 0 .. view.length)
        {
            const entry = view[i];
            assert(entry.density == f.density(i));
            mass += entry.density * (entry.bin.high - entry.bin.low);
        }
        assert(mass == 1);
        const floats = f.densityBins!float();
        assert(floats[0].density == f.density!float(0));
        assert(floats[1].density == f.density!float(1));
    }
    version(mir_stat_test_lifetime)
        () @safe { check!()(); }();
    else
        check!()();
}

// Every floating-point output type uses actual rounded regular-axis boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.stat.descriptive.histogram.axis: RegularAxis, VariableAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        alias A = RegularAxis!(uint, T, AxisOptions());
        uint[3] counts = [1, 1, 1];
        auto f = RelativeFrequencyAccumulator!(uint[], A)(counts[], A(3, T(0), T(1)));
        T mass = 0;
        foreach (i; 0 .. 3)
        {
            const bin = f.axis.bin(i);
            mass += f.density!T(i) * (bin.high - bin.low);
        }
        assert(approxEqual(mass, T(1), T.epsilon * 8, T(0)));
        alias V = VariableAxis!(uint, T*, AxisOptions());
        T[2] bounds = [T(0), T.min_normal];
        uint[1] one = [1];
        auto tiny = RelativeFrequencyAccumulator!(uint[], V)(one[], V(bounds[].sliced));
        assert(approxEqual(tiny.density!T(0), T(1) / T.min_normal, T.epsilon * 8, T(0)));
    }}
}

// Joint density views retain multidimensional coordinates for nested and strided storage.
version(mir_stat_test)
pure nothrow
unittest
{
    static void check()()
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.dynamic: transposed;
        import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
        alias A = RegularAxis!(uint, double, AxisOptions());
        auto nested = RelativeFrequencyAccumulator!(uint[][], A, A)(
            [[1u, 2u], [3u, 4u]], A(2, 0, 2), A(2, 0, 4));
        const view = nested.densityBins();
        assert(view.length == 4);
        assert(view[2].index!0 == 1 && view[2].index!1 == 0);
        assert(view[2].density == nested.density(1, 0));
        assert(view[2].relativeFrequency == nested.relativeFrequency(1, 0));
        auto tail = view[2 .. $];
        assert(tail.front.density == 0.15);
        auto storage = new uint[4];
        storage[] = [1u, 2u, 3u, 4u];
        auto strided = storage.sliced(2, 2).transposed;
        auto f = RelativeFrequencyAccumulator!(typeof(strided), A, A)(strided, A(2, 0, 2), A(2, 0, 4));
        foreach (entry; f.densityBins!float())
            assert(entry.density == f.density!float(entry.index!0, entry.index!1));
    }
    version(mir_stat_test_lifetime)
        () @safe { check!()(); }();
    else
        check!()();
}
