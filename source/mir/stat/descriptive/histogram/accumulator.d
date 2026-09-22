/++
This module contains a histogram accumulator and read-only bin views.

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

module mir.stat.descriptive.histogram.accumulator;

version(mir_stat_test)
private import mir.stat.descriptive.histogram.api.rc: rcMarginal;

private import mir.stat.descriptive.histogram.traits: ordinaryBinCount;

import mir.primitives: DeepElementType;
import mir.stat.descriptive.histogram.traits: isAxis;
import mir.stat.descriptive.histogram.internal.view: supportsBinView, JointArrayInfo;
import mir.stat.descriptive.histogram.internal.projection: validMarginalAxes;
private import mir.stat.descriptive.histogram.internal.cell:
    acceptsCellSamples, acceptsCellMerge, mergeCell;
import mir.qualifier: lightConst;
import std.meta: allSatisfy;
import std.traits: isNumeric, Unqual, isStaticArray;
import mir.ndslice.slice: isSlice;

// Weighted insertion follows implicit conversion rules for real numeric weights.
package template acceptsHistogramWeight(C, W)
{
    import std.traits: isIntegral, isFloatingPoint;
    enum acceptsHistogramWeight =
        (isIntegral!(Unqual!C) || isFloatingPoint!(Unqual!C)) &&
        (isIntegral!(Unqual!W) || isFloatingPoint!(Unqual!W)) && is(W : C);
}

private template isJointStorage(Storage, size_t rank)
{
    static if (isSlice!Storage)
        enum isJointStorage = Storage.N == rank;
    else
        enum isJointStorage = JointArrayInfo!Storage.rank == rank;
}

/++
Accumulator for numeric counts or per-bin statistical accumulators.

With numeric cells and one axis, each argument to put records an observation. With multiple
axes, put(x, y, ...) increments one joint bin. Storage must be an ndslice or
rectangular nested built-in array, with one dimension per axis.
Nested static arrays are copied into the accumulator; dynamic arrays and
ndslices share their backing counts. Keep axis definitions and storage shape
unchanged while using the accumulator.

Each storage dimension contains enabled underflow, ordinary
bins, and enabled overflow, in that order. Its length is the ordinary bin
count plus one for each enabled end bin. Axis indices still number ordinary
bins from zero; storage indices are shifted by one when underflow is enabled.
The same layout applies to one-dimensional histograms.

Bin views traverse ordinary joint bins with the last axis advancing fastest.
Use putWeighted(weight, coordinates...) to accumulate weights instead of unit
counts. Bin counts then represent sums of weights; this accumulator maintains
neither a separate observation count nor a running total.

Per-bin accumulators group observations by their coordinates and compute a
statistic from the samples in each group. As an example, use a Summator for more
sophisticated summation algorithms, MeanAccumulator for a mean, or
WMeanAccumulator for a weighted mean. The sample may represent a different
quantity from the coordinates: for example, grouping purchases by customer age
and accumulating purchase amounts gives total sales revenue for each age group.

Use putSample(sample, coordinates...) or
putWeightedSample(weight, sample, coordinates...) to update the selected cell.
Caller-provided cells retain their existing state, including their normal default
initialization. Numeric counting operations require numeric cells; sample
operations forward to the stored accumulator's put method. Marginalization
combines accumulator states using the same operation as histogram merging.

Storage requirements:
Storage is a built-in array or Mir ndslice whose elements are the bin cells.
One axis requires one-dimensional storage; multiple axes require a slice of
matching rank or rectangular nested arrays, including mixtures of static and
dynamic arrays. Strided ndslices are supported. Every dimension must match its
axis's storage extent, including enabled underflow/overflow bins. Construction
checks these extents and retains the supplied cell values; it does not reset them.

Cell requirements depend on the operation used:
$(UL
    $(LI Numeric cells support ordinary counting. Weighted counting additionally
        requires a real numeric weight implicitly convertible to the counter type.)
    $(LI putSample(sample, coordinates...) requires a nonnumeric cell with a
        callable cell.put(sample).)
    $(LI putWeightedSample(weight, sample, coordinates...) requires a nonnumeric
        cell with a callable cell.put(sample, weight).)
    $(LI Histogram merging requires matching cell and axis types. A cell must
        accept put(sourceCell) or += sourceCell with a const source cell;
        put takes precedence for nonnumeric cells.)
)
There is no required accumulator base type or fixed sample type. Custom cells
can implement only the operations they need. Updating requires mutable cells;
read-only storage can be used for bin views when the axes support bin descriptions
and the cells support const copying. Cell operations determine their own sample
and weight validity, allocation behavior, and function attributes.

The underflow/overflow total members currently require numeric cells.
Accumulator end bins can be read through bins!(BinCoverage.all).

If the `Axis` has an `options` member, the histogram may optionally allow
for overflow and underflow members.

Params:
    Storage = array or ndslice containing one numeric value or accumulator per stored bin
    Axis = the type of the axis used to create the histogram bins

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis),
    $(LREF RelativeFrequencyAccumulator)
+/
struct HistogramAccumulator(Storage, Axis...)
    if (Axis.length > 0 &&
        allSatisfy!(isAxis, Axis))
{
    import std.traits: isIterable, isSomeString;
    import mir.stat.descriptive.histogram.traits: includeOverflow, includeUnderflow,
        BinTypeOf, isCategoryAxis, acceptsAxisValue;
    static if (Axis.length > 1 && !isSlice!Storage)
        private alias StoredCountType = JointArrayInfo!Storage.Element;
    else
        private alias StoredCountType = DeepElementType!Storage;

    /// Type of one cell, independently of storage mutability; also see ValueType.
    alias CountType = Unqual!StoredCountType;
    /// Type of one stored numeric value or accumulator.
    alias ValueType = CountType;
    static if (Axis.length > 1)
    {
        static assert(isJointStorage!(Storage, Axis.length),
            "HistogramAccumulator: joint storage must be an ndslice or nested array with one dimension per axis");
    }
private:
    import mir.stat.descriptive.histogram.traits: axisStorageExtent = storageExtent;

    size_t storageIndex(size_t i, T)(T value)
    {
        // Match one-axis classification precedence and circular endpoints.
        static if (includeOverflow!(Axis[i]))
            if (axis[i].isOverflow(value))
                return axisStorageExtent(axis[i]) - 1;
        static if (includeUnderflow!(Axis[i]))
            if (axis[i].isUnderflow(value))
                return 0;
        auto index = axis[i].index(value);
        assert(index >= 0 && index < axis[i].N_bin,
            "HistogramAccumulator.put: joint bin index is out of range");
        return cast(size_t) index + includeUnderflow!(Axis[i]);
    }

    size_t[N] storageIndices(T...)(T coordinates)
    {
        size_t[N] indices;
        static foreach (i; 0 .. N)
            indices[i] = storageIndex!i(coordinates[i]);
        return indices;
    }

    // Validate every branch: checking only the first row would miss ragged arrays.
    static void validateArrayShape(size_t depth = 0, S)(auto ref const S storage,
        const ref size_t[N] shape)
    {
        assert(storage.length == shape[depth],
            "HistogramAccumulator.this: every storage dimension must match its axis");
        static if (depth + 1 < N)
            foreach (ref child; storage)
                validateArrayShape!(depth + 1)(child, shape);
    }

    private template acceptsMerge(H)
    {
        static if (is(Unqual!H == HistogramAccumulator!Args, Args...))
            enum acceptsMerge =
                is(Unqual!H == HistogramAccumulator!(Args[0], Axis)) &&
                is(Unqual!(H.CountType) == Unqual!CountType) &&
                acceptsCellMerge!StoredCountType;
        else
            enum acceptsMerge = false;
    }

    static void validateStorageShape(S)(auto ref const S storage,
        const ref size_t[N] shape)
    {
        static if (isSlice!S)
            assert(storage.shape == shape,
                "HistogramAccumulator: storage shape must match all axes");
        else
            validateArrayShape(storage, shape);
    }

    // Partial ndslices are handles; nested static arrays must remain references.
    static void mergeStorage(size_t depth = 0, D, S)(auto ref D destination,
        auto ref const S source)
    {
        foreach (i; 0 .. destination.length)
        {
            static if (depth + 1 == N)
                mergeCell(destination[i], source[i]);
            else
                mergeStorage!(depth + 1)(destination[i], source[i]);
        }
    }

    // Probe only the cell operation; coordinate checking is shared with counting.
    private template acceptsSamples(Samples...)
    {
        enum acceptsSamples = acceptsCellSamples!(StoredCountType, Samples);
    }

    // Keep nested static arrays as references, and preserve ndslice strides.
    private static void putArraySample(size_t depth = 0, S, Samples...)(
        ref S storage, const ref size_t[N] indices, auto ref Samples samples)
    {
        static if (depth + 1 == N)
            storage[indices[depth]].put(samples);
        else
            putArraySample!(depth + 1)(storage[indices[depth]], indices, samples);
    }

    // Recurse by reference so nested static arrays are updated in place.
    static void updateArray(bool weighted, size_t depth = 0, S)(ref S storage,
        const ref size_t[N] indices, CountType weight = CountType.init)
    {
        static if (depth + 1 == N)
        {
            static if (weighted)
                storage[indices[depth]] += weight;
            else
                storage[indices[depth]]++;
        }
        else
            updateArray!(weighted, depth + 1)(storage[indices[depth]], indices, weight);
    }

    // Fix one coordinate and sum the remaining dimensions. Indexed traversal
    // works for both nested arrays and ndslices, preserving their strides.
    static CountType axisEndTotal(size_t dimension, size_t depth = 0, S)(
        auto ref const S storage, size_t position)
    {
        static if (depth == N)
            return storage;
        else static if (depth == dimension)
            return axisEndTotal!(dimension, depth + 1)(storage[position], position);
        else
        {
            CountType result = 0;
            foreach (i; 0 .. storage.length)
                result += axisEndTotal!(dimension, depth + 1)(storage[i], position);
            return result;
        }
    }

public:

    ///
    Axis axis;

    /// All stored cells, including enabled underflow/overflow bins.
    Storage counts;

    // Shared projection implementation for GC, RC, and custom API factories.
    package(mir.stat.descriptive.histogram)
    auto projectMarginal(alias make, alias release, Context, dimensions...)(ref Context context) const
        if (acceptsCellMerge!CountType && validMarginalAxes!(N, dimensions))
    {
        import std.meta: staticMap;
        import mir.stat.descriptive.histogram.internal.projection: projectCells;

        template SelectedAxis(size_t dimension)
        {
            alias SelectedAxis = typeof(lightConst(axis[dimension]));
        }
        alias SelectedAxes = staticMap!(SelectedAxis, dimensions);
        SelectedAxes selected;
        size_t[N] sourceShape;
        static foreach (i; 0 .. N)
            sourceShape[i] = axisStorageExtent(axis[i]);
        validateStorageShape(counts, sourceShape);

        static foreach (i, dimension; dimensions)
        {
            selected[i] = lightConst(axis[dimension]);
        }
        auto result = make!(Unqual!CountType)(context, selected);
        static if (!is(typeof(release) == typeof(null)))
            scope(failure) release(context, result.counts);
        enum selectedDimensions = [dimensions];
        projectCells!(N, selectedDimensions)(result.counts, counts);
        return result;
    }

    /++
    Read-only random-access view of bins and their values.

    Numeric entries expose both value and count. Accumulator entries expose
    value as a const copy of their state; any referenced data remains shared
    and const-qualified. Copying an entry does not deep-copy owned or borrowed
    resources. Its referenced data must remain valid while the entry is used.

    The view copies the axis and storage handles, sharing the count buffer.
    Subsequent count updates are visible when an element is read. Replacing
    this accumulator's axis or storage does not redirect an existing view.
    Keep shared axis boundaries and the storage shape unchanged while using it.

    Available for axes with const bin-description access and supported storage.
    Coverage defaults to ordinary bins. BinCoverage.all also includes enabled
    underflow/overflow bins. The last axis advances fastest.
    Mutable and const histograms both return a view with a mutable cursor over
    read-only data. Custom axes must support mir.qualifier.lightConst.

    Params:
        coverage = ordinary bins by default, or all enabled stored bins
    See_also: $(LREF HistogramBinView)
    +/
    auto bins(BinCoverage coverage = BinCoverage.ordinary)() const
        if (supportsBinView!(Storage, Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        return HistogramBinView!(Storage, coverage, Axis)(counts, axis);
    }

    /++
    Borrow bins from static-array storage without copying its counts.
    Keep the accumulator alive and in place while using the view. Safety is
    inferred as @safe when borrow escape checking is enabled, otherwise @system.
    Moving or replacing the source while borrowed is prohibited by contract.
    +/
    auto bins(BinCoverage coverage = BinCoverage.ordinary)() return const
        if (isStaticArray!Storage && supportsBinView!(typeof(counts[]), Axis) &&
            (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
    {
        import mir.stat.internal.borrow: hasBorrowEscapeChecking, uncheckedBorrow;
        static if (!hasBorrowEscapeChecking)
            uncheckedBorrow();
        return HistogramBinView!(typeof(counts[]), coverage, Axis)(counts[], axis);
    }

    //
    enum N = Axis.length;

    /++
    Required storage length along an axis, including enabled underflow/overflow.
    Params:
        dimension = axis dimension, defaulting to zero
    +/
    size_t storageExtent(size_t dimension = 0)() const
        if (dimension < N)
    {
        return axisStorageExtent(axis[dimension]);
    }

    /++
    Construct an accumulator with storage matching the axes.
    Params:
        x = count storage including enabled underflow/overflow bins on every axis
        y = axes defining the bins
    +/
    this(Storage x, Axis y)
    {
        size_t[N] shape;
        static foreach (i; 0 .. N)
            shape[i] = axisStorageExtent(y[i]);
        validateStorageShape(x, shape);
        counts = x;
        axis = y;
    }

    ///
    void put(Range)(Range r)
        if (isNumeric!CountType && N == 1 &&
            isIterable!Range &&
            !(isCategoryAxis!(Axis[0]) && isSomeString!Range))
    {
        foreach(x; r)
        {
            put(x);
        }
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
    Record observations supplied as arguments.
    With one axis, each argument is a separate observation. With multiple axes,
    supply exactly one compatible coordinate per axis for a single observation.
    +/
    void put(T...)(T x)
        if (isNumeric!CountType && acceptsArguments!T)
    {
        static if (N == 1)
        {
            static foreach (i; 0 .. T.length)
                counts[storageIndex!0(x[i])]++;
        }
        else
        {
            // Resolve all coordinates before changing counts, including when
            // an axis rejects an observation or returns an invalid index.
            const indices = storageIndices(x);
            static if (isSlice!Storage)
                counts[indices]++;
            else
                updateArray!false(counts, indices);
        }
    }

    /++
    Record a sample in the accumulator selected by one coordinate per axis.
    For example, bin by temperature while accumulating mean response time.
    All coordinates are checked before the cell is updated. The cell's put
    method determines sample validity, attributes, and any allocation behavior.

    Params:
        sample = value passed to the selected cell's put method
        coordinates = one compatible coordinate per axis
    +/
    void putSample(S, T...)(auto ref S sample, T coordinates)
        if (T.length == N && acceptsArguments!T && acceptsSamples!S)
    {
        const indices = storageIndices(coordinates);
        static if (isSlice!Storage)
            counts[indices].put(sample);
        else
            putArraySample(counts, indices, sample);
    }

    /++
    Record a weighted sample in the selected accumulator.
    Weight comes first, as in putWeighted; the cell receives put(sample, weight).
    The cell determines weight validity and semantics, including zero weights.
    No separate histogram count or total is maintained. All coordinates are
    checked before the cell is updated.

    Params:
        weight = weight passed to the cell after the sample
        sample = value being accumulated
        coordinates = one compatible coordinate per axis
    +/
    void putWeightedSample(W, S, T...)(auto ref W weight, auto ref S sample, T coordinates)
        if (T.length == N && acceptsArguments!T && acceptsSamples!(S, W))
    {
        const indices = storageIndices(coordinates);
        static if (isSlice!Storage)
            counts[indices].put(sample, weight);
        else
            putArraySample(counts, indices, sample, weight);
    }

    /++
    Add a finite, nonnegative weight to one bin without maintaining a total.
    Supply exactly one coordinate per axis. Ordinary put continues to add one.
    Weight must be implicitly convertible to CountType and within its range;
    floating-point storage may round it to the counter precision. Fractional
    weights require floating-point counters. Counters must accommodate accumulated
    weights; overflow is not checked. Invalid weights and coordinates are rejected
    before changing storage. A zero weight still validates coordinates.

    Params:
        weight = finite, nonnegative contribution to the selected bin
        coordinates = one compatible coordinate per axis
    +/
    void putWeighted(W, T...)(W weight, T coordinates)
        if (acceptsHistogramWeight!(CountType, W) && T.length == N && acceptsArguments!T)
    {
        import std.traits: isFloatingPoint;
        import std.math: isFinite;
        static if (isFloatingPoint!(Unqual!W))
            assert(isFinite(weight), "HistogramAccumulator.putWeighted: weight must be finite");
        assert(weight >= 0 && weight <= CountType.max,
            "HistogramAccumulator.putWeighted: weight must be nonnegative and fit CountType");
        const CountType added = weight;
        const indices = storageIndices(coordinates);
        if (added == 0) return;
        static if (N == 1)
            counts[indices[0]] += added;
        else static if (isSlice!Storage)
            counts[indices] += added;
        else
            updateArray!true(counts, indices, added);
    }

    /++
    Merge cells from a histogram with matching axes and value type.
    Accumulator cells receive put(sourceCell) when supported; otherwise cells
    merge through +=, as numeric counters and Summator do. This merges full
    state rather than adding reported statistics such as means.
    Corresponding ordinary and underflow/overflow bins are merged. Source and
    destination storage may use different array nesting or ndslice layouts.
    Axes are compared using equality. All axes and both storage shapes are
    checked before any counts change.

    The source can be const. No allocation is performed. Each destination bin
    must have distinct storage; source storage may overlap the destination
    only at corresponding bin positions. Self-merging doubles numeric counts;
    accumulator cells must support merging their own state.
    Other overlapping layouts require a separate copy of the source first.

    Params:
        h = source histogram; its axis types and counter type must match
    +/
    void put(H)(auto ref const H h)
        if (acceptsMerge!H)
    {
        assert(axis == h.axis, "HistogramAccumulator.put: axes must match for merging");
        size_t[N] shape;
        static foreach (i; 0 .. N)
            shape[i] = axisStorageExtent(axis[i]);
        validateStorageShape(counts, shape);
        validateStorageShape(h.counts, shape);
        mergeStorage(counts, h.counts);
    }

    /++
    Sum current counts overflowing the selected axis.
    Includes every combination of bins on the other axes, including enabled
    underflow and overflow bins.
    Corner counts can contribute to totals for multiple axes, so these totals
    must not be added together to count distinct out-of-range observations.
    Takes time proportional to the product of the other storage dimensions;
    does not allocate.
    Params:
        dimension = zero-based axis number; overflow must be enabled on this axis
    +/
    CountType overflow(size_t dimension = 0)() const
        if (isNumeric!CountType && dimension < N && includeOverflow!(Axis[dimension]))
    {
        return axisEndTotal!dimension(counts, axisStorageExtent(axis[dimension]) - 1);
    }

    /++
    Sum current counts underflowing the selected axis.
    Includes every combination of bins on the other axes, including enabled
    underflow and overflow bins.
    Corner counts can contribute to totals for multiple axes, so these totals
    must not be added together to count distinct out-of-range observations.
    Takes time proportional to the product of the other storage dimensions;
    does not allocate.
    Params:
        dimension = zero-based axis number; underflow must be enabled on this axis
    +/
    CountType underflow(size_t dimension = 0)() const
        if (isNumeric!CountType && dimension < N && includeUnderflow!(Axis[dimension]))
    {
        return axisEndTotal!dimension(counts, 0);
    }

}

/++
Accumulate sales revenue by customer age to compare how much each age group
spends. Each purchase amount updates the Summator in the customer's age bin.
The caller selects the summation algorithm used within each bin.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summator, Summation;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;

    alias Cell = Summator!(double, Summation.pairwise);
    alias Age = RegularAxis!(double, AxisOptions());
    Cell[2] revenue;
    auto sales = HistogramAccumulator!(Cell[], Age)(revenue[], Age(2, 20.0, 60.0));
    sales.putSample(30.0, 25.0);
    sales.putSample(50.0, 35.0);
    sales.putSample(120.0, 45.0);

    // Ages [20,40) account for 80; ages [40,60) account for 120.
    // The view reports accumulator state, not a numeric observation count.
    auto bins = sales.bins;
    assert(bins[0].value.sum == 80.0);
    assert(bins[1].value.sum == 120.0);
}

/++
Compare service response times across machine temperatures to assess how
temperature relates to latency. Temperature selects the bin, and response time
updates its MeanAccumulator. Each bin reports the mean latency and request count.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.univariate: MeanAccumulator;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import mir.ndslice.slice: sliced;

    alias Cell = MeanAccumulator!(double, Summation.pairwise);
    alias Temperature = RegularAxis!(double, AxisOptions());
    Cell[2] buffer;
    auto storage = buffer[].sliced;
    auto timings = HistogramAccumulator!(typeof(storage), Temperature)(
        storage, Temperature(2, 20.0, 60.0));
    timings.putSample(100.0, 25.0);
    timings.putSample(200.0, 35.0);
    timings.putSample(400.0, 45.0);

    // The cooler bin has two requests averaging 150 ms; the warmer bin has one.
    auto bins = timings.bins;
    assert(bins[0].value.count == 2);
    assert(bins[0].value.mean == 150.0);
    assert(bins[1].value.mean == 400.0);

    // Reading a bin copies its current state. The view itself sees later updates.
    auto previous = bins[0];
    timings.putSample(300.0, 30.0);
    assert(previous.value.mean == 150.0);
    assert(bins[0].value.mean == 200.0);
}

/++
Regional sensor reports may each summarize a different number of readings.
Weight each report's mean by its reading count to recover the regional mean;
averaging report means without weights would give small reports too much influence.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.weighted: WMeanAccumulator, AssumeWeights;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;

    alias Cell = WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary);
    alias Coordinate = RegularAxis!(double, AxisOptions());
    Cell[2][2] storage;
    auto readings = HistogramAccumulator!(typeof(storage), Coordinate, Coordinate)(
        storage, Coordinate(2, 0.0, 2.0), Coordinate(2, 0.0, 2.0));

    // Weight first, then the reported mean, then latitude and longitude.
    readings.putWeightedSample(2.0, 10.0, 0.5, 1.5);
    readings.putWeightedSample(6.0, 30.0, 0.5, 1.5);
    assert(readings.counts[0][1].weight == 8.0);
    assert(readings.counts[0][1].wmean == 25.0);
    // Nested static storage is copied into the histogram, so inspect its cells.
    assert(storage[0][1].weight == 0.0);
}

// Sum and weighted-mean cells merge their full state across array and slice storage.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.sum: Summator, Summation;
    import mir.stat.descriptive.weighted: WMeanAccumulator, AssumeWeights;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    alias A = IntegralAxis!(int, AxisOptions());
    alias Sum = Summator!(double, Summation.pairwise);
    alias Mean = WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary);

    static foreach (Cell; AliasSeq!(Sum, Mean))
    {{
        Cell[2] buffer, otherBuffer;
        auto h = HistogramAccumulator!(Cell[], A)(buffer[], A(2, 0));
        auto storage = otherBuffer[].sliced;
        auto other = HistogramAccumulator!(typeof(storage), A)(storage, A(2, 0));
        static if (is(Cell == Sum))
        {
            h.putSample(10.0, 0);
            other.putSample(30.0, 0);
        }
        else
        {
            h.putWeightedSample(2.0, 10.0, 0);
            other.putWeightedSample(6.0, 30.0, 0);
        }
        h.put(other);
        static if (is(Cell == Sum))
            assert(h.bins.front.value.sum == 40.0);
        else
        {
            assert(h.bins.front.value.weight == 8.0);
            assert(h.bins.front.value.wmean == 25.0);
        }
        // Self-merging doubles the state, not the resulting mean.
        h.put(h);
        static if (is(Cell == Sum))
            assert(h.bins.front.value.sum == 80.0);
        else
        {
            assert(h.bins.front.value.weight == 16.0);
            assert(h.bins.front.value.wmean == 25.0);
        }
    }}
}

// A view must not expose mutable references held inside a custom cell.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static struct Cell
    {
        int[] data;
        void put(int sample) @safe pure nothrow @nogc { data[0] += sample; }
    }
    alias A = IntegralAxis!(int, AxisOptions());
    int[1] data;
    Cell[1] storage = [Cell(data[])];
    auto h = HistogramAccumulator!(Cell[], A)(storage[], A(1, 0));
    auto view = h.bins;
    auto entry = view.front;
    static assert(!__traits(compiles, { entry.value.data[0] = 10; }));
    static assert(!__traits(compiles, h.put(h))); // No cell merge operation.
    h.putSample(3, 0);
    assert(entry.value.data[0] == 3); // Referenced data is shared, not deep-copied.
}

// Rejected coordinates and incompatible axes leave accumulator cells untouched.
// Catching assertion failures requires @system; the successful paths are tested @safe.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    import mir.math.sum: Summation;
    import mir.stat.descriptive.univariate: MeanAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias Cell = MeanAccumulator!(double, Summation.pairwise);
    alias A = IntegralAxis!(int, AxisOptions());
    Cell[2] row0, row1;
    Cell[][2] rows = [row0[], row1[]];
    auto h = HistogramAccumulator!(Cell[][], A, A)(rows[], A(2, 0), A(2, 0));
    bool rejected;
    try { h.putSample(1.0, 0, 2); }
    catch (AssertError) { rejected = true; }
    assert(rejected);
    assert(row0[0].count == 0);
    auto other = HistogramAccumulator!(Cell[][], A, A)(rows[], A(2, 1), A(2, 0));
    rejected = false;
    try { h.put(other); }
    catch (AssertError) { rejected = true; }
    assert(rejected);
    assert(row0[0].count == 0);
    h.putSample(7.0, 1, 0);
    assert(row1[0].mean == 7.0);
    assert(h.bins[2].value.mean == 7.0);
}

version(mir_stat_test_lifetime)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.univariate: MeanAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias Cell = MeanAccumulator!(double, Summation.pairwise);
    alias A = IntegralAxis!(int, AxisOptions());
    alias Storage = Cell[2];
    alias H = HistogramAccumulator!(Storage, A);
    H h = H(Storage.init, A(2, 0));
    auto borrowed = h.bins;
    h.putSample(12.0, 0);
    assert(borrowed.front.value.mean == 12.0);
    static assert(!__traits(compiles, () @safe {
        H local = H(Storage.init, A(2, 0));
        return local.bins;
    }));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.stat.descriptive.univariate: MeanAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;

    alias Cell = MeanAccumulator!(double, Summation.pairwise);
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    Cell[16] buffer;
    // Preserve an existing cell, then merge from a different storage layout.
    buffer[4].put(10.0);
    auto storage = buffer[].sliced(4, 4).transposed;
    auto h = HistogramAccumulator!(typeof(storage), A, A)(storage, A(2, 0), A(2, 0));
    h.putSample(30.0, -1.0, 0.5);
    h.putSample(80.0, 2.0, 2.0);
    assert(h.counts[0, 1].mean == 20.0);
    assert(h.counts[3, 3].mean == 80.0);

    Cell[4][4] otherStorage;
    auto other = HistogramAccumulator!(typeof(otherStorage), A, A)(
        otherStorage, A(2, 0), A(2, 0));
    other.putSample(50.0, -1.0, 0.5);
    const source = other;
    h.put(source);
    assert(h.counts[0, 1].count == 3);
    assert(h.counts[0, 1].mean == 30.0);
    assert(h.counts[3, 3].mean == 80.0);

    import mir.stat.descriptive.histogram.accumulator: BinCoverage;
    auto all = h.bins!(BinCoverage.all);
    assert(all[1].isUnderflow!0);
    assert(all[1].value.mean == 30.0);
    assert(all.back.isOverflow!0 && all.back.isOverflow!1);
    assert(all.back.value.mean == 80.0);
    const readOnly = h;
    assert(readOnly.bins!(BinCoverage.all)[1].value.count == 3);
    static assert(!__traits(compiles, readOnly.putSample(1.0, 0.5, 0.5)));
    static assert(!__traits(compiles, all[1].value.put(1.0)));
    static assert(!__traits(compiles, h.putSample(1.0, 0.5)));
    static assert(!__traits(compiles, h.putSample(1.0, "bad", 0.5)));
    static assert(!__traits(compiles, h.putWeightedSample(1.0, 2.0, 0.5, 0.5)));
    static assert(!__traits(compiles, h.put(0.5, 0.5)));
    static assert(!__traits(compiles, h.putWeighted(1.0, 0.5, 0.5)));
    assert(h.rcMarginal!0().counts[0].count == 3);
    static assert(!__traits(compiles, all[0].count));
}

/// Accumulate weights in bins without maintaining a separate total.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    double[4] storage = 0; // underflow, two ordinary bins, overflow
    auto h = HistogramAccumulator!(double[], A)(storage[], A(2, 0));
    h.putWeighted(0.5, 0.25);
    h.putWeighted(1.5, 1.25);
    h.put(0.75); // Unweighted insertion still adds one.
    h.putWeighted(2.0, -1.0);
    assert(h.counts == [2.0, 1.5, 1.5, 0]);
    assert(h.underflow == 2);
    h.putWeighted(0.0, 1.25);
    assert(h.counts[2] == 1.5);
}

/// Allocate one-dimensional storage including both underflow and overflow.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    import mir.stat.descriptive.histogram.traits: storageExtent;

    alias A = IntegralAxis!(double,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));
    auto axis = A(3, 0.0);
    // N_bin counts ordinary bins; storageExtent includes the two end bins.
    auto counts = new uint[storageExtent(axis)];
    auto h = HistogramAccumulator!(uint[], A)(counts, axis);
    assert(axis.N_bin == 3 && h.storageExtent() == 5);

    h.put(-1.0, 0.5, 2.5, 3.0);
    assert(h.counts == [1u, 1u, 0u, 1u, 1u]);
    assert(h.underflow == 1 && h.overflow == 1);

    // View indices still describe ordinary bins, without the storage offset.
    auto bins = h.bins();
    assert(bins.length == 3 && bins.front.index == 0);
    assert(bins.front.count == 1 && bins.back.index == 2);

    // Dynamic storage shares all counts, including the end bins.
    counts[0] = 7;
    assert(h.underflow == 7);
}

// Check IntegralAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis = IntegralAxis!(double, AxisOptions())(5, 2.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(integralAxis))(counts, integralAxis);
    h.put([2.0, 2.5, 3.0, 3.5]);
    assert(counts == [2, 2, 0, 0, 0]);
    h.put(4.0);
    assert(counts == [2, 2, 1, 0, 0]);
}

// Check over/underflow IntegralAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis = IntegralAxis!(double, AxisOptions(false, true, true))(5, 2.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(integralAxis))(counts, integralAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
    h.put(3.0);
    assert(counts == [1, 0, 1, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);

    // Both flow counters can be inspected through a const reference.
    void checkFlows(ref const(typeof(h)) histogram) @safe pure nothrow @nogc
    {
        assert(histogram.overflow == 1);
        assert(histogram.underflow == 1);
    }
    checkFlows(h);
}

// Check over/underflow IntegralAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis, EnableUnderflow;

    auto integralAxis = IntegralAxis!(double, AxisOptions(EnableUnderflow(true)))(5, 2.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(integralAxis))(counts, integralAxis);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [1, 0, 0, 0, 0, 0]);
    assert(h.underflow == 1);
    h.put(3.0);
    assert(counts == [1, 0, 1, 0, 0, 0]);
    assert(h.underflow == 1);
}

// Check EnumAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, EnumAxis;

    enum Foo {
        A,
        B
    }
    EnumAxis!(Foo) enumAxis;
    size_t[] counts = [0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(enumAxis))(counts, enumAxis);
    h.put([Foo.A, Foo.B, Foo.B, Foo.B]);
    assert(counts == [1, 3]);
    h.put(Foo.A);
    assert(counts == [2, 3]);
}

// Check CategoryAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, CategoryAxis;

    enum Foo {
        A,
        B
    }
    CategoryAxis!(Foo, AxisOptions(false, true)) categoryAxis;
    size_t[] counts = [0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(categoryAxis))(counts, categoryAxis);
    h.put([Foo.A, Foo.B, Foo.B, Foo.B]);
    assert(counts == [1, 3, 0]);
    h.put(Foo.A);
    assert(counts == [2, 3, 0]);

    // Check strings
    h.put("B");
    assert(counts == [2, 4, 0]);
    assert(h.overflow == 0);
    h.put("C");
    assert(h.overflow == 1);
    h.put(["C", "D"]);
    assert(h.overflow == 3);
    h.put(["CD"]);
    assert(h.overflow == 4);
}

// Check RegularAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    auto regularAxis = RegularAxis!(double, AxisOptions())(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    h.put([2.0, 2.5, 3.0, 11.5]);
    assert(counts == [3, 0, 0, 0, 1]);
    h.put(7.0);
    assert(counts == [3, 0, 1, 0, 1]);
}

// Check over/underflow RegularAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    auto regularAxis = RegularAxis!(double, AxisOptions(false, true, true))(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check RegularAxis, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    auto regularAxis = RegularAxis!(double, AxisOptions(true))(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    h.put([2.5, 3.0, 3.5, 12.0]);
    assert(counts == [3, 0, 0, 0, 1]);
    h.put(7.0);
    assert(counts == [3, 0, 1, 0, 1]);
}

// Check over/underflow RegularAxis, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    auto regularAxis = RegularAxis!(double, AxisOptions(true, true, true))(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check TransformAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: log10;
    import mir.stat.descriptive.histogram.axis: AxisOptions, TransformAxis, inverseTransformMapping;

    auto transformAxis = TransformAxis!(double, log10, inverseTransformMapping!log10, AxisOptions())(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    h.put([10.0 ^^ 2.0, 10.0 ^^ 2.5, 10.0 ^^ 3.0, 10.0 ^^ 11.5]);
    assert(counts == [3, 0, 0, 0, 1]);
    h.put(10.0 ^^ 7.0);
    assert(counts == [3, 0, 1, 0, 1]);
}

// Check over/underflow TransformAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: log10;
    import mir.stat.descriptive.histogram.axis: AxisOptions, TransformAxis, inverseTransformMapping;

    auto transformAxis = TransformAxis!(double, log10, inverseTransformMapping!log10, AxisOptions(false, true, true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check Transform, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: log10;
    import mir.stat.descriptive.histogram.axis: AxisOptions, TransformAxis, inverseTransformMapping;

    auto transformAxis = TransformAxis!(double, log10, inverseTransformMapping!log10, AxisOptions(true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    h.put([10.0 ^^ 2.5, 10.0 ^^ 3.0, 10.0 ^^ 3.5, 10.0 ^^ 12.0]);
    assert(counts == [3, 0, 0, 0, 1]);
    h.put(10.0 ^^ 7.0);
    assert(counts == [3, 0, 1, 0, 1]);
}

// Check over/underflow TransformAxis, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: log10;
    import mir.stat.descriptive.histogram.axis: AxisOptions, TransformAxis, inverseTransformMapping;

    auto transformAxis = TransformAxis!(double, log10, inverseTransformMapping!log10, AxisOptions(true, true, true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check VariableAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, VariableAxis;

    auto axisSlice = [2.0, 3, 4, 5, 6, 7].sliced;
    auto variableAxis = VariableAxis!(double*, AxisOptions())(axisSlice);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(variableAxis))(counts, variableAxis);
    h.put([2.0, 2.5, 3.0, 3.5]);
    assert(counts == [2, 2, 0, 0, 0]);
    h.put(4.0);
    assert(counts == [2, 2, 1, 0, 0]);
}

// Check over/underflow VariableAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, VariableAxis;

    auto axisSlice = [2.0, 3, 4, 5, 6, 7].sliced;
    auto variableAxis = VariableAxis!(double*, AxisOptions(false, true, true))(axisSlice);
    size_t[] counts = [0, 0, 0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(variableAxis))(counts, variableAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [1, 0, 0, 0, 0, 0, 1]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check put HistogramAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis1 = IntegralAxis!(double, AxisOptions())(5, 2.0);
    size_t[] counts1 = [0, 0, 0, 0, 0];
    auto integralAxis2 = IntegralAxis!(double, AxisOptions())(5, 2.0);
    size_t[] counts2 = [0, 0, 0, 0, 0];

    auto h1 = HistogramAccumulator!(size_t[], typeof(integralAxis1))(counts1, integralAxis1);
    h1.put([2.0, 2.5, 3.0, 3.5, 4.0]);
    assert(counts1 == [2, 2, 1, 0, 0]);
    auto h2 = HistogramAccumulator!(size_t[], typeof(integralAxis2))(counts2, integralAxis2);
    h2.put([4.0, 5.0, 5.5, 6.0, 6.5]);
    assert(counts2 == [0, 0, 1, 2, 2]);
    h2.put(h1);
    assert(counts2 == [2, 2, 2, 2, 2]);
}

// Check put HistogramAccumulator with over/underflow
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        EnableOverflow, EnableUnderflow;

    auto integralAxis1 = IntegralAxis!(double, AxisOptions(EnableOverflow(true), EnableUnderflow(true)))(5, 2.0);
    size_t[] counts1 = [0, 0, 0, 0, 0, 0, 0];
    auto integralAxis2 = IntegralAxis!(double, AxisOptions(EnableOverflow(true), EnableUnderflow(true)))(5, 2.0);
    size_t[] counts2 = [0, 0, 0, 0, 0, 0, 0];

    auto h1 = HistogramAccumulator!(size_t[], typeof(integralAxis1))(counts1, integralAxis1);
    h1.put(-1.0);
    auto h2 = HistogramAccumulator!(size_t[], typeof(integralAxis2))(counts2, integralAxis2);
    h2.put(9.0);
    h2.put(h1);
    assert(h2.overflow == 1);
    assert(h2.underflow == 1);
}

// Check custom CircleAxis
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions;

    struct Point
    {
        double x;
        double y;
    }

    static struct CircleAxis(AxisOptions axisOptions)
    {
        alias CountType = size_t;
        alias BinType = Point;

        enum CountType N_bin = 1;

        CountType index(BinType x)
        {
            if (!isOverflow(x)) {
                return 0;
            } else {
                assert(0, "index: input may not overflow");
            }
        }

        bool isOverflow()(BinType x) const
        {
            return x.x * x.x + x.y + x.y < 1.0;
        }
    }

    auto circleAxis = CircleAxis!(AxisOptions())();
    size_t[2] count = 0;

    auto h = HistogramAccumulator!(size_t[2], typeof(circleAxis))(count, circleAxis);
    auto p = Point(0.25, 0.5);
    h.put(p);
}

/// Count pairs of observations in a joint histogram backed by built-in arrays.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias A = IntegralAxis!(double, AxisOptions());
    uint[][] counts = [[0u, 0u, 0u], [0u, 0u, 0u]];
    auto h = HistogramAccumulator!(uint[][], A, A)(counts, A(2, 0.0), A(3, 0.0));
    h.put(0.5, 1.5);
    h.put(0.75, 1.25);
    h.put(1.5, 2.5);

    // Rows select the first axis; columns select the second. Each pair adds
    // exactly one count, and the caller's dynamic arrays see the updates.
    assert(counts[0] == [0u, 2u, 0u]);
    assert(counts[1] == [0u, 0u, 1u]);
}

/// Use a two-dimensional ndslice over caller-supplied storage without allocation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias A = IntegralAxis!(double, AxisOptions());
    uint[6] buffer;
    auto counts = buffer[].sliced(2, 3);
    auto h = HistogramAccumulator!(typeof(counts), A, A)(
        counts, A(2, 0.0), A(3, 0.0));
    h.put(0.5, 1.5);
    h.put(1.5, 2.5);

    // The same joint coordinates are used with ndslice indexing. Its handle
    // shares the buffer, so updates are visible through either representation.
    assert(counts[0, 1] == 1 && counts[1, 2] == 1);
    assert(buffer == [0u, 1u, 0u, 0u, 0u, 1u]);
}

/// Count triples with one storage dimension per axis.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());

    // In D, the rightmost static-array dimension is the outermost: this
    // storage has two planes, three rows per plane, and four counts per row.
    uint[4][3][2] storage;
    auto h = HistogramAccumulator!(typeof(storage), A, A, A)(
        storage, A(2, 0.0), A(3, 0.0), A(4, 0.0));
    h.put(0.5, 1.5, 2.5);
    h.put(0.75, 1.25, 2.75);
    h.put(1.5, 2.5, 3.5);

    // Each triple increments exactly one joint bin. Fully static storage
    // is copied into the accumulator, so read the accumulated counts from h.
    assert(h.counts[0][1][2] == 2);
    assert(h.counts[1][2][3] == 1);
    assert(storage[0][1][2] == 0);
}

/// Joint flow bins retain both coordinates; member functions sum along one axis.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias A = IntegralAxis!(double,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));

    // Two ordinary x bins and three ordinary y bins, with an extra bin at
    // each end of both axes, require a four-row, five-column grid.
    uint[5][4] storage;
    auto h = HistogramAccumulator!(typeof(storage), A, A)(
        storage, A(2, 0.0), A(3, 0.0));
    h.put(-1.0, 1.5); // x underflow, ordinary y bin 1
    h.put(-1.0, 3.0); // x underflow, y overflow: a corner bin
    h.put(0.5, 3.0);  // ordinary x bin 0, y overflow

    // Storage index zero is underflow; ordinary indices are shifted by one.
    // The last index is overflow. The corner retains the association between
    // the two out-of-range coordinates, separately from either ordinary bin.
    assert(h.counts[0][2] == 1);
    assert(h.counts[0][4] == 1);
    assert(h.counts[1][4] == 1);

    // Choose the axis at compile time. These totals are calculated from the
    // current grid, including the other axis's underflow and overflow bins.
    // The corner contributes to both totals; adding them would count it twice.
    assert(h.underflow!0() == 2);
    assert(h.overflow!1() == 2);
    assert(h.overflow!0() == 0 && h.underflow!1() == 0);

    // Reading totals is also supported on a const histogram.
    const reader = h;
    assert(reader.underflow!0() == 2);
}

/// Three-dimensional ndslices retain joint underflow/overflow combinations.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias X = IntegralAxis!(double, AxisOptions(EnableUnderflow(true)));
    alias Y = IntegralAxis!(double, AxisOptions());
    alias Z = IntegralAxis!(double, AxisOptions(EnableOverflow(true)));

    // Two ordinary bins per axis, plus x underflow and z overflow, require
    // a 3-by-2-by-3 grid. The ndslice shares the caller's buffer.
    uint[18] buffer;
    auto counts = buffer[].sliced(3, 2, 3);
    auto h = HistogramAccumulator!(typeof(counts), X, Y, Z)(
        counts, X(2, 0.0), Y(2, 0.0), Z(2, 0.0));
    h.put(-1.0, 0.5, 2.0);
    h.put(-1.0, 1.5, 0.5);
    h.put(0.5, 1.5, 2.0);

    // Underflow occupies x storage index zero; z overflow occupies its last
    // index. The first observation is retained at their intersection.
    assert(counts[0, 0, 2] == 1);
    assert(counts[0, 1, 0] == 1);
    assert(counts[1, 1, 2] == 1);

    // Each member now sums a plane across the other two dimensions.
    // Their intersection contributes to both totals, but is stored only once.
    assert(h.underflow!0() == 2);
    assert(h.overflow!2() == 2);
    const reader = h;
    assert(reader.overflow!2() == 2);
}

/// Merge independently accumulated joint counts, including underflow/overflow bins.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias X = IntegralAxis!(double, AxisOptions(EnableUnderflow(true)));
    alias Y = IntegralAxis!(double, AxisOptions(EnableOverflow(true)));
    uint[3][3] storage;
    alias H = HistogramAccumulator!(typeof(storage), X, Y);
    auto first = H(storage, X(2, 0.0), Y(2, 0.0));
    auto second = H(storage, X(2, 0.0), Y(2, 0.0));
    first.put(0.5, 0.5);
    second.put(0.5, 0.5);
    second.put(-1.0, 2.0);

    // Both ordinary counts and the joint underflow/overflow corner are added.
    // The source can be const, and independent source storage is unchanged.
    const source = second;
    first.put(source);
    assert(first.counts[1][0] == 2);
    assert(first.counts[0][2] == 1);
    assert(first.underflow!0() == 1 && first.overflow!1() == 1);
    assert(source.counts[1][0] == 1 && source.counts[0][2] == 1);
}

/// Obtain a one-dimensional marginal by summing over the other axis.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.rc: rcMarginal;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    auto joint = HistogramAccumulator!(uint[][], A, A)(
        [[1u, 4u, 2u], [0u, 3u, 1u]], A(2, 0), A(3, 10));

    // Keep axis zero: each result count is the sum of one source row.
    auto first = joint.rcMarginal!0();
    assert(first.counts == [7u, 4u]);
    assert(first.bins.front.bin.low == 0);

    // Keep axis one instead: sum down each column, preserving that axis.
    auto second = joint.rcMarginal!1();
    assert(second.counts == [1u, 7u, 3u]);
    assert(second.bins.front.bin.low == 10);

    // Counts are an independent snapshot. Both histograms remain usable.
    joint.put(0, 10);
    assert(first.counts == [7u, 4u]);
    first.put(1);
    assert(first.counts == [7u, 5u] && joint.counts[1][0] == 0);
}

/// Discarded underflow/overflow bins still contribute to the marginal.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.api.rc: rcMarginal;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions(false, true, true));
    auto joint = HistogramAccumulator!(uint[][], A, A)(
        [[1u, 2u, 3u], [4u, 5u, 6u], [7u, 8u, 9u]], A(1, 0), A(1, 0));
    auto marginal = joint.rcMarginal!0();

    // Each row includes underflow, ordinary, and overflow on the discarded axis.
    // Retained-axis end bins remain distinct, so the corner counts are not lost.
    assert(marginal.underflow == 6);
    assert(marginal.bins.front.count == 15);
    assert(marginal.overflow == 24);
    uint total;
    foreach (entry; marginal.bins!(BinCoverage.all)()) total += entry.count;
    assert(total == 45);
}

// Circular endpoints must reach indexing even when flow counters are enabled.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.common: sqrt;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        RegularAxis, TransformAxis, VariableAxis;

    double square(double x) { return x * x; }

    void check(Axis)(Axis axis)
    {
        size_t[] counts = [0, 0, 0, 0];
        auto h = HistogramAccumulator!(size_t[], Axis)(counts, axis);
        assert(!axis.isOverflow(axis.high));
        assert(!axis.isUnderflow(axis.low));

        h.put(axis.high);
        h.put(axis.low);
        static if (Axis.options.isRightClosed)
            assert(h.counts[1 .. 3] == [0, 2]);
        else
            assert(h.counts[1 .. 3] == [2, 0]);
        assert(h.overflow == 0);
        assert(h.underflow == 0);

        h.put(axis.high + 1.0);
        h.put(axis.low - 1.0);
        assert(h.overflow == 1);
        assert(h.underflow == 1);
        static if (Axis.options.isRightClosed)
            assert(h.counts[1 .. 3] == [0, 2]);
        else
            assert(h.counts[1 .. 3] == [2, 0]);
    }

    static foreach (rightClosed; [false, true])
    {{
        enum options = AxisOptions(rightClosed, true, true, true);
        check(IntegralAxis!(double, options)(2, 1.0));
        check(RegularAxis!(double, options)(2, 1.0, 9.0));
        check(TransformAxis!(double, sqrt, square, options)(2, 1.0, 9.0));
        check(VariableAxis!(double*, options)([1.0, 4.0, 9.0].sliced));
    }}
}

/// Select the bins included in a histogram or relative frequency view.
enum BinCoverage
{
    /// Visit only ordinary bins.
    ordinary,
    /// Visit all stored bins, including enabled underflow/overflow bins.
    all,
}

/++
Axis-specific descriptions and the value read when a bin was accessed.

Indices refer to the original ordinary bins, including after slicing.
For underflow/overflow coordinates, index and bin assert; inspect isOrdinary,
isUnderflow, or isOverflow before accessing ordinary-bin metadata.
Elements are returned by value; assigning a numeric count does not change the
histogram. Accumulator values returned by views are const; referenced resources
remain shared rather than being deep-copied.
Select an axis with index!dimension or bin!dimension; dimension defaults to zero.

Params:
    Count = stored numeric value or accumulator type
    BinDescriptions = description type for each axis
+/
struct HistogramBin(Count, BinDescriptions...)
{
    import mir.functional: Tuple;
    private enum Kind { ordinary, underflow, overflow }
    private Kind[BinDescriptions.length] _kinds;
    private size_t[BinDescriptions.length] _indices;
    private Tuple!BinDescriptions _bins;

    /++
    Write bin coordinates and the recorded value to an output range.
    Numeric values use the label count; accumulator values use value and must
    support Mir formatting when this method is called.
    Numeric coordinates use labeled bounds because descriptions do not retain
    endpoint-closure options. Joint coordinates appear in axis order.
    Underflow/overflow coordinates use their names instead of ordinary bounds.
    Use standard range formatting on bins() to format multiple entries.
    Mir formatting uses a small expandable buffer without GC allocation.
    Attributes depend on the output writer and the values being formatted.
    +/
    void toString(Writer)(ref Writer writer) const
    {
        import mir.format: print;
        import mir.appender: scopedBuffer;
        import std.range.primitives: put;
        // Mir printers need both character and string put overloads. Buffering
        // also supports the character-only writer used by std.format.
        auto buffer = scopedBuffer!(char, 256);
        formatCoordinates(buffer);
        static if (isNumeric!Count)
            print(buffer, ": count=", count);
        else
            print(buffer, ": value=", value);
        put(writer, buffer.data);
    }

    package(mir.stat.descriptive.histogram)
    void formatCoordinates(Writer)(ref Writer writer) const
    {
        import mir.format: print;
        import std.range.primitives: put;
        import std.traits: hasMember;

        put(writer, "bin(");
        static foreach (dimension; 0 .. BinDescriptions.length)
        {
            static if (dimension != 0)
                put(writer, ", ");
            static if (BinDescriptions.length > 1)
                print(writer, "axis", dimension, "=");
            if (isUnderflow!dimension)
                put(writer, "underflow");
            else if (isOverflow!dimension)
                put(writer, "overflow");
            else
            {
                static if (hasMember!(BinDescriptions[dimension], "low") &&
                    hasMember!(BinDescriptions[dimension], "high"))
                {
                    static if (BinDescriptions.length > 1) put(writer, "(");
                    print(writer, "low=",
                        _bins[dimension].low, ", high=", _bins[dimension].high);
                    static if (BinDescriptions.length > 1) put(writer, ")");
                }
                else static if (hasMember!(BinDescriptions[dimension], "slot"))
                    print(writer, "slot=", _bins[dimension].slot);
                else
                    print(writer, _bins[dimension]);
            }
        }
        put(writer, ")");
    }

    /// Stored value at the time this element was read.
    Count value;

    /// Numeric count; an alias for value.
    static if (isNumeric!Count)
        alias count = value;

    /// Whether this coordinate is an ordinary bin; dimension defaults to zero.
    bool isOrdinary(size_t dimension = 0)() const @property
        if (dimension < BinDescriptions.length)
    {
        return _kinds[dimension] == Kind.ordinary;
    }

    /// Whether this coordinate is underflow; dimension defaults to zero.
    bool isUnderflow(size_t dimension = 0)() const @property
        if (dimension < BinDescriptions.length)
    {
        return _kinds[dimension] == Kind.underflow;
    }

    /// Whether this coordinate is overflow; dimension defaults to zero.
    bool isOverflow(size_t dimension = 0)() const @property
        if (dimension < BinDescriptions.length)
    {
        return _kinds[dimension] == Kind.overflow;
    }

    /++
    Original ordinary-bin index along one axis.
    Params:
        dimension = zero-based axis number, defaulting to zero
    +/
    size_t index(size_t dimension = 0)() const @property
        if (dimension < BinDescriptions.length)
    {
        assert(isOrdinary!dimension, "HistogramBin.index: coordinate is not an ordinary bin");
        return _indices[dimension];
    }

    /++
    Description of the bin along one axis.
    Params:
        dimension = zero-based axis number, defaulting to zero
    +/
    auto bin(size_t dimension = 0)() @property
        if (dimension < BinDescriptions.length)
    {
        assert(isOrdinary!dimension, "HistogramBin.bin: coordinate is not an ordinary bin");
        return _bins[dimension];
    }

    /// ditto
    auto bin(size_t dimension = 0)() const @property
        if (dimension < BinDescriptions.length)
    {
        assert(isOrdinary!dimension, "HistogramBin.bin: coordinate is not an ordinary bin");
        return _bins[dimension];
    }
}

/// Format one bin or a range of bins using standard D formatting.
version(mir_stat_test)
@safe pure
unittest
{
    import std.format: format;
    import mir.format: text;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(double, AxisOptions());
    auto h = HistogramAccumulator!(uint[], A)([2u, 1u], A(2, 0.0));
    const entry = h.bins.front;
    assert(format("%s", entry) == "bin(low=0.0, high=1.0): count=2");
    assert(text(entry) == "bin(low=0.0, high=1.0): count=2");
    // Standard range formatting supplies the brackets and separators.
    assert(format("%s", h.bins) ==
        "[bin(low=0.0, high=1.0): count=2, bin(low=1.0, high=2.0): count=1]");
    // Bounds are labeled; they do not imply an endpoint-closure convention.
}

/// Print a histogram with writeln or writefln, or choose precision per field.
version(mir_stat_test)
@safe
unittest
{
    import std.stdio: writeln, writefln;
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;

    alias A = IntegralAxis!(double, AxisOptions());
    auto h = HistogramAccumulator!(uint[], A)([2u, 1u], A(2, 0.0));

    // Call printHistogram(h) to write to stdout. The helper is compiled but
    // deliberately not called here, keeping documentation tests silent.
    void printHistogram(typeof(h) histogram)
    {
        writeln(histogram.bins());
        writefln("Histogram: %s", histogram.bins());
        foreach (entry; histogram.bins())
        {
            writeln(entry);
            // Format fields individually to control their numeric precision.
            writefln("low=%.2f, high=%.2f: count=%s",
                entry.bin.low, entry.bin.high, entry.count);
        }
    }

    // Check the corresponding text without performing console I/O.
    assert(format("%s", h.bins()) ==
        "[bin(low=0.0, high=1.0): count=2, bin(low=1.0, high=2.0): count=1]");
    assert(format("Histogram: %s", h.bins()) ==
        "Histogram: [bin(low=0.0, high=1.0): count=2, bin(low=1.0, high=2.0): count=1]");
    auto entry = h.bins().front;
    assert(format("low=%.2f, high=%.2f: count=%s",
        entry.bin.low, entry.bin.high, entry.count) ==
        "low=0.00, high=1.00: count=2");
}

// Joint coordinates and end-bin labels do not require ordinary-bin metadata.
version(mir_stat_test)
@safe pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    auto h = HistogramAccumulator!(uint[][], A, A)(
        [[0u, 1u, 0u], [0u, 2u, 0u], [0u, 0u, 3u]], A(1, 0.0), A(1, 0.0));
    auto entries = h.bins!(BinCoverage.all);
    assert(format("%s", entries[1]) == "bin(axis0=underflow, axis1=(low=0.0, high=1.0)): count=1");
    assert(format("%s", entries[8]) == "bin(axis0=overflow, axis1=overflow): count=3");
    assert(format("%s", entries[4]) == "bin(axis0=(low=0.0, high=1.0), axis1=(low=0.0, high=1.0)): count=2");
}

// Category labels, custom descriptions, and output-range writers.
version(mir_stat_test)
@safe pure
unittest
{
    import std.format: format;
    import mir.stat.descriptive.histogram.axis: EnumAxis;
    enum Color { red, blue }
    auto h = HistogramAccumulator!(uint[], EnumAxis!(Color))([1u, 2u], EnumAxis!(Color)());
    assert(format("%s", h.bins[1]) == "bin(slot=blue): count=2");

    static struct Description
    {
        string toString() const { return "custom"; }
    }
    HistogramBin!(uint, Description) custom;
    custom.count = 7;
    assert(format("%s", custom) == "bin(custom): count=7");

    static struct Writer
    {
        char[128] buffer;
        size_t length;
        void put(scope const(char)[] text) @safe pure nothrow @nogc
        {
            assert(length + text.length <= buffer.length);
            buffer[length .. length + text.length] = text;
            length += text.length;
        }
        void put(char value) @safe pure nothrow @nogc
        {
            assert(length < buffer.length);
            buffer[length++] = value;
        }
    }
    Writer writer;
    const entry = h.bins[0];
    entry.toString(writer);
    assert(writer.buffer[0 .. writer.length] == "bin(slot=red): count=1");
}

// Category lookup and joint entry formatting preserve the complete attribute
// set when counts and the output writer use reference-counted or scoped storage.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.appender: scopedBuffer;
    import mir.format: print;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: CategoryAxis, IntegralAxis, AxisOptions;

    enum Label { first, second }
    alias C = CategoryAxis!(Label, AxisOptions());
    alias A = IntegralAxis!(int, AxisOptions());
    auto counts = rcslice!uint(2, 2);
    auto h = HistogramAccumulator!(typeof(counts), C, A)(counts, C(), A(2, 0));
    // Both enum values and their string names use the same category.
    h.put(Label.first, 0);
    h.put("first", 0);
    h.put(Label.second, 1);

    auto writer = scopedBuffer!(char, 256);
    print(writer, h.bins().front);
    assert(writer.data == "bin(axis0=slot=first, axis1=(low=0, high=1)): count=2");
    assert(h.bins().back.count == 1);
}

// Numeric formatting into caller-provided storage is GC-free.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    static struct Writer
    {
        char[128] buffer;
        size_t length;
        void put(scope const(char)[] text) @safe pure nothrow @nogc
        {
            assert(length + text.length <= buffer.length);
            buffer[length .. length + text.length] = text;
            length += text.length;
        }
        void put(char value) @safe pure nothrow @nogc
        {
            assert(length < buffer.length);
            buffer[length++] = value;
        }
    }
    alias A = IntegralAxis!(double, AxisOptions());
    uint[2] counts = [2, 1];
    auto h = HistogramAccumulator!(uint[], A)(counts[], A(2, 0.0));
    const entry = h.bins.front;
    Writer writer;
    entry.toString(writer);
    assert(writer.buffer[0 .. writer.length] == "bin(low=0.0, high=1.0): count=2");
}

/++
Read-only random-access range of bins and counts in any dimension.

Usually obtained from a histogram's bins accessor. Includes zero-count bins;
coverage selects ordinary bins or all enabled underflow/overflow bins.
Every selected combination is visited once. The last axis advances fastest, regardless
of storage strides. Element indices retain their original per-axis coordinates.
Numeric descriptions expose low and high; category descriptions expose slot.

The view copies read-only axis and storage handles without allocating a result
array. Nested arrays with a dynamic outer dimension and Mir ndslices are supported.
Shared count updates are visible on later reads; previously returned counts are
values. Replacing the source's handles does not redirect an existing view.

Reference-counted handles retain ownership. Borrowed count storage and variable
axis boundaries must outlive the view. Built-in numeric bin descriptions copy
their bounds; custom descriptions may still refer to boundary storage.
Keep storage shape and shared axis boundaries unchanged. The accumulator's
static-array accessor borrows its internal counts instead of copying them.
Custom axes must support mir.qualifier.lightConst and const bin access that
cannot mutate shared boundaries.

Const sources produce mutable cursors over read-only data. A const view can be
indexed, saved, and sliced; save and slicing return independent mutable cursors.

Params:
    Storage = nested array with dynamic outer dimension or Mir ndslice of counts
    coverage = ordinary bins or all enabled stored bins
    Axis = axis types with const runtime bin-description access
+/
struct HistogramBinView(Storage, BinCoverage coverage, Axis...)
    if (supportsBinView!(Storage, Axis) &&
        (coverage == BinCoverage.ordinary || coverage == BinCoverage.all))
{
    import std.meta: staticMap;
    import mir.stat.descriptive.histogram.traits: includeUnderflow, includeOverflow;
    enum N = Axis.length;
    private alias ReadOnlyStorage = typeof(lightConst((const Storage).init));
    private template ReadOnlyAxisOf(A) { alias ReadOnlyAxisOf = typeof(lightConst((const A).init)); }
    private template DescriptionOf(A) { alias DescriptionOf = typeof((const A).init.bin(size_t.init)); }
    private alias ReadOnlyAxes = staticMap!(ReadOnlyAxisOf, Axis);
    private ReadOnlyStorage _counts;
    private ReadOnlyAxes _axes;
    private size_t[N] _shape;
    private size_t _begin;
    private size_t _end;

    static if (isSlice!ReadOnlyStorage)
        private alias StoredValue = DeepElementType!ReadOnlyStorage;
    else
        private alias StoredValue = JointArrayInfo!ReadOnlyStorage.Element;
    static if (isNumeric!StoredValue)
        private alias Count = Unqual!StoredValue;
    else
        private alias Count = const(Unqual!StoredValue);

    /// Type of each returned value.
    alias Element = HistogramBin!(Count, staticMap!(DescriptionOf, ReadOnlyAxes));

    /// Construct a view over storage matching the axes, including enabled joint end bins.
    this(const Storage counts, const Axis axes)
    {
        alias H = HistogramAccumulator!(Storage, Axis);
        size_t[N] storageShape;
        size_t length = 1;
        static foreach (i; 0 .. N)
        {{
            _shape[i] = ordinaryBinCount(axes[i]);
            storageShape[i] = H.axisStorageExtent(axes[i]);
            const extent = coverage == BinCoverage.all ? storageShape[i] : _shape[i];
            assert(extent == 0 || length <= size_t.max / extent,
                "HistogramBinView: traversal length overflows size_t");
            length *= extent;
            _axes[i] = lightConst(axes[i]);
        }}
        H.validateStorageShape(counts, storageShape);
        _counts = lightConst(counts);
        _end = length;
    }

    // Passing by value recurses in constructor resolution on LDC 1.28.1
    // when storage has a copy constructor. Copy the owning handles below.
    private this(ref const HistogramBinView source, size_t begin, size_t end)
    {
        _counts = lightConst(source._counts);
        static foreach (i; 0 .. N)
            _axes[i] = lightConst(source._axes[i]);
        _shape = source._shape;
        _begin = begin;
        _end = end;
    }

    private static Count readArrayCount(size_t depth = 0, S)(auto ref const S counts,
        const ref size_t[N] indices)
    {
        static if (depth + 1 == N)
            return counts[indices[depth]];
        else
            return readArrayCount!(depth + 1)(counts[indices[depth]], indices);
    }

    /// Number of remaining bins in the selected coverage.
    size_t length() const @property { return _end - _begin; }

    /// Whether all bins in this range have been consumed.
    bool empty() const @property { return _begin == _end; }

    /// First remaining element, returned by value.
    Element front() const @property
    {
        assert(!empty, "HistogramBinView.front: empty range");
        return this[0];
    }

    /// Last remaining element, returned by value.
    Element back() const @property
    {
        assert(!empty, "HistogramBinView.back: empty range");
        return this[length - 1];
    }

    /// Advance past the first remaining bin.
    void popFront()
    {
        assert(!empty, "HistogramBinView.popFront: empty range");
        ++_begin;
    }

    /// Remove the last remaining bin from this range.
    void popBack()
    {
        assert(!empty, "HistogramBinView.popBack: empty range");
        --_end;
    }

    /// Copy the traversal position, sharing the backing buffers.
    auto save() const @property
    {
        return HistogramBinView(this, _begin, _end);
    }

    /// Read an element relative to the current range.
    Element opIndex(size_t index) const
    {
        assert(index < length, "HistogramBinView: index is out of range");
        return readElement(_counts, _begin + index, _shape, _axes);
    }

    // Shared with relative frequency views without retaining handles inside their
    // scope-bound cursors. Callers validate shape at view construction.
    package(mir.stat.descriptive.histogram)
    static Element readElement(S, A...)(const S counts, size_t flat,
        const ref size_t[N] shape, const A axes)
        if (A.length == N)
    {
        Element result;
        size_t[N] storageIndices;
        static foreach (reverse; 0 .. N)
        {{
            enum dimension = N - 1 - reverse;
            enum hasUnderflow = includeUnderflow!(Axis[dimension]);
            enum hasOverflow = includeOverflow!(Axis[dimension]);
            const extent = shape[dimension] +
                (coverage == BinCoverage.all ? hasUnderflow + hasOverflow : 0);
            const position = flat % extent;
            flat /= extent;
            static if (coverage == BinCoverage.all)
            {
                storageIndices[dimension] = position;
                if (hasUnderflow && position == 0)
                    result._kinds[dimension] = Element.Kind.underflow;
                else if (hasOverflow && position == shape[dimension] + hasUnderflow)
                    result._kinds[dimension] = Element.Kind.overflow;
                else
                    result._indices[dimension] = position - hasUnderflow;
            }
            else
            {
                result._indices[dimension] = position;
                storageIndices[dimension] = position + hasUnderflow;
            }
            // End bins have no ordinary interval or category description.
            if (result.isOrdinary!dimension)
            {
                const readOnlyAxis = lightConst(axes[dimension]);
                result._bins[dimension] = readOnlyAxis.bin(result._indices[dimension]);
            }
        }}
        static if (isSlice!S)
            return Element(result._kinds, result._indices, result._bins, counts[storageIndices]);
        else
            return Element(result._kinds, result._indices, result._bins,
                readArrayCount(counts, storageIndices));
    }

    /// Return a subrange; element indices still refer to the original histogram.
    auto opSlice(size_t begin, size_t end) const
    {
        assert(begin <= end && end <= length,
            "HistogramBinView: slice is out of range");
        return HistogramBinView(this, _begin + begin, _begin + end);
    }

    /// Copy the full remaining range.
    auto opSlice() const { return save; }

    /// Support $ in index and slice expressions.
    size_t opDollar() const { return length; }
}

/// Iterate over numeric bins alongside their counts.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(double, AxisOptions());
    auto h = HistogramAccumulator!(uint[], Axis)([0u, 0u, 0u], Axis(3, 0.0));
    h.put([0.5, 1.0, 1.5, 2.5]);

    uint total;
    foreach (entry; h.bins)
    {
        assert(entry.bin.low == entry.index);
        assert(entry.bin.high == entry.index + 1);
        total += entry.count;
    }
    assert(total == 4);
}

/// Select each axis's index and description while traversing a joint histogram.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis, AxisOptions;
    alias X = IntegralAxis!(double, AxisOptions());
    enum Color { red, blue, green }
    alias Y = CategoryAxis!(Color, AxisOptions());
    uint[][] counts = [[1u, 2u, 3u], [4u, 5u, 6u]];
    auto h = HistogramAccumulator!(typeof(counts), X, Y)(counts, X(2, 0.0), Y());
    auto view = h.bins;

    // The last axis advances fastest: position four is ordinary bin (1, 1).
    // Each axis keeps its own description type: interval for X, category for Y.
    auto entry = view[4];
    assert(entry.index!0 == 1 && entry.index!1 == 1);
    assert(entry.bin!0.low == 1.0 && entry.bin!0.high == 2.0);
    assert(entry.bin!1.slot == Color.blue);
    assert(entry.count == 5);

    // Omitting the dimension selects axis zero, also in multidimensional views.
    assert(entry.index == entry.index!0);
    assert(entry.bin.low == entry.bin!0.low);

    // Slices retain original coordinates. Later reads see updated counts;
    // entries already returned keep their count values.
    auto tail = view[3 .. $];
    assert(tail.front.index!0 == 1 && tail.front.index!1 == 0);
    h.put(1.5, Color.blue);
    assert(view[4].count == 6 && entry.count == 5);
}

/// Joint traversal skips underflow/overflow bins while retaining ordinary indices.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias A = IntegralAxis!(double,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));
    uint[16] buffer;
    auto counts = buffer[].sliced(4, 4);
    auto h = HistogramAccumulator!(typeof(counts), A, A)(counts, A(2, 0.0), A(2, 0.0));
    h.put(-1.0, 0.5);
    h.put(0.5, 2.0);
    h.put(1.5, 0.5);

    // Four ordinary joint bins are visited, despite the sixteen storage cells.
    // Storage offsets for the underflow bins are handled by the view.
    const view = h.bins;
    assert(view.length == 4);
    assert(view[2].index!0 == 1 && view[2].index!1 == 0);
    assert(view[2].count == 1);
    uint total;
    foreach (entry; view.save)
        total += entry.count;
    assert(total == 1);
}

/// Index and slice a view without losing the original bin indices.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(double, AxisOptions());
    auto h = HistogramAccumulator!(uint[], Axis)([1u, 2u, 1u], Axis(3, 0.0));
    auto bins = h.bins;
    assert(bins[1].count == 2);
    assert(bins[1].bin.low == 1.0);

    auto middle = bins[1 .. $];
    assert(middle.length == 2);
    assert(middle.front.index == 1);
    assert(middle.back.index == 2);
}

/// Category bins expose a slot instead of interval boundaries.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, CategoryAxis;

    enum Label { first, second }
    alias Axis = CategoryAxis!(Label, AxisOptions());
    auto h = HistogramAccumulator!(uint[], Axis)([0u, 0u], Axis());
    h.put([Label.first, Label.second, Label.second]);

    auto bins = h.bins;
    assert(bins[0].bin.slot == Label.first);
    assert(bins[0].count == 1);
    assert(bins[1].bin.slot == Label.second);
    assert(bins[1].count == 2);
}

/// Saved views have independent positions and share subsequent count updates.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(double, AxisOptions());
    auto counts = rcslice!uint([1u, 2u]);
    auto h = HistogramAccumulator!(typeof(counts), Axis)(counts, Axis(2, 0.0));
    auto bins = h.bins;
    auto saved = bins.save;
    auto previous = bins.front;
    bins.popFront();
    h.put(0.5);

    assert(bins.front.index == 1);
    assert(saved.front.index == 0);
    assert(saved.front.count == 2);
    assert(previous.count == 1);

    previous.count = 100;
    assert(h.counts[0] == 2);
}

/// Const access preserves live counts while allowing independent traversal.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(uint[], Axis);
    auto h = H([1u, 2u, 3u], Axis(3, 0.0));

    // A reporting function needs only const access to the histogram.
    auto readBins(ref const H histogram) { return histogram.bins; }
    const fixed = readBins(h);

    // save copies the traversal position into a mutable cursor.
    // Advancing that cursor leaves the const view at the first bin.
    auto cursor = fixed.save;
    cursor.popFront();
    assert(cursor.front.index == 1);
    assert(fixed.front.index == 0);

    // The const view shares the count buffer; it does not freeze the data.
    // Adding 0.5 through h increments the first bin from 1 to 2.
    h.put(0.5);
    assert(fixed.front.count == 2);

    // Slicing creates another mutable cursor, here covering bins 1 and 2.
    // popBack removes bin 2 from this cursor's range, without changing
    // the histogram's bins or the range covered by fixed.
    auto subset = fixed[1 .. $];
    subset.popBack();
    assert(subset.length == 1);
    assert(fixed.length == 3);
}

// Range semantics and sharing for both supported owning storage forms.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    import std.range.primitives: isRandomAccessRange, hasLength, hasSlicing,
        hasAssignableElements, isInfinite;

    alias Axis = IntegralAxis!(double, AxisOptions(false, true, true));
    void check(Storage)(Storage counts, Storage replacement)
    {
        auto h = HistogramAccumulator!(Storage, Axis)(counts, Axis(3, 0.0));
        auto bins = h.bins;
        alias View = typeof(bins);
        static assert(isRandomAccessRange!View);
        static assert(hasLength!View && hasSlicing!View);
        static assert(!hasAssignableElements!View && !isInfinite!View);
        static assert(is(typeof(bins.front.count) == uint));
        bins[0].count = 10u; // Assigning to a returned temporary cannot update storage.
        assert(h.counts[0] == 0);

        h.put([-1.0, 0.5, 2.5, 4.0]);
        assert(h.underflow == 1 && h.overflow == 1);
        assert(bins.length == 3);
        assert(bins[0].count == 1 && bins[1].count == 0 && bins[2].count == 1);

        auto copy = bins.save;
        bins.popFront();
        bins.popBack();
        assert(bins.length == 1 && bins.front.index == 1);
        assert(bins.front == bins.back);
        auto sub = copy[1 .. 3][1 .. 2];
        assert(sub.front.index == 2);
        assert(copy[].length == 3);
        assert(copy[3 .. 3].empty);
        bins.popFront();
        assert(bins.empty && bins.length == 0);

        h.counts = replacement;
        h.axis[0] = Axis(3, 10.0);
        h.put(10.5);
        assert(copy.front.count == 1 && copy.front.bin.low == 0.0);
        assert(h.bins.front.count == 6 && h.bins.front.bin.low == 10.0);
    }
    check([0u, 0u, 0u, 0u, 0u], [0u, 5u, 0u, 0u, 0u]);
    check(rcslice!uint([0u, 0u, 0u, 0u, 0u]), rcslice!uint([0u, 5u, 0u, 0u, 0u]));
}

// All built-in axes preserve their existing bin descriptions.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        RegularAxis, TransformAxis, EnumAxis, CategoryAxis, VariableAxis;
    import mir.math.common: approxEqual;

    void checkNumeric(Axis)(Axis axis)
    {
        uint[] counts = [3u, 0u, 7u];
        auto h = HistogramAccumulator!(uint[], Axis)(counts, axis);
        const expectedAxis = axis;
        auto bins = h.bins;
        foreach (i; 0 .. bins.length)
        {
            auto expected = expectedAxis.bin(i);
            assert(bins[i].bin.low.approxEqual(expected.low));
            assert(bins[i].bin.high.approxEqual(expected.high));
            assert(bins[i].count == counts[i]);
        }
    }
    checkNumeric(IntegralAxis!(double, AxisOptions())(3, 0.0));
    checkNumeric(RegularAxis!(double, AxisOptions(true))(3, 0.0, 6.0));
    checkNumeric(RegularAxis!(double, AxisOptions(false, false, false, true))(3, 0.0, 6.0));
    checkNumeric(TransformAxis!(double, "a * 2", "a / 2", AxisOptions())(3, 0.0, 6.0));

    auto breaks = [0.0, 1.0, 3.0, 6.0].sliced;
    checkNumeric(VariableAxis!(double*, AxisOptions())(breaks));

    enum Label { first, second }
    alias Enum = EnumAxis!(Label);
    alias Category = CategoryAxis!(Label, AxisOptions());
    auto enums = HistogramAccumulator!(uint[], Enum)([2u, 4u], Enum()).bins;
    auto categories = HistogramAccumulator!(uint[], Category)([2u, 4u], Category()).bins;
    assert(enums[0].bin.slot == Label.first && enums[1].bin.slot == Label.second);
    assert(categories[0].bin.slot == Label.first && categories[1].count == 4);

    // The view retains break storage; returned numeric bins copy their bounds.
    auto makeView()
    {
        auto ownedBreaks = rcslice!double([0.0, 2.0, 5.0]);
        auto axis = VariableAxis!(typeof(ownedBreaks._iterator), AxisOptions())(ownedBreaks);
        auto counts = rcslice!uint([2u, 3u]);
        return HistogramAccumulator!(typeof(counts), typeof(axis))(counts, axis).bins;
    }
    auto owned = makeView();
    assert(owned[1].bin.low == 2.0 && owned[1].count == 3);
    auto description = owned[1].bin;
    description.low = 100.0;
    assert(owned[1].bin.low == 2.0);
    owned = typeof(owned).init;
    assert(description.low == 100.0 && description.high == 5.0);
}

// Invalid access and incompatible input are rejected.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(uint[], Axis);
    assertThrown!AssertError(H([0u], Axis(2, 0.0)).bins);
    auto bins = H([0u, 0u], Axis(2, 0.0)).bins;
    assertThrown!AssertError(bins[2]);
    assertThrown!AssertError(bins[size_t.max]);
    assertThrown!AssertError(bins[0 .. 3]);
    assertThrown!AssertError(bins[1 .. 0]);
    auto empty = bins[0 .. 0];
    assertThrown!AssertError(empty.front);
    assertThrown!AssertError(empty.back);
    assertThrown!AssertError(empty.popFront());
    assertThrown!AssertError(empty.popBack());

    alias Multi = HistogramAccumulator!(size_t[][], Axis, Axis);
    static assert(__traits(compiles, Multi.init.bins()));

    struct CountingAxis
    {
        alias CountType = uint;
        alias BinType = double;
        uint N_bin() const { return 2; }
        uint index(double x) const { return cast(uint) x; }
    }
    static assert(isAxis!CountingAxis);
    alias CountingOnly = HistogramAccumulator!(uint[], CountingAxis);
    static assert(!__traits(compiles, CountingOnly.init.bins()));
}

// Const sources produce mutable cursors over read-only storage handles.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    import std.algorithm: map, equal, find;
    import std.range: retro, take;
    import std.range.primitives: isRandomAccessRange;

    alias Axis = IntegralAxis!(double, AxisOptions());
    void check(Storage)(Storage counts)
    {
        alias H = HistogramAccumulator!(Storage, Axis);
        auto h = H(counts, Axis(3, 0.0));
        auto readBins(ref const H source) { return source.bins; }
        const fixed = readBins(h);
        auto cursor = fixed.save;
        static assert(is(typeof(cursor) == typeof(h.bins())));
        static assert(isRandomAccessRange!(typeof(cursor)));
        static assert(!__traits(compiles, fixed.popFront()));
        static assert(!__traits(compiles, fixed.popBack()));
        static assert(!__traits(compiles, {
            cursor._counts[0] = 10u;
        }));
        static assert(!__traits(compiles, {
            fixed._counts[0] = 10u;
        }));

        auto previous = fixed.front;
        h.put(0.5);
        assert(fixed.front.count == 2 && previous.count == 1);
        previous.count = 100;
        assert(h.counts[0] == 2);

        assert(cursor.map!(e => e.count).equal([2u, 2u, 3u]));
        assert(cursor.retro.map!(e => e.count).equal([3u, 2u, 2u]));
        assert(cursor.take(2).map!(e => e.count).equal([2u, 2u]));
        assert(cursor.find!(e => e.count == 3).front.index == 2);
        cursor.popFront();
        const advanced = cursor;
        auto saved = advanced.save;
        auto full = advanced[];
        auto tail = advanced[1 .. $];
        assert(saved.front.index == 1 && full.front.index == 1);
        assert(tail.front.index == 2);
        saved.popFront();
        full.popBack();
        assert(saved.front.index == 2 && full.back.index == 1);
        assert(advanced.length == 2 && fixed.length == 3);
        assert(advanced[0 .. 0].empty);
    }
    check([1u, 2u, 3u]);
    check(rcslice!uint([1u, 2u, 3u]));
}

// Both count and break ownership survive a const source and saved/sliced views.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.ndslice.slice: Slice;
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.axis: AxisOptions, VariableAxis, Bin;

    auto makeView()
    {
        auto breaks = rcslice!double([0.0, 1.0, 3.0, 6.0]);
        alias Axis = VariableAxis!(RCI!double, AxisOptions());
        auto counts = rcslice!uint([1u, 2u, 3u]);
        const h = HistogramAccumulator!(typeof(counts), Axis)(counts, Axis(breaks));
        const fixed = h.bins;
        return fixed.save[1 .. $];
    }
    auto view = makeView();
    auto saved = view.save;
    auto sliced = view[1 .. $];
    saved.popFront();
    assert(view.length == 2 && saved.length == 1 && sliced.length == 1);
    assert(view.front.count == 2 && view.back.count == 3);
    assert(view.front.bin.low == 1.0 && view.back.bin.high == 6.0);
    static assert(is(typeof(view._counts) == Slice!(RCI!(const uint))));
    static assert(is(typeof(view._axes[0]) ==
        VariableAxis!(RCI!(const double), AxisOptions())));
    auto bin = view.front.bin;
    static assert(is(typeof(bin) == Bin!double));
    view = typeof(view).init;
    // Each copy retains both buffers after the source view is released.
    assert(saved.front.count == 3 && sliced.front.count == 3);
    assert(saved.front.bin.low == 3.0 && sliced.front.bin.high == 6.0);
    saved = typeof(saved).init;
    sliced = typeof(sliced).init;
    assert(bin.low == 1.0 && bin.high == 3.0);
}

// Additional storage forms keep const data readable and traversal independent.
version(mir_stat_test)
pure nothrow
unittest
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    import std.algorithm: map, equal;

    alias Axis = IntegralAxis!(double, AxisOptions());
    uint[] backing = [1u, 99u, 2u, 99u, 3u, 99u];
    auto strided = Slice!(uint*, 1, SliceKind.universal)([3], [2], backing.ptr);
    const view = HistogramBinView!(typeof(strided), BinCoverage.ordinary, Axis)(strided, Axis(3, 0.0));
    assert(view.save.map!(e => e.count).equal([1u, 2u, 3u]));
    backing[2] = 4;
    assert(view[1].count == 4);

    const(uint)[] counts = [1u, 2u, 3u];
    const readOnly = HistogramBinView!(typeof(counts), BinCoverage.ordinary, Axis)(counts, Axis(3, 0.0));
    auto cursor = readOnly.save;
    assert(cursor.map!(e => e.count).equal([1u, 2u, 3u]));
    static assert(is(typeof(cursor.front.count) == uint));
}

// Custom axes must provide an ownership-preserving const conversion when needed.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: Bin;

    static struct ReferenceAxis
    {
        alias CountType = uint;
        alias BinType = double;
        double[] breaks;
        uint N_bin() const { return 1; }
        uint index(double x) const { return 0; }
        Bin!double bin(size_t i) const { return Bin!double(breaks[0], breaks[1]); }
    }
    static assert(!supportsBinView!(uint[], ReferenceAxis));

    static struct ReadOnlyAxis
    {
        alias CountType = uint;
        alias BinType = double;
        const(double)[] breaks;
        auto lightConst() const @property { return ReadOnlyAxis(breaks); }
        uint N_bin() const { return 1; }
        uint index(double x) const { return 0; }
        Bin!double bin(size_t i) const { return Bin!double(breaks[0], breaks[1]); }
    }
    static assert(supportsBinView!(uint[], ReadOnlyAxis));
    const view = HistogramBinView!(uint[], BinCoverage.ordinary, ReadOnlyAxis)(
        [2u], ReadOnlyAxis([0.0, 1.0]));
    assert(view.save.front.bin.high == 1.0 && view.front.count == 2);
}

// One axis accepts variadic batches; multiple axes require one coordinate each.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    auto h = HistogramAccumulator!(size_t[], A)([0UL, 0UL], A(2, 0.0));
    static assert(!__traits(compiles, h.put()));
    static assert(__traits(compiles, h.put(0.5, 1.5)));
    static assert(!__traits(compiles, h.put(0.5, "invalid")));
    h.put(0.5, 1.5);
    assert(h.counts == [1UL, 1UL]);
    h.put([0.5, 1.5]);
    assert(h.counts == [2UL, 2UL]);
    const double first = 0.5;
    immutable double second = 1.5;
    static assert(__traits(compiles, h.put(first, second)));
    h.put(first, second);
    const double[] readOnly = [0.5, 1.5];
    immutable double[] frozen = [0.5, 1.5];
    h.put(readOnly);
    h.put(frozen);
    assert(h.counts == [5UL, 5UL]);
    auto multi = HistogramAccumulator!(size_t[][], A, A)(
        [[0UL, 0UL], [0UL, 0UL]], A(2, 0.0), A(2, 0.0));
    static assert(__traits(compiles, multi.put(0.5, 1.5)));
    static assert(!__traits(compiles, multi.put(0.5, "invalid")));
    static assert(!__traits(compiles, multi.put(0.5)));
    static assert(!__traits(compiles, multi.put(0.5, 1.5, 0.5)));
    multi.put(0.5, 1.5);
    assert(multi.counts[0] == [0UL, 1UL]);
    assert(multi.counts[1] == [0UL, 0UL]);
}

// Validate storage shape before accepting an accumulator.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(size_t[], A);
    auto axis = A(2, 0.0);
    assertThrown!AssertError(H([0UL], axis));
    assertThrown!AssertError(H([0UL, 0UL, 0UL], axis));
    alias M = HistogramAccumulator!(size_t[][], A, A);
    assertThrown!AssertError(M([[0UL, 0UL]], axis, axis));
    assertThrown!AssertError(M([[0UL, 0UL], [0UL]], axis, axis));
}

// Construction, insertion, merging, and view traversal need no GC allocation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis, AxisOptions;
    static immutable uint[4] zero = [0, 0, 0, 0];
    static immutable double[4] samples = [-1.0, 0.5, 1.5, 3.0];
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    auto counts = rcslice!uint(zero[]);
    alias H = HistogramAccumulator!(typeof(counts), A);
    auto h = H(counts, A(2, 0.0));
    h.put(samples[]);
    auto other = H(rcslice!uint(zero[]), A(2, 0.0));
    other.put(0.5);
    h.put(other);
    assert(h.counts[1] == 2 && h.counts[2] == 1);
    assert(h.underflow == 1 && h.overflow == 1);
    auto view = h.bins();
    auto saved = view.save;
    auto tail = view[1 .. 2];
    view.popFront(); saved.popBack();
    assert(view.front.count == 1 && saved.back.count == 2);
    assert(tail.front.index == 1);
    tail.popFront();
    assert(tail.empty);

    // Enum and string category insertion are also usable without the GC.
    enum Label { first, second }
    alias C = CategoryAxis!(Label, AxisOptions());
    auto category = HistogramAccumulator!(typeof(counts), C)(rcslice!uint(zero[0 .. 2]), C());
    category.put(Label.second);
    assert(category.bins.back.count == 1);
    category.put("first");
    assert(category.bins.front.count == 1);
    alias FlowCategory = CategoryAxis!(Label, AxisOptions(false, true));
    auto withFlow = HistogramAccumulator!(typeof(counts), FlowCategory)(
        rcslice!uint(zero[0 .. 3]), FlowCategory());
    withFlow.put("unknown");
    assert(withFlow.overflow == 1);
}

// Joint counts preserve associations even when the marginals are identical.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    uint[3][2] zero;
    alias H = HistogramAccumulator!(typeof(zero), A, A);
    auto diagonal = H(zero, A(2, 0.0), A(3, 0.0));
    auto crossed = H(zero, A(2, 0.0), A(3, 0.0));
    diagonal.put(0.5, 0.5);
    diagonal.put(1.5, 1.5);
    crossed.put(0.5, 1.5);
    crossed.put(1.5, 0.5);
    assert(diagonal.counts[0] == [1u, 0u, 0u]);
    assert(diagonal.counts[1] == [0u, 1u, 0u]);
    assert(crossed.counts[0] == [0u, 1u, 0u]);
    assert(crossed.counts[1] == [1u, 0u, 0u]);
    assert(zero[0] == [0u, 0u, 0u] && zero[1] == [0u, 0u, 0u]);
    static assert(is(H.CountType == uint));
    static assert(__traits(compiles, diagonal.put(crossed)));
    const reader = diagonal;
    assert(reader.counts[0][0] == 1);
    static assert(!__traits(compiles, reader.put(0.5, 0.5)));
}

// Mixed static/dynamic arrays retain normal D storage ownership semantics.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    uint[3][2] buffer;
    buffer[1][2] = 4;
    auto fixedRows = buffer[];
    auto h = HistogramAccumulator!(typeof(fixedRows), A, A)(
        fixedRows, A(2, 0.0), A(3, 0.0));
    static assert(is(typeof(h).CountType == uint));
    assert(h.counts[1][2] == 4);
    h.put(1.5, 2.5);
    assert(buffer[1][2] == 5);

    uint[][2] dynamicRows = [buffer[0][], buffer[1][]];
    auto other = HistogramAccumulator!(typeof(dynamicRows), A, A)(
        dynamicRows, A(2, 0.0), A(3, 0.0));
    other.put(0.5, 2.5);
    assert(buffer[0][2] == 1 && buffer[1][2] == 5);
}

// Respect ndslice strides and allow different coordinate types on each axis.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis, AxisOptions;
    enum Label { first, second, third }
    alias A = IntegralAxis!(double, AxisOptions());
    alias C = CategoryAxis!(Label, AxisOptions());
    ulong[6] buffer;
    auto counts = buffer[].sliced(3, 2).transposed;
    auto h = HistogramAccumulator!(typeof(counts), A, C)(counts, A(2, 0.0), C());
    h.put(0.5, Label.third);
    h.put(1.5, "second");
    assert(h.counts[0, 2] == 1 && h.counts[1, 1] == 1);
    assert(buffer == [0UL, 0UL, 0UL, 1UL, 1UL, 0UL]);
    static assert(is(typeof(h).CountType == ulong));
    static assert(!__traits(compiles, h.put("invalid", Label.first)));
    static assert(!__traits(compiles, h.put(0.5)));
    static assert(!__traits(compiles, h.put(0.5, Label.first, 0.5)));
}

// Validate every dimension and leave all counts unchanged on invalid input.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(uint[][], A, A);
    auto x = A(2, 0.0);
    auto y = A(3, 0.0);
    assertThrown!AssertError(H([[0u, 0u, 0u]], x, y));
    assertThrown!AssertError(H([[0u, 0u, 0u], [0u, 0u]], x, y));
    assertThrown!AssertError(H([[0u, 0u, 0u], [0u, 0u, 0u, 0u]], x, y));
    uint[6] buffer;
    auto wrongShape = buffer[].sliced(3, 2);
    alias S = HistogramAccumulator!(typeof(wrongShape), A, A);
    assertThrown!AssertError(S(wrongShape, x, y));
    auto h = H([[0u, 0u, 0u], [0u, 0u, 0u]], x, y);
    assertThrown!AssertError(h.put(0.5, 3.0));
    assertThrown!AssertError(h.put(-1.0, 0.5));
    assert(h.counts[0] == [0u, 0u, 0u] && h.counts[1] == [0u, 0u, 0u]);
    auto s = S(buffer[].sliced(2, 3), x, y);
    assertThrown!AssertError(s.put(0.5, 3.0));
    assert(buffer == [0u, 0u, 0u, 0u, 0u, 0u]);

    // Custom axes are checked too, even if their index method omits bounds checks.
    struct UncheckedAxis
    {
        alias CountType = uint;
        alias BinType = int;
        enum N_bin = 3;
        int index(int value) const { return value; }
    }
    auto custom = HistogramAccumulator!(uint[][], A, UncheckedAxis)(
        h.counts, x, UncheckedAxis());
    assertThrown!AssertError(custom.put(0.5, -1));
    assertThrown!AssertError(custom.put(0.5, 3));
    assert(h.counts[0] == [0u, 0u, 0u] && h.counts[1] == [0u, 0u, 0u]);
}

// Reject unsupported storage ranks explicitly.
version(mir_stat_test)
unittest
{
    import mir.ndslice.slice: Slice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    static assert(!__traits(compiles, HistogramAccumulator!(uint[], A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][][], A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 1), A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 3), A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][], A, A, A).init));
}

// Every combination of independently enabled ends uses the same storage mapping.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    static foreach (flags; 0 .. 16)
    {{
        enum ux = (flags & 1) != 0;
        enum ox = (flags & 2) != 0;
        enum uy = (flags & 4) != 0;
        enum oy = (flags & 8) != 0;
        alias X = IntegralAxis!(double,
            AxisOptions(EnableUnderflow(ux), EnableOverflow(ox)));
        alias Y = IntegralAxis!(double,
            AxisOptions(EnableUnderflow(uy), EnableOverflow(oy)));
        enum rows = 2 + ux + ox;
        enum columns = 3 + uy + oy;
        uint[columns][rows] storage;
        auto arrayHist = HistogramAccumulator!(typeof(storage), X, Y)(
            storage, X(2, 0.0), Y(3, 0.0));
        uint[rows * columns] buffer;
        auto slice = buffer[].sliced(columns, rows).transposed;
        auto sliceHist = HistogramAccumulator!(typeof(slice), X, Y)(
            slice, X(2, 0.0), Y(3, 0.0));
        // Fill every ordinary, edge and corner cell once.
        foreach (r; 0 .. rows)
            foreach (c; 0 .. columns)
            {
                double x = cast(int) r - ux + 0.5;
                double y = cast(int) c - uy + 0.5;
                arrayHist.put(x, y);
                sliceHist.put(x, y);
            }
        foreach (r; 0 .. rows)
            foreach (c; 0 .. columns)
            {
                assert(arrayHist.counts[r][c] == 1);
                assert(sliceHist.counts[r, c] == 1);
            }
        static if (ux)
        {
            assert(arrayHist.underflow!0() == columns);
            assert(sliceHist.underflow!0() == columns);
        }
        else
            static assert(!__traits(compiles, arrayHist.underflow!0()));
        static if (ox)
        {
            assert(arrayHist.overflow!0() == columns);
            assert(sliceHist.overflow!0() == columns);
        }
        else
            static assert(!__traits(compiles, arrayHist.overflow!0()));
        static if (uy)
        {
            assert(arrayHist.underflow!1() == rows);
            assert(sliceHist.underflow!1() == rows);
        }
        else
            static assert(!__traits(compiles, arrayHist.underflow!1()));
        static if (oy)
        {
            assert(arrayHist.overflow!1() == rows);
            assert(sliceHist.overflow!1() == rows);
            // Flow totals reflect direct changes to shared caller storage.
            sliceHist.counts[0, columns - 1] += 2;
            const reader = sliceHist;
            assert(reader.overflow!1() == rows + 2);
        }
        else
            static assert(!__traits(compiles, arrayHist.overflow!1()));
        static assert(!__traits(compiles, arrayHist.overflow!2()));
        static assert(__traits(compiles, arrayHist.underflow()) == ux);
    }}
}

// Failed classification or malformed flow storage must not partially record a pair.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias F = IntegralAxis!(double,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));
    alias A = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(uint[][], F, A);
    auto x = F(2, 0.0);
    auto y = A(3, 0.0);
    assertThrown!AssertError(H([[0u, 0u, 0u], [0u, 0u, 0u]], x, y));
    assertThrown!AssertError(H([[0u, 0u, 0u], [0u, 0u, 0u],
        [0u, 0u, 0u], [0u, 0u]], x, y));
    uint[12] buffer;
    auto wrong = buffer[].sliced(3, 4);
    alias S = HistogramAccumulator!(typeof(wrong), F, A);
    assertThrown!AssertError(S(wrong, x, y));
    auto h = S(buffer[].sliced(4, 3), x, y);
    assertThrown!AssertError(h.put(-1.0, 3.0));
    assertThrown!AssertError(h.put(2.0, -1.0));
    assertThrown!AssertError(h.put(-1.0, double.nan));
    assertThrown!AssertError(h.put(double.nan, 0.5));
    foreach (count; buffer)
        assert(count == 0);
    assert(h.underflow!0() == 0 && h.overflow!0() == 0);

    // An enabled end on the second coordinate must not conceal an invalid first.
    alias ReversedStorage = uint[4][3];
    auto reversed = HistogramAccumulator!(ReversedStorage, A, F)(
        ReversedStorage.init, y, x);
    assertThrown!AssertError(reversed.put(-1.0, 2.0));
    assert(reversed.overflow!1() == 0);
}

// Preserve circular/right-closed endpoints, infinities, and unknown categories.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis,
        AxisOptions, EnableUnderflow, EnableOverflow, IsCircular, IsRightClosed;
    static foreach (rightClosed; [false, true])
    {{
        alias X = IntegralAxis!(double, AxisOptions(
            EnableUnderflow(true), EnableOverflow(true),
            IsCircular(true), IsRightClosed(rightClosed)));
        enum Label { first, second }
        alias Y = CategoryAxis!(Label, AxisOptions(EnableOverflow(true)));
        alias Storage = uint[3][4];
        auto h = HistogramAccumulator!(Storage, X, Y)(
            Storage.init, X(2, 0.0), Y());
        h.put(0.0, Label.first);
        h.put(2.0, Label.first);
        assert(h.counts[rightClosed ? 2 : 1][0] == 2);
        assert(h.underflow!0() == 0 && h.overflow!0() == 0);
        h.put(-double.infinity, "unknown");
        h.put(double.infinity, "second");
        assert(h.counts[0][2] == 1 && h.counts[3][1] == 1);
        assert(h.underflow!0() == 1 && h.overflow!0() == 1);
        assert(h.overflow!1() == 1);
    }}
}

// Dynamic storage shares preexisting counts; totals use the storage's counter type.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow, IsRightClosed;
    alias A = IntegralAxis!(double, AxisOptions(
        EnableUnderflow(true), EnableOverflow(true), IsRightClosed(true)));
    ulong[][] counts = [[0UL, 0UL, 0UL], [0UL, 0UL, 0UL], [0UL, 0UL, 0UL]];
    counts[0][0] = cast(ulong) uint.max + 1;
    auto h = HistogramAccumulator!(ulong[][], A, A)(counts, A(1, 0.0), A(1, 0.0));
    static assert(is(typeof(h.underflow!0()) == ulong));
    static assert(is(typeof(h.overflow!1()) == ulong));
    h.put(0.0, 0.0); // Right-closed lower endpoints underflow.
    h.put(1.0, 1.0); // Right-closed upper endpoints remain ordinary.
    assert(counts[0][0] == cast(ulong) uint.max + 2);
    assert(counts[1][1] == 1);
    assert(h.underflow!0() == cast(ulong) uint.max + 2);
    assert(h.overflow!1() == 0);
    counts[2][2] = 7;
    assert(h.overflow!0() == 7 && h.overflow!1() == 7);
}

// Adding enabled flow bins cannot wrap an axis extent before shape validation.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    struct HugeAxis
    {
        alias CountType = size_t;
        alias BinType = int;
        size_t N_bin = size_t.max;
        size_t index(int value) const { return 0; }
        bool isOverflow(int value) const { return value > 0; }
    }
    alias Storage = uint[1][1];
    alias H = HistogramAccumulator!(Storage, HugeAxis, HugeAxis);
    assertThrown!AssertError(H(Storage.init, HugeAxis(), HugeAxis()));
}

// Three-dimensional traversal preserves associations and strides at all boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias X = IntegralAxis!(double,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));
    alias Y = IntegralAxis!(double, AxisOptions(EnableUnderflow(true)));
    alias Z = IntegralAxis!(double, AxisOptions(EnableOverflow(true)));
    uint[5][4][4] storage;
    auto a = HistogramAccumulator!(typeof(storage), X, Y, Z)(
        storage, X(2, 0.0), Y(3, 0.0), Z(4, 0.0));
    uint[80] buffer;
    auto counts = buffer[].sliced(5, 4, 4).transposed!(2, 1, 0);
    auto b = HistogramAccumulator!(typeof(counts), X, Y, Z)(
        counts, X(2, 0.0), Y(3, 0.0), Z(4, 0.0));
    uint expectedUnderX, expectedOverX, expectedUnderY, expectedOverZ;
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4)
            foreach (k; 0 .. 5)
            {
                // Distinct counts make swapped dimensions visible in totals.
                uint repeats = 1 + i + 2 * j + 3 * k;
                foreach (_; 0 .. repeats)
                {
                    a.put(i - 0.5, j - 0.5, k + 0.5);
                    b.put(i - 0.5, j - 0.5, k + 0.5);
                }
                assert(a.counts[i][j][k] == repeats);
                assert(b.counts[i, j, k] == repeats);
                assert(buffer[(k * 4 + j) * 4 + i] == repeats);
                if (i == 0) expectedUnderX += repeats;
                if (i == 3) expectedOverX += repeats;
                if (j == 0) expectedUnderY += repeats;
                if (k == 4) expectedOverZ += repeats;
            }
    assert(a.underflow!0() == expectedUnderX && b.underflow!0() == expectedUnderX);
    assert(a.overflow!0() == expectedOverX && b.overflow!0() == expectedOverX);
    assert(a.underflow!1() == expectedUnderY && b.underflow!1() == expectedUnderY);
    assert(a.overflow!2() == expectedOverZ && b.overflow!2() == expectedOverZ);
    const reader = b;
    buffer[0] += 7;
    assert(reader.underflow!0() == expectedUnderX + 7);
    static assert(!__traits(compiles, b.overflow!1()));
    static assert(!__traits(compiles, b.underflow!2()));
    static assert(!__traits(compiles, b.overflow!3()));
}

// Higher ranks use the same insertion and total paths, with no special 3-D case.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableOverflow;
    alias A = IntegralAxis!(int, AxisOptions(EnableOverflow(true)));
    ulong[2][2][2][2] storage;
    auto a = HistogramAccumulator!(typeof(storage), A, A, A, A)(
        storage, A(1, 0), A(1, 0), A(1, 0), A(1, 0));
    ulong[16] buffer;
    auto counts = buffer[].sliced(2, 2, 2, 2);
    auto b = HistogramAccumulator!(typeof(counts), A, A, A, A)(
        counts, A(1, 0), A(1, 0), A(1, 0), A(1, 0));
    a.put(1, 0, 1, 1);
    b.put(1, 0, 1, 1);
    assert(a.counts[1][0][1][1] == 1 && b.counts[1, 0, 1, 1] == 1);
    static foreach (dimension; 0 .. 4)
    {
        assert(a.overflow!dimension() == (dimension == 1 ? 0 : 1));
        assert(b.overflow!dimension() == (dimension == 1 ? 0 : 1));
    }
    static assert(is(typeof(a).CountType == ulong));
    static assert(!__traits(compiles, a.put(0, 0, 0)));
    static assert(!__traits(compiles, a.put(0, 0, 0, 0, 0)));
}

// Validate every nested branch and every ndslice dimension before accepting storage.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced, Slice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    alias H = HistogramAccumulator!(uint[][][], A, A, A);
    auto axis = A(2, 0.0);
    uint[][][] counts = [[[0u, 0u], [0u, 0u]], [[0u, 0u], [0u]]];
    assertThrown!AssertError(H(counts, axis, axis, axis));
    counts[1][1] = [0u, 0u, 0u];
    assertThrown!AssertError(H(counts, axis, axis, axis));
    counts[1] = [[0u, 0u]];
    assertThrown!AssertError(H(counts, axis, axis, axis));
    counts = [[[0u, 0u], [0u, 0u]]];
    assertThrown!AssertError(H(counts, axis, axis, axis));
    uint[8] buffer;
    alias S = HistogramAccumulator!(Slice!(uint*, 3), A, A, A);
    assertThrown!AssertError(S(buffer[].sliced(1, 2, 4), axis, axis, axis));
    assertThrown!AssertError(S(buffer[].sliced(2, 1, 4), axis, axis, axis));
    assertThrown!AssertError(S(buffer[].sliced(2, 4, 1), axis, axis, axis));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 2), A, A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 4), A, A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][][][], A, A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 2)[], A, A, A).init));
}

// Resolve mixed coordinate types completely before recording a joint observation.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, CategoryAxis,
        AxisOptions, EnableUnderflow, EnableOverflow;
    alias X = IntegralAxis!(int, AxisOptions(EnableUnderflow(true)));
    enum Label { first, second }
    alias Y = CategoryAxis!(Label, AxisOptions(EnableOverflow(true)));
    alias Z = IntegralAxis!(double, AxisOptions());
    uint[2][3][3] counts;
    auto h = HistogramAccumulator!(typeof(counts), X, Y, Z)(
        counts, X(2, 0), Y(), Z(2, 0.0));
    assertThrown!AssertError(h.put(-1, "unknown", 2.0));
    assertThrown!AssertError(h.put(-1, "unknown", double.nan));
    assertThrown!AssertError(h.put(2, Label.first, 0.5));
    assert(h.underflow!0() == 0 && h.overflow!1() == 0);
    h.put(-1, "unknown", 1.5);
    h.put(1, Label.second, 0.5);
    assert(h.counts[0][2][1] == 1 && h.counts[2][1][0] == 1);
    assert(h.underflow!0() == 1 && h.overflow!1() == 1);
    static assert(!__traits(compiles, h.put("invalid", Label.first, 0.5)));
    static assert(!__traits(compiles, h.put(0, Label.first, "invalid")));
}

// Mixed static/dynamic nesting preserves caller ownership at three dimensions.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    uint[2][2][2] buffer;
    auto planes = buffer[];
    auto h = HistogramAccumulator!(typeof(planes), A, A, A)(
        planes, A(2, 0), A(2, 0), A(2, 0));
    h.put(1, 0, 1);
    assert(buffer[1][0][1] == 1);
    uint[][2][2] rows;
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 2)
            rows[i][j] = buffer[i][j][];
    auto other = HistogramAccumulator!(typeof(rows), A, A, A)(
        rows, A(2, 0), A(2, 0), A(2, 0));
    other.put(0, 1, 0);
    assert(buffer[0][1][0] == 1 && buffer[1][0][1] == 1);
    static assert(is(typeof(other).CountType == uint));
}

// Merge differing storage layouts by joint coordinate, including all end bins.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias X = IntegralAxis!(int,
        AxisOptions(EnableUnderflow(true), EnableOverflow(true)));
    alias Y = IntegralAxis!(int, AxisOptions());
    alias Z = IntegralAxis!(int, AxisOptions(EnableOverflow(true)));
    uint[3][2][4] storage;
    alias H = HistogramAccumulator!(typeof(storage), X, Y, Z);
    auto arrayHist = H(storage, X(2, 0), Y(2, 0), Z(2, 0));
    uint[24] buffer;
    auto counts = buffer[].sliced(3, 2, 4).transposed!(2, 1, 0);
    auto sliceHist = HistogramAccumulator!(typeof(counts), X, Y, Z)(
        counts, X(2, 0), Y(2, 0), Z(2, 0));
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 2)
            foreach (k; 0 .. 3)
            {
                arrayHist.counts[i][j][k] = 1 + i + 2 * j + 3 * k;
                counts[i, j, k] = 2;
            }
    const source = arrayHist;
    sliceHist.put(source);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 2)
            foreach (k; 0 .. 3)
            {
                assert(counts[i, j, k] == 3 + i + 2 * j + 3 * k);
                assert(source.counts[i][j][k] == 1 + i + 2 * j + 3 * k);
            }
    // Reverse the storage combination, using a const ndslice source.
    const sliceSource = sliceHist;
    arrayHist.put(sliceSource);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 2)
            foreach (k; 0 .. 3)
                assert(arrayHist.counts[i][j][k] == 4 + 2 * i + 4 * j + 6 * k);
    assert(arrayHist.underflow!0() == source.underflow!0() + sliceSource.underflow!0());
    assert(arrayHist.overflow!2() == source.overflow!2() + sliceSource.overflow!2());
    static assert(!__traits(compiles, source.put(arrayHist)));
}

// Axes and both complete shapes must be validated before any destination writes.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    alias H = HistogramAccumulator!(uint[][], A, A);
    auto make() { return H([[1u, 2u], [3u, 4u]], A(2, 0), A(2, 0)); }
    auto destination = make();
    auto source = make();
    source.axis[1] = A(2, 1);
    assertThrown!AssertError(destination.put(source));
    assert(destination.counts == [[1u, 2u], [3u, 4u]]);
    source.axis[1] = A(2, 0);
    source.counts[1] = [3u];
    assertThrown!AssertError(destination.put(source));
    assert(destination.counts == [[1u, 2u], [3u, 4u]]);
    source = make();
    destination.counts[1] = [3u];
    assertThrown!AssertError(destination.put(source));
    assert(destination.counts[0] == [1u, 2u]);
    uint[4] buffer = [1u, 2u, 3u, 4u];
    auto counts = buffer[].sliced(2, 2);
    auto s = HistogramAccumulator!(typeof(counts), A, A)(counts, A(2, 0), A(2, 0));
    s.counts = buffer[].sliced(1, 4);
    assertThrown!AssertError(s.put(source));
    assert(buffer == [1u, 2u, 3u, 4u]);
    destination = make();
    assertThrown!AssertError(destination.put(s));
    assert(destination.counts == [[1u, 2u], [3u, 4u]]);

    alias Wide = HistogramAccumulator!(ulong[][], A, A);
    static assert(!__traits(compiles, destination.put(Wide.init)));
    static assert(!__traits(compiles, destination.put(HistogramAccumulator!(uint[], A).init)));
}

// Self-merging and corresponding shared storage double each logical count once.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    uint[2][2] storage = [[1u, 2u], [3u, 4u]];
    auto a = HistogramAccumulator!(typeof(storage), A, A)(storage, A(2, 0), A(2, 0));
    a.put(a);
    uint[2][2] doubled = [[2u, 4u], [6u, 8u]];
    assert(a.counts == doubled);
    auto sharedRows = storage[];
    auto b = HistogramAccumulator!(typeof(sharedRows), A, A)(sharedRows, A(2, 0), A(2, 0));
    auto other = b;
    b.put(other);
    assert(storage == doubled);
    uint[4] buffer = [1u, 2u, 3u, 4u];
    auto counts = buffer[].sliced(2, 2);
    auto c = HistogramAccumulator!(typeof(counts), A, A)(counts, A(2, 0), A(2, 0));
    c.put(c);
    assert(buffer == [2u, 4u, 6u, 8u]);
}

// Accept temporary and read-only-storage sources without weakening destination constness.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    alias H = HistogramAccumulator!(uint[2][2], A, A);
    uint[2][2] initial = [[1u, 2u], [3u, 4u]];
    auto destination = H(initial, A(2, 0), A(2, 0));
    destination.put(H(initial, A(2, 0), A(2, 0)));
    assert(destination.counts[0] == [2u, 4u]);
    immutable uint[4] buffer = [1u, 2u, 3u, 4u];
    auto counts = buffer[].sliced(2, 2);
    auto source = HistogramAccumulator!(typeof(counts), A, A)(counts, A(2, 0), A(2, 0));
    destination.put(source);
    assert(destination.counts[0] == [3u, 6u]);
    assert(destination.counts[1] == [9u, 12u]);
    static assert(!__traits(compiles, source.put(destination)));
}

// Generalized view traversal respects strided layouts, coordinates, and cursor state.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    import std.range.primitives: isRandomAccessRange;
    alias X = IntegralAxis!(int, AxisOptions(EnableUnderflow(true)));
    alias Y = IntegralAxis!(int, AxisOptions());
    alias Z = IntegralAxis!(int, AxisOptions(EnableOverflow(true)));
    uint[36] buffer;
    auto counts = buffer[].sliced(4, 3, 3).transposed!(2, 1, 0);
    auto h = HistogramAccumulator!(typeof(counts), X, Y, Z)(
        counts, X(2, 0), Y(3, 0), Z(3, 0));
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 4)
                counts[i, j, k] = 100 * i + 10 * j + k;
    const view = h.bins;
    static assert(isRandomAccessRange!(typeof(view.save)));
    assert(view.length == 18);
    foreach (flat; 0 .. view.length)
    {
        auto entry = view[flat];
        assert(entry.index!0 == flat / 9);
        assert(entry.index!1 == flat / 3 % 3);
        assert(entry.index!2 == flat % 3);
        assert(entry.count == 100 * (flat / 9 + 1) + 10 * (flat / 3 % 3) + flat % 3);
        assert(entry.bin!2.low == flat % 3);
    }
    auto cursor = view.save;
    cursor.popFront();
    cursor.popBack();
    const saved = cursor.save;
    auto tail = saved[8 .. $];
    assert(tail.front.index!0 == 1 && tail.front.index!1 == 0 && tail.front.index!2 == 0);
    assert(view.length == 18 && saved.length == 16);
    assert(saved.front.index!2 == 1 && saved.back.index!2 == 1);
    auto previous = view[9];
    h.put(1, 0, 0);
    assert(view[9].count == previous.count + 1);
    static assert(!__traits(compiles, view.front.index!3));
    static assert(!__traits(compiles, view.front.bin!3));
    static assert(!__traits(compiles, { cursor._counts[0, 0, 0] = 0; }));
}

// Nested arrays and read-only count storage retain the same flat traversal order.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    uint[][][] counts = [[[1u, 2u], [3u, 4u]], [[5u, 6u], [7u, 8u]]];
    auto h = HistogramAccumulator!(typeof(counts), A, A, A)(counts, A(2, 0), A(2, 0), A(2, 0));
    auto view = h.bins;
    foreach (i; 0 .. 8)
        assert(view[i].count == i + 1);
    counts[1][0][1] = 12;
    assert(view[5].count == 12);
    const entry = view[5];
    assert(entry.index!0 == 1 && entry.bin!1.low == 0);
    static assert(!__traits(compiles, { entry.count = 0; }));
}

// Reference-counted count and boundary handles survive a local joint accumulator.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: VariableAxis, IntegralAxis, AxisOptions;
    auto makeView()
    {
        auto breaks = rcslice!double([0.0, 1.0, 3.0]);
        alias X = VariableAxis!(typeof(breaks._iterator), AxisOptions());
        alias Y = IntegralAxis!(int, AxisOptions());
        auto counts = rcslice!uint(2, 2);
        auto h = HistogramAccumulator!(typeof(counts), X, Y)(counts, X(breaks), Y(2, 0));
        h.put(2.0, 1);
        return h.bins.save[2 .. $];
    }
    auto view = makeView();
    assert(view.back.index!0 == 1 && view.back.index!1 == 1 && view.back.count == 1);
    auto description = view.back.bin!0;
    view = typeof(view).init;
    assert(description.low == 1.0 && description.high == 3.0);
}

// Static-array views borrow live counts instead of silently copying a snapshot.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.stat.internal.borrow: hasBorrowEscapeChecking;
    alias A = IntegralAxis!(int, AxisOptions());
    alias Storage = uint[2][2];
    alias H = HistogramAccumulator!(Storage, A, A);
    void check() @nogc
    {
        auto h = H(Storage.init, A(2, 0), A(2, 0));
        auto view = h.bins;
        h.put(1, 1);
        assert(view[3].count == 1);
        const reader = view;
        assert(reader.back.bin!1.low == 1);
        alias OneStorage = uint[2];
        auto one = HistogramAccumulator!(OneStorage, A)(OneStorage.init, A(2, 0));
        auto oneView = one.bins;
        one.put(1);
        assert(oneView.back.index == 1 && oneView.back.count == 1);
    }
    check();
    enum safeBorrow = __traits(compiles, () @safe {
        auto h = H(Storage.init, A(2, 0), A(2, 0));
        auto view = h.bins;
        h.put(1, 1);
        return view[3].count;
    });
    static assert(safeBorrow == hasBorrowEscapeChecking);
    static assert(!__traits(compiles, () @safe {
        auto h = H(Storage.init, A(2, 0), A(2, 0));
        return h.bins;
    }));
    static assert(!__traits(compiles, () @safe {
        auto h = H(Storage.init, A(2, 0), A(2, 0));
        return h.bins.save[1 .. $];
    }));
}

// Reject malformed joint view shapes and overflowing ordinary-bin products.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions, Bin;
    alias A = IntegralAxis!(int, AxisOptions());
    alias V = HistogramBinView!(uint[][], BinCoverage.ordinary, A, A);
    assertThrown!AssertError(V([[0u, 0u], [0u]], A(2, 0), A(2, 0)));
    auto view = V([[0u, 0u], [0u, 0u]], A(2, 0), A(2, 0));
    assertThrown!AssertError(view[4]);
    assertThrown!AssertError(view[0 .. 5]);
    assertThrown!AssertError(view[0 .. 0].front);
    static struct HugeAxis
    {
        alias CountType = size_t;
        alias BinType = int;
        size_t N_bin;
        size_t index(int x) const { return 0; }
        Bin!int bin(size_t i) const { return Bin!int(0, 1); }
    }
    alias S = Slice!(uint*, 2, SliceKind.universal);
    // Product validation must fail before inspecting any count buffer.
    assertThrown!AssertError(HistogramBinView!(S, BinCoverage.ordinary, HugeAxis, HugeAxis)(
        S.init, HugeAxis(size_t.max), HugeAxis(2)));
}


// Every 1D option combination uses the same storage layout and merge machinery.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.stat.descriptive.histogram.traits: storageExtent;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;

    static foreach (u; [false, true])
    static foreach (o; [false, true])
    {{
        alias A = IntegralAxis!(double, AxisOptions(false, o, u));
        enum n = 2 + u + o;
        auto axis = A(2, 0.0);
        uint[n] initial;
        initial[u] = 3;
        initial[u + 1] = 5;
        static if (u) initial[0] = 7;
        static if (o) initial[$ - 1] = 11;
        auto h = HistogramAccumulator!(typeof(initial), A)(initial, axis);
        assert(storageExtent(axis) == n && h.storageExtent!0() == n);
        static if (u) assert(h.underflow == 7);
        static if (o) assert(h.overflow == 11);
        h.put(0.5, 1.5);
        static if (u) h.put(-1.0);
        static if (o) h.put(2.0);
        assert(h.counts[u] == 4 && h.counts[u + 1] == 6);
        static if (u) assert(h.underflow == 8);
        static if (o) assert(h.overflow == 12);

        // Merge static arrays into strided storage, including both end counts.
        uint[2 * n] buffer;
        auto slice = buffer[].sliced.stride(2);
        auto destination = HistogramAccumulator!(typeof(slice), A)(slice, axis);
        const source = h;
        destination.put(source);
        foreach (i; 0 .. n)
        {
            assert(buffer[2 * i] == h.counts[i]);
            assert(buffer[2 * i + 1] == 0);
        }
        const bins = destination.bins();
        assert(bins.length == 2 && bins.front.count == 4 && bins.back.count == 6);
        destination.put(destination);
        assert(bins.front.count == 8 && bins.back.count == 12);
        static if (u) assert(destination.underflow == 16);
        static if (o) assert(destination.overflow == 24);
        assert(initial[u] == 3); // Static-array construction made an independent copy.
    }}
}

// Expanded storage is mandatory, and extent arithmetic cannot wrap.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.stat.descriptive.histogram.traits: storageExtent;
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    alias H = HistogramAccumulator!(uint[], A);
    assertThrown!AssertError(H([0u, 0u], A(2, 0.0)));
    assertThrown!AssertError(H([0u, 0u, 0u, 0u, 0u], A(2, 0.0)));
    static struct HugeAxis
    {
        alias CountType = size_t;
        alias BinType = int;
        size_t N_bin = size_t.max;
        size_t index(int) const { return 0; }
        bool isOverflow(int) const { return false; }
    }
    assertThrown!AssertError(storageExtent(HugeAxis()));
}


/// Include underflow and overflow while keeping ordinary indices unchanged.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    auto h = HistogramAccumulator!(uint[], A)([2u, 3u, 4u, 1u], A(2, 0.0));

    // Default traversal still returns only the two ordinary bins.
    assert(h.bins.length == 2);
    auto all = h.bins!(BinCoverage.all)();
    assert(all.length == 4);
    assert(all.front.isUnderflow && all.front.count == 2);
    assert(all.back.isOverflow && all.back.count == 1);

    // End bins have counts but no ordinary index or interval.
    // Check classification before requesting ordinary-bin metadata.
    auto firstOrdinary = all[1];
    assert(firstOrdinary.isOrdinary);
    assert(firstOrdinary.index == 0 && firstOrdinary.bin.low == 0.0);

    // Saved and sliced cursors preserve classification and share live counts.
    auto tail = all[1 .. $];
    h.put(2.5);
    assert(tail.back.isOverflow && tail.back.count == 2);
    assert(all.front.isUnderflow);
}

// Every enabled joint coordinate is visited once, independent of strides.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import std.range.primitives: isRandomAccessRange, hasSlicing;

    static foreach (u; [false, true])
    static foreach (o; [false, true])
    {{
        alias X = IntegralAxis!(int, AxisOptions(false, o, u));
        alias Y = IntegralAxis!(double, AxisOptions(false, true, true));
        enum rows = 2 + u + o;
        enum columns = 5;
        uint[rows * columns] buffer;
        auto storage = buffer[].sliced(columns, rows).transposed;
        foreach (i; 0 .. rows)
            foreach (j; 0 .. columns)
                storage[i, j] = cast(uint)(1 + i * columns + j);
        auto h = HistogramAccumulator!(typeof(storage), X, Y)(
            storage, X(2, 0), Y(3, 0.0));
        const all = h.bins!(BinCoverage.all)();
        static assert(isRandomAccessRange!(typeof(all.save)) && hasSlicing!(typeof(all.save)));
        assert(all.length == rows * columns);
        uint sum;
        foreach (i; 0 .. all.length)
        {
            const entry = all[i];
            const row = i / columns;
            const column = i % columns;
            assert(entry.count == i + 1);
            sum += entry.count;
            assert(entry.isUnderflow == (u && row == 0));
            assert(entry.isOverflow == (o && row == rows - 1));
            assert(entry.isOrdinary == !(entry.isUnderflow || entry.isOverflow));
            assert(entry.isUnderflow!1 == (column == 0));
            assert(entry.isOverflow!1 == (column == columns - 1));
            assert(entry.isOrdinary!1 == (column > 0 && column < columns - 1));
            if (entry.isOrdinary) assert(entry.index == row - u);
            if (entry.isOrdinary!1)
            {
                assert(entry.index!1 == column - 1);
                assert(entry.bin!1.low == column - 1);
            }
        }
        assert(sum == all.length * (all.length + 1) / 2);
        auto saved = all.save;
        saved.popFront();
        saved.popBack();
        assert(saved.length == all.length - 2);
        assert(all[1 .. $][1 .. $].front.count == 3);
        assert(all[$ .. $].empty);
        assert(h.bins.length == 6);
        foreach (entry; h.bins)
            assert(entry.isOrdinary && entry.isOrdinary!1);
        static assert(!__traits(compiles, all.front.isOverflow!2));
        static assert(!__traits(compiles, h.bins!(cast(BinCoverage) 99)()));
    }}
}

// Categorical and variable end bins cannot expose invalid ordinary descriptions.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: CategoryAxis, VariableAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    enum Label { first, second }
    alias X = CategoryAxis!(Label, AxisOptions(false, true));
    alias Y = VariableAxis!(double*, AxisOptions(false, true, true));
    auto y = Y([0.0, 1.0, 3.0].sliced);
    auto h = HistogramAccumulator!(uint[][], X, Y)(
        [[0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u], [0u, 0u, 0u, 0u]], X(), y);
    h.put("unknown", -1.0);
    const all = h.bins!(BinCoverage.all)();
    const corner = all[8];
    assert(corner.isOverflow && corner.isUnderflow!1 && corner.count == 1);
    assertThrown!AssertError(corner.index);
    assertThrown!AssertError(corner.bin);
    assertThrown!AssertError(corner.index!1);
    assertThrown!AssertError(corner.bin!1);
    auto mixed = all[9];
    assert(mixed.isOverflow && mixed.isOrdinary!1);
    assert(mixed.bin!1.low == 0 && mixed.bin!1.high == 1);
    assertThrown!AssertError(mixed.bin);
    assert(all[1].bin.slot == Label.first);
    assert(all[1].bin!1.high == 1);
}

// Owning all-bin views outlive an accumulator; static-array views remain borrowed.
version(mir_stat_test_lifetime)
@safe
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions(false, true, true));
    auto owning()
    {
        auto counts = rcslice!uint([2u, 3u, 4u, 1u]);
        auto h = HistogramAccumulator!(typeof(counts), A)(counts, A(2, 0));
        return h.bins!(BinCoverage.all)();
    }
    auto view = owning();
    assert(view.front.isUnderflow && view.front.count == 2);
    assert(view.back.isOverflow && view.back.count == 1);

    uint[4] buffer;
    auto local = HistogramAccumulator!(typeof(buffer), A)(buffer, A(2, 0));
    auto borrowed = local.bins!(BinCoverage.all)();
    local.put(-1);
    assert(borrowed.front.count == 1);
    static assert(!__traits(compiles, () @safe {
        uint[4] storage;
        auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(2, 0));
        return h.bins!(BinCoverage.all)();
    }));
    static assert(!__traits(compiles, () @safe {
        uint[4] storage;
        auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(2, 0));
        return h.bins!(BinCoverage.all)().save;
    }));
}


// In one dimension, disabled end bins are never synthesized.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    static foreach (u; [false, true])
    static foreach (o; [false, true])
    {{
        alias A = IntegralAxis!(int, AxisOptions(false, o, u));
        uint[2 + u + o] buffer;
        auto storage = buffer[].sliced;
        auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(2, 0));
        auto all = h.bins!(BinCoverage.all)();
        assert(all.length == buffer.length);
        assert(all.front.isUnderflow == u && all.back.isOverflow == o);
        assert(all[u].isOrdinary && all[u].index == 0);
        assert(all[u + 1].index == 1);
        h.put(0, 1);
        assert(all[u].count == 1 && all[u + 1].count == 1);
        static if (!u && !o)
            assert(all.front == h.bins.front && all.back == h.bins.back);
    }}
}

// Axis order, rectangular built-in arrays, and strided ndslices agree.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    alias A = IntegralAxis!(int, AxisOptions());
    uint[4][3][2] data;
    uint[24] backing;
    // A transposed [4,3,2] slice has logical shape [2,3,4].
    auto strided = backing[].sliced(4, 3, 2).transposed!(2, 1, 0);
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 4)
                data[i][j][k] = strided[i,j,k] = cast(uint)(1 + i*12 + j*4 + k);
    const arrays = HistogramAccumulator!(typeof(data), A, A, A)(
        data, A(2, 0), A(3, 10), A(4, 20));
    const slices = HistogramAccumulator!(typeof(strided), A, A, A)(
        strided, A(2, 0), A(3, 10), A(4, 20));
    auto a = arrays.rcMarginal!(2, 0)();
    auto b = slices.rcMarginal!(2, 0)();
    assert(a.counts.shape == [4, 2]);
    assert(a.axis[0].bin(0).low == 20 && a.axis[1].bin(0).low == 0);
    uint sum;
    foreach (k; 0 .. 4)
        foreach (i; 0 .. 2)
        {
            uint expected;
            foreach (j; 0 .. 3) expected += data[i][j][k];
            assert(a.counts[k,i] == expected && b.counts[k,i] == expected);
            sum += expected;
        }
    assert(sum == 300);
    static assert(!__traits(compiles, arrays.rcMarginal!()()));
    static assert(!__traits(compiles, arrays.rcMarginal!(0, 0)()));
    static assert(!__traits(compiles, arrays.rcMarginal!(0, 1, 2)()));
    static assert(!__traits(compiles, arrays.rcMarginal!3()));
    static assert(!__traits(compiles, arrays.rcMarginal!(-1)()));
    static assert(!__traits(compiles, arrays.rcMarginal!double()));
    static assert(!__traits(compiles, arrays.rcMarginal!(0.5)()));
    static assert(!__traits(compiles, a.rcMarginal!(0, 1)()));
    static assert(!__traits(compiles, a.rcMarginal!0().rcMarginal!0()));
}


// Copied RC boundaries survive their original owner; marginal counts are writable.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.axis: VariableAxis, IntegralAxis, AxisOptions;
    alias X = VariableAxis!(RCI!double, AxisOptions());
    alias Y = IntegralAxis!(int, AxisOptions());
    auto makeMarginal()
    {
        double[3] values = [0.0, 1.0, 4.0];
        auto boundaries = rcslice!double(values[]);
        uint[2][2] data = [[1u, 2u], [3u, 4u]];
        const h = HistogramAccumulator!(typeof(data), X, Y)(data, X(boundaries), Y(2, 0));
        return h.rcMarginal!0();
    }
    auto result = makeMarginal();
    assert(result.counts == [3u, 7u]);
    assert(result.axis[0].bin(1).high == 4);
    result.put(2.0);
    assert(result.counts == [3u, 8u]);
}

// Borrowed variable-axis boundaries remain borrowed, with read-only access.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis, IntegralAxis, AxisOptions;
    double[] boundaries = [0, 1, 4];
    alias X = VariableAxis!(double*, AxisOptions());
    alias Y = IntegralAxis!(int, AxisOptions());
    uint[2][2] data = [[1u, 2u], [3u, 4u]];
    const h = HistogramAccumulator!(typeof(data), X, Y)(
        data, X(boundaries[].sliced), Y(2, 0));
    auto result = h.rcMarginal!0();
    static assert(is(typeof(result.axis[0]) == VariableAxis!(const(double)*, AxisOptions())));
    assert(result.axis[0].bin(1).low == 1 && result.axis[0].bin(1).high == 4);
    result.put(2.0);
    assert(result.counts == [3u, 8u]);
}

// Owning counts can escape stack-backed sources; borrowed axis data cannot escape.
version(mir_stat_test_lifetime)
@safe @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis, IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    auto makeMarginal()
    {
        uint[2][2] data = [[1u, 2u], [3u, 4u]];
        auto h = HistogramAccumulator!(typeof(data), A, A)(data, A(2, 0), A(2, 0));
        return h.rcMarginal!0();
    }
    assert(makeMarginal().counts == [3u, 7u]);
    // A borrowed count slice is consumed during the call, not retained.
    auto fromSlice()
    {
        uint[4] data = [1u, 2u, 3u, 4u];
        auto storage = data[].sliced(2, 2);
        auto h = HistogramAccumulator!(typeof(storage), A, A)(storage, A(2, 0), A(2, 0));
        return h.rcMarginal!1();
    }
    assert(fromSlice().counts == [4u, 6u]);

    static assert(!__traits(compiles, () @safe {
        double[3] boundaries = [0, 1, 4];
        alias X = VariableAxis!(double*, AxisOptions());
        uint[2][2] data;
        auto h = HistogramAccumulator!(typeof(data), X, A)(
            data, X(boundaries[].sliced), A(2, 0));
        return h.rcMarginal!0();
    }));
}

// Validate the full source shape again before projecting externally shared arrays.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    auto data = [[1u, 2u], [3u, 4u]];
    auto h = HistogramAccumulator!(typeof(data), A, A)(data, A(2, 0), A(2, 0));
    data[1] = [3u];
    assertThrown!AssertError(h.rcMarginal!0());
}


// Marginalization depends on stored counts and axes, not bin-description support.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: CategoryAxis, AxisOptions;
    static struct CountOnlyAxis
    {
        alias CountType = uint;
        alias BinType = int;
        uint N_bin = 2;
        uint index(int value) const { return cast(uint) value; }
    }
    enum Label { first, second }
    alias C = CategoryAxis!(Label, AxisOptions(false, true));
    uint[3][2] data = [[1u, 2u, 3u], [4u, 5u, 6u]];
    auto h = HistogramAccumulator!(typeof(data), CountOnlyAxis, C)(data, CountOnlyAxis(), C());
    auto numeric = h.rcMarginal!0();
    assert(numeric.counts == [6u, 15u]);
    auto category = h.rcMarginal!1();
    assert(category.counts == [5u, 7u, 9u]);
    assert(category.bins.front.bin.slot == Label.first);
    assert(category.overflow == 9);
}


// Read-only counter storage still supports sums and remains non-writable.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    alias A = IntegralAxis!(int, AxisOptions(false, true, true));
    void check(S)(S storage)
    {
        auto h = HistogramAccumulator!(S, A, A)(storage, A(1, 0), A(1, 0));
        static assert(is(typeof(h).CountType == uint));
        assert(h.underflow == 6 && h.overflow == 24);
        assert(h.underflow!1 == 12 && h.overflow!1 == 18);
        const frozen = h;
        assert(frozen.underflow == 6 && frozen.overflow!1 == 18);
        auto marginal = h.rcMarginal!0();
        assert(marginal.counts == [6u, 15u, 24u]);
        marginal.put(0);
        assert(marginal.counts[1] == 16);
        static assert(!__traits(compiles, h.put(0, 0)));
        static assert(!__traits(compiles, h.put(h)));
        static assert(!__traits(compiles, { h.counts[0][0] = 0; }));
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

// A numeric entry is a snapshot, even when the view borrows local storage.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: VariableAxis, AxisOptions;

    static auto fromLocal() @safe pure nothrow @nogc
    {
        double[3] edges = [0, 1, 3];
        uint[2] counts = [1, 2];
        alias A = VariableAxis!(double*, AxisOptions());
        auto h = HistogramAccumulator!(uint[], A)(counts[], A(edges[].sliced));
        return h.bins[1];
    }
    const entry = fromLocal();
    assert(entry.index == 1 && entry.count == 2);
    assert(entry.bin.low == 1 && entry.bin.high == 3);

    version(mir_stat_test_lifetime)
    {
        // Copying an entry is safe; returning the borrowing view still is not.
        static assert(!__traits(compiles, () @safe {
            double[3] edges = [0, 1, 3];
            uint[2] counts = [1, 2];
            alias A = VariableAxis!(double*, AxisOptions());
            auto h = HistogramAccumulator!(uint[], A)(counts[], A(edges[].sliced));
            return h.bins;
        }));
    }
}

// Weighted updates share joint indexing for nested arrays and strided slices.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    alias A = IntegralAxis!(double, AxisOptions(false, true, true));
    double[4][4] nested = 0;
    auto h = HistogramAccumulator!(typeof(nested), A, A)(nested, A(2, 0), A(2, 0));
    h.putWeighted(0.5, 0.25, 1.25);
    h.putWeighted(2.0, -1.0, 3.0);
    assert(h.counts[1][2] == 0.5 && h.counts[0][3] == 2);
    auto marginal = h.rcMarginal!0();
    assert(marginal.counts == [2.0, 0.5, 0, 0]);
    double[16] raw = 0;
    auto view = raw[].sliced(4, 4).transposed;
    auto other = HistogramAccumulator!(typeof(view), A, A)(view, A(2, 0), A(2, 0));
    other.putWeighted(1.5, 0.25, 1.25);
    other.put(h);
    assert(other.counts[1, 2] == 2 && other.counts[0, 3] == 2);
    assert(raw[9] == 2);
}

// Integral weights are supported; fractional weights cannot silently truncate.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(double, AxisOptions());
    uint[2] storage = 0;
    auto h = HistogramAccumulator!(uint[], A)(storage[], A(2, 0));
    h.putWeighted(3u, 0.5);
    assert(h.counts == [3u, 0u]);
    static assert(!__traits(compiles, h.putWeighted(0.5, 0.5)));
    static assert(!__traits(compiles, h.putWeighted(1u, 0.5, 1.5)));
    static assert(!__traits(compiles, h.putWeighted(1u)));
    static assert(!__traits(compiles, h.putWeighted(1u, "invalid coordinate")));
}
