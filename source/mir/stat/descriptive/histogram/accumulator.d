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

import mir.primitives: DeepElementType;
import mir.stat.descriptive.histogram.traits: isAxis;
import mir.stat.descriptive.histogram.internal.view: supportsBinView;
import mir.qualifier: lightConst;
import std.meta: allSatisfy;
import std.traits: isNumeric, Unqual;
import mir.ndslice.slice: isSlice;

private template isJointStorage(Storage)
{
    import std.traits: isArray;
    static if (isSlice!Storage)
        enum isJointStorage = Storage.N == 2 && isNumeric!(DeepElementType!Storage);
    else static if (isArray!Storage && isArray!(typeof(Storage.init[0])))
        enum isJointStorage = isNumeric!(typeof(Storage.init[0][0]));
    else
        enum isJointStorage = false;
}

struct DenseStorage(Storage)
{
    import std.traits: isNumeric;

    Storage storage;

    void put(size_t i = 0)(size_t x)
        if (is(Storage : T[], T) ||
            is(Storage : T[N], T, size_t N))
    {
        storage[i]++;
    }

    void put(size_t i = 0)(size_t x)
        if (isNumeric!Storage)
    {
        storage++;
    }

    void put(size_t i = 0)()
        if (is(Storage : T[], T) ||
            is(Storage : T[N], T, size_t N))
    {
        storage[i]++;
    }

    void put(size_t i = 0)()
        if (isNumeric!Storage)
    {
        storage++;
    }
}

private
template put(size_t i)
{
    void put(Storage, T)(ref Storage storage, T x)
        if (isNumeric!T &&
            __traits(compiles, { storage[x]++; }))
    {
        storage[x]++;
    }
}

/++
Accumulator used to generate histogram.

With one axis, storage contains one count per ordinary bin. With two axes,
put(x, y) increments one joint bin. Storage must be a rank-2 ndslice or a
rectangular two-dimensional built-in array, with one dimension per axis.
Nested static arrays are copied into the accumulator; dynamic arrays and
ndslices share their backing counts. Keep axis definitions and storage shape
unchanged while using the accumulator.

Two-axis underflow, overflow, merging, and bin views are not yet supported.
More than two axes are also not yet supported.

If the `Axis` has an `options` member, the histogram may optionally allow
for overflow and underflow members.

Params:
    Axis = the type of the axis used to create the histogram bins

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis),
    $(LREF FrequencyAccumulator)
+/
struct HistogramAccumulator(Storage, Axis...)
    if (Axis.length > 0 &&
        allSatisfy!(isAxis, Axis))
{
    import std.meta: allSatisfy, anySatisfy, staticMap;
    import std.traits: hasMember, isIterable, isSomeString;
    import mir.primitives: hasShape, DeepElementType;
    import mir.stat.descriptive.histogram.traits: includeOverflow, includeUnderflow,
        BinTypeOf, isCategoryAxis, acceptsAxisValue;
    static if (Axis.length == 2 && !isSlice!Storage)
        /// Type of one joint-bin counter.
        alias CountType = typeof(Storage.init[0][0]);
    else
        /// Type of one bin counter.
        alias CountType = DeepElementType!Storage;
    static assert(Axis.length <= 2, "HistogramAccumulator: at most two axes are supported");
    static if (Axis.length == 2)
    {
        static assert(isJointStorage!Storage,
            "HistogramAccumulator: two axes require rank-2 ndslice or two-dimensional numeric array storage");
        static assert(!anySatisfy!(includeOverflow, Axis) &&
                      !anySatisfy!(includeUnderflow, Axis),
            "HistogramAccumulator: two-axis flow bins are not yet supported");
    }
private:
    static if (anySatisfy!(includeOverflow, Axis))
    {
        static if (N == 1) {
            ///
            DenseStorage!CountType overflowStorage;
        } else {
            ///
            DenseStorage!(CountType[N]) overflowStorage;
        }
    }

    static if (anySatisfy!(includeUnderflow, Axis))
    {
        static if (N == 1) {
            ///
            DenseStorage!CountType underflowStorage;
        } else {
            ///
            DenseStorage!(CountType[N]) underflowStorage;
        }
    }

public:

    ///
    Axis axis;

    ///
    Storage counts;

    /++
    Read-only random-access view of the ordinary bins and their counts.

    The view copies the axis and storage handles, sharing the count buffer.
    Subsequent count updates are visible when an element is read. Replacing
    this accumulator's axis or storage does not redirect an existing view.
    Keep shared axis boundaries and the storage shape unchanged while using it.

    Available for one axis with const bin-description access and supported
    one-dimensional storage. Underflow and overflow are excluded.
    Mutable and const histograms both return a view with a mutable cursor over
    read-only data. Custom axes must support mir.qualifier.lightConst.

    See_also: $(LREF HistogramBinView)
    +/
    auto bins()() const
        if (N == 1 && supportsBinView!(Storage, Axis[0]))
    {
        return HistogramBinView!(Storage, Axis[0])(counts, axis[0]);
    }

    //
    enum N = Axis.length;

    static if (anySatisfy!(includeOverflow, Axis))
    {
        ///
        alias OverflowType = typeof(overflowStorage.storage);
    }

    static if (anySatisfy!(includeUnderflow, Axis))
    {
        ///
        alias UnderflowType = typeof(underflowStorage.storage);
    }

    /++
    Construct an accumulator with storage matching the axes.
    Params:
        x = ordinary counts; one element per bin, or a grid matching both axes
        y = axes defining the bins
    +/
    this(Storage x, Axis y)
    {
        static if (N == 1)
            assert(x.length == y[0].N_bin,
                "HistogramAccumulator.this: storage length must match axis");
        else static if (isSlice!Storage)
        {
            assert(x.shape[0] == y[0].N_bin && x.shape[1] == y[1].N_bin,
                "HistogramAccumulator.this: storage shape must match both axes");
        }
        else
        {
            assert(x.length == y[0].N_bin,
                "HistogramAccumulator.this: row count must match first axis");
            foreach (ref row; x)
                assert(row.length == y[1].N_bin,
                    "HistogramAccumulator.this: every row must match second axis");
        }
        counts = x;
        axis = y;
    }

    ///
    void put(Range)(Range r)
        if (N == 1 &&
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
        if (acceptsArguments!T)
    {
        static if (N == 1)
        {
            static foreach (i; 0 .. T.length)
                putSingleImpl!(T[i], 0)(x[i]);
        }
        else
        {
            // Resolve both coordinates before changing counts, including when
            // an axis rejects an observation or returns an invalid index.
            auto row = axis[0].index(x[0]);
            auto column = axis[1].index(x[1]);
            assert(row >= 0 && row < axis[0].N_bin &&
                   column >= 0 && column < axis[1].N_bin,
                "HistogramAccumulator.put: joint bin index is out of range");
            static if (isSlice!Storage)
                counts[row, column]++;
            else
                counts[row][column]++;
        }
    }

    private
    void putSingleImpl(T, size_t i)(T x)
        if (acceptsAxisValue!(Axis[i], T))
    {
        static if (includeOverflow!(Axis[i]) && includeUnderflow!(Axis[i])) {
            if (axis[i].isOverflow(x)) {
                overflowStorage.put!i();
            } else if (axis[i].isUnderflow(x)) {
                underflowStorage.put!i();
            } else {
                counts.put!i(axis[i].index(x));
            }
        } else static if (!includeOverflow!(Axis[i]) && includeUnderflow!(Axis[i])) {
            if (axis[i].isUnderflow(x)) {
                underflowStorage.put!i();
            } else {
                counts.put!i(axis[i].index(x));
            }
        } else static if (includeOverflow!(Axis[i]) && !includeUnderflow!(Axis[i])) {
            if (axis[i].isOverflow(x)) {
                overflowStorage.put!i();
            } else {
                counts.put!i(axis[i].index(x));
            }
        } else {
            counts.put!i(axis[i].index(x));
        }
    }
    static if (N == 1)
    {
        /// Merge counts from another one-axis histogram with matching axes.
        void put(HistogramAccumulator!(Storage, Axis) h)
        {
            assert(axis == h.axis);
            counts[] += h.counts[];
            static if (includeOverflow!(Axis[0]))
                overflowStorage.storage += h.overflowStorage.storage;
            static if (includeUnderflow!(Axis[0]))
                underflowStorage.storage += h.underflowStorage.storage;
        }
    }

    static if (anySatisfy!(includeOverflow, Axis))
    {
        ///
        OverflowType overflow()() const
        {
            return overflowStorage.storage;
        }
    }

    static if (anySatisfy!(includeUnderflow, Axis))
    {
        ///
        UnderflowType underflow()() const
        {
            return underflowStorage.storage;
        }
    }
}

// Check IntegralAxis
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions())(5, 2.0);
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

    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions(false, true, true))(5, 2.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(integralAxis))(counts, integralAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
    h.put(3.0);
    assert(counts == [0, 1, 0, 0, 0]);
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

    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions(EnableUnderflow(true)))(5, 2.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(integralAxis))(counts, integralAxis);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.underflow == 1);
    h.put(3.0);
    assert(counts == [0, 1, 0, 0, 0]);
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
    EnumAxis!(size_t, Foo) enumAxis;
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
    CategoryAxis!(size_t, Foo, AxisOptions(false, true)) categoryAxis;
    size_t[] counts = [0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(categoryAxis))(counts, categoryAxis);
    h.put([Foo.A, Foo.B, Foo.B, Foo.B]);
    assert(counts == [1, 3]);
    h.put(Foo.A);
    assert(counts == [2, 3]);

    // Check strings
    h.put("B");
    assert(counts == [2, 4]);
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

    auto regularAxis = RegularAxis!(size_t, double, AxisOptions())(5, 2.0, 12.0);
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

    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(false, true, true))(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check RegularAxis, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(true))(5, 2.0, 12.0);
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

    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(true, true, true))(5, 2.0, 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(regularAxis))(counts, regularAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [0, 0, 0, 0, 0]);
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

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseTransformMapping!log10, AxisOptions())(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
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

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseTransformMapping!log10, AxisOptions(false, true, true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 1.0);
    assert(counts == [0, 0, 0, 0, 0]);
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

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseTransformMapping!log10, AxisOptions(true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
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

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseTransformMapping!log10, AxisOptions(true, true, true))(5, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(transformAxis))(counts, transformAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(10.0 ^^ 1.0);
    assert(counts == [0, 0, 0, 0, 0]);
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
    auto variableAxis = VariableAxis!(size_t, double*, AxisOptions())(axisSlice);
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
    auto variableAxis = VariableAxis!(size_t, double*, AxisOptions(false, true, true))(axisSlice);
    size_t[] counts = [0, 0, 0, 0, 0];

    auto h = HistogramAccumulator!(size_t[], typeof(variableAxis))(counts, variableAxis);
    assert(h.overflow == 0);
    assert(h.underflow == 0);
    h.put(13.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 0);
    h.put(1.0);
    assert(counts == [0, 0, 0, 0, 0]);
    assert(h.overflow == 1);
    assert(h.underflow == 1);
}

// Check put HistogramAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    auto integralAxis1 = IntegralAxis!(size_t, double, AxisOptions())(5, 2.0);
    size_t[] counts1 = [0, 0, 0, 0, 0];
    auto integralAxis2 = IntegralAxis!(size_t, double, AxisOptions())(5, 2.0);
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

    auto integralAxis1 = IntegralAxis!(size_t, double, AxisOptions(EnableOverflow(true), EnableUnderflow(true)))(5, 2.0);
    size_t[] counts1 = [0, 0, 0, 0, 0];
    auto integralAxis2 = IntegralAxis!(size_t, double, AxisOptions(EnableOverflow(true), EnableUnderflow(true)))(5, 2.0);
    size_t[] counts2 = [0, 0, 0, 0, 0];

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
@safe pure nothrow
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
    size_t[1] count = 0;

    auto h = HistogramAccumulator!(size_t[1], typeof(circleAxis))(count, circleAxis);
    auto p = Point(0.25, 0.5);
    h.put(p);
}

/// Count pairs of observations in a joint histogram backed by built-in arrays.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias A = IntegralAxis!(uint, double, AxisOptions());
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

    alias A = IntegralAxis!(uint, double, AxisOptions());
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
        size_t[] counts = [0, 0];
        auto h = HistogramAccumulator!(size_t[], Axis)(counts, axis);
        assert(!axis.isOverflow(axis.high));
        assert(!axis.isUnderflow(axis.low));

        h.put(axis.high);
        h.put(axis.low);
        static if (Axis.options.isRightClosed)
            assert(h.counts == [0, 2]);
        else
            assert(h.counts == [2, 0]);
        assert(h.overflow == 0);
        assert(h.underflow == 0);

        h.put(axis.high + 1.0);
        h.put(axis.low - 1.0);
        assert(h.overflow == 1);
        assert(h.underflow == 1);
        static if (Axis.options.isRightClosed)
            assert(h.counts == [0, 2]);
        else
            assert(h.counts == [2, 0]);
    }

    static foreach (rightClosed; [false, true])
    {{
        enum options = AxisOptions(rightClosed, true, true, true);
        check(IntegralAxis!(size_t, double, options)(2, 1.0));
        check(RegularAxis!(size_t, double, options)(2, 1.0, 9.0));
        check(TransformAxis!(size_t, double, sqrt, square, options)(2, 1.0, 9.0));
        check(VariableAxis!(size_t, double*, options)([1.0, 4.0, 9.0].sliced));
    }}
}

/++
A bin description and the count read when the element was accessed.

The index is the original ordinary-bin index, including when the view is sliced.
Elements are returned by value; assigning to count does not update the histogram.

Params:
    BinDescription = type returned by the axis's const bin accessor
    Count = histogram count type
+/
struct HistogramBin(BinDescription, Count)
{
    /// Original ordinary-bin index.
    size_t index;
    /// Axis-specific interval or category description.
    BinDescription bin;
    /// Count at the time this element was read.
    Count count;
}

/++
Read-only random-access range of ordinary bins and counts.

Usually obtained from a histogram's bins accessor. Includes zero-count bins;
underflow and overflow are excluded. Numeric bin descriptions expose low and
high, while enum and category descriptions expose slot. Interval closure and
circular behavior remain defined by the axis options.

The view copies the axis and storage handles without allocating a result array.
Dynamic arrays and one-dimensional Mir slices are supported. Shared count
updates are visible on later reads; previously returned counts are values.
Replacing the source histogram's axis or storage does not redirect the view.
Traversal and slicing change only the view's position.

Reference-counted handles retain their allocations. Borrowed storage, including
borrowed variable-axis breaks, must outlive the view and any bin descriptions
that refer to it. Keep storage shape and shared axis boundaries unchanged.
Custom axes must provide const bin access whose result does not allow mutation
of shared boundaries. They must also support mir.qualifier.lightConst: value-only
axes can be copied from const, while axes containing mutable references should
provide a lightConst accessor that preserves ownership and makes those references
read-only.

A const histogram can create a view with a mutable traversal position. A const
view can be indexed, saved, and sliced; save and slicing return independent
mutable cursors over the same read-only data. Updates through an existing mutable
histogram remain visible. Constness does not make the backing data immutable.

Params:
    Storage = dynamic array or one-dimensional Mir slice of numeric counts
    Axis = axis with const runtime bin-description access
+/
struct HistogramBinView(Storage, Axis)
    if (supportsBinView!(Storage, Axis))
{
    private alias ReadOnlyStorage = typeof(lightConst((const Storage).init));
    private alias ReadOnlyAxis = typeof(lightConst((const Axis).init));
    private ReadOnlyStorage _counts;
    private ReadOnlyAxis _axis;
    private size_t _begin;
    private size_t _end;

    /// Type of each returned value.
    alias Element = HistogramBin!(
        typeof((const ReadOnlyAxis).init.bin(size_t.init)),
        Unqual!(DeepElementType!ReadOnlyStorage));

    /// Construct a view with one count per ordinary bin.
    this(const Storage counts, const Axis axis)
    {
        assert(counts.length == axis.N_bin,
            "HistogramBinView: count length must match axis bin count");
        _counts = lightConst(counts);
        _axis = lightConst(axis);
        _begin = 0;
        _end = counts.length;
    }

    private this(ReadOnlyStorage counts, ReadOnlyAxis axis, size_t begin, size_t end)
    {
        _counts = counts;
        _axis = axis;
        _begin = begin;
        _end = end;
    }

    /// Number of remaining ordinary bins.
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
        return HistogramBinView(lightConst(_counts), lightConst(_axis), _begin, _end);
    }

    /// Read an element relative to the current range.
    Element opIndex(size_t index) const
    {
        assert(index < length, "HistogramBinView: index is out of range");
        auto originalIndex = _begin + index;
        return Element(originalIndex, _axis.bin(originalIndex), _counts[originalIndex]);
    }

    /// Return a subrange; element indices still refer to the original histogram.
    auto opSlice(size_t begin, size_t end) const
    {
        assert(begin <= end && end <= length,
            "HistogramBinView: slice is out of range");
        return HistogramBinView(lightConst(_counts), lightConst(_axis),
            _begin + begin, _begin + end);
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

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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

/// Index and slice a view without losing the original bin indices.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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
    alias Axis = CategoryAxis!(uint, Label, AxisOptions());
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

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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

    alias Axis = IntegralAxis!(uint, double, AxisOptions(false, true, true));
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
    check([0u, 0u, 0u], [5u, 0u, 0u]);
    check(rcslice!uint([0u, 0u, 0u]), rcslice!uint([5u, 0u, 0u]));
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
    checkNumeric(IntegralAxis!(uint, double, AxisOptions())(3, 0.0));
    checkNumeric(RegularAxis!(uint, double, AxisOptions(true))(3, 0.0, 6.0));
    checkNumeric(RegularAxis!(uint, double, AxisOptions(false, false, false, true))(3, 0.0, 6.0));
    checkNumeric(TransformAxis!(uint, double, "a * 2", "a / 2", AxisOptions())(3, 0.0, 6.0));

    auto breaks = [0.0, 1.0, 3.0, 6.0].sliced;
    checkNumeric(VariableAxis!(uint, double*, AxisOptions())(breaks));

    enum Label { first, second }
    alias Enum = EnumAxis!(uint, Label);
    alias Category = CategoryAxis!(uint, Label, AxisOptions());
    auto enums = HistogramAccumulator!(uint[], Enum)([2u, 4u], Enum()).bins;
    auto categories = HistogramAccumulator!(uint[], Category)([2u, 4u], Category()).bins;
    assert(enums[0].bin.slot == Label.first && enums[1].bin.slot == Label.second);
    assert(categories[0].bin.slot == Label.first && categories[1].count == 4);

    // The view, then the returned bin, retains reference-counted break storage.
    auto makeView()
    {
        auto ownedBreaks = rcslice!double([0.0, 2.0, 5.0]);
        auto axis = VariableAxis!(uint, typeof(ownedBreaks._iterator), AxisOptions())(ownedBreaks);
        auto counts = rcslice!uint([2u, 3u]);
        return HistogramAccumulator!(typeof(counts), typeof(axis))(counts, axis).bins;
    }
    auto owned = makeView();
    assert(owned[1].bin.low == 2.0 && owned[1].count == 3);
    auto description = owned[1].bin;
    owned = typeof(owned).init;
    assert(description.low == 2.0 && description.high == 5.0);
    static assert(!__traits(compiles, {
        description.low = 100.0;
    }));
}

// Invalid access and incompatible input are rejected.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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
    static assert(!__traits(compiles, Multi.init.bins()));

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

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
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
        alias Axis = VariableAxis!(uint, RCI!double, AxisOptions());
        auto counts = rcslice!uint([1u, 2u, 3u]);
        const h = HistogramAccumulator!(typeof(counts), Axis)(counts, Axis(breaks));
        const fixed = h.bins;
        return fixed.save[1 .. $];
    }
    auto view = makeView();
    assert(view.front.count == 2 && view.back.count == 3);
    assert(view.front.bin.low == 1.0 && view.back.bin.high == 6.0);
    static assert(is(typeof(view._counts) == Slice!(RCI!(const uint))));
    static assert(is(typeof(view._axis) ==
        VariableAxis!(uint, RCI!(const double), AxisOptions())));
    auto bin = view.front.bin;
    static assert(is(typeof(bin) == Bin!(Slice!(RCI!(const double)))));
    view = typeof(view).init;
    assert(bin.low == 1.0 && bin.high == 3.0);
}

// Additional storage forms keep const data readable and traversal independent.
version(mir_stat_test)
unittest
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis;
    import std.algorithm: map, equal;

    alias Axis = IntegralAxis!(uint, double, AxisOptions());
    uint[] backing = [1u, 99u, 2u, 99u, 3u, 99u];
    auto strided = Slice!(uint*, 1, SliceKind.universal)([3], [2], backing.ptr);
    const view = HistogramBinView!(typeof(strided), Axis)(strided, Axis(3, 0.0));
    assert(view.save.map!(e => e.count).equal([1u, 2u, 3u]));
    backing[2] = 4;
    assert(view[1].count == 4);

    const(uint)[] counts = [1u, 2u, 3u];
    const readOnly = HistogramBinView!(typeof(counts), Axis)(counts, Axis(3, 0.0));
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
    const view = HistogramBinView!(uint[], ReadOnlyAxis)(
        [2u], ReadOnlyAxis([0.0, 1.0]));
    assert(view.save.front.bin.high == 1.0 && view.front.count == 2);
}

// One axis accepts variadic batches; multiple axes require one coordinate each.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(size_t, double, AxisOptions());
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
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(size_t, double, AxisOptions());
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
    static immutable uint[2] zero = [0, 0];
    static immutable double[4] samples = [-1.0, 0.5, 1.5, 3.0];
    alias A = IntegralAxis!(uint, double, AxisOptions(false, true, true));
    auto counts = rcslice!uint(zero[]);
    alias H = HistogramAccumulator!(typeof(counts), A);
    auto h = H(counts, A(2, 0.0));
    h.put(samples[]);
    auto other = H(rcslice!uint(zero[]), A(2, 0.0));
    other.put(0.5);
    h.put(other);
    assert(h.counts[0] == 2 && h.counts[1] == 1);
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
    alias C = CategoryAxis!(uint, Label, AxisOptions());
    auto category = HistogramAccumulator!(typeof(counts), C)(rcslice!uint(zero[]), C());
    category.put(Label.second);
    assert(category.bins.back.count == 1);
    category.put("first");
    assert(category.bins.front.count == 1);
    alias FlowCategory = CategoryAxis!(uint, Label, AxisOptions(false, true));
    auto withFlow = HistogramAccumulator!(typeof(counts), FlowCategory)(
        rcslice!uint(zero[]), FlowCategory());
    withFlow.put("unknown");
    assert(withFlow.overflow == 1);
}

// Joint counts preserve associations even when the marginals are identical.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
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
    static assert(!__traits(compiles, diagonal.put(crossed)));
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
    alias A = IntegralAxis!(uint, double, AxisOptions());
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
    alias A = IntegralAxis!(uint, double, AxisOptions());
    alias C = CategoryAxis!(uint, Label, AxisOptions());
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
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, double, AxisOptions());
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

// Reject unsupported ranks and joint flow bins explicitly.
version(mir_stat_test)
unittest
{
    import mir.ndslice.slice: Slice;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions,
        EnableUnderflow, EnableOverflow;
    alias A = IntegralAxis!(uint, double, AxisOptions());
    alias U = IntegralAxis!(uint, double, AxisOptions(EnableUnderflow(true)));
    alias O = IntegralAxis!(uint, double, AxisOptions(EnableOverflow(true)));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[], A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][][], A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 1), A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(Slice!(uint*, 3), A, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][], U, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][], A, U).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][], O, A).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][], A, O).init));
    static assert(!__traits(compiles, HistogramAccumulator!(uint[][][], A, A, A).init));
}
