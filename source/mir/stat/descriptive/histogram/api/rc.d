/++
This module contains an API for creating reference-counted histograms.

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
    inverseTransformMapping, hasInverseTransformMapping, isTransformFunction;
import mir.stat.descriptive.histogram.traits: isAxis, storageExtent, acceptsBreakFunction;

/++
Params:
    x = input observations
    axis = axis defining the bins
+/
HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
    rchistogramImplBasic(Iterator, size_t N, SliceKind kind, Axis)(
               Slice!(Iterator, N, kind) x, Axis axis)
    if (isAxis!Axis)
{
    import mir.ndslice.allocation: mininitRcslice;

    auto counts = mininitRcslice!(Axis.CountType)(storageExtent(axis));
    foreach(ref e; counts) {
        e = 0;
    }
    auto h = HistogramAccumulator!(typeof(counts), Axis)(counts, axis);
    h.put(x);
    return h;
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
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    axisOptions = options
+/
private
template rchistogramImpl(CountType, BinType, alias Axis, AxisOptions axisOptions)
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: CategoryAxis, IntegralAxis, RegularAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), IntegralAxis!(CountType, BinType, axisOptions))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low)
        if (__traits(isSame, Axis, IntegralAxis))
    {
        import core.lifetime: move;

        auto integralAxis = IntegralAxis!(CountType, BinType, axisOptions)(N_bin, low);
        return .rchistogramImplBasic(slice.move, integralAxis);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), RegularAxis!(CountType, BinType, axisOptions))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, RegularAxis))
    {
        import core.lifetime: move;

        auto regularAxis = RegularAxis!(CountType, BinType, axisOptions)(N_bin, low, high);
        return .rchistogramImplBasic(slice.move, regularAxis);
    }

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), CategoryAxis!(CountType, BinType, axisOptions))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, CategoryAxis))
    {
        import core.lifetime: move;

        CategoryAxis!(CountType, BinType, axisOptions) categoryAxis;
        return .rchistogramImplBasic(slice.move, categoryAxis);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
+/
private
template rchistogramImpl(CountType, BinType, alias Axis)
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: EnumAxis;

    ///
    HistogramAccumulator!(Slice!(RCI!(CountType)), EnumAxis!(CountType, BinType))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, EnumAxis))
    {
        import core.lifetime: move;

        EnumAxis!(CountType, BinType) enumAxis;
        return .rchistogramImplBasic(slice.move, enumAxis);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
+/
private
template rchistogramImpl(CountType, BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions)
    if (__traits(isTemplate, Axis) &&
        isTransformFunction!(transform, BinType) &&
        isTransformFunction!(inverseTransform, BinType))
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        auto transformAxis = TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
        return .rchistogramImplBasic(slice.move, transformAxis);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    axisOptions = options
+/
private
template rchistogramImpl(CountType, BinType, alias Axis, alias transform, AxisOptions axisOptions)
    if (__traits(isTemplate, Axis) &&
        hasInverseTransformMapping!transform)
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
        rchistogramImpl(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        auto transformAxis = TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions)(N_bin, low, high);
        return .rchistogramImplBasic(slice.move, transformAxis);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Iterator = iterator used in slice
    Axis = type of axis
    axisOptions = options
+/
private
template rchistogramImpl(CountType, Iterator, alias Axis, AxisOptions axisOptions)
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/

    HistogramAccumulator!(Slice!(RCI!(CountType)), VariableAxis!(CountType, Iterator, axisOptions))
        rchistogramImpl(size_t N, SliceKind kindA, SliceKind kindB)(
                   Slice!(Iterator, N, kindA) dataSlice,
                   Slice!(Iterator, 1, kindB) axisSlice)
        if (__traits(isSame, Axis, VariableAxis))
    {
        import core.lifetime: move;

        auto variableAxis = VariableAxis!(CountType, Iterator, axisOptions)(axisSlice.move);
        return .rchistogramImplBasic(dataSlice.move, variableAxis.move);
    }
}

/++
Computes a reference-counted histogram of the inputs.
The allocated counts include enabled underflow/overflow bins, before and after
the ordinary bins respectively. Ordinary bin views exclude those end bins.

If the `Axis` has an `options` member, the histogram may optionally allow
for overflow and underflow members.

Params:
    slice = slice
    axis = axis

See_also:
    $(LREF HistogramAccumulator),
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
    rchistogram(Iterator, size_t N, SliceKind kind, Axis)(
               Slice!(Iterator, N, kind) slice, Axis axis)
    if (isAxis!Axis)
{
    import core.lifetime: move;
    return .rchistogramImplBasic(slice.move, axis);
}

/++
Choose the number of integral or regular bins using a rule on the observations.
The rule is called once with a light-scope view and must return a positive integer
representable by CountType. It must not mutate or retain the observation view.
Bounds remain explicit. The resulting histogram owns its count storage.

Params:
    CountType = count type
    BinType = axis value type
    Axis = IntegralAxis or RegularAxis
    breakFunction = callable returning the number of bins
    axisOptions = axis options
+/
template rchistogram(CountType, BinType, alias Axis, alias breakFunction,
    AxisOptions axisOptions = AxisOptions())
    if (isRuleAxis!Axis)
{
    /++
    Params:
        slice = input observations
        bounds = low for IntegralAxis; low and high for RegularAxis
    +/
    auto rchistogram(Iterator, size_t N, SliceKind kind, Bounds...)(
        Slice!(Iterator, N, kind) slice, Bounds bounds)
        if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
            validRuleBounds!(Axis, BinType, Bounds))
    {
        import mir.stat.descriptive.histogram.axis: integralAxis, regularAxis, IntegralAxis;
        static if (__traits(isSame, Axis, IntegralAxis))
            auto axis = integralAxis!(CountType, BinType, breakFunction, axisOptions)(slice, bounds);
        else
            auto axis = regularAxis!(CountType, BinType, breakFunction, axisOptions)(slice, bounds);
        return .rchistogramImplBasic(slice, axis);
    }
}

/++
Infer the axis value type from the bounds while specifying the count type.
Params:
    CountType = count type
    Axis = IntegralAxis or RegularAxis
    breakFunction = callable returning the number of bins
    axisOptions = axis options
+/
template rchistogram(CountType, alias Axis, alias breakFunction,
    AxisOptions axisOptions = AxisOptions())
    if (isRuleAxis!Axis)
{
    /// ditto
    auto rchistogram(Iterator, size_t N, SliceKind kind, BinType, Bounds...)(
        Slice!(Iterator, N, kind) slice, BinType low, Bounds rest)
        if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
            validRuleBounds!(Axis, BinType, BinType, Bounds))
    {
        return .rchistogram!(CountType, BinType, Axis, breakFunction, axisOptions)(slice, low, rest);
    }
}

/++
Use the default count type and infer the axis value type from the observations.
Params:
    Axis = IntegralAxis or RegularAxis
    breakFunction = callable returning the number of bins
    axisOptions = axis options
+/
template rchistogram(alias Axis, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (isRuleAxis!Axis)
{
    /// ditto
    auto rchistogram(Iterator, size_t N, SliceKind kind, Bounds...)(
        Slice!(Iterator, N, kind) slice, Bounds bounds)
        if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
            validRuleBounds!(Axis, typeof(slice).DeepElement, Bounds))
    {
        import mir.stat.descriptive.histogram.traits: DefaultCountType;
        import std.traits: Unqual;
        return .rchistogram!(DefaultCountType, Unqual!(typeof(slice).DeepElement),
            Axis, breakFunction, axisOptions)(slice, bounds);
    }
}

private template isRuleAxis(alias Axis)
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis;
    enum isRuleAxis = __traits(isSame, Axis, IntegralAxis) || __traits(isSame, Axis, RegularAxis);
}

private template validRuleBounds(alias Axis, BinType, Bounds...)
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis;
    static if (__traits(isSame, Axis, IntegralAxis))
        enum expected = 1;
    else
        enum expected = 2;
    static if (Bounds.length != expected)
        enum validRuleBounds = false;
    else static if (expected == 1)
        enum validRuleBounds = is(Bounds[0] : BinType);
    else
        enum validRuleBounds = is(Bounds[0] : BinType) && is(Bounds[1] : BinType);
}

/++
Params:
    Axis = type of axis
+/
template rchistogram(Axis)
    if (isAxis!Axis)
{
    import std.traits: isInstanceOf;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis,
        TransformAxis, EnumAxis, CategoryAxis, VariableAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low)
        if (isInstanceOf!(IntegralAxis, Axis))
    {
        import core.lifetime: move;

        auto integralAxis = Axis(N_bin, low);
        return .rchistogramImplBasic(slice.move, integralAxis);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
                    Slice!(Iterator, N, kind) slice,
                    CountType N_bin,
                    BinType low,
                    BinType high)
        if (isInstanceOf!(RegularAxis, Axis) || isInstanceOf!(TransformAxis, Axis))
    {
        import core.lifetime: move;

        auto axis = Axis(N_bin, low, high);
        return .rchistogramImplBasic(slice.move, axis);
    }

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
        rchistogram(Iterator, size_t N, SliceKind kind)(
                    Slice!(Iterator, N, kind) slice)
        if (isInstanceOf!(EnumAxis, Axis) || isInstanceOf!(CategoryAxis, Axis))
    {
        import core.lifetime: move;

        auto axis = Axis();
        return .rchistogramImplBasic(slice.move, axis);
    }

    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/
    HistogramAccumulator!(Slice!(RCI!(Axis.CountType)), Axis)
        rchistogram(DataIterator, AxisIterator, size_t N,
                    SliceKind kindA, SliceKind kindB)(
                    Slice!(DataIterator, N, kindA) dataSlice,
                    Slice!(AxisIterator, 1, kindB) axisSlice)
        if (isInstanceOf!(VariableAxis, Axis))
    {
        import core.lifetime: move;

        auto axis = Axis(axisSlice.move);
        return .rchistogramImplBasic(dataSlice.move, axis.move);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, BinType, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis, CategoryAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), IntegralAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low)
        if (__traits(isSame, Axis, IntegralAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), RegularAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, RegularAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low, high);
    }

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), CategoryAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, CategoryAxis) && is(BinType == enum))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
+/
template rchistogram(CountType, BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        isTransformFunction!(transform, BinType) &&
        isTransformFunction!(inverseTransform, BinType))
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    axisOptions = options
+/
template rchistogram(CountType, BinType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        hasInverseTransformMapping!transform)
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Iterator = iterator used in slice
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, Iterator, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: VariableAxis;

    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), VariableAxis!(CountType, Iterator, axisOptions))
        rchistogram(size_t N, SliceKind kindA, SliceKind kindB)(
                   Slice!(Iterator, N, kindA) dataSlice,
                   Slice!(Iterator, 1, kindB) axisSlice)
        if (__traits(isSame, Axis, VariableAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, Iterator, Axis, axisOptions)(dataSlice.move, axisSlice.move);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(BinType, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis, CategoryAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), IntegralAxis!(DefaultCountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   DefaultCountType N_bin,
                   BinType low)
        if (__traits(isSame, Axis, IntegralAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis, axisOptions)(slice.move, N_bin, low);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), RegularAxis!(DefaultCountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   DefaultCountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, RegularAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis, axisOptions)(slice.move, N_bin, low, high);
    }

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), CategoryAxis!(DefaultCountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, CategoryAxis) && is(BinType == enum))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis, axisOptions)(slice.move);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
+/
template rchistogram(BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        isTransformFunction!(transform, BinType) &&
        isTransformFunction!(inverseTransform, BinType))
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;


    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), TransformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   DefaultCountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis, transform, inverseTransform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
    transform = function to transform axis
    axisOptions = options
+/
template rchistogram(BinType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        hasInverseTransformMapping!transform)
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;


    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), TransformAxis!(DefaultCountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice,
                   DefaultCountType N_bin,
                   BinType low,
                   BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    Iterator = iterator used in slice
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(Iterator, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), VariableAxis!(DefaultCountType, Iterator, axisOptions))
        rchistogram(size_t N, SliceKind kindA, SliceKind kindB)(
                   Slice!(Iterator, N, kindA) dataSlice,
                   Slice!(Iterator, 1, kindB) axisSlice)
        if (__traits(isSame, Axis, VariableAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, Iterator, Axis, axisOptions)(dataSlice.move, axisSlice.move);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), IntegralAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, BinType)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low)
        if (__traits(isSame, Axis, IntegralAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), RegularAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, RegularAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    transform = function to transform axis
    inverseTransform = function to undo transform
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, TransformAxis) &&
            isTransformFunction!(transform, BinType) &&
            isTransformFunction!(inverseTransform, BinType))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    transform = function to transform axis
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        hasInverseTransformMapping!transform)
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: VariableAxis;

    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), VariableAxis!(CountType, IteratorB, axisOptions))
        rchistogram(IteratorA, size_t N, SliceKind kindA, IteratorB, SliceKind kindB)(
                   Slice!(IteratorA, N, kindA) dataSlice,
                   Slice!(IteratorB, 1, kindB) axisSlice)
        if (__traits(isSame, Axis, VariableAxis) &&
            is(DeepElementType!(Slice!(IteratorA, N, kindA)) : DeepElementType!(Slice!(IteratorB, 1, kindB))))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, IteratorB, Axis, axisOptions)(dataSlice.move, axisSlice.move);
    }
}

/++
Params:
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: CategoryAxis, IntegralAxis, RegularAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), IntegralAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
                   Slice!(Iterator, N, kind) slice,
                   CountType N_bin,
                   BinType low)
        if (__traits(isSame, Axis, IntegralAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low);
    }

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), RegularAxis!(CountType, BinType, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, RegularAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, axisOptions)(slice.move, N_bin, low, high);
    }

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), CategoryAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, CategoryAxis) && is(DeepElementType!(typeof(slice)) == enum))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, DeepElementType!(typeof(slice)), Axis, axisOptions)(slice.move);
    }
}

/++
Params:
    Axis = type of axis
+/
template rchistogram(alias Axis)
    if (__traits(isTemplate, Axis))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: EnumAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), EnumAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind))))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, EnumAxis) && is(DeepElementType!(typeof(slice)) == enum))
    {
        import core.lifetime: move;
        import mir.stat.descriptive.histogram.traits: DefaultCountType;

        return .rchistogramImpl!(DefaultCountType, DeepElementType!(typeof(slice)), Axis)(slice.move);
    }
}

/++
Params:
    Axis = type of axis
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
+/
template rchistogram(alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, TransformAxis) &&
            isTransformFunction!(transform, BinType) &&
            isTransformFunction!(inverseTransform, BinType))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    Axis = type of axis
    transform = function to transform axis
    axisOptions = options
+/
template rchistogram(alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) &&
        hasInverseTransformMapping!transform)
{
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    /++
    Params:
        slice = slice
        N_bin = number of bins
        low = the value of the smallest bin
        high = the value of the largest bin
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind, CountType, BinType)(
            Slice!(Iterator, N, kind) slice,
            CountType N_bin,
            BinType low,
            BinType high)
        if (__traits(isSame, Axis, TransformAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(slice.move, N_bin, low, high);
    }
}

/++
Params:
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: VariableAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        dataSlice = slice of data
        axisSlice = slice of axis breaks
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), VariableAxis!(DefaultCountType, IteratorB, axisOptions))
        rchistogram(IteratorA, size_t N, SliceKind kindA, IteratorB, SliceKind kindB)(
                   Slice!(IteratorA, N, kindA) dataSlice,
                   Slice!(IteratorB, 1, kindB) axisSlice)
        if (__traits(isSame, Axis, VariableAxis) &&
            is(DeepElementType!(Slice!(IteratorA, N, kindA)) : DeepElementType!(Slice!(IteratorB, 1, kindB))))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, IteratorB, Axis, axisOptions)(dataSlice.move, axisSlice.move);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    Axis = type of axis
+/
template rchistogram(BinType, alias Axis)
    if (__traits(isTemplate, Axis) && is(BinType == enum))
{
    import mir.stat.descriptive.histogram.axis: EnumAxis;
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(DefaultCountType)), EnumAxis!(DefaultCountType, BinType))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, EnumAxis))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(DefaultCountType, BinType, Axis)(slice.move);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Axis = type of axis
+/
template rchistogram(CountType, alias Axis)
    if (__traits(isTemplate, Axis) && !is(CountType == enum))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: EnumAxis;

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), EnumAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind))))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, EnumAxis) && is(DeepElementType!(typeof(slice)) == enum))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, DeepElementType!(typeof(slice)), Axis)(slice.move);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    Axis = type of axis
    axisOptions = options
+/
template rchistogram(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
    if (__traits(isTemplate, Axis) && !is(CountType == enum))
{
    import mir.primitives: DeepElementType;
    import mir.stat.descriptive.histogram.axis: CategoryAxis;

    /++
    Params:
        slice = slice
    +/
    HistogramAccumulator!(Slice!(RCI!(CountType)), CategoryAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions))
        rchistogram(Iterator, size_t N, SliceKind kind)(
                   Slice!(Iterator, N, kind) slice)
        if (__traits(isSame, Axis, CategoryAxis) && is(DeepElementType!(typeof(slice)) == enum))
    {
        import core.lifetime: move;

        return .rchistogramImpl!(CountType, DeepElementType!(typeof(slice)), Axis, axisOptions)(slice.move);
    }
}

/// Choose a regular-bin count using Sturges, retaining explicit bounds.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
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

// Explicit regular-axis types preserve their counter type and flow options.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, EnableOverflow;

    alias Axis = RegularAxis!(uint, double, AxisOptions());
    auto data = [0.0, 1, 4, 5, 6, 9, 10, 13, 14].sliced;
    auto h = data.rchistogram!Axis(3u, 0.0, 15.0);
    assert(h.counts == [3u, 3u, 3u]);
    static assert(is(typeof(h) == HistogramAccumulator!(Slice!(RCI!uint), Axis)));

    alias OverflowAxis = RegularAxis!(uint, double, AxisOptions(EnableOverflow(true)));
    auto withOverflow = [1.0, 6.0, 11.0, 20.0].sliced;
    auto flow = withOverflow.rchistogram!OverflowAxis(3u, 0.0, 15.0);
    assert(flow.counts == [1u, 1u, 1u, 1u]);
    assert(flow.overflow == 1);
    static assert(is(flow.CountType == uint));
}

// Explicit transform-axis types retain the custom transform and inverse.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: TransformAxis;

    static double transform(double x) { return x * x; }
    static double inverse(double x) { import mir.math.common: sqrt; return sqrt(x); }

    alias Axis = TransformAxis!(uint, double, transform, inverse, AxisOptions());
    auto data = [0.5, 1.0, 2.5, 3.5].sliced;
    auto h = data.rchistogram!Axis(4u, 0.0, 4.0);
    assert(h.counts == [2u, 1u, 0u, 1u]);
    assert(h.axis[0].bin(0).high == 2.0);
    static assert(is(typeof(h) == HistogramAccumulator!(Slice!(RCI!uint), Axis)));
}

// Explicit variable-axis types accept identical and different iterator types.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.axis: VariableAxis;

    static foreach (CountType; AliasSeq!(size_t, uint))
    {{
        auto data = [0.0, 0.5, 1.5, 2.5, 3.5].sliced;
        auto breaks = [0.0, 1.0, 3.0, 4.0].sliced;
        alias Axis = VariableAxis!(CountType, double*, AxisOptions());
        auto h = data.rchistogram!Axis(breaks);
        assert(h.counts == [2u, 2u, 1u]);
        static assert(is(typeof(h) == HistogramAccumulator!(Slice!(RCI!CountType), Axis)));

        alias RcAxis = VariableAxis!(CountType, RCI!double, AxisOptions());
        auto makeHistogram()
        {
            auto ownedBreaks = rcslice!double([0.0, 1.0, 3.0, 4.0]);
            return data.rchistogram!RcAxis(ownedBreaks);
        }
        auto mixed = makeHistogram();
        assert(mixed.counts == [2u, 2u, 1u]);
        static assert(is(typeof(mixed) == HistogramAccumulator!(Slice!(RCI!CountType), RcAxis)));
        // Break storage must survive the factory's local reference.
        mixed.put(2.0);
        assert(mixed.counts == [2u, 3u, 1u]);
        assert(mixed.axis[0].bin(1).high == 3.0);
    }}
}

// Other explicit-axis overloads use the same template-instance matching.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, EnumAxis, CategoryAxis;

    alias Integral = IntegralAxis!(uint, double, AxisOptions());
    auto h = [0.0, 0.5, 1.5].sliced.rchistogram!Integral(2u, 0.0);
    assert(h.counts == [2u, 1u]);
    static assert(is(h.CountType == uint));

    enum Label { first, second }
    auto labels = [Label.first, Label.second, Label.second].sliced;
    alias Enumerated = EnumAxis!(uint, Label);
    alias Categorized = CategoryAxis!(uint, Label, AxisOptions());
    auto e = labels.rchistogram!Enumerated();
    auto c = labels.rchistogram!Categorized();
    assert(e.counts == [1u, 2u]);
    assert(c.counts == [1u, 2u]);
    static assert(is(e.CountType == uint));
    static assert(is(c.CountType == uint));
}

// Infer the bin type while selecting the count type and both transforms.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: TransformAxis, IsRightClosed;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (right; [false, true])
    {{
        T[3] values = [10, 100, 1000];
        auto data = values[].sliced;
        enum options = AxisOptions(IsRightClosed(right));
        alias inverse = inverseTransformMapping!log10;
        auto inferred = rchistogram!(uint, TransformAxis, log10, inverse, options)(
            data, 2u, T(1), T(10000));
        auto explicitTypes = rchistogram!(uint, T, TransformAxis, log10, inverse, options)(
            data, 2u, T(1), T(10000));
        auto inferredInverse = rchistogram!(uint, TransformAxis, log10, options)(
            data, 2u, T(1), T(10000));
        static assert(is(typeof(inferred) == typeof(explicitTypes)));
        static assert(is(typeof(inferred).CountType == uint));
        assert(inferred.counts == (right ? [2u, 1u] : [1u, 2u]));
        auto defaultOptions = rchistogram!(uint, TransformAxis, log10, inverse)(
            data, 2u, T(1), T(10000));
        assert(defaultOptions.counts == [1u, 2u]);
        assert(inferred.counts == explicitTypes.counts);
        assert(inferred.counts == inferredInverse.counts);
        static assert(!__traits(compiles,
            rchistogram!(uint, TransformAxis, 42, inverse, options)(
                data, 2u, T(1), T(10000))));
        static assert(!__traits(compiles,
            rchistogram!(uint, TransformAxis, log10, 42, options)(
                data, 2u, T(1), T(10000))));
    }}
}


// Factory storage includes exactly the enabled underflow/overflow positions.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis;
    import mir.ndslice.slice: sliced;
    static foreach (u; [false, true])
    static foreach (o; [false, true])
    {{
        alias A = IntegralAxis!(uint, double, AxisOptions(false, o, u));
        auto h = [0.5, 1.5].sliced.rchistogram(A(2, 0.0));
        assert(h.counts.length == 2 + u + o);
        assert(h.counts[u] == 1 && h.counts[u + 1] == 1);
        static if (u) { h.put(-1.0); assert(h.counts[0] == 1); }
        static if (o) { h.put(2.0); assert(h.counts[$ - 1] == 1); }
        assert(h.bins.length == 2);
    }}
}

// Rules are invoked once; explicit axis construction gives identical results.
version(mir_stat_test)
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis,
        integralAxis, regularAxis, AxisOptions;
    import mir.stat.descriptive.histogram.breaks: sturges, scott, freedmanDiaconis;
    auto data = [0.0, 1, 2, 3, 4, 5].sliced;
    import std.meta: AliasSeq;
    static int calls;
    calls = 0;
    static size_t rule(S)(S values) { ++calls; return 3; }
    auto h = data.rchistogram!(RegularAxis, rule)(0.0, 6.0);
    assert(calls == 1 && h.counts == [2, 2, 2]);
    auto explicitAxis = data.regularAxis!rule(0.0, 6.0);
    assert(calls == 2 && data.rchistogram(explicitAxis).counts == h.counts);
    auto integral = data[0 .. 3].rchistogram!(uint, IntegralAxis, rule)(0.0);
    assert(calls == 3);
    static assert(is(integral.CountType == uint));
    auto expected = data[0 .. 3].rchistogram(data.integralAxis!(uint, double, rule)(0.0));
    assert(integral.counts == expected.counts);
    static foreach (builtin; AliasSeq!(sturges, scott, freedmanDiaconis, sturges!uint))
    {{
        auto result = data.rchistogram!(RegularAxis, builtin)(0.0, 6.0);
        auto axis = data.regularAxis!builtin(0.0, 6.0);
        assert(result.counts == data.rchistogram(axis).counts);
    }}
}

// Reject non-integer results and check narrowing before constructing the axis.
version(mir_stat_test)
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, IntegralAxis, regularAxis;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;
    import core.exception: AssertError;
    auto data = [0.0, 1].sliced;
    static foreach (value; AliasSeq!(0, -1, 256UL, ulong.max))
    {{
        static auto invalid(S)(S values) { return value; }
        assertThrown!AssertError(data.rchistogram!(ubyte, double, RegularAxis, invalid)(0.0, 2.0));
        assertThrown!AssertError(data.regularAxis!(ubyte, double, invalid)(0.0, 2.0));
    }}
    static double fractional(S)(S values) { return 2.5; }
    static bool boolean(S)(S values) { return true; }
    static uint wrongArgument(string value) { return 2; }
    static foreach (rule; AliasSeq!(fractional, boolean, wrongArgument, 42))
    {{
        static assert(!__traits(compiles, data.rchistogram!(RegularAxis, rule)(0.0, 2.0)));
    }}
    static uint two(S)(S values) { return 2; }
    static assert(!__traits(compiles, data.rchistogram!(IntegralAxis, two)(0.0, 2.0)));
    static assert(!__traits(compiles, data.rchistogram!(RegularAxis, two)(0.0)));
    static auto boundary(S)(S values) { return 255UL; }
    auto h = data.rchistogram!(ubyte, double, RegularAxis, boundary)(0.0, 255.0);
    assert(h.axis[0].N_bin == 255);
}
