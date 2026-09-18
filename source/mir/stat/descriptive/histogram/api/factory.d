/++
Shared initialization and axis overloads for histogram factories.

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

module mir.stat.descriptive.histogram.api.factory;

// Storage ownership stays with the caller. Validate before writing so a bad
// extent cannot clear unrelated storage before the constructor rejects it.
package auto initializeHistogram(Storage, Axis, Data)(Storage counts, Axis axis, Data data)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    auto h = HistogramAccumulator!(Storage, Axis)(counts, axis);
    // Floating-point .init is NaN; every counter must instead start at zero.
    foreach (ref count; h.counts)
        count = 0;
    h.put(data);
    return h;
}

// GC/RC storage already carries its own lifetime policy.
package struct NoAllocationContext {}

// Shared overloads keep allocation policy independent of axis construction.
package mixin template HistogramFactory(alias allocate, alias release = null)
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: AxisOptions,
        inverseTransformMapping, hasInverseTransformMapping, isTransformFunction,
        TransformAxis, acceptsTransformedBreakFunction;
    import mir.stat.descriptive.histogram.traits: isAxis, storageExtent, acceptsBreakFunction;

    // Distinct aliases keep recursive overload dispatch in the outer scope.
    private alias buildHistogram = factoryImplBasic;
    private alias buildAxisHistogram = factoryImpl;
    private alias dispatchHistogram = factory;
    import std.traits: ReturnType;
    // Infer storage without constructing or copying a stateful allocator.
    private auto allocationType(T, Context)(ref Context context)
    {
        return allocate!T(context, size_t.init);
    }
    private alias Storage(Context, T) = ReturnType!(allocationType!(T, Context));

    /++
    Params:
        x = input observations
        axis = axis defining the bins
    +/
    HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
        factoryImplBasic(Context, Iterator, size_t N, SliceKind kind, Axis)(
                   ref Context context, Slice!(Iterator, N, kind) x, Axis axis)
        if (isAxis!Axis)
    {
        auto counts = allocate!(Axis.CountType)(context, storageExtent(axis));
        // GC/RC storage manages its own lifetime; only caller allocation
        // needs an explicit failure handler.
        static if (!is(typeof(release) == typeof(null)))
            scope(failure) release(context, counts);
        import mir.stat.descriptive.histogram.api.factory: initializeHistogram;
        return initializeHistogram(counts, axis, x);
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    private
    template factoryImpl(CountType, BinType, alias Axis, AxisOptions axisOptions)
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: CategoryAxis, IntegralAxis, RegularAxis;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), IntegralAxis!(CountType, BinType, axisOptions))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low)
            if (__traits(isSame, Axis, IntegralAxis))
        {
            import core.lifetime: move;

            auto integralAxis = IntegralAxis!(CountType, BinType, axisOptions)(N_bin, low);
            return buildHistogram(context, slice.move, integralAxis);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), RegularAxis!(CountType, BinType, axisOptions))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, RegularAxis))
        {
            import core.lifetime: move;

            auto regularAxis = RegularAxis!(CountType, BinType, axisOptions)(N_bin, low, high);
            return buildHistogram(context, slice.move, regularAxis);
        }

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, CountType), CategoryAxis!(CountType, BinType, axisOptions))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, CategoryAxis))
        {
            import core.lifetime: move;

            CategoryAxis!(CountType, BinType, axisOptions) categoryAxis;
            return buildHistogram(context, slice.move, categoryAxis);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
    +/
    private
    template factoryImpl(CountType, BinType, alias Axis)
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: EnumAxis;

        ///
        HistogramAccumulator!(Storage!(Context, CountType), EnumAxis!(CountType, BinType))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, EnumAxis))
        {
            import core.lifetime: move;

            EnumAxis!(CountType, BinType) enumAxis;
            return buildHistogram(context, slice.move, enumAxis);
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
    template factoryImpl(CountType, BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions)
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            auto transformAxis = TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
            return buildHistogram(context, slice.move, transformAxis);
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
    template factoryImpl(CountType, BinType, alias Axis, alias transform, AxisOptions axisOptions)
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
            factoryImpl(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            auto transformAxis = TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions)(N_bin, low, high);
            return buildHistogram(context, slice.move, transformAxis);
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
    template factoryImpl(CountType, Iterator, alias Axis, AxisOptions axisOptions)
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: VariableAxis;
        /++
        Params:
            dataSlice = slice of data
            axisSlice = slice of axis breaks
        +/

        HistogramAccumulator!(Storage!(Context, CountType), VariableAxis!(CountType, Iterator, axisOptions))
            factoryImpl(Context, DataIterator, size_t N, SliceKind kindA, SliceKind kindB)(
                       ref Context context, Slice!(DataIterator, N, kindA) dataSlice,
                       // The returned axis may borrow these boundaries.
                       return scope Slice!(Iterator, 1, kindB) axisSlice)
            if (__traits(isSame, Axis, VariableAxis))
        {
            import core.lifetime: move;

            auto variableAxis = VariableAxis!(CountType, Iterator, axisOptions)(axisSlice.move);
            return buildHistogram(context, dataSlice.move, variableAxis.move);
        }
    }

    /++
    Computes a histogram of the inputs.
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
    HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
        factory(Context, Iterator, size_t N, SliceKind kind, Axis)(
                   ref Context context, Slice!(Iterator, N, kind) slice, Axis axis)
        if (isAxis!Axis)
    {
        import core.lifetime: move;
        return buildHistogram(context, slice.move, axis);
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
    template factory(CountType, BinType, alias Axis, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (isRuleAxis!Axis)
    {
        /++
        Params:
            slice = input observations
            bounds = low for IntegralAxis; low and high for RegularAxis
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind, Bounds...)(
            ref Context context, Slice!(Iterator, N, kind) slice, Bounds bounds)
            if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
                validRuleBounds!(Axis, BinType, Bounds))
        {
            import mir.stat.descriptive.histogram.axis: integralAxis, regularAxis, IntegralAxis;
            static if (__traits(isSame, Axis, IntegralAxis))
                auto axis = integralAxis!(CountType, BinType, breakFunction, axisOptions)(slice, bounds);
            else
                auto axis = regularAxis!(CountType, BinType, breakFunction, axisOptions)(slice, bounds);
            return buildHistogram(context, slice, axis);
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
    template factory(CountType, alias Axis, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (isRuleAxis!Axis)
    {
        /// ditto
        auto factory(Context, Iterator, size_t N, SliceKind kind, BinType, Bounds...)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, Bounds rest)
            if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
                validRuleBounds!(Axis, BinType, BinType, Bounds))
        {
            return dispatchHistogram!(CountType, BinType, Axis, breakFunction, axisOptions)(context, slice, low, rest);
        }
    }

    /++
    Use the default count type and infer the axis value type from the observations.
    Params:
        Axis = IntegralAxis or RegularAxis
        breakFunction = callable returning the number of bins
        axisOptions = axis options
    +/
    template factory(alias Axis, alias breakFunction, AxisOptions axisOptions = AxisOptions())
        if (isRuleAxis!Axis)
    {
        /// ditto
        auto factory(Context, Iterator, size_t N, SliceKind kind, Bounds...)(
            ref Context context, Slice!(Iterator, N, kind) slice, Bounds bounds)
            if (acceptsBreakFunction!(breakFunction, typeof(slice)) &&
                validRuleBounds!(Axis, typeof(slice).DeepElement, Bounds))
        {
            import mir.stat.descriptive.histogram.traits: DefaultCountType;
            import std.traits: Unqual;
            return dispatchHistogram!(DefaultCountType, Unqual!(typeof(slice).DeepElement),
                Axis, breakFunction, axisOptions)(context, slice, bounds);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        CountType = count type
        BinType = axis value type
        Axis = TransformAxis
        transform = forward transform
        inverseTransform = inverse transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(CountType, BinType, alias Axis, alias transform, alias inverseTransform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis))
    {
        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, typeof(slice)) &&
                isTransformFunction!(inverseTransform, BinType))
        {
            import mir.stat.descriptive.histogram.axis: transformAxis;
            auto axis = transformAxis!(CountType, BinType, transform, inverseTransform,
                breakFunction, axisOptions)(slice, low, high);
            return buildHistogram(context, slice, axis);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        CountType = count type
        Axis = TransformAxis
        transform = forward transform
        inverseTransform = inverse transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(CountType, alias Axis, alias transform, alias inverseTransform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis))
    {
        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, typeof(slice)) &&
                isTransformFunction!(inverseTransform, BinType))
        {
            return dispatchHistogram!(CountType, BinType, Axis, transform,
                inverseTransform, breakFunction, axisOptions)(context, slice, low, high);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        Axis = TransformAxis
        transform = forward transform
        inverseTransform = inverse transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(alias Axis, alias transform, alias inverseTransform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis))
    {
        import std.traits: Unqual;

        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, Unqual!(typeof(slice).DeepElement), typeof(slice)) &&
                isTransformFunction!(inverseTransform, Unqual!(typeof(slice).DeepElement)) && is(BinType : typeof(slice).DeepElement))
        {
            import mir.stat.descriptive.histogram.traits: DefaultCountType;
            return dispatchHistogram!(DefaultCountType, Unqual!(typeof(slice).DeepElement), Axis, transform,
                inverseTransform, breakFunction, axisOptions)(context, slice, low, high);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        CountType = count type
        BinType = axis value type
        Axis = TransformAxis
        transform = forward transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(CountType, BinType, alias Axis, alias transform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis) && hasInverseTransformMapping!transform)
    {
        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, typeof(slice)))
        {
            return dispatchHistogram!(CountType, BinType, Axis, transform,
                inverseTransformMapping!transform, breakFunction, axisOptions)(context, slice, low, high);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        CountType = count type
        Axis = TransformAxis
        transform = forward transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(CountType, alias Axis, alias transform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis) && hasInverseTransformMapping!transform)
    {
        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, typeof(slice)))
        {
            return dispatchHistogram!(CountType, BinType, Axis, transform,
                inverseTransformMapping!transform, breakFunction, axisOptions)(context, slice, low, high);
        }
    }

    /++
    Choose transformed-axis bin counts from transformed observations.
    Bounds remain in original units. The rule must return a positive integer count.
    Params:
        Axis = TransformAxis
        transform = forward transform
        breakFunction = rule applied to transformed observations
        axisOptions = axis options
    +/
    template factory(alias Axis, alias transform, alias breakFunction,
        AxisOptions axisOptions = AxisOptions())
        if (__traits(isSame, Axis, TransformAxis) && hasInverseTransformMapping!transform)
    {
        import std.traits: Unqual;

        /++
        Params:
            slice = input observations in original units
            low = lower bound in original units
            high = upper bound in original units
        +/
        auto factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
            ref Context context, Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (acceptsTransformedBreakFunction!(breakFunction, transform, Unqual!(typeof(slice).DeepElement), typeof(slice)) && is(BinType : typeof(slice).DeepElement))
        {
            import mir.stat.descriptive.histogram.traits: DefaultCountType;
            return dispatchHistogram!(DefaultCountType, Unqual!(typeof(slice).DeepElement), Axis, transform,
                inverseTransformMapping!transform, breakFunction, axisOptions)(context, slice, low, high);
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
    template factory(Axis)
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
        HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low)
            if (isInstanceOf!(IntegralAxis, Axis))
        {
            import core.lifetime: move;

            auto integralAxis = Axis(N_bin, low);
            return buildHistogram(context, slice.move, integralAxis);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                        ref Context context, Slice!(Iterator, N, kind) slice,
                        CountType N_bin,
                        BinType low,
                        BinType high)
            if (isInstanceOf!(RegularAxis, Axis) || isInstanceOf!(TransformAxis, Axis))
        {
            import core.lifetime: move;

            auto axis = Axis(N_bin, low, high);
            return buildHistogram(context, slice.move, axis);
        }

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
            factory(Context, Iterator, size_t N, SliceKind kind)(
                        ref Context context, Slice!(Iterator, N, kind) slice)
            if (isInstanceOf!(EnumAxis, Axis) || isInstanceOf!(CategoryAxis, Axis))
        {
            import core.lifetime: move;

            auto axis = Axis();
            return buildHistogram(context, slice.move, axis);
        }

        /++
        Params:
            dataSlice = slice of data
            axisSlice = slice of axis breaks
        +/
        HistogramAccumulator!(Storage!(Context, Axis.CountType), Axis)
            factory(Context, DataIterator, AxisIterator, size_t N,
                        SliceKind kindA, SliceKind kindB)(
                        ref Context context, Slice!(DataIterator, N, kindA) dataSlice,
                        Slice!(AxisIterator, 1, kindB) axisSlice)
            if (isInstanceOf!(VariableAxis, Axis))
        {
            import core.lifetime: move;

            auto axis = Axis(axisSlice.move);
            return buildHistogram(context, dataSlice.move, axis.move);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, BinType, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis, CategoryAxis;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), IntegralAxis!(CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low)
            if (__traits(isSame, Axis, IntegralAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), RegularAxis!(CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, RegularAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low, high);
        }

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, CountType), CategoryAxis!(CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, CategoryAxis) && is(BinType == enum))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, axisOptions)(context, slice.move);
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
    template factory(CountType, BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, transform, inverseTransform, axisOptions)(context, slice.move, N_bin, low, high);
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
    template factory(CountType, BinType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        Iterator = iterator used in slice
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, Iterator, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: VariableAxis;

        /++
        Params:
            dataSlice = slice of data
            axisSlice = slice of axis breaks
        +/
        HistogramAccumulator!(Storage!(Context, CountType), VariableAxis!(CountType, Iterator, axisOptions))
            factory(Context, size_t N, SliceKind kindA, SliceKind kindB)(
                       ref Context context, Slice!(Iterator, N, kindA) dataSlice,
                       Slice!(Iterator, 1, kindB) axisSlice)
            if (__traits(isSame, Axis, VariableAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, Iterator, Axis, axisOptions)(context, dataSlice.move, axisSlice.move);
        }
    }

    /++
    Params:
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    template factory(BinType, alias Axis, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, DefaultCountType), IntegralAxis!(DefaultCountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       DefaultCountType N_bin,
                       BinType low)
            if (__traits(isSame, Axis, IntegralAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), RegularAxis!(DefaultCountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       DefaultCountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, RegularAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low, high);
        }

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), CategoryAxis!(DefaultCountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, CategoryAxis) && is(BinType == enum))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis, axisOptions)(context, slice.move);
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
    template factory(BinType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, DefaultCountType), TransformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       DefaultCountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis, transform, inverseTransform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
        transform = function to transform axis
        axisOptions = options
    +/
    template factory(BinType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, DefaultCountType), TransformAxis!(DefaultCountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       DefaultCountType N_bin,
                       BinType low,
                       BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        Iterator = iterator used in slice
        Axis = type of axis
        axisOptions = options
    +/
    template factory(Iterator, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: VariableAxis;
        import mir.stat.descriptive.histogram.traits: DefaultCountType;

        /++
        Params:
            dataSlice = slice of data
            axisSlice = slice of axis breaks
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), VariableAxis!(DefaultCountType, Iterator, axisOptions))
            factory(Context, size_t N, SliceKind kindA, SliceKind kindB)(
                       ref Context context, Slice!(Iterator, N, kindA) dataSlice,
                       Slice!(Iterator, 1, kindB) axisSlice)
            if (__traits(isSame, Axis, VariableAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, Iterator, Axis, axisOptions)(context, dataSlice.move, axisSlice.move);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), IntegralAxis!(CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low)
            if (__traits(isSame, Axis, IntegralAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, CountType), RegularAxis!(CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, RegularAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low, high);
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
    template factory(CountType, alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, TransformAxis) &&
                isTransformFunction!(transform, BinType) &&
                isTransformFunction!(inverseTransform, BinType))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, transform, inverseTransform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        transform = function to transform axis
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, CountType), TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import mir.primitives: DeepElementType;
        import mir.stat.descriptive.histogram.axis: VariableAxis;

        /++
        Params:
            dataSlice = slice of data
            axisSlice = slice of axis breaks
        +/
        HistogramAccumulator!(Storage!(Context, CountType), VariableAxis!(CountType, IteratorB, axisOptions))
            factory(Context, IteratorA, size_t N, SliceKind kindA, IteratorB, SliceKind kindB)(
                       ref Context context, Slice!(IteratorA, N, kindA) dataSlice,
                       Slice!(IteratorB, 1, kindB) axisSlice)
            if (__traits(isSame, Axis, VariableAxis) &&
                is(DeepElementType!(Slice!(IteratorA, N, kindA)) : DeepElementType!(Slice!(IteratorB, 1, kindB))))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, IteratorB, Axis, axisOptions)(context, dataSlice.move, axisSlice.move);
        }
    }

    /++
    Params:
        Axis = type of axis
        axisOptions = options
    +/
    template factory(alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import std.traits: Unqual;
        import mir.primitives: DeepElementType;
        import mir.stat.descriptive.histogram.axis: CategoryAxis, IntegralAxis, RegularAxis;
        import mir.stat.descriptive.histogram.traits: DefaultCountType;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
        +/
        HistogramAccumulator!(Storage!(Context, Unqual!CountType), IntegralAxis!(Unqual!CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                       ref Context context, Slice!(Iterator, N, kind) slice,
                       CountType N_bin,
                       BinType low)
            if (__traits(isSame, Axis, IntegralAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(Unqual!CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low);
        }

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, Unqual!CountType), RegularAxis!(Unqual!CountType, BinType, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, RegularAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(Unqual!CountType, BinType, Axis, axisOptions)(context, slice.move, N_bin, low, high);
        }

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), CategoryAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, CategoryAxis) && is(DeepElementType!(typeof(slice)) == enum))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, DeepElementType!(typeof(slice)), Axis, axisOptions)(context, slice.move);
        }
    }

    /++
    Params:
        Axis = type of axis
    +/
    template factory(alias Axis)
        if (__traits(isTemplate, Axis))
    {
        import mir.primitives: DeepElementType;
        import mir.stat.descriptive.histogram.axis: EnumAxis;
        import mir.stat.descriptive.histogram.traits: DefaultCountType;

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), EnumAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind))))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, EnumAxis) && is(DeepElementType!(typeof(slice)) == enum))
        {
            import core.lifetime: move;
            import mir.stat.descriptive.histogram.traits: DefaultCountType;

            return buildAxisHistogram!(DefaultCountType, DeepElementType!(typeof(slice)), Axis)(context, slice.move);
        }
    }

    /++
    Params:
        Axis = type of axis
        transform = function to transform axis
        inverseTransform = function to undo transform
        axisOptions = options
    +/
    template factory(alias Axis, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis))
    {
        import std.traits: Unqual;
        import mir.stat.descriptive.histogram.axis: TransformAxis;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, Unqual!CountType), TransformAxis!(Unqual!CountType, BinType, transform, inverseTransform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, TransformAxis) &&
                isTransformFunction!(transform, BinType) &&
                isTransformFunction!(inverseTransform, BinType))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(Unqual!CountType, BinType, Axis, transform, inverseTransform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        Axis = type of axis
        transform = function to transform axis
        axisOptions = options
    +/
    template factory(alias Axis, alias transform, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis) &&
            hasInverseTransformMapping!transform)
    {
        import std.traits: Unqual;
        import mir.stat.descriptive.histogram.axis: TransformAxis;

        /++
        Params:
            slice = slice
            N_bin = number of bins
            low = the value of the smallest bin
            high = the value of the largest bin
        +/
        HistogramAccumulator!(Storage!(Context, Unqual!CountType), TransformAxis!(Unqual!CountType, BinType, transform, inverseTransformMapping!transform, axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind, CountType, BinType)(
                ref Context context, Slice!(Iterator, N, kind) slice,
                CountType N_bin,
                BinType low,
                BinType high)
            if (__traits(isSame, Axis, TransformAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(Unqual!CountType, BinType, Axis, transform, inverseTransformMapping!transform, axisOptions)(context, slice.move, N_bin, low, high);
        }
    }

    /++
    Params:
        Axis = type of axis
        axisOptions = options
    +/
    template factory(alias Axis, AxisOptions axisOptions = AxisOptions())
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
        HistogramAccumulator!(Storage!(Context, DefaultCountType), VariableAxis!(DefaultCountType, IteratorB, axisOptions))
            factory(Context, IteratorA, size_t N, SliceKind kindA, IteratorB, SliceKind kindB)(
                       ref Context context, Slice!(IteratorA, N, kindA) dataSlice,
                       Slice!(IteratorB, 1, kindB) axisSlice)
            if (__traits(isSame, Axis, VariableAxis) &&
                is(DeepElementType!(Slice!(IteratorA, N, kindA)) : DeepElementType!(Slice!(IteratorB, 1, kindB))))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, IteratorB, Axis, axisOptions)(context, dataSlice.move, axisSlice.move);
        }
    }

    /++
    Params:
        BinType = the type of the values that are compared in histogram bins
        Axis = type of axis
    +/
    template factory(BinType, alias Axis)
        if (__traits(isTemplate, Axis) && is(BinType == enum))
    {
        import mir.stat.descriptive.histogram.axis: EnumAxis;
        import mir.stat.descriptive.histogram.traits: DefaultCountType;

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, DefaultCountType), EnumAxis!(DefaultCountType, BinType))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, EnumAxis))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(DefaultCountType, BinType, Axis)(context, slice.move);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        Axis = type of axis
    +/
    template factory(CountType, alias Axis)
        if (__traits(isTemplate, Axis) && !is(CountType == enum))
    {
        import mir.primitives: DeepElementType;
        import mir.stat.descriptive.histogram.axis: EnumAxis;

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, CountType), EnumAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind))))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, EnumAxis) && is(DeepElementType!(typeof(slice)) == enum))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, DeepElementType!(typeof(slice)), Axis)(context, slice.move);
        }
    }

    /++
    Params:
        CountType = the type that is used to count in histogram bins
        Axis = type of axis
        axisOptions = options
    +/
    template factory(CountType, alias Axis, AxisOptions axisOptions = AxisOptions())
        if (__traits(isTemplate, Axis) && !is(CountType == enum))
    {
        import mir.primitives: DeepElementType;
        import mir.stat.descriptive.histogram.axis: CategoryAxis;

        /++
        Params:
            slice = slice
        +/
        HistogramAccumulator!(Storage!(Context, CountType), CategoryAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions))
            factory(Context, Iterator, size_t N, SliceKind kind)(
                       ref Context context, Slice!(Iterator, N, kind) slice)
            if (__traits(isSame, Axis, CategoryAxis) && is(DeepElementType!(typeof(slice)) == enum))
        {
            import core.lifetime: move;

            return buildAxisHistogram!(CountType, DeepElementType!(typeof(slice)), Axis, axisOptions)(context, slice.move);
        }
    }

}

// Run the same behavioral checks through all three public factories. Storage-specific
// examples and attribute checks remain beside the public APIs.
version(mir_stat_test)
{
    import mir.stat.descriptive.histogram.api.gc: histogram;
    import mir.stat.descriptive.histogram.api.rc: rchistogram;
    mixin FactoryTests!(histogram, true) gcTests;
    mixin FactoryTests!(rchistogram, false) rcTests;
    import mir.stat.descriptive.histogram.api.custom: customHistogramForTests;
    mixin FactoryTests!(customHistogramForTests, true) customTests;
}

version(mir_stat_test)
private mixin template FactoryTests(alias makeHistogram, bool gcCounts)
{
    mixin ConstFactoryTests!(makeHistogram, false);
    import mir.ndslice.slice: Slice;
    import mir.rc.array: RCI;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: AxisOptions, TransformAxis,
        inverseTransformMapping, isTransformFunction;
    static if (gcCounts)
        private alias CountStorage(T) = Slice!(T*);
    else
        private alias CountStorage(T) = Slice!(RCI!T);

    // Explicit regular-axis types preserve their counter type and flow options.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: RegularAxis, EnableOverflow;

        alias Axis = RegularAxis!(uint, double, AxisOptions());
        auto data = [0.0, 1, 4, 5, 6, 9, 10, 13, 14].sliced;
        auto h = makeHistogram!Axis(data, 3u, 0.0, 15.0);
        assert(h.counts == [3u, 3u, 3u]);
        static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!uint, Axis)));

        alias OverflowAxis = RegularAxis!(uint, double, AxisOptions(EnableOverflow(true)));
        auto withOverflow = [1.0, 6.0, 11.0, 20.0].sliced;
        auto flow = makeHistogram!OverflowAxis(withOverflow, 3u, 0.0, 15.0);
        assert(flow.counts == [1u, 1u, 1u, 1u]);
        assert(flow.overflow == 1);
        static assert(is(flow.CountType == uint));
    }

    // Explicit transform-axis types retain the custom transform and inverse.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: TransformAxis;

        static double transform(double x) { return x * x; }
        static double inverse(double x) { import mir.math.common: sqrt; return sqrt(x); }

        alias Axis = TransformAxis!(uint, double, transform, inverse, AxisOptions());
        auto data = [0.5, 1.0, 2.5, 3.5].sliced;
        auto h = makeHistogram!Axis(data, 4u, 0.0, 4.0);
        assert(h.counts == [2u, 1u, 0u, 1u]);
        assert(h.axis[0].bin(0).high == 2.0);
        static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!uint, Axis)));
    }

    // Explicit variable-axis types accept identical and different iterator types.
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
            auto h = makeHistogram!Axis(data, breaks);
            assert(h.counts == [2u, 2u, 1u]);
            static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!CountType, Axis)));

            alias RcAxis = VariableAxis!(CountType, RCI!double, AxisOptions());
            auto makeOwnedHistogram()
            {
                auto ownedBreaks = rcslice!double([0.0, 1.0, 3.0, 4.0]);
                return makeHistogram!RcAxis(data, ownedBreaks);
            }
            auto mixed = makeOwnedHistogram();
            assert(mixed.counts == [2u, 2u, 1u]);
            static assert(is(typeof(mixed) == HistogramAccumulator!(CountStorage!CountType, RcAxis)));
            // Break storage must survive the factory's local reference.
            mixed.put(2.0);
            assert(mixed.counts == [2u, 3u, 1u]);
            assert(mixed.axis[0].bin(1).high == 3.0);
        }}
    }

    // Other explicit-axis overloads use the same template-instance matching.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: IntegralAxis, EnumAxis, CategoryAxis;

        alias Integral = IntegralAxis!(uint, double, AxisOptions());
        auto h = makeHistogram!Integral([0.0, 0.5, 1.5].sliced, 2u, 0.0);
        assert(h.counts == [2u, 1u]);
        static assert(is(h.CountType == uint));

        enum Label { first, second }
        auto labels = [Label.first, Label.second, Label.second].sliced;
        alias Enumerated = EnumAxis!(uint, Label);
        alias Categorized = CategoryAxis!(uint, Label, AxisOptions());
        auto e = makeHistogram!Enumerated(labels);
        auto c = makeHistogram!Categorized(labels);
        assert(e.counts == [1u, 2u]);
        assert(c.counts == [1u, 2u]);
        static assert(is(e.CountType == uint));
        static assert(is(c.CountType == uint));
        // Template-only overloads infer the enum and use the default counter.
        auto inferredEnum = makeHistogram!EnumAxis(labels);
        auto inferredCategory = makeHistogram!CategoryAxis(labels);
        assert(inferredEnum.counts == [1, 2]);
        assert(inferredCategory.counts == [1, 2]);
    }

    // Infer the bin type while selecting the count type and both transforms.
    @safe pure nothrow
    unittest
    {
        void check()()
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
                auto inferred = makeHistogram!(uint, TransformAxis, log10, inverse, options)(
                    data, 2u, T(1), T(10000));
                auto explicitTypes = makeHistogram!(uint, T, TransformAxis, log10, inverse, options)(
                    data, 2u, T(1), T(10000));
                auto inferredInverse = makeHistogram!(uint, TransformAxis, log10, options)(
                    data, 2u, T(1), T(10000));
                static assert(is(typeof(inferred) == typeof(explicitTypes)));
                static assert(is(typeof(inferred).CountType == uint));
                assert(inferred.counts == (right ? [2u, 1u] : [1u, 2u]));
                auto defaultOptions = makeHistogram!(uint, TransformAxis, log10, inverse)(
                    data, 2u, T(1), T(10000));
                assert(defaultOptions.counts == [1u, 2u]);
                assert(inferred.counts == explicitTypes.counts);
                assert(inferred.counts == inferredInverse.counts);
                static assert(!__traits(compiles,
                    makeHistogram!(uint, TransformAxis, 42, inverse, options)(
                        data, 2u, T(1), T(10000))));
                static assert(!__traits(compiles,
                    makeHistogram!(uint, TransformAxis, log10, 42, options)(
                        data, 2u, T(1), T(10000))));
            }}
        }
        static if (gcCounts)
            check!()();
        else
            () @nogc { check!()(); }();
    }


    // Factory storage includes exactly the enabled underflow/overflow positions.
    @safe pure nothrow
    unittest
    {
        import mir.stat.descriptive.histogram.axis: IntegralAxis;
        import mir.ndslice.slice: sliced;
        static foreach (u; [false, true])
        static foreach (o; [false, true])
        {{
            alias A = IntegralAxis!(uint, double, AxisOptions(false, o, u));
            auto h = makeHistogram([0.5, 1.5].sliced, A(2, 0.0));
            assert(h.counts.length == 2 + u + o);
            assert(h.counts[u] == 1 && h.counts[u + 1] == 1);
            static if (u) { h.put(-1.0); assert(h.counts[0] == 1); }
            static if (o) { h.put(2.0); assert(h.counts[$ - 1] == 1); }
            assert(h.bins.length == 2);
        }}
    }

    // Rules are invoked once; explicit axis construction gives identical results.
    @safe nothrow
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
        auto h = makeHistogram!(RegularAxis, rule)(data, 0.0, 6.0);
        assert(calls == 1 && h.counts == [2, 2, 2]);
        auto explicitAxis = data.regularAxis!rule(0.0, 6.0);
        assert(calls == 2 && makeHistogram(data, explicitAxis).counts == h.counts);
        auto integral = makeHistogram!(uint, IntegralAxis, rule)(data[0 .. 3], 0.0);
        assert(calls == 3);
        static assert(is(integral.CountType == uint));
        auto expected = makeHistogram(data[0 .. 3], data.integralAxis!(uint, double, rule)(0.0));
        assert(integral.counts == expected.counts);
        static foreach (builtin; AliasSeq!(sturges, scott, freedmanDiaconis, sturges!uint))
        {{
            auto result = makeHistogram!(RegularAxis, builtin)(data, 0.0, 6.0);
            auto axis = data.regularAxis!builtin(0.0, 6.0);
            assert(result.counts == makeHistogram(data, axis).counts);
        }}
    }

    // Reject non-integer results and check narrowing before constructing the axis.
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
            assertThrown!AssertError(makeHistogram!(ubyte, double, RegularAxis, invalid)(data, 0.0, 2.0));
            assertThrown!AssertError(data.regularAxis!(ubyte, double, invalid)(0.0, 2.0));
        }}
        static double fractional(S)(S values) { return 2.5; }
        static bool boolean(S)(S values) { return true; }
        static uint wrongArgument(string value) { return 2; }
        static foreach (rule; AliasSeq!(fractional, boolean, wrongArgument, 42))
        {{
            static assert(!__traits(compiles, makeHistogram!(RegularAxis, rule)(data, 0.0, 2.0)));
        }}
        static uint two(S)(S values) { return 2; }
        static assert(!__traits(compiles, makeHistogram!(IntegralAxis, two)(data, 0.0, 2.0)));
        static assert(!__traits(compiles, makeHistogram!(RegularAxis, two)(data, 0.0)));
        static auto boundary(S)(S values) { return 255UL; }
        auto h = makeHistogram!(ubyte, double, RegularAxis, boundary)(data, 0.0, 255.0);
        assert(h.axis[0].N_bin == 255);
    }

    // Cover custom rules, precision/options overrides, and scalar/rule disambiguation.
    unittest
    {
        import mir.ndslice.slice: sliced, Slice;
        import mir.math.common: log2, exp2;
        import mir.stat.descriptive.histogram.axis: transformAxis;
        import mir.stat.descriptive.histogram.breaks: freedmanDiaconis, sturges;
        import std.exception: assertThrown;
        import core.exception: AssertError;
        static int calls;
        calls = 0;
        static uint rule(S)(S values)
        {
            ++calls;
            assert(values[0] == 0 && values[8] == 8);
            return 3;
        }
        auto data = [1.0, 2, 4, 8, 16, 32, 64, 128, 256].sliced;
        enum options = AxisOptions(false, true, true);
        auto a = makeHistogram!(uint, double, TransformAxis, log2, exp2, rule, options)(data, 1.0, 512.0);
        assert(calls == 1 && a.counts == [0, 3, 3, 3, 0]);
        auto b = makeHistogram!(uint, double, TransformAxis, log2, rule, options)(data, 1.0, 512.0);
        auto c = makeHistogram!(uint, TransformAxis, log2, exp2, rule, options)(data, 1.0, 512.0);
        auto d = makeHistogram!(uint, TransformAxis, log2, rule, options)(data, 1.0, 512.0);
        auto e = makeHistogram!(TransformAxis, log2, exp2, rule, options)(data, 1.0, 512.0);
        auto f = makeHistogram!(TransformAxis, log2, rule, options)(data, 1.0, 512.0);
        assert(calls == 6);
        assert(a.counts == b.counts && b.counts == c.counts && c.counts == d.counts);
        assert(d.counts == e.counts && e.counts == f.counts);
        static assert(is(a.CountType == uint));
        // Underflow/overflow still compare observations in original units.
        a.put(0.5); a.put(512.0);
        assert(a.underflow == 1 && a.overflow == 1);
        // Direct axis construction agrees with the factory and explicit mapping.
        import mir.ndslice.topology: map;
        auto direct = transformAxis!(log2, freedmanDiaconis)(data, 1.0, 512.0);
        assert(direct.N_bin == freedmanDiaconis(data.map!log2));
        assert(direct.N_bin == 3);
        // An ordinary function taking a slice is a rule, not a scalar transform.
        import mir.stat.descriptive.histogram.axis: transformedBreakData;
        alias Mapped = typeof(transformedBreakData!(double, log2)(data).lightScope);
        static uint typedRule(Mapped values) { return 3; }
        static assert(!isTransformFunction!(typedRule, double));
        auto typed = makeHistogram!(TransformAxis, log2, exp2, typedRule)(data, 1.0, 512.0);
        assert(typed.counts == [3, 3, 3]);
        static uint invalid(S)(S values) { return 0; }
        assertThrown!AssertError(makeHistogram!(TransformAxis, log2, invalid)(data, 1.0, 512.0));
        static double fractional(S)(S values) { return 3.5; }
        static assert(!__traits(compiles,
            makeHistogram!(TransformAxis, log2, fractional)(data, 1.0, 512.0)));
        // Explicit-count overloads still provide the original-data policy.
        auto originalCount = freedmanDiaconis(data);
        auto oldPolicy = makeHistogram!(TransformAxis, log2)(data, originalCount, 1.0, 512.0);
        assert(originalCount == 5 && oldPolicy.counts == [2, 2, 2, 2, 1]);
    }

    // Lazy transformed rules preserve attribute inference and string transforms.
    @safe pure nothrow
    unittest
    {
        void check()()
        {
            import mir.ndslice.slice: sliced;
            import mir.math.common: log2;
            import mir.stat.descriptive.histogram.breaks: sturges;
            static immutable data = [1.0f, 2, 4, 8];
            auto h = makeHistogram!(uint, float, TransformAxis,
                "log2(a)", "exp2(a)", sturges)(data[].sliced, 1.0f, 16.0f);
            auto inferred = makeHistogram!(TransformAxis, log2, sturges)(data[].sliced, 1.0f, 16.0f);
            assert(h.counts == inferred.counts);
            static assert(is(h.CountType == uint));
            static assert(is(h.axis[0].BinType == float));
        }
        static if (gcCounts)
            check!()();
        else
            () @nogc { check!()(); }();
    }

    // Inferred axis coordinates follow observations, even when bounds use another type.
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.math.common: log2, exp2;
        import mir.stat.descriptive.histogram.axis: transformAxis;
        import std.traits: Unqual;

        static uint doubleRule(S)(S values)
            if (is(Unqual!(S.DeepElement) == double))
        {
            assert(values[0] == 0 && values[3] == 3);
            return 2;
        }
        static uint floatRule(S)(S values)
            if (is(Unqual!(S.DeepElement) == float))
        { return 2; }
        // These transforms also accept only the selected coordinate type.
        static T forward(T)(T value) if (is(T == double)) { return log2(value); }
        static T inverse(T)(T value) if (is(T == double)) { return exp2(value); }

        auto data = [1.0, 2, 4, 8].sliced;
        auto a = makeHistogram!(TransformAxis, log2, doubleRule)(data, 1.0f, 16.0f);
        auto b = makeHistogram!(TransformAxis, forward, inverse, doubleRule)(data, 1.0f, 16.0f);
        static assert(is(a.axis[0].BinType == double));
        assert(a.counts == [2, 2] && b.counts == a.counts);
        auto c = data.transformAxis!(log2, doubleRule)(1.0f, 16.0f);
        auto d = data.transformAxis!(forward, inverse, doubleRule)(1.0f, 16.0f);
        static assert(is(c.BinType == double));
        assert(c.N_bin == 2 && d.N_bin == 2);

        // A rule accepting only the bounds' type must still be rejected.
        static assert(!__traits(compiles,
            makeHistogram!(TransformAxis, log2, floatRule)(data, 1.0f, 16.0f)));
        static assert(!__traits(compiles,
            data.transformAxis!(log2, exp2, floatRule)(1.0f, 16.0f)));
        // Explicit coordinate overrides continue to select float for the rule.
        auto explicitType = data.transformAxis!(uint, float, log2, floatRule)(1.0f, 16.0f);
        static assert(is(explicitType.BinType == float));
        assert(explicitType.N_bin == 2);
    }

    // Qualifiers on a bin-count value must not make newly allocated counters read-only.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis;
        import mir.math.common: log2, exp2;

        auto data = [1.0, 2, 2.5].sliced;
        import std.meta: AliasSeq;
        static foreach (CountArgument; AliasSeq!(uint, const(uint), immutable(uint)))
        {
            {
                CountArgument n = 2;
                auto integral = makeHistogram!IntegralAxis(data, n, 1.0);
                auto regular = makeHistogram!RegularAxis(data, n, 1.0, 3.0);
                auto transformed = makeHistogram!(TransformAxis, log2, exp2)(data, n, 1.0, 16.0);
                auto mapped = makeHistogram!(TransformAxis, log2)(data, n, 1.0, 16.0);
                static foreach (h; AliasSeq!(integral, regular, transformed, mapped))
                {
                    static assert(is(h.CountType == uint));
                    static assert(is(h.axis[0].CountType == uint));
                    h.put(1.5);
                }
                assert(integral.counts == [2, 2]);
                assert(regular.counts == [2, 2]);
                assert(transformed.counts == [4, 0]);
                assert(mapped.counts == [4, 0]);
                assert(n == 2);
                // An explicit counter override still takes precedence over the argument.
                auto overridden = makeHistogram!(ulong, double, RegularAxis)(data, n, 1.0, 3.0);
                static assert(is(overridden.CountType == ulong));
                overridden.put(1.5);
                assert(overridden.counts == [2, 2]);
            }
        }
    }

    // Observation and boundary iterators need not match; ownership follows the boundaries.
    @safe pure nothrow
    unittest
    {
        void check()()
        {
            import mir.ndslice.slice: sliced;
            import mir.ndslice.allocation: rcslice;
            import mir.stat.descriptive.histogram.axis: VariableAxis;

            static immutable samples = [0.5, 1.5, 2.5];
            static immutable edges = [0.0, 1.0, 3.0];
            auto boundaries = rcslice!double(edges);
            auto h = makeHistogram!VariableAxis(samples[].sliced, boundaries);
            static assert(is(typeof(h.axis[0]) == VariableAxis!(size_t, RCI!double, AxisOptions())));
            assert(h.counts == [1, 2]);
            // Release the caller's reference. The histogram must retain the boundaries.
            boundaries = typeof(boundaries).init;
            assert(h.axis[0].bin(1).high == 3.0);
            h.put(2.0);
            assert(h.counts == [1, 3]);

            // The reverse combination retains borrowed boundary semantics.
            auto observations = rcslice!double(samples);
            auto borrowed = makeHistogram!VariableAxis(observations, edges[].sliced);
            assert(borrowed.counts == [1, 2]);
            static assert(is(typeof(borrowed.axis[0]) ==
                VariableAxis!(size_t, immutable(double)*, AxisOptions())));

            // Counter overrides and underflow/overflow options use the same helper.
            enum options = AxisOptions(false, true, true);
            auto customized = makeHistogram!(uint, VariableAxis, options)(samples[].sliced, rcslice!double(edges));
            static assert(is(customized.CountType == uint));
            customized.put(-1.0);
            customized.put(3.0);
            assert(customized.counts == [1, 1, 2, 1]);
        }
        static if (gcCounts)
            check!()();
        else
            () @nogc { check!()(); }();
    }
    // Floating-point counters must start at zero, including empty end bins.
    @safe pure nothrow
    unittest
    {
        import std.meta: AliasSeq;
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: AxisOptions, RegularAxis;

        static foreach (T; AliasSeq!(float, double, real))
        {{
            alias Axis = RegularAxis!(T, double, AxisOptions(false, true, true));
            double[0] empty;
            auto h = makeHistogram(empty[].sliced, Axis(2, 0.0, 4.0));
            assert(h.counts == [0, 0, 0, 0]);
            // Once storage exists, counting requires no GC allocation.
            static void fill(H)(ref H value) @safe pure nothrow @nogc
            {
                value.put(-1.0);
                value.put(0.5);
                value.put(4.0);
            }
            fill(h);
            assert(h.counts == [1, 1, 0, 1]);
        }}
    }


    // Rank and layout describe observations, not the number of histogram axes.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.dynamic: transposed;
        import mir.stat.descriptive.histogram.axis: RegularAxis;

        double[6] values = [0.5, 0.25, 1.5, 0.25, 2.5, 0.25];
        auto matrix = values[].sliced(3, 2);
        auto all = makeHistogram!RegularAxis(matrix, 3u, 0.0, 3.0);
        assert(all.counts == [4, 1, 1]);
        auto transposedInput = makeHistogram!RegularAxis(matrix.transposed, 3u, 0.0, 3.0);
        assert(transposedInput.counts == [4, 1, 1]);
        // Selecting the first column skips every second element in storage.
        auto column = makeHistogram!RegularAxis(matrix.transposed[0], 3u, 0.0, 3.0);
        assert(column.counts == [1, 1, 1]);
    }

    // Owned counts cannot make an axis borrowing stack boundaries safe to escape.
    version(mir_stat_test_lifetime)
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.allocation: rcslice;
        import mir.stat.descriptive.histogram.axis: VariableAxis;

        static assert(!__traits(compiles, () @safe {
            double[3] edges = [0, 1, 3];
            double[1] values = [0.5];
            auto h = makeHistogram!VariableAxis(values[].sliced, edges[].sliced);
            return h;
        }));
        // Named slices exercise the ref branch of the public auto-ref wrapper.
        static assert(!__traits(compiles, () @safe {
            double[3] edges = [0, 1, 3];
            double[1] values = [0.5];
            auto boundaries = edges[].sliced;
            auto observations = values[].sliced;
            return makeHistogram!VariableAxis(observations, boundaries);
        }));
        // Changing only the boundary ownership permits the return.
        auto makeOwned() @safe pure nothrow
        {
            double[3] edges = [0, 1, 3];
            double[1] values = [0.5];
            auto h = makeHistogram!VariableAxis(values[].sliced, rcslice!double(edges[]));
            return h;
        }
        auto owned = makeOwned();
        assert(owned.counts == [1, 0]);
        assert(owned.axis[0].bin(1).high == 3.0);
    }
}

// Invalid storage is rejected before initialization can overwrite its contents.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import std.exception: assertThrown;
    import core.exception: AssertError;

    uint[3] counts = [7, 8, 9];
    double[0] observations;
    auto axis = RegularAxis!(uint, double, AxisOptions())(2u, 0.0, 2.0);
    assertThrown!AssertError(initializeHistogram(counts[].sliced, axis, observations[]));
    assert(counts == [7, 8, 9]);
}

// Exercise normalization and boundary ownership through every allocation policy.
version(mir_stat_test)
{
    import mir.stat.descriptive.histogram.api.gc: relativeFrequencyHistogram;
    import mir.stat.descriptive.histogram.api.rc: rcRelativeFrequencyHistogram;
    import mir.stat.descriptive.histogram.api.custom: customRelativeFrequencyForTests;
    mixin RelativeFrequencyFactoryTests!relativeFrequencyHistogram relativeGC;
    mixin RelativeFrequencyFactoryTests!rcRelativeFrequencyHistogram relativeRC;
    mixin RelativeFrequencyFactoryTests!customRelativeFrequencyForTests relativeCustom;
}

version(mir_stat_test)
private mixin template RelativeFrequencyFactoryTests(alias make)
{
    mixin ConstFactoryTests!(make, true);
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.dynamic: transposed;
        import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions, TransformAxis;
        import mir.math.common: log2, exp2;
        import std.math: isNaN;

        double[5] values = [-1, 0, 1, 2, 4];
        auto ordinary = make!RegularAxis(values[1 .. 4].sliced, 2u, 0.0, 4.0);
        assert(ordinary.count == 3);
        assert(ordinary.relativeFrequency(0) == 2.0 / 3);
        enum options = AxisOptions(false, true, true);
        auto all = make!(ulong, double, RegularAxis, options)(
            values[].sliced, 2u, 0.0, 4.0);
        static assert(is(all.CountType == ulong));
        assert(all.count == 5);
        assert(all.underflowRelativeFrequency() == 0.2);
        assert(all.overflowRelativeFrequency() == 0.2);
        assert(all.cumulativeRelativeFrequency(1) == 0.8);
        all.put(3.0);
        assert(all.count == 6 && all.relativeFrequency(1) == 2.0 / 6);

        alias A = RegularAxis!(uint, double, AxisOptions());
        double[0] empty;
        auto explicitAxis = make(empty[].sliced, A(2, 0, 4));
        assert(explicitAxis.count == 0 && isNaN(explicitAxis.relativeFrequency(0)));
        static uint two(S)(S data) { return 2; }
        double[4] powers = [1, 2, 4, 8];
        auto transformed = make!(TransformAxis, log2, exp2, two)(
            powers[].sliced(2, 2).transposed, 1.0, 16.0);
        assert(transformed.count == 4 && transformed.relativeFrequency(0) == 0.5);
    }

    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.ndslice.allocation: rcslice;
        import mir.stat.descriptive.histogram.axis: VariableAxis;
        static auto owned() @safe pure nothrow
        {
            double[2] values = [0.5, 2.0];
            auto edges = rcslice!double([0.0, 1, 3]);
            auto f = make!(uint, VariableAxis)(values[].sliced, edges);
            assert(edges.length == 3 && edges[2] == 3);
            edges = typeof(edges).init;
            return f;
        }
        auto f = owned();
        f.put(2.5);
        assert(f.count == 3 && f.relativeFrequency(1) == 2.0 / 3);
        auto saved = f.cumulativeRelativeFrequencies();
        assert(saved == [1.0 / 3, 1]);
    }

    version(mir_stat_test_lifetime)
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: VariableAxis;
        double[3] edges = [0, 1, 3];
        double[2] values = [0.5, 2.0];
        auto f = make!(uint, VariableAxis)(values[].sliced, edges[].sliced);
        assert(f.count == 2 && f.relativeFrequency(1) == 0.5);
        static assert(!__traits(compiles, () @safe {
            double[3] localEdges = [0, 1, 3];
            double[1] localValues = [0.5];
            auto boundaries = localEdges[].sliced;
            return make!(uint, VariableAxis)(localValues[].sliced, boundaries);
        }));
    }
}

// Const observations/boundaries do not make newly allocated counters const.
// Run through GC, RC, and caller-selected factories for both accumulator types.
version(mir_stat_test)
private mixin template ConstFactoryTests(alias make, bool relative)
{
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: RegularAxis, VariableAxis;

        const double[4] values = [0.5, 0.5, 1.5, 2.5];
        const observations = values[].sliced;
        const uint n = 3;
        auto mutableResult = make!RegularAxis(observations, n, 0.0, 3.0);
        static assert(is(mutableResult.CountType == uint));
        mutableResult.put(2.5);
        assert(mutableResult.counts == [2u, 1, 2]);
        static if (relative)
            assert(mutableResult.count == 5 && mutableResult.relativeFrequency(2) == 0.4);

        // Static storage lets this test cover const semantics without borrowing
        // local boundaries in builds that do not enable DIP1000.
        static const double[3] edges = [0, 1, 3];
        const boundaries = edges[].sliced;
        auto variable = make!(uint, VariableAxis)(observations, boundaries);
        variable.put(2.0);
        assert(variable.counts == [2u, 3]);

        const frozen = make!(uint, VariableAxis)(observations, boundaries);
        assert(frozen.counts == [2u, 2]);
        static assert(!__traits(compiles, frozen.put(0.5)));
        static assert(!__traits(compiles, { frozen.counts[0] = 0; }));
        static if (relative)
        {
            assert(frozen.count == 4 && frozen.relativeFrequency!float(0) == 0.5f);
            assert(frozen.cumulativeRelativeFrequency(1) == 1);
            assert(frozen.cumulativeRelativeFrequencies() == [0.5, 1]);
            double[2] destination;
            frozen.cumulativeRelativeFrequencies(destination[]);
            assert(destination[] == [0.5, 1]);
        }
        version(mir_stat_test_lifetime)
        {
            auto bins = frozen.bins();
            assert(bins.front.count == 2);
            bins.popFront();
            assert(bins.front.count == 2);
            static if (relative)
            {
                auto frequencies = frozen.relativeFrequencyBins();
                assert(frequencies.front.relativeFrequency == 0.5);
                auto cumulative = frozen.cumulativeRelativeFrequencyBins();
                cumulative.popFront();
                assert(cumulative.front.cumulativeRelativeFrequency == 1);
            }
        }
    }
}
