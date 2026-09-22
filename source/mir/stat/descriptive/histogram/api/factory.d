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
package auto initializeHistogram(bool insert = true, Storage, Axis, Data)(Storage counts, Axis axis, Data data)
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    auto h = HistogramAccumulator!(Storage, Axis)(counts, axis);
    // Floating-point .init is NaN; every counter must instead start at zero.
    foreach (ref count; h.counts)
        count = 0;
    static if (insert)
        h.put(data);
    return h;
}

// GC/RC storage already carries its own lifetime policy.
package struct NoAllocationContext {}

// Shared overloads keep allocation policy independent of axis construction.
package mixin template HistogramFactory(alias allocate, alias release = null, bool insert = true)
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.stat.descriptive.histogram.traits: isAxis, storageExtent, DefaultCountType;

    // Counter storage is selected here, independently of the axis metadata.
    auto factoryImplBasic(CountType = DefaultCountType, Context, Data, Axis)(
        ref Context context, scope Data data, Axis axis)
        if (isAxis!Axis)
    {
        import mir.stat.descriptive.histogram.api.factory: initializeHistogram;
        auto counts = allocate!CountType(context, storageExtent(axis));
        static if (!is(typeof(release) == typeof(null)))
            scope(failure) release(context, counts);
        return initializeHistogram!insert(counts, axis, data);
    }

    template factory(Options...)
    {
        auto factory(Context, Iterator, size_t N, SliceKind kind, Args...)(
            ref Context context, Slice!(Iterator, N, kind) data, auto ref Args args)
        {
            import std.traits: isNumeric, Unqual;
            static if (Options.length && is(Options[0]) && isNumeric!(Options[0]))
            {
                alias CountType = Unqual!(Options[0]);
                alias AxisSelection = Options[1 .. $];
            }
            else
            {
                alias CountType = DefaultCountType;
                alias AxisSelection = Options;
            }
            static if (AxisSelection.length == 0)
            {
                static assert(Args.length == 1 && isAxis!(Args[0]),
                    "Histogram factory: supply an axis instance or select an axis template");
                return factoryImplBasic!CountType(context, data, args[0]);
            }
            else
            {
                import mir.stat.descriptive.histogram.api.factory: constructHistogramAxis;
                auto axis = constructHistogramAxis!AxisSelection(data, args);
                import core.lifetime: move;
                return factoryImplBasic!CountType(context, data, axis.move);
            }
        }
    }
}

// Axis selection has no knowledge of counter storage or allocation policy.
package template constructHistogramAxis(Selection...)
{
    auto constructHistogramAxis(Data, Args...)(scope Data data, auto ref Args args)
    {
        import mir.stat.descriptive.histogram.traits: isAxis;
        import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
            RegularAxis, TransformAxis, VariableAxis, EnumAxis, CategoryAxis,
            integralAxis, regularAxis, transformAxis;
        import std.traits: Unqual;
        static if (Selection.length == 1 && is(Selection[0]) && isAxis!(Selection[0]))
        {
            alias Axis = Selection[0];
            return Axis(args);
        }
        else
        {
            static if (is(Selection[0]))
            {
                alias Coordinate = Selection[0];
                alias Axis = Selection[1];
                alias Rest = Selection[2 .. $];
            }
            else
            {
                alias Axis = Selection[0];
                alias Rest = Selection[1 .. $];
            }
            static if (Rest.length && is(typeof(Rest[$ - 1]) == AxisOptions))
            {
                enum options = Rest[$ - 1];
                alias Parameters = Rest[0 .. $ - 1];
            }
            else
            {
                enum options = AxisOptions();
                alias Parameters = Rest;
            }
            static if (__traits(isSame, Axis, VariableAxis))
            {
                static assert(Parameters.length == 0 && Args.length == 1);
                static if (is(Selection[0]))
                    return VariableAxis!(Coordinate, options)(args[0]);
                else
                {
                    import mir.stat.descriptive.histogram.axis: variableAxis;
                    return variableAxis!options(args[0]);
                }
            }
            else
            {
                static if (is(Selection[0]))
                    alias BinType = Coordinate;
                else static if ((__traits(isSame, Axis, TransformAxis) && Args.length == 2) ||
                    ((__traits(isSame, Axis, RegularAxis) || __traits(isSame, Axis, IntegralAxis)) && Parameters.length))
                    alias BinType = Unqual!(Data.DeepElement);
                else static if (Args.length)
                    alias BinType = Unqual!(Args[$ - 1]);
                else
                    alias BinType = Unqual!(Data.DeepElement);
                static if (__traits(isSame, Axis, IntegralAxis) || __traits(isSame, Axis, RegularAxis))
                {
                    static if (__traits(isSame, Axis, IntegralAxis))
                        alias makeAxis = integralAxis;
                    else
                        alias makeAxis = regularAxis;
                    static if (Parameters.length)
                    {
                        static assert(Parameters.length == 1);
                        return makeAxis!(BinType, Parameters[0], options)(data, args);
                    }
                    else
                        return Axis!(BinType, options)(args);
                }
                else static if (__traits(isSame, Axis, TransformAxis))
                {
                    static if (Args.length == 2)
                        return transformAxis!(BinType, Parameters, options)(data, args);
                    else
                        return transformAxis!(BinType, Parameters, options)(args);
                }
                else static if (__traits(isSame, Axis, EnumAxis))
                {
                    static assert(Parameters.length == 0 && Args.length == 0 && options == AxisOptions());
                    return EnumAxis!BinType();
                }
                else static if (__traits(isSame, Axis, CategoryAxis))
                {
                    static assert(Parameters.length == 0 && Args.length == 0);
                    return CategoryAxis!(BinType, options)();
                }
                else
                    static assert(0, "Histogram factory: unsupported axis template");
            }
        }
    }
}

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

    // Explicit regular-axis types preserve geometry and underflow/overflow options.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: RegularAxis, EnableOverflow;

        alias Axis = RegularAxis!(double, AxisOptions());
        auto data = [0.0, 1, 4, 5, 6, 9, 10, 13, 14].sliced;
        auto h = makeHistogram!Axis(data, 3u, 0.0, 15.0);
        assert(h.counts == [3u, 3u, 3u]);
        static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!size_t, Axis)));

        alias OverflowAxis = RegularAxis!(double, AxisOptions(EnableOverflow(true)));
        auto withOverflow = [1.0, 6.0, 11.0, 20.0].sliced;
        auto flow = makeHistogram!OverflowAxis(withOverflow, 3u, 0.0, 15.0);
        assert(flow.counts == [1u, 1u, 1u, 1u]);
        assert(flow.overflow == 1);
        static assert(is(flow.CountType == size_t));
    }

    // Explicit transform-axis types retain the custom transform and inverse.
    @safe pure nothrow
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: TransformAxis;

        static double transform(double x) { return x * x; }
        static double inverse(double x) { import mir.math.common: sqrt; return sqrt(x); }

        alias Axis = TransformAxis!(double, transform, inverse, AxisOptions());
        auto data = [0.5, 1.0, 2.5, 3.5].sliced;
        auto h = makeHistogram!Axis(data, 4u, 0.0, 4.0);
        assert(h.counts == [2u, 1u, 0u, 1u]);
        assert(h.axis[0].bin(0).high == 2.0);
        static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!size_t, Axis)));
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
            alias Axis = VariableAxis!(double*, AxisOptions());
            auto h = makeHistogram!(CountType, Axis)(data, breaks);
            assert(h.counts == [2u, 2u, 1u]);
            static assert(is(typeof(h) == HistogramAccumulator!(CountStorage!CountType, Axis)));

            alias RcAxis = VariableAxis!(RCI!double, AxisOptions());
            auto makeOwnedHistogram()
            {
                auto ownedBreaks = rcslice!double([0.0, 1.0, 3.0, 4.0]);
                return makeHistogram!(CountType, RcAxis)(data, ownedBreaks);
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

        alias Integral = IntegralAxis!(double, AxisOptions());
        auto h = makeHistogram!(uint, Integral)([0.0, 0.5, 1.5].sliced, 2u, 0.0);
        assert(h.counts == [2u, 1u]);
        static assert(is(h.CountType == uint));

        enum Label { first, second }
        auto labels = [Label.first, Label.second, Label.second].sliced;
        alias Enumerated = EnumAxis!(Label);
        alias Categorized = CategoryAxis!(Label, AxisOptions());
        auto e = makeHistogram!(uint, Enumerated)(labels);
        auto c = makeHistogram!(uint, Categorized)(labels);
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
            alias A = IntegralAxis!(double, AxisOptions(false, o, u));
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
        auto expected = makeHistogram(data[0 .. 3], data.integralAxis!(double, rule)(0.0));
        assert(integral.counts == expected.counts);
        static foreach (builtin; AliasSeq!(sturges, scott, freedmanDiaconis, sturges!uint))
        {{
            auto result = makeHistogram!(RegularAxis, builtin)(data, 0.0, 6.0);
            auto axis = data.regularAxis!builtin(0.0, 6.0);
            assert(result.counts == makeHistogram(data, axis).counts);
        }}
    }

    // Reject invalid rule results; bin count is independent of counter width.
    unittest
    {
        import mir.ndslice.slice: sliced;
        import mir.stat.descriptive.histogram.axis: RegularAxis, IntegralAxis, regularAxis;
        import std.exception: assertThrown;
        import std.meta: AliasSeq;
        import core.exception: AssertError;
        auto data = [0.0, 1].sliced;
        static foreach (value; AliasSeq!(0, -1))
        {{
            static auto invalid(S)(S values) { return value; }
            assertThrown!AssertError(makeHistogram!(ubyte, double, RegularAxis, invalid)(data, 0.0, 2.0));
            assertThrown!AssertError(data.regularAxis!(double, invalid)(0.0, 2.0));
        }}
        static double fractional(S)(S values) { return 2.5; }
        static bool boolean(S)(S values) { return true; }
        static uint wrongArgument(Unused = void)(string value) { return 2; }
        static foreach (rule; AliasSeq!(fractional, boolean, wrongArgument, 42))
        {{
            static assert(!__traits(compiles, makeHistogram!(RegularAxis, rule)(data, 0.0, 2.0)));
        }}
        static uint two(S)(S values) { return 2; }
        static assert(!__traits(compiles, makeHistogram!(IntegralAxis, two)(data, 0.0, 2.0)));
        static assert(!__traits(compiles, makeHistogram!(RegularAxis, two)(data, 0.0)));
        static auto boundary(S)(S values) { return 300UL; }
        auto h = makeHistogram!(ubyte, double, RegularAxis, boundary)(data, 0.0, 300.0);
        assert(h.axis[0].N_bin == 300 && h.counts.length == 300);
        static assert(is(h.CountType == ubyte));
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
        auto explicitType = data.transformAxis!(float, log2, floatRule)(1.0f, 16.0f);
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
                    static assert(is(h.CountType == size_t));
                    static assert(is(typeof(h.axis[0].N_bin()) == size_t));
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
            static assert(is(typeof(h.axis[0]) == VariableAxis!(RCI!double, AxisOptions())));
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
                VariableAxis!(immutable(double)*, AxisOptions())));

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
            alias Axis = RegularAxis!(double, AxisOptions(false, true, true));
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
    auto axis = RegularAxis!(double, AxisOptions())(2u, 0.0, 2.0);
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
        assert(ordinary.total == 3);
        assert(ordinary.relativeFrequency(0) == 2.0 / 3);
        enum options = AxisOptions(false, true, true);
        auto all = make!(ulong, double, RegularAxis, options)(
            values[].sliced, 2u, 0.0, 4.0);
        static assert(is(all.CountType == ulong));
        assert(all.total == 5);
        assert(all.underflowRelativeFrequency() == 0.2);
        assert(all.overflowRelativeFrequency() == 0.2);
        assert(all.cumulativeRelativeFrequency(1) == 0.8);
        all.put(3.0);
        assert(all.total == 6 && all.relativeFrequency(1) == 2.0 / 6);

        alias A = RegularAxis!(double, AxisOptions());
        double[0] empty;
        auto explicitAxis = make(empty[].sliced, A(2, 0, 4));
        assert(explicitAxis.total == 0 && isNaN(explicitAxis.relativeFrequency(0)));
        static uint two(S)(S data) { return 2; }
        double[4] powers = [1, 2, 4, 8];
        auto transformed = make!(TransformAxis, log2, exp2, two)(
            powers[].sliced(2, 2).transposed, 1.0, 16.0);
        assert(transformed.total == 4 && transformed.relativeFrequency(0) == 0.5);
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
        assert(f.total == 3 && f.relativeFrequency(1) == 2.0 / 3);
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
        assert(f.total == 2 && f.relativeFrequency(1) == 0.5);
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
        static assert(is(mutableResult.CountType == size_t));
        mutableResult.put(2.5);
        assert(mutableResult.counts == [2u, 1, 2]);
        static if (relative)
            assert(mutableResult.total == 5 && mutableResult.relativeFrequency(2) == 0.4);

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
            assert(frozen.total == 4 && frozen.relativeFrequency!float(0) == 0.5f);
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

// Compute ceil(cuberoot(n)) exactly, without floating-point rounding at cubes
// or overflow from cubing a candidate. This is a sample-size heuristic.
package size_t defaultPercentogramBinCount(size_t n) @safe pure nothrow @nogc
{
    assert(n > 0, "percentogram: observations must be nonempty");
    size_t low = 1;
    size_t high = size_t(1) << ((size_t.sizeof * 8 + 2) / 3);
    while (low < high)
    {
        const middle = low + (high - low) / 2;
        // middle^3 >= n is equivalent to middle > (n - 1) / middle^2.
        if (middle > (n - 1) / middle / middle)
            high = middle;
        else
            low = middle + 1;
    }
    return low;
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(defaultPercentogramBinCount(1) == 1);
    assert(defaultPercentogramBinCount(2) == 2);
    foreach (size_t root; [2, 5, 10, 100, 1000])
    {
        const cube = root * root * root;
        assert(defaultPercentogramBinCount(cube - 1) == root);
        assert(defaultPercentogramBinCount(cube) == root);
        assert(defaultPercentogramBinCount(cube + 1) == root + 1);
    }
    static if (size_t.sizeof == 8)
        assert(defaultPercentogramBinCount(size_t.max) == 2_642_246);
    else
        assert(defaultPercentogramBinCount(size_t.max) == 1_626);
}

// Quantile and count allocation policies stay paired: both results own their storage.
package auto buildPercentogram(alias allocate, alias quantiles, alias histogram,
    Data, P)(scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.ndslice.slice: isSlice, sliced;
    import mir.stat.descriptive.histogram.axis: variableAxis, AxisOptions;
    import std.traits: isIntegral;

    static if (isIntegral!P)
    {
        assert(probabilities > 0 && probabilities < size_t.max,
            "percentogram: bin count must be positive and leave room for an extra boundary");
        NoAllocationContext context;
        auto levels = allocate!double(context, cast(size_t) probabilities + 1);
        foreach (i; 0 .. levels.length)
            levels[i] = cast(double) i / probabilities;
        return buildPercentogram!(allocate, quantiles, histogram)(data, levels);
    }
    else
    {
        static if (isSlice!Data)
            scope auto observations = data;
        else
            scope auto observations = data[].sliced;
        static if (isSlice!P)
            scope auto levels = probabilities;
        else
            scope auto levels = probabilities[].sliced;
        validatePercentogramInputs(observations, levels);

        auto edges = quantiles(observations, levels);
        const distinct = preparePercentogramEdges(edges);
        edges = edges[0 .. distinct];
        // Quantiles may promote integral observations to floating-point boundaries.
        // Match the axis value type lazily without another observation buffer.
        import mir.ndslice.topology: as;
        import mir.primitives: DeepElementType;
        auto axis = variableAxis!(AxisOptions(false, true, true))(edges);
        return histogram(observations.as!(DeepElementType!(typeof(edges))), axis);
    }
}

version(mir_stat_test)
package void testPercentogramRejections(alias factory)()
{
    import core.exception: AssertError;
    import std.math: nextDown;
    import std.meta: AliasSeq;
    // Direct catches preserve nothrow inference while exercising contract failures.
    static void rejects(alias operation)()
    {
        bool rejected;
        try { operation(); }
        catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    rejects!(() { double[0] x; factory(x, 2); })();
    rejects!(() { double[2] x = [1, 1]; factory(x, 2); })();
    rejects!(() { double[3] x = [0, 0, 1]; double[2] p = [0, 0.25]; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; factory(x, 0); })();
    rejects!(() { double[2] x = [0, 1]; factory(x, -1); })();
    rejects!(() { double[2] x = [0, 1]; factory(x, size_t.max); })();
    rejects!(() { double[2] x = [0, double.nan]; factory(x, 2); })();
    rejects!(() { double[2] x = [0, double.infinity]; factory(x, 2); })();
    rejects!(() { double[2] x = [0, 1]; double[0] p; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; double[2] p = [-0.1, 1]; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; double[2] p = [0, 1.1]; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; double[3] p = [0, double.nan, 1]; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; double[4] p = [0, 0.5, 0.5, 1]; factory(x, p); })();
    rejects!(() { double[2] x = [0, 1]; double[4] p = [0, 0.75, 0.25, 1]; factory(x, p); })();
    static foreach (T; AliasSeq!(float, double, real))
    {{
        rejects!(() { T[2] x = [nextDown(T.max), T.max]; factory(x, 1); })();
    }}
}

// Use the same tie cases for each allocation policy and both probability APIs.
version(mir_stat_test)
package void testPercentogramDuplicates(alias factory)()
{
    import std.meta: AliasSeq;
    import std.math: nextUp, isFinite;
    // Type-7 quartiles select these sorted observations exactly. Every case
    // reduces to boundaries 0, 1, 2 before the final endpoint is extended.
    const int[5][4] samples = [
        [0, 1, 1, 1, 2], // duplicates only in the interior
        [0, 0, 0, 1, 2], // duplicates only at the minimum
        [0, 1, 2, 2, 2], // duplicates only at the maximum
        [0, 0, 1, 2, 2], // duplicates at both endpoints
    ];
    const uint[4] firstCounts = [1, 3, 1, 2];
    const double[5] probabilities = [0, 0.25, 0.5, 0.75, 1];
    static foreach (T; AliasSeq!(int, float, double, real))
    {{
        foreach (i; 0 .. samples.length)
        {
            T[5] data;
            // Reverse input order so preservation checks also catch in-place sorting.
            foreach (j; 0 .. data.length)
                data[j] = samples[i][$ - 1 - j];
            const original = data;
            auto explicitLevels = factory(data, probabilities);
            auto equalIntervals = factory(data, 4);
            assert(data == original);
            assert(probabilities[] == [0.0, 0.25, 0.5, 0.75, 1.0]);
            assert(explicitLevels.axis.N_bin == 2 && equalIntervals.axis.N_bin == 2);
            assert(explicitLevels.total == 5 && equalIntervals.total == 5);
            assert(explicitLevels.counts[1] == firstCounts[i]);
            assert(explicitLevels.counts[2] == 5 - firstCounts[i]);
            assert(equalIntervals.counts == explicitLevels.counts);
            double area = 0;
            foreach (j; 0 .. 2)
            {
                auto bin = explicitLevels.bins()[j].bin;
                auto other = equalIntervals.bins()[j].bin;
                assert(bin.low == j && bin.high > bin.low);
                if (j == 0)
                    assert(bin.high == 1);
                else
                    assert(bin.high == nextUp(typeof(bin.high)(2)));
                assert(other.low == bin.low && other.high == bin.high);
                const height = explicitLevels.density(j);
                assert(isFinite(height));
                area += height * cast(double) (bin.high - bin.low);
            }
            assert(area > 0.999999 && area < 1.000001);
        }
    }}
}

// Keep statistical rules shared without coupling custom allocation to GC/RC ownership.
package void validatePercentogramInputs(Observations, Levels)(scope Observations observations, scope Levels levels)
{
    import std.traits: isIntegral;
    import std.math: isFinite;
    static assert(Observations.N == 1 && Levels.N == 1,
        "percentogram: observations and probabilities must be one-dimensional");
    assert(observations.length > 0, "percentogram: observations must not be empty");
    foreach (x; observations)
    {
        static if (!isIntegral!(typeof(x)))
            assert(isFinite(x), "percentogram: observations must be finite");
    }
    assert(levels.length >= 2, "percentogram: at least two probabilities are required");
    assert(levels[0] >= 0 && levels[$ - 1] <= 1,
        "percentogram: probabilities must lie between zero and one");
    foreach (i; 1 .. levels.length)
        assert(levels[i] > levels[i - 1], "percentogram: probabilities must strictly increase");

}

// Compact in place but retain the complete allocation handle for manual cleanup.
package size_t preparePercentogramEdges(Edges)(scope Edges edges)
{
    import std.math: isFinite, nextUp;
    size_t distinct = 0;
    foreach (i; 0 .. edges.length)
    {
        assert(isFinite(edges[i]), "percentogram: quantile boundaries must be finite");
        if (distinct == 0 || edges[i] > edges[distinct - 1])
            edges[distinct++] = edges[i];
        else
            assert(edges[i] == edges[distinct - 1], "percentogram: boundaries must not decrease");
    }
    assert(distinct >= 2, "percentogram: at least two distinct boundaries are required");
    edges[distinct - 1] = nextUp(edges[distinct - 1]);
    assert(isFinite(edges[distinct - 1]), "percentogram: maximum requires a finite successor");
    return distinct;
}

// Test the same restricted intervals for owning and explicitly disposed results.
version(mir_stat_test)
package void testPercentogramIntervals(alias factory, alias release = null)()
{
    import mir.stat.descriptive.histogram.relative_frequency: Normalization;
    import std.math: nextUp, isNaN;
    static ref auto histogramOf(T)(return ref T value)
    {
        static if (__traits(hasMember, T, "histogram")) return value.histogram;
        else return value;
    }
    double[9] data = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const double[3][4] levels = [[0.25, 0.5, 0.75], [0.25, 0.5, 1],
        [0, 0.5, 0.75], [0.2, 0.5, 0.8]];
    const uint[4][4] expected = [[2, 2, 3, 2], [2, 2, 5, 0], [0, 4, 3, 2], [2, 2, 3, 2]];
    foreach (i; 0 .. levels.length)
    {
        auto result = factory(data, levels[i]);
        scope(exit) { static if (!is(typeof(release) == typeof(null))) release(result); }
        void check(H)(ref H p)
        {
            assert(p.total == 9 && p.counts == expected[i][]);
            assert(p.underflow == expected[i][0] && p.overflow == expected[i][3]);
            const ordinary = expected[i][1] + expected[i][2];
            assert(p.relativeFrequency(0) == cast(double) expected[i][1] / 9);
            assert(p.relativeFrequency!(double, Normalization.ordinary)(0) == cast(double) expected[i][1] / ordinary);
            assert(p.cumulativeRelativeFrequency!(double, Normalization.ordinary)(1) == 1);
            assert(p.cumulativeRelativeFrequency(1) == cast(double) (9 - expected[i][3]) / 9);
            double allArea = 0, ordinaryArea = 0;
            foreach (j; 0 .. 2)
            {
                auto bin = p.bins()[j].bin;
                const width = bin.high - bin.low;
                allArea += p.density(j) * width;
                ordinaryArea += p.density!(double, Normalization.ordinary)(j) * width;
            }
            assert(allArea > cast(double) ordinary / 9 - 1e-12 && allArea < cast(double) ordinary / 9 + 1e-12);
            assert(ordinaryArea > 1 - 1e-12 && ordinaryArea < 1 + 1e-12);
            const low = p.axis.low, high = p.axis.high;
            p.put(-1.0, 9.0);
            assert(p.total == 11 && p.underflow == expected[i][0] + 1 && p.overflow == expected[i][3] + 1);
            assert(p.axis.low == low && p.axis.high == high);
        }
        check(histogramOf(result));
    }
    {
        // Ties at both cutoffs belong to ordinary bins: seven of nine are retained.
        double[9] tied = [0, 1, 1, 1, 2, 3, 3, 3, 4];
        auto result = factory(tied, levels[0]);
        scope(exit) { static if (!is(typeof(release) == typeof(null))) release(result); }
        assert(histogramOf(result).counts == [1, 3, 4, 1]);
        assert(histogramOf(result).axis.high == nextUp(3.0));
    }
    {
        // An interpolated interval can contain no observed values at all.
        double[2] sparse = [0, 10];
        double[2] interval = [0.25, 0.75];
        auto result = factory(sparse, interval);
        scope(exit) { static if (!is(typeof(release) == typeof(null))) release(result); }
        assert(histogramOf(result).counts == [1, 0, 1]);
        assert(histogramOf(result).density(0) == 0);
        assert(isNaN(histogramOf(result).density!(double, Normalization.ordinary)(0)));
    }
    assert(data[] == [0.0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

// Reuse axis dispatch without first performing unweighted insertion. Observations
// remain available to bin-count rules; only the final insertion step differs.
package mixin template WeightedHistogramFactory(alias allocate, alias release = null)
{
    private mixin HistogramFactory!(allocate, release, false) emptyImplementation;

    template weightedFactory(Options...)
    {
        auto weightedFactory(Context, Data, Weights, Args...)(
            ref Context context, scope auto ref Data data,
            scope auto ref Weights weights, auto ref Args args)
        {
            import mir.ndslice.slice: isSlice, sliced;
            static if (isSlice!Data)
                scope auto observations = data.lightScope;
            else
                scope auto observations = data[].sliced;
            static if (isSlice!Weights)
                scope auto masses = weights.lightScope;
            else
                scope auto masses = weights[].sliced;
            static assert(observations.N == masses.N,
                "Weighted histogram observations and weights must have matching ranks");
            assert(observations.shape == masses.shape,
                "Weighted histogram observations and weights must have matching shapes");

            import mir.stat.descriptive.histogram.api.factory: insertWeighted;
            import std.traits: isNumeric;
            static if (Options.length && is(Options[0]) && isNumeric!(Options[0]))
                auto h = emptyImplementation.factory!Options(context, observations, args);
            else
                auto h = emptyImplementation.factory!(double, Options)(context, observations, args);
            static if (!is(typeof(release) == typeof(null)))
                scope(failure) release(context, h.counts);
            insertWeighted(h, observations, masses);
            return h;
        }
    }
}

package void insertWeighted(H, Data, Weights)(ref H h, scope Data data, scope Weights weights)
{
    // Recursing through matching shapes preserves logical pairing for arbitrary
    // strides, without allocating flattened copies or truncating either input.
    foreach (i; 0 .. data.length)
    {
        static if (Data.N == 1)
            h.putWeighted(weights[i], data[i]);
        else
            insertWeighted(h, data[i], weights[i]);
    }
}
