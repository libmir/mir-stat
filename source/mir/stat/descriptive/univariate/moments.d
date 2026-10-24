/++
Raw, central, and standardized moments.

Select the moment order with the template parameter `N`. Use a named moment
function or $(REF moment, mir, stat, descriptive, univariate, moments) with $(REF MomentAlgo, mir, stat, descriptive, univariate, moments) to select the kind of moment.

Skewness and kurtosis are implemented in $(MREF mir,stat,descriptive,univariate,skewness_kurtosis)
and re-exported here for compatibility.

Import `mir.stat.descriptive.univariate.moments` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF rawMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF centralMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF standardizedMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF moment, mir, stat, descriptive, univariate, moments)$(NBSP)
    ))
    $(TR $(TD Accumulator) $(TD
        $(REF MomentAccumulator, mir, stat, descriptive, univariate, moments)$(NBSP)
    ))
    $(TR $(TD Algorithm selection) $(TD
        $(REF MomentAlgo, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF StandardizedMomentAlgo, mir, stat, descriptive, univariate, moments)$(NBSP)
    ))
))

$(SCRIPT
// Preserve links to shape statistics previously documented on this page.
(function () {
    var owners = {"KurtosisAccumulator": "skewness_kurtosis", "KurtosisAlgo": "skewness_kurtosis", "SkewnessAccumulator": "skewness_kurtosis", "SkewnessAlgo": "skewness_kurtosis", "kurtosis": "skewness_kurtosis", "skewness": "skewness_kurtosis"};
    var anchor = window.location.hash.substring(1);
    var symbol = anchor.replace(/^\./, '').split('.')[0];
    if (Object.prototype.hasOwnProperty.call(owners, symbol)) {
        window.location.replace('mir_stat_descriptive_univariate_skewness_kurtosis.html#' + anchor);
    }
})();
)

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall, Ilya Yaroshenko
Copyright: 2022-3 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.univariate.moments;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType, stdevType;
import mir.stat.descriptive.univariate.means: mean, MeanAccumulator;
import mir.stat.descriptive.univariate.dispersions: variance, standardDeviation, VarianceAlgo, VarianceAccumulator;

/// Re-export shape statistics for existing moments imports.
public import mir.stat.descriptive.univariate.skewness_kurtosis: skewness, kurtosis, SkewnessAccumulator, KurtosisAccumulator, SkewnessAlgo, KurtosisAlgo;

///
struct MomentAccumulator(T, size_t N, Summation summation)
    if (N > 0 && isMutable!T)
{
    import std.traits: isIterable;

    ///
    Summator!(T, summation) summator;

    ///
    size_t count;

    ///
    F moment(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) summator.sum / cast(F) count;
    }

    ///
    F sumOfPower(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) summator.sum;
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        import mir.math.internal.powi: powi;
        import mir.primitives: hasShape;

        static if (hasShape!Range)
        {
            import mir.ndslice.topology: map;
            import mir.primitives: elementCount;

            count += r.elementCount;
            // Match scalar insertion: convert before taking the power, so a
            // narrower input does not overflow or underflow before summation.
            summator.put(r.map!(a => (cast(T) a).powi(N)));
        }
        else
        {
            foreach(x; r)
            {
                put(x);
            }
        }
    }

    ///
    void put(Range)(Range r, T m)
        if (isIterable!Range)
    {
        import mir.math.internal.powi: powi;
        import mir.primitives: hasShape;

        static if (hasShape!Range)
        {
            import mir.ndslice.internal: LeftOp;
            import mir.ndslice.topology: vmap, map;
            import mir.primitives: elementCount;

            count += r.elementCount;
            static if (N == 1)
            {
                summator.put(r.vmap(LeftOp!("-", T)(m))
                    );
            } else static if (N == 2) {
                summator.put(r.vmap(LeftOp!("-", T)(m)).map!"a * a"
                    );
            } else {
                summator.put(r.vmap(LeftOp!("-", T)(m)).
                               map!(a => a.powi(N))
                    );
            }
        }
        else
        {
            foreach(x; r)
            {
                put(x, m);
            }
        }
    }

    ///
    void put(Range)(Range r, T m, T s)
        if (isIterable!Range)
    {
        import mir.math.internal.powi: powi;
        import mir.primitives: hasShape;

        static if (hasShape!Range)
        {
            import mir.ndslice.internal: LeftOp;
            import mir.ndslice.topology: vmap, map;
            import mir.primitives: elementCount;

            count += r.elementCount;
            static if (N == 1)
            {
                summator.put(r.vmap(LeftOp!("-", T)(m)).
                               vmap(LeftOp!("*", T)(1 / s))
                    );
            } else static if (N == 2) {
                summator.put(r.vmap(LeftOp!("-", T)(m)).
                               vmap(LeftOp!("*", T)(1 / s)).
                               map!"a * a"
                    );
            } else {
                summator.put(r.vmap(LeftOp!("-", T)(m)).
                               vmap(LeftOp!("*", T)(1 / s)).
                               map!(a => a.powi(N))
                    );
            }

        }
        else
        {
            foreach(x; r)
            {
                put(x, m, s);
            }
        }
    }

    ///
    void put()(T x)
    {
        import mir.math.internal.powi;

        count++;
        summator.put(x.powi(N));
    }

    /++
    Accumulate the N-th power of $(D x - m).
    Use the same center for every observation contributing to this moment;
    previously accumulated powers are not adjusted when the center changes.
    +/
    void put()(T x, T m)
    {
        put(x - m);
    }

    /++
    Accumulate the N-th power of $(D (x - m) * (1 / s)), matching batch insertion.
    Use the same center and scale for every observation. The caller supplies
    the scale; this overload does not estimate it from the observations.
    +/
    void put()(T x, T m, T s)
    {
        put((x - m) * (1 / s));
    }

    /++
    Combine counts and accumulated powers without modifying the source.
    KBN, KB2, and precise summation preserve the source's correction terms
    rather than first rounding them to a single value. For centered or
    standardized moments, both accumulators must use the same center and scale.
    +/
    void put()(MomentAccumulator!(T, N, summation) m)
    {
        count += m.count;
        static if (summation == Summation.kbn || summation == Summation.kb2 ||
            summation == Summation.precise)
        {
            import mir.stat.internal.summation: mergeSummator;

            // Preserve corrections that would be lost by rounding the source
            // to one value. The by-value parameter also makes self-merges safe.
            mergeSummator(summator, m.summator);
        }
        else
            summator.put(m.summator.sum);
    }

    ///
    this(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        this.put(r.move);
    }

    ///
    this(Range)(Range r, T m)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        this.put(r.move, m);
    }

    ///
    this(Range)(Range r, T m, T s)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        this.put(r.move, m, s);
    }

    ///
    this()(T x)
    {
        this.put(x);
    }

    ///
    this()(T x, T m)
    {
        this.put(x, m);
    }

    ///
    this()(T x, T m, T s)
    {
        this.put(x, m, s);
    }
}

// Centered and standardized scalar constructors and non-shaped input ranges
// use the same fixed center/scale as array insertion.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.primitives: hasShape;

    static struct Input(T)
    {
        const(T)[] values;
        @property bool empty() const { return values.length == 0; }
        @property T front() const { return values[0]; }
        void popFront() { values = values[1 .. $]; }
    }
    static assert(!hasShape!(Input!double));

    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T[3] values = [-1, 1, 5];
        static foreach (N; 1 .. 5)
        {{
            alias M = MomentAccumulator!(T, N, Summation.naive);
            enum centeredSum = [2, 20, 56, 272][N - 1];
            enum scaledSum = [1, 5, 7, 17][N - 1];
            enum firstCentered = [-2, 4, -8, 16][N - 1];
            enum firstScaled = [-1, 1, -1, 1][N - 1];
            auto centered = M(values[0], T(1));
            auto scaled = M(values[0], T(1), T(2));
            assert(centered.count == 1 && centered.sumOfPower == firstCentered);
            assert(scaled.count == 1 && scaled.sumOfPower == firstScaled);
            foreach (x; values[1 .. $])
            {
                centered.put(x, T(1));
                scaled.put(x, T(1), T(2));
            }
            auto centeredRange = M(Input!T(values[]), T(1));
            auto scaledRange = M(Input!T(values[]), T(1), T(2));
            auto centeredBatch = M(values[], T(1));
            auto scaledBatch = M(values[], T(1), T(2));
            foreach (result; [centered, centeredRange, centeredBatch])
            {
                assert(result.count == 3 && result.sumOfPower == centeredSum);
                assert(result.moment.approxEqual(T(centeredSum) / 3));
            }
            foreach (result; [scaled, scaledRange, scaledBatch])
            {
                assert(result.count == 3 && result.sumOfPower == scaledSum);
                assert(result.moment.approxEqual(T(scaledSum) / 3));
            }
            centeredRange.put(Input!T(values[0 .. 0]), T(1));
            scaledRange.put(Input!T(values[0 .. 0]), T(1), T(2));
            assert(centeredRange.count == 3 && centeredRange.sumOfPower == centeredSum);
            assert(scaledRange.count == 3 && scaledRange.sumOfPower == scaledSum);
        }}
    }}
}

// Widen observations before taking powers in a shaped batch, as scalar
// insertion does. Powers of two give exact, independent expected results.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;

    alias M = MomentAccumulator!(double, 2, Summation.naive);
    static foreach (value; [0x1p80f, 0x1p-80f])
    {{
        const float[2] values = [value, value];
        enum double expected = value > 1 ? 0x1p160 : 0x1p-160;
        M scalar;
        foreach (x; values)
            scalar.put(x);
        auto batch = M(values[]);
        auto slice = M(values[].sliced);
        assert(scalar.moment == expected);
        assert(batch.count == 2 && batch.moment == expected);
        assert(slice.count == 2 && slice.moment == expected);
        assert(rawMoment!(double, 2, Summation.naive)(values[].sliced) == expected);

        const float[4] interleaved = [value, float.nan, value, float.nan];
        auto strided = interleaved[].sliced(2, 2).transposed[0];
        assert(M(strided).moment == expected);
    }}
}

// Merging preserves the correction terms needed by later cancellation.
// Exact powers of two make the expected remainder independent of rounding.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (mode; [Summation.kbn, Summation.kb2, Summation.precise])
    static foreach (N; [1, 3])
    {{
        alias M = MomentAccumulator!(T, N, mode);
        // The selected power is finite, but adding one to it cannot be
        // represented in a single T on this platform.
        enum T large = N == 1 ? T(0x1p100) : T(0x1p30);
        M source;
        source.put(large);
        source.put(T(1));
        M target;
        target.put(source);
        target.put(-large);
        assert(target.count == 3 && target.sumOfPower == 1);
        // Merging must not consume the source's corrections.
        source.put(-large);
        assert(source.count == 3 && source.sumOfPower == 1);
        M empty;
        target.put(empty);
        assert(target.count == 3 && target.sumOfPower == 1);
        target.put(target);
        assert(target.count == 6 && target.sumOfPower == 2);
        M fresh;
        fresh.put(empty);
        assert(fresh.count == 0 && fresh.sumOfPower == 0);
    }}
}

/// Raw moment
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    MomentAccumulator!(double, 2, Summation.naive) v;
    v.put(x);

    assert(v.moment.approxEqual(54.76562 / 12));

    v.put(4.0);
    assert(v.moment.approxEqual(70.76562 / 13));
}

// Raw Moment: test putting accumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    MomentAccumulator!(double, 2, Summation.naive) v;
    v.put(x);
    assert(v.moment.approxEqual(13.492188 / 6));

    MomentAccumulator!(double, 2, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.moment.approxEqual(54.76562 / 12));
}

// mir.complex test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    alias C = Complex!double;

    auto a = [C(1, 3), C(2), C(3)].sliced;
    auto x = a.center;

    MomentAccumulator!(C, 2, Summation.naive) v;
    v.put(x);
    assert(v.moment.approxEqual(C(-4, -6) / 3));
}

// Raw Moment: test std.complex
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import std.complex: Complex;
    import std.math.operations: isClose;

    auto a = [Complex!double(1.0, 3), Complex!double(2.0, 0), Complex!double(3.0, 0)].sliced;
    auto x = a.center;

    MomentAccumulator!(Complex!double, 2, Summation.naive) v;
    v.put(x);
    assert(v.moment.isClose(Complex!double(-4.0, -6.0) / 3));
}

/// Central moment
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    MomentAccumulator!(double, 2, Summation.naive) v;
    auto m = mean(x);
    v.put(x, m);
    assert(v.moment.approxEqual(54.76562 / 12));
}

// Central moment: dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    MomentAccumulator!(double, 2, Summation.naive) v;
    auto m = mean(x);
    v.put(x, m);
    assert(v.sumOfPower.approxEqual(54.76562));
}

// Central moment: withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    MomentAccumulator!(double, 2, Summation.naive) v;
    auto m = mean(x);
    v.put(x.asSlice.lightScope, m);
    assert(v.sumOfPower.approxEqual(54.76562));
}

// Central moment: Test N == 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    MomentAccumulator!(double, 1, Summation.naive) v;
    auto m = mean(x);
    v.put(x, m);
    assert(v.moment.approxEqual(0.0 / 12));
    assert(v.count == 12);
}

/// Standardized moment with scaled calculation
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance, VarianceAlgo, VarianceAccumulator;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto u = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    MomentAccumulator!(double, 3, Summation.naive) v;
    v.put(x, u.mean, u.variance(true).sqrt);
    assert(v.moment.approxEqual(12.000999 / 12));
    assert(v.count == 12);
}

// standardized moment: dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance, VarianceAlgo, VarianceAccumulator;
    import mir.math.common: approxEqual, sqrt;
    import mir.rc.array: RCArray;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto u = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    MomentAccumulator!(double, 3, Summation.naive) v;
    v.put(x, u.mean, u.variance(true).sqrt);
    assert(v.sumOfPower.approxEqual(12.000999));
}

// standardized moment: withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance, VarianceAlgo, VarianceAccumulator;
    import mir.math.common: approxEqual, sqrt;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto u = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    MomentAccumulator!(double, 3, Summation.naive) v;
    v.put(x.asSlice.lightScope, u.mean, u.variance(true).sqrt);
    assert(v.sumOfPower.approxEqual(12.000999));
}

// standardized moment: Test N == 2
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance, VarianceAlgo, VarianceAccumulator;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto u = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    MomentAccumulator!(double, 2, Summation.naive) v;
    v.put(x, u.mean, u.variance(true).sqrt);
    assert(v.moment.approxEqual(1.0));
    assert(v.count == 12);
}

// standardized moment: Test N == 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance, VarianceAlgo, VarianceAccumulator;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto u = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    MomentAccumulator!(double, 1, Summation.naive) v;
    v.put(x, u.mean, u.variance(true).sqrt);
    assert(v.moment.approxEqual(0.0));
    assert(v.count == 12);
}

/++
Calculates the n-th raw moment of the input.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

Params:
    F = controls type of output
    N = controls n-th raw moment
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The n-th raw moment of the input, must be floating point or complex type
+/
template rawMoment(F, size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.math.sum: ResolveSummationType;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!F rawMoment(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias G = typeof(return);
        MomentAccumulator!(G, N, ResolveSummationType!(summation, Range, G)) momentAccumulator;
        momentAccumulator.put(r.move);
        return momentAccumulator.moment;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!F rawMoment(scope const F[] ar...)
    {
        alias G = typeof(return);
        MomentAccumulator!(G, N, ResolveSummationType!(summation, const(G)[], G)) momentAccumulator;
        momentAccumulator.put(ar);
        return momentAccumulator.moment;
    }
}

/// ditto
template rawMoment(size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!Range rawMoment(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .rawMoment!(F, N, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!T rawMoment(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .rawMoment!(F, N, summation)(ar);
    }
}

/// ditto
template rawMoment(F, size_t N, string summation)
    if (N > 0)
{
    mixin("alias rawMoment = .rawMoment!(F, N, Summation." ~ summation ~ ");");
}

/// ditto
template rawMoment(size_t N, string summation)
    if (N > 0)
{
    mixin("alias rawMoment = .rawMoment!(N, Summation." ~ summation ~ ");");
}

/// Basic implementation
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(rawMoment!2([1.0, 2, 3]).approxEqual(14.0 / 3));
    assert(rawMoment!3([1.0, 2, 3]).approxEqual(36.0 / 3));

    assert(rawMoment!(float, 2)([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(55f / 6));
    static assert(is(typeof(rawMoment!(float, 2)([1, 2, 3])) == float));
}

// Moment orders retain their full size_t range in release builds too.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(rawMoment!(double, uint.max)(0.5) == 0);
    assert(rawMoment!(double, uint.max)(-1.0) == -1);
    assert(rawMoment!(double, int.max)(1.0) == 1);
    assert(rawMoment!(double, size_t.max)(-1.0) == -1);
}

/// Raw Moment of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    assert(x.rawMoment!2.approxEqual(54.76562 / 12));
}

/// Raw Moment of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.stat.transform: center;

    auto a = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto x = a.center;

    assert(x.rawMoment!2.approxEqual(54.76562 / 12));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;
    import mir.stat.transform: center;

    //Set sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto b = a * 10_000;
    auto x = b.center;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.rawMoment!2.approxEqual(2.0e208 / 4));

    assert(x.rawMoment!(2, "kbn").approxEqual(2.0e208 / 4));
    assert(x.rawMoment!(2, "kb2").approxEqual(2.0e208 / 4));
    assert(x.rawMoment!(2, "precise").approxEqual(2.0e208 / 4));
    assert(x.rawMoment!(double, 2, "precise").approxEqual(2.0e208 / 4));

    auto y = uint.max.repeat(3);
    auto z = y.rawMoment!(ulong, 2);
    assert(z.approxEqual(cast(double) (cast(ulong) uint.max) ^^ 2u));
    static assert(is(typeof(z) == double));
}

// mir.complex test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex: Complex;
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;

    alias C = Complex!double;

    auto x = [C(1, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    assert(x.rawMoment!2.approxEqual(C(-24, 80) / 4));
}

/++
rawMoment works for complex numbers and other user-defined types (that are either
implicitly convertible to floating point or if `isComplex` is true)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.complex: Complex;
    import std.math.operations: isClose;

    auto x = [Complex!double(1, 2), Complex!double(2, 3), Complex!double(3, 4), Complex!double(4, 5)].sliced;
    assert(x.rawMoment!2.isClose(Complex!double(-24, 80)/ 4));
}

// Full-width complex moment orders use logarithmically many products.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.complex: MirComplex = Complex;
    import std.complex: StdComplex = Complex;
    import std.meta: AliasSeq;

    static foreach (C; AliasSeq!(MirComplex!double, StdComplex!double))
    {
        assert(rawMoment!(C, size_t.max)(C(0, 1)) == C(0, -1));
        assert(rawMoment!(C, size_t.max - 1)(C(0, 1)) == C(-1, 0));
    }
}

/// Arbitrary raw moment
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(rawMoment!2(1.0, 2, 3).approxEqual(14.0 / 3));
    assert(rawMoment!(float, 2)(1, 2, 3).approxEqual(14f / 3));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].rawMoment!2.approxEqual(30.0 / 4));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.rawMoment!2.approxEqual(126.062500 / 12));
}

/++
Calculates the n-th central moment of the input.

For orders above one, inputs that cannot be converted to a slice must be
forward ranges. The mean and moment calculations use separate traversals of
the same observations.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

Params:
    F = controls type of output
    N = controls n-th central moment
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The n-th central moment of the input, must be floating point or complex type
+/
template centralMoment(F, size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.math.sum: ResolveSummationType;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!F centralMoment(Range)(Range r)
        if (isIterable!Range)
    {
        alias G = typeof(return);
        static if (N > 1) {
            MeanAccumulator!(G, ResolveSummationType!(summation, Range, G)) meanAccumulator;
            MomentAccumulator!(G, N, ResolveSummationType!(summation, Range, G)) momentAccumulator;
            static if (isSlice!Range) {
                meanAccumulator.put(r.lightScope);
                alias momentRange = r;
            } else static if (isConvertibleToSlice!Range) {
                import mir.ndslice.slice: toSlice;
                // Retain the converted owner while both passes borrow its storage.
                auto slice = r.toSlice;
                auto momentRange = slice.lightScope;
                meanAccumulator.put(momentRange);
            } else {
                import std.range.primitives: isForwardRange, save;
                static assert(isForwardRange!Range,
                    "centralMoment requires a forward range for multiple passes");
                // A shallow copy may share the original cursor.
                meanAccumulator.put(r.save);
                alias momentRange = r;
            }
            momentAccumulator.put(momentRange, meanAccumulator.mean);
            return momentAccumulator.moment;
        } else {
            return cast(G) 0.0;
        }
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!F centralMoment(scope const F[] ar...)
    {
        alias G = typeof(return);
        static if (N > 1) {
            MeanAccumulator!(G, ResolveSummationType!(summation, const(G)[], G)) meanAccumulator;
            MomentAccumulator!(G, N, ResolveSummationType!(summation, const(G)[], G)) momentAccumulator;
            meanAccumulator.put(ar);
            momentAccumulator.put(ar, meanAccumulator.mean);
            return momentAccumulator.moment;
        } else {
            return cast(G) 0.0;
        }
    }
}

/// ditto
template centralMoment(size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!Range centralMoment(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .centralMoment!(F, N, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!T centralMoment(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .centralMoment!(F, N, summation)(ar);
    }
}

/// ditto
template centralMoment(F, size_t N, string summation)
    if (N > 0)
{
    mixin("alias centralMoment = .centralMoment!(F, N, Summation." ~ summation ~ ");");
}

/// ditto
template centralMoment(size_t N, string summation)
    if (N > 0)
{
    mixin("alias centralMoment = .centralMoment!(N, Summation." ~ summation ~ ");");
}

/// Basic implementation
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(centralMoment!2([1.0, 2, 3]).approxEqual(2.0 / 3));
    assert(centralMoment!3([1.0, 2, 3]).approxEqual(0.0 / 3));

    assert(centralMoment!(float, 2)([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(17.5f / 6));
    static assert(is(typeof(centralMoment!(float, 2)([1, 2, 3])) == float));
}

/// Central Moment of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.centralMoment!2.approxEqual(54.76562 / 12));
}

/// Central Moment of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.centralMoment!2.approxEqual(54.76562 / 12));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;
    import mir.stat.transform: center;

    //Set sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto b = a * 10_000;
    auto x = b.center;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.centralMoment!2.approxEqual(2.0e208 / 4));

    assert(x.centralMoment!(2, "kbn").approxEqual(2.0e208 / 4));
    assert(x.centralMoment!(2, "kb2").approxEqual(2.0e208 / 4));
    assert(x.centralMoment!(2, "precise").approxEqual(2.0e208 / 4));
    assert(x.centralMoment!(double, 2, "precise").approxEqual(2.0e208 / 4));

    auto y = uint.max.repeat(3);
    auto z = y.centralMoment!(ulong, 2);
    assert(z.approxEqual(0.0));
    static assert(is(typeof(z) == double));
}

// mir.complex test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex: Complex;
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;

    alias C = Complex!double;

    auto x = [C(1, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    assert(x.centralMoment!2.approxEqual(C(0, 10) / 4));
}

/++
centralMoment works for complex numbers and other user-defined types (that are
either implicitly convertible to floating point or if `isComplex` is true)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.complex: Complex;
    import std.math.operations: isClose;

    auto x = [Complex!double(1, 2), Complex!double(2, 3), Complex!double(3, 4), Complex!double(4, 5)].sliced;
    assert(x.centralMoment!2.isClose(Complex!double(0, 10) / 4));
}

/// Arbitrary central moment
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(centralMoment!2(1.0, 2, 3).approxEqual(2.0 / 3));
    assert(centralMoment!(float, 2)(1, 2, 3).approxEqual(2f / 3));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].centralMoment!2.approxEqual(5.0 / 4));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.centralMoment!2.approxEqual(54.765625 / 12));
}

// test special casing
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.centralMoment!1.approxEqual(0.0 / 12));
}

///
enum StandardizedMomentAlgo
{
    /// Calculates n-th standardized moment as E(((x - u) / sigma) ^^ N)
    scaled,

    /// Calculates E((x - u) ^^ N) / (sigma ^^ N), using the population standard deviation sigma.
    centered
}

/++
Calculates the n-th standardized moment of the input.

For orders above two, inputs that cannot be converted to a slice must be
forward ranges. Calculating the center and scale must leave a traversal
available for the moment calculation.

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    F = controls type of output
    N = controls n-th standardized moment
    standardizedMomentAlgo = algorithm for standardizing moments (default: StandardizedMomentAlgo.scaled)
    varianceAlgo = algorithm for calculating variance (default: VarianceAlgo.twoPass)
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The n-th standardized moment of the input, must be floating point
+/
template standardizedMoment(F, size_t N,
                            StandardizedMomentAlgo standardizedMomentAlgo = StandardizedMomentAlgo.scaled,
                            VarianceAlgo varianceAlgo = VarianceAlgo.twoPass,
                            Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.math.sum: ResolveSummationType;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath stdevType!F standardizedMoment(Range)(Range r)
        if (isIterable!Range)
    {
        alias G = typeof(return);
        static if (N > 2) {
            static if (isSlice!Range) {
                auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(r.lightScope);
                alias momentRange = r;
            } else static if (isConvertibleToSlice!Range) {
                import mir.ndslice.slice: toSlice;
                // Retain the converted owner while both passes borrow its storage.
                auto slice = r.toSlice;
                auto momentRange = slice.lightScope;
                auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(momentRange);
            } else {
                import std.range.primitives: isForwardRange, save;
                static assert(isForwardRange!Range,
                    "standardizedMoment requires a forward range for multiple passes");
                // Preserve the observations for the subsequent moment pass.
                auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(r.save);
                alias momentRange = r;
            }
            MomentAccumulator!(G, N, ResolveSummationType!(summation, Range, G)) momentAccumulator;
            static if (standardizedMomentAlgo == StandardizedMomentAlgo.scaled) {
                import mir.math.common: sqrt;

                momentAccumulator.put(momentRange, varianceAccumulator.mean, varianceAccumulator.variance(true).sqrt);
                return momentAccumulator.moment;
            } else static if (standardizedMomentAlgo == StandardizedMomentAlgo.centered) {
                import mir.math.common: pow;

                momentAccumulator.put(momentRange, varianceAccumulator.mean);
                auto variance = varianceAccumulator.variance(true);
                auto result = momentAccumulator.moment / pow(variance, N / 2);
                static if (N % 2 != 0) {
                    import mir.math.common: sqrt;

                    // The integer power accounts for all but one factor of sigma.
                    return result / sqrt(variance);
                } else {
                    return result;
                }
            }
        } else static if (N == 2) {
            return cast(G) 1.0;
        } else static if (N == 1) {
            return cast(G) 0.0;
        }
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F standardizedMoment(scope const F[] ar...)
    {
        alias G = typeof(return);
        static if (N > 2) {
            auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, const(G)[], G))(ar);
            MomentAccumulator!(G, N, ResolveSummationType!(summation, const(G)[], G)) momentAccumulator;
            static if (standardizedMomentAlgo == StandardizedMomentAlgo.scaled) {
                import mir.math.common: sqrt;

                momentAccumulator.put(ar, varianceAccumulator.mean, varianceAccumulator.variance(true).sqrt);
                return momentAccumulator.moment;
            } else static if (standardizedMomentAlgo == StandardizedMomentAlgo.centered) {
                import mir.math.common: pow;

                momentAccumulator.put(ar, varianceAccumulator.mean);
                auto variance = varianceAccumulator.variance(true);
                auto result = momentAccumulator.moment / pow(variance, N / 2);
                static if (N % 2 != 0) {
                    import mir.math.common: sqrt;

                    // The integer power accounts for all but one factor of sigma.
                    return result / sqrt(variance);
                } else {
                    return result;
                }
            }
        } else static if (N == 2) {
            return cast(G) 1.0;
        } else static if (N == 1) {
            return cast(G) 0.0;
        }
    }
}

/// ditto
template standardizedMoment(size_t N,
                            StandardizedMomentAlgo standardizedMomentAlgo = StandardizedMomentAlgo.scaled,
                            VarianceAlgo varianceAlgo = VarianceAlgo.twoPass,
                            Summation summation = Summation.appropriate)
    if (N > 0)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath stdevType!Range standardizedMoment(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .standardizedMoment!(F, N, standardizedMomentAlgo, varianceAlgo, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T standardizedMoment(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .standardizedMoment!(F, N, standardizedMomentAlgo, varianceAlgo, summation)(ar);
    }
}

/// ditto
template standardizedMoment(F, size_t N, string standardizedMomentAlgo, string varianceAlgo = "twoPass", string summation = "appropriate")
    if (N > 0)
{
    mixin("alias standardizedMoment = .standardizedMoment!(F, N, StandardizedMomentAlgo." ~ standardizedMomentAlgo ~ ", VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template standardizedMoment(size_t N, string standardizedMomentAlgo, string varianceAlgo = "twoPass", string summation = "appropriate")
    if (N > 0)
{
    mixin("alias standardizedMoment = .standardizedMoment!(N, StandardizedMomentAlgo." ~ standardizedMomentAlgo ~ ", VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// Basic implementation
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(standardizedMoment!1([1.0, 2, 3]).approxEqual(0.0));
    assert(standardizedMoment!2([1.0, 2, 3]).approxEqual(1.0));
    assert(standardizedMoment!3([1.0, 2, 3]).approxEqual(0.0 / 3));
    assert(standardizedMoment!4([1.0, 2, 3]).approxEqual(4.5 / 3));

    assert(standardizedMoment!(float, 2)([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(6f / 6));
    static assert(is(typeof(standardizedMoment!(float, 2)([1, 2, 3])) == float));
}

// Asymmetric inputs exercise odd-order normalization in both overloads.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    // For [0, 0, 1], standardized deviations are [-1, -1, 2] / sqrt(2).
    // Thus the third, fourth, and fifth moments are 1/sqrt(2), 3/2,
    // and 5/(2*sqrt(2)), independently of a positive input scale.
    static foreach (F; AliasSeq!(float, double)) {
        foreach (scale; [F(0.25), F(1), F(8)]) {
            auto values = [F(0), F(0), scale];
            static foreach (N; AliasSeq!(3, 4, 5)) {{
                const expected = N == 3 ? F(1) / sqrt(F(2)) :
                    N == 4 ? F(1.5) : F(2.5) / sqrt(F(2));
                assert(standardizedMoment!(F, N, "centered")(values.sliced).approxEqual(expected));
                assert(standardizedMoment!(F, N, "centered")(F(0), F(0), scale).approxEqual(expected));
            }}
        }
    }
}

/// Standardized Moment of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.standardizedMoment!3.approxEqual(12.000999 / 12));
}

/// Standardized Moment of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.standardizedMoment!3.approxEqual(12.000999 / 12));
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 100_000_000_000;

    // The default algorithm is numerically stable in this case
    auto y = x.standardizedMoment!3;
    assert(y.approxEqual(12.000999 / 12));

    // The online algorithm is numerically unstable in this case
    auto z1 = x.standardizedMoment!(3, "scaled", "online");
    assert(!z1.approxEqual(12.000999 / 12));
    assert(!z1.approxEqual(y));

    // It is also numerically unstable when using StandardizedMomentAlgo.centered
    auto z2 = x.standardizedMoment!(3, "centered", "online");
    assert(!z2.approxEqual(12.000999 / 12));
    assert(!z2.approxEqual(y));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    //Set standardized moment algorithm, variance algorithm, sum algorithm, or output type

    auto a = [1.0, 1e98, 1, -1e98].sliced;
    auto x = a * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.standardizedMoment!3.approxEqual(0.0));

    assert(x.standardizedMoment!(3, "scaled", "online").approxEqual(0.0));
    assert(x.standardizedMoment!(3, "centered", "online").approxEqual(0.0));
    assert(x.standardizedMoment!(3, "scaled", "online", "kbn").approxEqual(0.0));
    assert(x.standardizedMoment!(3, "scaled", "online", "kb2").approxEqual(0.0));
    assert(x.standardizedMoment!(3, "scaled", "online", "precise").approxEqual(0.0));
    assert(x.standardizedMoment!(double, 3, "scaled", "online", "precise").approxEqual(0.0));

    auto y = [uint.max - 2, uint.max - 1, uint.max].sliced;
    auto z = y.standardizedMoment!(ulong, 3);
    assert(z == 0.0);
    static assert(is(typeof(z) == double));
}

/++
For integral slices, can pass output type as template parameter to ensure output
type is correct. By default, they get converted to double.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.standardizedMoment!3;
    assert(y.approxEqual(9.666455 / 12));
    static assert(is(typeof(y) == double));

    assert(x.standardizedMoment!(float, 3).approxEqual(9.666455f / 12));
}

/// Arbitrary standardized moment
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(standardizedMoment!3(1.0, 2, 3).approxEqual(0.0 / 3));
    assert(standardizedMoment!(float, 3)(1, 2, 3).approxEqual(0f / 3));
    assert(standardizedMoment!(float, 3, "centered")(1, 2, 3).approxEqual(0f / 3));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].standardizedMoment!3.approxEqual(0.0 / 4));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.standardizedMoment!3.approxEqual(12.000999 / 12));
}

// test special casing
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.standardizedMoment!1.approxEqual(0.0 / 12));
}

///
enum MomentAlgo
{
    /// nth raw moment, E(x ^^ n)
    raw,

    /// nth central moment, E((x - u) ^^ n)
    central,

    /// nth standardized moment, E(((x - u) / sigma) ^^ n)
    standardized
}

/++
Calculates the n-th moment of the input.

Params:
    F = controls type of output
    N = controls n-th standardized moment
    momentAlgo = type of moment to be calculated
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The n-th moment of the input, must be floating point or complex type
+/
template moment(F, size_t N,
                MomentAlgo momentAlgo,
                Summation summation = Summation.appropriate)
{
    import mir.math.sum: ResolveSummationType;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!F moment(Range)(Range r)
        if (isIterable!Range && momentAlgo != MomentAlgo.standardized)
    {
        import core.lifetime: move;

        alias G = typeof(return);
        static if (momentAlgo == MomentAlgo.raw) {
            return .rawMoment!(G, N, ResolveSummationType!(summation, Range, G))(r.move);
        } else static if (momentAlgo == MomentAlgo.central) {
            return .centralMoment!(G, N, ResolveSummationType!(summation, Range, G))(r.move);
        }
    }

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath stdevType!F moment(Range)(Range r)
        if (isIterable!Range && momentAlgo == MomentAlgo.standardized)
    {
        import core.lifetime: move;

        alias G = typeof(return);
        return .standardizedMoment!(G, N, StandardizedMomentAlgo.scaled, VarianceAlgo.twoPass, ResolveSummationType!(summation, Range, G))(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!F moment()(scope const F[] ar...)
        if (momentAlgo != MomentAlgo.standardized)
    {
        alias G = typeof(return);
        static if (momentAlgo == MomentAlgo.raw) {
            return .rawMoment!(G, N, ResolveSummationType!(summation, const(G)[], G))(ar);
        } else static if (momentAlgo == MomentAlgo.central) {
            return .centralMoment!(G, N, ResolveSummationType!(summation, const(G)[], G))(ar);
        }
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F moment()(scope const F[] ar...)
        if (momentAlgo == MomentAlgo.standardized)
    {
        alias G = typeof(return);
        return .standardizedMoment!(G, N, StandardizedMomentAlgo.scaled, VarianceAlgo.twoPass, ResolveSummationType!(summation, const(G)[], G))(ar);
    }
}

/// ditto
template moment(size_t N,
                MomentAlgo momentAlgo,
                Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath stdevType!Range moment(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .moment!(F, N, momentAlgo, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T moment(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .moment!(F, N, momentAlgo, summation)(ar);
    }
}

/// ditto
template moment(F, size_t N, string momentAlgo, string summation = "appropriate")
{
    mixin("alias moment = .moment!(F, N, MomentAlgo." ~ momentAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template moment(size_t N, string momentAlgo, string summation = "appropriate")
{
    mixin("alias moment = .moment!(N, MomentAlgo." ~ momentAlgo ~ ", Summation." ~ summation ~ ");");
}

/// Basic implementation
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(moment!(1, "raw")([1.0, 2, 3]).approxEqual(6.0 / 3));
    assert(moment!(2, "raw")([1.0, 2, 3]).approxEqual(14.0 / 3));
    assert(moment!(3, "raw")([1.0, 2, 3]).approxEqual(36.0 / 3));
    assert(moment!(4, "raw")([1.0, 2, 3]).approxEqual(98.0 / 3));

    assert(moment!(1, "central")([1.0, 2, 3]).approxEqual(0.0 / 3));
    assert(moment!(2, "central")([1.0, 2, 3]).approxEqual(2.0 / 3));
    assert(moment!(3, "central")([1.0, 2, 3]).approxEqual(0.0 / 3));
    assert(moment!(4, "central")([1.0, 2, 3]).approxEqual(2.0 / 3));

    assert(moment!(1, "standardized")([1.0, 2, 3]).approxEqual(0.0));
    assert(moment!(2, "standardized")([1.0, 2, 3]).approxEqual(1.0));
    assert(moment!(3, "standardized")([1.0, 2, 3]).approxEqual(0.0 / 3));
    assert(moment!(4, "standardized")([1.0, 2, 3]).approxEqual(4.5 / 3));

    assert(moment!(float, 2, "standardized")([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(6f / 6));
    static assert(is(typeof(moment!(float, 2, "standardized")([1, 2, 3])) == float));
}

/// Standardized Moment of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.moment!(3, "standardized").approxEqual(12.000999 / 12));
}

/// Standardized Moment of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.moment!(3, "standardized").approxEqual(12.000999 / 12));
}

/++
For integral slices, can pass output type as template parameter to ensure output
type is correct. By default, they get converted to double.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.moment!(3, "standardized");
    assert(y.approxEqual(9.666455 / 12));
    static assert(is(typeof(y) == double));

    assert(x.moment!(float, 3, "standardized").approxEqual(9.666455f / 12));
}

/// Arbitrary standardized moment
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(moment!(3, "standardized")(1.0, 2, 3).approxEqual(0.0 / 3));
    assert(moment!(float, 3, "standardized")(1, 2, 3).approxEqual(0f / 3));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].moment!(3, "standardized").approxEqual(0.0 / 4));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.moment!(3, "standardized").approxEqual(12.000999 / 12));
}

// Public multi-pass functions save forward ranges before consuming their cursor.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;

    static struct Input(T, bool replayable)
    {
        const(T)[] data;
        size_t[] cursor;
        size_t ownPosition;
        size_t position() const { return cursor.length ? cursor[0] : ownPosition; }
        bool empty() const { return position == data.length; }
        T front() const { return data[position]; }
        void popFront() { if (cursor.length) ++cursor[0]; else ++ownPosition; }
        static if (replayable)
            auto save() { return typeof(this)(data, null, position); }
    }

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T[3] values = [0, 1, 5];
        // Explicit range instantiation also supports array-to-slice conversion.
        alias central = centralMoment!(T, 3, Summation.naive);
        alias standard = standardizedMoment!(T, 3);
        assert(central!(const(T)[])(values[]) == 6);
        assert(standard!(const(T)[])(values[]).approxEqual(standard(values[])));
        size_t[1] cursor;
        auto input = Input!(T, true)(values[], cursor[]);
        assert(centralMoment!(T, 3, Summation.naive)(input).approxEqual(T(6)));
        assert(cursor[0] == values.length);
        cursor[0] = 0;
        assert(moment!(T, 3, MomentAlgo.central, Summation.naive)(input).approxEqual(T(6)));
        static foreach (algo; [VarianceAlgo.twoPass, VarianceAlgo.hybrid, VarianceAlgo.online, VarianceAlgo.naive])
        static foreach (method; [StandardizedMomentAlgo.scaled, StandardizedMomentAlgo.centered])
        {{
            cursor[0] = 0;
            auto expected = standardizedMoment!(T, 3, method, algo, Summation.naive)(values[]);
            auto result = standardizedMoment!(T, 3, method, algo, Summation.naive)(input);
            assert(result.approxEqual(expected));
            assert(cursor[0] == values.length);
        }}
        cursor[0] = 0;
        assert(moment!(T, 3, MomentAlgo.standardized, Summation.naive)(input)
            .approxEqual(standardizedMoment!(T, 3)(values[])));
        cursor[0] = 0;
        auto single = Input!(T, false)(values[], cursor[]);
        static assert(!__traits(compiles, centralMoment!(T, 2)(single)));
        static assert(!__traits(compiles, standardizedMoment!(T, 3)(single)));
        // Constant low-order results and raw moments do not need another pass.
        assert(centralMoment!(T, 1)(single) == 0);
        assert(standardizedMoment!(T, 2)(single) == 1);
        assert(cursor[0] == 0);
        assert(rawMoment!(T, 1, Summation.naive)(single) == 2);
        assert(cursor[0] == values.length);
    }}
}

// Owning arrays are converted once and borrowed for both moment passes.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        auto owner = RCArray!T(3);
        owner[0] = 0;
        owner[1] = 1;
        owner[2] = 5;
        assert(centralMoment!(T, 3, Summation.naive)(owner) == 6);
        assert(moment!(T, 3, MomentAlgo.central, Summation.naive)(owner) == 6);
        static foreach (method; [StandardizedMomentAlgo.scaled, StandardizedMomentAlgo.centered])
        {{
            auto expected = standardizedMoment!(T, 3, method)(owner.asSlice.lightScope);
            assert(standardizedMoment!(T, 3, method)(owner).approxEqual(expected));
        }}
        assert(moment!(T, 3, MomentAlgo.standardized)(owner)
            .approxEqual(standardizedMoment!(T, 3)(owner.asSlice.lightScope)));
        // The caller retains ownership and the observations are unchanged.
        assert(owner.length == 3);
        assert(owner[0] == 0 && owner[1] == 1 && owner[2] == 5);
    }}
}
