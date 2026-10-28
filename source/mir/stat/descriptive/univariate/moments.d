/++
Raw, central, and standardized moments, including weighted raw, central, and standardized moments.

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
        $(REF weightedRawMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF weightedCentralMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
        $(REF weightedStandardizedMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
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

// Keep the 18 cases in separate functions to reduce Windows DMD's
// optimized compilation cost. Production calls inside each case may inline.
version(mir_stat_test)
pragma(inline, false)
private void checkCompensatedMomentMerge(T, Summation mode, int N)() pure nothrow @nogc
{
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
        checkCompensatedMomentMerge!(T, mode, N)();
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

// Remove this guard once the supported Windows DMD versions fix #24118.
// Check every entry and traversal boundary: forwarding makes a by-value
// parameter look like an lvalue, but cannot repair corruption on entry.
package(mir.stat.descriptive.univariate) mixin template WeightedMomentCheckArguments(alias values, alias weights)
{
    version (DigitalMars) version (Win64)
    {
        import std.traits: Unqual;
        import std.meta: AliasSeq;
        static foreach (input; AliasSeq!(values, weights))
            static if (!__traits(isRef, input))
                static assert(!is(Unqual!(typeof(input)) == float[1]) &&
                    !is(Unqual!(typeof(input)) == double[1]),
                    "Windows x64 DMD issue #24118: store one-element float/double arrays " ~
                    "in named variables before calling weighted moments");
    }
}

// Borrow inputs through the public wrappers and traversal. Besides avoiding
// static-array copies, this protects named arrays from DMD #24118 (Windows
// x64, introduced in 2.094.0). Public entry points separately reject affected
// temporaries, which auto ref would pass by value.
// https://github.com/dlang/dmd/issues/24118
// Traverse by logical index, preserving independent strides.
private void weightedMomentEach(Sink, X, W)(ref Sink sink, scope auto ref X x, scope auto ref W w)
{
    mixin WeightedMomentCheckArguments!(x, w);
    import mir.ndslice.slice: toSlice;
    import std.traits: isStaticArray;
    import std.range.primitives: isInputRange, isForwardRange;
    static if (isConvertibleToSlice!X && !isSlice!X && !isStaticArray!X)
        weightedMomentEach(sink, x.toSlice, w);
    else static if (isConvertibleToSlice!W && !isSlice!W && !isStaticArray!W)
        weightedMomentEach(sink, x, w.toSlice);
    else static if ((isStaticArray!X || isSlice!X) && (isStaticArray!W || isSlice!W))
    {
        static if (isSlice!X && isSlice!W)
        {
            static assert(X.N == W.N, "weighted moments: ranks must match");
            assert(x.shape == w.shape, "weighted moments: shapes must match");
        }
        else
            assert(x.length == w.length, "weighted moments: lengths must match");
        foreach (i; 0 .. x.length)
            weightedMomentEach(sink, x[i], w[i]);
    }
    else static if (isStaticArray!X && isInputRange!W)
    {
        auto remaining = w;
        static if (isForwardRange!W) remaining = w.save;
        foreach (i; 0 .. x.length)
        {
            assert(!remaining.empty, "weighted moments: lengths must match");
            weightedMomentEach(sink, x[i], remaining.front);
            remaining.popFront;
        }
        assert(remaining.empty, "weighted moments: lengths must match");
    }
    else static if (isInputRange!X && isStaticArray!W)
    {
        auto remaining = x;
        static if (isForwardRange!X) remaining = x.save;
        foreach (i; 0 .. w.length)
        {
            assert(!remaining.empty, "weighted moments: lengths must match");
            weightedMomentEach(sink, remaining.front, w[i]);
            remaining.popFront;
        }
        assert(remaining.empty, "weighted moments: lengths must match");
    }
    else static if (isInputRange!X && isInputRange!W)
    {
        auto xr = x;
        auto wr = w;
        static if (isForwardRange!X) xr = x.save;
        static if (isForwardRange!W) wr = w.save;
        while (!xr.empty && !wr.empty)
        {
            weightedMomentEach(sink, xr.front, wr.front);
            xr.popFront;
            wr.popFront;
        }
        assert(xr.empty && wr.empty, "weighted moments: lengths must match");
    }
    else
        sink.put(x, w);
}

private template weightedMomentReplayable(R)
{
    import mir.ndslice.slice: toSlice;
    import std.traits: isStaticArray;
    import std.range.primitives: isInputRange, isForwardRange, ElementType;
    static if (isStaticArray!R)
        enum weightedMomentReplayable = weightedMomentReplayable!(typeof(R.init[0]));
    else static if (isConvertibleToSlice!R && !isSlice!R)
        enum weightedMomentReplayable = weightedMomentReplayable!(typeof(R.init.toSlice));
    else static if (isSlice!R)
        enum weightedMomentReplayable = weightedMomentReplayable!(typeof(R.init[0]));
    else static if (isInputRange!R)
        enum weightedMomentReplayable = isForwardRange!R && weightedMomentReplayable!(ElementType!R);
    else
        enum weightedMomentReplayable = true;
}

// Return zero before converting an excluded observation. Positive weights must
// survive conversion, just as in weighted variance.
private T weightedMomentWeight(T, W)(W w)
{
    import std.math: isFinite;
    const raw = cast(real) w;
    assert(isFinite(raw) && raw >= 0, "weighted moments: weights must be finite and nonnegative");
    if (raw == 0) return 0;
    const weight = cast(T) w;
    assert(isFinite(weight) && weight > 0, "weighted moments: weight must fit result type");
    return weight;
}

// Restore value / valueDivisor * (scale * residualScale * 2^^shift)^^order
// without forming an underflowing quotient or an overflowing power.
// Saturating exponents are far beyond any supported floating-point range;
// powers of a fixed positive scale all move in the same direction.
private T restoreWeightedMoment(T)(T value, T scale, size_t order, int shift = 0, T residualScale = 1, T valueDivisor = 1)
{
    import std.math: frexp, ldexp;
    import std.algorithm: clamp;
    if (value == 0 || scale == 0) return 0;
    int e;
    T result = frexp(value, e);
    long exponent = e;
    if (valueDivisor != 1)
    {
        const divisorFraction = frexp(valueDivisor, e);
        exponent -= e;
        result = frexp(result / divisorFraction, e);
        exponent += e;
    }
    T factor = frexp(scale, e);
    long factorExponent = e + shift;
    const residualFraction = frexp(residualScale, e);
    factorExponent += e;
    factor = frexp(factor * residualFraction, e);
    factorExponent += e;
    enum long limit = int.max / 4;
    while (order)
    {
        if (order & 1)
        {
            result = frexp(result * factor, e);
            exponent = clamp(exponent + factorExponent + e, -limit, limit);
        }
        order >>= 1;
        if (order)
        {
            factor = frexp(factor * factor, e);
            factorExponent = clamp(2 * factorExponent + e, -limit, limit);
        }
    }
    return ldexp(result, cast(int) exponent);
}

private struct WeightedRawMomentState(T, size_t N, Summation summation)
{
    import mir.math.internal.powi: powi;
    import std.math: abs, isFinite;
    Summator!(T, summation) mass, powers;
    T weightScale = 0, valueScale = 0;

    void put(X, W)(X x, W w)
    {
        const weight = weightedMomentWeight!T(w);
        if (weight == 0) return;
        const value = cast(T) x;
        assert(isFinite(value), "weighted moments: positive-weight observations must be finite");
        if (weight > weightScale)
        {
            const ratio = weightScale / weight;
            mass = cast(T) mass.sum * ratio;
            powers = cast(T) powers.sum * ratio;
            weightScale = weight;
        }
        const normalizedWeight = weight / weightScale;
        mass.put(normalizedWeight);
        if (normalizedWeight == 0) return;
        const magnitude = abs(value);
        if (magnitude > valueScale)
        {
            powers = cast(T) powers.sum * powi(valueScale / magnitude, N);
            valueScale = magnitude;
        }
        if (valueScale != 0)
            powers.put(normalizedWeight * powi(value / valueScale, N));
    }

    T result()
    {
        assert(weightScale > 0, "weighted moments: at least one positive weight is required");
        return restoreWeightedMoment(cast(T) powers.sum / cast(T) mass.sum, valueScale, N);
    }
}

private struct WeightedMomentBounds(T)
{
    T origin = 0, scale = 0, weightScale = 0, minimum = 0, maximum = 0;
    bool halfDifferences;

    T difference(T value) const
    {
        import std.math: isFinite;
        const delta = value - origin;
        if (!halfDifferences) return delta;
        return isFinite(delta) ? delta / 2 : value / 2 - origin / 2;
    }

    void put(X, W)(X x, W w)
    {
        import std.math: abs, isFinite;
        const weight = weightedMomentWeight!T(w);
        if (weight == 0) return;
        const value = cast(T) x;
        assert(isFinite(value), "weighted moments: positive-weight observations must be finite");
        if (weightScale == 0) minimum = maximum = value;
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        // A distant negligible-weight first row must not anchor the mean.
        if (weight > weightScale)
        {
            weightScale = weight;
            origin = value;
        }
    }

    void prepare()
    {
        import std.math: abs, isFinite;
        halfDifferences = !isFinite(minimum - origin) || !isFinite(maximum - origin);
        const left = abs(difference(minimum)), right = abs(difference(maximum));
        scale = left > right ? left : right;
    }
}

private struct WeightedCenteredMomentPass(T, size_t N, Summation summation, bool standardized = false)
{
    import mir.math.internal.powi: powi;
    WeightedMomentBounds!T bounds;
    Summator!(T, summation) mass, powers;
    static if (standardized) Summator!(T, summation) squares;
    T center = 0, divisor = 1;

    void put(X, W)(X x, W w)
    {
        if (w == 0) return;
        const weight = cast(T) w / bounds.weightScale;
        if (weight == 0) return;
        const delta = bounds.difference(cast(T) x) / bounds.scale;
        mass.put(weight);
        const residual = (delta - center) / divisor;
        powers.put(weight * powi(residual, N));
        static if (standardized) squares.put(weight * residual * residual);
    }
}

// Shared scaled calculation: keep variance and higher powers in the same
// coordinates so standardized moments need not form dimensional powers.
private T weightedCenteredMoment(T, size_t N, Summation summation, bool standardized, X, W)
    (scope ref X values, scope ref W weights)
{
    static if (N > (standardized ? 2 : 1))
        static assert(weightedMomentReplayable!X && weightedMomentReplayable!W,
            "weighted centered moments require replayable inputs for this order");
    WeightedMomentBounds!T bounds;
    weightedMomentEach(bounds, values, weights);
    assert(bounds.weightScale > 0, "weighted moments: at least one positive weight is required");
    static if (N == 1) return T(0);
    else static if (standardized && N == 2) return T(1);
    else
    {
        bounds.prepare();
        if (bounds.scale == 0) return standardized ? T.nan : T(0);
        enum mode = ResolveSummationType!(summation, T[], T);
        WeightedCenteredMomentPass!(T, 1, mode) meanPass;
        meanPass.bounds = bounds;
        weightedMomentEach(meanPass, values, weights);
        WeightedCenteredMomentPass!(T, N, mode, standardized) momentPass;
        momentPass.bounds = bounds;
        momentPass.center = cast(T) meanPass.powers.sum / cast(T) meanPass.mass.sum;
        import std.math: abs;
        import std.algorithm: max;
        // Normalize by the actual largest centered residual. A fixed
        // extra factor of two would erase high moments of unit spreads.
        momentPass.divisor = max(abs(bounds.difference(bounds.minimum) / bounds.scale - momentPass.center),
            abs(bounds.difference(bounds.maximum) / bounds.scale - momentPass.center));
        weightedMomentEach(momentPass, values, weights);
        static if (standardized)
        {
            import std.math: frexp, ldexp, sqrt;
            const mass = cast(T) momentPass.mass.sum;
            const squares = cast(T) momentPass.squares.sum;
            if (squares == 0) return T.nan;
            // The physical scale cancels. Split mass / squares into a
            // fraction and exponent before taking its square root: forming
            // the variance or its reciprocal can underflow or overflow.
            int massExponent, squaresExponent;
            const massFraction = frexp(mass, massExponent);
            const squaresFraction = frexp(squares, squaresExponent);
            const exponent = massExponent - squaresExponent;
            const factor = sqrt(ldexp(massFraction / squaresFraction, exponent % 2));
            // Likewise, keep powers / mass split until the final ldexp.
            return restoreWeightedMoment(cast(T) momentPass.powers.sum,
                factor, N, exponent / 2, T(1), mass);
        }
        else
        {
            const moment = cast(T) momentPass.powers.sum / cast(T) momentPass.mass.sum;
            return restoreWeightedMoment(moment, bounds.scale, N, bounds.halfDifferences ? 1 : 0, momentPass.divisor);
        }
    }
}

/++
Weighted average of the N-th powers of real-valued observations.

Use a raw moment when zero is the meaningful reference point: order one is the
weighted mean, and order two is the weighted mean square. Formula:
sum(w * x^^N) / sum(w).

Counts, probabilities, and relative emphasis all use the same descriptive
formula, so there is no weight-meaning or sample-correction selector. Counts
match rawMoment on the expanded observations, up to rounding. Multiplying all
weights by the same positive constant leaves the mathematical result unchanged.
Weights need not sum to one. Unlike wmean, signed weights are not accepted.

Weights must be finite and nonnegative, with at least one positive weight.
Zero weights exclude observations, including NaN and infinity. Positive weights
must stay positive and finite after conversion to the result type, and their
observations must be finite in that type. Preconditions use assertions.

Inputs may be arrays, slices with independent strides, or equally shaped nested
ranges. Raw moments traverse once, consuming single-pass cursors and saving
forward ranges. Inputs are not mutated and no observation buffer is allocated.
On Windows x64 DMD, store one-element float or double static arrays in named
variables before passing them as values or weights. Temporary arrays of these
types are rejected to avoid incorrect results from
$(LINK2 https://github.com/dlang/dmd/issues/24118, DMD issue 24118).

Weights and values are scaled internally to avoid intermediate overflow.
Rescaling rounds accumulated sums; tiny weight ratios and normalized powers can
underflow, and cancellation can limit odd-moment accuracy. Results outside the
result type's range may overflow or underflow. Choose a sufficiently wide result
type when converting large integers; subtracting an offset changes raw moments.

Params:
    F = floating result/working type, or a type promoted to floating point
    N = positive moment order
    summation = summation algorithm; appropriate selects pairwise summation
+/
template weightedRawMoment(F, size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    /// values and weights must have matching shapes or lengths.
    stdevType!F weightedRawMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        alias T = stdevType!F;
        WeightedRawMomentState!(T, N, ResolveSummationType!(summation, T[], T)) state;
        weightedMomentEach(state, values, weights);
        return state.result;
    }
}

/// ditto
template weightedRawMoment(size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.primitives: DeepElementType;
    /// ditto
    stdevType!(DeepElementType!X) weightedRawMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        return .weightedRawMoment!(stdevType!(DeepElementType!X), N, summation)(values, weights);
    }
}

/// Counts and probabilities describe the same weighted distribution.
@safe pure nothrow @nogc
unittest
{
    double[2] values = [0, 10];
    int[2] counts = [1, 3];
    double[2] probabilities = [0.25, 0.75];
    assert(weightedRawMoment!1(values, counts) == 7.5);
    assert(weightedRawMoment!2(values, counts) == 75);
    assert(weightedRawMoment!2(values, probabilities) == 75);
}

/++
Weighted average of the N-th powers of deviations from the weighted mean.
Formula: sum(w * (x - weightedMean)^^N) / sum(w).

Use central moments to describe spread and asymmetry around the mean. The first
central moment is zero. The second is population weighted variance. The third
is negative when the cubed deviations contribute more on the lower side of the
mean; its units are the observation units cubed, so it is not standardized skewness.

These are descriptive moments, without a sample correction. Count weights agree
with centralMoment on expanded observations. Probabilities and relative weights
use the same formula, and common weight scaling leaves the result unchanged.
For a sample-corrected second moment, use weightedVariance with frequencySample
or relativeSample instead; those estimate a different quantity.

Input, weight, result-type, and rounding contracts follow weightedRawMoment.
Orders above one require replayable inputs at every nesting level: arrays,
slices, or forward ranges. Three traversals find an anchor among the largest-weight observations and scales, estimate
the mean displacement, and accumulate centered powers. Anchored differences
preserve small spreads around a large common offset. Order one still traverses
and validates both inputs before returning zero. Constant observations give zero.
Subtracting a common integer offset before conversion can preserve small
differences in central moments without changing their mathematical value.

Params:
    F = floating result/working type, or a type promoted to floating point
    N = positive moment order
    summation = summation algorithm; appropriate selects pairwise summation
+/
template weightedCentralMoment(F, size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    /// values and weights must have matching shapes or lengths.
    stdevType!F weightedCentralMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        return weightedCenteredMoment!(stdevType!F, N, summation, false)(values, weights);
    }
}

/// ditto
template weightedCentralMoment(size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.primitives: DeepElementType;
    /// ditto
    stdevType!(DeepElementType!X) weightedCentralMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        return .weightedCentralMoment!(stdevType!(DeepElementType!X), N, summation)(values, weights);
    }
}

/// Describing spread and estimating sample variance are different questions.
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.descriptive.univariate.dispersions: weightedVariance, WeightedMomentNormalization;
    alias Policy = WeightedMomentNormalization;
    double[2] values = [0, 10];
    int[2] counts = [1, 3]; // Represents [0, 10, 10, 10].
    assert(weightedCentralMoment!1(values, counts) == 0);
    assert(weightedCentralMoment!2(values, counts).approxEqual(18.75));
    assert(weightedCentralMoment!3(values, counts).approxEqual(-93.75));
    assert(weightedCentralMoment!2(values, counts).approxEqual(weightedVariance(values, counts)));
    assert(weightedVariance!(Policy.frequencySample)(values, counts).approxEqual(25));
    assert(weightedVariance!(Policy.relativeSample)(values, counts).approxEqual(50));
}

// Reference identities across output types, moment orders, and summation modes.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.descriptive.univariate.dispersions: weightedVariance;
    import mir.stat.descriptive.univariate.means: wmean;
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        T[4] values = [-2, 0, 1, 5];
        int[4] counts = [1, 2, 3, 1];
        T[7] expanded = [-2, 0, 0, 1, 1, 1, 5];
        T[4] ones = [1, 1, 1, 1];
        static foreach (N; [1, 2, 3, 4, 5, 8])
        {
            assert(weightedRawMoment!(N, mode)(values, counts)
                .approxEqual(rawMoment!(N, mode)(expanded[].sliced), T(128) * T.epsilon, T(0)));
            assert(weightedCentralMoment!(N, mode)(values, counts)
                .approxEqual(centralMoment!(N, mode)(expanded[].sliced), T(128) * T.epsilon, T(0)));
            assert(weightedRawMoment!(N, mode)(values, ones)
                .approxEqual(rawMoment!(N, mode)(values[].sliced), T(128) * T.epsilon, T(0)));
            assert(weightedCentralMoment!(N, mode)(values, ones)
                .approxEqual(centralMoment!(N, mode)(values[].sliced), T(128) * T.epsilon, T(0)));
        }
        assert(weightedRawMoment!(1, mode)(values, counts).approxEqual(wmean(values[], counts[]), T(32) * T.epsilon));
        assert(weightedCentralMoment!(2, mode)(values, counts).approxEqual(weightedVariance(values, counts), T(64) * T.epsilon));
        static assert(is(typeof(weightedRawMoment!2(values, counts)) == T));
        static assert(is(typeof(weightedCentralMoment!2(values, counts)) == T));
    }}
    int[2] integers = [50_000, 100_000];
    int[2] weights = [1, 1];
    assert(weightedRawMoment!2(integers, weights) == 6.25e9);
    static assert(is(typeof(weightedRawMoment!2(integers, weights)) == double));
    static assert(is(typeof(weightedCentralMoment!(float, 2)(integers, weights)) == float));
}

// Scaling, exclusion, representable results despite unrepresentable powers,
// and centered spreads too small to survive rounding the mean to one value.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: isInfinity;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[3] values = [0, 1, 5];
        foreach (scale; [T.min_normal, T(1), T.max / 4])
        {
            T[3] weights = [scale, 2 * scale, 3 * scale];
            assert(weightedRawMoment!2(values, weights).approxEqual(T(77) / 6, T(32) * T.epsilon, T(0)));
            assert(weightedCentralMoment!3(values, weights).approxEqual(-T(41) / 54, T(128) * T.epsilon, T(0)));
        }
        T[2] adjacent = [1, 1 + T.epsilon], equal = [1, 1];
        assert(weightedCentralMoment!2(adjacent, equal) == (T.epsilon / 2) * (T.epsilon / 2));
        T[3] masked = [T.nan, 7, T.infinity], mask = [0, 1, 0];
        assert(weightedRawMoment!2(masked, mask) == 49);
        assert(weightedCentralMoment!3(masked, mask) == 0);
        T[2] opposite = [-T.max, T.max];
        assert(weightedRawMoment!1(opposite, equal) == 0);
        assert(weightedCentralMoment!3(opposite, equal) == 0);
        assert(weightedCentralMoment!2(opposite, equal).isInfinity);
        T[2] constant = [T.max, T.max];
        assert(weightedRawMoment!1(constant, equal) == T.max);
        assert(weightedCentralMoment!4(constant, equal) == 0);
        T[2] zero = [0, 0];
        assert(weightedRawMoment!4(zero, equal) == 0);
    }}
    // x^^2 overflows, but the weighted mean square fits.
    double[2] large = [1e160, 0], smallWeight = [1e-20, 1];
    assert(weightedRawMoment!2(large, smallWeight).approxEqual(1e300, 1e-14, 0));
    double[4] shifted = [1e12 + 1, 1e12 + 2, 1e12 + 4, 1e12 + 8];
    double[4] small = [1, 2, 4, 8], weights = [1, 3, 2, 1];
    static foreach (N; [2, 3, 4])
        assert(weightedCentralMoment!N(shifted, weights).approxEqual(weightedCentralMoment!N(small, weights), 1e-14, 0));
    double[3] outlier = [-1e20, 1, 2], outlierWeights = [1e-80, 1, 1];
    assert(weightedCentralMoment!2(outlier, outlierWeights).approxEqual(0.25, 1e-14, 0));
    double[2] tiny = [1e-160, -1e-160], equal = [1, 1];
    assert(weightedCentralMoment!2(tiny, equal).approxEqual(cast(double) 1e-320L, 1e-3, 0));
}

// Paired logical traversal accepts independent strides, nested ranges, and
// const inputs, without requiring either input to own contiguous storage.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import std.algorithm: filter;
    const double[6] values = [1, 2, 3, 4, 5, 6];
    double[6] storage = [1, 4, 2, 5, 3, 6], weights = [1, 2, 3, 4, 5, 6];
    static foreach (fun; AliasSeq!(weightedRawMoment, weightedCentralMoment, weightedStandardizedMoment))
    {{
        const expected = fun!3(values[], weights[]);
        assert(fun!3(values[].sliced(2, 3), storage[].sliced(3, 2).transposed).approxEqual(expected));
        assert(fun!3(values[].filter!(x => true), weights[].filter!(x => true)).approxEqual(expected));
        assert(fun!3(values, weights[].filter!(x => true)).approxEqual(expected));
        assert(fun!3(values[].filter!(x => true), weights).approxEqual(expected));
        const(double)[][2] nested = [values[0 .. 2], values[2 .. $]];
        double[][2] nestedWeights = [weights[0 .. 2], weights[2 .. $]];
        assert(fun!3(nested, nestedWeights).approxEqual(expected));
    }}
}

version(mir_stat_test)
pure unittest
{
    import core.exception: AssertError;
    import std.meta: AliasSeq;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;
    double[2] values = [0, 10], weights = [1, 3];
    static foreach (fun; AliasSeq!(weightedRawMoment, weightedCentralMoment, weightedStandardizedMoment))
    static foreach (N; [1, 2, 3])
    {{
        foreach (bad; [-1.0, double.nan, double.infinity])
        {
            double[2] invalid = [1, bad];
            assertThrown!AssertError(fun!N(values, invalid));
        }
        double[2] zero = [0, 0];
        assertThrown!AssertError(fun!N(values, zero));
        assertThrown!AssertError(fun!N(values[0 .. 0], weights[0 .. 0]));
        assertThrown!AssertError(fun!N(values, weights[0 .. 1]));
        assertThrown!AssertError(fun!N(values[].sliced(1, 2), weights[].sliced(2, 1)));
        double[2] nonfinite = [0, double.infinity];
        assertThrown!AssertError(fun!N(nonfinite, weights));
        double[2] underflow = [1, double.min_normal];
        assertThrown!AssertError(fun!(float, N)(values, underflow));
        double[2] overflow = [1, double.max];
        assertThrown!AssertError(fun!(float, N)(values, overflow));
    }}
}

// A forward range's save must detach shared cursor state. A plain copy does
// not provide a second traversal, and a nested single-pass range is rejected.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    struct Cursor(bool replayable)
    {
        const(double)[] data;
        size_t[] cursor;
        size_t position;
        bool detached;
        size_t index() const { return detached ? position : cursor[0]; }
        bool empty() const { return index == data.length; }
        double front() const { return data[index]; }
        void popFront() { if (detached) ++position; else ++cursor[0]; }
        static if (replayable)
        {
            @property auto save() return scope
            {
                auto result = this;
                result.position = index;
                result.detached = true;
                return result;
            }
        }
    }
    double[3] values = [0, 1, 5], weights = [1, 2, 3];
    size_t[1] xi, wi;
    auto x = Cursor!true(values[], xi[]);
    auto w = Cursor!true(weights[], wi[]);
    assert(weightedCentralMoment!3(x, w).approxEqual(-41.0 / 54));
    assert(xi[0] == 0 && wi[0] == 0);
    assert(weightedRawMoment!2(x, w).approxEqual(77.0 / 6));
    assert(weightedStandardizedMoment!3(x, w).approxEqual(weightedStandardizedMoment!3(values, weights)));
    assert(xi[0] == 0 && wi[0] == 0);
    auto input = Cursor!false(values[], xi[]);
    auto inputWeights = Cursor!false(weights[], wi[]);
    static assert(!__traits(compiles, weightedCentralMoment!2(input, w)));
    static assert(!__traits(compiles, weightedCentralMoment!2(x, inputWeights)));
    Cursor!false[1] nested = [input];
    static assert(!__traits(compiles, weightedCentralMoment!2(nested, nested)));
    static assert(!__traits(compiles, weightedStandardizedMoment!3(input, w)));
    static assert(!__traits(compiles, weightedStandardizedMoment!3(x, inputWeights)));
    static assert(!__traits(compiles, weightedStandardizedMoment!3(nested, nested)));
    assert(weightedRawMoment!2(input, inputWeights).approxEqual(77.0 / 6));
    assert(xi[0] == values.length && wi[0] == weights.length);
    xi[0] = wi[0] = 0;
    assert(weightedCentralMoment!1(input, inputWeights) == 0);
    assert(xi[0] == values.length && wi[0] == weights.length);
    xi[0] = wi[0] = 0;
    assert(weightedStandardizedMoment!2(input, inputWeights) == 1);
    assert(xi[0] == values.length && wi[0] == weights.length);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.rc.array: RCArray;
    import mir.math.common: approxEqual;
    auto values = RCArray!double(3);
    auto weights = RCArray!double(3);
    values[0] = 0; values[1] = 1; values[2] = 5;
    weights[0] = 1; weights[1] = 2; weights[2] = 3;
    assert(weightedRawMoment!2(values, weights).approxEqual(77.0 / 6));
    assert(weightedCentralMoment!3(values, weights).approxEqual(-41.0 / 54));
    assert(weightedStandardizedMoment!3(values, weights).approxEqual(weightedStandardizedMoment!3(values.asSlice, weights.asSlice)));
    assert(values.length == 3 && values[0] == 0 && values[2] == 5);
    assert(weights.length == 3 && weights[0] == 1 && weights[2] == 3);
}

// Large orders must not overflow the internal exponent bookkeeping.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: isInfinity;
    double[2] symmetric = [-1, 1], equal = [1, 1];
    assert(weightedCentralMoment!1024(symmetric, equal) == 1);
    assert(weightedCentralMoment!(size_t.max)(symmetric, equal) == 0);
    double[1] value = [-1], weight = [1];
    assert(weightedRawMoment!(size_t.max)(value, weight) == -1);
    value[0] = 2;
    assert(weightedRawMoment!(size_t.max)(value, weight).isInfinity);
    value[0] = 0.5;
    assert(weightedRawMoment!(size_t.max)(value, weight) == 0);
}

// DMD #24118: an affected argument can be either observations or weights.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    version (DigitalMars)
    {
        version (Win64) enum rejectTemporaries = true;
        else enum rejectTemporaries = false;
    }
    else enum rejectTemporaries = false;

    static foreach (T; AliasSeq!(float, double, const(float), const(double)))
    static foreach (fun; AliasSeq!(weightedRawMoment, weightedCentralMoment, weightedStandardizedMoment))
    static foreach (explicitType; [false, true])
    {{
        static if (explicitType)
            alias args = AliasSeq!(double, 2);
        else
            alias args = AliasSeq!(2);
        enum expected = __traits(identifier, fun) == "weightedRawMoment" ? 4 :
            __traits(identifier, fun) == "weightedCentralMoment" ? 0 : 1;
        T[1] values = [-2], weights = [1];
        assert(fun!args(values, weights) == expected);
        assert(fun!args(values[], weights[]) == expected);
        static if (rejectTemporaries)
        {
            static assert(!__traits(compiles, fun!args(cast(T[1])[-2], weights)));
            static assert(!__traits(compiles, fun!args(values, cast(T[1])[1])));
            static assert(!__traits(compiles, fun!args(cast(T[1])[-2], cast(T[1])[1])));
        }
        else
        {
            assert(fun!args(cast(T[1])[-2], weights) == expected);
            assert(fun!args(values, cast(T[1])[1]) == expected);
            assert(fun!args(cast(T[1])[-2], cast(T[1])[1]) == expected);
        }
        // Other static-array temporaries remain usable on Windows DMD too.
        assert(fun!args(cast(T[2])[-2, -2], cast(T[2])[1, 1]) == expected);
        assert(fun!args(cast(int[1])[-2], cast(int[1])[1]) == expected);
    }}
}

/++
Weighted N-th central moment divided by the N-th power of the population
standard deviation. The result has no units: use it to compare distribution
shapes measured on different scales.

Order three is population weighted skewness. Order four is raw population
weighted kurtosis; subtract three to obtain excess kurtosis. Counts,
probabilities, and relative weights use the same descriptive formula, with no
sample bias correction. Multiplying all weights by a positive constant leaves
the mathematical result unchanged.

As with standardizedMoment, orders one and two return zero and one, including
for constant observations. Unlike the unweighted shortcuts, these orders
traverse and validate the weighted inputs. Higher orders return NaN when the
positive-weight observations are constant or the accumulated scaled sum of
squared deviations underflows to zero. Empty inputs and inputs with no positive weights violate preconditions.

Input, result-type, weight validity, and Windows DMD temporary-array restrictions
follow weightedRawMoment. Orders above two require replayable inputs at every
nesting level. Three traversals find a center and scale, estimate the mean,
and accumulate the variance and higher centered power. No observation buffer
is allocated, and forward-range cursors are saved.

The calculation keeps the variance and higher powers in scaled coordinates,
avoiding overflow or underflow caused solely by the measurement units. Tiny
relative weights and normalized powers may still underflow, cancellation can
limit odd-order accuracy, and a result outside the result type's range can
overflow. Changing the sign of all observations reverses odd-order moments
and leaves even-order moments unchanged, up to rounding.

Params:
    F = floating result/working type, or a type promoted to floating point
    N = positive moment order
    summation = summation algorithm; appropriate selects pairwise summation
+/
template weightedStandardizedMoment(F, size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    /// values and weights must have matching shapes or lengths.
    stdevType!F weightedStandardizedMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        return weightedCenteredMoment!(stdevType!F, N, summation, true)(values, weights);
    }
}

/// ditto
template weightedStandardizedMoment(size_t N, Summation summation = Summation.appropriate)
    if (N > 0)
{
    import mir.primitives: DeepElementType;
    /// ditto
    stdevType!(DeepElementType!X) weightedStandardizedMoment(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        mixin WeightedMomentCheckArguments!(values, weights);
        return .weightedStandardizedMoment!(stdevType!(DeepElementType!X), N, summation)(values, weights);
    }
}

/// Compare shape independently of the measurement units.
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: sqrt;
    double[2] values = [0, 10], weights = [1, 3];
    double[2] centimeters = [0, 1000], probabilities = [0.25, 0.75];
    assert(weightedStandardizedMoment!3(values, weights).approxEqual(-2 / sqrt(3.0)));
    assert(weightedStandardizedMoment!4(values, weights).approxEqual(7.0 / 3));
    assert(weightedStandardizedMoment!4(centimeters, probabilities).approxEqual(7.0 / 3));
    // Excess kurtosis uses a normal distribution's raw kurtosis of three.
    assert((weightedStandardizedMoment!4(values, weights) - 3).approxEqual(-2.0 / 3));
}

// Population identities and scale invariance, including dimensional moments
// that overflow or underflow even though standardized moments are finite.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import std.math: sqrt, isNaN;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        T[2] values = [0, 10], weights = [1, 3], probabilities = [0.25, 0.75];
        T[4] expanded = [0, 10, 10, 10];
        static foreach (N; [3, 4, 5, 6, 8])
        {{
            const result = weightedStandardizedMoment!(T, N, mode)(values, weights);
            assert(result.approxEqual(standardizedMoment!N(expanded[].sliced), T(128) * T.epsilon, T(0)));
            assert(result.approxEqual(weightedStandardizedMoment!(N, mode)(values, probabilities), T(128) * T.epsilon, T(0)));
        }}
        assert(weightedStandardizedMoment!(3, mode)(values, weights).approxEqual(-2 / sqrt(T(3)), T(64) * T.epsilon, T(0)));
        assert(weightedStandardizedMoment!(4, mode)(values, weights).approxEqual(T(7) / 3, T(64) * T.epsilon, T(0)));
        assert(weightedStandardizedMoment!(5, mode)(values, weights).approxEqual(-T(20) / (3 * sqrt(T(3))), T(128) * T.epsilon, T(0)));
        assert(weightedStandardizedMoment!(6, mode)(values, weights).approxEqual(T(61) / 9, T(128) * T.epsilon, T(0)));
        T[2] symmetric = [-T.max, T.max], equal = [1, 1];
        assert(weightedStandardizedMoment!(3, mode)(symmetric, equal) == 0);
        assert(weightedStandardizedMoment!(4, mode)(symmetric, equal) == 1);
        symmetric = [-T.min_normal, T.min_normal];
        assert(weightedStandardizedMoment!(4, mode)(symmetric, equal) == 1);
        T[2] adjacent = [1, 1 + T.epsilon];
        assert(weightedStandardizedMoment!(4, mode)(adjacent, equal) == 1);
        T[3] masked = [T.nan, 7, T.infinity], mask = [0, 1, 0];
        assert(weightedStandardizedMoment!(1, mode)(masked, mask) == 0);
        assert(weightedStandardizedMoment!(2, mode)(masked, mask) == 1);
        assert(weightedStandardizedMoment!(3, mode)(masked, mask).isNaN);
        assert(weightedStandardizedMoment!(4, mode)(masked, mask).isNaN);
        static assert(is(typeof(weightedStandardizedMoment!3(values, weights)) == T));
    }}
    int[2] integers = [0, 10], counts = [1, 3];
    static assert(is(typeof(weightedStandardizedMoment!3(integers, counts)) == double));
    static assert(is(typeof(weightedStandardizedMoment!(float, 3)(integers, counts)) == float));
    double[2] symmetric = [-1, 1], equal = [1, 1];
    assert(weightedStandardizedMoment!1024(symmetric, equal) == 1);
    assert(weightedStandardizedMoment!(size_t.max)(symmetric, equal) == 0);
    double[3] shifted = [1e12, 1e12 + 1, 1e12 + 5], plain = [0, 1, 5], reflected = [0, -1, -5], w = [1, 2, 3];
    static foreach (N; [3, 4, 5, 6])
    {{
        const expected = weightedStandardizedMoment!N(plain, w);
        assert(weightedStandardizedMoment!N(shifted, w).approxEqual(expected, 1e-13, 0));
        assert(weightedStandardizedMoment!N(reflected, w).approxEqual(N % 2 ? -expected : expected, 1e-13, 0));
    }}
    // A tiny weight can produce a large but finite standardized fourth moment.
    double[2] rare = [0, 1], rareWeights = [1, 1e-300];
    assert(weightedStandardizedMoment!4(rare, rareWeights).approxEqual(1e300, 1e-13, 0));
}

// Normalization must not erase surviving subnormal sums before the final
// standardized result is restored. The expected fourth moment was computed
// from exact rational representations of these double inputs.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: sqrt;
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        double[3] values = [-1e-100, 1e-100, 1];
        double[3] weights = [1, 1, 0x1p-1074];
        assert(weightedStandardizedMoment!(4, mode)(values, weights)
            .approxEqual(2.4703282292062325233e76, 5e-14, 0));
        // Here even squares / mass underflows, but skewness is finite.
        values = [0, 0, 1];
        assert(weightedStandardizedMoment!(3, mode)(values, weights)
            .approxEqual(sqrt(2.0) * 0x1p537, 5e-14, 0));
        float[3] floatValues = [0, 0, 1], floatWeights = [1, 1, 0x1p-149f];
        assert(weightedStandardizedMoment!(3, mode)(floatValues, floatWeights)
            .approxEqual(0x1p75f, 16 * float.epsilon, 0));
    }}
}
