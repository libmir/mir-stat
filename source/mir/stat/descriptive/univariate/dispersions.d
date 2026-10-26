/++
Variance, standard deviation, and related measures of dispersion.

Use the named functions for standard summaries, or $(REF dispersion, mir, stat, descriptive, univariate, dispersions)
to choose the centering, transformation, and summary operations.
$(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions) selects the variance calculation algorithm.

Weighted spread uses $(LREF weightedVariance) and $(LREF weightedStandardDeviation).
Select population, frequency-sample, or relative-weight-sample normalization through
$(LREF WeightedVarianceNormalization); population is the default.

Import `mir.stat.descriptive.univariate.dispersions` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF variance, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF standardDeviation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF weightedVariance, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF weightedStandardDeviation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF coefficientOfVariation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF dispersion, mir, stat, descriptive, univariate, dispersions)$(NBSP)
    ))
    $(TR $(TD Accumulator) $(TD
        $(REF VarianceAccumulator, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF WVarianceAccumulator, mir, stat, descriptive, univariate, dispersions)$(NBSP)
    ))
    $(TR $(TD Algorithm selection) $(TD
        $(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF WeightedVarianceNormalization, mir, stat, descriptive, univariate, dispersions)$(NBSP)
    ))
))

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

module mir.stat.descriptive.univariate.dispersions;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIntegral, isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType, stdevType;
import mir.stat.descriptive.univariate.means: mean, MeanAccumulator;
import mir.stat.descriptive.univariate.quantiles: median;

// DMD's optimized inliner can diagnose division by zero in a guarded
// integer mean read. __traits(compiles) succeeds before that optimization,
// so use a compiler condition until a fixed DMD version is established.
version (DigitalMars)
    private enum workaroundDmdGuardedIntegerMean = true;
else
    private enum workaroundDmdGuardedIntegerMean = false;

/++
Variance algorithms.

See Also:
    $(WEB en.wikipedia.org/wiki/Algorithms_for_calculating_variance, Algorithms for calculating variance).
+/
enum VarianceAlgo
{
    /++
    Performs Welford's online algorithm for updating variance. Can also `put`
    another VarianceAccumulator of different types, which uses the parallel
    algorithm from Chan et al., described above.
    +/
    online,

    /++
    Calculates variance using E(x^^2) - E(x)^2 (alowing for adjustments for
    population/sample variance). This algorithm can be numerically unstable. As
    in:
    E(x ^^ 2) - E(x) ^^ 2
    +/
    naive,

    /++
    Calculates variance using a two-pass algorithm whereby the input is first
    centered and then the sum of squares is calculated from that. As in:
    E((x - E(x)) ^^ 2)

    Non-slice ranges must be forward ranges so each pass can traverse the same
    observations.
    +/
    twoPass,

    /++
    Calculates variance assuming the mean of the dataseries is zero.
    +/
    assumeZeroMean,

    /++
    Uses the two-pass algorithm for slices, slice-like objects, and forward ranges.
    Single-pass inputs and individual data points use the online algorithm.
    +/
    hybrid
}

// Transfer stored square-sum corrections only when their representation and
// summator type match. Naive variance stores raw squares; online, two-pass,
// and hybrid variance store centered squares. Other sources still need their
// public getter's conversion. Kahan does not transfer its correction through
// +=, and pairwise direct merging is unsafe in supported dependency releases.
private template canMergeVarianceSquares(T, Summation summation, V, string member)
{
    static if (__traits(hasMember, V, member) &&
        (summation == Summation.kbn || summation == Summation.kb2 ||
         summation == Summation.precise))
        enum canMergeVarianceSquares = is(typeof(__traits(getMember, V, member)) ==
            Summator!(T, summation));
    else
        enum canMergeVarianceSquares = false;
}

///
struct VarianceAccumulator(T, VarianceAlgo varianceAlgo, Summation summation)
    if (isMutable!T && varianceAlgo == VarianceAlgo.naive)
{
    import mir.math.sum: Summator;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;

    ///
    private Summator!(T, summation) summatorOfSquares;

    ///
    this(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        this.put(r.move);
    }

    ///
    this()(T x)
    {
        this.put(x);
    }


    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        foreach(x; r)
        {
            this.put(x);
        }
    }

    ///
    void put()(T x)
    {
        meanAccumulator.put(x);
        summatorOfSquares.put(x * x);
    }

    ///
    void put(U, Summation sumAlgo)(VarianceAccumulator!(U, varianceAlgo, sumAlgo) v)
    {
        meanAccumulator.put(v.meanAccumulator);
        static if (canMergeVarianceSquares!(T, summation, typeof(v), "summatorOfSquares"))
            summatorOfSquares += v.summatorOfSquares;
        else
            summatorOfSquares.put(v.sumOfSquares!T);
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() const @property
    {
        return meanAccumulator.mean!F;
    }
    ///
    F sumOfSquares(F = T)()
    {
        return cast(F) summatorOfSquares.sum;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return sumOfSquares!F - count * mean!F * mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "VarianceAccumulator.varaince: count must be larger than one");
    }
    do
    {
        return sumOfSquares!F / (count + isPopulation - 1) -
            mean!F * mean!F * count / (count + isPopulation - 1);
    }
}

/// naive
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));

    v.put(4.0);
    assert(v.variance(true).approxEqual(57.01923 / 13));
    assert(v.variance(false).approxEqual(57.01923 / 12));
}

// Can put VarianceAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive) v;
    v.put(x);
    VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    v.variance(true).shouldApprox == 54.76562 / 12;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive)(x1);
    v1.variance(true).should == 2;
    v1.centeredSumOfSquares.should == 10;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive)(x2);
    v2.variance(true).should == 8;
}

///
struct VarianceAccumulator(T, VarianceAlgo varianceAlgo, Summation summation)
    if (isMutable!T && varianceAlgo == VarianceAlgo.online)
{
    import mir.math.sum: Summator;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;

    ///
    private Summator!(T, summation) centeredSummatorOfSquares;

    ///
    this(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        this.put(r.move);
    }

    ///
    this()(T x)
    {
        this.put(x);
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        foreach(x; r)
        {
            this.put(x);
        }
    }

    ///
    void put()(T x)
    {
        T delta = x;
        static if (workaroundDmdGuardedIntegerMean && isIntegral!T)
        {
            // DMD can fold the guarded mean of a known-empty integer
            // accumulator into division by zero. Keep the guard and mean
            // calculation together in a conditional expression.
            delta = count ? x - meanAccumulator.mean : x;
        }
        else if (count > 0)
        {
            delta -= meanAccumulator.mean;
        }
        meanAccumulator.put(x);
        centeredSummatorOfSquares.put(delta * (x - meanAccumulator.mean));
    }

    ///
    void put(U, VarianceAlgo varAlgo, Summation sumAlgo)(VarianceAccumulator!(U, varAlgo, sumAlgo) v)
        if (varAlgo != VarianceAlgo.assumeZeroMean)
    {
        // Keep the small KBN merge inline: an out-of-line call otherwise
        // forces accumulator copies and spills in LDC's ordinary merge loop.
        version (LDC)
            static if (isFloatingPoint!T && summation == Summation.kbn &&
                canMergeVarianceSquares!(T, summation, typeof(v), "centeredSummatorOfSquares"))
                pragma(inline, true);
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        T delta;
        delta = 0;
        if (oldCount > 0) {
            delta = v.mean!T - meanAccumulator.mean;
        }
        // With an empty destination, keep the correction zero without squaring
        // the source mean. Share the updates with the ordinary merge path.
        meanAccumulator.put!T(v.meanAccumulator);
        static if (canMergeVarianceSquares!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            // Merge the saved correction before adding the shift between means;
            // rounding either together with the source sum can discard it.
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            static if (isIntegral!T)
                centeredSummatorOfSquares.put(delta * delta * v.count * oldCount / count);
            else
                centeredSummatorOfSquares.put(
                    delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
        }
        else static if (isIntegral!T)
        {
            // Preserve integer division after multiplication: the count ratio
            // alone would truncate to zero for two nonempty accumulators.
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
                + delta * delta * v.count * oldCount / count);
        }
        else
        {
            // Apply the count ratio before multiplying by delta twice, avoiding
            // intermediate overflow when the final correction is representable.
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
                + delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
        }
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() const @property
    {
        return meanAccumulator.mean!F;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "VarianceAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
}

/// online
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));

    v.put(4.0);
    assert(v.variance(true).approxEqual(57.01923 / 13));
    assert(v.variance(false).approxEqual(57.01923 / 12));
}

// can put slices
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    v.put(y);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (online)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Keep the 18 cases in separate functions to reduce Windows DMD's
// optimized compilation cost. Production calls inside each case may inline.
version(mir_stat_test)
pragma(inline, false)
private void checkVarianceMergeIdentities(T, VarianceAlgo algo, Summation mode)() @safe pure nothrow @nogc
{
    import std.math: fabs;

    alias Accumulator = VarianceAccumulator!(T, algo, mode);
    Accumulator empty, values;
    values.put(T(1));
    values.put(T(3));
    values.put(empty);
    assert(values.count == 2 && values.variance(false) == 2);
    empty.put(empty);
    assert(empty.count == 0);
    empty.put(values);
    assert(empty.count == 2 && empty.variance(false) == 2);

    Accumulator constant, destination;
    constant.put(T.max / 8);
    constant.put(T.max / 8);
    destination.put(constant);
    assert(destination.count == 2 && destination.variance(false) == 0);
    assert(constant.count == 2 && constant.variance(false) == 0);

    values.put(values);
    assert(values.count == 4);
    assert(fabs(values.variance(false) - T(4) / 3) <= 8 * T.epsilon);
}

// Empty merges are identities; nonempty and self merges preserve the statistics.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (algo; AliasSeq!(VarianceAlgo.online, VarianceAlgo.hybrid))
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kbn))
        checkVarianceMergeIdentities!(T, algo, mode)();
}

// Explicit integer accumulators retain multiplication before integer division.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    static foreach (algo; AliasSeq!(VarianceAlgo.online, VarianceAlgo.hybrid))
    {{
        alias Accumulator = VarianceAccumulator!(ulong, algo, Summation.naive);
        Accumulator left, right, empty;
        left.put(0UL);
        right.put(2UL);
        left.put(right);
        assert(left.count == 2);
        assert(left.centeredSumOfSquares == 2);
        assert(left.variance(false) == 2);
        left.put(empty);
        assert(left.count == 2 && left.centeredSumOfSquares == 2);
        empty.put(left);
        assert(empty.count == 2 && empty.centeredSumOfSquares == 2);
    }}
}

// Count factors and squaring can overflow before a finite merge correction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (algo; AliasSeq!(VarianceAlgo.online, VarianceAlgo.hybrid))
    {{
        alias Accumulator = VarianceAccumulator!(T, algo, Summation.naive);
        // Binary powers give an exact correction of 8*x*x for two groups of 16.
        static if (T.max_exp == 128)
            enum T x = 0x1p60f;
        else static if (T.max_exp == 1024)
            enum T x = 0x1p508;
        else
            enum T x = 0x1p8189L;
        Accumulator left, right;
        foreach (i; 0 .. 16)
        {
            left.put(T(0));
            right.put(x);
        }
        left.put(right);
        assert(left.count == 32);
        assert(left.centeredSumOfSquares == x * (x * 8));
        assert(fabs(left.variance(false) / (x * (x * (T(8) / 31))) - 1)
            <= 8 * T.epsilon);

        // For two singletons the correction is x*x/2, even when x*x overflows.
        static if (T.max_exp == 128)
            enum T large = 0x1.4p64f;
        else static if (T.max_exp == 1024)
            enum T large = 0x1.4p512;
        else
            enum T large = 0x1.4p8192L;
        Accumulator first, second;
        first.put(T(0));
        second.put(large);
        first.put(second);
        assert(first.count == 2);
        assert(first.variance(false) == large * (large / 2));
    }}
}

// Can put accumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    auto w = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// complex
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex: Complex;

    auto x = [Complex!double(1.0, 3), Complex!double(2), Complex!double(3)].sliced;

    VarianceAccumulator!(Complex!double, VarianceAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(Complex!double(-4.0, -6) / 3));
    assert(v.variance(false).approxEqual(Complex!double(-4.0, -6) / 2));
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive)(x1);
    v1.variance(true).should == 2;
    v1.centeredSumOfSquares.should == 10;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive)(x2);
    v2.variance(true).should == 8;
}

///
struct VarianceAccumulator(T, VarianceAlgo varianceAlgo, Summation summation)
    if (isMutable!T && varianceAlgo == VarianceAlgo.twoPass)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange, isForwardRange, save;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;

    ///
    private Summator!(T, summation) centeredSummatorOfSquares;

    ///
    this(Iterator, size_t N, SliceKind kind)(
         Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: vmap, map;

        meanAccumulator.put(slice.lightScope);
        centeredSummatorOfSquares.put(slice.vmap(LeftOp!("-", T)(meanAccumulator.mean)).map!(naryFun!"a * a"));
    }

    ///
    this(SliceLike)(SliceLike x)
        if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice);
    }

    ///
    this(Range)(Range range)
        if (isForwardRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import std.algorithm: map;
        // A copy may share its cursor; save an independent first traversal.
        meanAccumulator.put(range.save);

        auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a");
        centeredSummatorOfSquares.put(centeredRangeMultiplier);
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() const @property
    {
        return meanAccumulator.mean;
    }
    ///
    F centeredSumOfSquares(F = T)() const @property
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "SkewnessAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
}

/// twoPass
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    assert(v.centeredSumOfSquares.approxEqual(54.76562));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.math.sum: sum;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x);
    assert(v.centeredSumOfSquares.sum.approxEqual(54.76562));
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x1);
    v1.variance(true).should == 2;
    v1.centeredSumOfSquares.should == 10;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(x2);
    v2.variance(true).should == 8;
}

///
struct VarianceAccumulator(T, VarianceAlgo varianceAlgo, Summation summation)
    if (isMutable!T && varianceAlgo == VarianceAlgo.assumeZeroMean)
{
    import mir.math.sum: Summator;
    import mir.ndslice.slice: Slice, SliceKind;

    private size_t _count;
    ///
    private Summator!(T, summation) centeredSummatorOfSquares;

    ///
    this(Range)(Range r)
        if (isIterable!Range)
    {
        this.put(r);
    }

    ///
    this()(T x)
    {
        this.put(x);
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        foreach(x; r)
        {
            this.put(x);
        }
    }

    ///
    void put()(T x)
    {
        _count++;
        centeredSummatorOfSquares.put(x * x);
    }

    ///
    void put(U, Summation sumAlgo)(VarianceAccumulator!(U, varianceAlgo, sumAlgo) v)
    {
        _count += v.count;
        static if (canMergeVarianceSquares!(T, summation, typeof(v), "centeredSummatorOfSquares"))
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F mean(F = T)() const @property
    {
        return cast(F) 0;
    }
    ///
    MeanAccumulator!(T, summation) meanAccumulator()()
    {
        typeof(return) m = { _count, T(0) };
        return m;
    }
    ///
    F centeredSumOfSquares(F = T)() const @property
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
}

/// assumeZeroMean
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.transform: center;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    VarianceAccumulator!(double, VarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
    v.put(4.0);
    assert(v.variance(true).approxEqual(70.76562 / 13));
    assert(v.variance(false).approxEqual(70.76562 / 12));
}

// can put slices
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.transform: center;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    VarianceAccumulator!(double, VarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(13.492188 / 6));
    assert(v.variance(false).approxEqual(13.492188 / 5));

    v.put(y);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// can put two accumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.transform: center;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    VarianceAccumulator!(double, VarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(13.492188 / 6));
    assert(v.variance(false).approxEqual(13.492188 / 5));

    VarianceAccumulator!(double, VarianceAlgo.assumeZeroMean, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// complex
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex: Complex;
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [Complex!double(1.0, 3), Complex!double(2), Complex!double(3)].sliced;
    auto x = a.center;

    VarianceAccumulator!(Complex!double, VarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(Complex!double(-4.0, -6) / 3));
    assert(v.variance(false).approxEqual(Complex!double(-4.0, -6) / 2));
}

///
struct VarianceAccumulator(T, VarianceAlgo varianceAlgo, Summation summation)
    if (isMutable!T && varianceAlgo == VarianceAlgo.hybrid)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange, isForwardRange, save;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;

    ///
    private Summator!(T, summation) centeredSummatorOfSquares;

    ///
    this(Iterator, size_t N, SliceKind kind)(
         Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: vmap, map;

        meanAccumulator.put(slice.lightScope);
        centeredSummatorOfSquares.put(slice.vmap(LeftOp!("-", T)(meanAccumulator.mean)).map!(naryFun!"a * a"));
    }

    ///
    this(SliceLike)(SliceLike x)
        if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice);
    }

    ///
    this(Range)(Range range)
        if (isIterable!Range && !isConvertibleToSlice!Range)
    {
        static if (isForwardRange!Range && is(elementType!Range : T))
        {
            import std.algorithm: map;
            // A copy may share its cursor; save an independent first traversal.
            meanAccumulator.put(range.save);

            auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a");
            centeredSummatorOfSquares.put(centeredRangeMultiplier);
        } else {
            this.put(range);
        }
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        static if (isInputRange!Range && is(elementType!Range : T) &&
            (isConvertibleToSlice!Range || isForwardRange!Range)) {
            auto v = typeof(this)(r);
            this.put(v);
        } else{
            foreach(x; r)
            {
                this.put(x);
            }
        }
    }

    ///
    void put()(T x)
    {
        T delta = x;
        static if (workaroundDmdGuardedIntegerMean && isIntegral!T)
        {
            // DMD can fold the guarded mean of a known-empty integer
            // accumulator into division by zero. Keep the guard and mean
            // calculation together in a conditional expression.
            delta = count ? x - meanAccumulator.mean : x;
        }
        else if (count > 0)
        {
            delta -= meanAccumulator.mean;
        }
        meanAccumulator.put(x);
        centeredSummatorOfSquares.put(delta * (x - meanAccumulator.mean));
    }

    ///
    void put(U, VarianceAlgo varAlgo, Summation sumAlgo)(VarianceAccumulator!(U, varAlgo, sumAlgo) v)
        if (varAlgo != VarianceAlgo.assumeZeroMean)
    {
        // Keep the small KBN merge inline: an out-of-line call otherwise
        // forces accumulator copies and spills in LDC's ordinary merge loop.
        version (LDC)
            static if (isFloatingPoint!T && summation == Summation.kbn &&
                canMergeVarianceSquares!(T, summation, typeof(v), "centeredSummatorOfSquares"))
                pragma(inline, true);
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        T delta;
        delta = 0;
        if (oldCount > 0) {
            delta = v.mean!T - meanAccumulator.mean;
        }
        // With an empty destination, keep the correction zero without squaring
        // the source mean. Share the updates with the ordinary merge path.
        meanAccumulator.put!T(v.meanAccumulator);
        static if (canMergeVarianceSquares!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            // Merge the saved correction before adding the shift between means;
            // rounding either together with the source sum can discard it.
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            static if (isIntegral!T)
                centeredSummatorOfSquares.put(delta * delta * v.count * oldCount / count);
            else
                centeredSummatorOfSquares.put(
                    delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
        }
        else static if (isIntegral!T)
        {
            // Preserve integer division after multiplication: the count ratio
            // alone would truncate to zero for two nonempty accumulators.
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
                + delta * delta * v.count * oldCount / count);
        }
        else
        {
            // Apply the count ratio before multiplying by delta twice, avoiding
            // intermediate overflow when the final correction is representable.
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
                + delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
        }
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() const @property
    {
        return meanAccumulator.mean!F;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "VarianceAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
}

/// online
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive)(x);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));

    v.put(4.0);
    assert(v.variance(true).approxEqual(57.01923 / 13));
    assert(v.variance(false).approxEqual(57.01923 / 12));
}

// can put slices
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive)(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    v.put(y);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (hybrid)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    VarianceAccumulator!(double, VarianceAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (online)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    VarianceAccumulator!(double, VarianceAlgo.online, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// Can put accumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(12.55208 / 6));
    assert(v.variance(false).approxEqual(12.55208 / 5));

    auto w = VarianceAccumulator!(double, VarianceAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.variance(true).approxEqual(54.76562 / 12));
    assert(v.variance(false).approxEqual(54.76562 / 11));
}

// complex
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex: Complex;

    auto x = [Complex!double(1.0, 3), Complex!double(2), Complex!double(3)].sliced;

    VarianceAccumulator!(Complex!double, VarianceAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.variance(true).approxEqual(Complex!double(-4.0, -6) / 3));
    assert(v.variance(false).approxEqual(Complex!double(-4.0, -6) / 2));
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: chunks, iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive)(x1);
    v1.variance(true).should == 2;
    v1.centeredSumOfSquares.should == 10;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive)(x2);
    v2.variance(true).should == 8;
    VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive) v3;
    v3.put(x1.chunks(1));
    v3.centeredSumOfSquares.should == 10;
    auto v4 = VarianceAccumulator!(double, VarianceAlgo.hybrid, Summation.naive)(x1.chunks(1));
    v4.centeredSumOfSquares.should == 10;
}

/++
Calculates the variance of the input

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

Large integer observations that differ by small amounts can lose those
differences when converted to the floating-point accumulator type. For example,
the five `long` values `(1L << 60) + [0, 1, 2, 3, 4]` have sample variance
`2.5`, but the default calculation returns zero because their conversions to
`double` are identical. A more accurate summation algorithm cannot recover
differences already lost during conversion.

When the data has a known common offset, subtracting it in integer arithmetic
before calling `variance` can preserve the differences. Ensure that the
subtractions fit the integer type. This translation preserves variance about
the mean, but must not be used with `VarianceAlgo.assumeZeroMean`, which
calculates squared distances from zero.

Floating-point observations can also lose accuracy when their spread is very
small compared with their common offset. The two-pass calculation measures
distances from a rounded mean. If the exact mean lies between representable
values, this can overestimate the variance even when the observations
themselves are represented exactly. For example, the two `double` values
`1.0` and `1.0 + double.epsilon` have sample variance
`double.epsilon * double.epsilon / 2`; the default calculation returns twice
that value. The absolute error is tiny in this example, but the relative
error is large.

For tightly clustered floating-point data, subtracting a nearby common offset
before calling `variance` can improve accuracy by allowing the mean of the
differences to be represented more accurately. Choose an offset for which
the subtractions remain finite. As with integer centering, this workaround
applies to variance about the mean, not `VarianceAlgo.assumeZeroMean`.

Params:
    F = controls type of output
    varianceAlgo = algorithm for calculating variance (default: VarianceAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    The variance of the input, must be floating point or complex type
+/
template variance(
    F,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population variance, false if sample variance (default)
    +/
    @fmamath meanType!F variance(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias G = typeof(return);
        auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(r.move);
        return varianceAccumulator.variance(isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!F variance(scope const F[] ar...)
    {
        alias G = typeof(return);
        auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, const(G)[], G))(ar);
        return varianceAccumulator.variance(false);
    }
}

/// ditto
template variance(
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population variance, false if sample variance (default)
    +/
    @fmamath meanType!Range variance(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .variance!(F, varianceAlgo, summation)(r.move, isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!T variance(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .variance!(F, varianceAlgo, summation)(ar);
    }
}

/// ditto
template variance(F, string varianceAlgo, string summation = "appropriate")
{
    mixin("alias variance = .variance!(F, VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template variance(string varianceAlgo, string summation = "appropriate")
{
    mixin("alias variance = .variance!(VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.complex.math: capproxEqual = approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex;
    alias C = Complex!double;

    assert(variance([1.0, 2, 3]).approxEqual(2.0 / 2));
    assert(variance([1.0, 2, 3], true).approxEqual(2.0 / 3));

    assert(variance([C(1, 3), C(2), C(3)]).capproxEqual(C(-4, -6) / 2));

    assert(variance!float([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(17.5 / 5));

    static assert(is(typeof(variance!float([1, 2, 3])) == float));
}

/// Variance of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.variance.approxEqual(54.76562 / 11));
}

/// Variance of matrix
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

    assert(x.variance.approxEqual(54.76562 / 11));
}

/// Column variance of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0,  1.0, 1.5, 2.0],
        [3.5, 4.25, 2.0, 7.5],
        [5.0,  1.0, 1.5, 0.0]
    ].fuse;
    auto result = [13.16667 / 2, 7.041667 / 2, 0.1666667 / 2, 30.16667 / 2];

    // Use byDim or alongDim with map to compute variance of row/column.
    assert(x.byDim!1.map!variance.all!approxEqual(result));
    assert(x.alongDim!0.map!variance.all!approxEqual(result));

    // FIXME
    // Without using map, computes the variance of the whole slice
    // assert(x.byDim!1.variance == x.sliced.variance);
    // assert(x.alongDim!0.variance == x.sliced.variance);
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

    auto x = a + 1_000_000_000;

    auto y = x.variance;
    assert(y.approxEqual(54.76562 / 11));

    // The naive algorithm is numerically unstable in this case
    auto z0 = x.variance!"naive";
    assert(!z0.approxEqual(y));

    auto z1 = x.variance!"online";
    assert(z1.approxEqual(54.76562 / 11));

    // But the two-pass algorithm provides a consistent answer
    auto z2 = x.variance!"twoPass";
    assert(z2.approxEqual(y));

    // And the assumeZeroMean algorithm is way off
    auto z3 = x.variance!"assumeZeroMean";
    assert(z3.approxEqual(1.2e19 / 11));
}

/// Center tightly clustered observations before calculating their variance.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice : sliced;
    import mir.ndslice.topology : map;

    const double[2] observations = [1.0, 1.0 + double.epsilon];
    // Subtract the common offset lazily; no intermediate array is needed.
    auto centered = observations[].sliced.map!(x => x - 1.0);
    const expected = double.epsilon * double.epsilon / 2;
    assert(centered.variance == expected);
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

    //Set population variance, variance algorithm, sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto x = a * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.variance(false).approxEqual(2.0e208 / 3));
    assert(x.variance(true).approxEqual(2.0e208 / 4));

    assert(x.variance!("online").approxEqual(2.0e208 / 3));
    assert(x.variance!("online", "kbn").approxEqual(2.0e208 / 3));
    assert(x.variance!("online", "kb2").approxEqual(2.0e208 / 3));
    assert(x.variance!("online", "precise").approxEqual(2.0e208 / 3));
    assert(x.variance!(double, "online", "precise").approxEqual(2.0e208 / 3));
    assert(x.variance!(double, "online", "precise")(true).approxEqual(2.0e208 / 4));

    auto y = uint.max.repeat(3);
    auto z = y.variance!ulong;
    assert(z == 0.0);
    static assert(is(typeof(z) == double));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.variance;
    assert(y.approxEqual(50.91667 / 11));
    static assert(is(typeof(y) == double));

    assert(x.variance!float.approxEqual(50.91667 / 11));
}

/++
Variance works for complex numbers and other user-defined types (provided they
can be converted to a floating point or complex type)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex;
    alias C = Complex!double;

    auto x = [C(1, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    assert(x.variance.approxEqual((C(0, 10)) / 3));
}

/// Compute variance along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;

    assert(x.variance.approxEqual(17.5 / 5));

    auto m0 = [4.5, 4.5, 4.5];
    assert(x.alongDim!0.map!variance.all!approxEqual(m0));
    assert(x.alongDim!(-2).map!variance.all!approxEqual(m0));

    auto m1 = [1.0, 1.0];
    assert(x.alongDim!1.map!variance.all!approxEqual(m1));
    assert(x.alongDim!(-1).map!variance.all!approxEqual(m1));

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!variance.all!approxEqual(repeat(3600.0 / 2, 3, 4, 5)));
}

/// Arbitrary variance
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(variance(1.0, 2, 3) == 1.0);
    assert(variance!float(1, 2, 3) == 1f);
}

// UCFS test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].variance.approxEqual(5.0 / 3));
}

// testing types are right along dimension
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!variance;
    assert(y.all!approxEqual([0.5, 0.5]));
    static assert(is(meanType!(typeof(y)) == double));
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

    assert(x.sliced.variance.approxEqual(54.76562 / 11));
    assert(x.sliced.variance!float.approxEqual(54.76562 / 11));
}

/*
Recover a representable standard deviation when squaring the observations or
residuals overflowed or underflowed. Scaling by a power of two keeps nearby
observations distinct; subtracting a fixed normalized observation avoids a
rounded mean creating a spread in constant inputs. The original variance is
computed first, so its contracts and ordinary results remain unchanged.
*/
private T recoverStandardDeviation(T, VarianceAlgo algo, Range)(
    Range values, bool isPopulation, T originalVariance)
{
    import std.math : abs, frexp, ldexp, isFinite, sqrt;
    import std.range.primitives : empty, front, save;

    if (values.empty)
        return sqrt(originalVariance);
    const first = cast(T) values.front;
    // Zero can mean either identical observations or underflowed squares.
    // Equality is cheaper to check than computing the scale for every value.
    if (originalVariance == 0 && isFinite(first))
    {
        static if (algo == VarianceAlgo.assumeZeroMean)
            const canBeConstant = first == 0;
        else
            enum canBeConstant = true;
        if (canBeConstant)
        {
            bool equal = true;
            foreach (x; values.save)
                if (cast(T) x != first)
                {
                    equal = false;
                    break;
                }
            if (equal)
                return 0;
        }
    }

    T maximum = 0;
    foreach (x; values.save)
    {
        const value = cast(T) x;
        if (!isFinite(value))
            return sqrt(originalVariance);
        if (abs(value) > maximum)
            maximum = abs(value);
    }
    if (maximum == 0)
        return sqrt(originalVariance);
    int exponent;
    frexp(maximum, exponent);
    const scale = ldexp(T(0.5), exponent);
    static if (algo == VarianceAlgo.assumeZeroMean)
        enum anchor = T(0);
    else
        const anchor = first / scale;
    size_t count;
    Summator!(T, Summation.kbn) sum;
    foreach (x; values.save)
    {
        sum.put(cast(T) x / scale - anchor);
        ++count;
    }
    static if (algo == VarianceAlgo.assumeZeroMean)
        enum center = T(0);
    else
        const center = sum.sum / count;
    Summator!(T, Summation.kbn) squares;
    foreach (x; values)
    {
        const residual = (cast(T) x / scale - anchor) - center;
        squares.put(residual * residual);
    }
    const v = squares.sum / (count + isPopulation - 1);
    return sqrt(v) * scale;
}

/++
Calculates the standard deviation of the input

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

For replayable floating-point inputs, a nonfinite, negative, zero, or
subnormal variance triggers a scaled recalculation. This can preserve a
standard deviation even when its square is not representable. Recovery uses
compensated two-pass variance, except that `assumeZeroMean` retains its
zero-centered definition. Ordinary positive normal variances use the selected
algorithm unchanged. Single-pass inputs do not receive this recovery.

Integer inputs retain the conversion limitation described for $(LREF variance):
large integers separated by small differences can produce a zero standard
deviation. The scaled floating-point recovery does not repair this loss of
integer precision.

Positive normal variances can still be inaccurate for tightly clustered
floating-point observations, as described for $(LREF variance). Taking their
square root does not remove that error, and these results do not trigger
scaled recovery. Subtracting a nearby common offset before calling
`standardDeviation` can help, except when using `VarianceAlgo.assumeZeroMean`.

Params:
    F = controls type of output
    varianceAlgo = algorithm for calculating variance (default: VarianceAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    The standard deviation of the input, must be floating point type type
+/
template standardDeviation(
    F,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import mir.math.common: sqrt;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population standard deviation, false if sample standard deviation (default)
    +/
    @fmamath stdevType!F standardDeviation(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        alias G = typeof(return);
        import std.range.primitives : isForwardRange, ElementType, save;
        import std.traits : isFloatingPoint;
        static if (isSlice!Range)
        {
            import mir.ndslice.topology : flattened;
            auto flat = r.flattened;
            static if (isForwardRange!(typeof(flat)) && isFloatingPoint!(ElementType!(typeof(flat))))
                auto saved = flat.save;
        }
        else static if (isForwardRange!Range && isFloatingPoint!(ElementType!Range))
            auto saved = r.save;
        const v = r.move.variance!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(isPopulation);
        static if (__traits(compiles, saved))
        {
            import std.math : isFinite;
            if (!isFinite(v) || v < G.min_normal)
                return recoverStandardDeviation!(G, varianceAlgo)(saved.move, isPopulation, v);
        }
        return v.sqrt;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F standardDeviation(scope const F[] ar...)
    {
        alias G = typeof(return);
        const v = ar.variance!(G, varianceAlgo, ResolveSummationType!(summation, const(G)[], G));
        static if (isFloatingPoint!F)
        {
            import std.math : isFinite;
            if (!isFinite(v) || v < G.min_normal)
                return recoverStandardDeviation!(G, varianceAlgo)(ar, false, v);
        }
        return v.sqrt;
    }
}

// Save the forward cursor independently before computing variance.
version(mir_stat_test)
unittest
{
    static struct Forward
    {
        const(double)[] data;
        size_t* position;
        size_t ownPosition;
        bool empty() const { return (position ? *position : ownPosition) == data.length; }
        double front() const { return data[position ? *position : ownPosition]; }
        void popFront() { if (position) ++*position; else ++ownPosition; }
        auto save() { return Forward(data, null, position ? *position : ownPosition); }
    }
    static struct Single
    {
        const(double)[] data;
        size_t* position;
        bool empty() const { return *position == data.length; }
        double front() const { return data[*position]; }
        void popFront() { ++*position; }
    }
    size_t position;
    // Standard-deviation recovery also needs an independent saved cursor.
    double[3] spread = [-1e200, 0, 1e200];
    position = 0;
    assert(standardDeviation!(VarianceAlgo.online)(Forward(spread[], &position)) == 1e200);
    assert(position == 3);
    position = 0;
    assert(standardDeviation!(VarianceAlgo.online)(Single(spread[], &position)) == double.infinity);
    assert(position == 3);
}

// Preserve assumeZeroMean's existing boundary results when recovery has no scale.
// Unlike the centered algorithms, this algorithm has no minimum-count contract.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math : isNaN;

    double[] empty;
    // Population variance divides zero by zero; recovery must not read front.
    assert(empty.standardDeviation!(VarianceAlgo.assumeZeroMean)(true).isNaN);
    // A singleton sample also divides zero by zero. All observations are zero,
    // so scaling cannot recover a standard deviation from the NaN variance.
    double[1] zero = [0.0];
    assert(zero[].standardDeviation!(VarianceAlgo.assumeZeroMean).isNaN);
    assert(zero[].standardDeviation!(VarianceAlgo.assumeZeroMean)(true) == 0);
}

// Recover standard deviations whose squared values are outside the type's range.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta : AliasSeq;
    import std.math : sqrt, abs, isNaN;
    import mir.ndslice.slice : sliced;
    import mir.ndslice.dynamic : transposed;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (magnitude; [T.max / 4, T.min_normal, T.min_normal * T.epsilon])
        {
            T[3] values = [-magnitude, 0, magnitude];
            static foreach (algo; AliasSeq!(VarianceAlgo.naive, VarianceAlgo.online,
                VarianceAlgo.twoPass, VarianceAlgo.hybrid, VarianceAlgo.assumeZeroMean))
            {{
                assert(values[].standardDeviation!(T, algo) == magnitude);
                const population = values[].standardDeviation!(T, algo)(true);
                const expected = magnitude * sqrt(T(2) / 3);
                assert(abs(population - expected) <= 4 * T.epsilon * magnitude
                    + T.min_normal * T.epsilon);
            }}
        }
        T[3] extremes = [-T.max, 0, T.max];
        assert(extremes[].standardDeviation!(T, VarianceAlgo.naive) == T.max);
        assert(standardDeviation!T(-T.max, T(0), T.max) == T.max);
        T[3] largeConstant;
        largeConstant[] = T.max / 4;
        const rms = largeConstant[].standardDeviation!(T, VarianceAlgo.assumeZeroMean)(true);
        assert(abs(rms / (T.max / 4) - 1) <= 2 * T.epsilon);
        T[5] identical;
        identical[] = T.max;
        assert(identical[].standardDeviation == 0);
        T[3] zero;
        zero[] = 0;
        assert(zero[].standardDeviation == 0);
        T[3] constant = [T(3), T(3), T(3)];
        assert(constant[].standardDeviation!(T, VarianceAlgo.assumeZeroMean)(true) == 3);
        T[2] invalid = [T.infinity, T.infinity];
        assert(invalid[].standardDeviation.isNaN);
    }}
    double[3] tiny = [-2e-162, 0, 2e-162];
    assert(abs(tiny[].standardDeviation / 2e-162 - 1) <= 2 * double.epsilon);
    const double[6] matrix = [-1e200, 0, 1e200, -1e200, 0, 1e200];
    assert(matrix[].sliced(2, 3).transposed[0].standardDeviation == 0);
    const expected = 1e200 * sqrt(4.0 / 5);
    assert(abs(matrix[].sliced(2, 3).transposed.standardDeviation / expected - 1)
        <= 4 * double.epsilon);
}

/// ditto
template standardDeviation(
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population standard deviation, false if sample standard deviation (default)
    +/
    @fmamath stdevType!Range standardDeviation(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .standardDeviation!(F, varianceAlgo, summation)(r.move, isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T standardDeviation(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .standardDeviation!(F, varianceAlgo, summation)(ar);
    }
}

/// ditto
template standardDeviation(F, string varianceAlgo, string summation = "appropriate")
{
    mixin("alias standardDeviation = .standardDeviation!(F, VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template standardDeviation(string varianceAlgo, string summation = "appropriate")
{
    mixin("alias standardDeviation = .standardDeviation!(VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    assert(standardDeviation([1.0, 2, 3]).approxEqual(sqrt(2.0 / 2)));
    assert(standardDeviation([1.0, 2, 3], true).approxEqual(sqrt(2.0 / 3)));

    assert(standardDeviation!float([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(sqrt(17.5 / 5)));

    static assert(is(typeof(standardDeviation!float([1, 2, 3])) == float));
}

/// Standard deviation of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.standardDeviation.approxEqual(sqrt(54.76562 / 11)));
}

/// Standard deviation of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.standardDeviation.approxEqual(sqrt(54.76562 / 11)));
}

/// Column standard deviation of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0,  1.0, 1.5, 2.0],
        [3.5, 4.25, 2.0, 7.5],
        [5.0,  1.0, 1.5, 0.0]
    ].fuse;
    auto result = [13.16667 / 2, 7.041667 / 2, 0.1666667 / 2, 30.16667 / 2].map!sqrt;

    // Use byDim or alongDim with map to compute standardDeviation of row/column.
    assert(x.byDim!1.map!standardDeviation.all!approxEqual(result));
    assert(x.alongDim!0.map!standardDeviation.all!approxEqual(result));

    // FIXME
    // Without using map, computes the standardDeviation of the whole slice
    // assert(x.byDim!1.standardDeviation == x.sliced.standardDeviation);
    // assert(x.alongDim!0.standardDeviation == x.sliced.standardDeviation);
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 1_000_000_000;

    auto y = x.standardDeviation;
    assert(y.approxEqual(sqrt(54.76562 / 11)));

    // Prefer two-pass for large offsets. Naive variance can lose the spread;
    // recovery catches zero or negative results, but not an inaccurate positive result.
    // The two-pass algorithm provides a consistent answer
    auto z1 = x.standardDeviation!"twoPass";
    assert(z1.approxEqual(y));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;

    //Set population standard deviation, standardDeviation algorithm, sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto x = a * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.standardDeviation(false).approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation(true).approxEqual(sqrt(2.0e208 / 4)));

    assert(x.standardDeviation!("online").approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation!("online", "kbn").approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation!("online", "kb2").approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation!("online", "precise").approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation!(double, "online", "precise").approxEqual(sqrt(2.0e208 / 3)));
    assert(x.standardDeviation!(double, "online", "precise")(true).approxEqual(sqrt(2.0e208 / 4)));

    auto y = uint.max.repeat(3);
    auto z = y.standardDeviation!ulong;
    assert(z == 0.0);
    static assert(is(typeof(z) == double));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.standardDeviation;
    assert(y.approxEqual(sqrt(50.91667 / 11)));
    static assert(is(typeof(y) == double));

    assert(x.standardDeviation!float.approxEqual(sqrt(50.91667 / 11)));
}

/++
Variance works for other user-defined types (provided they
can be converted to a floating point)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    static struct Foo {
        float x;
        alias x this;
    }

    Foo[] foo = [Foo(1f), Foo(2f), Foo(3f)];
    assert(foo.standardDeviation == 1f);
}

/// Compute standard deviation along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;

    assert(x.standardDeviation.approxEqual(sqrt(17.5 / 5)));

    auto m0 = repeat(sqrt(4.5), 3);
    assert(x.alongDim!0.map!standardDeviation.all!approxEqual(m0));
    assert(x.alongDim!(-2).map!standardDeviation.all!approxEqual(m0));

    auto m1 = [1.0, 1.0];
    assert(x.alongDim!1.map!standardDeviation.all!approxEqual(m1));
    assert(x.alongDim!(-1).map!standardDeviation.all!approxEqual(m1));

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!standardDeviation.all!approxEqual(repeat(sqrt(3600.0 / 2), 3, 4, 5)));
}

/// Arbitrary standard deviation
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: sqrt;

    assert(standardDeviation(1.0, 2, 3) == 1.0);
    assert(standardDeviation!float(1, 2, 3) == 1f);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    assert([1.0, 2, 3, 4].standardDeviation.approxEqual(sqrt(5.0 / 3)));
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!standardDeviation;
    assert(y.all!approxEqual([sqrt(0.5), sqrt(0.5)]));
    static assert(is(meanType!(typeof(y)) == double));
}

version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.standardDeviation.approxEqual(sqrt(54.76562 / 11)));
    assert(x.sliced.standardDeviation!float.approxEqual(sqrt(54.76562 / 11)));
}

/++
Calculates the dispersion of the input.

For an input `x`, this function first centers `x` by subtracting each `e` in `x`
by the result of `centralTendency`, then it transforms the centered values using
the function `transform`, and then finally summarizes that information using
the `summarize` funcion.

The default functions provided are equivalent to calculating the population
variance. The `centralTendency` default is the `mean` function, which results
in the input being centered about the mean. The default `transform` function
will square the centered values. The default `summarize` function is `mean`,
which will return the mean of the squared centered values.

Params:
    centralTendency = function that will produce the value that the input is centered about, default is `mean`
    transform = function to transform centered values, default squares the centered values
    summarize = function to summarize the transformed centered values, default is `mean`

Returns:
    The dispersion of the input
+/
template dispersion(
    alias centralTendency = mean,
    alias transform = "a * a",
    alias summarize = mean)
{
    import mir.functional: naryFun;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind, sliced;

    static if (__traits(isSame, naryFun!transform, transform))
    {
        /++
        Params:
            slice = slice
        +/
        @fmamath auto dispersion(Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
        {
            import mir.ndslice.topology: map;
            import mir.stat.transform: center;

            return summarize(slice.center!centralTendency.map!transform);
        }

        /// ditto
        @fmamath auto dispersion(T)(scope const T[] ar...)
        {
            return dispersion(ar.sliced);
        }

        /// ditto
        @fmamath auto dispersion(SliceLike)(SliceLike x)
            if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
        {
            import mir.ndslice.slice: toSlice;
            return dispersion(x.toSlice);
        }
    }
    else
        alias dispersion = .dispersion!(centralTendency, naryFun!transform, summarize);
}

/// Simple examples
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.complex: Complex;
    import mir.complex.math: capproxEqual = approxEqual;
    import mir.functional: naryFun;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    alias C = Complex!double;

    assert(dispersion([1.0, 2, 3]).approxEqual(2.0 / 3));

    assert(dispersion([C(1.0, 3), C(2), C(3)]).capproxEqual(C(-4, -6) / 3));

    assert(dispersion!(mean!float, "a * a", mean!float)([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(17.5 / 6));

    static assert(is(typeof(dispersion!(mean!float, "a ^^ 2", mean!float)([1, 2, 3])) == float));
}

/// Dispersion of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.dispersion.approxEqual(54.76562 / 12));
}

/// Dispersion of matrix
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

    assert(x.dispersion.approxEqual(54.76562 / 12));
}

/// Column dispersion of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0,  1.0, 1.5, 2.0],
        [3.5, 4.25, 2.0, 7.5],
        [5.0,  1.0, 1.5, 0.0]
    ].fuse;
    auto result = [13.16667 / 3, 7.041667 / 3, 0.1666667 / 3, 30.16667 / 3];

    // Use byDim or alongDim with map to compute dispersion of row/column.
    assert(x.byDim!1.map!dispersion.all!approxEqual(result));
    assert(x.alongDim!0.map!dispersion.all!approxEqual(result));

    // FIXME
    // Without using map, computes the dispersion of the whole slice
    // assert(x.byDim!1.dispersion == x.sliced.dispersion);
    // assert(x.alongDim!0.dispersion == x.sliced.dispersion);
}

/// Can also set functions to change type of dispersion that is used
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.stat.descriptive.univariate.quantiles: median;

    import mir.functional: naryFun;
    import mir.math.common: approxEqual, fabs, sqrt;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    alias square = naryFun!"a * a";

    // Other population variance examples
    assert(x.dispersion.approxEqual(54.76562 / 12));
    assert(x.dispersion!mean.approxEqual(54.76562 / 12));
    assert(x.dispersion!(mean, square).approxEqual(54.76562 / 12));
    assert(x.dispersion!(mean, square, mean).approxEqual(54.76562 / 12));

    // Population standard deviation
    assert(x.dispersion!(mean, square, mean).sqrt.approxEqual(sqrt(54.76562 / 12)));

    // Mean absolute deviation about the mean
    assert(x.dispersion!(mean, fabs, mean).approxEqual(21.0 / 12));
    //Mean absolute deviation about the median
    assert(x.dispersion!(median, fabs, mean).approxEqual(19.25000 / 12));
    //Median absolute deviation about the mean
    assert(x.dispersion!(mean, fabs, median).approxEqual(1.43750));
    //Median absolute deviation about the median
    assert(x.dispersion!(median, fabs, median).approxEqual(1.25000));
}

/++
For integral slices, pass output type to `centralTendency`, `transform`, and
`summary` functions as template parameter to ensure output type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.functional: naryFun;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    alias square = naryFun!"a * a";

    auto y = x.dispersion;
    assert(y.approxEqual(50.91667 / 12));
    static assert(is(typeof(y) == double));

    assert(x.dispersion!(mean!float, square, mean!float).approxEqual(50.91667 / 12));
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

    auto x = [C(1.0, 2), C(2.0, 3), C(3.0, 4), C(4.0, 5)].sliced;
    assert(x.dispersion.approxEqual(C(0.0, 10.0) / 4));
}

// std.complex test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.complex: complex;
    import std.math.operations: isClose;

    auto x = [complex(1.0, 2), complex(2, 3), complex(3, 4), complex(4, 5)].sliced;
    assert(x.dispersion.isClose(complex(0.0, 10.0) / 4));
}

/++
Dispersion works for complex numbers and other user-defined types (provided that
the `centralTendency`, `transform`, and `summary` functions are defined for those
types)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.complex: Complex;
    import std.math.operations: isClose;

    auto x = [Complex!double(1, 2), Complex!double(2, 3), Complex!double(3, 4), Complex!double(4, 5)].sliced;
    assert(x.dispersion.isClose(Complex!double(0, 10) / 4));
}

/// Compute mean tensors along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;

    assert(x.dispersion.approxEqual(17.5 / 6));

    auto m0 = [2.25, 2.25, 2.25];
    assert(x.alongDim!0.map!dispersion.all!approxEqual(m0));
    assert(x.alongDim!(-2).map!dispersion.all!approxEqual(m0));

    auto m1 = [2.0 / 3, 2.0 / 3];
    assert(x.alongDim!1.map!dispersion.all!approxEqual(m1));
    assert(x.alongDim!(-1).map!dispersion.all!approxEqual(m1));

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!dispersion.all!approxEqual(repeat(1800.0 / 2, 3, 4, 5)));
}

/// Arbitrary dispersion
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.functional: naryFun;
    import mir.math.common: approxEqual;

    alias square = naryFun!"a * a";

    assert(dispersion(1.0, 2, 3).approxEqual(2.0 / 3));
    assert(dispersion!(mean!float, square, mean!float)(1, 2, 3).approxEqual(2f / 3));
}

// UFCS UT
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    assert([1.0, 2, 3, 4].dispersion.approxEqual(5.0 / 4));
}

// Confirm type output is correct
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!dispersion;
    assert(y.all!approxEqual([0.25, 0.25]));
    static assert(is(meanType!(typeof(y)) == double));
}

// @nogc UT
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.dispersion.approxEqual(54.76562 / 12));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    assert(x.dispersion.approxEqual(54.76562 / 12));
}

/++
Calculates the coefficient of variation of the input.

The coefficient of variation is calculated by dividing either the population or
sample (default) standard deviation by the mean of the input. According to
wikipedia, "the coefficient of variation should be computed computed for data
measured on a ratio scale, that is, scales that have a meaningful zero and hence
allow for relative comparison of two measurements." In addition, for "small- and
moderately-sized datasets", the coefficient of variation is biased, even when
using the sample standard deviation.

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    F = controls type of output
    varianceAlgo = algorithm for calculating variance (default: VarianceAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The coefficient of varition of the input, must be floating point type

See_also:
    $(WEB en.wikipedia.org/wiki/Coefficient_of_variation, Coefficient of variation)
+/
template coefficientOfVariation(
    F,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import mir.math.common: sqrt;
    import mir.math.sum: ResolveSummationType;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population variance, false if sample variance (default)
    +/
    @fmamath stdevType!F coefficientOfVariation(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias G = typeof(return);
        auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, Range, G))(r.move);
        assert(varianceAccumulator.mean!G > 0, "coefficientOfVariation: mean must be larger than zero");
        return varianceAccumulator.variance!G(isPopulation).sqrt / varianceAccumulator.mean!G;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F coefficientOfVariation(scope const F[] ar...)
    {
        alias G = typeof(return);
        auto varianceAccumulator = VarianceAccumulator!(G, varianceAlgo, ResolveSummationType!(summation, const(G)[], G))(ar);
        assert(varianceAccumulator.mean!G > 0, "coefficientOfVariation: mean must be larger than zero");
        return varianceAccumulator.variance!G(false).sqrt / varianceAccumulator.mean!G;
    }
}

/// ditto
template coefficientOfVariation(
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population variance, false if sample variance (default)
    +/
    @fmamath stdevType!Range coefficientOfVariation(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .coefficientOfVariation!(F, varianceAlgo, summation)(r.move, isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T coefficientOfVariation(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .coefficientOfVariation!(F, varianceAlgo, summation)(ar);
    }
}

///
template coefficientOfVariation(F, string varianceAlgo, string summation = "appropriate")
{
    mixin("alias coefficientOfVariation = .coefficientOfVariation!(F, VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template coefficientOfVariation(string varianceAlgo, string summation = "appropriate")
{
    mixin("alias coefficientOfVariation = .coefficientOfVariation!(VarianceAlgo." ~ varianceAlgo ~ ", Summation." ~ summation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(coefficientOfVariation([1.0, 2, 3]).approxEqual(1.0 / 2.0));
    assert(coefficientOfVariation([1.0, 2, 3], true).approxEqual(0.816497 / 2.0));

    assert(coefficientOfVariation!float([0, 1, 2, 3, 4, 5].sliced(3, 2)).approxEqual(1.870829 / 2.5));

    static assert(is(typeof(coefficientOfVariation!float([1, 2, 3])) == float));
}

/// Coefficient of variation of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.coefficientOfVariation.approxEqual(2.231299 / 2.437500));
}

/// Coefficient of variation of matrix
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

    assert(x.coefficientOfVariation.approxEqual(2.231299 / 2.437500));
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

    auto x = a + 1_000_000_000;

    auto y = x.coefficientOfVariation;
    assert(y.approxEqual(2.231299 / 1_000_000_002.437500));

    // The naive variance algorithm is numerically unstable in this case, but
    // the difference is small as coefficientOfVariation is a ratio
    auto z0 = x.coefficientOfVariation!"naive";
    assert(!z0.approxEqual(y, 0x1p-20f, 0x1p-30f));

    // But the two-pass algorithm provides a consistent answer
    auto z1 = x.coefficientOfVariation!"twoPass";
    assert(z1.approxEqual(y));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: mean;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    // Set population standard deviation, standardDeviation algorithm, sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto x = a * 10_000;

    bool populationTrue = true;

    /++
    For this case, failing to use a summation algorithm results in an assert
    error because the mean is zero due to floating point precision issues.
    +/
    //assert(x.coefficientOfVariation!("online").approxEqual(8.164966e103 / 0.0));

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.coefficientOfVariation!("online", "kbn").approxEqual(8.164966e103 / 5000.0));
    assert(x.coefficientOfVariation!("online", "kb2").approxEqual(8.164966e103 / 5000.0));
    assert(x.coefficientOfVariation!("online", "precise").approxEqual(8.164966e103 / 5000.0));
    assert(x.coefficientOfVariation!(double, "online", "precise").approxEqual(8.164966e103 / 5000.0));
    assert(x.coefficientOfVariation!(double, "online", "precise")(populationTrue).approxEqual(7.071068e103 / 5000.0));


    auto y = [uint.max - 2, uint.max - 1, uint.max].sliced;
    auto z = y.coefficientOfVariation!ulong;
    assert(z == (1.0 / (cast(double) uint.max - 1)));
    static assert(is(typeof(z) == double));
    assert(y.coefficientOfVariation!(ulong, "online") == (1.0 / (cast(double) uint.max - 1)));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.coefficientOfVariation;
    assert(y.approxEqual(2.151462f / 2.416667));
    static assert(is(typeof(y) == double));

    assert(x.coefficientOfVariation!float.approxEqual(2.151462f / 2.416667));
}

/++
coefficientOfVariation works for other user-defined types (provided they
can be converted to a floating point)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    static struct Foo {
        float x;
        alias x this;
    }

    Foo[] foo = [Foo(1f), Foo(2f), Foo(3f)];
    assert(foo.coefficientOfVariation.approxEqual(1f / 2f));
}

/// Arbitrary coefficientOfVariation
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(coefficientOfVariation(1.0, 2, 3).approxEqual(1.0 / 2.0));
    assert(coefficientOfVariation!float(1, 2, 3).approxEqual(1f / 2f));
}

// Dynamic array / UFCS
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert(coefficientOfVariation([1.0, 2, 3, 4]).approxEqual(1.290994 / 2.50));
    assert([1.0, 2, 3, 4].coefficientOfVariation.approxEqual(1.290994 / 2.50));
}

// Check type of alongDim result
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!coefficientOfVariation;
    assert(y.all!approxEqual([0.707107 / 1.50, 0.707107 / 3.50]));
    static assert(is(meanType!(typeof(y)) == double));
}

// @nogc test
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.coefficientOfVariation.approxEqual(2.231299 / 2.437500));
    assert(x.sliced.coefficientOfVariation!float.approxEqual(2.231299 / 2.437500));
}

// Copies can share a cursor even when save provides an independent traversal.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    import std.range.primitives: isForwardRange, isInputRange;

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
        static assert(isForwardRange!(Input!(T, true)));
        static assert(isInputRange!(Input!(T, false)));
        static assert(!isForwardRange!(Input!(T, false)));
        static foreach (algo; [VarianceAlgo.twoPass, VarianceAlgo.hybrid])
        {{
            size_t[1] cursor;
            auto result = VarianceAccumulator!(T, algo, Summation.naive)(Input!(T, true)(values[], cursor[]));
            assert(result.count == 3 && result.mean == 2);
            assert(result.variance(true).approxEqual(T(14) / 3));
            assert(cursor[0] == values.length);
        }}
        size_t[1] cursor;
        auto single = Input!(T, false)(values[], cursor[]);
        alias TwoPass = VarianceAccumulator!(T, VarianceAlgo.twoPass, Summation.naive);
        static assert(!__traits(compiles, TwoPass(single)));
        alias Hybrid = VarianceAccumulator!(T, VarianceAlgo.hybrid, Summation.naive);
        auto result = Hybrid(single);
        assert(result.count == 3 && result.mean == 2);
        assert(result.variance(true).approxEqual(T(14) / 3));
        assert(cursor[0] == values.length);
        cursor[0] = 0;
        Hybrid appended;
        appended.put(T(-2));
        appended.put(single);
        assert(appended.count == 4 && appended.mean == 1);
        assert(appended.variance(true).approxEqual(T(6.5)));
        assert(cursor[0] == values.length);
    }}
}

// Retain low-order square sums when merging compensated accumulators.
version(mir_stat_test)
pure nothrow
unittest
{
    static foreach (mode; [Summation.kbn, Summation.kb2, Summation.precise])
    {
        static foreach (algo; [VarianceAlgo.naive, VarianceAlgo.assumeZeroMean])
        {{
            alias A = VarianceAccumulator!(double, algo, mode);
            A source, result;
            source.put(0x1p27);
            source.put(1.0);
            result.put(1.0);
            result.put(1.0);
            result.put(source);
            // 2^54 + 3 rounds to 2^54 + 4; scalarizing the source loses 1.
            static if (algo == VarianceAlgo.naive)
                assert(result.sumOfSquares == 0x1p54 + 4);
            else
                assert(result.centeredSumOfSquares == 0x1p54 + 4);
            assert(result.count == 4 && source.count == 2);
        }}
        static foreach (algo; [VarianceAlgo.online, VarianceAlgo.hybrid])
        {{
            alias A = VarianceAccumulator!(double, algo, mode);
            alias B = VarianceAccumulator!(double, VarianceAlgo.twoPass, mode);
            auto source = B([0x1p27, -0x1p27, 1.0, -1.0]);
            A result;
            result.put([1.0, -1.0, 1.0, -1.0]);
            result.put(source);
            // Exact M2 is 2^55 + 6, rounded to 2^55 + 8.
            assert(result.centeredSumOfSquares == 0x1p55 + 8);
            assert(result.count == 8 && result.mean == 0);
            A copied;
            copied.put(source);
            copied.put([1.0, -1.0, 1.0, -1.0]);
            assert(copied.centeredSumOfSquares == result.centeredSumOfSquares);
            result.put(result);
            assert(result.count == 16);
            assert(result.centeredSumOfSquares == 0x1p56 + 16);
            assert(source.count == 4 && source.centeredSumOfSquares == 0x1p55);

            // A raw-square source must still be converted to centered squares.
            auto raw = VarianceAccumulator!(double, VarianceAlgo.naive, mode)([2.0, 4.0]);
            A shifted;
            shifted.put([6.0, 8.0]);
            shifted.put(raw);
            assert(shifted.mean == 5 && shifted.centeredSumOfSquares == 20);
            A centered;
            centered.put([2.0, 4.0]);
            A other;
            other.put([6.0, 8.0]);
            centered.put(other);
            assert(centered.mean == 5 && centered.centeredSumOfSquares == 20);
            A empty;
            centered.put(empty);
            assert(centered.count == 4 && centered.centeredSumOfSquares == 20);
        }}
    }
}

/++
Choose what the weights mean and whether to apply a sample correction.

Use population to describe the spread of the weighted distribution you have.
Use frequencySample when each weight counts actual observations in a sample.
Use relativeSample when each row is one observation, weights specify relative
emphasis, and you want a sample correction under the assumptions below.

For example, values [0, 10] with weights [1, 3] have the same weighted mean,
7.5, in all three modes. Their variances differ: 18.75 for population, 25 for
frequencySample, and 50 for relativeSample. These answer different questions;
the larger answer is not automatically more accurate.

Select the meaning explicitly. Integer weights do not automatically mean
frequencies, and floating-point weights such as [1.0, 3.0] can be frequencies.
See the examples on $(LREF weightedVariance) for executable calls, and
$(REF descriptive, mir, stat) for shared guidance on weight meanings.
+/
enum WeightedVarianceNormalization
{
    /++
    Describe the spread of a weighted distribution, without a sample correction.
    This is the default.

    Use this for a probability distribution, a complete weighted dataset, or a
    descriptive summary when you do not want to estimate a population variance
    from a sample. For example, an outcome is 0 with probability 25 percent and
    10 with probability 75 percent. Its variance is 18.75. Either [0.25, 0.75]
    or [1, 3] expresses those probabilities; weights need not sum to one.

    Multiplying every weight by the same positive constant leaves the result
    unchanged. One positive-weight observation is sufficient and has variance 0.

    Formula: sum(w*(x-mean)*(x-mean)) / sum(w).
    +/
    population,
    /++
    Estimate variance from a sample stored as values and occurrence counts.

    Use this to avoid expanding repeated observations. Values [0, 10] and counts
    [1, 3] represent the four observations [0, 10, 10, 10]. The result is 25,
    exactly the sample-variance definition used by ordinary variance on that
    expanded sample (up to floating-point rounding).

    Counts must be nonnegative whole numbers, and their total must exceed one.
    Fractional counts such as 1.5 are invalid. A single stored row with count 3
    represents three identical observations and has sample variance 0.

    Do not normalize counts to sum to one. Changing [1, 3] to [2, 6] means an
    eight-observation sample, so its sample correction changes: the variance
    becomes 150/7, approximately 21.43. This mode applies the familiar division
    by sample size minus one, using total count as the sample size.

    Formula: sum(w*(x-mean)*(x-mean)) / (sum(w)-1).
    +/
    frequencySample,
    /++
    Apply a sample correction when weights give observations relative emphasis.

    Use this when each row is one independent observation and weights are fixed
    independently of the observed values. For example, two observations [0, 10]
    have preassigned weights [1, 3]: the second receives three times the emphasis,
    but it is still one observation, not three repeated observations. The
    corrected variance is 50. Weights [0.25, 0.75] give the same result.

    At least two rows must have positive weights. Fractional weights are allowed,
    and multiplying all weights by the same positive constant leaves the result
    unchanged. Equal positive weights recover ordinary sample variance. Unequal
    weights concentrate the information in fewer observations, which this
    correction accounts for under its statistical assumptions.

    The correction is unbiased when observations are independent, share a common
    mean and variance, and weights are fixed. It is not a general correction for
    survey sampling or measurements with different known variances. It estimates
    the variance of the observations, not the uncertainty of their weighted mean.

    Formula: write W = sum(w), Q = sum(w*w), and
    M2 = sum(w*(x-mean)*(x-mean)); the result is M2/(W-Q/W).
    +/
    relativeSample
}
// Traverse paired inputs by logical index, independently of storage strides.
// Save forward ranges at every nesting level; single-pass ranges are consumed.
private import mir.primitives: DeepElementType;

private void weightedVarianceEach(Sink, X, W)(ref Sink sink, scope X x, scope W w)
{
    import mir.ndslice.slice: toSlice;
    import std.range.primitives: isInputRange, isForwardRange;
    static if (isConvertibleToSlice!X && !isSlice!X)
        weightedVarianceEach(sink, x.toSlice, w);
    else static if (isConvertibleToSlice!W && !isSlice!W)
        weightedVarianceEach(sink, x, w.toSlice);
    else static if (isSlice!X && isSlice!W)
    {
        static assert(X.N == W.N, "weighted variance: ranks must match");
        assert(x.shape == w.shape, "weighted variance: shapes must match");
        foreach (i; 0 .. x.length)
            weightedVarianceEach(sink, x[i], w[i]);
    }
    else static if (isInputRange!X && isInputRange!W)
    {
        static if (isForwardRange!X) x = x.save;
        static if (isForwardRange!W) w = w.save;
        while (!x.empty && !w.empty)
        {
            weightedVarianceEach(sink, x.front, w.front);
            x.popFront;
            w.popFront;
        }
        assert(x.empty && w.empty, "weighted variance: lengths must match");
    }
    else
        sink.put(x, w);
}

private template weightedVarianceReplayable(R)
{
    import mir.ndslice.slice: toSlice;
    import std.range.primitives: isInputRange, isForwardRange, ElementType;
    static if (isConvertibleToSlice!R && !isSlice!R)
        enum weightedVarianceReplayable = weightedVarianceReplayable!(typeof(R.init.toSlice));
    else static if (isSlice!R)
        enum weightedVarianceReplayable = weightedVarianceReplayable!(typeof(R.init[0]));
    else static if (isInputRange!R)
        enum weightedVarianceReplayable = isForwardRange!R && weightedVarianceReplayable!(ElementType!R);
    else
        enum weightedVarianceReplayable = true;
}

/++
Mergeable weighted variance accumulator for real-valued observations.

Weights must be finite and nonnegative. Zero weights exclude observations,
including nonfinite observations. Positive-weight observations must be finite.
Frequency-sample weights must additionally be whole numbers, with total no
greater than 2^^T.mant_dig so that counts are represented exactly. Preconditions
are checked by assertions. Empty insertion and merging an empty accumulator are
no-ops; reading variance or mean from an empty accumulator is invalid.

Weights are scaled internally by their maximum, avoiding overflow in squared
weights and preserving common-scale invariance over a wide dynamic range.
Positive weights must remain positive and finite when converted to T. Ratios,
centered differences, and moment arithmetic are still subject to T's floating
point range and rounding. Very small relative weights can underflow. Subtract a
common integer offset before insertion when conversion to T would lose data.

online uses weighted centered updates. twoPass uses a corrected centered second
pass for batches and requires replayable inputs. hybrid uses twoPass when inputs
are replayable and online otherwise. Scalar insertion always uses online updates.
All modes support merging through put(other). The mean uses a recentered leading
value and a trailing correction to preserve small differences as it moves.
Centered moments carry a separate residual scale: standard deviation can remain
representable even when variance overflows or underflows. Weight ratios and
normalized residuals can still underflow at extreme relative dynamic ranges.

The summation policy applies to accumulated weights and moments; appropriate
selects pairwise summation for the batch functions. Rescaling and
merging transfer rounded sums, so low-order summator corrections are not preserved
across those operations. Results need not be bitwise identical across partitions.

Unlike wmean, signed weights are not accepted and zero weights exclude values.
+/
struct WVarianceAccumulator(T,
    WeightedVarianceNormalization normalization = WeightedVarianceNormalization.population,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.kbn)
    if (isFloatingPoint!T && isMutable!T &&
        (varianceAlgo == VarianceAlgo.online || varianceAlgo == VarianceAlgo.twoPass ||
         varianceAlgo == VarianceAlgo.hybrid))
{
    private alias Sum = Summator!(T, summation);
    private Sum mass, squares, pairs, moment2;
    private T scale = 0;
    private T origin = 0;
    private T offset = 0;
    private T momentScale = 0;
    private real frequency = 0;
    private size_t positiveCount;

    private static void multiply(ref Sum s, T factor)
    {
        T value = cast(T) s.sum * factor;
        // Scalar assignment resets the summator without copying its storage.
        s = value;
    }

    private void rescale(T newScale)
    {
        if (scale == 0) { scale = newScale; return; }
        if (newScale <= scale) return;
        T ratio = scale / newScale;
        multiply(mass, ratio);
        multiply(moment2, ratio);
        // Two multiplications avoid underflow in ratio*ratio before scaling.
        multiply(squares, ratio);
        multiply(squares, ratio);
        multiply(pairs, ratio);
        multiply(pairs, ratio);
        scale = newScale;
    }

    // Keep the mean as a leading value plus a trailing correction. Recenter
    // after each update so a distant initial observation cannot fix the origin.
    private void shiftMean(T shift)
    {
        T leading = origin + shift;
        T roundedShift = leading - origin;
        T trailing = (origin - (leading - roundedShift)) + (shift - roundedShift);
        trailing += offset;
        origin = leading + trailing;
        T roundedTrailing = origin - leading;
        offset = (leading - (origin - roundedTrailing)) + (trailing - roundedTrailing);
    }

    // Store M2 / momentScale^^2. The scale is a residual magnitude rather
    // than its square, so standard deviation can survive unrepresentable M2.
    private void addMoment(T delta, T coefficient)
    {
        import std.math: abs;
        if (delta == 0 || coefficient == 0) return;
        T magnitude = abs(delta);
        if (magnitude > momentScale)
        {
            T ratio = momentScale / magnitude;
            multiply(moment2, ratio);
            multiply(moment2, ratio);
            momentScale = magnitude;
        }
        T normalized = delta / momentScale;
        moment2.put((normalized * coefficient) * normalized);
    }

    private void mergeMoment(T value, T valueScale)
    {
        if (valueScale == 0 || value == 0) return;
        if (valueScale > momentScale)
        {
            T ratio = momentScale / valueScale;
            multiply(moment2, ratio);
            multiply(moment2, ratio);
            momentScale = valueScale;
        }
        T ratio = valueScale / momentScale;
        moment2.put((value * ratio) * ratio);
    }
    /// Insert one observation with unit weight.
    void put(X)(X x) if (is(X : T)) { put(x, 1); }

    /// Insert one observation and its weight.
    void put(X, W)(scope X x, scope W w) if (is(X : T) && is(W : real))
    {
        import std.math: isFinite, floor;
        const raw = cast(real) w;
        assert(isFinite(raw) && raw >= 0, "weighted variance: weights must be finite and nonnegative");
        if (raw == 0) return;
        const weight = cast(T) w;
        assert(isFinite(weight) && weight > 0, "weighted variance: weight must fit accumulator type");
        const value = cast(T) x;
        assert(isFinite(value), "weighted variance: positive-weight observations must be finite");
        static if (normalization == WeightedVarianceNormalization.frequencySample)
        {
            enum real maxFrequency = 2.0L ^^ T.mant_dig;
            assert(raw == floor(raw) && raw <= maxFrequency - frequency,
                "weighted variance: frequencies must be whole numbers with an exactly representable total");
            frequency += raw;
        }
        rescale(weight);
        T a = cast(T) mass.sum;
        T b = weight / scale;
        mass.put(b);
        squares.put(b * b);
        pairs.put(a * b);
        if (positiveCount == 0)
            origin = value;
        else
        {
            T delta = (value - origin) - offset;
            T total = cast(T) mass.sum;
            addMoment(delta, (a / total) * b);
            shiftMean(delta * (b / total));
        }
        ++positiveCount;
    }

    private struct CenteredPass
    {
        T origin, offset, scale;
        T momentScale = 0;
        Sum residuals, centered;
        void put(A, B)(A a, B b)
        {
            import std.math: abs;
            if (b == 0) return;
            T weight = cast(T) b / scale;
            // An underflowed mass must not change the residual scale.
            if (weight == 0) return;
            T delta = (cast(T) a - origin) - offset;
            if (delta == 0) return;
            T magnitude = abs(delta);
            if (magnitude > momentScale)
            {
                T ratio = momentScale / magnitude;
                multiply(residuals, ratio);
                multiply(centered, ratio);
                multiply(centered, ratio);
                momentScale = magnitude;
            }
            T normalized = delta / momentScale;
            residuals.put(weight * normalized);
            centered.put((weight * normalized) * normalized);
        }
    }
    /// Insert arrays, slices, or equally shaped nested/input ranges.
    void put(X, W)(scope X x, scope W w) if (isIterable!X && isIterable!W)
    {
        WVarianceAccumulator batch;
        weightedVarianceEach(batch, x, w);
        enum replayable = weightedVarianceReplayable!X && weightedVarianceReplayable!W;
        static if (varianceAlgo == VarianceAlgo.twoPass)
            static assert(replayable, "weighted variance: twoPass requires replayable inputs");
        static if (varianceAlgo != VarianceAlgo.online && replayable)
        {
            if (batch.positiveCount)
            {
                CenteredPass second;
                second.origin = batch.origin;
                second.offset = batch.offset;
                second.scale = batch.scale;
                weightedVarianceEach(second, x, w);
                T residual = cast(T) second.residuals.sum;
                T total = cast(T) batch.mass.sum;
                T corrected = cast(T) second.centered.sum - (residual / total) * residual;
                // A tiny negative result can arise from the correction's rounding.
                batch.moment2 = corrected < 0 ? T(0) : corrected;
                batch.momentScale = second.momentScale;
                batch.shiftMean((residual / total) * second.momentScale);
            }
        }
        put(batch);
    }

    /// Merge a partition with the same type, normalization, algorithm and summation.
    void put(scope const ref WVarianceAccumulator other)
    {
        if (other.positiveCount == 0) return;
        // Snapshot scalar state before mutation, including for self-merges.
        const otherCount = other.positiveCount;
        const otherFrequency = other.frequency;
        const otherOrigin = other.origin;
        const otherOffset = other.offset;
        const otherScale = other.scale;
        const otherMomentScale = other.momentScale;
        T otherMass = cast(T) other.mass.sum;
        T otherSquares = cast(T) other.squares.sum;
        T otherPairs = cast(T) other.pairs.sum;
        T otherMoment = cast(T) other.moment2.sum;
        if (positiveCount == 0)
        {
            positiveCount = otherCount;
            frequency = otherFrequency;
            origin = otherOrigin;
            offset = otherOffset;
            scale = otherScale;
            mass.put(otherMass);
            squares.put(otherSquares);
            pairs.put(otherPairs);
            moment2.put(otherMoment);
            momentScale = otherMomentScale;
            return;
        }
        static if (normalization == WeightedVarianceNormalization.frequencySample)
        {
            enum real maxFrequency = 2.0L ^^ T.mant_dig;
            assert(otherFrequency <= maxFrequency - frequency,
                "weighted variance: merged frequency total is too large");
            frequency += otherFrequency;
        }
        T delta = (otherOrigin - origin) + (otherOffset - offset);
        rescale(otherScale);
        T ratio = otherScale / scale;
        T a = cast(T) mass.sum;
        T b = otherMass * ratio;
        mass.put(b);
        T total = cast(T) mass.sum;
        squares.put((otherSquares * ratio) * ratio);
        pairs.put((otherPairs * ratio) * ratio);
        pairs.put(a * b);
        mergeMoment(otherMoment * ratio, otherMomentScale);
        addMoment(delta, (a / total) * b);
        shiftMean(delta * (b / total));
        positiveCount += otherCount;
    }
    /// Number of positive-weight input rows, not the sum of frequency counts.
    size_t count() const @property { return positiveCount; }

    /// Total weight. May overflow even when normalized variance is representable.
    T weight() const @property { return scale * cast(T) mass.sum; }

    /// Weighted mean; requires at least one positive-weight observation.
    T mean() const @property
    {
        assert(positiveCount, "weighted variance: no positive-weight observations");
        return origin + offset;
    }

    /// Kish weight-concentration diagnostic W*W/Q, not a universal degrees of freedom.
    T effectiveSampleSize() const @property
    {
        assert(positiveCount, "weighted variance: no positive-weight observations");
        T total = cast(T) mass.sum;
        return total / (cast(T) squares.sum / total);
    }

    private T varianceDenominator() const
    {
        assert(positiveCount, "weighted variance: no positive-weight observations");
        T total = cast(T) mass.sum;
        static if (normalization == WeightedVarianceNormalization.population)
            T denominator = total;
        else static if (normalization == WeightedVarianceNormalization.frequencySample)
        {
            assert(frequency > 1, "weighted variance: total frequency must exceed one");
            T denominator = cast(T) ((frequency - 1) / scale);
        }
        else
        {
            static assert(normalization == WeightedVarianceNormalization.relativeSample);
            // W-Q/W = 2*sum(i<j, wi*wj)/W. Avoid cancellation near nEff=1.
            T denominator = 2 * (cast(T) pairs.sum / total);
        }
        assert(denominator > 0, "weighted variance: insufficient effective weight");
        return denominator;
    }

    /// Variance with the normalization selected by the template argument.
    T variance() const @property
    {
        T normalized = cast(T) moment2.sum / varianceDenominator();
        return (normalized * momentScale) * momentScale;
    }

    /// Standard deviation calculated without first forming the unscaled variance.
    T standardDeviation() const @property
    {
        import std.math: sqrt;
        T denominator = varianceDenominator();
        return (sqrt(cast(T) moment2.sum) / sqrt(denominator)) * momentScale;
    }
}
/++
Weighted variance of paired values and weights, without materializing the inputs.

Choose the normalization according to what your weights represent:
$(UL
    $(LI $(D population), the default: describe the spread of the weighted
        distribution itself, for example outcomes with known probabilities.)
    $(LI $(D frequencySample): each weight is the number of times that value
        occurred in a sample. Apply the same sample correction as if you had
        listed every observation separately.)
    $(LI $(D relativeSample): each row is one observation with a fixed relative
        emphasis. Apply the sample correction for independent observations with
        a common mean and variance.)
)
See $(LREF WeightedVarianceNormalization) for when to choose each option,
rescaling behavior and assumptions. All modes require finite, nonnegative
weights and ignore zero-weight observations, even if their values are NaN.
See $(LREF WVarianceAccumulator) for numerical limits and streaming use.

Examples:
--------------------
import mir.stat.descriptive.univariate.dispersions;
import mir.math.common: approxEqual;

alias N = WeightedVarianceNormalization;
auto values = [0.0, 10.0];
auto weights = [1.0, 3.0];

// A distribution with 25% of its mass at 0 and 75% at 10.
assert(weightedVariance(values, weights).approxEqual(18.75));

// Four sampled observations: [0, 10, 10, 10].
assert(weightedVariance!(N.frequencySample)(values, weights).approxEqual(25));

// Two sampled observations, with preassigned relative emphasis 1:3.
assert(weightedVariance!(N.relativeSample)(values, weights).approxEqual(50));

// Relative weights may be normalized; frequency counts must not be.
auto probabilities = [0.25, 0.75];
assert(weightedVariance(values, probabilities).approxEqual(18.75));
assert(weightedVariance!(N.relativeSample)(values, probabilities).approxEqual(50));

// Standard deviation uses the same choice and returns the original data units.
assert(weightedStandardDeviation!(N.frequencySample)(values, weights).approxEqual(5));

// Use the same policy when observations arrive one at a time.
WVarianceAccumulator!(double, N.frequencySample) accumulator;
accumulator.put(0.0, 1);
accumulator.put(10.0, 3);
assert(accumulator.variance.approxEqual(25));
--------------------

Input arrays, slices (including strided tensors), and ranges must have matching
shapes/lengths. Integral observations infer double; floating observations retain
their type. F explicitly selects a floating result/accumulator type. Weights are
converted to that type. No input is mutated; single-pass cursors are consumed.
+/
template weightedVariance(F,
    WeightedVarianceNormalization normalization = WeightedVarianceNormalization.population,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    stdevType!F weightedVariance(X, W)(scope X values, scope W weights)
        if (isIterable!X && isIterable!W)
    {
        alias T = stdevType!F;
        WVarianceAccumulator!(T, normalization, varianceAlgo,
            ResolveSummationType!(summation, T[], T)) accumulator;
        accumulator.put(values, weights);
        return accumulator.variance;
    }
}

/// ditto
template weightedVariance(
    WeightedVarianceNormalization normalization = WeightedVarianceNormalization.population,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    stdevType!(DeepElementType!X) weightedVariance(X, W)(scope X values, scope W weights)
        if (isIterable!X && isIterable!W)
    {
        return .weightedVariance!(stdevType!(DeepElementType!X), normalization, varianceAlgo, summation)(values, weights);
    }
}

/++
Weighted spread expressed in the same units as the observations.

This is the square root of $(LREF weightedVariance). For example, if observations
are durations in seconds, variance is in squared seconds and standard deviation
is in seconds, which is often easier to interpret. Scaled centered moments allow a finite standard deviation even when the variance
would overflow or underflow. The frequency-count example
[0, 10] with counts [1, 3] has variance 25 and standard deviation 5.

Choose population, frequencySample or relativeSample for the same reasons as in
$(LREF WeightedVarianceNormalization). Taking the square root does not change
what the weights mean. The same input requirements and numerical limits apply.
The sample correction is a correction for variance; taking its square root does
not in general produce an unbiased estimator of population standard deviation.
See $(LREF weightedVariance) for executable examples.
+/
template weightedStandardDeviation(F,
    WeightedVarianceNormalization normalization = WeightedVarianceNormalization.population,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    stdevType!F weightedStandardDeviation(X, W)(scope X values, scope W weights)
        if (isIterable!X && isIterable!W)
    {
        alias T = stdevType!F;
        WVarianceAccumulator!(T, normalization, varianceAlgo,
            ResolveSummationType!(summation, T[], T)) accumulator;
        accumulator.put(values, weights);
        return accumulator.standardDeviation;
    }
}

/// ditto
template weightedStandardDeviation(
    WeightedVarianceNormalization normalization = WeightedVarianceNormalization.population,
    VarianceAlgo varianceAlgo = VarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    stdevType!(DeepElementType!X) weightedStandardDeviation(X, W)(scope X values, scope W weights)
        if (isIterable!X && isIterable!W)
    {
        return .weightedStandardDeviation!(stdevType!(DeepElementType!X), normalization, varianceAlgo, summation)(values, weights);
    }
}


/// The same data has different sample corrections under different weight meanings.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias N = WeightedVarianceNormalization;
    double[2] x = [0, 10];
    double[2] w = [1, 3];
    assert(weightedVariance(x[], w[]).approxEqual(18.75));
    assert(weightedVariance!(N.frequencySample)(x[], w[]).approxEqual(25));
    assert(weightedVariance!(N.relativeSample)(x[], w[]).approxEqual(50));
    assert(weightedStandardDeviation!(N.frequencySample)(x[], w[]).approxEqual(5));
    WVarianceAccumulator!(double, N.relativeSample) a;
    a.put(x[], w[]);
    assert(a.count == 2 && a.weight == 4);
    assert(a.mean.approxEqual(7.5));
    assert(a.effectiveSampleSize.approxEqual(1.6));
    assert(a.standardDeviation.approxEqual(weightedStandardDeviation!(N.relativeSample)(x[], w[])));
}

// Equal weights, expanded frequencies, split/merge and batch/stream agreement.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    alias N = WeightedVarianceNormalization;
    double[4] x = [1, 2, 4, 8];
    int[4] w = [1, 3, 2, 1];
    double[7] expanded = [1, 2, 2, 2, 4, 4, 8];
    double[4] ones = [1, 1, 1, 1];
    static foreach (algo; [VarianceAlgo.online, VarianceAlgo.twoPass, VarianceAlgo.hybrid])
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        assert((weightedVariance!(N.population, algo, mode)(x[], ones[])).approxEqual(variance(x[], true)));
        assert((weightedVariance!(N.relativeSample, algo, mode)(x[], ones[])).approxEqual(variance(x[])));
        assert((weightedVariance!(N.frequencySample, algo, mode)(x[], w[])).approxEqual(variance(expanded[])));
        static foreach (norm; [N.population, N.frequencySample, N.relativeSample])
        {{
            alias A = WVarianceAccumulator!(double, norm, algo, mode);
            A batch, stream, left, right, empty;
            batch.put(x[], w[]);
            foreach (i; 0 .. x.length) stream.put(x[i], w[i]);
            left.put(x[0 .. 2], w[0 .. 2]);
            right.put(x[2 .. $], w[2 .. $]);
            left.put(empty);
            left.put(right);
            assert(left.variance.approxEqual(batch.variance));
            assert(stream.variance.approxEqual(batch.variance));
            assert(left.mean.approxEqual(batch.mean));
            assert(left.effectiveSampleSize.approxEqual(batch.effectiveSampleSize));
            empty.put(empty);
            empty.put(batch);
            assert(empty.variance == batch.variance);
        }}
    }}
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[2] v = [1, 1 + T.epsilon];
        T[2] equal = [1, 1];
        auto expected = T.epsilon * T.epsilon / 2;
        assert((weightedVariance!(N.relativeSample)(v[], equal[])).approxEqual(expected, T(1e-5), T(0)));
        static assert(is(typeof(weightedVariance(v[], equal[])) == T));
    }}
    static assert(is(typeof(weightedVariance(w[], w[])) == double));
    static assert(is(typeof(weightedVariance!float(x[], w[])) == float));
}

// Scale invariance, a dominant weight, zero-weight exclusion, and large offsets.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias N = WeightedVarianceNormalization;
    double[2] x = [0, 10];
    static foreach (factor; [1e-300, 1.0, 1e300])
    {{
        double[2] w = [factor, 3 * factor];
        assert(weightedVariance(x[], w[]).approxEqual(18.75));
        assert((weightedVariance!(N.relativeSample)(x[], w[])).approxEqual(50));
    }}
    double[2] unequal = [1, 1e-20];
    assert((weightedVariance!(N.relativeSample)(x[], unequal[])).approxEqual(50));
    double[3] nonfinite = [double.nan, 5, double.infinity];
    double[3] masked = [0, 2, 0];
    assert(weightedVariance(nonfinite[], masked[]) == 0);
    assert((weightedVariance!(N.frequencySample)(nonfinite[], masked[])) == 0);
    double[4] shifted = [1e12 + 1, 1e12 + 2, 1e12 + 4, 1e12 + 8];
    double[4] small = [1, 2, 4, 8];
    double[4] weights = [1, 3, 2, 1];
    assert(weightedVariance(shifted[], weights[]).approxEqual(weightedVariance(small[], weights[]), 1e-14, 0));
    WVarianceAccumulator!double a, b;
    a.put(shifted[0 .. 2], weights[0 .. 2]);
    b.put(shifted[2 .. $], weights[2 .. $]);
    a.put(b);
    assert(a.variance.approxEqual(weightedVariance(small[], weights[]), 1e-14, 0));
}

// Paired traversal preserves independent tensor strides and range lengths.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.math.common: approxEqual;
    import std.algorithm: filter;
    double[6] x = [1, 2, 3, 4, 5, 6];
    double[6] storageWeights = [1, 4, 2, 5, 3, 6];
    double[6] flatWeights = [1, 2, 3, 4, 5, 6];
    const expected = weightedVariance(x[], flatWeights[]);
    assert(weightedVariance(x[].sliced(2, 3), storageWeights[].sliced(3, 2).transposed).approxEqual(expected));
    assert(weightedVariance(x[].filter!(a => true), flatWeights[]).approxEqual(expected));
    assert(weightedVariance(x[].filter!(a => true), flatWeights[].filter!(a => true)).approxEqual(expected));
    const double[6] immutableValues = x;
    assert(weightedVariance(immutableValues[], flatWeights[]).approxEqual(expected));

}

version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.algorithm: filter;
    alias N = WeightedVarianceNormalization;
    double[2] x = [0, 10];
    double[2] w = [1, 3];
    foreach (bad; [-1.0, double.nan, double.infinity])
    {
        auto invalid = w;
        invalid[0] = bad;
        assertThrown!AssertError(weightedVariance(x[], invalid[]));
    }
    double[2] zero = [0, 0];
    assertThrown!AssertError(weightedVariance(x[], zero[]));
    assertThrown!AssertError(weightedVariance(x[0 .. 0], w[0 .. 0]));
    assertThrown!AssertError(weightedVariance(x[], w[0 .. 1]));
    assertThrown!AssertError(weightedVariance(x[].filter!(a => true), w[0 .. 1].filter!(a => true)));
    assertThrown!AssertError(weightedVariance(x[0 .. 1].filter!(a => true), w[].filter!(a => true)));
    assertThrown!AssertError(weightedVariance!(N.relativeSample)(x[0 .. 1], w[0 .. 1]));
    assertThrown!AssertError(weightedVariance!(N.frequencySample)(x[0 .. 1], w[0 .. 1]));
    double[2] fractional = [1, 1.5];
    assertThrown!AssertError(weightedVariance!(N.frequencySample)(x[], fractional[]));
    WVarianceAccumulator!(float, N.frequencySample) counts;
    assertThrown!AssertError(counts.put(1, (1u << 24) + 1u));
    counts.put(1, 1u << 24);
    assertThrown!AssertError(counts.put(2, 1));
    auto badValues = x;
    badValues[0] = double.nan;
    assertThrown!AssertError(weightedVariance(badValues[], w[]));
}

// Weighted variance is also an allocation-free histogram profile cell.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    import mir.math.common: approxEqual;
    alias Cell = WVarianceAccumulator!(double, WeightedVarianceNormalization.frequencySample);
    alias Axis = RegularAxis!(double, AxisOptions());
    Cell[2] storage;
    auto h = HistogramAccumulator!(typeof(storage), Axis)(storage, Axis(2, 0.0, 2.0));
    h.putWeightedSample(0.0, 0.5, 1.0);
    h.putWeightedSample(10.0, 0.5, 3.0);
    assert(h.counts[0].variance.approxEqual(25));
}

// Single-pass cursors must be consumed only once; explicit twoPass rejects them.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    static struct Cursor
    {
        double[] values;
        size_t[] position;
        bool empty() const @property { return position[0] == values.length; }
        double front() const @property { return values[position[0]]; }
        void popFront() { ++position[0]; }
    }
    double[4] x = [1, 2, 4, 8];
    double[4] w = [1, 3, 2, 1];
    size_t[1] ix, iw;
    auto values = Cursor(x[], ix[]);
    auto weights = Cursor(w[], iw[]);
    static assert(!__traits(compiles, weightedVariance!(WeightedVarianceNormalization.population,
        VarianceAlgo.twoPass)(values, weights)));
    assert(weightedVariance(values, weights).approxEqual(weightedVariance(x[], w[])));
    assert(ix[0] == x.length && iw[0] == w.length);
    WVarianceAccumulator!double a;
    a.put(x[], w[]);
    const expected = a.variance;
    a.put(a);
    assert(a.count == 8 && a.weight.approxEqual(14));
    assert(a.variance.approxEqual(expected));
    auto before = a.variance;
    a.put(x[0 .. 0], w[0 .. 0]);
    a.put(double.nan, 0);
    assert(a.variance == before);
}

// Jagged nested arrays traverse corresponding elements without flattening copies.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    double[][] rows = [[1.0, 2], [3.0, 4, 5, 6]];
    double[][] weights = [[1.0, 2], [3.0, 4, 5, 6]];
    double evaluate(scope double[][] x, scope double[][] w) @nogc
    {
        return weightedVariance(x, w);
    }
    double[6] flat = [1, 2, 3, 4, 5, 6];
    assert(evaluate(rows, weights).approxEqual(weightedVariance(flat[], flat[])));
}
// Preserve standard deviation even when variance cannot be represented.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: ldexp, sqrt;
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    alias N = WeightedVarianceNormalization;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (norm; [N.population, N.frequencySample, N.relativeSample])
    static foreach (algo; [VarianceAlgo.online, VarianceAlgo.twoPass, VarianceAlgo.hybrid])
    {{
        foreach (exponent; [T.min_exp / 2 - T.mant_dig, T.max_exp / 2 + T.mant_dig])
        {
            T span = ldexp(T(1), exponent);
            T[2] x = [0, span];
            T[2] w = [1, 3];
            T factor = norm == N.population ? sqrt(T(3)) / 4 :
                norm == N.frequencySample ? T(0.5) : 1 / sqrt(T(2));
            T expected = span * factor;
            auto actual = weightedStandardDeviation!(norm, algo)(x[], w[]);
            assert(actual.approxEqual(expected, T.epsilon * 32, T(0)));
            alias A = WVarianceAccumulator!(T, norm, algo);
            A stream, left, right;
            stream.put(x[0], w[0]);
            stream.put(x[1], w[1]);
            left.put(x[0], w[0]);
            right.put(x[1], w[1]);
            left.put(right);
            assert(stream.standardDeviation.approxEqual(expected, T.epsilon * 32, T(0)));
            assert(left.standardDeviation.approxEqual(expected, T.epsilon * 32, T(0)));
        }
    }}
}

// A negligible first row must not anchor subsequent means far from the data.
// The pairwise identity sum(i<j,wi*wj*(xi-xj)^2)/W^2 gives 11/9 here
// to double precision, without subtracting large raw moments.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias N = WeightedVarianceNormalization;
    double[4] x = [0, 1e16, 1e16 + 2, 1e16 + 2];
    double[4] w = [1e-32, 1, 1, 1];
    static foreach (algo; [VarianceAlgo.online, VarianceAlgo.twoPass, VarianceAlgo.hybrid])
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        alias A = WVarianceAccumulator!(double, N.population, algo, mode);
        assert((weightedVariance!(N.population, algo, mode)(x[], w[])).approxEqual(11.0/9, 1e-14, 0));
        foreach (split; 0 .. x.length + 1)
        {
            A left, right, reverse;
            left.put(x[0 .. split], w[0 .. split]);
            right.put(x[split .. x.length], w[split .. w.length]);
            reverse.put(right);
            reverse.put(left);
            left.put(right);
            assert(left.variance.approxEqual(11.0/9, 1e-14, 0));
            assert(reverse.variance.approxEqual(11.0/9, 1e-14, 0));
        }
    }}
}

// A normalized weight that underflows must not erase other observations' spread.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    double[3] x = [0, 2, 1e300];
    double[3] w = [1e308, 1e308, double.min_normal / 2];
    assert(weightedVariance(x[], w[]).approxEqual(1));
    assert(weightedStandardDeviation(x[], w[]).approxEqual(1));
}
