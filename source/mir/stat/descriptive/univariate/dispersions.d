/++
Variance, standard deviation, and related measures of dispersion.

Use the named functions for standard summaries, or $(REF dispersion, mir, stat, descriptive, univariate, dispersions)
to choose the centering, transformation, and summary operations.
$(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions) selects the variance calculation algorithm.

Import `mir.stat.descriptive.univariate.dispersions` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF variance, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF standardDeviation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF coefficientOfVariation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        $(REF dispersion, mir, stat, descriptive, univariate, dispersions)$(NBSP)
    ))
    $(TR $(TD Accumulator) $(TD
        $(REF VarianceAccumulator, mir, stat, descriptive, univariate, dispersions)$(NBSP)
    ))
    $(TR $(TD Algorithm selection) $(TD
        $(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions)$(NBSP)
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
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType, stdevType;
import mir.stat.descriptive.univariate.means: mean, MeanAccumulator;
import mir.stat.descriptive.univariate.quantiles: median;

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
    +/
    twoPass,

    /++
    Calculates variance assuming the mean of the dataseries is zero.
    +/
    assumeZeroMean,

    /++
    When slices, slice-like objects, or ranges are the inputs, uses the two-pass
    algorithm. When an individual data-point is added, uses the online algorithm.
    +/
    hybrid
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
        if (count > 0) {
            delta -= meanAccumulator.mean;
        }
        meanAccumulator.put(x);
        centeredSummatorOfSquares.put(delta * (x - meanAccumulator.mean));
    }

    ///
    void put(U, VarianceAlgo varAlgo, Summation sumAlgo)(VarianceAccumulator!(U, varAlgo, sumAlgo) v)
        if (varAlgo != VarianceAlgo.assumeZeroMean)
    {
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
        // Apply the count ratio before multiplying by delta twice, avoiding
        // intermediate overflow when the final correction is representable.
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
            + delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
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

// Empty merges are identities; nonempty and self merges preserve the statistics.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (algo; AliasSeq!(VarianceAlgo.online, VarianceAlgo.hybrid))
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kbn))
    {{
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
    import std.range: isInputRange;

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
        if (isInputRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import std.algorithm: map;
        meanAccumulator.put(range);

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
    import std.range: isInputRange;

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
        static if (isInputRange!Range && is(elementType!Range : T))
        {
            import std.algorithm: map;
            meanAccumulator.put(range);

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
        static if (isInputRange!Range && is(elementType!Range : T)) {
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
        if (count > 0) {
            delta -= meanAccumulator.mean;
        }
        meanAccumulator.put(x);
        centeredSummatorOfSquares.put(delta * (x - meanAccumulator.mean));
    }

    ///
    void put(U, VarianceAlgo varAlgo, Summation sumAlgo)(VarianceAccumulator!(U, varAlgo, sumAlgo) v)
        if (varAlgo != VarianceAlgo.assumeZeroMean)
    {
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
        // Apply the count ratio before multiplying by delta twice, avoiding
        // intermediate overflow when the final correction is representable.
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T
            + delta * (delta * (cast(T) v.count * (cast(T) oldCount / count))));
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
