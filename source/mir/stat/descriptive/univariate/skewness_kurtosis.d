/++
Skewness and kurtosis.

Use $(REF skewness, mir, stat, descriptive, univariate, skewness_kurtosis) and $(REF kurtosis, mir, stat, descriptive, univariate, skewness_kurtosis) to summarize distribution shape.
Each statistic has an accumulator type and an enum selecting its calculation
algorithm. See the function parameters for population and sample conventions
and for raw versus excess kurtosis.

Import `mir.stat.descriptive.univariate.skewness_kurtosis` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF skewness, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
        $(REF kurtosis, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
    ))
    $(TR $(TD Accumulators) $(TD
        $(REF SkewnessAccumulator, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
        $(REF KurtosisAccumulator, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
    ))
    $(TR $(TD Algorithm selection) $(TD
        $(REF SkewnessAlgo, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
        $(REF KurtosisAlgo, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
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

module mir.stat.descriptive.univariate.skewness_kurtosis;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType, stdevType;
import mir.stat.descriptive.univariate.means: mean, MeanAccumulator;
import mir.stat.descriptive.univariate.dispersions: variance, standardDeviation, VarianceAlgo, VarianceAccumulator;

/++
Skewness algorithms.

See_also:
    $(WEB en.wikipedia.org/wiki/Skewness, Skewness),
    $(WEB en.wikipedia.org/wiki/Algorithms_for_calculating_variance, Algorithms for calculating variance)
+/
enum SkewnessAlgo
{
    /++
    Similar to Welford's algorithm for updating variance, but adjusted for skewness.
    Can also `put` another SkewnessAccumulator of the same type, which
    uses the parallel algorithm from Terriberry that extends the work of Chan et
    al.
    +/
    online,

    /++
    Calculates skewness using
    (E(x^^3) - 3 * mu * sigma ^^ 2 + mu ^^ 3) / (sigma ^^ 3)

    This algorithm can be numerically unstable.
    +/
    naive,

    /++
    Calculates skewness by first calculating the mean, then calculating
    E((x - E(x)) ^^ 3) / (E((x - E(x)) ^^ 2) ^^ 1.5)
    +/
    twoPass,

    /++
    Calculates skewness by first calculating the mean, then the standard deviation, then calculating
    E(((x - E(x)) / (E((x - E(x)) ^^ 2) ^^ 0.5)) ^^ 3)
    +/
    threePass,

    /++
    Calculates skewness assuming the mean of the input is zero.
    +/
    assumeZeroMean,

    /++
    When slices, slice-like objects, or ranges are the inputs, uses the two-pass
    algorithm. When an individual data-point is added, uses the online algorithm.
    +/
    hybrid
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.naive)
{
    import mir.functional: naryFun;
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S summatorOfSquares;
    ///
    private S summatorOfCubes;

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
        T x2 = x * x;
        summatorOfSquares.put(x2);
        summatorOfCubes.put(x2 * x);
    }

    void put(U, Summation sumAlgo)(SkewnessAccumulator!(U, skewnessAlgo, sumAlgo) v)
    {
        meanAccumulator.put(v.meanAccumulator);
        summatorOfSquares.put(v.sumOfSquares!T);
        summatorOfCubes.put(v.sumOfCubes!T);
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() @property
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "SkewnessAccumulator.varaince: count must be larger than one");
    }
    do
    {
        return sumOfSquares!F / (count + isPopulation - 1) -
            mean!F * mean!F * count / (count + isPopulation - 1);
    }
    ///
    F sumOfCubes(F = T)()
    {
        return cast(F) summatorOfCubes.sum;
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
    F centeredSumOfCubes(F = T)()
    {
        F mu = mean!F;
        return sumOfCubes!F - 3 * mu * sumOfSquares!F + 2 * count * mu * mu * mu;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(variance(true) > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;

        return scaledSumOfCubes!F(isPopulation) * count /
            ((count + isPopulation - 1) * (count + 2 * isPopulation - 2));
        /+ equivalent to
        F mu = mean!F;
        F avg_centeredSumOfCubes = sumOfCubes!F / count - 3 * mu * variance!F(true) - (mu * mu * mu);
        F var = variance!F(isPopulation);
        return avg_centeredSumOfCubes / (var * var.sqrt) *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
}

/// naive
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) v;
    v.put(x);
    v.skewness(true).shouldApprox == (117.005859 / 12) / pow(54.765625 / 12, 1.5);
    v.skewness(false).shouldApprox == (117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0);

    v.put(4.0);
    v.skewness(true).shouldApprox == (100.238166 / 13) / pow(57.019231 / 13, 1.5);
    v.skewness(false).shouldApprox == (100.238166 / 13) / pow(57.019231 / 12, 1.5) * (13.0 ^^ 2) / (12.0 * 11.0);
}

// check two-dimensional
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: pow;
    import mir.math.sum: Summation;
    import mir.ndslice.fuse: fuse;
    import mir.test: shouldApprox;

    auto x = [[0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
              [2.0, 7.5, 5.0, 1.0, 1.5, 0.00]].fuse;

    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) v;
    v.put(x);
    v.skewness(true).shouldApprox == (117.005859 / 12) / pow(54.765625 / 12, 1.5);
    v.skewness(false).shouldApprox == (117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0);
}

// Can put SkewnessAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: pow;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) v;
    v.put(x);
    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    v.skewness(true).shouldApprox == (117.005859 / 12) / pow(54.765625 / 12, 1.5);
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
    auto v1 = SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive)(x1);
    v1.skewness(true).should == 0;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive)(x2);
    v2.skewness(true).should == 0;
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.online)
{
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;

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
        T deltaOld = x;
        if (count > 0) {
            deltaOld -= mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (count - 1) * (count - 2) / (count * count) -
                                    3 * deltaOld * centeredSumOfSquares / count);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, SkewnessAlgo skewAlgo, Summation sumAlgo)(SkewnessAccumulator!(U, skewAlgo, sumAlgo) v)
        if (skewAlgo != SkewnessAlgo.assumeZeroMean)
    {
        size_t oldCount = count;
        T delta = v.mean;
        if (oldCount > 0) {
            delta -= mean;
        }
        meanAccumulator.put!T(v.meanAccumulator);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                    delta * delta * delta * v.count * oldCount * (oldCount - v.count) / (count * count) +
                                    3 * delta * (oldCount * v.centeredSumOfSquares!T - v.count * centeredSumOfSquares!T) / count);
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * v.count * oldCount / count);
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() @property
    {
        return meanAccumulator.mean!F;
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
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
}

/// online
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.skewness(true).approxEqual((117.005859 / 12) / pow(54.765625 / 12, 1.5)));
    assert(v.skewness(false).approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));

    v.put(4.0);
    assert(v.skewness(true).approxEqual((100.238166 / 13) / pow(57.019231 / 13, 1.5)));
    assert(v.skewness(false).approxEqual((100.238166 / 13) / pow(57.019231 / 12, 1.5) * (13.0 ^^ 2) / (12.0 * 11.0)));
}

// Can put slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    v.put(y);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (threePass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// check variance/scaledSumOfCubes
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive) v;
    v.put(x);
    auto varP = x.variance!"online"(true);
    auto varS = x.variance!"online"(false);
    v.variance(true).shouldApprox == varP;
    v.variance(false).shouldApprox == varS;
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.twoPass)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;

        meanAccumulator.put(slice.lightScope);

        auto sliceMap = slice.vmap(LeftOp!("-", T)(mean)).map!(naryFun!"a * a", naryFun!"a * a * a");
        centeredSummatorOfSquares.put(sliceMap.map!"a[0]");
        centeredSummatorOfCubes.put(sliceMap.map!"a[1]");
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

        auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a");
        centeredSummatorOfSquares.put(centeredRangeMultiplier.map!"a[0]");
        centeredSummatorOfCubes.put(centeredRangeMultiplier.map!"a[1]");
    }

const:

    ///
    size_t count()()
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "SkewnessAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
}

/// twoPass
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(x);
    assert(v.skewness(true).approxEqual(12.000999 / 12));
    assert(v.skewness(false).approxEqual(12.000999 / 12 * sqrt(12.0 * 11.0) / 10.0));
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(x);
    v.scaledSumOfCubes(true).shouldApprox == 12.000999;
}

// check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(x);
    v.scaledSumOfCubes(true).shouldApprox == 12.000999;
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
    auto v1 = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(x1);
    v1.skewness(true).should == 0;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(x2);
    v2.skewness(true).should == 0;
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.threePass)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S scaledSummatorOfCubes;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;
        import mir.math.common: sqrt;

        meanAccumulator.put(slice.lightScope);
        centeredSummatorOfSquares.put(slice.vmap(LeftOp!("-", T)(mean)).map!(naryFun!"a * a"));

        T stdP = variance!T(true).sqrt;
        assert(stdP > 0, "SkewnessAccumulator.this: must divide by positive standard deviation");

        scaledSummatorOfCubes.put(slice.
            vmap(LeftOp!("-", T)(mean)).
            vmap(LeftOp!("*", T)(1 / stdP)).
            map!(naryFun!"a * a * a"));
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
        import mir.math.common: sqrt;
        import std.algorithm: map;

        meanAccumulator.put(range);
        auto centeredRange = range.map!(a => (a - mean));
        centeredSummatorOfSquares.put(centeredRange.map!"a * a");
        T stdP = variance!T(true).sqrt;
        auto scaledRange = centeredRange.map!(a => a / stdP);
        scaledSummatorOfCubes.put(scaledRange.map!"a * a * a");
    }

const:

    ///
    size_t count()()
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "SkewnessAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        import mir.math.common: sqrt;
        F varP = variance!F(true); // based on using the population variance as divisor above
        return scaledSumOfCubes!F * varP * varP.sqrt;
    }
    ///
    F scaledSumOfCubes(F = T)()
    {
        return cast(F) scaledSummatorOfCubes.sum;
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
    }
    do
    {
        // formula for other skewness accumulators doesn't work here since we are
        // enforcing the the scaledSumOfCubes uses population variance and not that it can switch
        import mir.math.common: sqrt;
        return scaledSumOfCubes!F / (count + 2 * isPopulation - 2) *
                sqrt(cast(F) (count + isPopulation - 1) / count);
        /+ Equivalent to
        return scaledSumOfCubes!F / count *
                sqrt(cast(F) count * (count + isPopulation - 1)) / (count + 2 * isPopulation - 2)
        +/
    }
}

/// threePass
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(x);
    assert(v.skewness(true).approxEqual(12.000999 / 12));
    assert(v.skewness(false).approxEqual(12.000999 / 12 * sqrt(12.0 * 11.0) / 10.0));
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(x);
    assert(v.scaledSumOfCubes.approxEqual(12.000999));
}

// check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.math.sum: Summation;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(x);
    assert(v.scaledSumOfCubes.approxEqual(12.000999));
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
    auto v1 = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(x1);
    v1.skewness(true).should == 0;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(x2);
    v2.skewness(true).should == 0;
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.assumeZeroMean)
{
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;

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
        T x2 = x * x;
        centeredSummatorOfSquares.put(x2);
        centeredSummatorOfCubes.put(x2 * x);
    }

    ///
    void put(U, Summation sumAlgo)(SkewnessAccumulator!(U, skewnessAlgo, sumAlgo) v)
    {
        _count += v.count;
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F mean(F = T)() @property
    {
        return cast(F) 0;
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
    MeanAccumulator!(T, summation) meanAccumulator()()
    {
        typeof(return) m = { _count, T(0) };
        return m;
    }
    ///
    F centeredSumOfCubes(F = T)() @property
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F centeredSumOfSquares(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation) @property
    {
        import mir.math.common: sqrt;

        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
}

/// assumeZeroMean
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.skewness(true).approxEqual((117.005859 / 12) / pow(54.765625 / 12, 1.5)));
    assert(v.skewness(false).approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * 12.0 ^^ 2 / (11.0 * 10.0)));

    v.put(4.0);
    assert(v.skewness(true).approxEqual((181.005859 / 13) / pow(70.765625 / 13, 1.5)));
    assert(v.skewness(false).approxEqual((181.005859 / 13) / pow(70.765625 / 12, 1.5) * 13.0 ^^ 2 / (12.0 * 11.0)));
}

// Can put slices
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(-11.206543));
    assert(v.centeredSumOfSquares.approxEqual(13.49219));

    v.put(y);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator
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

    SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfCubes.approxEqual(-11.206543));
    assert(v.centeredSumOfSquares.approxEqual(13.49219));

    SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// check variance/scaledSumOfCubes
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    auto varP = x.variance!"assumeZeroMean"(true);
    auto varS = x.variance!"assumeZeroMean"(false);
    v.variance(true).shouldApprox == varP;
    v.variance(false).shouldApprox == varS;
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
}

///
struct SkewnessAccumulator(T, SkewnessAlgo skewnessAlgo, Summation summation)
    if (isMutable!T && skewnessAlgo == SkewnessAlgo.hybrid)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;

        meanAccumulator.put(slice.lightScope);

        auto sliceMap = slice.vmap(LeftOp!("-", T)(mean)).map!(naryFun!"a * a", naryFun!"a * a * a");
        centeredSummatorOfSquares.put(sliceMap.map!"a[0]");
        centeredSummatorOfCubes.put(sliceMap.map!"a[1]");
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
        static if (isInputRange!Range && is(elementType!Range : T)) {
            import std.algorithm: map;
            meanAccumulator.put(range);

            auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a");
            centeredSummatorOfSquares.put(centeredRangeMultiplier.map!"a[0]");
            centeredSummatorOfCubes.put(centeredRangeMultiplier.map!"a[1]");
        } else {
            this.put(range);
        }
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
        static if (isInputRange!Range && is(elementType!Range : T)) {
            auto v = typeof(this)(r);
            this.put(v);
        } else {
            foreach(x; r)
            {
                this.put(x);
            }
        }
    }

    ///
    void put()(T x)
    {
        T deltaOld = x;
        if (count > 0) {
            deltaOld -= mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (count - 1) * (count - 2) / (count * count) -
                                    3 * deltaOld * centeredSumOfSquares / count);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, SkewnessAlgo skewAlgo, Summation sumAlgo)(SkewnessAccumulator!(U, skewAlgo, sumAlgo) v)
        if (skewAlgo != SkewnessAlgo.assumeZeroMean)
    {
        size_t oldCount = count;
        T delta = v.mean;
        if (oldCount > 0) {
            delta -= mean;
        }
        meanAccumulator.put!T(v.meanAccumulator);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                    delta * delta * delta * v.count * oldCount * (oldCount - v.count) / (count * count) +
                                    3 * delta * (oldCount * v.centeredSumOfSquares!T - v.count * centeredSumOfSquares!T) / count);
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * v.count * oldCount / count);
    }

const:

    ///
    size_t count() @property
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)() @property
    {
        return meanAccumulator.mean!F;
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
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
}

/// hybrid
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.skewness(true).approxEqual((117.005859 / 12) / pow(54.765625 / 12, 1.5)));
    assert(v.skewness(false).approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));

    v.put(4.0);
    assert(v.skewness(true).approxEqual((100.238166 / 13) / pow(57.019231 / 13, 1.5)));
    assert(v.skewness(false).approxEqual((100.238166 / 13) / pow(57.019231 / 12, 1.5) * (13.0 ^^ 2) / (12.0 * 11.0)));
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    v.scaledSumOfCubes(true).shouldApprox == 12.000999;
}

// check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    v.scaledSumOfCubes(true).shouldApprox == 12.000999;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.algorithm: map;
    import std.range: chunks, iota;

    auto x1 = iota(0, 5);
    auto v1 = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x1);
    v1.skewness(true).should == 0;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x2);
    v2.skewness(true).should == 0;
    SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive) v3;
    v3.put(x1.chunks(1));
    v3.skewness(true).should == 0;
    auto v4 = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x1.chunks(1));
    v4.skewness(true).should == 0;
}

// Can put slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    v.put(y);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (online)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.online, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put SkewnessAccumulator (threePass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfCubes.approxEqual(4.071181));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = SkewnessAccumulator!(double, SkewnessAlgo.threePass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfCubes.approxEqual(117.005859));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// check variance/scaledSumOfCubes
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x);
    auto varP = x.variance!"twoPass"(true);
    auto varS = x.variance!"twoPass"(false);
    v.variance(true).shouldApprox == varP;
    v.variance(false).shouldApprox == varS;
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
}

/++
Calculates the skewness of the input

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    F = controls type of output
    skewnessAlgo = algorithm for calculating skewness (default: SkewnessAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The skewness of the input, must be floating point or complex type

See_also:
    $(LREF SkewnessAlgo)
+/
template skewness(F,
                  SkewnessAlgo skewnessAlgo = SkewnessAlgo.hybrid,
                  Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population skewness, false if sample skewness (default)
    +/
    @fmamath stdevType!F skewness(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        alias G = typeof(return);
        auto skewnessAccumulator = SkewnessAccumulator!(G, skewnessAlgo, ResolveSummationType!(summation, Range, G))(r.move);
        return skewnessAccumulator.skewness(isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F skewness(scope const F[] ar...)
    {
        alias G = typeof(return);
        auto skewnessAccumulator = SkewnessAccumulator!(G, skewnessAlgo, ResolveSummationType!(summation, const(G)[], G))(ar);
        return skewnessAccumulator.skewness(false);
    }
}

/// ditto
template skewness(SkewnessAlgo skewnessAlgo = SkewnessAlgo.hybrid,
                  Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population skewness, false if sample skewness (default)
    +/
    @fmamath stdevType!Range skewness(Range)(Range r, bool isPopulation = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        alias F = typeof(return);
        return .skewness!(F, skewnessAlgo, summation)(r.move, isPopulation);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T skewness(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .skewness!(F, skewnessAlgo, summation)(ar);
    }
}

/// ditto
template skewness(F, string skewnessAlgo, string summation = "appropriate")
{
    mixin("alias skewness = .skewness!(F, SkewnessAlgo." ~ skewnessAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template skewness(string skewnessAlgo, string summation = "appropriate")
{
    mixin("alias skewness = .skewness!(SkewnessAlgo." ~ skewnessAlgo ~ ", Summation." ~ summation ~ ");");
}

/// Simple example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    assert(skewness([1.0, 2, 3]).approxEqual(0.0));

    assert(skewness([1.0, 2, 4]).approxEqual((2.222222 / 3) / pow(4.666667 / 2, 1.5) * (3.0 ^^ 2) / (2.0 * 1.0)));
    assert(skewness([1.0, 2, 4], true).approxEqual((2.222222 / 3) / pow(4.666667 / 3, 1.5)));

    assert(skewness!float([0, 1, 2, 3, 4, 6].sliced(3, 2)).approxEqual(0.462910));

    static assert(is(typeof(skewness!float([1, 2, 3])) == float));
}

/// Skewness of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.skewness.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));
}

/// Skewness of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.skewness.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));
}

/// Column skewness of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0,  1.0, 1.5, 2.0],
        [3.5, 4.25, 2.0, 7.5],
        [5.0,  1.0, 1.5, 0.0]
    ].fuse;
    auto result = [-1.090291, 1.732051, 1.732051, 1.229809];

    // Use byDim or alongDim with map to compute skewness of row/column.
    assert(x.byDim!1.map!skewness.all!approxEqual(result));
    assert(x.alongDim!0.map!skewness.all!approxEqual(result));

    // FIXME
    // Without using map, computes the skewness of the whole slice
    // assert(x.byDim!1.skewness == x.sliced.skewness);
    // assert(x.alongDim!0.skewness == x.sliced.skewness);
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow, sqrt;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 100_000_000_000;

    // The online algorithm is numerically unstable in this case
    auto y = x.skewness!"online";
    assert(!y.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));

    // The naive algorithm has an assert error in this case because standard
    // deviation is calculated naively as zero. The skewness formula would then
    // be dividing by zero.
    //auto z0 = x.skewness!(real, "naive");

    // However, the two-pass and three-pass algorithms are numerically stable in this case
    auto z1 = x.skewness!"twoPass";
    assert(z1.approxEqual(12.000999 / 12 * sqrt(12.0 * 11.0) / 10.0));
    assert(!z1.approxEqual(y));
    auto z2 = x.skewness!"threePass";
    assert(z2.approxEqual((12.000999 / 12) * sqrt(12.0 * 11.0) / 10.0));
    assert(!z2.approxEqual(y));

    // And the assumeZeroMean algorithm provides the incorrect answer, as expected
    auto z3 = x.skewness!"assumeZeroMean";
    assert(!z3.approxEqual(y));
}

// Alt version with x a tenth of above's value
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow, sqrt;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 10_000_000_000;

    // The online algorithm is numerically stable in this case
    auto y = x.skewness!"online";
    assert(y.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));

    // The naive algorithm has an assert error in this case because standard
    // deviation is calculated naively as zero. The skewness formula would then
    // be dividing by zero.
    //auto z0 = x.skewness!(real, "naive");

    // The two-pass algorithm is  numerically stable in this case
    auto z1 = x.skewness!"twoPass";
    assert(z1.approxEqual(12.000999 / 12 * sqrt(12.0 * 11.0) / 10.0));
    assert(z1.approxEqual(y));

    // However, the three-pass algorithm is numerically stable in this case
    auto z2 = x.skewness!"threePass";
    assert(z2.approxEqual((12.000999 / 12) * sqrt(12.0 * 11.0) / 10.0));
    assert(z2.approxEqual(y));

    // And the assumeZeroMean algorithm provides the incorrect answer, as expected
    auto z3 = x.skewness!"assumeZeroMean";
    assert(!z3.approxEqual(y));
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

    //Set population skewness, skewness algorithm, sum algorithm or output type

    auto a = [1.0, 1e98, 1, -1e98].sliced;
    auto x = a * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and squaring `x`, the first and third numbers in the slice have precision
    too low to be included in the centered sum of squares.
    +/
    assert(x.skewness(false).approxEqual(0.0));
    assert(x.skewness(true).approxEqual(0.0));

    assert(x.skewness!("online").approxEqual(0.0));
    assert(x.skewness!("online", "kbn").approxEqual(0.0));
    assert(x.skewness!("online", "kb2").approxEqual(0.0));
    assert(x.skewness!("online", "precise").approxEqual(0.0));
    assert(x.skewness!(double, "online", "precise").approxEqual(0.0));
    assert(x.skewness!(double, "online", "precise")(true).approxEqual(0.0));

    auto y = [uint.max - 2, uint.max - 1, uint.max].sliced;
    auto z = y.skewness!(ulong, "threePass");
    assert(z == 0.0);
    static assert(is(typeof(z) == double));
}

/++
For integral slices, can pass output type as template parameter to ensure output
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

    auto y = x.skewness;
    assert(y.approxEqual(0.925493));
    static assert(is(typeof(y) == double));

    assert(x.skewness!float.approxEqual(0.925493));
}

/++
Skewness works for other user-defined types (provided they
can be converted to a floating point)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    Foo[] foo = [Foo(1f), Foo(2f), Foo(3f)];
    assert(foo.skewness == 0f);
}

/// Compute skewness along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0, 1, 3],
        [3.0, 4, 5],
        [6.0, 7, 7],
    ].fuse;

    assert(x.skewness.approxEqual(-0.308571));

    auto m0 = [0, 0.0, 0.0];
    assert(x.alongDim!0.map!skewness.all!approxEqual(m0));
    assert(x.alongDim!(-2).map!skewness.all!approxEqual(m0));

    auto m1 = [0.935220, 0.0, -1.732051];
    assert(x.alongDim!1.map!skewness.all!approxEqual(m1));
    assert(x.alongDim!(-1).map!skewness.all!approxEqual(m1));
    assert(iota(3, 4, 5, 6).as!double.alongDim!0.map!skewness.all!approxEqual(repeat(0.0, 4, 5, 6)));
}

/// Arbitrary skewness
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(skewness(1.0, 2, 3) == 0.0);
    assert(skewness!float(1, 2, 3) == 0f);
}

// Check skewness vector UFCS
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    assert([1.0, 2, 3, 4].skewness.approxEqual(0.0));
}

// Double-check correct output types
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([3, 3], 1);
    auto y = x.alongDim!1.map!skewness;
    assert(y.all!approxEqual([0.0, 0.0, 0.0]));
    static assert(is(stdevType!(typeof(y)) == double));
}

// @nogc skewness test
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.skewness.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));
    assert(x.sliced.skewness!float.approxEqual((117.005859 / 12) / pow(54.765625 / 11, 1.5) * (12.0 ^^ 2) / (11.0 * 10.0)));
}

// Test skewness with values
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.stat.transform: center;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.skewness.approxEqual(1.149008));
    assert(x.skewness(true).approxEqual(1.000083));
    assert(x.skewness!"naive".approxEqual(1.149008));
    assert(x.skewness!"naive"(true).approxEqual(1.000083));
    assert(x.skewness!"online".approxEqual(1.149008));
    assert(x.skewness!"online"(true).approxEqual(1.000083));
    assert(x.skewness!"twoPass".approxEqual(1.149008));
    assert(x.skewness!"twoPass"(true).approxEqual(1.000083));
    assert(x.skewness!"threePass".approxEqual(1.149008));
    assert(x.skewness!"threePass"(true).approxEqual(1.000083));

    auto y = x.center;
    assert(y.skewness!"assumeZeroMean".approxEqual(1.149008));
    assert(y.skewness!"assumeZeroMean"(true).approxEqual(1.000083));
}

/++
Kurtosis algorithms.

See_also:
    $(WEB en.wikipedia.org/wiki/Kurtosis, Kurtosis),
    $(WEB en.wikipedia.org/wiki/Algorithms_for_calculating_variance, Algorithms for calculating variance)
+/
enum KurtosisAlgo
{
    /++
    Similar to Welford's algorithm for updating variance, but adjusted for
    kurtosis. Can also `put` another KurtosisAccumulator of the same type, which
    uses the parallel algorithm from Terriberry that extends the work of Chan et
    al.
    +/
    online,
    /++
    Calculates kurtosis using
    (E(x^^4) - 4 * E(x) * E(x ^^ 3) + 6 * (E(x) ^^ 2) E(X ^^ 2) + 3 E(x) ^^ 4) / sigma ^ 2
    (allowing for adjustments for population/sample kurtosis).

    This algorithm can be numerically unstable.
    +/
    naive,

    /++
    Calculates kurtosis by first calculating the mean, then calculating
    E((x - E(x)) ^^ 4) / (E((x - E(x)) ^^ 2) ^^ 2)
    +/
    twoPass,

    /++
    Calculates kurtosis by first calculating the mean, then the standard deviation, then calculating
    E(((x - E(x)) / (E((x - E(x)) ^^ 2) ^^ 0.5)) ^^ 4)
    +/
    threePass,

    /++
    Calculates kurtosis assuming the mean of the input is zero.
    +/
    assumeZeroMean,

    /++
    When slices, slice-like objects, or ranges are the inputs, uses the two-pass
    algorithm. When an individual data-point is added, uses the online algorithm.
    +/
    hybrid
}

// Make sure skew algos and kurtosis algos match up
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.conv: to;
    assert(SkewnessAlgo.online.to!int == KurtosisAlgo.online.to!int);
    assert(SkewnessAlgo.naive.to!int == KurtosisAlgo.naive.to!int);
    assert(SkewnessAlgo.twoPass.to!int == KurtosisAlgo.twoPass.to!int);
    assert(SkewnessAlgo.threePass.to!int == KurtosisAlgo.threePass.to!int);
    assert(SkewnessAlgo.assumeZeroMean.to!int == KurtosisAlgo.assumeZeroMean.to!int);
    assert(SkewnessAlgo.hybrid.to!int == KurtosisAlgo.hybrid.to!int);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.naive)
{
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S summatorOfSquares;
    ///
    private S summatorOfCubes;
    ///
    private S summatorOfQuarts;

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
        T x2 = x * x;
        summatorOfSquares.put(x2);
        summatorOfCubes.put(x2 * x);
        summatorOfQuarts.put(x2 * x2);
    }

    ///
    void put(U, Summation sumAlgo)(KurtosisAccumulator!(U, kurtosisAlgo, sumAlgo) v)
    {
        meanAccumulator.put(v.meanAccumulator);
        summatorOfSquares.put(v.sumOfSquares!T);
        summatorOfCubes.put(v.sumOfCubes!T);
        summatorOfQuarts.put(v.sumOfQuarts!T);
    }

const:
    ///
    size_t count()
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "SkewnessAccumulator.varaince: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F sumOfSquares(F = T)()
    {
        return cast(F) summatorOfSquares.sum;
    }
    ///
    F sumOfCubes(F = T)()
    {
        return cast(F) summatorOfCubes.sum;
    }
    ///
    F sumOfQuarts(F = T)()
    {
        return cast(F) summatorOfQuarts.sum;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return sumOfSquares!F - count * mean!F * mean!F;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        F mu = mean!F;
        return sumOfCubes!F - 3 * mu * sumOfSquares!F + 2 * count * mu * mu * mu;
    }
    ///
    F centeredSumOfQuarts(F = T)()
    {
        F mu = mean!F;
        F mu2 = mu * mu;
        return sumOfQuarts!F - 4 * mu * sumOfCubes!F + 6 * mu2 * sumOfSquares!F - 3 * count * mu2 * mu2;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        F var = variance!F(isPopulation);
        return centeredSumOfQuarts!F / (var * var);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSumOfSquares > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
        assert(variance(true) > 0, "KurtosisAccumulator.kurtosis: variance must be larger than zero");
    }
    do
    {
        F mult1 = cast(F) count * (count + isPopulation - 1) * (count - isPopulation + 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F s = centeredSumOfSquares!F;
        return centeredSumOfQuarts!F / (s * s) * mult1 + 3 * (isRaw - mult2);
    }
}

/// naive
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: pow;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v;
    v.put(x);

    v.kurtosis(true, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0);
    v.kurtosis(true, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0);
    v.kurtosis(false, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0) + 3;

    v.skewness(true).shouldApprox == (117.005859 / 12) / pow(54.765625 / 12, 1.5);

    v.put(4.0);
    v.kurtosis(true, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0);
    v.kurtosis(true, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0);
    v.kurtosis(false, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0) + 3;

    v.skewness(true).shouldApprox == (100.238166 / 13) / pow(57.019231 / 13, 1.5);
}

// check two-dimensional
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: pow;
    import mir.math.sum: Summation;
    import mir.ndslice.fuse: fuse;
    import mir.test: shouldApprox;

    auto x = [[0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
              [2.0, 7.5, 5.0, 1.0, 1.5, 0.00]].fuse;

    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v;
    v.put(x);
    v.kurtosis(true, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0);
}

// Can put KurtosisAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: pow;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v;
    v.put(x);
    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    v.kurtosis(true, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0);
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v1;
    v1.put(x1);
    v1.kurtosis(false, true).shouldApprox == 1.8;
    auto x2 = x1.map!(a => 2 * a);
    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v2;
    v2.put(x2);
    v2.kurtosis(false, true).shouldApprox == 1.8;
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v;
    v.put(x);
    auto varP = x.variance!"naive"(true);
    auto varS = x.variance!"naive"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"naive"(true);
    v.skewness(false).shouldApprox == x.skewness!"naive"(false);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.online)
{
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;
    ///
    private S centeredSummatorOfQuarts;

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
        T deltaOld = x;
        if (count > 0) {
            deltaOld -= mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        centeredSummatorOfQuarts.put(deltaOld * deltaOld * deltaOld * deltaOld * ((count - 1) * (count * count - 3 * count + 3)) / (count * count * count) +
                                6 * deltaOld * deltaOld * centeredSumOfSquares!T / (count * count) -
                                4 * deltaOld * centeredSumOfCubes!T / count);
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (count - 1) * (count - 2) / (count * count) -
                               3 * deltaOld * centeredSumOfSquares!T / count);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, KurtosisAlgo kurtAlgo, Summation sumAlgo)(KurtosisAccumulator!(U, kurtAlgo, sumAlgo) v)
    {
        size_t oldCount = count;
        T delta = v.mean;
        if (oldCount > 0) {
            delta -= mean;
        }
        meanAccumulator.put!T(v.meanAccumulator);
        centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T +
                               delta * delta * delta * delta * ((v.count * oldCount) * (oldCount * oldCount - v.count * oldCount + v.count * v.count)) / (count * count * count) +
                               6 * delta * delta * ((oldCount * oldCount) * v.centeredSumOfSquares!T + (v.count * v.count) * centeredSumOfSquares!T) / (count * count) +
                               4 * delta * (oldCount * v.centeredSumOfCubes!T - v.count * centeredSumOfCubes!T) / count);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                               delta * delta * delta * v.count * oldCount * (oldCount - v.count) / (count * count) +
                               3 * delta * (oldCount * v.centeredSumOfSquares!T - v.count * centeredSumOfSquares!T) / count);
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * v.count * oldCount / count);
    }

const:

    ///
    size_t count()
    {
        return meanAccumulator.count;
    }
    ///
    F centeredSumOfQuarts(F = T)()
    {
        return cast(F) centeredSummatorOfQuarts.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F/ (var * var.sqrt);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        F var = variance!F(isPopulation);
        return centeredSumOfQuarts!F/ (var * var);
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "KurtosisAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
        assert(variance(true) > 0, "KurtosisAccumulator.kurtosis: variance must be larger than zero");
    }
    do
    {
        F mult1 = cast(F) count * (count + isPopulation - 1) * (count - isPopulation + 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F s = centeredSumOfSquares!F;
        return centeredSumOfQuarts!F / (s * s) * mult1 + 3 * (isRaw - mult2);
    }
}

/// online
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    v.kurtosis(true, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0);
    v.kurtosis(true, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0);
    v.kurtosis(false, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0) + 3;

    v.put(4.0);
    v.kurtosis(true, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0);
    v.kurtosis(true, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0);
    v.kurtosis(false, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0) + 3;
}

// Can put slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    v.put(y);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (threePass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (assumeZeroMean)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto b = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;
    auto y = b.center;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(622.639052)); //note: different from above due to inconsistent centering
    assert(v.centeredSumOfSquares.approxEqual(52.885417)); //note: different from above due to inconsistent centering
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive) v;
    v.put(x);
    auto varP = x.variance!"online"(true);
    auto varS = x.variance!"online"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"online"(true);
    v.skewness(false).shouldApprox == x.skewness!"online"(false);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.twoPass)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes; // only included to facilitate adding to online
    ///
    private S centeredSummatorOfQuarts;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;

        meanAccumulator.put(slice.lightScope);

        auto sliceMap = slice.vmap(LeftOp!("-", T)(mean)).map!(naryFun!"a * a", naryFun!"(a * a) * a", naryFun!"(a * a) * (a * a)");
        centeredSummatorOfSquares.put(sliceMap.map!"a[0]");
        centeredSummatorOfCubes.put(sliceMap.map!"a[1]");
        centeredSummatorOfQuarts.put(sliceMap.map!"a[2]");
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

        auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a", "a * a * a * a");
        centeredSummatorOfSquares.put(centeredRangeMultiplier.map!"a[0]");
        centeredSummatorOfCubes.put(centeredRangeMultiplier.map!"a[1]");
        centeredSummatorOfQuarts.put(centeredRangeMultiplier.map!"a[2]");
    }

const:

    ///
    size_t count()()
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "SkewnessAccumulator.variance: count must be larger than 1");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F centeredSumOfQuarts(F = T)()
    {
        return cast(F) centeredSummatorOfQuarts.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        auto var = variance!F(isPopulation);
        return centeredSumOfCubes!F / (var * var.sqrt);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        auto var = variance!F(isPopulation);
        return centeredSumOfQuarts!F / (var * var);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "KurtosisAccumulator.skewness: count must be larger than two");
        assert(centeredSumOfSquares > 0, "KurtosisAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
    }
    do
    {
        F mult1 = cast(F) count * (count + isPopulation - 1) * (count - isPopulation + 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F s = centeredSumOfSquares!F;
        return centeredSumOfQuarts!F / (s * s) * mult1 + 3 * (isRaw - mult2);
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

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x);
    assert(v.kurtosis(true, true).approxEqual(38.062853 / 12));
    assert(v.kurtosis(true, false).approxEqual(38.062853 / 12 - 3.0));
    assert(v.kurtosis(false, true).approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(v.kurtosis(false, false).approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x);
    v.scaledSumOfQuarts(true).shouldApprox == 38.062853;
}

// check dynamic slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x);
    v.scaledSumOfQuarts(true).shouldApprox == 38.062853;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x1);
    v1.kurtosis(false, true).shouldApprox == 1.8;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x2);
    v2.kurtosis(false, true).shouldApprox == 1.8;
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(x);
    auto varP = x.variance!"twoPass"(true);
    auto varS = x.variance!"twoPass"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"twoPass"(true);
    v.skewness(false).shouldApprox == x.skewness!"twoPass"(false);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.threePass)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S scaledSummatorOfCubes; //only included to facilitate adding to online accumulator
    ///
    private S scaledSummatorOfQuarts;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;
        import mir.math.common: sqrt;

        meanAccumulator.put(slice.lightScope);
        auto centeredSlice = slice.vmap(LeftOp!("-", T)(mean));
        centeredSummatorOfSquares.put(centeredSlice.map!(naryFun!"a * a"));

        assert(variance(true) > 0, "KurtosisAccumulator.this: must divide by positive standard deviation");

        auto sliceMap = centeredSlice.
            vmap(LeftOp!("*", T)(1 / variance(true).sqrt)).
            map!(naryFun!"(a * a) * a", naryFun!"(a * a) * (a * a)");
        scaledSummatorOfCubes.put(sliceMap.map!"a[0]");
        scaledSummatorOfQuarts.put(sliceMap.map!"a[1]");
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
        import mir.math.common: sqrt;
        import std.algorithm: map;

        meanAccumulator.put(range);
        auto centeredRange = range.map!(a => (a - mean));
        centeredSummatorOfSquares.put(centeredRange.map!"a * a");
        auto rangeMap = centeredRange.
            map!(a => a / variance(true).sqrt).
            map!("(a * a) * a", "(a * a) * (a * a)");
        scaledSummatorOfCubes.put(rangeMap.map!"a[0]");
        scaledSummatorOfQuarts.put(rangeMap.map!"a[1]");
    }

const:

    ///
    size_t count()()
    {
        return meanAccumulator.count;
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "SkewnessAccumulator.variance: count must be larger than 1");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        import mir.math.common: sqrt;
        // variance consistent with that used for scaledSumOfQuarts above
        auto varP = variance!F(true);
        return scaledSumOfCubes!F * varP * varP.sqrt;
    }
    ///
    F centeredSumOfQuarts(F = T)()
    {
        // variance consistent with that used for scaledSumOfQuarts above
        auto varP = variance!F(true);
        return scaledSumOfQuarts!F * varP * varP;
    }
    ///
    F scaledSumOfCubes(F = T)()
    {
        return cast(F) scaledSummatorOfCubes.sum;
    }
    ///
    F scaledSumOfQuarts(F = T)()
    {
        return cast(F) scaledSummatorOfQuarts.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        return scaledSumOfCubes!F * (count + isPopulation - 1) * sqrt(cast(F) count + isPopulation - 1) / count / sqrt(cast(F) count);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        return scaledSumOfQuarts!F * (count + isPopulation - 1) * (count + isPopulation - 1) / cast(F) count / cast(F) count;
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "KurtosisAccumulator.skewness: count must be larger than two");
    }
    do
    {
        // formula for other kurtosis accumulators doesn't work here since we are
        // enforcing the the scaledSumOfCubes uses population variance and not that it can switch
        import mir.math.common: sqrt;
        return scaledSumOfCubes!F / (count + 2 * isPopulation - 2) *
                sqrt(cast(F) (count + isPopulation - 1) / count);
        /+ Equivalent to
        return scaledSumOfCubes!F / count *
                sqrt(cast(F) count * (count + isPopulation - 1)) / (count + 2 * isPopulation - 2)
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
    }
    do
    {
        // formula for other kurtosis accumulators doesn't work here since we are
        // enforcing the scaling uses population variance and not that it can switch
        F mult1 = cast(F) (count + isPopulation - 1) * (count - isPopulation + 1) / (count * (count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));

        return scaledSumOfQuarts!F * mult1 + 3 * (isRaw - mult2);
    }
}

/// threePass
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x);
    assert(v.kurtosis(true, true).approxEqual(38.062853 / 12));
    assert(v.kurtosis(true, false).approxEqual(38.062853 / 12 - 3.0));
    assert(v.kurtosis(false, true).approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(v.kurtosis(false, false).approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x);
    assert(v.scaledSumOfQuarts.approxEqual(38.062853));
}

// check dynamic slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.math.sum: Summation;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x);
    assert(v.scaledSumOfQuarts.approxEqual(38.062853));
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto v1 = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x1);
    v1.kurtosis(false, true).shouldApprox == 1.8;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x2);
    v2.kurtosis(false, true).shouldApprox == 1.8;
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(x);
    auto varP = x.variance!"twoPass"(true);
    auto varS = x.variance!"twoPass"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"threePass"(true);
    v.skewness(false).shouldApprox == x.skewness!"threePass"(false);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.assumeZeroMean)
{
    import mir.math.sum: Summator;
    import std.traits: isIterable;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;
    ///
    private S centeredSummatorOfQuarts;

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
        T x2 = x * x;
        centeredSummatorOfSquares.put(x2);
        centeredSummatorOfCubes.put(x2 * x);
        centeredSummatorOfQuarts.put(x2 * x2);
    }

    ///
    void put(U, Summation sumAlgo)(KurtosisAccumulator!(U, kurtosisAlgo, sumAlgo) v)
    {
        _count += v.count;
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
        centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T);
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F mean(F = T)() @property
    {
        return cast(F) 0;
    }
    MeanAccumulator!(T, summation) meanAccumulator()()
    {
        typeof(return) m = { _count, T(0) };
        return m;
    }
    ///
    F variance(F = T)(bool isPopulation) @property
    in
    {
        assert(count > 1, "KurtosisAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F centeredSumOfQuarts(F = T)() @property
    {
        return cast(F) centeredSummatorOfQuarts.sum;
    }
    ///
    F centeredSumOfCubes(F = T)() @property
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F centeredSumOfSquares(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F/ (var * var.sqrt);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        F var = variance!F(isPopulation);
        return centeredSumOfQuarts!F/ (var * var);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
        assert(variance(true) > 0, "KurtosisAccumulator.kurtosis: variance must be larger than zero");
    }
    do
    {
        F mult1 = cast(F) count * (count + isPopulation - 1) * (count - isPopulation + 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F s = centeredSumOfSquares!F;
        return centeredSumOfQuarts!F / (s * s) * mult1 + 3 * (isRaw - mult2);
    }
}

/// assumeZeroMean
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.kurtosis(true, true).approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0)));
    assert(v.kurtosis(true, false).approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0) - 3.0));
    assert(v.kurtosis(false, false).approxEqual(792.784119 / pow(54.765625 / 11, 2.0) * (12.0 * 13.0) / (11.0 * 10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
    assert(v.kurtosis(false, true).approxEqual(792.784119 / pow(54.765625 / 11, 2.0) * (12.0 * 13.0) / (11.0 * 10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0) + 3.0));

    v.put(4.0);
    assert(v.kurtosis(true, true).approxEqual((1048.784119 / 13) / pow(70.765625 / 13, 2.0)));
    assert(v.kurtosis(true, false).approxEqual((1048.784119 / 13) / pow(70.765625 / 13, 2.0) - 3.0));
    assert(v.kurtosis(false, false).approxEqual(1048.784119 / pow(70.765625 / 12, 2.0) * (13.0 * 14.0) / (12.0 * 11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0)));
    assert(v.kurtosis(false, true).approxEqual(1048.784119 / pow(70.765625 / 12, 2.0) * (13.0 * 14.0) / (12.0 * 11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0) + 3.0));
}

// Can put slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(52.44613647));
    assert(v.centeredSumOfSquares.approxEqual(13.4921875));

    v.put(y);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = a.center;
    auto x = b[0 .. 6];
    auto y = b[6 .. $];

    KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(52.44613647));
    assert(v.centeredSumOfSquares.approxEqual(13.4921875));

    KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive) w;
    w.put(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive)(x);
    auto varP = x.variance!"assumeZeroMean"(true);
    auto varS = x.variance!"assumeZeroMean"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"assumeZeroMean"(true);
    v.skewness(false).shouldApprox == x.skewness!"assumeZeroMean"(false);
}

///
struct KurtosisAccumulator(T, KurtosisAlgo kurtosisAlgo, Summation summation)
    if (isMutable!T && kurtosisAlgo == KurtosisAlgo.hybrid)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import std.range: isInputRange;
    import std.traits: isIterable;

    ///
    private MeanAccumulator!(T, summation) meanAccumulator;
    ///
    alias S = Summator!(T, summation);
    ///
    private S centeredSummatorOfSquares;
    ///
    private S centeredSummatorOfCubes;
    ///
    private S centeredSummatorOfQuarts;

    ///
    this(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        import mir.functional: naryFun;
        import mir.ndslice.topology: vmap, map;
        import mir.ndslice.internal: LeftOp;

        meanAccumulator.put(slice.lightScope);

        auto sliceMap = slice.vmap(LeftOp!("-", T)(mean)).map!(naryFun!"a * a", naryFun!"(a * a) * a", naryFun!"(a * a) * (a * a)");
        centeredSummatorOfSquares.put(sliceMap.map!"a[0]");
        centeredSummatorOfCubes.put(sliceMap.map!"a[1]");
        centeredSummatorOfQuarts.put(sliceMap.map!"a[2]");
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
        static if (isInputRange!Range && is(elementType!Range : T)) {
            import std.algorithm: map;
            meanAccumulator.put(range);

            auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a", "a * a * a * a");
            centeredSummatorOfSquares.put(centeredRangeMultiplier.map!"a[0]");
            centeredSummatorOfCubes.put(centeredRangeMultiplier.map!"a[1]");
            centeredSummatorOfQuarts.put(centeredRangeMultiplier.map!"a[2]");
        } else {
            this.put(range);
        }
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
        static if (isInputRange!Range && is(elementType!Range : T)) {
            auto v = typeof(this)(r);
            this.put(v);
        } else {
            foreach(x; r)
            {
                this.put(x);
            }
        }
    }

    ///
    void put()(T x)
    {
        T deltaOld = x;
        if (count > 0) {
            deltaOld -= mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        centeredSummatorOfQuarts.put(deltaOld * deltaOld * deltaOld * deltaOld * ((count - 1) * (count * count - 3 * count + 3)) / (count * count * count) +
                                6 * deltaOld * deltaOld * centeredSumOfSquares!T / (count * count) -
                                4 * deltaOld * centeredSumOfCubes!T / count);
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (count - 1) * (count - 2) / (count * count) -
                               3 * deltaOld * centeredSumOfSquares!T / count);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, KurtosisAlgo kurtAlgo, Summation sumAlgo)(KurtosisAccumulator!(U, kurtAlgo, sumAlgo) v)
    {
        size_t oldCount = count;
        T delta = v.mean;
        if (oldCount > 0) {
            delta -= mean;
        }
        meanAccumulator.put!T(v.meanAccumulator);
        centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T +
                               delta * delta * delta * delta * ((v.count * oldCount) * (oldCount * oldCount - v.count * oldCount + v.count * v.count)) / (count * count * count) +
                               6 * delta * delta * ((oldCount * oldCount) * v.centeredSumOfSquares!T + (v.count * v.count) * centeredSumOfSquares!T) / (count * count) +
                               4 * delta * (oldCount * v.centeredSumOfCubes!T - v.count * centeredSumOfCubes!T) / count);
        centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                               delta * delta * delta * v.count * oldCount * (oldCount - v.count) / (count * count) +
                               3 * delta * (oldCount * v.centeredSumOfSquares!T - v.count * centeredSumOfSquares!T) / count);
        centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * v.count * oldCount / count);
    }

const:

    ///
    size_t count()
    {
        return meanAccumulator.count;
    }
    ///
    F centeredSumOfQuarts(F = T)()
    {
        return cast(F) centeredSummatorOfQuarts.sum;
    }
    ///
    F centeredSumOfCubes(F = T)()
    {
        return cast(F) centeredSummatorOfCubes.sum;
    }
    ///
    F centeredSumOfSquares(F = T)()
    {
        return cast(F) centeredSummatorOfSquares.sum;
    }
    ///
    F scaledSumOfCubes(F = T)(bool isPopulation)
    {
        import mir.math.common: sqrt;
        F var = variance!F(isPopulation);
        return centeredSumOfCubes!F/ (var * var.sqrt);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        F var = variance!F(isPopulation);
        return centeredSumOfQuarts!F/ (var * var);
    }
    ///
    F mean(F = T)()
    {
        return meanAccumulator.mean!F;
    }
    ///
    F variance(F = T)(bool isPopulation)
    in
    {
        assert(count > 1, "KurtosisAccumulator.variance: count must be larger than one");
    }
    do
    {
        return centeredSumOfSquares!F / (count + isPopulation - 1);
    }
    ///
    F skewness(F = T)(bool isPopulation)
    in
    {
        assert(count > 2, "SkewnessAccumulator.skewness: count must be larger than two");
        assert(centeredSummatorOfSquares.sum > 0, "SkewnessAccumulator.skewness: variance must be larger than zero");
    }
    do
    {
        import mir.math.common: sqrt;
        F s = centeredSumOfSquares!F;
        return centeredSumOfCubes!F / (s * s.sqrt) * count * sqrt(cast(F) count + isPopulation - 1) /
            (count + 2 * isPopulation - 2);
        /+ Equivalent to
        return scaledSumOfCubes!F(isPopulation) / count *
                (cast(F) count * count / ((count + isPopulation - 1) * (count + 2 * isPopulation - 2)));
        +/
    }
    ///
    F kurtosis(F = T)(bool isPopulation, bool isRaw)
    in
    {
        assert(count > 3, "KurtosisAccumulator.kurtosis: count must be larger than three");
        assert(variance(true) > 0, "KurtosisAccumulator.kurtosis: variance must be larger than zero");
    }
    do
    {
        F mult1 = cast(F) count * (count + isPopulation - 1) * (count - isPopulation + 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F mult2 = cast(F) (count + isPopulation - 1) * (count + isPopulation - 1) / ((count + 2 * isPopulation - 2) * (count + 3 * isPopulation - 3));
        F s = centeredSumOfSquares!F;
        return centeredSumOfQuarts!F / (s * s) * mult1 + 3 * (isRaw - mult2);
    }
}

/// hybrid
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    v.kurtosis(true, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0);
    v.kurtosis(true, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0);
    v.kurtosis(false, true).shouldApprox == (792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0) + 3;

    v.put(4.0);
    v.kurtosis(true, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0);
    v.kurtosis(true, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) - 3;
    v.kurtosis(false, false).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0);
    v.kurtosis(false, true).shouldApprox == (745.608180 / 13) / pow(57.019231 / 13, 2.0) * (12.0 * 14.0) / (11.0 * 10.0) - 3.0 * (12.0 * 12.0) / (11.0 * 10.0) + 3;
}

// check withAsSlice
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    v.scaledSumOfQuarts(true).shouldApprox == 38.062853;
}

// check dynamic slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                  2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    v.scaledSumOfQuarts(true).shouldApprox == 38.062853;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;
    import std.algorithm: map;
    import std.range: chunks, iota;

    auto x1 = iota(0, 5);
    auto v1 = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x1);
    v1.kurtosis(false, true).shouldApprox == 1.8;
    auto x2 = x1.map!(a => 2 * a);
    auto v2 = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x2);
    v2.kurtosis(false, true).shouldApprox == 1.8;
    KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive) v3;
    v3.put(x1.chunks(1));
    v3.kurtosis(false, true).shouldApprox == 1.8;
    auto v4 = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x1.chunks(1));
    v4.kurtosis(false, true).shouldApprox == 1.8;
}

// Can put slice
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive) v;
    v.put(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    v.put(y);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (online)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.online, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (threePass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    assert(v.centeredSumOfQuarts.approxEqual(46.944607));
    assert(v.centeredSumOfSquares.approxEqual(12.552083));

    auto w = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(792.784119));
    assert(v.centeredSumOfSquares.approxEqual(54.765625));
}

// Can put KurtosisAccumulator (assumeZeroMean)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25].sliced;
    auto b = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto x = a.center;
    auto y = b.center;

    auto v = KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive)(x);
    auto w = KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.naive)(y);
    v.put(w);
    assert(v.centeredSumOfQuarts.approxEqual(622.639052)); //note: different from above due to inconsistent centering
    assert(v.centeredSumOfSquares.approxEqual(52.885417)); //note: different from above due to inconsistent centering
}

// check scaledSumOfCubes/scaledSumOfQuarts/skewness
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.dispersions: variance;
    import mir.math.common: sqrt;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    KurtosisAccumulator!(double, KurtosisAlgo.hybrid, Summation.naive) v;
    v.put(x);
    auto varP = x.variance!"twoPass"(true);
    auto varS = x.variance!"twoPass"(false);
    v.scaledSumOfCubes(true).shouldApprox == v.centeredSumOfCubes / (varP * varP.sqrt);
    v.scaledSumOfCubes(false).shouldApprox == v.centeredSumOfCubes / (varS * varS.sqrt);
    v.scaledSumOfQuarts(true).shouldApprox == v.centeredSumOfQuarts / (varP * varP);
    v.scaledSumOfQuarts(false).shouldApprox == v.centeredSumOfQuarts / (varS * varS);
    v.skewness(true).shouldApprox == x.skewness!"hybrid"(true);
    v.skewness(false).shouldApprox == x.skewness!"hybrid"(false);
}

/++
Calculates the kurtosis of the input

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    F = controls type of output
    kurtosisAlgo = algorithm for calculating kurtosis (default: KurtosisAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)

Returns:
    The kurtosis of the input, must be floating point

See_also:
    $(LREF KurtosisAlgo)
+/
template kurtosis(
    F,
    KurtosisAlgo kurtosisAlgo = KurtosisAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population kurtosis, false if sample kurtosis (default)
        isRaw = true if raw kurtosis, false if excess kurtosis (default)
    +/
    @fmamath stdevType!F kurtosis(Range)(Range r, bool isPopulation = false, bool isRaw = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        alias G = typeof(return);
        auto kurtosisAccumulator = KurtosisAccumulator!(G, kurtosisAlgo, ResolveSummationType!(summation, Range, G))(r.move);
        return kurtosisAccumulator.kurtosis(isPopulation, isRaw);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!F kurtosis(scope const F[] ar...)
    {
        alias G = typeof(return);
        auto kurtosisAccumulator = KurtosisAccumulator!(G, kurtosisAlgo, ResolveSummationType!(summation, const(G)[], G))(ar);
        return kurtosisAccumulator.kurtosis(false, false);
    }
}

/// ditto
template kurtosis(
    KurtosisAlgo kurtosisAlgo = KurtosisAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        isPopulation = true if population kurtosis, false if sample kurtosis (default)
        isRaw = true if raw kurtosis, false if excess kurtosis (default)
    +/
    @fmamath stdevType!Range kurtosis(Range)(Range r, bool isPopulation = false, bool isRaw = false)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        alias F = typeof(return);
        return .kurtosis!(F, kurtosisAlgo, summation)(r.move, isPopulation, isRaw);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath stdevType!T kurtosis(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .kurtosis!(F, kurtosisAlgo, summation)(ar);
    }
}

/// ditto
template kurtosis(F, string kurtosisAlgo, string summation = "appropriate")
{
    mixin("alias kurtosis = .kurtosis!(F, KurtosisAlgo." ~ kurtosisAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template kurtosis(string kurtosisAlgo, string summation = "appropriate")
{
    mixin("alias kurtosis = .kurtosis!(KurtosisAlgo." ~ kurtosisAlgo ~ ", Summation." ~ summation ~ ");");
}

/// Simple example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    assert(kurtosis([1.0, 2, 3, 4]).approxEqual(-1.2));

    assert(kurtosis([1.0, 2, 4, 5]).approxEqual((34.0 / 4) / pow(10.0 / 4, 2.0) * (3.0 * 5.0) / (2.0 * 1.0) - 3.0 * (3.0 * 3.0) / (2.0 * 1.0)));
    // population excess kurtosis
    assert(kurtosis([1.0, 2, 4, 5], true).approxEqual((34.0 / 4) / pow(10.0 / 4, 2.0) - 3.0));
    // sample raw kurtosis
    assert(kurtosis([1.0, 2, 4, 5], false, true).approxEqual((34.0 / 4) / pow(10.0 / 4, 2.0) * (3.0 * 5.0) / (2.0 * 1.0) - 3.0 * (3.0 * 3.0) / (2.0 * 1.0) + 3.0));
    // population raw kurtosis
    assert(kurtosis([1.0, 2, 4, 5], true, true).approxEqual((34.0 / 4) / pow(10.0 / 4, 2.0)));

    assert(kurtosis!float([0, 1, 2, 3, 4, 6].sliced(3, 2)).approxEqual(-0.2999999));

    static assert(is(typeof(kurtosis!float([1, 2, 3])) == float));
}

/// Kurtosis of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.kurtosis.approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
}

/// Kurtosis of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.kurtosis.approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
}

/// Column kurtosis of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0,  1.0,  1.5, 2.0],
        [3.5, 4.25,  2.0, 7.5],
        [5.0,  1.0,  1.5, 0.0],
        [1.5,  4.5, 4.75, 0.5]
    ].fuse;
    auto result = [-2.067182, -5.918089, 3.504056, 2.690240];

    // Use byDim or alongDim with map to compute kurtosis of row/column.
    assert(x.byDim!1.map!kurtosis.all!approxEqual(result));
    assert(x.alongDim!0.map!kurtosis.all!approxEqual(result));

    // FIXME
    // Without using map, computes the kurtosis of the whole slice
    // assert(x.byDim!1.kurtosis == x.sliced.kurtosis);
    // assert(x.alongDim!0.kurtosis == x.sliced.kurtosis);
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 100_000_000_000;

    // The online algorithm is numerically unstable in this case
    auto y = x.kurtosis!"online";
    assert(!y.approxEqual((792.78411865 / 12) / pow(54.76562500 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));

    // The naive algorithm has an assert error in this case because standard
    // deviation is calculated naively as zero. The kurtosis formula would then
    // be dividing by zero.
    //auto z0 = x.kurtosis!(real, "naive");

    // The two-pass algorithm is also numerically unstable in this case
    auto z1 = x.kurtosis!"twoPass";
    assert(!z1.approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(!z1.approxEqual(y));

    // However, the three-pass algorithm is numerically stable in this case
    auto z2 = x.kurtosis!"threePass";
    assert(z2.approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(!z2.approxEqual(y));

    // And the assumeZeroMean algorithm provides the incorrect answer, as expected
    auto z3 = x.kurtosis!"assumeZeroMean";
    assert(!z3.approxEqual(y));
}

// Alt version with x a hundred of above's value
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    auto x = a + 1_000_000_000;

    // The online algorithm is numerically stable in this case
    auto y = x.kurtosis!"online";
    assert(y.approxEqual((792.78411865 / 12) / pow(54.76562500 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));

    // The naive algorithm has an assert error in this case because standard
    // deviation is calculated naively as zero. The kurtosis formula would then
    // be dividing by zero.
    //auto z0 = x.kurtosis!(real, "naive");

    // The two-pass algorithm is  numerically stable in this case
    auto z1 = x.kurtosis!"twoPass";
    assert(z1.approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(z1.approxEqual(y));

    // However, the three-pass algorithm is numerically stable in this case
    auto z2 = x.kurtosis!"threePass";
    assert(z2.approxEqual(38.062853 / 12 * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)) + 3.0);
    assert(z2.approxEqual(y));

    // And the assumeZeroMean algorithm provides the incorrect answer, as expected
    auto z3 = x.kurtosis!"assumeZeroMean";
    assert(!z3.approxEqual(y));
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

    // Set population/sample kurtosis, excess/raw kurtosis, kurtosis algorithm,
    // sum algorithm or output type

    auto a = [1.0, 1e72, 1, -1e72].sliced;
    auto x = a * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect. Further, after centering
    and taking `x` to the fourth power, the first and third numbers in the slice
    have precision too low to be included in the centered sum of cubes.
    +/
    assert(x.kurtosis.approxEqual(1.5));
    assert(x.kurtosis(false).approxEqual(1.5));
    assert(x.kurtosis(true).approxEqual(-1.0));
    assert(x.kurtosis(true, true).approxEqual(2.0));
    assert(x.kurtosis(false, true).approxEqual(4.5));

    assert(x.kurtosis!("online").approxEqual(1.5));
    assert(x.kurtosis!("online", "kbn").approxEqual(1.5));
    assert(x.kurtosis!("online", "kb2").approxEqual(1.5));
    assert(x.kurtosis!("online", "precise").approxEqual(1.5));
    assert(x.kurtosis!(double, "online", "precise").approxEqual(1.5));
    assert(x.kurtosis!(double, "online", "precise")(true).approxEqual(-1.0));
    assert(x.kurtosis!(double, "online", "precise")(true, true).approxEqual(2.0));

    auto y = [uint.max - 3, uint.max - 2, uint.max - 1, uint.max].sliced;
    auto z = y.kurtosis!(ulong, "threePass");
    assert(z.approxEqual(-1.2));
    static assert(is(typeof(z) == double));
}

/++
For integral slices, can pass output type as template parameter to ensure output
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

    auto y = x.kurtosis;
    assert(y.approxEqual(0.223394));
    static assert(is(typeof(y) == double));

    assert(x.kurtosis!float.approxEqual(0.223394));
}

/++
Kurtosis works for other user-defined types (provided they can be converted to a
floating point)
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

    Foo[] foo = [Foo(1f), Foo(2f), Foo(3f), Foo(4f)];
    assert(foo.kurtosis.approxEqual(-1.2f));
}

/// Compute kurtosis along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0,  1,  3,  5],
        [3.0,  4,  5,  7],
        [6.0,  7, 10, 11],
        [9.0, 12, 15, 12]
    ].fuse;

    assert(x.kurtosis.approxEqual(-0.770040));

    auto m0 = [-1.200000, -0.152893, -1.713859, -3.869005];
    assert(x.alongDim!0.map!kurtosis.all!approxEqual(m0));
    assert(x.alongDim!(-2).map!kurtosis.all!approxEqual(m0));

    auto m1 = [-1.699512, 0.342857, -4.339100, 1.500000];
    assert(x.alongDim!1.map!kurtosis.all!approxEqual(m1));
    assert(x.alongDim!(-1).map!kurtosis.all!approxEqual(m1));

    assert(iota(4, 5, 6, 7).as!double.alongDim!0.map!kurtosis.all!approxEqual(repeat(-1.2, 5, 6, 7)));
}

/// Arbitrary kurtosis
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(kurtosis(1.0, 2, 3, 4).approxEqual(-1.2));
    assert(kurtosis!float(1, 2, 3, 4).approxEqual(-1.2f));
}

// Check kurtosis vector UFCS
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].kurtosis.approxEqual(-1.2));
}

// Double-check correct output types
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([4, 4], 1);
    auto y = x.alongDim!1.map!kurtosis;
    assert(y.all!approxEqual([-1.2, -1.2, -1.2, -1.2]));
    static assert(is(stdevType!(typeof(y)) == double));
}

// @nogc kurtosis test
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.kurtosis.approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
    assert(x.sliced.kurtosis!float.approxEqual((792.784119 / 12) / pow(54.765625 / 12, 2.0) * (11.0 * 13.0) / (10.0 * 9.0) - 3.0 * (11.0 * 11.0) / (10.0 * 9.0)));
}

// Test all using values
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.stat.transform: center;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.kurtosis.approxEqual(1.006470));
    assert(x.kurtosis(false, true).approxEqual(4.006470));
    assert(x.kurtosis(true).approxEqual(0.171904));
    assert(x.kurtosis(true, true).approxEqual(3.171904));

    assert(x.kurtosis!"naive".approxEqual(1.006470));
    assert(x.kurtosis!"naive"(false, true).approxEqual(4.006470));
    assert(x.kurtosis!"naive"(true).approxEqual(0.171904));
    assert(x.kurtosis!"naive"(true, true).approxEqual(3.171904));

    assert(x.kurtosis!"online".approxEqual(1.006470));
    assert(x.kurtosis!"online"(false, true).approxEqual(4.006470));
    assert(x.kurtosis!"online"(true).approxEqual(0.171904));
    assert(x.kurtosis!"online"(true, true).approxEqual(3.171904));

    assert(x.kurtosis!"twoPass".approxEqual(1.006470));
    assert(x.kurtosis!"twoPass"(false, true).approxEqual(4.006470));
    assert(x.kurtosis!"twoPass"(true).approxEqual(0.171904));
    assert(x.kurtosis!"twoPass"(true, true).approxEqual(3.171904));

    assert(x.kurtosis!"threePass".approxEqual(1.006470));
    assert(x.kurtosis!"threePass"(false, true).approxEqual(4.006470));
    assert(x.kurtosis!"threePass"(true).approxEqual(0.171904));
    assert(x.kurtosis!"threePass"(true, true).approxEqual(3.171904));

    auto y = x.center;
    assert(y.kurtosis!"assumeZeroMean".approxEqual(1.006470));
    assert(y.kurtosis!"assumeZeroMean"(false, true).approxEqual(4.006470));
    assert(y.kurtosis!"assumeZeroMean"(true).approxEqual(0.171904));
    assert(y.kurtosis!"assumeZeroMean"(true, true).approxEqual(3.171904));
}
