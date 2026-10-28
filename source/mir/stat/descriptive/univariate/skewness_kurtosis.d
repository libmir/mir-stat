/++
Skewness and kurtosis.

Use $(REF skewness, mir, stat, descriptive, univariate, skewness_kurtosis) and $(REF kurtosis, mir, stat, descriptive, univariate, skewness_kurtosis) to summarize distribution shape.
Each unweighted statistic has an accumulator type and an enum selecting its calculation
algorithm. See the function parameters for population and sample conventions
and for raw versus excess kurtosis.

Import `mir.stat.descriptive.univariate.skewness_kurtosis` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF weightedSkewness, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
        $(REF weightedKurtosis, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
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

// Merge stored moments only when their representation and summation state match.
// Add recentering corrections separately: combining them with a rounded incoming
// total would discard the low-order parts before the destination can retain them.
// Three-pass higher moments are scaled, and naive moments are uncentered;
// their getters must still perform the conversion to centered moments.
// Mixed types/modes retain the existing conversion through the public getter.
// Kahan is excluded because Summator's += does not transfer its correction.
// Keep pairwise scalar merging: supported mir-algorithm releases have a counter
// defect in direct merges that can overflow the partial-sum buffer.
private template canMergeShapeMoment(T, Summation summation, V, string member)
{
    static if (__traits(hasMember, V, member) &&
        (summation == Summation.kbn || summation == Summation.kb2 ||
         summation == Summation.precise))
        enum canMergeShapeMoment = is(typeof(__traits(getMember, V, member)) ==
            Summator!(T, summation));
    else
        enum canMergeShapeMoment = false;
}

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

    Inputs that cannot be converted to a slice must be forward ranges.
    +/
    twoPass,

    /++
    Calculates skewness by first calculating the mean, then the standard deviation, then calculating
    E(((x - E(x)) / (E((x - E(x)) ^^ 2) ^^ 0.5)) ^^ 3)

    Inputs that cannot be converted to a slice must be forward ranges.
    +/
    threePass,

    /++
    Calculates skewness assuming the mean of the input is zero.
    +/
    assumeZeroMean,

    /++
    Uses the two-pass algorithm for slices, slice-like objects, and forward ranges.
    Single-pass inputs and individual data points use the online algorithm.
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
        static if (canMergeShapeMoment!(T, summation, typeof(v), "summatorOfSquares"))
        {
            summatorOfSquares += v.summatorOfSquares;
        }
        else
            summatorOfSquares.put(v.sumOfSquares!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "summatorOfCubes"))
        {
            summatorOfCubes += v.summatorOfCubes;
        }
        else
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

        return scaledSumOfCubes!F(isPopulation) *
            (cast(F) count / (count - 2 + 2 * isPopulation)) / (count - 1 + isPopulation);
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
        // The first observation has zero centered moments. Avoid raising x
        // to a power that could overflow before multiplication by zero.
        T deltaOld = T(0);
        if (count > 0) {
            deltaOld = x - mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * ((cast(T) count - 2) * inverseCount) -
                                    3 * deltaOld * centeredSumOfSquares * inverseCount);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, SkewnessAlgo skewAlgo, Summation sumAlgo)(SkewnessAccumulator!(U, skewAlgo, sumAlgo) v)
        if (skewAlgo != SkewnessAlgo.assumeZeroMean)
    {
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        if (!oldCount) {
            // No recentering is needed; avoid evaluating powers of a possibly
            // large mean only to multiply them by a zero destination count.
            meanAccumulator.put!T(v.meanAccumulator);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
            {
                centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            }
            else
                centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
            {
                centeredSummatorOfCubes += v.centeredSummatorOfCubes;
            }
            else
                centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
            return;
        }
        T delta = v.mean;
        delta -= mean;
        // Subtract counts before conversion, but preserve the sign when the
        // incoming batch is larger. size_t subtraction alone would wrap.
        T countDifference = oldCount >= v.count
            ? cast(T) (oldCount - v.count) : -cast(T) (v.count - oldCount);
        meanAccumulator.put!T(v.meanAccumulator);
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        T oldFraction = cast(T) oldCount * inverseCount;
        T newFraction = cast(T) v.count * inverseCount;
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes.put(delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                    3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
            centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                        delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                        3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares.put(delta * delta * newFraction * oldCount);
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * newFraction * oldCount);
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
    import std.range: isInputRange, isForwardRange, save;

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
        if (isForwardRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import std.algorithm: map;
        // Save every traversal that is followed by another pass over the input.
        meanAccumulator.put(range.save);

        auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a");
        centeredSummatorOfSquares.put(centeredRangeMultiplier.save.map!"a[0]");
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
    import std.range: isInputRange, isForwardRange, save;

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
        if (isForwardRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import mir.math.common: sqrt;
        import std.algorithm: map;

        // Save every traversal that is followed by another pass over the input.
        meanAccumulator.put(range.save);
        auto centeredRange = range.map!(a => (a - mean));
        centeredSummatorOfSquares.put(centeredRange.save.map!"a * a");
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
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
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
    import std.range: isInputRange, isForwardRange, save;
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
        static if (isForwardRange!Range && is(elementType!Range : T)) {
            import std.algorithm: map;
            // Save every traversal that is followed by another pass over the input.
            meanAccumulator.put(range.save);

            auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a");
            centeredSummatorOfSquares.put(centeredRangeMultiplier.save.map!"a[0]");
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
        static if (isInputRange!Range && is(elementType!Range : T) &&
            (isConvertibleToSlice!Range || isForwardRange!Range)) {
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
        // The first observation has zero centered moments. Avoid raising x
        // to a power that could overflow before multiplication by zero.
        T deltaOld = T(0);
        if (count > 0) {
            deltaOld = x - mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * ((cast(T) count - 2) * inverseCount) -
                                    3 * deltaOld * centeredSumOfSquares * inverseCount);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, SkewnessAlgo skewAlgo, Summation sumAlgo)(SkewnessAccumulator!(U, skewAlgo, sumAlgo) v)
        if (skewAlgo != SkewnessAlgo.assumeZeroMean)
    {
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        if (!oldCount) {
            // No recentering is needed; avoid evaluating powers of a possibly
            // large mean only to multiply them by a zero destination count.
            meanAccumulator.put!T(v.meanAccumulator);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
            {
                centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            }
            else
                centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
            {
                centeredSummatorOfCubes += v.centeredSummatorOfCubes;
            }
            else
                centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
            return;
        }
        T delta = v.mean;
        delta -= mean;
        // Subtract counts before conversion, but preserve the sign when the
        // incoming batch is larger. size_t subtraction alone would wrap.
        T countDifference = oldCount >= v.count
            ? cast(T) (oldCount - v.count) : -cast(T) (v.count - oldCount);
        meanAccumulator.put!T(v.meanAccumulator);
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        T oldFraction = cast(T) oldCount * inverseCount;
        T newFraction = cast(T) v.count * inverseCount;
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes.put(delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                    3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
            centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                        delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                        3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares.put(delta * delta * newFraction * oldCount);
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * newFraction * oldCount);
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
    import mir.math.common: approxEqual;
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
    // Incremental merging can leave roundoff in the symmetric third moment.
    assert(v3.skewness(true).approxEqual(0.0, 8 * double.epsilon, 8 * double.epsilon));
    auto v4 = SkewnessAccumulator!(double, SkewnessAlgo.hybrid, Summation.naive)(x1.chunks(1));
    assert(v4.skewness(true).approxEqual(0.0, 8 * double.epsilon, 8 * double.epsilon));
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

/// Column skewness of matrix.
/// The mapped result is lazy; see the $(MREF mir,stat,descriptive,univariate)
/// overview for storing results in allocated or caller-provided memory.
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

    Inputs that cannot be converted to a slice must be forward ranges.
    +/
    twoPass,

    /++
    Calculates kurtosis by first calculating the mean, then the standard deviation, then calculating
    E(((x - E(x)) / (E((x - E(x)) ^^ 2) ^^ 0.5)) ^^ 4)

    Inputs that cannot be converted to a slice must be forward ranges.
    +/
    threePass,

    /++
    Calculates kurtosis assuming the mean of the input is zero.
    +/
    assumeZeroMean,

    /++
    Uses the two-pass algorithm for slices, slice-like objects, and forward ranges.
    Single-pass inputs and individual data points use the online algorithm.
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
        static if (canMergeShapeMoment!(T, summation, typeof(v), "summatorOfSquares"))
        {
            summatorOfSquares += v.summatorOfSquares;
        }
        else
            summatorOfSquares.put(v.sumOfSquares!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "summatorOfCubes"))
        {
            summatorOfCubes += v.summatorOfCubes;
        }
        else
            summatorOfCubes.put(v.sumOfCubes!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "summatorOfQuarts"))
        {
            summatorOfQuarts += v.summatorOfQuarts;
        }
        else
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
        // The bias corrections are ratios of nearby counts. Do not form
        // their integer products or large intermediate floating powers.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = cast(F) count * correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount);
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);
        F s = centeredSumOfSquares!F;
        return (centeredSumOfQuarts!F / s) * (mult1 / s) + 3 * (isRaw - mult2);
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
        // The first observation has zero centered moments. Avoid raising x
        // to a power that could overflow before multiplication by zero.
        T deltaOld = T(0);
        if (count > 0) {
            deltaOld = x - mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        centeredSummatorOfQuarts.put(deltaOld * deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * (1 - 3 * inverseCount + 3 * inverseCount * inverseCount) +
                                6 * deltaOld * deltaOld * centeredSumOfSquares!T * inverseCount * inverseCount -
                                4 * deltaOld * centeredSumOfCubes!T * inverseCount);
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * ((cast(T) count - 2) * inverseCount) -
                               3 * deltaOld * centeredSumOfSquares!T * inverseCount);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, KurtosisAlgo kurtAlgo, Summation sumAlgo)(KurtosisAccumulator!(U, kurtAlgo, sumAlgo) v)
    {
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        if (!oldCount) {
            // No recentering is needed; avoid evaluating powers of a possibly
            // large mean only to multiply them by a zero destination count.
            meanAccumulator.put!T(v.meanAccumulator);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
            {
                centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            }
            else
                centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
            {
                centeredSummatorOfCubes += v.centeredSummatorOfCubes;
            }
            else
                centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfQuarts"))
            {
                centeredSummatorOfQuarts += v.centeredSummatorOfQuarts;
            }
            else
                centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T);
            return;
        }
        T delta = v.mean;
        delta -= mean;
        // Subtract counts before conversion, but preserve the sign when the
        // incoming batch is larger. size_t subtraction alone would wrap.
        T countDifference = oldCount >= v.count
            ? cast(T) (oldCount - v.count) : -cast(T) (v.count - oldCount);
        meanAccumulator.put!T(v.meanAccumulator);
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        T oldFraction = cast(T) oldCount * inverseCount;
        T newFraction = cast(T) v.count * inverseCount;
        // a*b*(a*a-a*b+b*b)/n^3 = a*(b/n)*((a/n)^2-(a/n)*(b/n)+(b/n)^2).
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfQuarts"))
        {
            centeredSummatorOfQuarts.put(delta * delta * delta * delta * newFraction * oldCount * (oldFraction * oldFraction - oldFraction * newFraction + newFraction * newFraction) +
                               6 * delta * delta * (oldFraction * oldFraction * v.centeredSumOfSquares!T + newFraction * newFraction * centeredSumOfSquares!T) +
                               4 * delta * (oldFraction * v.centeredSumOfCubes!T - newFraction * centeredSumOfCubes!T));
            centeredSummatorOfQuarts += v.centeredSummatorOfQuarts;
        }
        else
            centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T +
                                   delta * delta * delta * delta * newFraction * oldCount * (oldFraction * oldFraction - oldFraction * newFraction + newFraction * newFraction) +
                                   6 * delta * delta * (oldFraction * oldFraction * v.centeredSumOfSquares!T + newFraction * newFraction * centeredSumOfSquares!T) +
                                   4 * delta * (oldFraction * v.centeredSumOfCubes!T - newFraction * centeredSumOfCubes!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes.put(delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                               3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
            centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                   delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                   3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares.put(delta * delta * newFraction * oldCount);
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * newFraction * oldCount);
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
        // The bias corrections are ratios of nearby counts. Do not form
        // their integer products or large intermediate floating powers.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = cast(F) count * correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount);
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);
        F s = centeredSumOfSquares!F;
        return (centeredSumOfQuarts!F / s) * (mult1 / s) + 3 * (isRaw - mult2);
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
    import std.range: isInputRange, isForwardRange, save;

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
        if (isForwardRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import std.algorithm: map;
        // Save every traversal that is followed by another pass over the input.
        meanAccumulator.put(range.save);

        auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a", "a * a * a * a");
        centeredSummatorOfSquares.put(centeredRangeMultiplier.save.map!"a[0]");
        centeredSummatorOfCubes.put(centeredRangeMultiplier.save.map!"a[1]");
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
        // The bias corrections are ratios of nearby counts. Do not form
        // their integer products or large intermediate floating powers.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = cast(F) count * correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount);
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);
        F s = centeredSumOfSquares!F;
        return (centeredSumOfQuarts!F / s) * (mult1 / s) + 3 * (isRaw - mult2);
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
    import std.range: isInputRange, isForwardRange, save;

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
        if (isForwardRange!Range && !isConvertibleToSlice!Range && is(elementType!Range : T))
    {
        import mir.math.common: sqrt;
        import std.algorithm: map;

        // Save every traversal that is followed by another pass over the input.
        meanAccumulator.put(range.save);
        auto centeredRange = range.map!(a => (a - mean));
        centeredSummatorOfSquares.put(centeredRange.save.map!"a * a");
        auto rangeMap = centeredRange.
            map!(a => a / variance(true).sqrt).
            map!("(a * a) * a", "(a * a) * (a * a)");
        scaledSummatorOfCubes.put(rangeMap.save.map!"a[0]");
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
        F correction = cast(F) (count - 1 + isPopulation) / count;
        return scaledSumOfCubes!F * correction * sqrt(correction);
    }
    ///
    F scaledSumOfQuarts(F = T)(bool isPopulation)
    {
        F correction = cast(F) (count - 1 + isPopulation) / count;
        return scaledSumOfQuarts!F * correction * correction;
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
        // The stored fourth sum is already scaled by population variance.
        // Use count ratios for the bias correction, then divide by count.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount) / count;
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);

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
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
            centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfQuarts"))
        {
            centeredSummatorOfQuarts += v.centeredSummatorOfQuarts;
        }
        else
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
        // The bias corrections are ratios of nearby counts. Do not form
        // their integer products or large intermediate floating powers.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = cast(F) count * correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount);
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);
        F s = centeredSumOfSquares!F;
        return (centeredSumOfQuarts!F / s) * (mult1 / s) + 3 * (isRaw - mult2);
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
    import std.range: isInputRange, isForwardRange, save;
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
        static if (isForwardRange!Range && is(elementType!Range : T)) {
            import std.algorithm: map;
            // Save every traversal that is followed by another pass over the input.
            meanAccumulator.put(range.save);

            auto centeredRangeMultiplier = range.map!(a => (a - mean)).map!("a * a", "a * a * a", "a * a * a * a");
            centeredSummatorOfSquares.put(centeredRangeMultiplier.save.map!"a[0]");
            centeredSummatorOfCubes.put(centeredRangeMultiplier.save.map!"a[1]");
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
        static if (isInputRange!Range && is(elementType!Range : T) &&
            (isConvertibleToSlice!Range || isForwardRange!Range)) {
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
        // The first observation has zero centered moments. Avoid raising x
        // to a power that could overflow before multiplication by zero.
        T deltaOld = T(0);
        if (count > 0) {
            deltaOld = x - mean;
        }
        meanAccumulator.put(x);
        T deltaNew = x - mean;
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        centeredSummatorOfQuarts.put(deltaOld * deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * (1 - 3 * inverseCount + 3 * inverseCount * inverseCount) +
                                6 * deltaOld * deltaOld * centeredSumOfSquares!T * inverseCount * inverseCount -
                                4 * deltaOld * centeredSumOfCubes!T * inverseCount);
        centeredSummatorOfCubes.put(deltaOld * deltaOld * deltaOld * (cast(T) (count - 1) * inverseCount) * ((cast(T) count - 2) * inverseCount) -
                               3 * deltaOld * centeredSumOfSquares!T * inverseCount);
        centeredSummatorOfSquares.put(deltaOld * deltaNew);
    }

    ///
    void put(U, KurtosisAlgo kurtAlgo, Summation sumAlgo)(KurtosisAccumulator!(U, kurtAlgo, sumAlgo) v)
    {
        // An empty source has no mean and contributes no observations.
        if (!v.count)
            return;
        size_t oldCount = count;
        if (!oldCount) {
            // No recentering is needed; avoid evaluating powers of a possibly
            // large mean only to multiply them by a zero destination count.
            meanAccumulator.put!T(v.meanAccumulator);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
            {
                centeredSummatorOfSquares += v.centeredSummatorOfSquares;
            }
            else
                centeredSummatorOfSquares.put(v.centeredSumOfSquares!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
            {
                centeredSummatorOfCubes += v.centeredSummatorOfCubes;
            }
            else
                centeredSummatorOfCubes.put(v.centeredSumOfCubes!T);
            static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfQuarts"))
            {
                centeredSummatorOfQuarts += v.centeredSummatorOfQuarts;
            }
            else
                centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T);
            return;
        }
        T delta = v.mean;
        delta -= mean;
        // Subtract counts before conversion, but preserve the sign when the
        // incoming batch is larger. size_t subtraction alone would wrap.
        T countDifference = oldCount >= v.count
            ? cast(T) (oldCount - v.count) : -cast(T) (v.count - oldCount);
        meanAccumulator.put!T(v.meanAccumulator);
        // Form count ratios before multiplying: integer count powers can wrap,
        // and floating count powers can overflow even when the result is finite.
        T inverseCount = T(1) / count;
        T oldFraction = cast(T) oldCount * inverseCount;
        T newFraction = cast(T) v.count * inverseCount;
        // a*b*(a*a-a*b+b*b)/n^3 = a*(b/n)*((a/n)^2-(a/n)*(b/n)+(b/n)^2).
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfQuarts"))
        {
            centeredSummatorOfQuarts.put(delta * delta * delta * delta * newFraction * oldCount * (oldFraction * oldFraction - oldFraction * newFraction + newFraction * newFraction) +
                               6 * delta * delta * (oldFraction * oldFraction * v.centeredSumOfSquares!T + newFraction * newFraction * centeredSumOfSquares!T) +
                               4 * delta * (oldFraction * v.centeredSumOfCubes!T - newFraction * centeredSumOfCubes!T));
            centeredSummatorOfQuarts += v.centeredSummatorOfQuarts;
        }
        else
            centeredSummatorOfQuarts.put(v.centeredSumOfQuarts!T +
                                   delta * delta * delta * delta * newFraction * oldCount * (oldFraction * oldFraction - oldFraction * newFraction + newFraction * newFraction) +
                                   6 * delta * delta * (oldFraction * oldFraction * v.centeredSumOfSquares!T + newFraction * newFraction * centeredSumOfSquares!T) +
                                   4 * delta * (oldFraction * v.centeredSumOfCubes!T - newFraction * centeredSumOfCubes!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfCubes"))
        {
            centeredSummatorOfCubes.put(delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                               3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
            centeredSummatorOfCubes += v.centeredSummatorOfCubes;
        }
        else
            centeredSummatorOfCubes.put(v.centeredSumOfCubes!T +
                                   delta * delta * delta * newFraction * oldCount * (countDifference * inverseCount) +
                                   3 * delta * (oldFraction * v.centeredSumOfSquares!T - newFraction * centeredSumOfSquares!T));
        static if (canMergeShapeMoment!(T, summation, typeof(v), "centeredSummatorOfSquares"))
        {
            centeredSummatorOfSquares.put(delta * delta * newFraction * oldCount);
            centeredSummatorOfSquares += v.centeredSummatorOfSquares;
        }
        else
            centeredSummatorOfSquares.put(v.centeredSumOfSquares!T + delta * delta * newFraction * oldCount);
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
        // The bias corrections are ratios of nearby counts. Do not form
        // their integer products or large intermediate floating powers.
        F correction = cast(F) (count - 1 + isPopulation) / (count - 2 + 2 * isPopulation);
        F inverseAdjustedCount = F(1) / (count - 3 + 3 * isPopulation);
        F mult1 = cast(F) count * correction *
            ((cast(F) (count - isPopulation) + 1) * inverseAdjustedCount);
        F mult2 = correction * (cast(F) (count - 1 + isPopulation) * inverseAdjustedCount);
        F s = centeredSumOfSquares!F;
        return (centeredSumOfQuarts!F / s) * (mult1 / s) + 3 * (isRaw - mult2);
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

/// Column kurtosis of matrix.
/// The mapped result is lazy; see the $(MREF mir,stat,descriptive,univariate)
/// overview for storing results in allocated or caller-provided memory.
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

    // The exact sample excess is 32076/143585. Allow float roundoff in
    // the moment sums and subtraction of the bias correction near four.
    assert(x.kurtosis!float.approxEqual(32076.0f / 143585,
        16 * float.epsilon, 16 * float.epsilon));
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

// Parallel merges preserve signed count differences and empty identities.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (hybrid; [false, true])
    {{
        alias S = SkewnessAccumulator!(T,
            hybrid ? SkewnessAlgo.hybrid : SkewnessAlgo.online, Summation.naive);
        alias K = KurtosisAccumulator!(T,
            hybrid ? KurtosisAlgo.hybrid : KurtosisAlgo.online, Summation.naive);
        static foreach (A; AliasSeq!(S, K))
        {{
            // The incoming batch is larger; reversing the merge must agree.
            A left, right;
            left.put(T(0));
            right.put(T(1));
            right.put(T(2));
            auto reverse = right;
            reverse.put(left);
            left.put(right);
            foreach (result; [left, reverse])
            {
                assert(result.count == 3 && result.mean == 1);
                assert(result.centeredSumOfSquares.approxEqual(T(2)));
                assert(result.centeredSumOfCubes == 0);
                static if (is(A == K))
                    assert(result.centeredSumOfQuarts.approxEqual(T(2)));
            }
            // Empty merges do not contaminate a valid accumulator.
            A empty;
            left.put(empty);
            assert(left.count == 3 && left.mean == 1);
            assert(left.centeredSumOfSquares.approxEqual(T(2)));
            assert(left.centeredSumOfCubes == 0);
            static if (is(A == K))
                assert(left.centeredSumOfQuarts.approxEqual(T(2)));
            empty.put(empty);
            assert(empty.count == 0 && empty.centeredSumOfSquares == 0);
            // Copying into an empty target needs no powers of the large mean.
            static if (is(A == K))
                alias Batch = KurtosisAccumulator!(T, KurtosisAlgo.twoPass, Summation.naive);
            else
                alias Batch = SkewnessAccumulator!(T, SkewnessAlgo.twoPass, Summation.naive);
            auto large = Batch([T(0x1p100), T(0x1p100)]);
            empty.put(large);
            assert(empty.count == 2 && empty.mean == T(0x1p100));
            assert(empty.centeredSumOfSquares == 0);
            assert(empty.centeredSumOfCubes == 0);
            static if (is(A == K))
                assert(empty.centeredSumOfQuarts == 0);
            // By-value source parameters preserve self-merge behavior.
            left.put(left);
            assert(left.count == 6 && left.mean == 1);
            assert(left.centeredSumOfSquares.approxEqual(T(4)));
            assert(left.centeredSumOfCubes == 0);
            static if (is(A == K))
                assert(left.centeredSumOfQuarts.approxEqual(T(4)));
        }}
    }}
}

// Multi-pass ranges need independent cursors for every accumulated power.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;

    static struct Input(T, bool replayable)
    {
        const(T)[] values;
        size_t[] cursor;
        size_t ownPosition;
        size_t position() const { return cursor.length ? cursor[0] : ownPosition; }
        bool empty() const { return position == values.length; }
        T front() const { return values[position]; }
        void popFront() { if (cursor.length) ++cursor[0]; else ++ownPosition; }
        static if (replayable)
            auto save() { return typeof(this)(values, null, position); }
    }
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T[3] values = [0, 1, 5];
        static foreach (algo; [SkewnessAlgo.twoPass, SkewnessAlgo.threePass, SkewnessAlgo.hybrid])
        {{
            size_t[1] cursor;
            alias A = SkewnessAccumulator!(T, algo, Summation.naive);
            auto result = A(Input!(T, true)(values[], cursor[]));
            assert(result.count == 3 && result.mean == 2);
            assert(result.centeredSumOfSquares.approxEqual(T(14)));
            assert(result.centeredSumOfCubes.approxEqual(T(18)));
            assert(result.skewness(true).approxEqual(A(values[]).skewness(true)));
            assert(cursor[0] == values.length);
            static if (algo != SkewnessAlgo.hybrid)
                static assert(!__traits(compiles, A(Input!(T, false)(values[], cursor[]))));
        }}
        static foreach (algo; [KurtosisAlgo.twoPass, KurtosisAlgo.threePass, KurtosisAlgo.hybrid])
        {{
            size_t[1] cursor;
            alias A = KurtosisAccumulator!(T, algo, Summation.naive);
            auto result = A(Input!(T, true)(values[], cursor[]));
            assert(result.count == 3 && result.mean == 2);
            assert(result.centeredSumOfSquares.approxEqual(T(14)));
            assert(result.centeredSumOfCubes.approxEqual(T(18)));
            assert(result.centeredSumOfQuarts.approxEqual(T(98)));
            assert(result.scaledSumOfQuarts(true).approxEqual(A(values[]).scaledSumOfQuarts(true)));
            assert(cursor[0] == values.length);
            static if (algo != KurtosisAlgo.hybrid)
                static assert(!__traits(compiles, A(Input!(T, false)(values[], cursor[]))));
        }}
        alias S = SkewnessAccumulator!(T, SkewnessAlgo.hybrid, Summation.naive);
        alias K = KurtosisAccumulator!(T, KurtosisAlgo.hybrid, Summation.naive);
        static foreach (A; AliasSeq!(S, K))
        static foreach (replayable; [false, true])
        {{
            size_t[1] cursor;
            auto input = Input!(T, replayable)(values[], cursor[]);
            auto constructed = A(input);
            assert(constructed.count == 3 && constructed.mean == 2);
            assert(constructed.centeredSumOfCubes.approxEqual(T(18)));
            static if (is(A == K))
                assert(constructed.centeredSumOfQuarts.approxEqual(T(98)));
            assert(cursor[0] == values.length);
            cursor[0] = 0;
            A appended;
            appended.put(T(-2));
            appended.put(input);
            assert(appended.count == 4 && appended.mean == 1);
            assert(appended.centeredSumOfSquares.approxEqual(T(26)));
            assert(appended.centeredSumOfCubes.approxEqual(T(36)));
            static if (is(A == K))
                assert(appended.centeredSumOfQuarts.approxEqual(T(338)));
            assert(cursor[0] == values.length);
        }}
    }}
}

// Compile each of the 48 cases separately: expanding the entire matrix into
// one function causes excessive work in Windows DMD's optimized inliner.
// Production calls inside each case remain eligible for inlining.
version(mir_stat_test)
pragma(inline, false)
private void checkFirstObservation(T, A)() @safe pure nothrow
{
    import mir.math.common: approxEqual;
    foreach (sign; [-1, 1]) {
        A result;
        // Cubing or taking the fourth power overflows, but four copies
        // can still be summed without overflowing the mean accumulator.
        T x = sign * (T.max / 8);
        foreach (i; 0 .. 4) {
            result.put(x);
            assert(result.count == i + 1 && result.mean == x);
            assert(result.centeredSumOfSquares == 0);
            assert(result.centeredSumOfCubes == 0);
            static if (__traits(hasMember, A, "centeredSumOfQuarts"))
                assert(result.centeredSumOfQuarts == 0);
        }
    }

    // Subsequent observations still update each centered sum.
    A result;
    foreach (T x; [T(0), T(1), T(5)])
        result.put(x);
    assert(result.count == 3 && result.mean == 2);
    assert(result.centeredSumOfSquares.approxEqual(T(14), T(8) * T.epsilon, T(8) * T.epsilon));
    assert(result.centeredSumOfCubes.approxEqual(T(18), T(8) * T.epsilon, T(8) * T.epsilon));
    static if (__traits(hasMember, A, "centeredSumOfQuarts"))
        assert(result.centeredSumOfQuarts.approxEqual(T(98), T(8) * T.epsilon, T(8) * T.epsilon));
}

// A large first observation must not contaminate zero centered moments.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (sum; [Summation.naive, Summation.kbn, Summation.kb2, Summation.precise])
    static foreach (A; AliasSeq!(
        SkewnessAccumulator!(T, SkewnessAlgo.online, sum),
        SkewnessAccumulator!(T, SkewnessAlgo.hybrid, sum),
        KurtosisAccumulator!(T, KurtosisAlgo.online, sum),
        KurtosisAccumulator!(T, KurtosisAlgo.hybrid, sum)))
        checkFirstObservation!(T, A)();
}

// Count powers must not overflow while accumulating valid finite moments.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;

    // Build repeated observations through the public API without a huge array.
    static A repeated(A, T)(T value, size_t n)
    {
        A result, block;
        block.put(value);
        while (n) {
            if (n & 1)
                result.put(block);
            n >>= 1;
            if (n)
                block.put(block);
        }
        return result;
    }

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (A; AliasSeq!(
        SkewnessAccumulator!(T, SkewnessAlgo.online, Summation.naive),
        SkewnessAccumulator!(T, SkewnessAlgo.hybrid, Summation.naive),
        KurtosisAccumulator!(T, KurtosisAlgo.online, Summation.naive),
        KurtosisAccumulator!(T, KurtosisAlgo.hybrid, Summation.naive)))
    {{
        enum size_t largeCount = size_t.sizeof == 8 ? (1UL << 32) : (1U << 17);
        foreach (size_t n; [size_t(256), size_t(65536), largeCount]) {
            auto result = repeated!A(T(0), n);
            result.put(repeated!A(T(1), n));
            // Half zeros, half ones: mean=1/2, M2=n/2, M3=0, M4=n/8.
            assert(result.count == 2 * n && result.mean == T(0.5));
            assert(result.centeredSumOfSquares == T(n) / 2);
            assert(result.centeredSumOfCubes == 0);
            assert(result.skewness(true) == 0);
            static if (__traits(hasMember, A, "centeredSumOfQuarts")) {
                assert(result.centeredSumOfQuarts == T(n) / 8);
                assert(result.kurtosis(true, false).approxEqual(T(-2)));
            }
        }

        static if (size_t.sizeof == 8) {
            // Converting these counts before subtracting would lose the
            // one-observation difference in float and double.
            enum size_t group = 1UL << 54;
            auto unequal = repeated!A(T(0), group + 1);
            unequal.put(repeated!A(T(1), group));
            assert(unequal.centeredSumOfCubes.approxEqual(T(0.25)));
            auto reversed = repeated!A(T(1), group);
            reversed.put(repeated!A(T(0), group + 1));
            assert(reversed.centeredSumOfCubes.approxEqual(T(0.25)));
        }
        enum size_t n = 1U << 22;
        auto result = repeated!A(T(0), n - 1);
        result.put(T(1));
        // Evaluate the centered observations directly, independently of the
        // online count coefficients: n-1 copies of -1/n and one of 1-1/n.
        T zeroDeviation = -T(1) / n;
        T oneDeviation = 1 + zeroDeviation;
        T m2 = (n - 1) * zeroDeviation * zeroDeviation + oneDeviation * oneDeviation;
        T m3 = (n - 1) * zeroDeviation * zeroDeviation * zeroDeviation +
            oneDeviation * oneDeviation * oneDeviation;
        assert(result.centeredSumOfSquares.approxEqual(m2, T(16) * T.epsilon, T(16) * T.epsilon));
        assert(result.centeredSumOfCubes.approxEqual(m3, T(16) * T.epsilon, T(16) * T.epsilon));
        static if (__traits(hasMember, A, "centeredSumOfQuarts")) {
            T m4 = (n - 1) * zeroDeviation * zeroDeviation * zeroDeviation * zeroDeviation +
                oneDeviation * oneDeviation * oneDeviation * oneDeviation;
            assert(result.centeredSumOfQuarts.approxEqual(m4, T(16) * T.epsilon, T(16) * T.epsilon));
        }
    }}
}

// Keep the 18 cases in separate functions to reduce Windows DMD's
// optimized compilation cost. Production calls inside each case may inline.
version(mir_stat_test)
pragma(inline, false)
private void checkLargeCountKurtosis(T, KurtosisAlgo algo)() @safe pure nothrow
{
    import mir.math.common: approxEqual;

    enum size_t largeCount = size_t.sizeof == 8 ? (1UL << 33) : (1U << 17);
    foreach (size_t n; [size_t(4096), largeCount, size_t.max - 1]) {
        KurtosisAccumulator!(T, algo, Summation.naive) result;
        // Equal numbers of -1 and +1: mean=0, M2=M4=n, M3=0.
        static if (algo == KurtosisAlgo.assumeZeroMean)
            result._count = n;
        else
            result.meanAccumulator.count = n;
        static if (algo == KurtosisAlgo.naive) {
            result.summatorOfSquares.put(T(n));
            result.summatorOfQuarts.put(T(n));
        } else {
            result.centeredSummatorOfSquares.put(T(n));
            static if (algo == KurtosisAlgo.threePass)
                result.scaledSummatorOfQuarts.put(T(n));
            else
                result.centeredSummatorOfQuarts.put(T(n));
        }
        assert(result.skewness(true) == 0);
        assert(result.skewness(false) == 0);
        assert(result.kurtosis(true, true).approxEqual(T(1)));
        assert(result.kurtosis(true, false).approxEqual(T(-2)));
        // The sample excess simplifies to -2*(n-1)/(n-3).
        T sampleExcess = -T(2) * (T(n - 1) / (n - 3));
        assert(result.kurtosis(false, false).approxEqual(sampleExcess));
        assert(result.kurtosis(false, true).approxEqual(sampleExcess + 3));
        static if (algo == KurtosisAlgo.threePass) {
            T correction = T(n - 1) / n;
            assert(result.scaledSumOfQuarts(true).approxEqual(T(n)));
            assert(result.scaledSumOfQuarts(false).approxEqual(T(n) * correction * correction));
        }
    }
}

// Isolate large-count normalization from the algorithms that collect moments.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (algo; [KurtosisAlgo.naive, KurtosisAlgo.online,
        KurtosisAlgo.twoPass, KurtosisAlgo.threePass,
        KurtosisAlgo.assumeZeroMean, KurtosisAlgo.hybrid])
        checkLargeCountKurtosis!(T, algo)();
}

// Separate the cases so Windows DMD's optimized inliner does not combine the
// entire large-count test matrix into one function. Production calls inside
// each case remain eligible for inlining.
version(mir_stat_test)
pragma(inline, false)
private void checkLargeCountSkewness(T, SkewnessAlgo algo)() @safe pure nothrow
{
    enum size_t n = size_t.sizeof == 8 ? (1UL << 33) : (1U << 17);
    SkewnessAccumulator!(T, algo, Summation.naive) result;
    // Equal numbers of -1 and +1 have zero skewness.
    static if (algo == SkewnessAlgo.assumeZeroMean)
        result._count = n;
    else
        result.meanAccumulator.count = n;
    static if (algo == SkewnessAlgo.naive)
        result.summatorOfSquares.put(T(n));
    else
        result.centeredSummatorOfSquares.put(T(n));
    assert(result.skewness(true) == 0);
    assert(result.skewness(false) == 0);
}

// The naive skewness correction must not multiply counts as integers either.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (algo; [SkewnessAlgo.naive, SkewnessAlgo.online,
        SkewnessAlgo.twoPass, SkewnessAlgo.threePass,
        SkewnessAlgo.assumeZeroMean, SkewnessAlgo.hybrid])
        checkLargeCountSkewness!(T, algo)();
}

// Preserve corrections when raw or zero-centered moment sums are merged.
version(mir_stat_test)
pure nothrow
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (sum; [Summation.kbn, Summation.kb2, Summation.precise])
    static foreach (algo; [SkewnessAlgo.naive, SkewnessAlgo.assumeZeroMean])
    {{
        T large = 1;
        foreach (i; 0 .. T.mant_dig / 3 + 2)
            large *= 2;
        alias A = SkewnessAccumulator!(T, algo, sum);
        A source, result, copied;
        source.put(large);
        source.put(T(1));
        result.put(-large);
        result.put(source);
        copied.put(source);
        copied.put(-large);
        static if (algo == SkewnessAlgo.naive) {
            assert(result.sumOfCubes == 1);
            assert(copied.sumOfCubes == 1);
        } else {
            assert(result.centeredSumOfCubes == 1);
            assert(copied.centeredSumOfCubes == 1);
        }
    }}
}

// Keep the 54 compensated merge cases separately optimized instead of
// expanding them into a single large function in Windows DMD.
version(mir_stat_test)
pragma(inline, false)
private void checkCenteredMomentMerge(T, Summation sum,
    KurtosisAlgo destinationAlgo, KurtosisAlgo sourceAlgo)() pure nothrow
{
    T large = 1;
    foreach (i; 0 .. T.mant_dig)
        large *= 2;
    alias A = KurtosisAccumulator!(T, destinationAlgo, sum);
    KurtosisAccumulator!(T, sourceAlgo, sum) source;
    source.meanAccumulator.put(T(0));
    source.centeredSummatorOfSquares.put(large);
    source.centeredSummatorOfSquares.put(T(1));
    source.centeredSummatorOfCubes.put(large);
    source.centeredSummatorOfCubes.put(T(1));
    source.centeredSummatorOfQuarts.put(large);
    source.centeredSummatorOfQuarts.put(T(1));
    A result, copied, tail;
    tail.meanAccumulator.put(T(0));
    tail.centeredSummatorOfSquares.put(T(1));
    tail.centeredSummatorOfCubes.put(-large);
    tail.centeredSummatorOfQuarts.put(T(1));
    result.put(tail);
    result.put(source);
    copied.put(source);
    copied.put(tail);
    foreach (ref value; [result, copied]) {
        assert(value.centeredSumOfSquares == large + 2);
        assert(value.centeredSumOfCubes == 1);
        assert(value.centeredSumOfQuarts == large + 2);
    }
    assert(source.centeredSumOfCubes == large);
}

// Isolate state transfer from recentering with synthetic, equal-mean moments.
// These fixtures test summation state, not moments of an observation dataset.
version(mir_stat_test)
pure nothrow
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (sum; [Summation.kbn, Summation.kb2, Summation.precise])
    static foreach (destinationAlgo; [KurtosisAlgo.online, KurtosisAlgo.hybrid])
    static foreach (sourceAlgo; [KurtosisAlgo.online, KurtosisAlgo.hybrid, KurtosisAlgo.twoPass])
        checkCenteredMomentMerge!(T, sum, destinationAlgo, sourceAlgo)();
}

// Only equivalent moment representations and summator types transfer state.
version(mir_stat_test)
unittest
{
    alias Centered = KurtosisAccumulator!(double, KurtosisAlgo.twoPass, Summation.kbn);
    alias Scaled = KurtosisAccumulator!(double, KurtosisAlgo.threePass, Summation.kbn);
    static assert(canMergeShapeMoment!(double, Summation.kbn, Centered, "centeredSummatorOfCubes"));
    static assert(canMergeShapeMoment!(double, Summation.kbn, Scaled, "centeredSummatorOfSquares"));
    static assert(!canMergeShapeMoment!(double, Summation.kbn, Scaled, "centeredSummatorOfCubes"));
    static assert(!canMergeShapeMoment!(float, Summation.kbn, Centered, "centeredSummatorOfCubes"));
    static assert(!canMergeShapeMoment!(double, Summation.kb2, Centered, "centeredSummatorOfCubes"));
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    static foreach (algo; [SkewnessAlgo.online, SkewnessAlgo.hybrid])
    {{
        alias A = SkewnessAccumulator!(double, algo, Summation.kbn);
        A source, tail, result, copied;
        enum double large = 9007199254740992.0;
        source.meanAccumulator.put(0.0);
        source.centeredSummatorOfSquares.put(large);
        source.centeredSummatorOfSquares.put(1.0);
        source.centeredSummatorOfCubes.put(large);
        source.centeredSummatorOfCubes.put(1.0);
        tail.meanAccumulator.put(0.0);
        tail.centeredSummatorOfSquares.put(1.0);
        tail.centeredSummatorOfCubes.put(-large);
        result.put(tail);
        result.put(source);
        copied.put(source);
        copied.put(tail);
        assert(result.centeredSumOfSquares == large + 2);
        assert(copied.centeredSumOfSquares == large + 2);
        assert(result.centeredSumOfCubes == 1);
        assert(copied.centeredSumOfCubes == 1);
    }}
    static foreach (algo; [KurtosisAlgo.naive, KurtosisAlgo.assumeZeroMean])
    {{
        alias A = KurtosisAccumulator!(double, algo, Summation.kbn);
        A source, result;
        source.put(262144.0);
        source.put(1.0);
        result.put(-262144.0);
        result.put(source);
        static if (algo == KurtosisAlgo.naive)
            assert(result.sumOfCubes == 1);
        else
            assert(result.centeredSumOfCubes == 1);
    }}
}

// Repeated pairwise batch merges must not use the dependency's direct-state
// merge: its counter adjustment can exhaust the partial-sum buffer.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (A; AliasSeq!(
        SkewnessAccumulator!(double, SkewnessAlgo.assumeZeroMean, Summation.pairwise),
        KurtosisAccumulator!(double, KurtosisAlgo.assumeZeroMean, Summation.pairwise)))
    {{
        A batch, result;
        batch.put(1.0);
        batch.put(1.0);
        foreach (i; 0 .. 100) {
            result.put(batch);
            assert(result.count == 2 * (i + 1));
            assert(result.centeredSumOfSquares == 2.0 * (i + 1));
            assert(result.centeredSumOfCubes == 2.0 * (i + 1));
            static if (__traits(hasMember, A, "centeredSumOfQuarts"))
                assert(result.centeredSumOfQuarts == 2.0 * (i + 1));
        }
        assert(batch.count == 2);
        assert(batch.centeredSumOfSquares == 2);
    }}
}

/++
Population weighted skewness: the third weighted standardized moment.

Use skewness to describe asymmetry without measurement units. Positive values
mean that cubed deviations on the higher side of the mean contribute more;
negative values mean the lower side contributes more. Zero alone does not
establish symmetry.

Counts, probabilities, and relative weights use the same descriptive formula.
Scaling all weights by a positive constant leaves the mathematical result
unchanged. Counts agree with population skewness on expanded observations.
This function does not apply a sample bias correction; unlike skewness, it has
no population/sample selector.

Weights must be finite and nonnegative, with at least one positive weight.
Zero-weight observations are excluded, including NaN and infinity. Inputs must
be replayable arrays, slices, or nested forward ranges with matching shapes.
Constant positive-weight observations return NaN. Result-type conversion,
numerical limits, and the Windows x64 DMD temporary-array restriction follow
$(REF weightedStandardizedMoment, mir, stat, descriptive, univariate, moments).

Params:
    F = floating result/working type, or a type promoted to floating point
    summation = summation algorithm; appropriate selects pairwise summation
+/
template weightedSkewness(F, Summation summation = Summation.appropriate)
{
    /// values and weights must have matching shapes or lengths.
    stdevType!F weightedSkewness(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        import mir.stat.descriptive.univariate.moments: weightedStandardizedMoment, WeightedMomentCheckArguments;
        mixin WeightedMomentCheckArguments!(values, weights);
        return weightedStandardizedMoment!(F, 3, summation)(values, weights);
    }
}

/// ditto
template weightedSkewness(Summation summation = Summation.appropriate)
{
    import mir.primitives: DeepElementType;
    /// ditto
    stdevType!(DeepElementType!X) weightedSkewness(X, W)(scope auto ref X values, scope auto ref W weights)
        if (isIterable!X && isIterable!W)
    {
        import mir.stat.descriptive.univariate.moments: WeightedMomentCheckArguments;
        mixin WeightedMomentCheckArguments!(values, weights);
        return .weightedSkewness!(stdevType!(DeepElementType!X), summation)(values, weights);
    }
}

/// Counts and probabilities describe the same asymmetry.
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: sqrt;
    double[2] values = [0, 10], probabilities = [0.25, 0.75];
    int[2] counts = [1, 3]; // Represents [0, 10, 10, 10].
    // The mean is 7.5: the lone lower observation is farther from the mean.
    assert(weightedSkewness(values, counts).approxEqual(-2 / sqrt(3.0)));
    assert(weightedSkewness(values, probabilities).approxEqual(-2 / sqrt(3.0)));
}

/++
Population weighted kurtosis, reported as excess kurtosis by default.

Raw kurtosis is the fourth weighted standardized moment. It measures the
weighted fourth power of deviations relative to the variance squared, making
it sensitive to extreme observations. Excess kurtosis subtracts three, so a
normal distribution has excess kurtosis zero and raw kurtosis three. Use raw
kurtosis when a formula asks for the fourth standardized moment; use excess
kurtosis to compare with the normal-distribution reference.

The isRaw option follows kurtosis. Counts, probabilities, and relative weights
share the same descriptive formula, without sample bias correction. Counts
agree with population kurtosis on expanded observations, and positive common
weight scaling leaves the mathematical result unchanged.

Input, weight, result-type, numerical, and compiler restrictions follow
weightedSkewness. Constant positive-weight observations return NaN for both
raw and excess kurtosis.

Params:
    F = floating result/working type, or a type promoted to floating point
    summation = summation algorithm; appropriate selects pairwise summation
+/
template weightedKurtosis(F, Summation summation = Summation.appropriate)
{
    /++
    Params:
        values = observations
        weights = weights with matching shape or length
        isRaw = true for raw kurtosis, false for excess kurtosis (default)
    +/
    stdevType!F weightedKurtosis(X, W)(scope auto ref X values, scope auto ref W weights, bool isRaw = false)
        if (isIterable!X && isIterable!W)
    {
        import mir.stat.descriptive.univariate.moments: weightedStandardizedMoment, WeightedMomentCheckArguments;
        mixin WeightedMomentCheckArguments!(values, weights);
        const raw = weightedStandardizedMoment!(F, 4, summation)(values, weights);
        return isRaw ? raw : raw - 3;
    }
}

/// ditto
template weightedKurtosis(Summation summation = Summation.appropriate)
{
    import mir.primitives: DeepElementType;
    /// ditto
    stdevType!(DeepElementType!X) weightedKurtosis(X, W)(scope auto ref X values, scope auto ref W weights, bool isRaw = false)
        if (isIterable!X && isIterable!W)
    {
        import mir.stat.descriptive.univariate.moments: WeightedMomentCheckArguments;
        mixin WeightedMomentCheckArguments!(values, weights);
        return .weightedKurtosis!(stdevType!(DeepElementType!X), summation)(values, weights, isRaw);
    }
}

/// Choose excess or raw kurtosis for the quantity you want to report.
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    double[2] values = [0, 10];
    int[2] counts = [1, 3];
    assert(weightedKurtosis(values, counts).approxEqual(-2.0 / 3));
    assert(weightedKurtosis(values, counts, true).approxEqual(7.0 / 3));
}

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
        const T[2] values = [0, 10], weights = [1, 3], probabilities = [0.25, 0.75];
        T[4] expanded = [0, 10, 10, 10];
        const skew = weightedSkewness!(T, mode)(values, weights);
        const raw = weightedKurtosis!(T, mode)(values, weights, true);
        assert(skew.approxEqual(-2 / sqrt(T(3)), T(64) * T.epsilon, T(0)));
        assert(raw.approxEqual(T(7) / 3, T(64) * T.epsilon, T(0)));
        assert(skew.approxEqual(skewness(expanded[].sliced, true), T(64) * T.epsilon, T(0)));
        assert(raw.approxEqual(kurtosis(expanded[].sliced, true, true), T(64) * T.epsilon, T(0)));
        assert(weightedSkewness!mode(values, probabilities).approxEqual(skew));
        assert(weightedKurtosis!mode(values, probabilities).approxEqual(raw - 3));
        assert(weightedKurtosis!mode(values, weights, false).approxEqual(raw - 3));
        static assert(is(typeof(weightedSkewness(values, weights)) == T));
        static assert(is(typeof(weightedKurtosis(values, weights)) == T));
        T[2] extremes = [-T.max, T.max], equal = [1, 1];
        assert(weightedSkewness!mode(extremes, equal) == 0);
        assert(weightedKurtosis!mode(extremes, equal) == -2);
        assert(weightedKurtosis!mode(extremes, equal, true) == 1);
        T[3] excluded = [T.nan, 7, T.infinity], mask = [0, 1, 0];
        assert(weightedSkewness!mode(excluded, mask).isNaN);
        assert(weightedKurtosis!mode(excluded, mask).isNaN);
        assert(weightedKurtosis!mode(excluded, mask, true).isNaN);
    }}
    int[2] integers = [0, 10], counts = [1, 3];
    static assert(is(typeof(weightedSkewness(integers, counts)) == double));
    static assert(is(typeof(weightedKurtosis!(float)(integers, counts)) == float));
    static assert(is(typeof(weightedSkewness!(float)(integers, counts)) == float));
    assert(weightedKurtosis!(float)(integers, counts, true).approxEqual(7f / 3));
}

version(mir_stat_test)
pure unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.algorithm: filter;
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    double[6] values = [0, 1, 2, 3, 4, 5], weights = [1, 2, 3, 4, 5, 6];
    double[6] storage = [1, 4, 2, 5, 3, 6];
    static foreach (fun; AliasSeq!(weightedSkewness, weightedKurtosis))
    {{
        const expected = fun(values, weights);
        assert(fun(values[].filter!(x => true), weights[]).approxEqual(expected));
        assert(fun(values[].sliced(2, 3), storage[].sliced(3, 2).transposed).approxEqual(expected));
        assertThrown!AssertError(fun(values[0 .. 0], weights[0 .. 0]));
        assertThrown!AssertError(fun(values[], weights[0 .. 1]));
        foreach (bad; [-1.0, double.nan, double.infinity])
        {
            double[2] invalidWeights = [1, bad];
            assertThrown!AssertError(fun(values[0 .. 2], invalidWeights));
        }
        double[2] zero = [0, 0];
        assertThrown!AssertError(fun(values[0 .. 2], zero));
    }}
    static struct Input
    {
        double[] data;
        double front() { return data[0]; }
        bool empty() { return data.length == 0; }
        void popFront() { data = data[1 .. $]; }
    }
    auto input = Input(values[]);
    static assert(!__traits(compiles, weightedSkewness(input, weights)));
    static assert(!__traits(compiles, weightedKurtosis(values, input)));
}

// Every wrapper must guard its own by-value boundary before forwarding.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: isNaN;
    import std.meta: AliasSeq;
    version (DigitalMars)
    {
        version (Win64) enum rejectTemporaries = true;
        else enum rejectTemporaries = false;
    }
    else enum rejectTemporaries = false;
    static foreach (T; AliasSeq!(float, double, const(float), const(double)))
    static foreach (fun; AliasSeq!(weightedSkewness, weightedKurtosis))
    static foreach (explicitType; [false, true])
    {{
        static if (explicitType) alias args = AliasSeq!(double, Summation.naive);
        else alias args = AliasSeq!(Summation.naive);
        T[1] values = [-2], weights = [1];
        assert(fun!args(values, weights).isNaN);
        assert(fun!args(values[], weights[]).isNaN);
        static if (rejectTemporaries)
        {
            static assert(!__traits(compiles, fun!args(cast(T[1])[-2], weights)));
            static assert(!__traits(compiles, fun!args(values, cast(T[1])[1])));
            static assert(!__traits(compiles, fun!args(cast(T[1])[-2], cast(T[1])[1])));
        }
        else
            assert(fun!args(cast(T[1])[-2], cast(T[1])[1]).isNaN);
        assert(fun!args(cast(T[2])[-2, -2], cast(T[2])[1, 1]).isNaN);
    }}
}
