/++
Weight interpretation for weighted statistics.

WeightInterpretation describes what a weight represents. For weighted variance,
standard deviation, coefficient of variation, covariance, skewness, and kurtosis,
the separate runtime argument isPopulation selects whether to apply a sample
correction. It defaults to false, matching the unweighted functions. Pass true
to describe the weighted distribution without a sample correction.

For values [0, 10] and weights [1, 3], population variance is 18.75.
With frequency interpretation, sample variance is 25: the represented sample
is [0, 10, 10, 10]. With relative interpretation, sample variance is 50: there
are two observations, with the second receiving three times the emphasis.
These answer different questions; the larger answer is not automatically better.

Migration:
The former WeightedMomentNormalization and WeightedVarianceNormalization names
are removed. Replace population with relative interpretation and isPopulation=true;
frequencySample with frequency interpretation and isPopulation=false; and
relativeSample with relative interpretation and isPopulation=false.
Calls without options now select relative sample correction, rather than
population moments. For example, weightedVariance(x, w) now estimates sample
variance; weightedVariance(x, w, true) preserves the former descriptive result.
Weighted kurtosis takes isPopulation before isRaw, matching unweighted kurtosis.
An old weightedKurtosis(x, w, true) call requesting raw population kurtosis
therefore becomes weightedKurtosis(x, w, true, true).

Weighted quantiles, interquartile range, and median absolute deviation also use
WeightInterpretation, separately from their quantile algorithm. Relative weights
describe emphasis on the supplied rows; frequency weights describe repeated
observations in an expanded sample. Frequency interpretation defaults to type7
and also supports type8. See
$(REF WeightedQuantileAlgo, mir, stat, descriptive, univariate, quantiles)
for examples and the different defaults for relative weights. These functions
have no population/sample correction. Dedicated weightedMedian keeps its fixed
averaged-inverse-CDF convention without an interpretation selector.

Mean and correlation retain their existing APIs. Neither needs a sample
normalization selector: the mean has no such correction, and the common
covariance/variance correction cancels in correlation.

Template option order:
Weighted variance, standard deviation, coefficient of variation, and covariance
accept the weight interpretation and calculation algorithm in either order.
Weighted skewness and kurtosis likewise accept the interpretation and summation
algorithm in either order. An explicit result type stays first; for accumulators,
the required element type stays first. Any trailing summation option on variance
and covariance stays last. Supplying only an algorithm uses relative weights;
supplying only an interpretation uses the function's default algorithm.
Weighted quantile, interquartile range, and median absolute deviation follow the
same ordering rule for interpretation and algorithm across allocation variants.

Prefer interpretation first when explaining what the weights represent. Algorithm
first is useful when adding a weight interpretation to an existing algorithm choice.
Both spellings select the same implementation and give the same result.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.weighting;

/++
Choose explicitly what weights mean. Integer-valued weights do not automatically
mean frequencies; floating-point weights such as [1.0, 3.0] can be counts.
This selection governs validation and, when requested, the sample correction.
+/
enum WeightInterpretation
{
    /++
    Each row is one observation with relative emphasis. This is the default.
    Weights must be finite and nonnegative; zero weights exclude observations.
    Multiplying all weights by a positive constant leaves the mathematical
    result unchanged, so [1, 3] and [0.25, 0.75] express the same relative weights.

    Sample variance and covariance assume fixed weights and independent
    observations with common second moments. Their denominator is W-Q/W,
    where W=sum(w) and Q=sum(w*w). At least two positive-weight rows are needed.
    Sample skewness and kurtosis require at least three and four positive-weight
    rows respectively, and a common distribution with the required moments.
    Their corrected standardized ratios are not generally unbiased.

    These corrections are not general corrections for survey designs or
    measurements with different known variances. For known probabilities or
    a descriptive distribution, pass isPopulation=true instead.
    +/
    relative,
    /++
    Each weight counts occurrences. Values [0, 10] with counts [1, 3] represent
    [0, 10, 10, 10] without expanding the input. Use this for compact samples
    of repeated observations. Sample results match the corresponding unweighted
    definitions on the expanded sample, up to floating-point rounding.

    Counts must be finite nonnegative whole numbers with a total representable
    under the statistic's numerical limits. This validation also applies with
    isPopulation=true. Do not normalize counts: [2, 6] represents eight
    observations and can give different sample corrections from [1, 3].

    Sample variance and covariance require total count above one; skewness
    requires at least three and kurtosis at least four. Population results do
    not require these sample minima. Zero counts exclude observations.
    +/
    frequency
}

/// Choose the weight meaning and calculation algorithm in either order.
@safe pure nothrow @nogc unittest
{
    import mir.stat.descriptive.univariate.dispersions: weightedVariance, VarianceAlgo;
    import mir.math.common: approxEqual;
    alias W = WeightInterpretation;
    alias A = VarianceAlgo;
    double[2] values = [0, 10], counts = [1, 3];
    assert(weightedVariance!(W.frequency, A.twoPass)(values[], counts[]).approxEqual(25));
    assert(weightedVariance!(A.twoPass, W.frequency)(values[], counts[]).approxEqual(25));
    assert(weightedVariance!(double, A.twoPass, W.frequency)(values[], counts[]).approxEqual(25));
    // An algorithm alone retains the default relative-weight interpretation.
    assert(weightedVariance!(A.twoPass)(values[], counts[]).approxEqual(50));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.stat.descriptive.univariate.dispersions:
        weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation,
        WVarianceAccumulator, VarianceAlgo;
    import mir.stat.descriptive.multivariate:
        weightedCovariance, WCovarianceAccumulator, CovarianceAlgo;
    import mir.stat.descriptive.univariate.skewness_kurtosis: weightedSkewness, weightedKurtosis;
    import mir.math.sum: Summation;
    import std.meta: AliasSeq;

    // Aliases must select the exact original template, preserving its constraints,
    // inferred attributes, auto-ref handling, defaults, and accumulator identity.
    static foreach (w; [WeightInterpretation.relative, WeightInterpretation.frequency])
    {
        static foreach (a; [VarianceAlgo.online, VarianceAlgo.twoPass, VarianceAlgo.hybrid])
        {
            static foreach (fun; AliasSeq!(weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation))
            {
                static assert(__traits(isSame, fun!(a, w), fun!(w, a)));
                static assert(__traits(isSame, fun!(float, a, w, Summation.naive), fun!(float, w, a, Summation.naive)));
                static assert(__traits(isSame, fun!a, fun!(WeightInterpretation.relative, a)));
                static assert(__traits(isSame, fun!(double, a), fun!(double, WeightInterpretation.relative, a)));
            }
            static assert(is(WVarianceAccumulator!(double, a, w, Summation.naive) ==
                WVarianceAccumulator!(double, w, a, Summation.naive)));
            static assert(is(WVarianceAccumulator!(double, a) ==
                WVarianceAccumulator!(double, WeightInterpretation.relative, a)));
        }
        static foreach (a; [CovarianceAlgo.online, CovarianceAlgo.twoPass, CovarianceAlgo.hybrid])
        {
            static assert(__traits(isSame, weightedCovariance!(a, w), weightedCovariance!(w, a)));
            static assert(__traits(isSame, weightedCovariance!(float, a, w, Summation.naive),
                weightedCovariance!(float, w, a, Summation.naive)));
            static assert(__traits(isSame, weightedCovariance!a, weightedCovariance!(WeightInterpretation.relative, a)));
            static assert(__traits(isSame, weightedCovariance!(double, a), weightedCovariance!(double, WeightInterpretation.relative, a)));
            static assert(is(WCovarianceAccumulator!(double, a, w, Summation.naive) ==
                WCovarianceAccumulator!(double, w, a, Summation.naive)));
            static assert(is(WCovarianceAccumulator!(double, a) ==
                WCovarianceAccumulator!(double, WeightInterpretation.relative, a)));
        }
        static foreach (fun; AliasSeq!(weightedSkewness, weightedKurtosis))
        {
            static assert(__traits(isSame, fun!(Summation.kbn, w), fun!(w, Summation.kbn)));
            static assert(__traits(isSame, fun!(float, Summation.kbn, w), fun!(float, w, Summation.kbn)));
            static assert(__traits(isSame, fun!(Summation.kbn), fun!(WeightInterpretation.relative, Summation.kbn)));
            static assert(__traits(isSame, fun!(double, Summation.kbn), fun!(double, WeightInterpretation.relative, Summation.kbn)));
        }
    }
    double[4] x = [1, 2, 4, 8], weights = [1, 2, 3, 4];
    alias W = WeightInterpretation;
    foreach (isPopulation; [false, true])
        assert(weightedCovariance!(CovarianceAlgo.twoPass, W.frequency)(x[], x[], weights[], isPopulation) ==
            weightedCovariance!(W.frequency, CovarianceAlgo.twoPass)(x[], x[], weights[], isPopulation));
    foreach (isPopulation; [false, true])
    {
        assert(weightedSkewness!(Summation.kbn, W.frequency)(x, weights, isPopulation) ==
            weightedSkewness!(W.frequency, Summation.kbn)(x, weights, isPopulation));
        foreach (isRaw; [false, true])
            assert(weightedKurtosis!(double, Summation.kbn, W.frequency)(x, weights, isPopulation, isRaw) ==
                weightedKurtosis!(double, W.frequency, Summation.kbn)(x, weights, isPopulation, isRaw));
    }
    // Different enum types are not interchangeable; repeated selectors are invalid.
    static assert(!__traits(compiles, weightedVariance!(CovarianceAlgo.twoPass)(x[], weights[])));
    static assert(!__traits(compiles, weightedVariance!(VarianceAlgo.twoPass, VarianceAlgo.online)(x[], weights[])));
    static assert(!__traits(compiles, weightedVariance!(W.relative, W.frequency)(x[], weights[])));
    static assert(!__traits(compiles, weightedCovariance!(VarianceAlgo.twoPass)(x[], x[], weights[])));
    static assert(!__traits(compiles, weightedCovariance!(CovarianceAlgo.twoPass, CovarianceAlgo.online)(x[], x[], weights[])));
    static assert(!__traits(compiles, weightedCovariance!(W.relative, W.frequency)(x[], x[], weights[])));
}

/// The defaults match unweighted sample statistics; true requests population results.
@safe pure nothrow @nogc unittest
{
    import mir.stat.descriptive.univariate.dispersions:
        variance, standardDeviation, coefficientOfVariation,
        weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation,
        WVarianceAccumulator;
    import mir.stat.descriptive.multivariate:
        covariance, weightedCovariance, WCovarianceAccumulator, WCorrelationAccumulator;
    import mir.stat.descriptive.univariate.skewness_kurtosis:
        skewness, kurtosis, weightedSkewness, weightedKurtosis;
    import mir.math.common: approxEqual;
    double[4] x = [1, 2, 4, 8], y = [3, 1, 6, 5], ones = [1, 1, 1, 1];
    assert(weightedCovariance(x[], y[], ones[]).approxEqual(covariance(x[], y[])));
    assert(weightedSkewness(x, ones).approxEqual(skewness(x[])));
    assert(weightedKurtosis(x, ones).approxEqual(kurtosis(x[])));
    assert(weightedVariance(x[], ones[]).approxEqual(variance(x[])));
    assert(weightedStandardDeviation(x[], ones[]).approxEqual(standardDeviation(x[])));
    assert(weightedCoefficientOfVariation(x, ones).approxEqual(coefficientOfVariation(x[])));
    static foreach (interpretation; [WeightInterpretation.relative, WeightInterpretation.frequency])
    {{
        WVarianceAccumulator!(double, interpretation) spread;
        WCovarianceAccumulator!(double, interpretation) paired;
        paired.put(x[], y[], ones[]);
        spread.put(x[], ones[]);
        // One state can answer both queries, in either order, repeatedly.
        foreach (isPopulation; [true, false, true, false])
        {
            assert(weightedSkewness!interpretation(x, ones, isPopulation).approxEqual(skewness(x[], isPopulation)));
            foreach (isRaw; [false, true])
                assert(weightedKurtosis!interpretation(x, ones, isPopulation, isRaw).approxEqual(kurtosis(x[], isPopulation, isRaw)));
            assert(paired.covariance(isPopulation).approxEqual(covariance(x[], y[], isPopulation)));
            assert(spread.variance(isPopulation).approxEqual(variance(x[], isPopulation)));
            assert(spread.standardDeviation(isPopulation).approxEqual(standardDeviation(x[], isPopulation)));
            assert(spread.coefficientOfVariation(isPopulation).approxEqual(coefficientOfVariation(x[], isPopulation)));
        }
    }}
    WCorrelationAccumulator!double correlationState;
    correlationState.put(x[], y[], ones[]);
    assert(correlationState.covariance.approxEqual(covariance(x[], y[], true)));
}

version(mir_stat_test)
@system unittest
{
    import mir.stat.descriptive.univariate.dispersions:
        weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation;
    import mir.stat.descriptive.multivariate: weightedCovariance;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.meta: AliasSeq;
    double[1] single = [7], count = [1];
    assert(weightedVariance!(WeightInterpretation.frequency)(single[], count[], true) == 0);
    assertThrown!AssertError(weightedVariance(single[], count[]));
    assert(weightedCovariance!(WeightInterpretation.frequency)(single[], single[], count[], true) == 0);
    assertThrown!AssertError(weightedCovariance(single[], single[], count[]));
    double[4] x = [1, 2, 3, 4], fractional = [1, 2, 3, 0.5];
    static foreach (fun; AliasSeq!(weightedVariance, weightedStandardDeviation,
        weightedCoefficientOfVariation))
    {
        // Population mode must still validate the count interpretation.
        assertThrown!AssertError(fun!(WeightInterpretation.frequency)(x[], fractional[], true));
    }
}

version(mir_stat_test)
@system unittest
{
    import mir.stat.descriptive.multivariate: weightedCovariance;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[4] x = [1, 2, 3, 4], fractional = [1, 2, 3, 0.5];
    assertThrown!AssertError(weightedCovariance!(WeightInterpretation.frequency)(x[], x[], fractional[], true));
}

version(mir_stat_test)
@system unittest
{
    import mir.stat.descriptive.univariate.skewness_kurtosis: weightedSkewness, weightedKurtosis;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.meta: AliasSeq;
    import std.math: isNaN;
    double[1] single = [7], count = [1];
    assert(weightedSkewness!(WeightInterpretation.frequency)(single, count, true).isNaN);
    assert(weightedKurtosis!(WeightInterpretation.frequency)(single, count, true).isNaN);
    double[4] x = [1, 2, 3, 4], fractional = [1, 2, 3, 0.5];
    static foreach (fun; AliasSeq!(weightedSkewness, weightedKurtosis))
    {
        // Population queries still require valid occurrence counts.
        assertThrown!AssertError(fun!(WeightInterpretation.frequency)(x, fractional, true));
        static assert(!__traits(compiles, fun!(WeightInterpretation.relative, WeightInterpretation.frequency)(x, fractional)));
    }
}
