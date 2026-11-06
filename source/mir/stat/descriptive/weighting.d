/++
Weight interpretation and normalization for weighted statistics.

WeightInterpretation describes what a weight represents for weighted variance,
standard deviation, coefficient of variation, and WVarianceAccumulator.
The separate runtime argument isPopulation selects whether to apply a sample
correction. It defaults to false, matching the unweighted functions. Pass true
to describe the weighted distribution without a sample correction.

For values [0, 10] and weights [1, 3], population variance is 18.75.
With frequency interpretation, sample variance is 25: the represented sample
is [0, 10, 10, 10]. With relative interpretation, sample variance is 50: there
are two observations, with the second receiving three times the emphasis.
These answer different questions; the larger answer is not automatically better.

Migration:
For these variance APIs, replace WeightedMomentNormalization.population with
WeightInterpretation.relative and isPopulation=true; frequencySample with
frequency and isPopulation=false; and relativeSample with relative and
isPopulation=false. The WeightedVarianceNormalization alias is removed.
Calls without options now select relative sample correction: weightedVariance(x, w)
estimates sample variance; weightedVariance(x, w, true) preserves the former
descriptive result. The same migration applies to standard deviation, CV, and
the corresponding accumulator getters.

Weighted covariance, skewness, and kurtosis continue to use
$(LREF WeightedMomentNormalization), with population normalization by default.
Their argument order and behavior are unchanged. Mean and correlation retain
their existing APIs; neither needs a sample normalization selector.

Template option order:
Weighted variance, standard deviation, and coefficient of variation accept the
weight interpretation and calculation algorithm in either order. An explicit
result type stays first; WVarianceAccumulator requires its element type first.
Any trailing summation option stays last. An algorithm alone uses relative
weights; an interpretation alone uses the function's default algorithm.
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

    Sample variance assumes fixed weights and independent
    observations with common second moments. Its denominator is W-Q/W,
    where W=sum(w) and Q=sum(w*w). At least two positive-weight rows are needed.

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

    Sample variance requires total count above one. Population results do
    not require this sample minimum. Zero counts exclude observations.
    +/
    frequency
}

/++
Choose normalization for weighted covariance, skewness, and kurtosis.
Weighted variance, standard deviation, and coefficient of variation instead
use $(LREF WeightInterpretation) and a separate runtime isPopulation argument.
The variance formulas below illustrate the corresponding statistical definitions.

Use population to describe the weighted distribution you have.
Use frequencySample when each weight counts actual observations in a sample.
Use relativeSample when each row is one observation, weights specify relative
emphasis, and you want a sample correction under the assumptions below.

For example, values [0, 10] with weights [1, 3] have the same weighted mean,
7.5, in all three modes. Their variances differ: 18.75 for population, 25 for
frequencySample, and 50 for relativeSample. These answer different questions;
the larger answer is not automatically more accurate.

Select the meaning explicitly. Integer weights do not automatically mean
frequencies, and floating-point weights such as [1.0, 3.0] can be frequencies.
See $(REF weightedCovariance, mir, stat, descriptive, multivariate) for executable calls, and
$(REF descriptive, mir, stat) for shared guidance on weight meanings.
+/
enum WeightedMomentNormalization
{
    /++
    Describe a weighted distribution, without a sample correction.
    This is the default.

    Use this for a probability distribution, a complete weighted dataset, or a
    descriptive summary when you do not want to estimate a population statistic
    from a sample. For example, an outcome is 0 with probability 25 percent and
    10 with probability 75 percent. Its variance is 18.75. Either [0.25, 0.75]
    or [1, 3] expresses those probabilities; weights need not sum to one.

    Multiplying every weight by the same positive constant leaves the result
    unchanged. One positive-weight observation has variance 0; skewness and
    kurtosis are undefined for constant observations and return NaN.

    Variance formula: sum(w*(x-mean)*(x-mean)) / sum(w).
    +/
    population,
    /++
    Apply a sample correction to observations stored as values and occurrence counts.

    Use this to avoid expanding repeated observations. Values [0, 10] and counts
    [1, 3] represent the four observations [0, 10, 10, 10]. The variance is 25,
    exactly the sample-variance definition used by ordinary variance on that
    expanded sample (up to floating-point rounding).

    Counts must be nonnegative whole numbers. Variance and covariance require
    total count above one, skewness at least three, and kurtosis at least four.
    Fractional counts such as 1.5 are invalid. A single stored row with count 3
    represents three identical observations and has sample variance 0.

    Do not normalize counts to sum to one. Changing [1, 3] to [2, 6] means an
    eight-observation sample, so its sample correction changes: the variance
    becomes 150/7, approximately 21.43. For variance this mode divides by
    sample size minus one, using total count as the sample size. Skewness and
    kurtosis use different corrections based on that same total count.

    Variance formula: sum(w*(x-mean)*(x-mean)) / (sum(w)-1).
    +/
    frequencySample,
    /++
    Apply a sample correction when weights give observations relative emphasis.

    Use this when each row is one independent observation and weights are fixed
    independently of the observed values. For example, two observations [0, 10]
    have preassigned weights [1, 3]: the second receives three times the emphasis,
    but it is still one observation, not three repeated observations. The
    corrected variance is 50. Weights [0.25, 0.75] give the same result.

    Variance and covariance require at least two positive-weight rows, skewness
    at least three, and kurtosis at least four. Fractional weights are allowed,
    and multiplying all weights by the same positive constant leaves the result
    unchanged. Equal positive weights recover ordinary sample variance. Unequal
    weights concentrate the information in fewer observations, which this
    correction accounts for under its statistical assumptions.

    The variance correction is unbiased when observations are independent, share a common
    mean and variance, and weights are fixed. It is not a general correction for
    survey sampling or measurements with different known variances. It estimates
    the variance of the observations, not the uncertainty of their weighted mean.

    Variance formula: write W = sum(w), Q = sum(w*w), and
    M2 = sum(w*(x-mean)*(x-mean)); the result is M2/(W-Q/W).
    +/
    relativeSample
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
    import mir.stat.descriptive.multivariate: CovarianceAlgo;
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
    }
    double[4] x = [1, 2, 4, 8], weights = [1, 2, 3, 4];
    alias W = WeightInterpretation;
    // Different enum types are not interchangeable; repeated selectors are invalid.
    static assert(!__traits(compiles, weightedVariance!(CovarianceAlgo.twoPass)(x[], weights[])));
    static assert(!__traits(compiles, weightedVariance!(VarianceAlgo.twoPass, VarianceAlgo.online)(x[], weights[])));
    static assert(!__traits(compiles, weightedVariance!(W.relative, W.frequency)(x[], weights[])));
}

/// The defaults match unweighted sample statistics; true requests population results.
@safe pure nothrow @nogc unittest
{
    import mir.stat.descriptive.univariate.dispersions:
        variance, standardDeviation, coefficientOfVariation,
        weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation,
        WVarianceAccumulator;
    import mir.math.common: approxEqual;
    double[4] x = [1, 2, 4, 8], ones = [1, 1, 1, 1];
    assert(weightedVariance(x[], ones[]).approxEqual(variance(x[])));
    assert(weightedStandardDeviation(x[], ones[]).approxEqual(standardDeviation(x[])));
    assert(weightedCoefficientOfVariation(x, ones).approxEqual(coefficientOfVariation(x[])));
    static foreach (interpretation; [WeightInterpretation.relative, WeightInterpretation.frequency])
    {{
        WVarianceAccumulator!(double, interpretation) spread;
        spread.put(x[], ones[]);
        // One state can answer both queries, in either order, repeatedly.
        foreach (isPopulation; [true, false, true, false])
        {
            assert(spread.variance(isPopulation).approxEqual(variance(x[], isPopulation)));
            assert(spread.standardDeviation(isPopulation).approxEqual(standardDeviation(x[], isPopulation)));
            assert(spread.coefficientOfVariation(isPopulation).approxEqual(coefficientOfVariation(x[], isPopulation)));
        }
    }}
}

version(mir_stat_test)
@system unittest
{
    import mir.stat.descriptive.univariate.dispersions:
        weightedVariance, weightedStandardDeviation, weightedCoefficientOfVariation;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.meta: AliasSeq;
    double[1] single = [7], count = [1];
    assert(weightedVariance!(WeightInterpretation.frequency)(single[], count[], true) == 0);
    assertThrown!AssertError(weightedVariance(single[], count[]));
    double[4] x = [1, 2, 3, 4], fractional = [1, 2, 3, 0.5];
    static foreach (fun; AliasSeq!(weightedVariance, weightedStandardDeviation,
        weightedCoefficientOfVariation))
    {
        // Population mode must still validate the count interpretation.
        assertThrown!AssertError(fun!(WeightInterpretation.frequency)(x[], fractional[], true));
    }
}
