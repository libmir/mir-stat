/++
Shared normalization for weighted statistics.

$(LREF WeightedMomentNormalization) is the single policy used by weighted
variance, standard deviation, coefficient of variation, covariance, skewness, and kurtosis. The policy
selects the meaning of weights and whether to apply a sample correction;
each statistic supplies its own formula and minimum sample size.

Variance and covariance corrections assume fixed weights and independent
observations with common second moments. Relative sample skewness and kurtosis
assume a common distribution with the required finite higher moments. Their
corrected standardized ratios are not generally unbiased. See
$(REF weightedSkewness, mir, stat, descriptive, univariate, skewness_kurtosis) and
$(REF weightedKurtosis, mir, stat, descriptive, univariate, skewness_kurtosis)
for these assumptions and examples. Correlation has no normalization selector
because the common covariance/variance correction cancels.

The older WeightedVarianceNormalization name in the dispersions module remains
an alias to this enum. Existing source imports and template arguments work;
the defining module and type name in compiled symbols have changed, so rebuild
consumers when upgrading.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.weighting;

/++
Choose what the weights mean and whether to apply a sample correction.

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
See the examples on $(REF weightedVariance, mir, stat, descriptive, univariate, dispersions) for executable calls, and
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
