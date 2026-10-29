/++
This package publicly imports `mir.stat.descriptive.*` modules.

$(BOOKTABLE ,
    $(TR
        $(TH Modules)
        $(TH Description)
    )
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram)) $(TD Histograms ))
    $(TR $(TDNW $(MREF mir,stat,descriptive,aliases)) $(TD Aliases for common functions ))
    $(TR $(TDNW $(MREF mir,stat,descriptive,multivariate)) $(TD Multivariate Descriptive statistics ))
    $(TR $(TDNW $(MREF mir,stat,descriptive,univariate)) $(TD Univariate descriptive statistics, including ordinary and weighted quantiles ))
    $(TR $(TDNW $(MREF mir,stat,descriptive,weighted)) $(TD Compatibility imports for weighted means and sums ))
)

Choosing weights:
Frequency weights count repeated observations. Values [10, 20] with weights
[2, 1] represent the sample [10, 10, 20]. Use them for a table of counts when
you want the result to agree with listing every observation separately.
Relative weights give observations different amounts of influence. Weights
[2, 1] and [20, 10] describe the same relative influence. Use them, for example,
to describe the distribution of simulated outcomes with different probabilities.
They need not sum to one.

Choose the calculation that matches the question:
$(TABLE
$(TR $(TH Statistic) $(TH Choice and example))
$(TR $(TD Weighted mean) $(TD Both meanings use the same formula. Values [0, 10]
with weights [1, 3] have mean 7.5. Multiplying all weights by ten leaves it unchanged.))
$(TR $(TD Weighted sum) $(TD The same inputs give 30. Multiplying all weights by
ten gives 300. Counts give the total of the expanded sample; relative weights
give a weighted total. Normalize weights yourself if you want an average.))
$(TR $(TD Weighted variance) $(TD Use population to describe the weighted
distribution: the example gives 18.75. Use frequencySample for a sample stored
as counts: it gives 25. Use relativeSample for fixed relative weights on
independent observations with a common mean and variance: it gives 50.
Population and relativeSample are unchanged by scaling all weights;
frequencySample changes because the represented sample size changes.))
$(TR $(TD Weighted raw and central moments) $(TD Counts, probabilities, and
relative emphasis use the same descriptive formula. For values [0, 10] with
weights [1, 3], the second raw moment is 75 and the second central moment is
18.75. Scaling all weights leaves both unchanged. Use weightedVariance for
sample corrections; weightedCentralMoment always describes the distribution.))
$(TR $(TD Weighted standardized moments) $(TD Use these to compare shape
without measurement units. Order three describes skewness; order four is raw
kurtosis. Counts, probabilities, and relative weights share the same formula.
There is no sample bias correction. Subtract three from order four for excess
kurtosis, which is zero for a normal distribution.))
$(TR $(TD Weighted skewness and kurtosis) $(TD Use weightedSkewness for
asymmetry and weightedKurtosis for tail-sensitive shape. Population is the
default. WeightedMomentNormalization also selects frequencySample for repeated
sample observations or relativeSample for independent rows with fixed relative
emphasis. Sample corrections need at least three observations for skewness and
four for kurtosis, counting
expanded occurrences in frequency mode or positive-weight rows in relative mode.
These corrected standardized ratios are not generally unbiased. Kurtosis defaults
to excess; pass isRaw=true to add three. For values [0, 10] and counts [1, 3],
population excess kurtosis is -2/3, while frequency-sample excess kurtosis is 4.
Relative sample correction is unavailable for these two rows.))
$(TR $(TD Weighted coefficient of variation) $(TD Divide weighted standard deviation
by the positive weighted mean to compare relative spread. Use the same three
normalization choices as variance. For values [0, 10] and weights [1, 3], the
population ratio is about 0.577, frequency-sample ratio 0.667, and relative-sample
ratio 0.943. Sample correction does not generally make this ratio unbiased.))
$(TR $(TD Weighted covariance) $(TD Use WeightedMomentNormalization with the same
three choices as variance. For x=[0,10], y=[0,20], weights=[1,3], covariance is
37.5 for population, 50 for frequencySample, and 100 for relativeSample.))
$(TR $(TD Weighted correlation) $(TD The same inputs have correlation 1. A common
sample correction cancels, so no normalization option is needed. Scaling all
weights leaves Pearson correlation unchanged. Constant inputs return NaN.))
$(TR $(TD Weighted quantiles) $(TD Select frequencyType7 or frequencyType8 to
match the corresponding quantiles of an expanded sample. Other algorithms
treat weights as relative masses and are unchanged by scaling all weights.
For values [0, 10] with weights [2, 2], the first quartile is 0 under
frequencyType7 and 2.5 under relative-mass type7.))
$(TR $(TD Weighted histograms) $(TD Bin totals add weights, so scaling all
weights scales the totals. Relative frequencies divide by the total and stay
unchanged. Per-bin accumulators use their own statistical policies.))
)

For weighted percentograms, choosing the number of bins is separate from
choosing their quantile boundaries. The default counts positive-weight rows.
WeightedBinCountBasis.frequencyTotal uses the represented number of observations;
combine it with a frequency quantile algorithm to match expanded data.
WeightedBinCountBasis.effectiveSampleSize can request coarser bins when a few
relative weights dominate. It is a resolution heuristic, not a general measure
of uncertainty. See $(MREF mir, stat, descriptive, histogram, api) for examples.

These choices do not make every kind of weight interchangeable. In particular,
relativeSample is not a survey-design correction or the uncertainty of a mean
formed using inverse-variance weights. See
$(REF WeightedMomentNormalization, mir, stat, descriptive, weighting)
and $(REF WeightedQuantileAlgo, mir, stat, descriptive, univariate, quantiles)
for each estimator's assumptions.

Weight validity also depends on the function. Weighted means and sums allow
signed algebraic weights and evaluate zero-weight products: NaN times zero
still contributes NaN. Weighted raw, central, and standardized moments, skewness, kurtosis, variance, covariance, correlation, and quantiles require finite,
nonnegative weights and exclude zero-weight observations, even nonfinite ones.
Numeric histograms require finite, nonnegative weights; accumulator cells
decide their own rules. AssumeWeights.sumToOne in the mean API is only a
promise that skips division, not a choice of weight meaning or a validation step.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall, Ilya Yaroshenko

Copyright: 2026 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))

+/

module mir.stat.descriptive;

///
public import mir.stat.descriptive.aliases;
///
public import mir.stat.descriptive.multivariate;
///
public import mir.stat.descriptive.univariate;
///
public import mir.stat.descriptive.weighted;
///
public import mir.stat.descriptive.weighting;
///
public import mir.stat.descriptive.histogram;
