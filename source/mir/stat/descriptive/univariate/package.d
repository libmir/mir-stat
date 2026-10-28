/++
Univariate descriptive statistics.

Import `mir.stat.descriptive.univariate` to access the functions below, or import
a specialized module directly. The package re-exports the public declarations
so existing imports remain valid.

Note that used specialized summing algorithms execute more primitive operations
than vanilla summation. Therefore, if in certain cases maximum speed is required
at expense of precision, one can use $(REF_ALTTEXT $(TT Summation.fast), Summation.fast, mir, math, sum)$(NBSP).

## Grouped reductions and result storage

Use $(NDSLICEREF topology, byDim) or
$(NDSLICEREF topology, alongDim) to select groups, then
$(NDSLICEREF topology, map) to calculate a statistic for each group.
For a matrix, `x.byDim!1.map!mean` calculates column means, whereas
`x.mean` returns one mean over the numeric matrix.

The mapped view is lazy: accessing an element calculates that group's statistic.
Repeated access can repeat the calculation, and changes to the input affect later
results. Keep the underlying observations alive while using the view.
To calculate once and retain results, use `slice` for GC-managed storage,
`rcslice` for reference-counted storage, `makeSlice` for a custom allocator,
or assign into an existing output slice. These choices use the same statistical
calculation; they differ in storage and ownership.

`sliced` creates or reshapes a view; it does not materialize a mapped result.
The following example shows all four ways to store the calculated column means.
The caller-provided output must have the matching shape and should not overlap
input observations that have yet to be read.

------
import mir.stat.descriptive.univariate.means : mean;
import mir.ndslice.slice : sliced;
import mir.ndslice.topology : byDim, alongDim, map;
import mir.ndslice.allocation : slice, rcslice, makeSlice;
import std.experimental.allocator : dispose;
import std.experimental.allocator.mallocator : Mallocator;

double[6] data = [1, 2, 3, 4, 5, 6];
auto x = data[].sliced(2, 3);
assert(x.mean == 3.5); // One scalar over the whole matrix.

auto means = x.byDim!1.map!mean;
assert(means == [2.5, 3.5, 4.5]);
assert(x.alongDim!0.map!mean == means);

auto gcMeans = means.slice;
auto rcMeans = means.rcslice;
auto customMeans = makeSlice(Mallocator.instance, means);
scope(exit) Mallocator.instance.dispose(customMeans.field);

double[3] buffer;
auto output = buffer[].sliced;
output[] = means;

assert(gcMeans == [2.5, 3.5, 4.5]);
assert(rcMeans == gcMeans);
assert(customMeans == gcMeans);
assert(output == gcMeans);

// The view reads current input; stored results retain calculated values.
data[0] = 101;
assert(means[0] == 52.5);
assert(gcMeans[0] == 2.5);
assert(rcMeans[0] == 2.5);
assert(customMeans[0] == 2.5);
assert(output[0] == 2.5);

------

For weighted statistics, apply the weights explicitly within each group, for
example `x.byDim!1.map!(group => group.wmean(w))`. Here `w` supplies one
weight per row and is reused for each column; it is not implicitly broadcast
over the matrix. The same storage choices apply to the mapped results.

Direct grouped overloads are a separate proposal tracked in
$(LINK2 https://github.com/libmir/mir-stat/issues/226, issue 226).
No new overload or allocation family is needed to use the composition above.

$(SCRIPT inhibitQuickIndex = 1;)
$(SCRIPT
// Retain links to declarations and members from the former single-module page.
(function () {
    var owners = {"EntropyAccumulator": "entropies", "GMeanAccumulator": "means", "KurtosisAccumulator": "skewness_kurtosis", "KurtosisAlgo": "skewness_kurtosis", "MapSummator": "internal_summation", "MeanAccumulator": "means", "MomentAccumulator": "moments", "MomentAlgo": "moments", "QuantileAlgo": "quantiles", "SkewnessAccumulator": "skewness_kurtosis", "SkewnessAlgo": "skewness_kurtosis", "StandardizedMomentAlgo": "moments", "VarianceAccumulator": "dispersions", "VarianceAlgo": "dispersions", "WeightedQuantileAlgo": "quantiles", "centralMoment": "moments", "coefficientOfVariation": "dispersions", "dispersion": "dispersions", "entropy": "entropies", "entropyType": "entropies", "gmean": "means", "gmeanType": "means", "hmean": "means", "hmeanType": "means", "interquartileRange": "quantiles", "kurtosis": "skewness_kurtosis", "makeQuantile": "quantiles", "makeWeightedQuantile": "quantiles", "mean": "means", "meanType": "internal_types", "median": "quantiles", "medianAbsoluteDeviation": "quantiles", "moment": "moments", "quantile": "quantiles", "quantileInto": "quantiles", "quantileType": "quantiles", "rawMoment": "moments", "rcWeightedQuantile": "quantiles", "rcquantile": "quantiles", "skewness": "skewness_kurtosis", "standardDeviation": "dispersions", "standardizedMoment": "moments", "statType": "internal_types", "stdevType": "internal_types", "variance": "dispersions", "weightedQuantile": "quantiles"};
    var anchor = window.location.hash.substring(1);
    var symbol = anchor.replace(/^\./, '').split('.')[0];
    if (Object.prototype.hasOwnProperty.call(owners, symbol)) {
        window.location.replace('mir_stat_descriptive_univariate_' + owners[symbol] + '.html#' + anchor);
    }
})();
)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Module) $(TH Public declarations))
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,means) $(BR)
        $(SMALL Arithmetic, geometric, harmonic, and weighted means.))
        $(TD
            $(B Functions:)
            $(LEGACY_ANCHOR mean)$(REF mean, mir, stat, descriptive, univariate, means)$(NBSP)
            $(LEGACY_ANCHOR gmean)$(REF gmean, mir, stat, descriptive, univariate, means)$(NBSP)
            $(LEGACY_ANCHOR hmean)$(REF hmean, mir, stat, descriptive, univariate, means)$(NBSP)
            $(REF wmean, mir, stat, descriptive, univariate, means)$(NBSP)
            $(BR)
            $(B Accumulators:)
            $(LEGACY_ANCHOR MeanAccumulator)$(REF MeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
            $(LEGACY_ANCHOR GMeanAccumulator)$(REF GMeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
            $(REF WMeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
            $(BR)
            $(B Weight assumptions:)
            $(REF AssumeWeights, mir, stat, descriptive, univariate, means)$(NBSP)
        )
    )
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,sums) $(BR)
        $(SMALL Weighted sums.)) $(TD
        $(B Function:)
        $(REF wsum, mir, stat, descriptive, univariate, sums)$(NBSP)
        $(BR)
        $(B Accumulator:)
        $(REF WSummator, mir, stat, descriptive, univariate, sums)$(NBSP)))
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,quantiles) $(BR)
        $(SMALL Medians, quantiles, and robust measures of spread.))
        $(TD
            $(B Ordinary quantiles:)
            $(LEGACY_ANCHOR median)$(REF median, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR quantile)$(REF quantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR rcquantile)$(REF rcquantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR makeQuantile)$(REF makeQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR quantileInto)$(REF quantileInto, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(BR)
            $(B Weighted quantiles:)
            $(LEGACY_ANCHOR weightedQuantile)$(REF weightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR rcWeightedQuantile)$(REF rcWeightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR makeWeightedQuantile)$(REF makeWeightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(BR)
            $(B Spread:)
            $(LEGACY_ANCHOR interquartileRange)$(REF interquartileRange, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR medianAbsoluteDeviation)$(REF medianAbsoluteDeviation, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(BR)
            $(B Algorithms:)
            $(LEGACY_ANCHOR QuantileAlgo)$(REF QuantileAlgo, mir, stat, descriptive, univariate, quantiles)$(NBSP)
            $(LEGACY_ANCHOR WeightedQuantileAlgo)$(REF WeightedQuantileAlgo, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        )
    )
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,dispersions) $(BR)
        $(SMALL Variance, standard deviation, and related measures.))
        $(TD
            $(B Functions:)
            $(REF variance, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF standardDeviation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF weightedVariance, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF weightedStandardDeviation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF coefficientOfVariation, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF dispersion, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(BR)
            $(B Accumulator:)
            $(REF VarianceAccumulator, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF WVarianceAccumulator, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(BR)
            $(B Algorithm:)
            $(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions)$(NBSP)
            $(REF WeightedVarianceNormalization, mir, stat, descriptive, univariate, dispersions)$(NBSP)
        )
    )
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,moments) $(BR)
        $(SMALL Raw, central, and standardized moments.))
        $(TD
            $(B Functions:)
            $(REF weightedRawMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(REF weightedCentralMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(REF weightedStandardizedMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(LEGACY_ANCHOR rawMoment)$(REF rawMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(LEGACY_ANCHOR centralMoment)$(REF centralMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(LEGACY_ANCHOR standardizedMoment)$(REF standardizedMoment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(LEGACY_ANCHOR moment)$(REF moment, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(BR)
            $(B Accumulator:)
            $(LEGACY_ANCHOR MomentAccumulator)$(REF MomentAccumulator, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(BR)
            $(B Algorithms:)
            $(LEGACY_ANCHOR MomentAlgo)$(REF MomentAlgo, mir, stat, descriptive, univariate, moments)$(NBSP)
            $(LEGACY_ANCHOR StandardizedMomentAlgo)$(REF StandardizedMomentAlgo, mir, stat, descriptive, univariate, moments)$(NBSP)
        )
    )
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,skewness_kurtosis) $(BR)
        $(SMALL Skewness and kurtosis.))
        $(TD
            $(B Functions:)
            $(LEGACY_ANCHOR skewness)$(REF skewness, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
            $(LEGACY_ANCHOR kurtosis)$(REF kurtosis, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
            $(BR)
            $(B Accumulators:)
            $(LEGACY_ANCHOR SkewnessAccumulator)$(REF SkewnessAccumulator, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
            $(LEGACY_ANCHOR KurtosisAccumulator)$(REF KurtosisAccumulator, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
            $(BR)
            $(B Algorithms:)
            $(LEGACY_ANCHOR SkewnessAlgo)$(REF SkewnessAlgo, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
            $(LEGACY_ANCHOR KurtosisAlgo)$(REF KurtosisAlgo, mir, stat, descriptive, univariate, skewness_kurtosis)$(NBSP)
        )
    )
    $(TR $(TD $(MREF mir,stat,descriptive,univariate,entropies) $(BR)
        $(SMALL Entropy and its accumulator.))
        $(TD
            $(B Function:)
            $(REF entropy, mir, stat, descriptive, univariate, entropies)$(NBSP)
            $(BR)
            $(B Accumulator:)
            $(REF EntropyAccumulator, mir, stat, descriptive, univariate, entropies)$(NBSP)
        )
    )
))

$(B Summation utility)

$(LEGACY_ANCHOR MapSummator)$(REF MapSummator, mir, stat, descriptive, univariate, internal, summation) applies a function to each input before summing.
It remains publicly available through this package; its implementation is in
$(MREF mir,stat,descriptive,univariate,internal,summation).

$(B Internal type helpers)

These helpers are internal to Mir: `quantileType` has `package(mir.stat)`
visibility, and the others have `package(mir)` visibility. They are listed here
for existing internal documentation links and are not public application APIs.

$(UL
    $(LI $(REF entropyType, mir, stat, descriptive, univariate, entropies))
    $(LI $(LEGACY_ANCHOR gmeanType)$(REF gmeanType, mir, stat, descriptive, univariate, means))
    $(LI $(LEGACY_ANCHOR hmeanType)$(REF hmeanType, mir, stat, descriptive, univariate, means))
    $(LI $(LEGACY_ANCHOR meanType)$(REF meanType, mir, stat, descriptive, univariate, internal, types))
    $(LI $(LEGACY_ANCHOR quantileType)$(REF quantileType, mir, stat, descriptive, univariate, quantiles))
    $(LI $(LEGACY_ANCHOR statType)$(REF statType, mir, stat, descriptive, univariate, internal, types))
    $(LI $(LEGACY_ANCHOR stdevType)$(REF stdevType, mir, stat, descriptive, univariate, internal, types))
)

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Several functions are borrowed from
$(HTTP mir-algorithm.$(MIR_SITE)/mir_math_stat.html, mir.math.stat). An additional
$(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions) is provided in this code, which is the new default.

Authors: John Michael Hall, Ilya Yaroshenko

Copyright: 2022-3 Mir Stat Authors.

Macros:
LEGACY_ANCHOR = <span id="$0"></span><span id=".$0"></span>
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.univariate;

///
public import mir.math.sum: Summation;

///
public import mir.stat.descriptive.univariate.means;

///
public import mir.stat.descriptive.univariate.sums;

///
public import mir.stat.descriptive.univariate.quantiles;

///
public import mir.stat.descriptive.univariate.dispersions: dispersion, variance, standardDeviation, coefficientOfVariation, VarianceAlgo, VarianceAccumulator,
    weightedVariance, weightedStandardDeviation, WVarianceAccumulator,
    WeightedVarianceNormalization, WeightedMomentNormalization;

///
public import mir.stat.descriptive.univariate.moments;

///
public import mir.stat.descriptive.univariate.entropies: entropy, EntropyAccumulator, entropyType;

///
public import mir.stat.descriptive.univariate.internal.types;

///
public import mir.stat.descriptive.univariate.internal.summation;

// Keep the historical import path usable alongside direct family imports.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import legacy = mir.stat.descriptive.univariate;
    import means = mir.stat.descriptive.univariate.means;
    import quantiles = mir.stat.descriptive.univariate.quantiles;
    import dispersions = mir.stat.descriptive.univariate.dispersions;
    import moments = mir.stat.descriptive.univariate.moments;
    import skewnessKurtosis = mir.stat.descriptive.univariate.skewness_kurtosis;
    import entropies = mir.stat.descriptive.univariate.entropies;
    import mir.stat.descriptive.univariate: mean, entropy, dispersion;

    static assert(__traits(isSame, mean, means.mean));
    static assert(__traits(isSame, legacy.MeanAccumulator, means.MeanAccumulator));
    static assert(__traits(isSame, legacy.QuantileAlgo, quantiles.QuantileAlgo));
    static assert(__traits(isSame, legacy.VarianceAccumulator, dispersions.VarianceAccumulator));
    static assert(__traits(isSame, legacy.weightedVariance, dispersions.weightedVariance));
    static assert(__traits(isSame, legacy.weightedStandardDeviation, dispersions.weightedStandardDeviation));
    static assert(__traits(isSame, legacy.WVarianceAccumulator, dispersions.WVarianceAccumulator));
    static assert(is(legacy.WeightedMomentNormalization == dispersions.WeightedVarianceNormalization));
    assert(legacy.weightedVariance([0.0, 10.0], [1.0, 3.0]) == 18.75);
    static assert(__traits(isSame, legacy.MomentAccumulator, moments.MomentAccumulator));
    static assert(__traits(isSame, moments.skewness, skewnessKurtosis.skewness));
    static assert(__traits(isSame, legacy.skewness, skewnessKurtosis.skewness));
    static assert(__traits(isSame, moments.kurtosis, skewnessKurtosis.kurtosis));
    static assert(__traits(isSame, legacy.kurtosis, skewnessKurtosis.kurtosis));
    static assert(__traits(isSame, moments.SkewnessAccumulator, skewnessKurtosis.SkewnessAccumulator));
    static assert(__traits(isSame, legacy.SkewnessAccumulator, skewnessKurtosis.SkewnessAccumulator));
    static assert(__traits(isSame, moments.KurtosisAccumulator, skewnessKurtosis.KurtosisAccumulator));
    static assert(__traits(isSame, legacy.KurtosisAccumulator, skewnessKurtosis.KurtosisAccumulator));
    static assert(__traits(isSame, moments.SkewnessAlgo, skewnessKurtosis.SkewnessAlgo));
    static assert(__traits(isSame, legacy.SkewnessAlgo, skewnessKurtosis.SkewnessAlgo));
    static assert(__traits(isSame, moments.KurtosisAlgo, skewnessKurtosis.KurtosisAlgo));
    static assert(__traits(isSame, legacy.KurtosisAlgo, skewnessKurtosis.KurtosisAlgo));
    static assert(__traits(isSame, entropy, entropies.entropy));
    static assert(__traits(isSame, dispersion, dispersions.dispersion));
    static assert(__traits(compiles, legacy.MapSummator!("a * a", double, Summation.naive)));

    assert(mir.stat.descriptive.univariate.mean([1.0, 3.0]) == 2);
    assert(mir.stat.descriptive.univariate.dispersion([1.0, 3.0]) == 1);
    assert(mir.stat.descriptive.univariate.entropy([1.0]) == 0);
    assert(legacy.median([1.0, 3.0]) == 2);
    assert(legacy.rawMoment!1([1.0, 3.0]) == 2);
}

// Keep the grouped-reduction overview example executable.
version(mir_stat_test)
unittest
{
    import mir.stat.descriptive.univariate.means : mean;
    import mir.ndslice.slice : sliced;
    import mir.ndslice.topology : byDim, alongDim, map;
    import mir.ndslice.allocation : slice, rcslice, makeSlice;
    import std.experimental.allocator : dispose;
    import std.experimental.allocator.mallocator : Mallocator;

    double[6] data = [1, 2, 3, 4, 5, 6];
    auto x = data[].sliced(2, 3);
    assert(x.mean == 3.5); // One scalar over the whole matrix.

    auto means = x.byDim!1.map!mean;
    assert(means == [2.5, 3.5, 4.5]);
    assert(x.alongDim!0.map!mean == means);

    auto gcMeans = means.slice;
    auto rcMeans = means.rcslice;
    auto customMeans = makeSlice(Mallocator.instance, means);
    scope(exit) Mallocator.instance.dispose(customMeans.field);

    double[3] buffer;
    auto output = buffer[].sliced;
    output[] = means;

    assert(gcMeans == [2.5, 3.5, 4.5]);
    assert(rcMeans == gcMeans);
    assert(customMeans == gcMeans);
    assert(output == gcMeans);

    // The view reads current input; stored results retain calculated values.
    data[0] = 101;
    assert(means[0] == 52.5);
    assert(gcMeans[0] == 2.5);
    assert(rcMeans[0] == 2.5);
    assert(customMeans[0] == 2.5);
    assert(output[0] == 2.5);
}
