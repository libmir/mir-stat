/++
Univariate descriptive statistics. This package preserves the original import
path and re-exports the specialized modules below. Each module keeps its
algorithms, examples, and unit tests together.

$(UL
    $(LI $(MREF mir,stat,descriptive,univariate,means))
    $(LI $(MREF mir,stat,descriptive,univariate,quantiles))
    $(LI $(MREF mir,stat,descriptive,univariate,dispersions))
    $(LI $(MREF mir,stat,descriptive,univariate,moments))
    $(LI $(MREF mir,stat,descriptive,univariate,entropies))
)

$(LEGACY_ANCHOR MapSummator)$(REF MapSummator, mir, stat, descriptive, univariate, internal, summation)


Note that used specialized summing algorithms execute more primitive operations
than vanilla summation. Therefore, if in certain cases maximum speed is required
at expense of precision, one can use $(REF_ALTTEXT $(TT Summation.fast), Summation.fast, mir, math, sum)$(NBSP).

$(SCRIPT inhibitQuickIndex = 1;)
$(SCRIPT
// Retain links to declarations and members from the former single-module page.
(function () {
    var owners = {"EntropyAccumulator": "entropies", "GMeanAccumulator": "means", "KurtosisAccumulator": "moments", "KurtosisAlgo": "moments", "MapSummator": "internal_summation", "MeanAccumulator": "means", "MomentAccumulator": "moments", "MomentAlgo": "moments", "QuantileAlgo": "quantiles", "SkewnessAccumulator": "moments", "SkewnessAlgo": "moments", "StandardizedMomentAlgo": "moments", "VarianceAccumulator": "dispersions", "VarianceAlgo": "dispersions", "WeightedQuantileAlgo": "quantiles", "centralMoment": "moments", "coefficientOfVariation": "dispersions", "dispersion": "dispersions", "entropy": "entropies", "entropyType": "entropies", "gmean": "means", "gmeanType": "means", "hmean": "means", "hmeanType": "means", "interquartileRange": "quantiles", "kurtosis": "moments", "makeQuantile": "quantiles", "makeWeightedQuantile": "quantiles", "mean": "means", "meanType": "internal_types", "median": "quantiles", "medianAbsoluteDeviation": "quantiles", "moment": "moments", "quantile": "quantiles", "quantileInto": "quantiles", "quantileType": "quantiles", "rawMoment": "moments", "rcWeightedQuantile": "quantiles", "rcquantile": "quantiles", "skewness": "moments", "standardDeviation": "dispersions", "standardizedMoment": "moments", "statType": "internal_types", "stdevType": "internal_types", "variance": "dispersions", "weightedQuantile": "quantiles"};
    var anchor = window.location.hash.substring(1);
    var symbol = anchor.replace(/^\./, '').split('.')[0];
    if (Object.prototype.hasOwnProperty.call(owners, symbol)) {
        window.location.replace('mir_stat_descriptive_univariate_' + owners[symbol] + '.html#' + anchor);
    }
})();
)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Symbols))
    $(TR $(TD Location) $(TD
        $(LEGACY_ANCHOR gmean)$(REF gmean, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR hmean)$(REF hmean, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR mean)$(REF mean, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR median)$(REF median, mir, stat, descriptive, univariate, quantiles)
    ))
    $(TR $(TD Deviation) $(TD
        $(LEGACY_ANCHOR dispersion)$(REF dispersion, mir, stat, descriptive, univariate, dispersions)
        $(LEGACY_ANCHOR entropy)$(REF entropy, mir, stat, descriptive, univariate, entropies)
        $(LEGACY_ANCHOR interquartileRange)$(REF interquartileRange, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR medianAbsoluteDeviation)$(REF medianAbsoluteDeviation, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR quantile)$(REF quantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR rcquantile)$(REF rcquantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR makeQuantile)$(REF makeQuantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR quantileInto)$(REF quantileInto, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR weightedQuantile)$(REF weightedQuantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR rcWeightedQuantile)$(REF rcWeightedQuantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR makeWeightedQuantile)$(REF makeWeightedQuantile, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR standardDeviation)$(REF standardDeviation, mir, stat, descriptive, univariate, dispersions)
        $(LEGACY_ANCHOR variance)$(REF variance, mir, stat, descriptive, univariate, dispersions)
    ))
    $(TR $(TD Higher Moments, etc.) $(TD
        $(LEGACY_ANCHOR kurtosis)$(REF kurtosis, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR skewness)$(REF skewness, mir, stat, descriptive, univariate, moments)
    ))
    $(TR $(TD Other Moment Functions) $(TD
        $(LEGACY_ANCHOR centralMoment)$(REF centralMoment, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR coefficientOfVariation)$(REF coefficientOfVariation, mir, stat, descriptive, univariate, dispersions)
        $(LEGACY_ANCHOR moment)$(REF moment, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR rawMoment)$(REF rawMoment, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR standardizedMoment)$(REF standardizedMoment, mir, stat, descriptive, univariate, moments)
    ))
    $(TR $(TD Accumulators) $(TD
        $(LEGACY_ANCHOR EntropyAccumulator)$(REF EntropyAccumulator, mir, stat, descriptive, univariate, entropies)
        $(LEGACY_ANCHOR GMeanAccumulator)$(REF GMeanAccumulator, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR KurtosisAccumulator)$(REF KurtosisAccumulator, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR MeanAccumulator)$(REF MeanAccumulator, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR MomentAccumulator)$(REF MomentAccumulator, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR SkewnessAccumulator)$(REF SkewnessAccumulator, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR VarianceAccumulator)$(REF VarianceAccumulator, mir, stat, descriptive, univariate, dispersions)
    ))
    $(TR $(TD Algorithms) $(TD
        $(LEGACY_ANCHOR KurtosisAlgo)$(REF KurtosisAlgo, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR MomentAlgo)$(REF MomentAlgo, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR QuantileAlgo)$(REF QuantileAlgo, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR WeightedQuantileAlgo)$(REF WeightedQuantileAlgo, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR SkewnessAlgo)$(REF SkewnessAlgo, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR StandardizedMomentAlgo)$(REF StandardizedMomentAlgo, mir, stat, descriptive, univariate, moments)
        $(LEGACY_ANCHOR VarianceAlgo)$(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions)
    ))
    $(TR $(TD Types) $(TD
        $(LEGACY_ANCHOR entropyType)$(REF entropyType, mir, stat, descriptive, univariate, entropies)
        $(LEGACY_ANCHOR gmeanType)$(REF gmeanType, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR hmeanType)$(REF hmeanType, mir, stat, descriptive, univariate, means)
        $(LEGACY_ANCHOR meanType)$(REF meanType, mir, stat, descriptive, univariate, internal, types)
        $(LEGACY_ANCHOR quantileType)$(REF quantileType, mir, stat, descriptive, univariate, quantiles)
        $(LEGACY_ANCHOR statType)$(REF statType, mir, stat, descriptive, univariate, internal, types)
        $(LEGACY_ANCHOR stdevType)$(REF stdevType, mir, stat, descriptive, univariate, internal, types)
    ))
))

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Several functions are borrowed from
$(HTTP mir-algorithm.$(MIR_SITE)/mir_math_stat.html, mir.math.stat). An additional
$(LEGACY_ANCHOR VarianceAlgo)$(REF VarianceAlgo, mir, stat, descriptive, univariate, dispersions) is provided in this code, which is the new default.

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
public import mir.stat.descriptive.univariate.quantiles;

///
public import mir.stat.descriptive.univariate.dispersions: dispersion, variance, standardDeviation, coefficientOfVariation, VarianceAlgo, VarianceAccumulator;

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
    import entropies = mir.stat.descriptive.univariate.entropies;
    import mir.stat.descriptive.univariate: mean, entropy, dispersion;

    static assert(__traits(isSame, mean, means.mean));
    static assert(__traits(isSame, legacy.MeanAccumulator, means.MeanAccumulator));
    static assert(__traits(isSame, legacy.QuantileAlgo, quantiles.QuantileAlgo));
    static assert(__traits(isSame, legacy.VarianceAccumulator, dispersions.VarianceAccumulator));
    static assert(__traits(isSame, legacy.MomentAccumulator, moments.MomentAccumulator));
    static assert(__traits(isSame, entropy, entropies.entropy));
    static assert(__traits(isSame, dispersion, dispersions.dispersion));
    static assert(__traits(compiles, legacy.MapSummator!("a * a", double, Summation.naive)));

    assert(mir.stat.descriptive.univariate.mean([1.0, 3.0]) == 2);
    assert(mir.stat.descriptive.univariate.dispersion([1.0, 3.0]) == 1);
    assert(mir.stat.descriptive.univariate.entropy([1.0]) == 0);
    assert(legacy.median([1.0, 3.0]) == 2);
    assert(legacy.rawMoment!1([1.0, 3.0]) == 2);
}
