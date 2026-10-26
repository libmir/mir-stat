/++
Factory functions for histogram counts, relative frequencies, and per-bin summaries.

This package re-exports $(MREF mir, stat, descriptive, histogram, api, gc),
$(MREF mir, stat, descriptive, histogram, api, rc), and
$(MREF mir, stat, descriptive, histogram, api, custom). The table below maps each
result to its storage-policy variants. For axes and direct accumulation over
existing storage, see $(MREF mir, stat, descriptive, histogram).

Weighted APIs take all observation inputs before weights, followed by axis or
probability configuration. Custom-allocation factories take the allocator first.
For example: `weightedHistogram(x, y, weights, xAxis, yAxis)` and
`rcWeightedPercentogram(values, weights, probabilities)`. Incremental updates use
`putWeighted(x, y, weight)` or `putWeightedSample(sample, x, y, weight)`.
Existing weight-first calls must be reordered, even when their argument types
still allow them to compile.

Weight validity depends on the result. Numeric weighted histograms require
finite, nonnegative weights but allow all-zero counts. Computing weighted
quantile boundaries for percentograms or quantile axes requires at least one
positive weight and ignores zero-weight observations. Histograms of accumulator
cells delegate weight semantics to the cell: for example,
$(REF WMeanAccumulator, mir, stat, descriptive, univariate, means) permits signed
weights and does not skip zero-weight products. These rules do not change with
the storage policy.

Choose a storage policy, then the kind of result:
$(TABLE
$(TR $(TH Result) $(TH GC storage) $(TH Reference-counted storage) $(TH Caller-selected allocator))
$(TR $(TD Counts or accumulator cells) $(TD histogram) $(TD rchistogram) $(TD makeHistogram))
$(TR $(TD Relative frequencies) $(TD relativeFrequencyHistogram) $(TD rcRelativeFrequencyHistogram) $(TD makeRelativeFrequencyHistogram))
$(TR $(TD Weighted counts or accumulator cells) $(TD weightedHistogram) $(TD rcWeightedHistogram) $(TD makeWeightedHistogram))
$(TR $(TD Weighted relative frequencies) $(TD weightedRelativeFrequencyHistogram) $(TD rcWeightedRelativeFrequencyHistogram) $(TD makeWeightedRelativeFrequencyHistogram))
$(TR $(TD Quantile-based bins) $(TD percentogram) $(TD rcpercentogram) $(TD makePercentogram))
$(TR $(TD Weighted quantile-based bins) $(TD weightedPercentogram) $(TD rcWeightedPercentogram) $(TD makeWeightedPercentogram))
$(TR $(TD Quantile axis from observations) $(TD quantileAxis) $(TD rcQuantileAxis) $(TD makeQuantileAxis))
$(TR $(TD Quantile axis from weighted observations) $(TD weightedQuantileAxis) $(TD rcWeightedQuantileAxis) $(TD makeWeightedQuantileAxis))
$(TR $(TD Quantile axis from supplied boundaries) $(TD quantileAxisFromBoundaries) $(TD rcQuantileAxisFromBoundaries) $(TD makeQuantileAxisFromBoundaries))
$(TR $(TD Projection onto selected axes) $(TD marginal) $(TD rcMarginal) $(TD makeMarginal))
)

Start with observations and an axis template or instance to count values into
one axis. Built-in arrays and Mir slices are accepted. Supply only axis instances
to histogram or relative-frequency factories to create empty counts, including
joint counts for multiple axes, then insert observations through put or putWeighted.
Populate numeric joint counts by passing one coordinate collection per explicit
axis: histogram(x, y, xAxis, yAxis). Weighted construction uses
weightedHistogram(x, y, weights, xAxis, yAxis). All input shapes must match.
A single multidimensional observation slice still contributes to one axis.

For per-bin summaries, choose an accumulator cell type such as Summator or
MeanAccumulator. Histogram factories can allocate empty cells or populate them
from samples and separate coordinate collections. Weighted factories also accept
weights. Relative-frequency factories instead require numeric counts or weights
that can be summed and normalized.

GC and RC factories manage their count storage automatically. The make factories
take an allocator first and require explicit cleanup for fixed storage.
Select AdaptiveCounts!() in histogram, relative-frequency, or percentogram factories to use
unweighted counters that widen from ubyte to ulong. Select AdaptiveCounts!ushort
(or uint or ulong) to start at a wider type. GC factories use GC ownership;
RC and custom factories use RC ownership for these adaptive counts.
Adaptive counts require no custom disposal and do not support weighted updates
or merging. Relative-frequency totals remain ulong and must not overflow. Ordinary custom histograms
expose caller-allocated counts; makePercentogram returns a wrapper whose dispose
method releases boundaries and fixed counts through the allocator; adaptive
counts are released automatically. Allocating counts does not take
ownership of borrowed axis boundaries.

Weighted percentograms use weights both to choose quantile boundaries and to
accumulate bin masses. Pass observations before weights; select a WeightedQuantileAlgo
explicitly to change the default inverseCDF definition. They use fixed numeric
counters and the same GC, RC, or explicit-disposal ownership policies as percentograms.

The default weighted percentogram chooses the number of bins from the number
of rows with positive weight. Pass WeightedBinCountBasis.frequencyTotal or
WeightedBinCountBasis.effectiveSampleSize as the final argument to select a
different sample-size basis. Callers can also calculate and supply their own
bin count. See $(REF descriptive, mir, stat) for shared guidance on weight meanings.

For example, after reweighting simulated outcomes, a few outcomes may carry
most of the probability. Kish effective sample size measures this imbalance:
it equals the number of positive-weight rows when those weights are equal and becomes
smaller as the weights become more concentrated. Using it in the bin-count
rule can request fewer bins, which can give a coarser summary of the reweighted
distribution. It can also hide detail among outcomes with smaller weights.

Frequency tables have a different interpretation. A row with weight 100
represents 100 observations of that value. Using the sum of the frequencies
in the bin-count rule gives the same requested number of bins as listing all
those observations separately. Select a frequency quantile algorithm as well
when you want the quantile boundaries to match that expanded sample.

In both examples, the policy changes only the requested number of bins. The selected
quantile algorithm still determines their boundaries, and the original weights
determine the amount accumulated in each bin.

To compare samples using the same boundaries, first construct a quantile axis from
a reference sample and probability levels or a bin count. Pass it to percentogram
or weightedPercentogram with the observations to count. Use the FromBoundaries
factories when another calculation supplies the quantiles. Preparation copies or
allocates boundaries, combines duplicates, and includes the upper quantile endpoint;
subsequent reuse leaves those boundaries unchanged. The existing convenience
percentogram factories also expose a prepared QuantileAxis that can be reused.

Quantile axes work with histogram and relative-frequency factories as ordinary
explicit axes. Selection algorithms and supplied boundaries preserve integral coordinate types.
Interpolating algorithms use floating-point coordinates. Direct histogram insertion
requires matching coordinate types. Percentogram overloads convert observations;
conversion to a floating axis preserves integral observations' interval membership,
while conversion to an integral axis requires exact representability.
Counts use the selected histogram factory's allocation policy independently of the
axis's boundary policy. Custom quantile-axis factories return an owner exposing axis
and dispose. A makePercentogram call with that axis allocates only counts and returns
a relative-frequency accumulator: dispose its fixed counts separately, and dispose
the boundary owner only after all histograms and views are finished using it.

See $(MREF mir, stat, descriptive, histogram, api, rc) for axis-selection examples,
$(MREF mir, stat, descriptive, histogram, api, gc) for GC factories, and
$(MREF mir, stat, descriptive, histogram, api, custom) for allocation and disposal.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

+/

module mir.stat.descriptive.histogram.api;

public import mir.stat.descriptive.histogram.api.rc;
public import mir.stat.descriptive.histogram.api.gc;
public import mir.stat.descriptive.histogram.api.custom;

/// Reweighted simulations can concentrate most probability on a few outcomes.
/// Use effective sample size to request coarser resolution in that situation.
/// This is an optional choice: it can also reduce detail among lighter outcomes.
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.rc: rcWeightedPercentogram, WeightedBinCountBasis;

    double[27] outcomes, weights;
    foreach (i; 0 .. outcomes.length)
        outcomes[i] = i;
    weights[] = 1;
    weights[13] = 26; // One outcome carries half the total probability.

    auto result = rcWeightedPercentogram(outcomes, weights,
        WeightedBinCountBasis.effectiveSampleSize); // Requests two bins, rather than three.
    auto explicitBins = rcWeightedPercentogram(outcomes, weights, 2);
    assert(result.counts == explicitBins.counts);
    assert(result.total == 52);
}

/// A frequency table represents repeated observations without storing every copy.
/// Use total frequency for bin selection when the requested resolution should
/// match a percentogram constructed from the expanded sample.
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.api.rc:
        rcWeightedPercentogram, rcpercentogram, WeightedBinCountBasis;
    import mir.stat.descriptive.univariate: WeightedQuantileAlgo;

    double[3] outcomes = [0, 10, 20];
    uint[3] frequencies = [2, 3, 4];
    double[9] expanded = [0, 0, 10, 10, 10, 20, 20, 20, 20];

    auto compact = rcWeightedPercentogram!(
        double, WeightedQuantileAlgo.frequencyType7)(
            outcomes, frequencies, WeightedBinCountBasis.frequencyTotal);
    auto repeated = rcpercentogram(expanded);

    assert(compact.counts == repeated.counts);
    // Repeated quantiles leave two ordinary bins in both representations.
    assert(compact.axis.N_bin == 2);
    assert(compact.axis.N_bin == repeated.axis.N_bin);
    foreach (i; 0 .. compact.axis.N_bin)
    {
        assert(compact.bins()[i].bin.low == repeated.bins()[i].bin.low);
        assert(compact.bins()[i].bin.high == repeated.bins()[i].bin.high);
    }
}
