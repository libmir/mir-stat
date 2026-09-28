/++
Factory functions for histogram counts, relative frequencies, and per-bin summaries.

Choose a storage policy, then the kind of result:
$(TABLE
$(TR $(TH Result) $(TH GC storage) $(TH Reference-counted storage) $(TH Caller-selected allocator))
$(TR $(TD Counts or accumulator cells) $(TD histogram) $(TD rchistogram) $(TD makeHistogram))
$(TR $(TD Relative frequencies) $(TD relativeFrequencyHistogram) $(TD rcRelativeFrequencyHistogram) $(TD makeRelativeFrequencyHistogram))
$(TR $(TD Weighted counts or accumulator cells) $(TD weightedHistogram) $(TD rcWeightedHistogram) $(TD makeWeightedHistogram))
$(TR $(TD Weighted relative frequencies) $(TD weightedRelativeFrequencyHistogram) $(TD rcWeightedRelativeFrequencyHistogram) $(TD makeWeightedRelativeFrequencyHistogram))
$(TR $(TD Quantile-based bins) $(TD percentogram) $(TD rcpercentogram) $(TD makePercentogram))
$(TR $(TD Projection onto selected axes) $(TD marginal) $(TD rcMarginal) $(TD makeMarginal))
)

Start with observations and an axis template or instance to count values into
one axis. Built-in arrays and Mir slices are accepted. Supply only axis instances
to histogram or relative-frequency factories to create empty counts, including
joint counts for multiple axes, then insert observations through put or putWeighted.
Populate numeric joint counts by passing one coordinate collection per explicit
axis: histogram(x, y, xAxis, yAxis). Weighted construction uses
weightedHistogram(weights, x, y, xAxis, yAxis). All input shapes must match.
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

See $(REF rc, mir, stat, descriptive, histogram, api) for axis-selection examples,
$(REF gc, mir, stat, descriptive, histogram, api) for GC factories, and
$(REF custom, mir, stat, descriptive, histogram, api) for allocation and disposal.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

+/

module mir.stat.descriptive.histogram.api;

public import mir.stat.descriptive.histogram.api.rc;
public import mir.stat.descriptive.histogram.api.gc;
public import mir.stat.descriptive.histogram.api.custom;
