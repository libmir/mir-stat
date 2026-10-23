/++
One-dimensional and joint histograms for counts, weights, and per-bin statistics.

Import this package to use the public histogram modules listed below. Start with
the factories in $(MREF mir, stat, descriptive, histogram, api) to allocate counts
and populate them from observations, or supply axis instances alone to create an
empty histogram for incremental updates.

A histogram combines axes, which map coordinates to bins, with storage for the
bin contents. Numeric cells count observations or sum weights; accumulator cells
compute statistics from samples grouped by their coordinates. One coordinate
collection per axis constructs joint counts. A multidimensional observation slice
passed with a single axis instead contributes all its elements to that one axis.

$(BOOKTABLE ,
    $(TR $(TH Module) $(TH Purpose))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,api))
        $(TD Factories with garbage-collected, reference-counted, or caller-allocated storage;
        percentograms, prepared quantile axes, and marginal projections.))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,axis))
        $(TD Integral, regular, transformed, variable-width, enum, category, and quantile axes;
        bin descriptions and boundary options.))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,breaks))
        $(TD Sturges, Scott, and Freedman-Diaconis rules for choosing a bin count.))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,accumulator))
        $(TD Incremental counting, weighted insertion, per-bin accumulator cells,
        merging, and read-only bin views over supplied storage.))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,relative_frequency))
        $(TD Maintained totals, relative frequencies, one-dimensional cumulative
        frequencies, and densities for numeric bin geometry.))
    $(TR $(TDNW $(MREF mir,stat,descriptive,histogram,traits))
        $(TD Axis traits, storage extents, default counter type, and adaptive counter selection.))
)

Use regular axes for equal-width bins, variable axes for supplied boundaries,
and quantile axes or percentograms for boundaries prepared from observations.
Axis options control interval closure and supported underflow, overflow, and
circular behavior. Ordinary bins exclude the enabled underflow and overflow
cells; bin views expose ordinary bins by default.

Storage policy controls count ownership independently of axis ownership.
Borrowed axis boundaries must outlive the histograms and views using them.
For direct control over existing arrays or Mir slices, construct a
$(REF HistogramAccumulator, mir, stat, descriptive, histogram, accumulator)
or $(REF RelativeFrequencyAccumulator, mir, stat, descriptive, histogram, relative_frequency).
These constructors retain existing cell values; factory-created numeric counts
start at zero before observations are inserted.

Relative frequencies normalize over all recorded counts by default, including
enabled underflow and overflow. Ordinary-bin normalization conditions on bins
that are ordinary on every axis. Density divides bin probability by numeric bin
width or joint bin volume. Percentograms prepare quantile boundaries once;
inserting more observations does not recompute those boundaries.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

+/

module mir.stat.descriptive.histogram;

public import mir.stat.descriptive.histogram.relative_frequency;
public import mir.stat.descriptive.histogram.breaks;
public import mir.stat.descriptive.histogram.accumulator;
public import mir.stat.descriptive.histogram.axis;
public import mir.stat.descriptive.histogram.traits;
public import mir.stat.descriptive.histogram.api;
