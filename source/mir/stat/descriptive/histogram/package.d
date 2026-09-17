/++
This module contains algorithms for creating histograms.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

+/

module mir.stat.descriptive.histogram;

public import mir.stat.descriptive.histogram.frequency;
public import mir.stat.descriptive.histogram.breaks;
public import mir.stat.descriptive.histogram.accumulator;
public import mir.stat.descriptive.histogram.axis;
public import mir.stat.descriptive.histogram.traits;
public import mir.stat.descriptive.histogram.api;

// TODO: Construction conveniences
// - Add factories for GC-backed storage and caller-selected allocation strategies.
//
// Possible later extensions
// - Support per-bin accumulators, such as MeanAccumulator.
// - Add weighted histograms and frequencies.
// - Add counters that widen dynamically when their current representation fills.
// - Replace the Phobos sorted-range dependency in VariableAxis with a Mir
//   equivalent; VariableAxis already uses a binary-search-based lookup.
