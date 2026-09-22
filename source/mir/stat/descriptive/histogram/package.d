/++
This module contains algorithms for creating histograms.

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

// Possible later extensions
// - Add allocating factories and marginalization for per-bin accumulators.
// - Add counters that widen dynamically when their current representation fills.
