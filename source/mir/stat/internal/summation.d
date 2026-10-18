/++
Internal summator merging for statistical accumulators.

`mergeSummator` combines accumulated summation state. Matching summator types
are merged directly when supported, preserving partial sums and compensation
terms. Otherwise, the source sum is converted to the destination element type
before being added. Callers manage observation counts and any additional
statistical corrections.

The helper has `package(mir.stat)` visibility and is intended for use inside
mir.stat; it is not a public application API. See
$(MREF mir,stat,descriptive,univariate) for the public statistical functions.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.summation;

import mir.math.sum : Summator, Summation;

/*
Preserve partial sums and corrections when matching summator types support a
direct merge. Otherwise retain scalar conversion to the destination element
type. In particular, decimal summators currently lack a direct merge, and
cross-type merges retain the existing conversion path and its constraints.

Capability detection is compile-time only. Counts and any statistical merge
corrections remain the caller's responsibility. Source and target must be
distinct summator objects; callers supporting self merges must first copy the
source, as MeanAccumulator's by-value merge parameter does. The source is not
modified.
*/
package(mir.stat) void mergeSummator(T, U, Summation mode)(
    ref Summator!(T, mode) target, ref const Summator!(U, mode) source)
{
    static if (is(T == U) && __traits(compiles, target += source))
        target += source;
    else
        target.put(cast(T) source);
}

// Both paths accept a const source without changing its accumulated state.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    Summator!(double, Summation.kbn) source;
    source.put(0x1p54);
    source.put(1.0);
    const saved = source;
    Summator!(double, Summation.kbn) target;
    mergeSummator(target, saved);
    target.put(-0x1p54);
    assert(target.sum == 1);
    // The source's saved correction remains intact.
    source.put(-0x1p54);
    assert(source.sum == 1);

    Summator!(double, Summation.decimal) decimalSource;
    decimalSource.put(2.0);
    const savedDecimal = decimalSource;
    Summator!(double, Summation.decimal) decimalTarget;
    decimalTarget.put(1.0);
    mergeSummator(decimalTarget, savedDecimal);
    assert(decimalTarget.sum == 3);
    assert(savedDecimal.sum == 2);
}
