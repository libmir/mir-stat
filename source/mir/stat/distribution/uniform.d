/++
This module contains algorithms for the continuous $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.uniform;

import mir.internal.utility: isFloatingPoint;

/++
Computes the uniform probability density function (PDF).

Params:
    x = value to evaluate PDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformPDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    // Opposite finite bounds can have an unrepresentable full width.
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return T(0.5) / (upper * T(0.5) - lower * T(0.5));
    return 1.0L / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformPDF == 1);
    assert(0.5.uniformPDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformPDF(1.0, 3.0).approxEqual(0.5));
}

/++
Computes the uniform cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformCDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (x * T(0.5) - lower * T(0.5))
            / (upper * T(0.5) - lower * T(0.5));
    return (x - lower) / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformCDF == 0.5);
    assert(0.5.uniformCDF(0.0, 1.5).approxEqual(1.0 / 3));
    assert(2.5.uniformCDF(1.0, 3.0).approxEqual(3.0 / 4));
}

/++
Computes the uniform complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformCCDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (upper * T(0.5) - x * T(0.5))
            / (upper * T(0.5) - lower * T(0.5));
    return (upper - x) / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformCCDF == 0.5);
    assert(0.5.uniformCCDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformCCDF(1.0, 3.0).approxEqual(1.0 / 4));
}

/++
Computes the uniform inverse cumulative distribution function (InvCDF)

Params:
    p = value to evaluate InvCDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformInvCDF(T)(const T p, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (lower < upper, "lower must be less than upper")
{
    if (p == 0) return lower;
    if (p == 1) return upper;
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (1 - p) * lower + p * upper;
    return lower + p * width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformInvCDF == 0.5);
    assert((1.0 / 3).uniformInvCDF(0.0, 1.5).approxEqual(0.5));
    assert((3.0 / 4).uniformInvCDF(1.0, 3.0).approxEqual(2.5));
}

/++
Computes the uniform log probability density function (LPDF)

Params:
    x = value to evaluate LPDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformLPDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    import mir.math.common: log;
    import mir.math.constant: LN2;

    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return -log(upper * T(0.5) - lower * T(0.5)) - T(LN2);
    return -log(width);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(0.5.uniformLPDF == 0);
    assert(0.5.uniformLPDF(0.0, 1.5).approxEqual(-log(1.5)));
    assert(1.5.uniformLPDF(1.0, 3.0).approxEqual(-log(2.0)));
}

// Finite bounds remain usable even when their difference overflows.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, log;
    import mir.math.constant: LN2;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T m = T.max;
        assert(uniformCDF(T(0), -m, m) == T(0.5));
        assert(uniformCCDF(T(0), -m, m) == T(0.5));
        assert(uniformCDF(-m, -m, m) == 0);
        assert(uniformCDF(m, -m, m) == 1);
        assert(uniformCCDF(-m, -m, m) == 1);
        assert(uniformCCDF(m, -m, m) == 0);
        assert(uniformPDF(T(0), -m, m) > 0);
        assert(approxEqual(uniformPDF(T(0), -m, m), T(0.5)/m,
            T(8)*T.epsilon, T(0)));
        assert(approxEqual(uniformLPDF(T(0), -m, m), -log(m)-T(LN2),
            T(8)*T.epsilon, T(0)));
        assert(uniformInvCDF(T(0), -m, m) == -m);
        assert(uniformInvCDF(T(1), -m, m) == m);
        assert(uniformInvCDF(T(0.5), -m, m) == 0);
        // Unequal bounds prevent a symmetry-only implementation from passing.
        const T lower = -m, upper = m*T(0.5);
        foreach (p; [T(0.25), T(0.5), T(0.75)])
        {
            const T x = uniformInvCDF(p, lower, upper);
            assert(x >= lower && x <= upper);
            assert(approxEqual(uniformCDF(x, lower, upper), p, T(8)*T.epsilon));
            assert(approxEqual(uniformCCDF(x, lower, upper), 1-p, T(8)*T.epsilon));
        }
        // Exact inverse endpoints must not depend on cancellation in the width.
        assert(uniformInvCDF(T(1), -m*T(0.5), T(1)) == 1);
    }}
}

// Narrow intervals can round interior quantiles to a bound, but must retain
// exact probability endpoints, bounded results, and monotonicity.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T tiny = nextUp(T(0));
        const T[2][7] intervals = [[T(0), tiny], [tiny, 2*tiny], [-tiny, tiny],
            [T(1), nextUp(T(1))], [-T.max, T.max],
            [-T.max, T.max/T(2)], [-T.max/T(2), T(-1)]];
        foreach (bounds; intervals)
        {
            const T lower = bounds[0], upper = bounds[1];
            assert(uniformCDF(lower, lower, upper) == 0);
            assert(uniformCDF(upper, lower, upper) == 1);
            assert(uniformCCDF(lower, lower, upper) == 1);
            assert(uniformCCDF(upper, lower, upper) == 0);
            T previous = lower;
            foreach (p; [T(0), tiny, T.epsilon/T(2), T(0.25), T(0.5),
                T(0.75), nextDown(T(1)), T(1)])
            {
                const T x = uniformInvCDF(p, lower, upper);
                assert(x >= lower && x <= upper);
                assert(x >= previous);
                previous = x;
            }
            assert(uniformInvCDF(T(0), lower, upper) == lower);
            assert(uniformInvCDF(T(1), lower, upper) == upper);
        }
    }}
}
