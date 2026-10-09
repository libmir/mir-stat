/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.cauchy;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Cauchy probability density function (PDF).

Params:
    x = value to evaluate PDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution)
+/
@safe pure nothrow @nogc
T cauchyPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.constant: M_1_PI;

    const T square = x * x;
    if (square == T.infinity)
        // 1/x^2 is negligible here, but the density can still be subnormal.
        return (T(M_1_PI) / x) / x;
    return T(M_1_PI) / (1 + square);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate PDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T cauchyPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return cauchyPDF((x - location) / scale) / scale;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    cauchyPDF(-3.0).shouldApprox == 0.03183099;
    cauchyPDF(-2.0).shouldApprox == 0.06366198;
    cauchyPDF(-1.0).shouldApprox == 0.1591549;
    cauchyPDF(0.0).shouldApprox == 0.3183099;
    cauchyPDF(1.0).shouldApprox == 0.1591549;
    cauchyPDF(2.0).shouldApprox == 0.06366198;
    cauchyPDF(3.0).shouldApprox == 0.03183099;

    // Can include location/scale
    cauchyPDF(-3.0, 1, 2).shouldApprox == 0.03183099;
    cauchyPDF(-2.0, 1, 2).shouldApprox == 0.04897075;
    cauchyPDF(-1.0, 1, 2).shouldApprox == 0.07957747;
    cauchyPDF(0.0, 1, 2).shouldApprox == 0.127324;
    cauchyPDF(1.0, 1, 2).shouldApprox == 0.1591549;
    cauchyPDF(2.0, 1, 2).shouldApprox == 0.127324;
    cauchyPDF(3.0, 1, 2).shouldApprox == 0.07957747;
}

/++
Computes the Cauchy cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution)
+/
@safe pure nothrow @nogc
T cauchyCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.constant: M_1_PI;
    import std.math.trigonometry: atan;

    return 0.5 + T(M_1_PI) * atan(x);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate CDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T cauchyCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return cauchyCDF((x - location) / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    cauchyCDF(-3.0).shouldApprox == 0.1024164;
    cauchyCDF(-2.0).shouldApprox == 0.1475836;
    cauchyCDF(-1.0).shouldApprox == 0.25;
    cauchyCDF(0.0).shouldApprox == 0.5;
    cauchyCDF(1.0).shouldApprox == 0.75;
    cauchyCDF(2.0).shouldApprox == 0.8524164;
    cauchyCDF(3.0).shouldApprox == 0.8975836;

    // Can include location/scale
    cauchyCDF(-3.0, 1, 2).shouldApprox == 0.1475836;
    cauchyCDF(-2.0, 1, 2).shouldApprox == 0.187167;
    cauchyCDF(-1.0, 1, 2).shouldApprox == 0.25;
    cauchyCDF(0.0, 1, 2).shouldApprox == 0.3524164;
    cauchyCDF(1.0, 1, 2).shouldApprox == 0.5;
    cauchyCDF(2.0, 1, 2).shouldApprox == 0.6475836;
    cauchyCDF(3.0, 1, 2).shouldApprox == 0.75;
}

/++
Computes the Cauchy complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution)
+/
@safe pure nothrow @nogc
T cauchyCCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    return cauchyCDF(-x);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate CCDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T cauchyCCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return cauchyCDF((location - x) / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    cauchyCCDF(-3.0).shouldApprox == 0.8975836;
    cauchyCCDF(-2.0).shouldApprox == 0.8524164;
    cauchyCCDF(-1.0).shouldApprox == 0.75;
    cauchyCCDF(0.0).shouldApprox == 0.5;
    cauchyCCDF(1.0).shouldApprox == 0.25;
    cauchyCCDF(2.0).shouldApprox == 0.1475836;
    cauchyCCDF(3.0).shouldApprox == 0.1024164;

    // Can include location/scale
    cauchyCCDF(-3.0, 1, 2).shouldApprox == 0.8524164;
    cauchyCCDF(-2.0, 1, 2).shouldApprox == 0.812833;
    cauchyCCDF(-1.0, 1, 2).shouldApprox == 0.75;
    cauchyCCDF(0.0, 1, 2).shouldApprox == 0.6475836;
    cauchyCCDF(1.0, 1, 2).shouldApprox == 0.5;
    cauchyCCDF(2.0, 1, 2).shouldApprox == 0.3524164;
    cauchyCCDF(3.0, 1, 2).shouldApprox == 0.25;
}

/++
Computes the Cauchy inverse cumulative distribution function (InvCDF).

Uses reciprocal tangents in the tails so small probabilities are not lost
by subtracting one half.

Params:
    p = value to evaluate InvCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution)
+/
@safe pure nothrow @nogc
T cauchyInvCDF(T)(const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.constant: PI;
    import mir.math.common: fabs;
    import std.math.trigonometry: tan;

    const T centered = p - T(0.5);
    if (fabs(centered) <= T(0.25))
        return tan(T(PI) * centered);
    return cauchyInvCDF(p, T(0), T(1));
}

/++
Ditto, with location and scale parameters.

Params:
    p = value to evaluate InvCDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T cauchyInvCDF(T)(const T p, const T location, const T scale)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.constant: PI, M_1_PI;
    import std.math.trigonometry: tan;

    if (p >= T(0.25) && p <= T(0.75))
        return location + scale * tan(T(PI) * (p - T(0.5)));
    if (p == 0)
        return -T.infinity;
    if (p == 1)
        return T.infinity;
    if (p > 0 && p < 1)
    {
        if (p < T.min_normal)
        {
            // Avoid rounding pi*p to a subnormal before taking its inverse.
            // cot(pi*p) = 1/(pi*p) to far better than working precision here.
            const T ratio = scale / p;
            const T tail = ratio < T.infinity ? ratio * T(M_1_PI)
                : (scale * T(M_1_PI)) / p;
            return location - tail;
        }
        if (p < T(0.25))
            return location - scale / tan(T(PI) * p);
        return location + scale / tan(T(PI) * (1 - p));
    }
    assert(0, "Should not be here");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    cauchyInvCDF(0.0).shouldApprox == -double.infinity;
    cauchyInvCDF(0.1).shouldApprox == -3.077684;
    cauchyInvCDF(0.2).shouldApprox == -1.376382;
    cauchyInvCDF(0.3).shouldApprox == -0.7265425;
    cauchyInvCDF(0.4).shouldApprox == -0.3249197;
    cauchyInvCDF(0.5).shouldApprox == 0.0;
    cauchyInvCDF(0.6).shouldApprox == 0.3249197;
    cauchyInvCDF(0.7).shouldApprox == 0.7265425;
    cauchyInvCDF(0.8).shouldApprox == 1.376382;
    cauchyInvCDF(0.9).shouldApprox == 3.077684;
    cauchyInvCDF(1.0).shouldApprox == double.infinity;

    // Can include location/scale
    cauchyInvCDF(0.2, 1, 2).shouldApprox == -1.752764;
    cauchyInvCDF(0.4, 1, 2).shouldApprox == 0.3501606;
    cauchyInvCDF(0.6, 1, 2).shouldApprox == 1.649839;
    cauchyInvCDF(0.8, 1, 2).shouldApprox == 3.752764;
}

/++
Computes the Cauchy log probability density function (LPDF).

Params:
    x = value to evaluate LPDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cauchy_distribution, Cauchy Distribution)
+/
@safe pure nothrow @nogc
T cauchyLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: log, fabs;
    import mir.stat.constant: LOGPI;

    const T square = x * x;
    if (square == T.infinity)
        return -T(LOGPI) - 2 * log(fabs(x));
    return -T(LOGPI) - log(1 + square);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate LPDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T cauchyLPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;

    return cauchyLPDF((x - location) / scale) - log(scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    cauchyLPDF(-3.0).shouldApprox == log(0.03183099);
    cauchyLPDF(-2.0).shouldApprox == log(0.06366198);
    cauchyLPDF(-1.0).shouldApprox == log(0.1591549);
    cauchyLPDF(0.0).shouldApprox == log(0.3183099);
    cauchyLPDF(1.0).shouldApprox == log(0.1591549);
    cauchyLPDF(2.0).shouldApprox == log(0.06366198);
    cauchyLPDF(3.0).shouldApprox == log(0.03183099);

    // Can include location/scale
    cauchyLPDF(-3.0, 1, 2).shouldApprox == log(0.03183099);
    cauchyLPDF(-2.0, 1, 2).shouldApprox == log(0.04897075);
    cauchyLPDF(-1.0, 1, 2).shouldApprox == log(0.07957747);
    cauchyLPDF(0.0, 1, 2).shouldApprox == log(0.127324);
    cauchyLPDF(1.0, 1, 2).shouldApprox == log(0.1591549);
    cauchyLPDF(2.0, 1, 2).shouldApprox == log(0.127324);
    cauchyLPDF(3.0, 1, 2).shouldApprox == log(0.07957747);
}

// Extreme tails retain densities and logarithms when squaring overflows.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp, nextUp;
    import mir.math.common: approxEqual, log;
    import mir.math.constant: M_1_PI;
    import mir.stat.constant: LOGPI;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = ldexp(T(1), T.max_exp / 2);
        // 1/x^2 is negligible; evaluating the divisions separately is safe.
        const T expected = (T(M_1_PI) / x) / x;
        assert(expected > 0);
        foreach (v; [x, -x])
        {
            assert(approxEqual(cauchyPDF(v), expected, 8 * T.epsilon, 2 * nextUp(T(0))));
            assert(approxEqual(cauchyLPDF(v), -T(LOGPI) - 2 * log(x), 8 * T.epsilon, T(0)));
        }
    }}
}

// Extreme probabilities retain tail and scaled quantiles.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp, nextUp, nextDown;
    import mir.math.common: approxEqual;
    import mir.math.constant: M_1_PI;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T p = T.epsilon * T.epsilon;
        assert(approxEqual(cauchyInvCDF(p), -T(M_1_PI) / p, 8 * T.epsilon, T(0)));
        const T upper = nextDown(T(1));
        assert(approxEqual(cauchyInvCDF(upper), T(M_1_PI) / (1 - upper), 8 * T.epsilon, T(0)));
        const T tiny = nextUp(T(0));
        assert(approxEqual(cauchyInvCDF(tiny, T(0), tiny), -T(M_1_PI), 8 * T.epsilon, T(0)));
        // scale/p overflows, but division by pi leaves a finite result.
        const T scaledTail = ldexp(T(M_1_PI), T.max_exp);
        assert(approxEqual(cauchyInvCDF(T.min_normal / 2, T(0), T(2)),
            -scaledTail, 8 * T.epsilon, T(0)));
        // The normal-probability tail path also preserves small scales.
        assert(approxEqual(cauchyInvCDF(T.min_normal, T(0), T.min_normal),
            -T(M_1_PI), 8 * T.epsilon, T(0)));
        assert(cauchyInvCDF(T(0)) == -T.infinity);
        assert(cauchyInvCDF(T(1)) == T.infinity);
        assert(cauchyInvCDF(T(.5)) == 0);
        foreach (cutoff; [T(.25), T(.75)])
        {
            const T at = cauchyInvCDF(cutoff);
            assert(approxEqual(at, cutoff == T(.25) ? T(-1) : T(1), 8 * T.epsilon, T(0)));
            assert(approxEqual(cauchyInvCDF(nextDown(cutoff)), at, 8 * T.epsilon, T(0)));
            assert(approxEqual(cauchyInvCDF(nextUp(cutoff)), at, 8 * T.epsilon, T(0)));
        }
    }}
}
