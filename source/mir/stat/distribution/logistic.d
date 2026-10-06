/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.logistic;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Logistic probability density function (PDF).

Params:
    x = value to evaluate PDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp, fabs;

    const T exp_x = exp(-fabs(x));
    return exp_x / ((1 + exp_x) * (1 + exp_x));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate PDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return logisticPDF((x - location) / scale) / scale;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticPDF(-2.0).shouldApprox == 0.1049936;
    logisticPDF(-1.0).shouldApprox == 0.1966119;
    logisticPDF(-0.5).shouldApprox == 0.2350037;
    logisticPDF(0.0).shouldApprox == 0.25;
    logisticPDF(0.5).shouldApprox == 0.2350037;
    logisticPDF(1.0).shouldApprox == 0.1966119;
    logisticPDF(2.0).shouldApprox == 0.1049936;

    // Can also provide location/scale parameters
    logisticPDF(-1.0, 2.0, 3.0).shouldApprox == 0.06553731;
    logisticPDF(1.0, 2.0, 3.0).shouldApprox == 0.08106072;
    logisticPDF(4.0, 2.0, 3.0).shouldApprox == 0.07471913;
}

/++
Computes the Logistic cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp;

    if (x < 0)
    {
        const T exp_x = exp(x);
        return exp_x / (1 + exp_x);
    }
    return 1 / (1 + exp(-x));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate CDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T logisticCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return logisticCDF((x - location) / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticCDF(-2.0).shouldApprox == 0.1192029;
    logisticCDF(-1.0).shouldApprox == 0.2689414;
    logisticCDF(-0.5).shouldApprox == 0.3775407;
    logisticCDF(0.0).shouldApprox == 0.5;
    logisticCDF(0.5).shouldApprox == 0.6224593;
    logisticCDF(1.0).shouldApprox == 0.7310586;
    logisticCDF(2.0).shouldApprox == 0.8807971;

    // Can also provide location/scale parameters
    logisticCDF(-1.0, 2.0, 3.0).shouldApprox == 0.2689414;
    logisticCDF(1.0, 2.0, 3.0).shouldApprox == 0.4174298;
    logisticCDF(4.0, 2.0, 3.0).shouldApprox == 0.6607564;
}

/++
Computes the Logistic complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    return logisticCDF(-x);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate CCDF
    location = location parameter
    scale = scale parameter
+/
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return logisticCCDF((x - location) / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticCCDF(-2.0).shouldApprox == 0.8807971;
    logisticCCDF(-1.0).shouldApprox == 0.7310586;
    logisticCCDF(-0.5).shouldApprox == 0.6224593;
    logisticCCDF(0.0).shouldApprox == 0.5;
    logisticCCDF(0.5).shouldApprox == 0.3775407;
    logisticCCDF(1.0).shouldApprox == 0.2689414;
    logisticCCDF(2.0).shouldApprox == 0.1192029;

    // Can also provide location/scale parameters
    logisticCCDF(-1.0, 2.0, 3.0).shouldApprox == 0.7310586;
    logisticCCDF(1.0, 2.0, 3.0).shouldApprox == 0.5825702;
    logisticCCDF(4.0, 2.0, 3.0).shouldApprox == 0.3392436;
}

/++
Computes the Logistic inverse cumulative distribution function (InvCDF).

Params:
    p = value to evaluate InvCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticInvCDF(T)(const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;

    return log(p / (1 - p));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    p = value to evaluate InvCDF
    location = location parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticInvCDF(T)(const T p, const T location, const T scale)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (scale > 0, "scale must be greater than zero")
{
    return location + scale * logisticInvCDF(p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticInvCDF(0.0).shouldApprox == -double.infinity;
    logisticInvCDF(0.25).shouldApprox == -1.098612;
    logisticInvCDF(0.5).shouldApprox == 0.0;
    logisticInvCDF(0.75).shouldApprox == 1.098612;
    logisticInvCDF(1.0).shouldApprox == double.infinity;

    // Can also provide location/scale parameters
    logisticInvCDF(0.2, 2, 3).shouldApprox == -2.158883;
    logisticInvCDF(0.4, 2, 3).shouldApprox == 0.7836047;
    logisticInvCDF(0.6, 2, 3).shouldApprox == 3.216395;
    logisticInvCDF(0.8, 2, 3).shouldApprox == 6.158883;
}

/++
Computes the Logistic log probability density function (LPDF).

Params:
    x = value to evaluate LPDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp, fabs;
    import mir.math.internal.log1p: log1p;

    const T magnitude = fabs(x);
    return -magnitude - 2 * log1p(exp(-magnitude));
}

// Evaluate tails without overflowing exp(-x), including representable small
// probabilities that would be lost by taking the reciprocal of exp(x).
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: exp, approxEqual;
    import std.math: isNaN;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = 1000;
        assert(logisticPDF(-x) == logisticPDF(x));
        assert(logisticLPDF(-x) == -x);
        assert(logisticLPDF(x) == -x);
        assert(logisticCCDF(-x) == 1);
        assert(logisticCDF(x) == 1);
        assert(logisticPDF(T.infinity) == 0);
        assert(logisticPDF(-T.infinity) == 0);
        assert(logisticCDF(-T.infinity) == 0);
        assert(logisticCCDF(-T.infinity) == 1);
        assert(logisticLPDF(-T.infinity) == -T.infinity);
        assert(isNaN(logisticPDF(T.nan)));
        assert(isNaN(logisticCDF(T.nan)));
        assert(isNaN(logisticCCDF(T.nan)));
        assert(isNaN(logisticLPDF(T.nan)));
        const T tail = T.max_exp * T(0.695);
        const T expected = exp(-tail);
        assert(expected > 0);
        assert(approxEqual(logisticCDF(-tail) / expected, T(1), T.epsilon * 8, T(0)));
        assert(approxEqual(logisticPDF(-tail) / expected, T(1), T.epsilon * 8, T(0)));
        assert(logisticCCDF(tail) == logisticCDF(-tail));
        assert(logisticPDF(-x, T(0), T(1)) == logisticPDF(-x));
        assert(logisticLPDF(-x, T(0), T(1)) == -x);
    }}
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Params:
    x = value to evaluate LPDF
    location = location parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;

    return logisticLPDF((x - location) / scale) - log(scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    logisticLPDF(-2.0).shouldApprox == log(0.1049936);
    logisticLPDF(-1.0).shouldApprox == log(0.1966119);
    logisticLPDF(-0.5).shouldApprox == log(0.2350037);
    logisticLPDF(0.0).shouldApprox == log(0.25);
    logisticLPDF(0.5).shouldApprox == log(0.2350037);
    logisticLPDF(1.0).shouldApprox == log(0.1966119);
    logisticLPDF(2.0).shouldApprox == log(0.1049936);

    // Can also provide location/scale parameters
    logisticLPDF(-1.0, 2.0, 3.0).shouldApprox == log(0.06553731);
    logisticLPDF(1.0, 2.0, 3.0).shouldApprox == log(0.08106072);
    logisticLPDF(4.0, 2.0, 3.0).shouldApprox == log(0.07471913);
}
