/**
 * Normal Distribution
 *
 * Copyright: Based on the CEPHES math library, which is
 *            Copyright (C) 1994 Stephen L. Moshier (moshier@world.std.com).
 * Authors:   Stephen L. Moshier, ported to D by Don Clugston and David Nadlinger. Adopted to Mir by Ilya Yaroshenko.
*/
/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.normal;

///
public import mir.math.func.normal: normalPDF, normalCDF, normalInvCDF;

import mir.internal.utility: isFloatingPoint;

/++
Computes the normal complementary cumulative distribution function (CCDF)

Params:
    x = value to evaluate CCDF
    mean = mean
    stdDev = standard deviation

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution)
+/
@safe pure nothrow @nogc
T normalCCDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    return normalCCDF(standardizedNormal(x, mean, stdDev));
}

/// ditto
@safe pure nothrow @nogc
T normalCCDF(T)(const T a)
    if (isFloatingPoint!T)
{
    return normalCDF(-a);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.normalCCDF.approxEqual(1 - normalCDF(0.5)));
    assert(0.5.normalCCDF(0, 1.5).approxEqual(1 - normalCDF(0.5, 0, 1.5)));
    assert(1.5.normalCCDF(1, 3).approxEqual(1 - normalCDF(1.5, 1, 3)));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert((-3.0).normalCCDF.approxEqual(1 - normalCDF(-3.0)));
    assert(3.0.normalCCDF.approxEqual(1 - normalCDF(3.0)));
    assert((-7.0).normalCCDF(1, 4).approxEqual(1 - normalCDF(-7.0, 1, 4)));
    assert(9.0.normalCCDF(1, 4).approxEqual(1 - normalCDF(9.0, 1, 4)));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(normalCCDF(double.infinity).approxEqual(1 - normalCDF(double.infinity)));
    assert(normalCCDF(-double.infinity).approxEqual(1 - normalCDF(-double.infinity)));
}

/++
Computes the normal log probability density function (LPDF)

Params:
    x = value to evaluate LPDF
    mean = mean
    stdDev = standard deviation

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution)
+/
@safe pure nothrow @nogc
T normalLPDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    import mir.math.common: log;
    return normalLPDF(standardizedNormal(x, mean, stdDev)) - log(stdDev);
}

/// ditto
@safe pure nothrow @nogc
T normalLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.stat.constant: LOGSQRT2PI;

    // Halve before squaring: x*x can overflow while x*x/2 still fits.
    return (-T(0.5) * x) * x - T(LOGSQRT2PI);
}

// Divide first only when subtraction overflows for finite observations.
// In that case x and mean have opposite signs, so the divisions cannot
// introduce cancellation between infinities. Keep the usual rounding otherwise.
private @safe pure nothrow @nogc
T standardizedNormal(T)(T x, T mean, T stdDev)
{
    import mir.math.common: fabs;
    const T delta = x - mean;
    if (fabs(delta) == T.infinity && fabs(x) <= T.max && fabs(mean) <= T.max)
        return x / stdDev - mean / stdDev;
    return delta / stdDev;
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: fabs, sqrt, log;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // x*x overflows, but the halved square and log density are finite.
        const T x = sqrt(T.max) * T(1.125);
        const T expected = -(x / 2) * x;
        assert(fabs(normalLPDF(x) / expected - 1) <= 4 * T.epsilon);
        const T scale = T.max * T(0.75);
        assert(fabs(normalLPDF(scale, -scale, scale) -
            (normalLPDF(T(2)) - log(scale))) <= 4 * T.epsilon * log(scale));
        assert(fabs(normalCCDF(scale, -scale, scale) / normalCCDF(T(2)) - 1)
            <= 4 * T.epsilon);
        assert(normalLPDF(T.infinity) == -T.infinity);
        assert(normalLPDF(-T.infinity) == -T.infinity);
        assert(normalCCDF(T.infinity, T(0), T(1)) == 0);
        assert(normalCCDF(-T.infinity, T(0), T(1)) == 1);
        assert(normalLPDF(T.nan) != normalLPDF(T.nan));
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(0.5.normalLPDF.approxEqual(log(normalPDF(0.5))));
    assert(0.5.normalLPDF(0, 1.5).approxEqual(log(normalPDF(0.5, 0, 1.5))));
    assert(1.5.normalLPDF(1, 3).approxEqual(log(normalPDF(1.5, 1, 3))));
}
