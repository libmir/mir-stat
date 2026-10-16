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
    import mir.math.common: exp, fabs;

    const T z = (x - location) / scale;
    const T density = logisticPDF(z);
    if (density < T.min_normal && scale < 1)
    {
        // In this tail, 1 + exp(-abs(z)) rounds to one. Divide between
        // four exponential factors. Half-exponentials can themselves become
        // subnormal when a subnormal scale restores a representable density.
        const T quarterDecay = exp(-fabs(z) / 4);
        return ((quarterDecay / scale) * quarterDecay) * quarterDecay * quarterDecay;
    }
    return density / scale;
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

// Small scales must restore representable densities in both tails.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp;
    import mir.math.common: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // 90-digit references; the logistic denominator rounds to one here.
        static if (T.max_exp == 128)
        {
            enum T exponent = 120;
            enum T tinyExponent = 191;
            enum T tinyExpected = 8.00245628516475785138969687948202631e-39L;
            enum T expected = 6.52291358810724124141821303035571164e-15L;
            enum T smallExponent = 90;
            enum T smallExpected = 1.63880252479810308607222164267638868e-39L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T exponent = 800;
            enum T tinyExponent = 1450;
            enum T tinyExpected = 3.79504197181520648721376487356687376e-307L;
            enum T expected = 1.64842823987804998528024178106714402e-40L;
            enum T smallExponent = 710;
            enum T smallExpected = 8.95257245135025991216632140458264468e-309L;
        }
        else static if (T.max_exp == 16384)
        {
            enum T exponent = 11420;
            enum T tinyExponent = 22750;
            // The reference uses a 64-bit significand; adjust for the
            // smaller minimum subnormal of wider formats.
            const T tinyExpected = ldexp(T(1.73306716928370349897078093711548328e-4930L),
                T.mant_dig - 64);
            enum T expected = 6.76714733679691181158578826955688255e-29L;
            enum T smallExponent = 11358;
            enum T smallExpected = 3.83976405302577104196109564353492666e-4933L;
        }
        const T scale = ldexp(T(1), 2 - T.max_exp);
        const T tolerance = T.mant_dig > 53 ? 2 * T.epsilon * exponent : 4 * T.epsilon;
        foreach (T sign; [T(-1), T(1)])
        {
            const T actual = logisticPDF(sign * exponent * scale, T(0), scale);
            assert(actual > 0);
            assert(fabs(actual / expected - 1) < tolerance);
            const T smallActual = logisticPDF(sign * smallExponent / 2, T(0), T(0.5));
            assert(fabs(smallActual - smallExpected) <=
                tolerance * smallExpected + 2 * T.min_normal * T.epsilon);
        }
        const T tinyScale = T.min_normal * T.epsilon;
        const T tinyActual = logisticPDF(tinyExponent * tinyScale, T(0), tinyScale);
        assert(tinyActual > 0);
        assert(fabs(tinyActual / tinyExpected - 1) < 2 * tolerance);
        assert(logisticPDF(T(0), T(0), scale) == T(0.25) / scale);
        assert(logisticPDF(T.infinity, T(0), scale) == 0);
        assert(logisticPDF(-T.infinity, T(0), scale) == 0);
        assert(logisticPDF(T(1), T(0), T.infinity) == 0);
    }}

    enum double ctfeDensity = logisticPDF(800 * 1e-300, 0.0, 1e-300);
    static assert(fabs(ctfeDensity / 3.667874584177395251e-48 - 1) < 2e-13);
    const double density = logisticPDF(800 * 1e-300, 0.0, 1e-300);
    assert(fabs(density / 3.667874584177395251e-48 - 1) < 2e-13);
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
