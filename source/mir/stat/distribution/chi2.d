/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: Ilia Ki, John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/
module mir.stat.distribution.chi2;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Chi-squared probability density function (PDF).

Params:
    x = value to evaluate PDF
    k = degrees of freedom

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution)
+/
@safe pure nothrow @nogc
T chi2PDF(T)(const T x, const uint k)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (k >= 1, "k must be greater than or equal to 1")
{
    import mir.stat.distribution.gamma: gammaPDF;

    return gammaPDF(x, T(k) * 0.5f, 2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.2.chi2PDF(2).shouldApprox == 0.4524187;
}

/++
Computes the Chi-squared cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    k = degrees of freedom

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution)
+/
@safe pure nothrow @nogc
T chi2CDF(T)(const T x, const uint k)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (k >= 1, "k must be greater than or equal to 1")
{
    import mir.stat.distribution.gamma: gammaCDF;

    return gammaCDF(x, T(k) * 0.5f, 2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.2.chi2CDF(2).shouldApprox == 0.09516258;
}

/++
Computes the Chi-squared complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    k = degrees of freedom

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution)
+/
@safe pure nothrow @nogc
T chi2CCDF(T)(const T x, const uint k)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (k >= 1, "k must be greater than or equal to 1")
{
    import mir.stat.distribution.gamma: gammaCCDF;

    return gammaCCDF(x, T(k) * 0.5f , 2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.2.chi2CCDF(2).shouldApprox == 0.9048374;
}

/++
Computes the Chi-squared inverse cumulative distribution function (InvCDF).

Params:
    p = value to evaluate InvCDF
    k = degrees of freedom

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution)
+/
@safe pure nothrow @nogc
T chi2InvCDF(T)(const T p, const uint k)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (k >= 1, "k must be greater than or equal to 1")
{
    import mir.stat.distribution.gamma: gammaInvCDF;

    return gammaInvCDF(p, T(k) * 0.5f, 2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.09516258.chi2InvCDF(2).shouldApprox == 0.2;
}

/++
Computes the Chi-squared log probability density function (LPDF).

Params:
    x = value to evaluate LPDF
    k = degrees of freedom

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution)
+/
@safe pure nothrow @nogc
T chi2LPDF(T)(const T x, const uint k)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (k >= 1, "k must be greater than or equal to 1")
{
    import mir.stat.distribution.gamma: gammaLPDF;
    return gammaLPDF(x, T(k) * T(0.5), T(2));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.2.chi2LPDF(2).shouldApprox == -0.7931472;
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.0.chi2LPDF(2).shouldApprox == -0.6931472;
    0.0.chi2LPDF(1).shouldApprox == double.infinity;
    0.0.chi2LPDF(3).shouldApprox == -double.infinity;
}

// The gamma representation must preserve the same endpoint and tiny-tail limits.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    foreach (k; [1u, 2u, 4u])
    {
        assert(chi2LPDF(double.infinity, k) == -double.infinity);
        assert(chi2PDF(double.infinity, k) == 0);
    }
    assert(approxEqual(chi2CDF(cast(double)4.94065645841246544177e-324L, 1),
        1.77350488860362726888307738673937e-162, 2e-13, 0.0));
}
