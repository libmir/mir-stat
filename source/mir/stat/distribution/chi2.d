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
    import mir.math.common: fabs, exp;
    if (k >= 64 && fabs(x - T(k)) < T(k) * T(0.5))
        return exp(chi2CenteredLPDF(x, T(k)));
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
    import mir.math.common: fabs;
    if (k >= 64 && fabs(x - T(k)) < T(k) * T(0.5))
        return chi2CenteredLPDF(x, T(k));
    if (x == 0)
        return gammaLPDF(x, T(k) * T(0.5), T(2));
    if (x == T.infinity)
        return -T.infinity;
    // Retain the compact direct formula outside the cancellation-prone center.
    import mir.math.common: log;
    import mir.math.constant: LN2;
    import std.mathspecial: logGamma;
    return (T(k) * T(0.5) - 1) * (log(x) - T(LN2)) - x * T(0.5)
        - T(logGamma(T(k) * T(0.5))) - T(LN2);
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

// Near the mean, combine the linear terms before evaluating logarithms:
// a*(log(1+d)-d)-log(1+d), where a=k/2 and d=(x-k)/k.
// This avoids cancellation between terms proportional to k*log(k).
private @safe pure nothrow @nogc
T chi2CenteredLPDF(T)(const T x, const T k)
{
    pragma(inline, false);
    import mir.math.common: log;
    import mir.stat.constant: LOGSQRT2PI;
    const T d = (x - k) / k;
    const T t = d / (2 + d);
    const T t2 = t * t;
    // |t| < 1/3 here. Each extra term reduces the truncation error by
    // at least nine; choose enough terms for the significand of T.
    enum terms = (T.mant_dig + 2) / 3 + 2;
    T polynomial = T(1) / (2 * terms + 1);
    static foreach (j; 0 .. terms - 1)
        polynomial = T(1) / (2 * (terms - j - 1) + 1) + t2 * polynomial;
    const T sum = t * t2 * polynomial;
    // Reuse log(1+d)=2*(t+sum); no second logarithm is needed.
    const T a = k * T(0.5);
    const T r = 1 / a, r2 = r * r;
    // At a>=32 the first omitted Stirling term is below 1e-24.
    const T correction = r * (T(1)/12 + r2 * (-T(1)/360 + r2 * (T(1)/1260
        + r2 * (-T(1)/1680 + r2 * (T(1)/1188 + r2 * (-T(691)/360360 + r2 / 156))))));
    return a * (-d * t + 2 * sum) - 2 * (t + sum) - T(0.5) * log(a)
        - T(0.693147180559945309417232121458176568L) - T(LOGSQRT2PI) - correction;
}

// Independent 100-digit gamma-density references at a large-degree center.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual, exp;
    assert(approxEqual(chi2LPDF(1e9, 1_000_000_000u),
        -11.6271450421245176412365740098587893, 4e-15));
    assert(approxEqual(chi2PDF(1e9, 1_000_000_000u),
        8.920620579277085476024753442300458e-6, 2e-13));
    foreach (x; [33.0, 64.0, 95.0])
        assert(approxEqual(chi2PDF(x, 64), exp(chi2LPDF(x, 64)), 2e-14));
}

// Independent references exercise both sides of the centered-formula boundary.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {
        static foreach (i, x; [32, 33, 64, 95, 96])
        {{
            enum expected = [
                -8.8351203444420375730972571193646059734352672053L,
                -8.3811989257726732335953826838646688865926682666L,
                -3.3475577470837329811630613541611323630947630401L,
                -6.6027496916167946598648462268466975454853889485L,
                -6.7781393957306371398446547747663091293630599128L][i];
            assert(approxEqual(chi2LPDF(T(x), 64), T(expected), T(128)*T.epsilon));
        }}
        assert(approxEqual(chi2LPDF(T(1e9), 1_000_000_000u),
            T(-11.6271450421245176412365740098587893025485486841L), T(8)*T.epsilon));
    }
}
