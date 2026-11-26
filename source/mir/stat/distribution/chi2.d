/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Chi-squared_distribution, Chi-squared Distribution).

These functions accept a positive integer number of degrees of freedom
$(D k) and nonnegative observations. A chi-squared distribution is a gamma
distribution with shape $(D k / 2.0) and scale two. The inverse CDF accepts
both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: Ilia Ki, John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/
module mir.stat.distribution.chi2;

import mir.internal.utility: isFloatingPoint;
import mir.bignum.fp: Fp;

// Preserve an odd degree count exactly when forming the gamma shape.
private @safe pure nothrow @nogc
T chi2Shape(T, I)(const I k)
    if (is(T == Fp!bits, size_t bits) && __traits(isIntegral, I))
    in (k > 0 && k <= uint.max, "degrees of freedom must be in [1, uint.max]")
{
    auto shape = T(cast(uint)k);
    --shape.exponent;
    return shape;
}

/++
Computes the Chi-squared probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D k == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D k == 2)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == 0), $(D k > 2)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    k = positive integer degrees of freedom

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

/// The density at zero changes at two degrees of freedom.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(chi2PDF(0.0, 1u) == double.infinity);
    assert(chi2PDF(0.0, 2u) == 0.5);
    assert(chi2PDF(0.0, 3u) == 0.0);
}

/++
Computes the chi-squared density using guarded gamma arithmetic.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    I = integer type of the degrees of freedom
    x = nonnegative observation, including positive infinity
    k = degrees of freedom from one through $(D uint.max)

Returns:
    The density as $(D T).
    At zero it is positive infinity for one degree of freedom, one half for two, and zero otherwise. At positive infinity it is zero.
+/
@safe pure nothrow @nogc
T chi2PDF(T, I)(const T x, const I k)
    if (is(T == Fp!bits, size_t bits) && __traits(isIntegral, I))
{
    import mir.stat.distribution.gamma: gammaPDF;
    return gammaPDF(x, chi2Shape!T(k), T(2));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(chi2PDF(T(0), 2) == T(0.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    alias T = Fp!64;
    alias U = UInt!64;
    const x = 0.5;
    // For k = 1 at x = 1/2 the density is exp(-1/4)/sqrt(pi).
    const expected = T(false, -65, U(0xe0f7e524d2808a98UL));
    assert(chi2PDF(T(x), 1) == expected);
    assert(approxEqual(chi2PDF(x, 1), cast(double)expected, 32 * double.epsilon, 0.0));
    const center = T(U(uint.max));
    assert(chi2PDF(center, uint.max) == T(false, -81, U(0x906eba8245005164UL)));
    const inf = T(false, long.max, U(0));
    assert(chi2PDF(T(0), 1) == inf);
    assert(chi2PDF(T(0), 3) == T(0));
    assert(chi2PDF(inf, 3) == T(0));
}

/++
Computes the Chi-squared cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    k = positive integer degrees of freedom

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

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    k = positive integer degrees of freedom

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

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    k = positive integer degrees of freedom

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

Returns the natural logarithm of the density.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D k == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D k == 2)) $(TD $(D -log(2))))
    $(TR $(TD $(D x == 0), $(D k > 2)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    k = positive integer degrees of freedom

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

/++
Computes the chi-squared log density using guarded gamma arithmetic.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    I = integer type of the degrees of freedom
    x = nonnegative observation, including positive infinity
    k = degrees of freedom from one through $(D uint.max)

Returns:
    The natural logarithm of the density as $(D T).
    At zero it is positive infinity for one degree of freedom, $(D -log(2)) for two, and negative infinity otherwise. At positive infinity it is negative infinity.
+/
@safe pure nothrow @nogc
T chi2LPDF(T, I)(const T x, const I k)
    if (is(T == Fp!bits, size_t bits) && __traits(isIntegral, I))
{
    import mir.stat.distribution.gamma: gammaLPDF;
    return gammaLPDF(x, chi2Shape!T(k), T(2));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    const limit = chi2LPDF(T(0), 1);
    assert(limit.isInfinity && !limit.sign);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    alias T = Fp!64;
    alias U = UInt!64;
    const x = 0.5;
    const expected = T(true, -64, U(0xd28682473d0de85fUL));
    assert(chi2LPDF(T(x), 1) == expected);
    assert(approxEqual(chi2LPDF(x, 1), cast(double)expected, 32 * double.epsilon, 0.0));
    const negativeInfinity = T(true, long.max, U(0));
    assert(chi2LPDF(T(0), 2) == T(true, -64, U(0xb17217f7d1cf79acUL)));
    assert(chi2LPDF(T(0), 3) == negativeInfinity);
    assert(chi2LPDF(T(false, long.max, U(0)), 1) == negativeInfinity);
    alias V = Fp!128;
    enum centerLogDensity = UInt!128.fromHexString("c5b1a19bbd67fa771ad3ef8a2d4e700c");
    assert(chi2LPDF(V(U(uint.max)), uint.max) == V(true, -124, centerLogDensity));
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
    import mir.stat.internal.gamma: centeredGammaLPDF;
    return centeredGammaLPDF(k * T(0.5), (x - k) / k,
        T(0.693147180559945309417232121458176568L));
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
