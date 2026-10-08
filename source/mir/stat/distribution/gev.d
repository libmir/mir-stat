/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: Ilia Ki, John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.gev;

import mir.internal.utility: isFloatingPoint;

import mir.math.common: fabs, exp, pow, log;

/++
Computes the generalized extreme value (GEV) probability density function (PDF).

Params:
    x = value to evaluate
    mu = location
    sigma = scale
    xi = shape

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution)
+/
@safe pure nothrow @nogc
T gevPDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (xi >= 0 || x <= mu - sigma / xi, "if xi is less than zero, x must be less than or equal to mu - sigma / xi")
    in (xi <= 0 || x >= mu - sigma / xi, "if xi is greater than zero, xi must be greater than or equal to mu - sigma / xi")
{
    auto s = (x - mu) / sigma;
    if (xi.fabs <= T.min_normal)
    {
        auto t = exp(-s);
        if (t == T.infinity)
            return 0;
        return t * exp(-t) / sigma;
    }
    auto v = 1 + xi * s;
    if (v <= 0)
        return xi == -1 ? 1 / sigma : xi < -1 ? T.infinity : T(0);
    auto a = pow(v, -1 / xi);
    if (a == T.infinity)
        return 0;
    return a * exp(-a) / (v * sigma);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    gevPDF(-3, 2, 3, -0.5).shouldApprox == 0.02120353011709564;
    gevPDF(-1, 2, 3, +0.5).shouldApprox == 0.04884170370329114;
    gevPDF(-1, 2, 3, 0.0).shouldApprox == 0.05979135957800574;
}

// Checking v <= 0 branch
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    gevPDF(-1.0, 0, 1, 1).should == 0;
}

/++
Computes the generalized extreme value (GEV) cumulatve distribution function (CDF).

Params:
    x = value to evaluate
    mu = location
    sigma = scale
    xi = shape

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution)
+/
@safe pure nothrow @nogc
T gevCDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (xi >= 0 || x <= mu - sigma / xi, "if xi is less than zero, x must be less than or equal to mu - sigma / xi")
    in (xi <= 0 || x >= mu - sigma / xi, "if xi is greater than zero, xi must be greater than or equal to mu - sigma / xi")
{
    auto s = (x - mu) / sigma;
    if (xi.fabs <= T.min_normal)
        return exp(-exp(-s));
    auto v = 1 + xi * s;
    if (v <= 0)
        return xi > 0 ? 0 : 1;
    auto a = pow(v, -1 / xi);
    return exp(-a);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    gevCDF(-3, 2, 3, -0.5).shouldApprox == 0.034696685646156494;
    gevCDF(-1, 2, 3, +0.5).shouldApprox == 0.01831563888873418;
    gevCDF(-1, 2, 3, 0.0).shouldApprox == 0.06598803584531254;
}

// Checking v <= 0 branch
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    gevCDF(-1.0, 0, 1, 1).should == 0;
    gevCDF(1.0, 0, 1, -1).should == 1;
}

/++
Computes the generalized extreme value (GEV) complementary cumulatve distribution function (CCDF).

Params:
    x = value to evaluate
    mu = location
    sigma = scale
    xi = shape

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution)
+/
@safe pure nothrow @nogc
T gevCCDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (xi >= 0 || x <= mu - sigma / xi, "if xi is less than zero, x must be less than or equal to mu - sigma / xi")
    in (xi <= 0 || x >= mu - sigma / xi, "if xi is greater than zero, xi must be greater than or equal to mu - sigma / xi")
{
    return 1 - gevCDF(x, mu, sigma, xi);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    gevCCDF(-3, 2, 3, -0.5).shouldApprox == 0.965303314353844;
    gevCCDF(-1, 2, 3, +0.5).shouldApprox == 0.981684361111266;
    gevCCDF(-1, 2, 3, 0.0).shouldApprox == 0.934011964154687;
}

// Checking v <= 0 branch
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    gevCCDF(-1.0, 0, 1, 1).should == 1;
    gevCCDF(1.0, 0, 1, -1).should == 0;
}

/++
Computes the generalized extreme value (GEV) inverse cumulative distribution function (InvCDF).

Params:
    p = value to evaluate
    mu = location
    sigma = scale
    xi = shape

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution)
+/
@safe pure nothrow @nogc
T gevInvCDF(T)(const T p, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    auto logp = log(p);
    if (xi.fabs <= T.min_normal)
        return mu - sigma * log(-logp);
    return mu + (pow(-logp, -xi) - 1) * sigma / xi;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    gevInvCDF(0.034696685646156494, 2, 3, -0.5).shouldApprox == -3;
    gevInvCDF(0.01831563888873418, 2, 3, +0.5).shouldApprox == -1;
    gevInvCDF(0.06598803584531254, 2, 3, 0.0).shouldApprox == -1;
}

/++
Computes the generalized extreme value (GEV) log probability density function (LPDF).

Params:
    x = value to evaluate
    mu = location
    sigma = scale
    xi = shape

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution, Generalized Extreme Value (GEV) Distribution)
+/
@safe pure nothrow @nogc
T gevLPDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (xi >= 0 || x <= mu - sigma / xi, "if xi is less than zero, x must be less than or equal to mu - sigma / xi")
    in (xi <= 0 || x >= mu - sigma / xi, "if xi is greater than zero, xi must be greater than or equal to mu - sigma / xi")
{
    import mir.math.common: log;

    auto s = (x - mu) / sigma;
    if (xi.fabs <= T.min_normal)
    {
        auto t = exp(-s);
        if (t == T.infinity)
            return -T.infinity;
        // Avoid underflow in exp(-s) followed by log, even for finite s.
        return -s - t - log(sigma);
    }
    auto v = 1 + xi * s;
    if (v <= 0)
        return xi == -1 ? -log(sigma) : xi < -1 ? T.infinity : -T.infinity;
    // Keep the logarithm even when the corresponding power underflows.
    const T h = log(v) / xi;
    const T a = exp(-h);
    if (a == T.infinity)
        return -T.infinity;
    return -(1 + xi) * h - a - log(sigma);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    gevLPDF(-3, 2, 3, -0.5).shouldApprox == -3.85358759620891;
    gevLPDF(-1, 2, 3, +0.5).shouldApprox == -3.01917074698827;
    gevLPDF(-1, 2, 3, 0.0).shouldApprox == -2.81689411712715;
}

// Checking v <= 0 branch
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    gevLPDF(-1.0, 0, 1, 1).shouldApprox == -double.infinity;
}

// Nonzero-shape log-densities remain finite after intermediate underflow or overflow.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // The power is below even the extended-precision subnormal range.
        const T expected = T(-21217.038044136922491357270448038231720304588745L);
        assert(approxEqual(gevLPDF(T(1e12), T(0), T(1), T(1) / 1024),
            expected, 8 * T.epsilon, T(0)));

        // v == 2, but v * sigma would overflow for this finite scale.
        const T sigma = T.max * T(0.75);
        const T expectedScaled = -2 * log(T(2)) - T(0.5) - log(sigma);
        assert(approxEqual(gevLPDF(sigma, T(0), sigma, T(1)),
            expectedScaled, 8 * T.epsilon, T(0)));

        foreach (xi; [T(-2), T(-1), T(-0.5), T(0.25), T(1)])
        {
            // Ordinary interior values still agree with the density.
            const T x = T(0.125);
            assert(approxEqual(gevLPDF(x, T(0), T(2), xi),
                log(gevPDF(x, T(0), T(2), xi)), 16 * T.epsilon, 16 * T.epsilon));
        }
        assert(gevLPDF(T.infinity, T(0), T(1), T(1)) == -T.infinity);
        assert(gevLPDF(-T.infinity, T(0), T(1), T(-1)) == -T.infinity);
    }}
}

// Zero shape retains the density scaling of the general distribution.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: exp, log, approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (sigma; [T(0.25), T(1), T(3)])
        {
            const T mu = 2;
            const T z = 1;
            const T x = mu + sigma * z;
            const T standard = gevPDF(z, T(0), T(1), T(0));
            assert(approxEqual(gevPDF(x, mu, sigma, T(0)), standard / sigma,
                32 * T.epsilon, T(0)));
            assert(approxEqual(gevLPDF(x, mu, sigma, T(0)),
                gevLPDF(z, T(0), T(1), T(0)) - log(sigma),
                32 * T.epsilon, 32 * T.epsilon));
            const T expectedLog = -z - exp(-z) - log(sigma);
            assert(approxEqual(gevLPDF(x, mu, sigma, T(0)), expectedLog,
                32 * T.epsilon, 32 * T.epsilon));
            assert(approxEqual(gevPDF(x, mu, sigma, T(0)), exp(expectedLog),
                32 * T.epsilon, T(0)));
        }
    }}
}

// The upper support endpoint has a shape-dependent density limit.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: log, approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T mu = 2;
        const T sigma = 3;
        assert(gevPDF(mu + sigma, mu, sigma, T(-1)) == 1 / sigma);
        assert(gevLPDF(mu + sigma, mu, sigma, T(-1)) == -log(sigma));
        assert(gevPDF(mu + 2 * sigma, mu, sigma, T(-0.5)) == 0);
        assert(gevLPDF(mu + 2 * sigma, mu, sigma, T(-0.5)) == -T.infinity);
        assert(gevPDF(mu + sigma / 2, mu, sigma, T(-2)) == T.infinity);
        assert(gevLPDF(mu + sigma / 2, mu, sigma, T(-2)) == T.infinity);
        assert(gevPDF(mu - sigma, mu, sigma, T(1)) == 0);
        assert(gevLPDF(mu - sigma, mu, sigma, T(1)) == -T.infinity);

        // Finite log-densities survive PDF underflow in the Gumbel upper tail.
        const T x = 1000;
        assert(approxEqual(gevLPDF(x, T(0), T(1), T(0)), -x,
            8 * T.epsilon, T(0)));
        assert(gevPDF(T.infinity, T(0), T(1), T(0)) == 0);
        assert(gevLPDF(T.infinity, T(0), T(1), T(0)) == -T.infinity);
        assert(gevPDF(-T.infinity, T(0), T(1), T(0)) == 0);
        assert(gevLPDF(-T.infinity, T(0), T(1), T(0)) == -T.infinity);

        // A positive shape can also overflow the intermediate power near its lower endpoint.
        const T xi = T(1) / 512;
        const T lowerTail = -512 + T.epsilon * 512;
        assert(gevPDF(lowerTail, T(0), T(1), xi) == 0);
        assert(gevLPDF(lowerTail, T(0), T(1), xi) == -T.infinity);
    }}
}
