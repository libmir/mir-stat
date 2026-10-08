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

Evaluates the upper tail directly, rather than subtracting the CDF from one,
so small probabilities are retained.

Implementation_Notes:
    With $(D s = (x - mu) / sigma), computes the interior power through
    $(D h = log(1 + xi * s) / xi) and $(D exp(-h)). For float and double
    precision, uses $(D log1p) when $(D abs(xi * s) < 1.0 / 8) to avoid
    cancellation in the logarithm. Extended precision always uses $(D log1p).
    The choice follows $(D T.mant_dig), since $(D real) varies between targets.
    DMD 2.102 uses a compensated logarithm instead of its problematic
    $(D log1p) implementation.

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
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    import std.math: log1p;

    // Evaluate the tail before the CDF can round to one.
    const T s = (x - mu) / sigma;
    if (xi.fabs <= T.min_normal)
        return oneMinusExpNeg(exp(-s));
    const T u = xi * s;
    const T v = 1 + u;
    if (v <= 0)
        return xi > 0 ? 1 : 0;

    T h;
    if (fabs(u) < T.epsilon)
        // log1p(u) / xi approaches s, including when u underflows to zero.
        h = s;
    else if (u == T.infinity)
        // Finite xi and s can overflow their product while h remains finite.
        // For infinite s this also retains the appropriate infinite h.
        h = (log(fabs(xi)) + log(fabs(s))) / xi;
    else
    {
        static if (__VERSION__ == 2102)
            // mir.math.internal.log1p also avoids this frontend's log1p.
            // Its plain log(1 + u) fallback would lose small shapes here.
            // The tiny-u branch above ensures v - 1 is nonzero.
            h = s * (log(v) / (v - 1));
        else static if (T.mant_dig > 53)
            h = log1p(u) / xi;
        else
            h = fabs(u) < T(0.125) ? log1p(u) / xi : log(v) / xi;
    }
    return oneMinusExpNeg(exp(-h));
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

// Direct tails retain small probabilities for zero, positive, and negative shapes.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T expected = T(4.248354255291588986304977843631582169098337L * 1e-18L);
        assert(approxEqual(gevCCDF(T(40), T(0), T(1), T(0)) / expected,
            T(1), 8 * T.epsilon, T(0)));
        assert(approxEqual(gevCCDF(T(122), T(2), T(3), T(0)) / expected,
            T(1), 8 * T.epsilon, T(0)));

        const T x = T(1e20);
        // Exponentiation amplifies error in the logarithm by its magnitude.
        assert(approxEqual(gevCCDF(x, T(0), T(1), T(1)) / (1 / x),
            T(1), 4 * T.epsilon * log(x), T(0)));
        const T nearUpper = 4 * (1 - T.epsilon);
        const T tiny = T.epsilon * T.epsilon * T.epsilon * T.epsilon;
        assert(approxEqual(gevCCDF(nearUpper, T(0), T(1), T(-0.25)) / tiny,
            T(1), 4 * T.epsilon * fabs(log(tiny)), T(0)));

        assert(gevCCDF(T(4), T(0), T(1), T(-0.25)) == 0);
        assert(gevCCDF(T(-4), T(0), T(1), T(0.25)) == 1);
        assert(gevCCDF(T.infinity, T(0), T(1), T(0)) == 0);
        assert(gevCCDF(-T.infinity, T(0), T(1), T(0)) == 1);
        assert(gevCCDF(T.infinity, T(0), T(1), T(1)) == 0);
        assert(gevCCDF(-T.infinity, T(0), T(1), T(-1)) == 1);

        foreach (xi; [T(-2), T(-1), T(-0.5), T(0), T(0.25), T(1)])
            assert(approxEqual(gevCDF(T(0.125), T(0), T(2), xi)
                + gevCCDF(T(0.125), T(0), T(2), xi), T(1), 8 * T.epsilon, T(0)));
    }}
}

// Small shapes approach the Gumbel limit without rounding 1 + xi * s to one.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T expected = T(0.30779937244465364613457800281721023851L);
        foreach (xi; [T.epsilon / 16, -T.epsilon / 16,
            2 * T.min_normal, -2 * T.min_normal])
            assert(approxEqual(gevCCDF(T(1), T(0), T(1), xi),
                expected, 8 * T.epsilon, T(0)));

        // The product can be subnormal or round to zero.
        const T atLocation = T(0.63212055882855767840447622983853913255L);
        assert(approxEqual(gevCCDF(nextUp(T(0)), T(0), T(1), 2 * T.min_normal),
            atLocation, 8 * T.epsilon, T(0)));

        // Overflow of a finite product does not imply a zero tail probability.
        assert(approxEqual(gevCCDF(T(2), T(0), T(1), T.max),
            atLocation, 8 * T.epsilon, T(0)));
        assert(approxEqual(gevCCDF(T(-2), T(0), T(1), -T.max),
            atLocation, 8 * T.epsilon, T(0)));
    }}
}

// Both sides of the logarithm cutoff agree with independent reference values.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, nextUp;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T[2][2] cases = [
            [T(0.125), T(0.32277000913152354465704295941556212577L)],
            [T(-0.125), T(0.29079376683063948552877145237498560910L)]];
        foreach (entry; cases)
            foreach (s; [nextDown(T(1)), T(1), nextUp(T(1))])
                // Moving s by one ULP changes the probability by O(epsilon).
                assert(approxEqual(gevCCDF(s, T(0), T(1), entry[0]),
                    entry[1], 16 * T.epsilon, T(0)));
    }}
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
