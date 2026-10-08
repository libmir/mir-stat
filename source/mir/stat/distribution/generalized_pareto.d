/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.generalized_pareto;

import mir.internal.utility: isFloatingPoint;
import mir.stat.internal.shape_transform: log1pScaled, expm1Scaled;

/++
Computes the generalized pareto probability density function (PDF).

Params:
    x = value to evaluate PDF
    mu = location parameter
    sigma = scale parameter
    xi = shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution)
+/
@safe pure nothrow @nogc
T generalizedParetoPDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (sigma > 0, "sigma must be greater than zero")
    in (x >= mu, "x must be greater than or equal to mu")
    in (xi >= 0 || (xi < 0 && x <= (mu - sigma / xi)), "if xi is less than zero, x must be less than mu - sigma / xi")
{
    import mir.math.common: exp;

    const T z = (x - mu) / sigma;
    if (xi != 0) {
        const T u = xi * z;
        const T v = 1 + u;
        if (v <= 0)
            return xi == -1 ? 1 / sigma : xi < -1 ? T.infinity : T(0);
        static if (T.mant_dig > 53)
        {
            import mir.math.common: powi;
            // Small integer powers avoid extended-precision transcendental
            // functions. Limit the exponent to eight so rounding 1 + u near
            // one contributes at most about four epsilons of relative error.
            // Larger exponents retain the stable logarithmic calculation.
            const T exponent = -(1 / xi + 1);
            if (exponent >= 0 && exponent <= 8
                && exponent == cast(int) exponent)
                return powi(v, cast(int) exponent) / sigma;
        }
        return exp(-(1 + xi) * log1pScaled(z, xi)) / sigma;
    } else {
        return exp(-z) / sigma;
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.generalizedParetoPDF(1, 1, 0.5).shouldApprox == 1;
    2.0.generalizedParetoPDF(1, 1, 0.5).shouldApprox == 0.2962963;
    3.0.generalizedParetoPDF(2, 3, 0.25).shouldApprox == 0.2233923;
    5.0.generalizedParetoPDF(2, 3, 0).shouldApprox == 0.1226264803904808;
}

/++
Computes the generalized pareto cumulative distribution function (CDF).

Evaluates the probability using $(D log1p) and $(D expm1) near cancellation,
preserving small probabilities and the limit as the shape approaches zero.

Params:
    x = value to evaluate CDF
    mu = location parameter
    sigma = scale parameter
    xi = shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution)
+/
@safe pure nothrow @nogc
T generalizedParetoCDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (sigma > 0, "sigma must be greater than zero")
    in (x >= mu, "x must be greater than or equal to mu")
    in (xi >= 0 || (xi < 0 && x <= (mu - sigma / xi)), "if xi is less than zero, x must be less than mu - sigma / xi")
{
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;

    const T z = (x - mu) / sigma;
    if (1 + xi * z <= 0)
        return 1;
    return oneMinusExpNeg(log1pScaled(z, xi));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.generalizedParetoCDF(1, 1, 0.5).shouldApprox == 0;
    2.0.generalizedParetoCDF(1, 1, 0.5).shouldApprox == 0.5555556;
    3.0.generalizedParetoCDF(2, 3, 0.25).shouldApprox == 0.273975;
    5.0.generalizedParetoCDF(2, 3, 0).shouldApprox == 0.6321206;
}

/++
Computes the generalized pareto complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    mu = location parameter
    sigma = scale parameter
    xi = shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution)
+/
@safe pure nothrow @nogc
T generalizedParetoCCDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (sigma > 0, "sigma must be greater than zero")
    in (x >= mu, "x must be greater than or equal to mu")
    in (xi >= 0 || (xi < 0 && x <= (mu - sigma / xi)), "if xi is less than zero, x must be less than mu - sigma / xi")
{
    import mir.math.common: exp;

    const T z = (x - mu) / sigma;
    const T h = 1 + xi * z <= 0 ? T.infinity : log1pScaled(z, xi);
    return exp(-h);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.generalizedParetoCCDF(1, 1, 0.5).shouldApprox == 1;
    2.0.generalizedParetoCCDF(1, 1, 0.5).shouldApprox == 0.4444444;
    3.0.generalizedParetoCCDF(2, 3, 0.25).shouldApprox == 0.726025;
    5.0.generalizedParetoCCDF(2, 3, 0).shouldApprox == 0.3678794;
}

/++
Computes the generalized pareto inverse cumulative distribution function (InvCDF).

Uses $(D expm1) to preserve small nonzero shapes, and a stable logarithm of
$(D 1 - p) to retain tiny probabilities.

Params:
    p = value to evaluate InvCDF
    mu = location parameter
    sigma = scale parameter
    xi = shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution)
+/
@safe pure nothrow @nogc
T generalizedParetoInvCDF(T)(const T p, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (sigma > 0, "sigma must be greater than zero")
{
    import mir.stat.internal.neg_log1m: negLog1m;

    return mu + sigma * expm1Scaled(negLog1m(p), xi);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.generalizedParetoInvCDF(1, 1, 0.5).shouldApprox == 1;
    0.5555556.generalizedParetoInvCDF(1, 1, 0.5).shouldApprox == 2;
    0.273975.generalizedParetoInvCDF(2, 3, 0.25).shouldApprox == 3;
    0.6321206.generalizedParetoInvCDF(2, 3, 0).shouldApprox == 5;    
}

/++
Computes the generalized pareto log probability density function (LPDF).

Params:
    x = value to evaluate LPDF
    mu = location parameter
    sigma = scale parameter
    xi = shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution)
+/
@safe pure nothrow @nogc
T generalizedParetoLPDF(T)(const T x, const T mu, const T sigma, const T xi)
    if (isFloatingPoint!T)
    in (sigma > 0, "sigma must be greater than zero")
    in (x >= mu, "x must be greater than or equal to mu")
    in (xi >= 0 || (xi < 0 && x <= (mu - sigma / xi)), "if xi is less than zero, x must be less than mu - sigma / xi")
{
    import mir.math.common: log;


    const T z = (x - mu) / sigma;
    if (xi != 0) {
        if (xi == -1)
            return -log(sigma);
        if (1 + xi * z <= 0)
            return xi < -1 ? T.infinity : -T.infinity;
        return -(1 + xi) * log1pScaled(z, xi) - log(sigma);
    } else {
        return -z - log(sigma);
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    1.0.generalizedParetoLPDF(1, 1, 0.5).shouldApprox == log(generalizedParetoPDF(1.0, 1, 1, 0.5));
    2.0.generalizedParetoLPDF(1, 1, 0.5).shouldApprox == log(generalizedParetoPDF(2.0, 1, 1, 0.5));
    3.0.generalizedParetoLPDF(2, 3, 0.25).shouldApprox == log(generalizedParetoPDF(3.0, 2, 3, 0.25));
    5.0.generalizedParetoLPDF(2, 3, 0).shouldApprox == log(generalizedParetoPDF(5.0, 2, 3, 0));
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
            const T standard = generalizedParetoPDF(z, T(0), T(1), T(0));
            assert(approxEqual(generalizedParetoPDF(x, mu, sigma, T(0)), standard / sigma,
                32 * T.epsilon, T(0)));
            assert(approxEqual(generalizedParetoLPDF(x, mu, sigma, T(0)),
                generalizedParetoLPDF(z, T(0), T(1), T(0)) - log(sigma),
                32 * T.epsilon, 32 * T.epsilon));
            const T expectedLog = -z - log(sigma);
            assert(approxEqual(generalizedParetoLPDF(x, mu, sigma, T(0)), expectedLog,
                32 * T.epsilon, 32 * T.epsilon));
            assert(approxEqual(generalizedParetoPDF(x, mu, sigma, T(0)), exp(expectedLog),
                32 * T.epsilon, T(0)));
        }
    }}
}

// Small nonzero shapes retain the exponential limit, including inverse CDFs.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    import std.math: nextDown, nextUp;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (xi; [T.epsilon / 64, -T.epsilon / 64])
        {
            const T tail = T(0.36787944117144232159552377016146086745L);
            assert(approxEqual(generalizedParetoCDF(T(1), T(0), T(1), xi),
                1 - tail, 16 * T.epsilon, T(0)));
            assert(approxEqual(generalizedParetoCCDF(T(1), T(0), T(1), xi),
                tail, 16 * T.epsilon, T(0)));
            assert(approxEqual(generalizedParetoPDF(T(1), T(0), T(1), xi),
                tail, 16 * T.epsilon, T(0)));
            assert(approxEqual(generalizedParetoLPDF(T(1), T(0), T(1), xi),
                T(-1), 16 * T.epsilon, T(0)));
            assert(approxEqual(generalizedParetoInvCDF(T(.5), T(0), T(1), xi),
                T(0.69314718055994530941723212145817656808L), 16 * T.epsilon, T(0)));
        }
        foreach (xi; [T(-.5), T(0), T(.5)])
        {
            foreach (tiny; [nextUp(T(0)), T.min_normal, T.epsilon / 16])
            {
                assert(generalizedParetoCDF(tiny, T(0), T(1), xi) == tiny);
                assert(generalizedParetoInvCDF(tiny, T(0), T(1), xi) == tiny);
            }
            foreach (p; [T(.125), T(.5), T(.875)])
                assert(approxEqual(generalizedParetoCDF(
                    generalizedParetoInvCDF(p, T(2), T(3), xi), T(2), T(3), xi),
                    p, 32 * T.epsilon, T(0)));
            assert(generalizedParetoInvCDF(T(0), T(2), T(3), xi) == 2);
            assert(generalizedParetoInvCDF(T(1), T(2), T(3), xi) == (xi < 0 ? T(8) : T.infinity));
        }
        foreach (xi; [T(-2), T(-1), T(-.5)])
        {
            const T end = -1 / xi;
            assert(generalizedParetoCDF(end, T(0), T(1), xi) == 1);
            assert(generalizedParetoCCDF(end, T(0), T(1), xi) == 0);
            assert(generalizedParetoPDF(end, T(0), T(1), xi) ==
                (xi == -1 ? T(1) : xi < -1 ? T.infinity : T(0)));
            assert(generalizedParetoLPDF(end, T(0), T(1), xi) ==
                (xi == -1 ? T(0) : xi < -1 ? T.infinity : -T.infinity));
        }
        // Small integer powers and the boundary of the logarithm cutoff.
        assert(approxEqual(generalizedParetoPDF(T(1), T(0), T(1), T(-.25)),
            T(0.421875), 8 * T.epsilon, T(0)));
        assert(approxEqual(generalizedParetoPDF(T(1), T(0), T(1), T(-.125)),
            T(0.392695903778076171875L), 8 * T.epsilon, T(0)));
        // Exponent nine remains on the logarithmic path.
        assert(approxEqual(generalizedParetoPDF(T(1), T(0), T(1), T(-.1)),
            T(0.387420489L), 8 * T.epsilon, T(0)));
        foreach (xi; [T(-.25), T(-.125)])
            assert(approxEqual(generalizedParetoPDF(T.epsilon, T(0), T(1), xi),
                1 - (1 + xi) * T.epsilon, 4 * T.epsilon, T(0)));
        foreach (x; [nextDown(T(.5)), T(.5), nextUp(T(.5))])
            assert(approxEqual(generalizedParetoPDF(x, T(0), T(1), T(-.25)),
                T(0.669921875), 16 * T.epsilon, T(0)));
        assert(approxEqual(generalizedParetoInvCDF(T(.5), T(0), T(1), T(.5)),
            T(0.82842712474619009760337744841939615714L), 16 * T.epsilon, T(0)));
        assert(approxEqual(generalizedParetoInvCDF(T(.5), T(0), T(1), T(-.5)),
            T(0.58578643762690495119831127579030192143L), 16 * T.epsilon, T(0)));
    }}
}
