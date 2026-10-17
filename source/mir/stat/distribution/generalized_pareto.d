/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Generalized_Pareto_distribution, Generalized Pareto Distribution).

The parameters are finite location $(D mu), positive finite scale $(D sigma),
and finite shape $(D xi). For $(D xi >= 0), the support is $(D x >= mu).
For $(D xi < 0), it is $(D mu <= x && x <= b), where $(D b = mu - sigma / xi).
The forward functions check these support restrictions with contracts.
Shape zero gives an exponential distribution shifted by $(D mu), with
rate $(D 1 / sigma).

The special-value tables assume parameters in the stated domain. Finite
expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts
does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.generalized_pareto;

import mir.internal.utility: isFloatingPoint;
import mir.stat.internal.shape_transform: log1pScaled, expm1Scaled;
import mir.stat.internal.standardize: standardize;

/++
Computes the generalized pareto probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.
Here $(D b = mu - sigma / xi) denotes a finite support boundary.
Boundary formulas describe the limiting values; rounding the calculation of
$(D b) can place an observation slightly inside or outside the support.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == mu)) $(TD $(D 1 / sigma)))
    $(TR $(TD $(D x == b), $(D -1 < xi && xi < 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == b), $(D xi == -1)) $(TD $(D 1 / sigma)))
    $(TR $(TD $(D x == b), $(D xi < -1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == T.infinity), $(D xi >= 0)) $(TD $(D 0)))
    $(TR $(TD Outside the support) $(TD Contract violation))
    $(TR $(TD NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    mu = finite location parameter
    sigma = positive finite scale parameter
    xi = finite shape parameter

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

    const T z = standardize(x, mu, sigma);
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
        const T decay = exp(-z);
        if (decay < T.min_normal && sigma < 1)
        {
            // Scale before forming the small density. Quarter-exponentials
            // retain precision even when sigma is the smallest subnormal.
            const T quarterDecay = exp(-z / 4);
            return ((quarterDecay / sigma) * quarterDecay) * quarterDecay * quarterDecay;
        }
        return decay / sigma;
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

/// Negative shape gives a finite upper endpoint with a shape-dependent density.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    // With location zero and scale two, the upper endpoint is -2 / xi.
    assert(generalizedParetoPDF(4.0, 0.0, 2.0, -0.5) == 0.0);
    assert(generalizedParetoPDF(2.0, 0.0, 2.0, -1.0) == 0.5);
    assert(generalizedParetoPDF(1.0, 0.0, 2.0, -2.0) == double.infinity);
}

// The zero-shape exponential limit must retain scaled tail densities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp;
    import mir.math.common: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Independent 90-digit references at exact binary inputs.
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
        const T actual = generalizedParetoPDF(exponent * scale, T(0), scale, T(0));
        assert(actual > 0);
        assert(fabs(actual / expected - 1) < tolerance);
        const T smallActual = generalizedParetoPDF(smallExponent / 2, T(0), T(0.5), T(0));
        assert(fabs(smallActual - smallExpected) <=
            tolerance * smallExpected + 2 * T.min_normal * T.epsilon);
        const T tinyScale = T.min_normal * T.epsilon;
        const T tinyActual = generalizedParetoPDF(tinyExponent * tinyScale, T(0), tinyScale, T(0));
        assert(tinyActual > 0);
        assert(fabs(tinyActual / tinyExpected - 1) < 2 * tolerance);
        // A nonzero location with exact subtraction retains the same density.
        assert(generalizedParetoPDF(T(0), -exponent * scale, scale, T(0)) == actual);
        assert(generalizedParetoPDF(T(0), T(0), scale, T(0)) == 1 / scale);
        assert(generalizedParetoPDF(T.infinity, T(0), scale, T(0)) == 0);
    }}

    // Probe the underflowing primitive, leaving PDF regressions visible.
    static if (__traits(compiles, {
        import mir.math.common: exp;
        enum double probe = exp(-800.0);
    }))
    {
        enum double density = generalizedParetoPDF(800 * 1e-300, 0.0, 1e-300, 0.0);
        static assert(fabs(density / 3.667874584177395251e-48 - 1) < 2e-13);
    }
}

/++
Computes the generalized pareto cumulative distribution function (CDF).

Evaluates the probability using $(D log1p) and $(D expm1) near cancellation,
preserving small probabilities and the limit as the shape approaches zero.

Returns the lower-tail probability $(D P(X <= x)).
Here $(D b = mu - sigma / xi) denotes a finite support boundary.
Boundary formulas describe the limiting values; rounding the calculation of
$(D b) can place an observation slightly inside or outside the support.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == mu)) $(TD $(D 0)))
    $(TR $(TD $(D x == b), $(D xi < 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity), $(D xi >= 0)) $(TD $(D 1)))
    $(TR $(TD Outside the support) $(TD Contract violation))
    $(TR $(TD NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    mu = finite location parameter
    sigma = positive finite scale parameter
    xi = finite shape parameter

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

    const T z = standardize(x, mu, sigma);
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

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.
Here $(D b = mu - sigma / xi) denotes a finite support boundary.
Boundary formulas describe the limiting values; rounding the calculation of
$(D b) can place an observation slightly inside or outside the support.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == mu)) $(TD $(D 1)))
    $(TR $(TD $(D x == b), $(D xi < 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity), $(D xi >= 0)) $(TD $(D 0)))
    $(TR $(TD Outside the support) $(TD Contract violation))
    $(TR $(TD NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    mu = finite location parameter
    sigma = positive finite scale parameter
    xi = finite shape parameter

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

    const T z = standardize(x, mu, sigma);
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

Returns the quantile at cumulative probability $(D p).
Here $(D b = mu - sigma / xi) denotes a finite support boundary.
Boundary formulas describe the limiting values; rounding the calculation of
$(D b) can place an observation slightly inside or outside the support.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D mu)))
    $(TR $(TD $(D p == 1), $(D xi < 0)) $(TD $(D b)))
    $(TR $(TD $(D p == 1), $(D xi >= 0)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    mu = finite location parameter
    sigma = positive finite scale parameter
    xi = finite shape parameter

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

Returns the natural logarithm of the density.
Here $(D b = mu - sigma / xi) denotes a finite support boundary.
Boundary formulas describe the limiting values; rounding the calculation of
$(D b) can place an observation slightly inside or outside the support.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == mu)) $(TD $(D -log(sigma))))
    $(TR $(TD $(D x == b), $(D -1 < xi && xi < 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == b), $(D xi == -1)) $(TD $(D -log(sigma))))
    $(TR $(TD $(D x == b), $(D xi < -1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == T.infinity), $(D xi >= 0)) $(TD $(D -T.infinity)))
    $(TR $(TD Outside the support) $(TD Contract violation))
    $(TR $(TD NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    mu = finite location parameter
    sigma = positive finite scale parameter
    xi = finite shape parameter

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


    const T z = standardize(x, mu, sigma);
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

// A finite standardized observation can survive an overflowing subtraction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, fabs, log;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // (max - -max) / max is exactly two.
        const T expected = (T(1) / 9) / T.max;
        const T actual = generalizedParetoPDF(T.max, -T.max, T.max, T(1));
        assert(actual > 0);
        assert(fabs(actual - expected) <= 4 * T.min_normal * T.epsilon);
        assert(approxEqual(generalizedParetoLPDF(T.max, -T.max, T.max, T(1)),
            -2 * log(T(3)) - log(T.max), 16 * T.epsilon, T(0)));
        assert(approxEqual(generalizedParetoCDF(T.max, -T.max, T.max, T(1)),
            T(2) / 3, 16 * T.epsilon, T(0)));
        assert(approxEqual(generalizedParetoCCDF(T.max, -T.max, T.max, T(1)),
            T(1) / 3, 16 * T.epsilon, T(0)));
        foreach (xi; [T(0), T(-.25)])
        {
            assert(approxEqual(generalizedParetoLPDF(T.max, -T.max, T.max, xi),
                generalizedParetoLPDF(T(2), T(0), T(1), xi) - log(T.max),
                16 * T.epsilon, T(0)));
            assert(generalizedParetoCDF(T.max, -T.max, T.max, xi) ==
                generalizedParetoCDF(T(2), T(0), T(1), xi));
            assert(generalizedParetoCCDF(T.max, -T.max, T.max, xi) ==
                generalizedParetoCCDF(T(2), T(0), T(1), xi));
        }
    }}
}
