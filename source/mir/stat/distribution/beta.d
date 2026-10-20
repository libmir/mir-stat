/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution).

An alternate parameterization of this distribution is provided in $(MREF mir,stat,distribution,beta_proportion).

Both shape parameters must be positive. Observations must lie in the closed
interval $(D [0, 1]); the forward functions do not extend outside it. The
shape $(D alpha) controls the density limit at zero, while $(D beta) controls
the limit at one. The inverse CDF accepts both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.beta;

import mir.internal.utility: isFloatingPoint;

/++
Computes the beta probability density function (PDF).

The direct calculation is retained when its intermediate values remain
finite and normal. Otherwise, a logarithmic fallback avoids losing a
representable density through intermediate overflow or underflow. Its
accuracy follows $(LREF betaLPDF), including the limits for smaller shapes.

Returns a density, which may exceed one; it is not a point probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D alpha < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D alpha == 1)) $(TD $(D beta)))
    $(TR $(TD $(D x == 0), $(D alpha > 1)) $(TD $(D 0)))
    $(TR $(TD $(D x == 1), $(D beta < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 1), $(D beta == 1)) $(TD $(D alpha)))
    $(TR $(TD $(D x == 1), $(D beta > 1)) $(TD $(D 0)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    alpha = positive shape controlling the lower endpoint
    beta = positive shape controlling the upper endpoint

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution)
+/
@safe pure nothrow @nogc
T betaPDF(T)(const T x, const T alpha, const T beta)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (alpha > 0, "alpha must be greater than zero")
    in (beta > 0, "beta must be greater than zero")
{
    import mir.math.common: pow, exp;
    import std.mathspecial: betaFunc = beta;

    // Evaluate endpoint limits directly, including shape=1 where the
    // density equals the opposite shape even if the beta function underflows.
    if (x == 0)
        return alpha < 1 ? T.infinity : alpha > 1 ? T(0) : beta;
    if (x == 1)
        return beta < 1 ? T.infinity : beta > 1 ? T(0) : alpha;
    const T left = pow(x, alpha - 1);
    const T right = pow(1 - x, beta - 1);
    const T numerator = left * right;
    const denominator = betaFunc(alpha, beta);
    // A normal quotient alone is not enough: a subnormal power or product
    // may already have lost precision before division restores its scale.
    if (left >= T.min_normal && right >= T.min_normal
        && numerator >= T.min_normal && numerator < T.infinity
        && denominator >= T.min_normal && denominator < T.infinity)
        return numerator / denominator;

    // Keep the logarithm and exponential in real until the final conversion;
    // rounding a large-magnitude log density to T can lose relative accuracy.
    return cast(T) exp(betaLPDF(cast(real) x, cast(real) alpha, cast(real) beta));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaPDF(1, 1) == 1);
    assert(0.75.betaPDF(1, 2).approxEqual(0.5));
    assert(0.25.betaPDF(0.5, 4).approxEqual(0.9228516));
}

// Recover densities lost by the direct powers and beta normalization.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: sqrt, log, nextUp;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Independent 380-digit beta-density references.
        assert(approxEqual(betaPDF(T(0.5), T(1000), T(1000)),
            T(35.67802229170864146047613447641492876937L), 128 * T.epsilon));
        assert(approxEqual(betaPDF(T(0.5), T(1e20L), T(1e20L)),
            T(11283791670.95512573894748429162675780971L), 128 * T.epsilon));
        // For double, the direct numerator and denominator are subnormal.
        assert(approxEqual(betaPDF(T(0.5), T(530), T(530)),
            T(25.97111325938743245428429197134466467901L), 128 * T.epsilon));
        // Both direct intermediates underflow even with 80-bit real, but
        // their quotient is representable (zero remains correct for float).
        enum real tail = 6.212155979696773806263160451679998014417e-279L;
        const T tailTolerance = cast(T) (8 * real.epsilon * (1 - log(tail))
            + 4 * T.epsilon);
        assert(approxEqual(betaPDF(T(0.375), T(10000), T(10000)),
            T(tail), tailTolerance, T(0)));
        // Symmetric shapes have peak density asymptotic to 2*sqrt(a/pi).
        const T huge = T.max / 2 + T.max / 4;
        const T peak = cast(T) (2 * sqrt(cast(real) huge)
            / sqrt(3.141592653589793238462643383279502884197L));
        // Exponentiation magnifies log-density rounding. Allow a few ULPs
        // of the real logarithm, plus the final conversion to T.
        const T peakTolerance = cast(T) (8 * real.epsilon
            * (1 + log(cast(real) peak)) + 4 * T.epsilon);
        assert(approxEqual(betaPDF(T(0.5), huge, huge), peak, peakTolerance));
        assert(betaPDF(T(0.25), huge, huge) == 0); // genuine result underflow
        assert(betaPDF(T(0), T(1), T.max) == T.max);
        assert(betaPDF(T(1), T.max, T(1)) == T.max);

        foreach (a; [T(0.5), T(1), T(2), T(1000)])
        foreach (b; [T(0.5), T(1), T(2), T(1000)])
        {
            assert(betaPDF(T(0), a, b) == (a < 1 ? T.infinity : a > 1 ? T(0) : b));
            assert(betaPDF(T(1), a, b) == (b < 1 ? T.infinity : b > 1 ? T(0) : a));
        }
        // With beta=1 and alpha=2 the exact density is 2*x, even when x
        // itself is subnormal. Use an absolute allowance for final rounding.
        const T tiny = nextUp(T(0));
        assert(approxEqual(betaPDF(tiny, T(2), T(1)), 2 * tiny,
            128 * T.epsilon, tiny));
    }}
}

/// Each shape controls one endpoint; a shape of one gives a finite endpoint density.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(betaPDF(0.0, 0.5, 2.0) == double.infinity);
    assert(betaPDF(0.0, 1.0, 2.0) == 2.0);
    assert(betaPDF(0.0, 2.0, 2.0) == 0.0);
    assert(betaPDF(1.0, 2.0, 1.0) == 2.0);
}

/++
Computes the beta cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == 1)) $(TD $(D 1)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    alpha = positive shape controlling the lower endpoint
    beta = positive shape controlling the upper endpoint

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution)
+/
@safe pure nothrow @nogc
T betaCDF(T)(const T x, const T alpha, const T beta)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (alpha > 0, "alpha must be greater than zero")
    in (beta > 0, "beta must be greater than zero")
{
    import std.mathspecial: betaIncomplete;

    return betaIncomplete(alpha, beta, x);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaCDF(1, 1).approxEqual(0.5));
    assert(0.75.betaCDF(1, 2).approxEqual(0.9375));
    assert(0.25.betaCDF(0.5, 4).approxEqual(0.8588867));
}

/++
Computes the beta complementary cumulative distribution function (CCDF).

For small $(D x), forming $(D 1 - x) can discard significant digits. Simply
subtracting the CDF from one is also inaccurate when the remaining upper tail
is tiny. For $(D x < 1.0 / 16), $(D alpha <= 1), and $(D beta <= 128), a
fallback can instead add the probability between $(D x) and an interior
point to the upper tail starting at that point. Other parameter regimes use
Phobos's complementary incomplete-beta function, whose accuracy depends on
the installed Phobos version. Versions without that function use the reflected
calculation with the subtraction performed in $(D real).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == 1)) $(TD $(D 0)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    alpha = positive shape controlling the lower endpoint
    beta = positive shape controlling the upper endpoint

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution)
+/
@safe pure nothrow @nogc
T betaCCDF(T)(const T x, const T alpha, const T beta)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (alpha > 0, "alpha must be greater than zero")
    in (beta > 0, "beta must be greater than zero")
{
    static import std.mathspecial;

    if (x < T(0.0625) && alpha <= 1 && beta <= 128)
        return cast(T) betaSmallInputCCDF(x, alpha, beta);
    static if (__traits(hasMember, std.mathspecial, "betaIncompleteCompl"))
        return std.mathspecial.betaIncompleteCompl(alpha, beta, x);
    else
        // Older Phobos has no complement API. Promote before subtracting so
        // float inputs retain their displacement from one.
        return std.mathspecial.betaIncomplete(beta, alpha, 1 - cast(real) x);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaCCDF(1, 1).approxEqual(0.5));
    assert(0.75.betaCCDF(1, 2).approxEqual(0.0625));
    assert(0.25.betaCCDF(0.5, 4).approxEqual(0.1411133));
}

// Compute the area to the right of x as two pieces:
//
//     x -------- c ---------------- 1
//       interval      upper tail
//
// Q(x) = integral(x..c, density) + Q(c). Neither piece is obtained by
// subtracting a probability close to one. This helper is deliberately bounded
// to alpha <= 1 and beta <= 128; it is not a replacement incomplete-beta API.
// A compile-time limit lets tests exercise exhaustion without invalid inputs.
private @safe pure nothrow @nogc
real betaSmallInputCCDF(int seriesLimit = 10_000)(real x, real a, real b)
{
    import std.math: exp, expm1, log, fabs;
    import std.mathspecial: betaIncomplete, gamma;

    if (x == 0)
        return 1;

    // Choose a power-of-two anchor so 1-c is exact. Requiring b*c <= 1
    // also keeps the interval's binomial-series coefficients manageable.
    // In this parameter range c is at least 1/128.
    real c = 0.5L;
    while (b * c > 1)
        c *= 0.5L;
    if (x >= c)
        return betaIncomplete(b, a, 1 - x);

    // With subnormal b the distribution's endpoint mass dominates. This
    // ratio also handles both shapes being subnormal without overflowing B.
    // For these small x, the omitted correction is below real precision.
    if (b < real.min_normal)
        return a / (a + b);

    // 1/B(a,b) = a * b/(a+b) * Gamma(1+a+b)/Gamma(1+a)/Gamma(1+b).
    // Shifted gamma arguments avoid Gamma(a) overflow for tiny a. The shape
    // bounds keep these gamma values representable even in double precision.
    const real gammaRatio = gamma(1 + a + b) / gamma(1 + b) / gamma(1 + a);
    const real logRatio = log(x) - log(c);

    // Expand (1-t)^(b-1) and integrate each term from x to c. Factor a out
    // of the entire result: intermediate series terms then stay representable
    // even when the final probability is subnormal.
    // The first term is (1-(x/c)^a)/a. expm1 preserves it when a is tiny;
    // its limit -log(x/c) avoids division of an already rounded subnormal.
    real sum = fabs(a * logRatio) < real.epsilon
        ? -logRatio : -expm1(a * logRatio) / a;
    real ratioPower = exp(a * logRatio);
    const real ratio = x / c;
    real coefficient = 1;
    foreach (n; 1 .. seriesLimit)
    {
        coefficient *= (n - b) * c / n;
        ratioPower *= ratio;
        const real term = coefficient / (a + n) * (1 - ratioPower);
        sum += term;
        if (fabs(term) > real.epsilon * fabs(sum))
            continue;

        real anchorPerA;
        if (a <= real.epsilon && b <= real.epsilon)
        {
            // Here c=1/2. The limiting tail is a/(a+b); the first-order
            // correction at one half vanishes. Avoid a fragile Phobos call.
            anchorPerA = 1 / (a + b);
        }
        else if (a < real.min_normal)
        {
            // Evaluate the anchor with a normal surrogate, then restore the
            // endpoint-mass ratio. Changing this tiny shape has negligible
            // higher-order effect, but avoids overflow inside Phobos.
            const real normalA = 16 * real.min_normal;
            anchorPerA = betaIncomplete(b, normalA, 1 - c) / normalA
                * ((normalA + b) / (a + b));
        }
        else
            anchorPerA = betaIncomplete(b, a, 1 - c) / a;

        const real result = a * (anchorPerA
            + exp(a * log(c)) * gammaRatio * (b / (a + b)) * sum);
        // The two pieces may round slightly outside the probability bounds.
        return result > 1 ? 1 : result < 0 ? 0 : result;
    }
    // Do not silently return an unconverged probability.
    return real.nan;
}

// Exhaustion must report failure rather than return a partial probability.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: isNaN, fabs;

    // For alpha=1, beta=2 the series needs its second (zero) term to
    // establish convergence. Stop after the first, nonzero term instead.
    assert(isNaN(betaSmallInputCCDF!2(0.03125L, 1.0L, 2.0L)));
    const real expected = (1 - 0.03125L) * (1 - 0.03125L);
    assert(fabs(betaSmallInputCCDF(0.03125L, 1.0L, 2.0L) - expected)
        < 32 * real.epsilon);
}

// Recover small-input upper tails without rounding away tiny probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: exp, expm1, log, log1p, nextUp, nextDown;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // For beta=1, Q(x) = 1-x^alpha, evaluated independently with expm1.
        foreach (a; [T(0.01), T.epsilon * T.epsilon])
        foreach (x; [T(1e-20L), nextDown(T(0.0625)), T(0.0625),
            nextUp(T(0.0625)), T(0.25), nextDown(T(1))])
        {
            const T expected = -expm1(a * log(x));
            const T value = betaCCDF(x, a, T(1));
            assert(value >= 0 && value <= 1);
            assert(approxEqual(value, expected, 512 * T.epsilon, T(0)));
        }
        // The result itself may be subnormal; allow its final rounding step.
        const T tiny = nextUp(T(0));
        const T expected = -expm1(tiny * log(T(1e-20L)));
        assert(approxEqual(betaCCDF(T(1e-20L), tiny, T(1)), expected,
            512 * T.epsilon, 2 * tiny));
        assert(approxEqual(betaCCDF(T(1e-20L), tiny, tiny), T(0.5),
            512 * T.epsilon, T(0)));
        assert(betaCCDF(T(0), T(0.01), T(3)) == 1);
        assert(betaCCDF(T(1), T(0.01), T(3)) == 0);

        // Independent 380-digit incomplete-beta references; decimal input
        // rounding in T is included in the error allowance.
        assert(approxEqual(betaCCDF(T(1e-20L), T(0.03), T(7.25)),
            T(0.7295273104045321893412536502921225024L),
            1024 * T.epsilon, T(0)));
        assert(approxEqual(betaCCDF(T(0.001), T(0.1), T(128)),
            T(0.1540456317238734585412937362254863490L),
            1024 * T.epsilon, T(0)));
        // For alpha=1, Q(x)=(1-x)^beta. Check the anchor and shape-bound
        // transitions, including the unchanged path outside the fallback.
        foreach (b; [nextDown(T(128)), T(128), nextUp(T(128))])
        foreach (x; [nextDown(T(0.0078125)), T(0.0078125), nextUp(T(0.0078125))])
            // Evaluate the identity in real: older Phobos float/double
            // log1p implementations are inaccurate near this boundary.
            assert(approxEqual(betaCCDF(x, T(1), b),
                cast(T) exp(cast(real) b * log1p(-cast(real) x)),
                1024 * T.epsilon, T(0)));
    }}
}

/++
Computes the beta inverse cumulative distribution function (InvCDF).

For lower-half probabilities whose quantiles are near one, a bounded search
keeps the original probability intact instead of subtracting it from one.
Its accuracy remains limited by the underlying incomplete beta calculation.

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D 1)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    alpha = positive shape controlling the lower endpoint
    beta = positive shape controlling the upper endpoint

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution)
+/
@safe pure nothrow @nogc
T betaInvCDF(T)(const T p, const T alpha, const T beta)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (alpha > 0, "alpha must be greater than zero")
    in (beta > 0, "beta must be greater than zero")
{
    import std.mathspecial: betaIncompleteInverse, betaIncomplete;

    // Phobos can lose a small p by switching to 1-p when its lower search
    // bound exceeds .95: https://github.com/dlang/phobos/issues/11103.
    // For alpha <= beta the median is <= .5, so this cannot affect p <= .5.
    if (p > 0 && p <= 0.5 && alpha > beta
        && betaIncomplete(alpha, beta, 0.95L) < p)
        return cast(T) betaInverseLowerTail(p, alpha, beta);

    return betaIncompleteInverse(alpha, beta, p);
}

// The caller has established F(.95) < p <= .5. Maintain a bracket around
// the quantile and compare against p throughout; never form its complement.
// The default compile-time budget is unchanged when testing early exhaustion.
private real betaInverseLowerTail(int iterationLimit = 3 * real.mant_dig + 4)
    (real p, real a, real b)
    @safe pure nothrow @nogc
{
    import std.mathspecial: betaIncomplete, logGamma;
    import std.math: log, log1p, exp, fabs, isFinite, nextUp, nextDown;

    real lower = 0.95L, upper = 1;
    real x = lower + (upper - lower) / 2;
    real previous = upper - lower;
    const normalizer = logGamma(a + b) - logGamma(a) - logGamma(b);
    // After a bounded number of Newton attempts, use only bisection.
    // This also handles an unusable derivative without changing the target.
    foreach (iteration; 0 .. iterationLimit)
    {
        const y = betaIncomplete(a, b, x);
        if (y < p)
            lower = x;
        else
            upper = x;
        if (y == p)
            return x;

        real next = lower + (upper - lower) / 2;
        if (iteration < 2 * real.mant_dig)
        {
            const density = exp((a - 1) * log(x) + (b - 1) * log1p(-x)
                + normalizer);
            if (density > 0 && isFinite(density))
            {
                const step = (y - p) / density;
                // A tiny estimated step is insufficient if the derivative
                // is inaccurate. Check that a neighboring value brackets p.
                if (fabs(step) <= real.epsilon * x)
                {
                    const neighbor = y < p ? nextUp(x) : nextDown(x);
                    const neighborY = betaIncomplete(a, b, neighbor);
                    if (y < p ? neighborY >= p : neighborY <= p)
                        return x;
                }
                const proposed = x - step;
                // Reject overshoots and steps that do not shrink promptly.
                if (proposed > lower && proposed < upper
                    && fabs(step) < previous / 2)
                    next = proposed;
            }
        }
        previous = fabs(next - x);
        if (next == lower || next == upper)
            return next;
        x = next;
    }
    return lower + (upper - lower) / 2;
}

// An exhausted search returns the midpoint of its updated bracket.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: fabs, pow;

    // F(.975)=.975^1000 is below .5, so one iteration raises the lower
    // bound to the initial midpoint while the upper bound remains one.
    const real first = 0.95L + (1 - 0.95L) / 2;
    const real expected = first + (1 - first) / 2;
    assert(betaInverseLowerTail!1(0.5L, 1000.0L, 1.0L) == expected);
    // The normal budget still resolves the quantile, not this coarse estimate.
    assert(fabs(betaInverseLowerTail(0.5L, 1000.0L, 1.0L)
        - pow(0.5L, 1.0L / 1000)) < 32 * real.epsilon);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaInvCDF(1, 1).approxEqual(0.5));
    assert(0.9375.betaInvCDF(1, 2).approxEqual(0.75));
    assert(0.8588867.betaInvCDF(0.5, 4).approxEqual(0.25));
}

// Preserve small lower-tail probabilities near one (Phobos #11103).
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: pow, fabs;
    import std.meta: AliasSeq;

    // A representable quantile can already give exactly the requested CDF.
    // Exercise that early return at the initial midpoint of the search, using
    // real throughout so conversion of p does not change the target value.
    const real midpoint = 0.95L + (1 - 0.95L) / 2;
    const real midpointP = betaCDF(midpoint, 1000.0L, 1.0L);
    assert(midpointP > 0 && midpointP < 0.5L);
    assert(betaInvCDF(midpointP, 1000.0L, 1.0L) == midpoint);
    // Independently check the shape-one identity, rather than relying only
    // on agreement between the forward and inverse special functions.
    assert(fabs(pow(midpointP, 1.0L / 1000) - midpoint) < 32 * real.epsilon);

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Allow final T rounding and the error of the underlying real CDF.
        const tolerance = 4 * T.epsilon + 32 * real.epsilon;
        foreach (p; [T(1e-30L), T(1e-20L), T(1e-10L), T(0.1L), T(0.5L)])
        {
            const expected = pow(cast(real) p, 1.0L / 1000);
            assert(fabs(betaInvCDF(p, T(1000), T(1)) - expected) < tolerance);
        }
        // Independent 120-decimal-digit mpmath inverse references; these
        // exercise non-unit shapes as well as the exact power identity.
        assert(fabs(betaInvCDF(T(1e-20L), T(1000), T(0.5L))
            - 0.95734473886760864595377827077806545919L) < tolerance);
        assert(fabs(betaInvCDF(T(1e-20L), T(1000), T(2))
            - 0.95126906395568540596314010302548721432L) < tolerance);
        assert(fabs(betaInvCDF(T(1e-30L), T(10000), T(2))
            - 0.99268854321259998873544195239523927870L) < tolerance);

        // Cross the .95 search boundary using the exact F(x)=x^1000 case.
        T previous = 0;
        foreach (x; [0.9499L, 0.95L, 0.9501L])
        {
            const p = cast(T) pow(x, 1000.0L);
            const actual = betaInvCDF(p, T(1000), T(1));
            assert(actual >= previous);
            assert(fabs(actual - pow(cast(real) p, 1.0L / 1000)) < tolerance);
            previous = actual;
        }
        assert(betaInvCDF(T(0), T(1000), T(2)) == 0);
        assert(betaInvCDF(T(1), T(1000), T(2)) == 1);
        assert(fabs(betaInvCDF(T(0.75L), T(2), T(1))
            - pow(0.75L, 0.5L)) < tolerance);
        // Do not force a quantile below one when its true value rounds to one.
        static if (real.mant_dig < 100)
            assert(betaInvCDF(T(0.5L), T(1), T(0.01L)) == 1);
    }}
}

/++
Computes the beta log probability density function (LPDF).

When both shapes are at least 16, a centered calculation avoids subtracting
large, nearly equal logarithms near the peak. Smaller shapes retain the
direct logarithmic formula and its accuracy limits.

Returns the natural logarithm of the density.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D alpha < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D alpha == 1)) $(TD $(D log(beta))))
    $(TR $(TD $(D x == 0), $(D alpha > 1)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == 1), $(D beta < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 1), $(D beta == 1)) $(TD $(D log(alpha))))
    $(TR $(TD $(D x == 1), $(D beta > 1)) $(TD $(D -T.infinity)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    alpha = positive shape controlling the lower endpoint
    beta = positive shape controlling the upper endpoint

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Distribution)
+/
@safe pure nothrow @nogc
T betaLPDF(T)(const T x, const T alpha, const T beta)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (alpha > 0, "alpha must be greater than zero")
    in (beta > 0, "beta must be greater than zero")
{
    import mir.math.internal.log_beta: logBeta;
    import mir.math.internal.xlogy: xlogy, xlog1py;

    if (alpha >= 16 && beta >= 16 && alpha < T.infinity && beta < T.infinity)
        return cast(T) betaLargeShapeLPDF(x, alpha, beta);
    return xlogy(alpha - 1, x) + xlog1py(beta - 1, -x) - logBeta(alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(0.5.betaLPDF(1, 1).approxEqual(log(betaPDF(0.5, 1, 1))));
    assert(0.75.betaLPDF(1, 2).approxEqual(log(betaPDF(0.75, 1, 2))));
    assert(0.25.betaLPDF(0.5, 4).approxEqual(log(betaPDF(0.25, 0.5, 4))));
}

// Stirling's log-gamma correction, with r=1/z and z>=16. The first omitted
// term is smaller than 2e-23 at z=16, below double/80-bit real precision.
private @safe pure nothrow @nogc
real betaStirlingCorrection(real r)
{
    const r2 = r * r;
    return r * (1.0L / 12 + r2 * (-1.0L / 360 + r2 * (1.0L / 1260
        + r2 * (-1.0L / 1680 + r2 * (1.0L / 1188 + r2 * (-691.0L / 360360
        + r2 * (1.0L / 156 + r2 * (-3617.0L / 122400
        + r2 * (43867.0L / 244188)))))))));
}

// Recover rounding discarded by a*b. Splitting each operand into two parts
// also works on DMD, whose std.math.fma need not provide a fused operation.
// Callers scale the operands into [0,1], so splitting cannot overflow.
private @safe pure nothrow @nogc
real betaProductError(real a, real b, real product)
{
    enum real splitter = (1UL << ((real.mant_dig + 1) / 2)) + 1.0L;
    const ca = splitter * a;
    const ah = ca - (ca - a);
    const al = a - ah;
    const cb = splitter * b;
    const bh = cb - (cb - b);
    const bl = b - bh;
    return ((ah * bh - product) + ah * bl + al * bh) + al * bl;
}

// log(1+z)-z for |z|<=1/2, without subtracting its nearly equal linear term.
// With t=z/(2+z), log(1+z)=2*(t+t^3/3+t^5/5+...), and 2*t-z=-z*t.
// Here |t|<=1/3, so the series converges rapidly even at the boundary.
private @safe pure nothrow @nogc
real betaLog1pmx(real z)
{
    const t = z / (2 + z);
    const t2 = t * t;
    real power = t * t2;
    real sum = 0;
    foreach (n; 0 .. 100)
    {
        const term = power / (3 + 2 * n);
        const next = sum + term;
        if (next == sum)
            break;
        sum = next;
        power *= t2;
    }
    return -z * t + 2 * sum;
}

// For large shapes, the usual formula subtracts huge log-gamma terms to
// recover a comparatively small log-density. Center the calculation at
// p=a/(a+b) instead: a*log(x/p)+b*log((1-x)/(1-p)), plus normalization.
// Near p, the two linear terms cancel mathematically. Remove them before
// evaluation rather than asking floating-point subtraction to cancel them.
private @safe pure nothrow @nogc
real betaLargeShapeLPDF(real x, real a, real b)
{
    import std.math: log, log1p, fabs, frexp, ldexp;

    if (x == 0 || x == 1)
        return -real.infinity;
    // Reflection keeps x<=1/2; its complement can then be compensated below.
    if (x > 0.5L)
    {
        x = 1 - x;
        const tmp = a;
        a = b;
        b = tmp;
    }
    const small = a < b ? a : b;
    const large = a < b ? b : a;
    const ratio = small / large;
    const logRatioSum = log1p(ratio);

    // d=a*(1-x)-b*x measures the displacement from p without rounding p
    // first. That matters when a very narrow peak lies between adjacent
    // floating-point values. Power-of-two scaling avoids overflow, and the
    // product residuals preserve the displacement when the products agree.
    int exponent;
    frexp(large, exponent);
    const scaledA = ldexp(a, -exponent);
    const scaledB = ldexp(b, -exponent);
    const q = 1 - x;
    const aq = scaledA * q;
    const bx = scaledB * x;
    const d = (aq - bx) + (betaProductError(scaledA, q, aq)
        - betaProductError(scaledB, x, bx) + scaledA * ((1 - q) - x));
    const za = -d / scaledA;
    const zb = d / scaledB;
    real centered;
    if (fabs(za) <= 0.5L && fabs(zb) <= 0.5L)
        centered = a * betaLog1pmx(za) + b * betaLog1pmx(zb);
    else
    {
        // Away from the peak, ordinary logarithms avoid rounding 1+z to
        // zero in an extreme tail. Cancellation is no longer severe here.
        const logSmallRatio = log(small) - log(large) - logRatioSum;
        const logP = a < b ? logSmallRatio : -logRatioSum;
        const logQ = a < b ? -logRatioSum : logSmallRatio;
        // Combine before restoring the scale: an individual negative term
        // can overflow even when the combined log-density is representable.
        centered = large * ((a / large) * (log(x) - logP)
            + (b / large) * (log1p(-x) - logQ));
    }
    // Stirling normalization, rearranged so a+b is never formed. In
    // particular, a*b/(a+b) = small/(1+small/large), without a*b overflow.
    return centered - log(x) - log1p(-x)
        + 0.5L * (log(small) - logRatioSum)
        - 0.9189385332046727417803297364056176398614L // log(sqrt(2*pi))
        - betaStirlingCorrection(1 / a) - betaStirlingCorrection(1 / b)
        + betaStirlingCorrection((1 / large) / (1 + ratio));
}

// Large-shape log densities must retain the small result after normalization.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log, log1p, nextUp, nextDown, isFinite;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Independent 380-digit mpmath references, including a skewed peak.
        assert(approxEqual(betaLPDF(T(0.5), T(1000), T(1000)),
            T(3.574534877131522080143340146965737860325L), 64 * T.epsilon));
        assert(approxEqual(betaLPDF(T(0.5), T(1e20L), T(1e20L)),
            T(23.14663316757570206252418299262528928826L), 64 * T.epsilon));
        const T shape = T(1e20L);
        // Powers of two preserve the exact 1:3 ratio in every tested type.
        const T skewShape = T(0x1p64L);
        assert(approxEqual(betaLPDF(T(0.25), skewShape, 3 * skewShape),
            T(22.79190664205935824212027792802960787739L), 64 * T.epsilon));
        assert(approxEqual(betaLPDF(T(0.01), T(1000), T(1e6L)),
            T(-6740.101350546231158367374280582065411379L), 64 * T.epsilon));

        // At the symmetric peak, log density approaches log(2*sqrt(a/pi)).
        // This remains finite even when a+a overflows the result type.
        const T huge = T.max / 2 + T.max / 4;
        const real peak = log(2.0L) + (log(cast(real) huge)
            - log(3.141592653589793238462643383279502884197L)) / 2;
        assert(approxEqual(betaLPDF(T(0.5), huge, huge), cast(T) peak,
            64 * T.epsilon));
        assert(betaLPDF(T(0), huge, huge) == -T.infinity);
        assert(betaLPDF(T(1), huge, huge) == -T.infinity);
        // A far-tail term may overflow before combining with the other term.
        const T far = betaLPDF(T(0.75), huge, huge);
        assert(isFinite(far));
        const real tailPerShape = log(0.75L);
        assert(approxEqual(far / huge, cast(T) tailPerShape, 64 * T.epsilon));

        // Check either side of the centered-series and large-shape switches.
        foreach (a; [nextDown(T(16)), T(16), nextUp(T(16))])
        foreach (x; [nextDown(T(0.25)), T(0.25), nextUp(T(0.25))])
        {
            const real center = betaLPDF(0.5L, cast(real) a, cast(real) a);
            const real delta = cast(real) x - 0.5L;
            const T expected = cast(T) (center
                + (cast(real) a - 1) * log1p(-4 * delta * delta));
            assert(approxEqual(betaLPDF(x, a, a), expected, 1024 * T.epsilon));
        }
        // Adjacent values near a narrow peak exercise compensated products.
        foreach (x; [nextDown(T(0.5)), nextUp(T(0.5))])
        {
            const real delta = cast(real) x - 0.5L;
            const real expected = betaLPDF(0.5L, cast(real) shape, cast(real) shape)
                + (cast(real) shape - 1) * log1p(-4 * delta * delta);
            assert(approxEqual(betaLPDF(x, shape, shape), cast(T) expected,
                64 * T.epsilon));
        }
        // The peak at 1/3 is not representable. These independent references
        // use the exact stored input, not the mathematical value 1/3.
        static if (T.mant_dig == 24)
            enum T offPeak = -844424921743325.2414846717984407322334L;
        else static if (T.mant_dig == 53)
            enum T offPeak = 35.03683565001478463916689363816247455347L;
        else static if (T.mant_dig == 64)
            enum T offPeak = 35.03976533681629268049036671728932642385L;
        static if (T.mant_dig == 24 || T.mant_dig == 53 || T.mant_dig == 64)
            assert(approxEqual(betaLPDF(T(1) / 3, T(0x1p100L), T(0x1p101L)),
                offPeak, 64 * T.epsilon));
    }}
}

// Keep the original float input when delegating outside the bounded fallback.
// Forming 1-x in float rounds both inputs to one. References were computed
// independently with high-precision incomplete-beta arithmetic.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(approxEqual(betaCCDF(0x1p-30f, 0.5f, 129f),
        0.999609267860570611262336622325055137f, 4 * float.epsilon, 0.0f));
    assert(approxEqual(betaCCDF(0x1p-30f, 0.125f, 1000.5f),
        0.812841910175758869528918774417910939f, 4 * float.epsilon, 0.0f));
}
