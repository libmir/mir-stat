/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution).

An alternate parameterization of the $(MREF mir,stat,distribution,beta)
distribution in terms of its mean and the sum of its shape parameters.

The shape parameters are computed as $(D mu * kappa) and
$(D (1 - mu) * kappa). Computing both products directly preserves the
smaller shape when the mean is close to zero or one.

The mean must satisfy $(D 0 < mu < 1), and the concentration $(D kappa) must
be positive. Larger concentration reduces the variance at a fixed mean:
$(D mu * (1 - mu) / (kappa + 1)). Concentration need not be an integer or an
actual observation count. Observations must lie in $(D [0, 1]).

For the tables below, $(D alpha = mu * kappa) and
$(D beta = (1 - mu) * kappa). For built-in overloads both computed shapes
must remain positive and finite in the working type. The inverse CDF accepts
both probability endpoints.

The extended-precision overloads form the lower shape exactly and carry both
shape exponents separately. They enclose the complementary shape on a grid
with twice the input precision plus 64 bits. If the shape lies between grid
points, tails and quantiles require agreement at both bounds. Interior
densities propagate the shape displacement through a log-density derivative
bound. At the exact mean with large shapes, the density uses the known
concentration directly in its Stirling normalization. Both bounds must round
to the same result. Unresolved cases return NaN. The guarded beta integration
and inversion limits apply.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

Macros:
DISTREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, distribution, $1)$(NBSP)

+/

module mir.stat.distribution.beta_proportion;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: position;
import mir.internal.utility: isFloatingPoint;

/++
Computes the beta proportion probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.
The table uses $(D alpha = mu * kappa) and $(D beta = (1 - mu) * kappa).

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
    mu = mean, strictly between zero and one
    kappa = positive sum of the two shape parameters

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution),
    $(DISTREF beta, betaPDF)
+/
@safe pure nothrow @nogc
T betaProportionPDF(T)(const T x, const T mu, const T kappa)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (mu > 0, "mu must be greater than zero")
    in (mu < 1, "mu must be less than one")
    in (kappa > 0, "kappa must be greater than zero")
{
    import mir.stat.distribution.beta: betaPDF;

    immutable T alpha = mu * kappa;
    immutable T beta = (1 - mu) * kappa;
    return betaPDF(x, alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaProportionPDF(0.5, 2) == 1);
    assert(0.75.betaProportionPDF((1.0 / 3), 3).approxEqual(0.5));
    assert(0.25.betaProportionPDF((1.0 / 9), 4.5).approxEqual(0.9228516));
}

/// At a fixed mean of one half, concentration changes both endpoint densities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(betaProportionPDF(0.0, 0.5, 1.0) == double.infinity);
    assert(betaProportionPDF(0.0, 0.5, 2.0) == 1.0);
    assert(betaProportionPDF(0.0, 0.5, 4.0) == 0.0);
    assert(betaProportionPDF(1.0, 0.5, 4.0) == 0.0);
}

/++
Evaluates the beta-proportion PDF in guarded extended precision.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    x = observation in the closed unit interval
    mu = finite mean strictly between zero and one
    kappa = finite positive concentration

Returns:
    The density as $(D T), with the beta density limits at zero and one.
    Returns NaN when the shape bounds or the underlying beta evaluation are unresolved.
+/
@safe pure nothrow @nogc
T betaProportionPDF(T)(const T x, const T mu, const T kappa)
    if (is(T == Fp!size, size_t size) && size <= 512)
    in (x.exponent != long.max && (!x.sign || !x.coefficient) && position(x, T(1)) <= 0)
    in (mu.exponent != long.max && !mu.sign && !!mu.coefficient && position(mu, T(1)) < 0)
    in (kappa.exponent != long.max && !kappa.sign && !!kappa.coefficient)
{
    import mir.stat.internal.fp_beta_proportion: betaProportion, BetaOperation;
    return betaProportion!(BetaOperation.pdf)(x, mu, kappa);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(betaProportionPDF(T(0.5), T(0.5), T(4)) == T(1.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    // With shapes (3,1), both parameterizations round the same 3*x^2 midpoint up.
    const x = T(false, -32, UInt!64(3000000009UL));
    assert(betaProportionPDF(x, T(0.75), T(4)) == T(false, -63, UInt!64(13500000081000000122UL)));
}

/++
Computes the beta proportion cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).
The table uses $(D alpha = mu * kappa) and $(D beta = (1 - mu) * kappa).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == 1)) $(TD $(D 1)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    mu = mean, strictly between zero and one
    kappa = positive sum of the two shape parameters

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution),
    $(DISTREF beta, betaCDF)
+/
@safe pure nothrow @nogc
T betaProportionCDF(T)(const T x, const T mu, const T kappa)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (mu > 0, "mu must be greater than zero")
    in (mu < 1, "mu must be less than one")
    in (kappa > 0, "kappa must be greater than zero")
{
    import mir.stat.distribution.beta: betaCDF;

    immutable T alpha = mu * kappa;
    immutable T beta = (1 - mu) * kappa;
    return betaCDF(x, alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaProportionCDF(0.5, 2).approxEqual(0.5));
    assert(0.75.betaProportionCDF((1.0 / 3), 3).approxEqual(0.9375));
    assert(0.25.betaProportionCDF((1.0 / 9), 4.5).approxEqual(0.8588867));
}

/++
Evaluates the beta-proportion CDF in guarded extended precision.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    x = observation in the closed unit interval
    mu = finite mean strictly between zero and one
    kappa = finite positive concentration

Returns:
    The lower probability as $(D T), zero at zero and one at one.
    Returns NaN when the shape bounds or the underlying beta evaluation are unresolved.
+/
@safe pure nothrow @nogc
T betaProportionCDF(T)(const T x, const T mu, const T kappa)
    if (is(T == Fp!size, size_t size) && size <= 512)
    in (x.exponent != long.max && (!x.sign || !x.coefficient) && position(x, T(1)) <= 0)
    in (mu.exponent != long.max && !mu.sign && !!mu.coefficient && position(mu, T(1)) < 0)
    in (kappa.exponent != long.max && !kappa.sign && !!kappa.coefficient)
{
    import mir.stat.internal.fp_beta_proportion: betaProportion, BetaOperation;
    return betaProportion!(BetaOperation.cdf)(x, mu, kappa);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(betaProportionCDF(T(0.5), T(0.5), T(4)) == T(0.5));
}

/++
Computes the beta proportion complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.
The table uses $(D alpha = mu * kappa) and $(D beta = (1 - mu) * kappa).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == 1)) $(TD $(D 0)))
    $(TR $(TD Outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    mu = mean, strictly between zero and one
    kappa = positive sum of the two shape parameters

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution),
    $(DISTREF beta, betaCCDF)
+/
@safe pure nothrow @nogc
T betaProportionCCDF(T)(const T x, const T mu, const T kappa)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (mu > 0, "mu must be greater than zero")
    in (mu < 1, "mu must be less than one")
    in (kappa > 0, "kappa must be greater than zero")
{
    import mir.stat.distribution.beta: betaCCDF;

    immutable T alpha = mu * kappa;
    immutable T beta = (1 - mu) * kappa;
    return betaCCDF(x, alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaProportionCCDF(0.5, 2).approxEqual(0.5));
    assert(0.75.betaProportionCCDF((1.0 / 3), 3).approxEqual(0.0625));
    assert(0.25.betaProportionCCDF((1.0 / 9), 4.5).approxEqual(0.1411133));
}

/++
Evaluates the beta-proportion CCDF in guarded extended precision.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    x = observation in the closed unit interval
    mu = finite mean strictly between zero and one
    kappa = finite positive concentration

Returns:
    The strict upper probability as $(D T), one at zero and zero at one.
    Returns NaN when the shape bounds or the underlying beta evaluation are unresolved.
+/
@safe pure nothrow @nogc
T betaProportionCCDF(T)(const T x, const T mu, const T kappa)
    if (is(T == Fp!size, size_t size) && size <= 512)
    in (x.exponent != long.max && (!x.sign || !x.coefficient) && position(x, T(1)) <= 0)
    in (mu.exponent != long.max && !mu.sign && !!mu.coefficient && position(mu, T(1)) < 0)
    in (kappa.exponent != long.max && !kappa.sign && !!kappa.coefficient)
{
    import mir.stat.internal.fp_beta_proportion: betaProportion, BetaOperation;
    return betaProportion!(BetaOperation.ccdf)(x, mu, kappa);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(betaProportionCCDF(T(0.5), T(0.5), T(4)) == T(0.5));
}

/++
Computes the beta proportion inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).
The table uses $(D alpha = mu * kappa) and $(D beta = (1 - mu) * kappa).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D 1)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    mu = mean, strictly between zero and one
    kappa = positive sum of the two shape parameters

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution),
    $(DISTREF beta, betaInvCDF)
+/
@safe pure nothrow @nogc
T betaProportionInvCDF(T)(const T p, const T mu, const T kappa)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (mu > 0, "mu must be greater than zero")
    in (mu < 1, "mu must be less than one")
    in (kappa > 0, "kappa must be greater than zero")
{
    import mir.stat.distribution.beta: betaInvCDF;

    immutable T alpha = mu * kappa;
    immutable T beta = (1 - mu) * kappa;
    return betaInvCDF(p, alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.betaProportionInvCDF(0.5, 2).approxEqual(0.5));
    assert(0.9375.betaProportionInvCDF((1.0 / 3), 3).approxEqual(0.75));
    assert(0.8588867.betaProportionInvCDF((1.0 / 9), 4.5).approxEqual(0.25));
}

/++
Evaluates the beta-proportion quantile in guarded extended precision.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    p = cumulative probability in the closed unit interval
    mu = finite mean strictly between zero and one
    kappa = finite positive concentration

Returns:
    The quantile as $(D T), zero at probability zero and one at probability one.
    Returns NaN when the shape bounds or the underlying beta evaluation are unresolved.
+/
@safe pure nothrow @nogc
T betaProportionInvCDF(T)(const T p, const T mu, const T kappa)
    if (is(T == Fp!size, size_t size) && size <= 512)
    in (p.exponent != long.max && (!p.sign || !p.coefficient) && position(p, T(1)) <= 0)
    in (mu.exponent != long.max && !mu.sign && !!mu.coefficient && position(mu, T(1)) < 0)
    in (kappa.exponent != long.max && !kappa.sign && !!kappa.coefficient)
{
    import mir.stat.internal.fp_beta_proportion: betaProportion, BetaOperation;
    return betaProportion!(BetaOperation.quantile)(p, mu, kappa);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(betaProportionInvCDF(T(0.5), T(0.5), T(4)) == T(0.5));
}

/++
Computes the beta proportion log probability density function (LPDF).

Returns the natural logarithm of the density.
The table uses $(D alpha = mu * kappa) and $(D beta = (1 - mu) * kappa).

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
    mu = mean, strictly between zero and one
    kappa = positive sum of the two shape parameters

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Beta_distribution, Beta Proportion Distribution),
    $(DISTREF beta, betaLPDF)
+/
@safe pure nothrow @nogc
T betaProportionLPDF(T)(const T x, const T mu, const T kappa)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (x <= 1, "x must be less than or equal to 1")
    in (mu > 0, "mu must be greater than zero")
    in (mu < 1, "mu must be less than one")
    in (kappa > 0, "kappa must be greater than zero")
{
    import mir.stat.distribution.beta: betaLPDF;

    immutable T alpha = mu * kappa;
    immutable T beta = (1 - mu) * kappa;
    return betaLPDF(x, alpha, beta);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(0.5.betaProportionLPDF(0.5, 2).approxEqual(log(betaProportionPDF(0.5, 0.5, 2))));
    assert(0.75.betaProportionLPDF((1.0 / 3), 3).approxEqual(log(betaProportionPDF(0.75, (1.0 / 3), 3))));
    assert(0.25.betaProportionLPDF((1.0 / 9), 4.5).approxEqual(log(betaProportionPDF(0.25, (1.0 / 9), 4.5))));
}

/++
Evaluates the beta-proportion LPDF in guarded extended precision.

Params:
    T = supported $(D Fp) type with at most 512 significand bits
    x = observation in the closed unit interval
    mu = finite mean strictly between zero and one
    kappa = finite positive concentration

Returns:
    The natural log density as $(D T), including the logarithmic endpoint limits.
    Returns NaN when the shape bounds or the underlying beta evaluation are unresolved.
+/
@safe pure nothrow @nogc
T betaProportionLPDF(T)(const T x, const T mu, const T kappa)
    if (is(T == Fp!size, size_t size) && size <= 512)
    in (x.exponent != long.max && (!x.sign || !x.coefficient) && position(x, T(1)) <= 0)
    in (mu.exponent != long.max && !mu.sign && !!mu.coefficient && position(mu, T(1)) < 0)
    in (kappa.exponent != long.max && !kappa.sign && !!kappa.coefficient)
{
    import mir.stat.internal.fp_beta_proportion: betaProportion, BetaOperation;
    return betaProportion!(BetaOperation.lpdf)(x, mu, kappa);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(betaProportionLPDF(T(0.5), T(0.5), T(2)) == T(0));
}

// Both floating-point families see exactly (alpha,beta) = (3,5).
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    alias T = Fp!64;
    assert(betaProportionPDF(T(0.25), T(0.375), T(8)) == T(2.076416015625));
    assert(betaProportionPDF(0.25, 0.375, 8.0) == 2.076416015625);
    assert(betaProportionCDF(T(0.25), T(0.375), T(8)) == T(0.24359130859375));
    assert(approxEqual(betaProportionCDF(0.25, 0.375, 8.0), 0.24359130859375));
    assert(betaProportionCCDF(T(0.25), T(0.375), T(8)) == T(0.75640869140625));
    assert(approxEqual(betaProportionCCDF(0.25, 0.375, 8.0), 0.75640869140625));
    import mir.bignum.fixed: UInt;
    import std.math: log;
    assert(betaProportionLPDF(T(0.25), T(0.375), T(8)) ==
        T(false, -64, UInt!64(0xbb0b711a39b2473bUL)));
    assert(approxEqual(betaProportionLPDF(0.25, 0.375, 8.0), log(2.076416015625), 64 * double.epsilon));
    assert(approxEqual(betaProportionInvCDF(0.5, 0.5, 4.0), 0.5, 64 * double.epsilon));
    assert(betaProportionInvCDF(T(0), T(0.375), T(8)) == T(0));
    assert(betaProportionCDF(T(1), T(0.375), T(8)) == T(1));
    assert(betaProportionCCDF(T(0), T(0.375), T(8)) == T(1));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    // The products around one differ by as little as 2^-126.
    const aboveHalf = T(false, -64, U(0x8000000000000001UL));
    const belowHalf = T(false, -65, U.max - 1);
    const belowTwo = T(false, -63, U.max - 1);
    assert(betaProportionPDF(T(0), aboveHalf, belowTwo).isInfinity);
    assert(betaProportionPDF(T(1), belowHalf, belowTwo).isInfinity);
    assert(betaProportionPDF(T(0), aboveHalf, T(false, -63, U.max)) == T(0));

    const tiny = T(false, -1000, U(1));
    assert(betaProportionPDF(T(1), tiny, T(1)).isInfinity); // beta strictly below one
    assert(betaProportionPDF(T(1), tiny, T(2)) == T(0));
    // Part of alpha, but not its low product bits, fits the kappa grid.
    const skew = T(false, -192, U(0x8000000000000001UL));
    assert(betaProportionPDF(T(1), skew, T(false, -63, U(0x8000000000000001UL))) == T(0));
    const large = T(false, 1000, U(1));
    assert(position(betaProportionPDF(T(0), tiny, large), large) == 0);
    // For kappa=1, B(mu,1-mu)=pi/sin(pi*mu). At x=mu the leading
    // log density is mu*log(mu), with an O(mu^2) correction.
    assert(betaProportionLPDF(tiny, tiny, T(1)) == T(true, -1054, U(0xad496b6802e49cd2UL)));
    assert(position(betaProportionPDF(T(0.5), tiny, T(1)), T(false, -999, U(1))) == 0);
    assert(betaProportionPDF(T(0.75), tiny, T(1)) == T(false, -1063, U(0xaaaaaaaaaaaaaaabUL)));
    // Q(1/2) = mu*log(2) + O(mu^2).
    assert(betaProportionCCDF(T(0.5), tiny, T(1)) == T(false, -1064, U(0xb17217f7d1cf79acUL)));
    assert(betaProportionInvCDF(T(0.5), tiny, T(1)) == T(0));
    const floor = T(false, long.min, U(1));
    // Both shapes are half the least representable Fp!64.
    assert(betaProportionPDF(T(0.5), T(0.5), floor) == floor);
    assert(betaProportionCDF(T(0.5), T(0.5), floor) == T(0.5));
    assert(betaProportionInvCDF(T(0.5), T(0.5), floor) == T(0.5));
    // beta = (2^192 - 1)*2^400 fits exactly on the grid below kappa.
    // One step above mu, the log density rounded from a 450-digit
    // Stirling reference has coefficient 2^64 - 1.
    const boundaryMean = T(false, -192, U(1));
    const nextMean = T(false, -255, U(0x8000000000000001UL));
    assert(betaProportionLPDF(nextMean, boundaryMean, T(false, 592, U(1))) ==
        T(true, 209, U.max));
    // At the exact mean the height is 2^850/sqrt(2*pi) up to a relative
    // correction of order 2^-400. An independently rounded shape moves
    // the center by many standard deviations.
    const mean = T(false, -400, U(1));
    const concentration = T(false, 1300, U(1));
    assert(betaProportionLPDF(mean, mean, concentration) == T(false, -54, U(0x93106501a4df3b16UL)));
    assert(betaProportionPDF(mean, mean, concentration) == T(false, 785, U(0xcc42299ea1b28468UL)));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    // MPFR log-gamma normalization and independent binomial integration
    // at the exact dyadic mean and concentration, including their low bits.
    const x = T(false, -65, U(0xc000000000000001UL));
    const mu = T(false, -65, U(0xbfffffffffffffffUL));
    const k = T(false, -61, U(0xe000000000000001UL));
    assert(betaProportionPDF(x, mu, k) == T(false, -62, U(0x863c6bea7d839f8cUL)));
    assert(betaProportionLPDF(x, mu, k) == T(false, -64, U(0xbd9f9919ab62f6e1UL)));
    assert(betaProportionCDF(x, mu, k) == T(false, -64, U(0x86c1ef6a6a8f216dUL)));
    assert(betaProportionCCDF(x, mu, k) == T(false, -65, U(0xf27c212b2ae1bd27UL)));
    assert(betaProportionInvCDF(T(false, -65, U.max), mu, k) == T(false, -65, U(0xb9988a0ba34a14f9UL)));
}

// A mean close to one must retain the small complementary shape in every API.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, log;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T mu = nextDown(T(1));
        const T smallShape = 3 * (1 - mu);
        // As alpha approaches 3 and beta approaches zero, PDF(1/2) is
        // beta/2 + O(beta^2), and CDF(1/2) is beta*(log(2)-5/8) + O(beta^2).
        // The finite-shape corrections here are within the stated tolerance.
        const T density = smallShape / 2;
        const T probability = smallShape * (log(T(2)) - T(0.625));
        assert(approxEqual(betaProportionPDF(T(0.5), mu, T(3)), density,
            64 * T.epsilon, T(0)));
        assert(approxEqual(betaProportionLPDF(T(0.5), mu, T(3)), log(density),
            64 * T.epsilon, T(0)));
        assert(approxEqual(betaProportionCDF(T(0.5), mu, T(3)), probability,
            64 * T.epsilon, T(0)));
        assert(approxEqual(betaProportionInvCDF(probability, mu, T(3)), T(0.5),
            64 * T.epsilon, T(0)));

        // Near the upper endpoint the CDF is large enough that an incorrect
        // small shape also causes an observable error in its complement.
        const T x = nextDown(T(1));
        const T cdf = smallShape * (-log(1 - x) - x - x * x / 2);
        assert(approxEqual(betaProportionCCDF(x, mu, T(3)), 1 - cdf,
            T(0), 2 * T.epsilon));

        // Swapping the mean reflects the density about one half.
        assert(approxEqual(betaProportionPDF(T(0.5), 1 - mu, T(3)), density,
            64 * T.epsilon, T(0)));
    }}
}
