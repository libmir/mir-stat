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
$(D beta = (1 - mu) * kappa). Both computed shapes must remain positive
and finite in the working floating-point type. The inverse CDF accepts
both probability endpoints.

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
