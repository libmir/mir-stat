/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion, Cornish-Fisher Expansion).

The Cornish-Fisher expansion adjusts normal quantiles using skewness and
excess kurtosis. It is an approximation, not a distribution with a guaranteed
CDF or density. Even with finite inputs, the truncated expansion can produce
quantiles that are not increasing with probability or lie outside the intended
distribution's support.

Use finite moment parameters and a positive finite standard deviation. These
moment requirements are not checked by contracts. Probability is checked by
the underlying normal inverse CDF. With nonzero corrections, use interior
probabilities and assess accuracy for the intended distribution, especially
in the tails.

The special-value tables assume parameters in the stated domain. Finite
expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts
does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.cornish_fisher;

import mir.internal.utility: isFloatingPoint;

/++
Approximates the inverse CDF of a continuous distribution using the Cornish-Fisher expansion.

It is generally recommended to only use the Cornish-Fisher expansion with
distributions that are similar to the normal distribution. Extreme values of
`skewness` or `excessKurtosis` can result in poorer approximations.
The truncated expansion is not guaranteed to be monotone in `p`, and
nonzero corrections do not in general define valid endpoint quantiles.
When both corrections are zero, the result is the normal quantile,
including its infinite endpoints.

Returns a Cornish-Fisher quantile approximation.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0), zero skewness and excess kurtosis) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D mu - std * skewness / 6)))
    $(TR $(TD $(D p == 1), zero skewness and excess kurtosis) $(TD $(D T.infinity)))
    $(TR $(TD $(D p == 0) or $(D p == 1), nonzero corrections) $(TD No valid endpoint quantile is guaranteed))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation in the normal inverse CDF))
)

Params:
    p = cumulative probability in [0, 1]; use interior probabilities with nonzero corrections
    mu = finite mean
    std = positive finite standard deviation
    skewness = skewness
    excessKurtosis = excess kurtosis (kurtosis - 3)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion, Cornish-Fisher Expansion)
+/
T cornishFisherInvCDF(T)(const T p, const T mu, const T std, const T skewness, const T excessKurtosis)
    if (isFloatingPoint!T)
{
    import mir.stat.distribution.normal: normalInvCDF;
    if (skewness == 0 && excessKurtosis == 0 && std > 0)
        return normalInvCDF(p, mu, std);
    return mu + std * cornishFisherInvCDF(p, skewness, excessKurtosis);
}

///
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.test: shouldApprox;

    0.99.cornishFisherInvCDF(0, 1, 0.1, 1).shouldApprox == 2.629904;
    0.99.cornishFisherInvCDF(0.1, 0.2, 0.1, 1).shouldApprox == 0.6259808;
}

/// Zero corrections recover normal quantiles, including the endpoints.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(cornishFisherInvCDF(0.5, 3.0, 2.0, 0.0, 0.0) == 3.0);
    assert(cornishFisherInvCDF(0.0) == -double.infinity);
    assert(cornishFisherInvCDF(1.0) == double.infinity);
}

/++
Ditto, but assumes mu = 0 and std = 1

Returns a Cornish-Fisher quantile approximation.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0), zero skewness and excess kurtosis) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D -skewness / 6)))
    $(TR $(TD $(D p == 1), zero skewness and excess kurtosis) $(TD $(D T.infinity)))
    $(TR $(TD $(D p == 0) or $(D p == 1), nonzero corrections) $(TD No valid endpoint quantile is guaranteed))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation in the normal inverse CDF))
)

Params:
    p = cumulative probability in [0, 1]; use interior probabilities with nonzero corrections
    skewness = skewness (default = 0)
    excessKurtosis = excess kurtosis (kurtosis - 3) (default = 0)
+/
T cornishFisherInvCDF(T)(const T p, const T skewness = 0, const T excessKurtosis = 0)
    if (isFloatingPoint!T)
{
    import mir.stat.distribution.normal: normalInvCDF;

    T x = normalInvCDF(p);
    return x.cornishFisherInvCDFImpl(skewness, excessKurtosis);
}

///
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.test: shouldApprox;

    0.5.cornishFisherInvCDF.shouldApprox == 0;
    0.5.cornishFisherInvCDF(1).shouldApprox == -0.1666667;
    0.5.cornishFisherInvCDF(-1, 0).shouldApprox == 0.1666667;

    0.9.cornishFisherInvCDF(0.1, 0).shouldApprox == 1.292868;
    0.9.cornishFisherInvCDF(0.1, 1).shouldApprox ==  1.220374;
    0.9.cornishFisherInvCDF(0.1, -1).shouldApprox == 1.365363;

    0.99.cornishFisherInvCDF(0.1, 0).shouldApprox == 2.396116;
    0.99.cornishFisherInvCDF(0.1, 1).shouldApprox == 2.629904;
    0.99.cornishFisherInvCDF(0.1, -1).shouldApprox == 2.162328;

    0.01.cornishFisherInvCDF(0.1, 0).shouldApprox == -2.249053  ;
    0.01.cornishFisherInvCDF(0.1, 1).shouldApprox == -2.482841;
    0.01.cornishFisherInvCDF(0.1, -1).shouldApprox == -2.015265;
}

package(mir.stat)
T cornishFisherInvCDFImpl(T)(const T x, const T skewness, const T excessKurtosis)
    if (isFloatingPoint!T)
{
    import mir.stat.distribution.normal: normalInvCDF;

    // Avoid zero times infinity in unused correction terms at p=0 or p=1.
    if (skewness == 0 && excessKurtosis == 0)
        return x;
    T x2 = x * x;
    T x3 = x2 * x;
    return x + (x2 - 1) * skewness / 6 + (x3 - 3 * x) * excessKurtosis / 24 - (2 * x3 - 5 * x) * skewness * skewness / 36;
}

//
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.test: shouldApprox;

    0.0.cornishFisherInvCDFImpl(0, 0).shouldApprox == 0;
    0.0.cornishFisherInvCDFImpl(1, 0).shouldApprox == -0.1666667;
    0.0.cornishFisherInvCDFImpl(-1, 0).shouldApprox == 0.1666667;

    1.281552.cornishFisherInvCDFImpl(0.1, 0).shouldApprox == 1.292868;
    1.281552.cornishFisherInvCDFImpl(0.1, 1).shouldApprox ==  1.220374;
    1.281552.cornishFisherInvCDFImpl(0.1, -1).shouldApprox == 1.365363;

    2.326348.cornishFisherInvCDFImpl(0.1, 0).shouldApprox == 2.396116;
    2.326348.cornishFisherInvCDFImpl(0.1, 1).shouldApprox == 2.629904;
    2.326348.cornishFisherInvCDFImpl(0.1, -1).shouldApprox == 2.162328;

    (-2.326348).cornishFisherInvCDFImpl(0.1, 0).shouldApprox == -2.249053  ;
    (-2.326348).cornishFisherInvCDFImpl(0.1, 1).shouldApprox == -2.482841;
    (-2.326348).cornishFisherInvCDFImpl(0.1, -1).shouldApprox == -2.015265;
}

// Zero corrections reduce exactly to the normal quantile, including endpoints.
version(mir_stat_test)
@safe pure @nogc nothrow unittest
{
    import std.meta: AliasSeq;
    import mir.stat.distribution.normal: normalInvCDF;
    static foreach (T; AliasSeq!(float, double, real))
    {
        foreach (p; [T(0), T(0.1), T(0.5), T(0.9), T(1)])
        {
            assert(cornishFisherInvCDF(p) == normalInvCDF(p));
            assert(cornishFisherInvCDF(p, T(3), T(2), T(0), T(0)) ==
                normalInvCDF(p, T(3), T(2)));
        }
    }
}
