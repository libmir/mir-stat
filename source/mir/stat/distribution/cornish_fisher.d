/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion, Cornish-Fisher Expansion).

The Cornish-Fisher expansion adjusts normal quantiles using skewness and
excess kurtosis. It is an approximation, not a distribution with a guaranteed
CDF or density. Even with finite inputs, the truncated expansion can produce
quantiles that are not increasing with probability or lie outside the intended
distribution's support.

Use finite moment parameters and a positive finite standard deviation.
The built-in overloads leave the moment requirements unchecked and use
the normal inverse CDF to check the probability. The $(D Fp) overloads
require interior probabilities when either correction is nonzero and use
guarded arithmetic throughout the expansion and location-scale transform.
Reference tests cover 32-, 64-, 128-, and 256-bit significands, with
$(D Fp!32) tested on 32-bit targets. Universal correct rounding is unproven.
Assess the approximation for the intended distribution, especially in the tails.

The special-value tables assume parameters in the stated domain. Finite
expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts
does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.cornish_fisher;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityLess, fpProbabilityValid;
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
Evaluates the location-scale Cornish-Fisher quantile approximation with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    p = Cumulative probability. Use an interior value when either correction is nonzero.
    mu = Finite location.
    std = Positive finite standard deviation.
    skewness = Finite skewness.
    excessKurtosis = Finite excess kurtosis.

Returns:
    The truncated expansion in type T. Zero corrections give the normal quantile,
    including its infinite endpoints. At one half the result is
    $(D mu - std * skewness / 6). Cancellation beyond working precision can lose significant
    bits. The expansion itself can be nonmonotone even when evaluated accurately.
+/
@safe pure nothrow @nogc
T cornishFisherInvCDF(T)(const T p, const T mu, const T std, const T skewness, const T excessKurtosis)
    if (is(T == Fp!size, size_t size))
    in (!mu.isSpecial && !std.isSpecial && !std.sign && !!std.coefficient)
    in (!skewness.isSpecial && !excessKurtosis.isSpecial)
    in (fpProbabilityValid(p))
    in ((!skewness.coefficient && !excessKurtosis.coefficient) ||
        (p.coefficient && fpProbabilityLess(p, T(1))))
{
    return correctedQuantile(p, mu, std, skewness, excessKurtosis);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(cornishFisherInvCDF(T(0.5), T(3), T(2), T(6), T(1)) == T(1));
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

/++
Evaluates the standardized Cornish-Fisher quantile approximation with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    p = Cumulative probability. Use an interior value when either correction is nonzero.
    skewness = Finite skewness, zero by default.
    excessKurtosis = Finite excess kurtosis, zero by default.

Returns:
    The truncated expansion in type T. Zero corrections give the normal quantile,
    including its infinite endpoints. At one half the result is
    $(D -skewness / 6). Cancellation beyond working precision can lose significant
    bits. The expansion itself can be nonmonotone even when evaluated accurately.
+/
@safe pure nothrow @nogc
T cornishFisherInvCDF(T)(const T p, const T skewness = T(0), const T excessKurtosis = T(0))
    if (is(T == Fp!size, size_t size))
{
    return cornishFisherInvCDF(p, T(0), T(1), skewness, excessKurtosis);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(cornishFisherInvCDF(T(0.5), T(3)) == T(-0.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.distribution.normal: normalInvCDF;
    alias T = Fp!64;
    const median = T(0.5);
    // Zero corrections take the same path as the normal quantile.
    foreach (p; [T(0), T(0.125), median, T(1)])
    {
        assert(cornishFisherInvCDF(p) == normalInvCDF(p));
        assert(cornishFisherInvCDF(p, T(3), T(2), T(0), T(0)) == normalInvCDF(p, T(3), T(2)));
    }

    // At the median, kurtosis drops out and mu - std*skewness/6 is rational.
    assert(cornishFisherInvCDF(median, T(7), T(4), T(0), T(2)) == T(7));
    assert(cornishFisherInvCDF(median, T(7), T(3), T(14), T(2)) == T(0));
    const huge = T(false, long.max - 1, UInt!64.max);
    assert(cornishFisherInvCDF(median, huge, huge, T(12), T(0)) ==
        T(true, long.max - 1, UInt!64.max));
    const least = T(false, long.min, UInt!64(1));
    assert(cornishFisherInvCDF(median, T(0), least, T(-3), T(0)) == T(0));
    assert(cornishFisherInvCDF(median, T(0), least, T(-9), T(0)) == T(false, long.min, UInt!64(2)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    alias T = Fp!64;
    const p = T(0.875);
    const skew = T(0.25);
    const kurt = T(-0.75);
    // Bisect MPFR erfc at 1,536 bits and evaluate the original expansion
    // with rational arithmetic at the represented inputs.
    assert(cornishFisherInvCDF(p, T(1), T(2), skew, kurt) ==
        T(false, -62, UInt!64(0xdd4927790b17e8d1UL)));
    assert(approxEqual(cornishFisherInvCDF(0.875, 1.0, 2.0, 0.25, -0.75),
        3.457589977461224, 8 * double.epsilon));
    const location = T(true, -63, UInt!64(0x9d4927790b17e8d1UL));
    assert(cornishFisherInvCDF(p, location, T(1), skew, kurt) ==
        T(false, -129, UInt!64(0xac4ae9d83b998330UL)));

    // k = 4*s*s/3 cancels the cubic coefficient before scaling.
    auto bigSkew = T(3), bigKurt = T(12), smallScale = T(1);
    bigSkew.exponent += 8000;
    bigKurt.exponent += 16000;
    smallScale.exponent -= 16000;
    assert(cornishFisherInvCDF(p, T(0), smallScale, bigSkew, bigKurt) ==
        T(true, -65, UInt!64(0x933ea603d82b3b2aUL)));

    // s*s is outside the Fp exponent range, but std*s*s = 2^100.
    // The terms independent of s and linear in s are negligible at this p.
    auto extremeSkew = T(1);
    extremeSkew.exponent += (1L << 62) + 50;
    const leastScale = T(false, long.min, UInt!64(1));
    assert(cornishFisherInvCDF(p, T(0), leastScale, extremeSkew, T(0)) ==
        T(false, 33, UInt!64(0x9a02cd9a7d67e0f5UL)));

    const below = T(false, -65, UInt!64.max);
    assert(cornishFisherInvCDF(below, T(1), T(2), skew, kurt) ==
        T(false, -64, UInt!64(0xeaaaaaaaaaaaaaa8UL)));
    // With s=6, k=48, mu=std=1 the entire expansion reduces to z*z.
    // At the neighbor below one half this rounds to pi * 2^-129.
    assert(cornishFisherInvCDF(below, T(1), T(1), T(6), T(48)) ==
        T(false, -191, UInt!64(0xc90fdaa22168c235UL)));
    auto tail = T(1);
    tail.exponent -= 9000;
    assert(cornishFisherInvCDF(tail, T(1), T(2), skew, kurt) ==
        T(false, -47, UInt!64(0xbe5117a308d3dd38UL)));

    alias W = Fp!128;
    enum wide = W(false, -126, UInt!128.fromHexString("dd4927790b17e8d12b12ba760ee660cc"));
    assert(cornishFisherInvCDF!W(W(0.875), W(1), W(2), W(0.25), W(-0.75)) == wide);
    static if (size_t.sizeof == 4)
    {
        alias N = Fp!32;
        // Construct dyadics without narrowing a double significand to 32 bits.
        auto sevenEighths = N(7), quarter = N(1), minusThreeQuarters = N(-3);
        sevenEighths.exponent -= 3;
        quarter.exponent -= 2;
        minusThreeQuarters.exponent -= 2;
        assert(cornishFisherInvCDF(sevenEighths, N(1), N(2), quarter, minusThreeQuarters) ==
            N(false, -30, UInt!32(0xdd492779)));
    }
}

version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled)
    {
        alias T = Fp!64;
        assertThrown!AssertError(cornishFisherInvCDF(T(0), T(1)));
        assertThrown!AssertError(cornishFisherInvCDF(T(1), T(0), T(1)));
        assertThrown!AssertError(cornishFisherInvCDF(T(double.nan)));
        assertThrown!AssertError(cornishFisherInvCDF(T(0.5), T(double.infinity)));
        assertThrown!AssertError(cornishFisherInvCDF(T(0.5), T(0), T(0), T(1), T(1)));
    }
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

// Use a common denominator so cancellation in the polynomial coefficients
// takes place before multiplication by the guarded normal quantile.
private @safe pure nothrow @nogc
Fp!bits correctedQuantile(uint bits)(Fp!bits p, Fp!bits mu, Fp!bits std,
    Fp!bits skewness, Fp!bits excessKurtosis)
{
    import mir.stat.distribution.normal: normalInvCDF;
    import mir.stat.internal.fp_location_scale: addOffset, half, position;
    import mir.stat.internal.fp_normal: standardNormalQuantileMagnitude;
    import mir.stat.internal.fp_shape_transform: ShapeTransform;
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias Term = B.Term;
    alias S = B.S;
    if (!skewness.coefficient && !excessKurtosis.coefficient)
        return normalInvCDF(p, mu, std);

    const side = position(p, half!bits());
    const scale = S.from(std);
    if (!side)
    {
        if (!skewness.coefficient) return mu;
        const offset = scale.multiply(S.from(skewness)).divide(S.from(T(6)));
        return addOffset!bits(mu, offset, !skewness.sign);
    }

    static Term times(Term a, S factor)
    {
        if (a.present) a.value = a.value.multiply(factor);
        return a;
    }
    const z = standardNormalQuantileMagnitude!bits(p, side < 0);
    const z2 = z.multiply(z);
    const denominator = S.from(T(72));
    const kurt = B.term(excessKurtosis);
    auto skew2 = B.term(skewness);
    if (skew2.present)
        skew2 = Term(skew2.value.multiply(skew2.value), false, true);

    // 72 times the odd part is z * ((3*k - 4*s*s)*z*z + 72 - 9*k + 10*s*s).
    auto cubicSkew = times(skew2, S.from(T(4)));
    cubicSkew.negative = true;
    const cubic = B.combine(times(kurt, S.from(T(3))), cubicSkew);
    auto linearKurt = times(kurt, S.from(T(9)));
    linearKurt.negative = !linearKurt.negative;
    const linearCorrection = B.combine(linearKurt, times(skew2, S.from(T(10))));
    auto odd = B.combineThree(times(cubic, z2), linearCorrection, B.term(T(72)));
    odd = times(odd, scale.multiply(z));
    odd.negative ^= side < 0;

    // Group the constant with the location before adding the z*z term.
    auto skewScale = times(B.term(skewness), scale.multiply(S.from(T(12))));
    const quadratic = times(skewScale, z2);
    skewScale.negative = !skewScale.negative;
    const constant = B.combine(times(B.term(mu), denominator), skewScale);
    return B.finish(B.combineThree(odd, quadratic, constant), denominator);
}
